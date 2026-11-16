"""Run the Day 7 agent on one split using Day 6 pgvector and an OpenAI-compatible chat API."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6_ROOT = ROOT.parent / "K5-L3B-Day06-NguyenHuuManh-2A202604015-DataIngestionEngineering"
sys.path.insert(0, str(ROOT / "starter"))
from agentic import run_agent
from baseline import TfidfIndex
from src.day6_loader import load_answerable_questions, load_day6_stack, load_unanswerable_cases
from src.local_model import (
    ChatEndpointError, ChatTransportError, OpenAICompatibleChatController,
)
from src.retrieval import HybridRetriever, TfidfRetriever


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _error_trace(case: dict, split: str, exc: Exception) -> dict:
    return {
        "query_id": case["id"], "split": split,
        "status": "error", "answer": "", "citations": [],
        "events": [{"event": "error", "error_type": type(exc).__name__,
                    "message": str(exc)}],
    }


def _pending_trace(case: dict, split: str) -> dict:
    return {
        "query_id": case["id"], "split": split,
        "status": "pending", "answer": "", "citations": [],
        "events": [{"event": "pending",
                    "message": "Not attempted after provider/transport failure; resume the run later."}],
    }


def main() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env", override=False)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day6-root", type=Path, default=DEFAULT_DAY6_ROOT)
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data" / "day6_nist_questions.json")
    parser.add_argument("--unanswerable", type=Path,
                        default=ROOT / "data" / "nist_unanswerable_cases.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--strategy", default="C",
                        help="Day 6 dense index strategy used by dense/hybrid retrieval")
    parser.add_argument("--retrieval", choices=("dense_day6", "tfidf", "hybrid_rrf"),
                        default="hybrid_rrf")
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL") or os.getenv("LOCAL_MODEL_NAME"))
    parser.add_argument("--base-url", default=os.getenv(
        "CHAT_BASE_URL", os.getenv("LOCAL_MODEL_BASE_URL", "https://api.openai.com/v1")))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1200,
                        help="context character budget; dev default selected from retrieval benchmark")
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="timeout per hosted/local chat request, seconds")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed cases and retry errors/pending cases from prior traces")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "agent")
    args = parser.parse_args()
    if not args.model:
        parser.error("Set CHAT_MODEL or pass --model with your provider's model name.")
    api_key = os.getenv("CHAT_API_KEY")
    hostname = (urlparse(args.base_url).hostname or "").casefold()
    if hostname not in ("localhost", "127.0.0.1", "::1") and not api_key:
        parser.error("Set CHAT_API_KEY in the environment for a hosted chat endpoint; do not pass it on the command line.")

    answerable = load_answerable_questions(args.questions, args.split)
    unanswerable = load_unanswerable_cases(args.unanswerable, args.split)
    if not unanswerable:
        raise ValueError(f"No unanswerable cases found for split {args.split}")
    cases = answerable + unanswerable
    cases_for_metrics = [
        {"id": case["id"], "split": args.split,
         "answerable": bool(case.get("answerable"))}
        for case in cases
    ]

    output = args.output / args.split
    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.json"
    traces_path = output / "traces.json"
    metadata_path = output / "run_metadata.json"
    _write_json(cases_path, cases_for_metrics)

    previous = {}
    if args.resume and traces_path.exists():
        previous_traces = json.loads(traces_path.read_text(encoding="utf-8"))
        previous = {trace["query_id"]: trace for trace in previous_traces}
        if metadata_path.exists():
            old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            config = {
                "strategy": args.strategy, "retrieval": args.retrieval,
                "chat_model": args.model, "chat_base_url": args.base_url,
                "top_k": args.top_k, "context_budget_chars": args.budget,
                "max_searches": args.max_searches,
            }
            if any(old_metadata.get(key) != value for key, value in config.items()):
                raise ValueError("Cannot resume: prior run configuration differs from this run.")

    _, _, dense, chunks = load_day6_stack(args.day6_root, strategy=args.strategy)
    lexical = TfidfRetriever(TfidfIndex(chunks))
    retrievers = {
        "dense_day6": dense,
        "tfidf": lexical,
        "hybrid_rrf": HybridRetriever(dense, lexical),
    }
    retriever = retrievers[args.retrieval]
    controller = OpenAICompatibleChatController(
        args.model, base_url=args.base_url, api_key=api_key, timeout=args.timeout,
    )

    traces = []
    stopped_for_provider = False
    for index, case in enumerate(cases):
        old_trace = previous.get(case["id"])
        if old_trace and old_trace.get("status") in ("answered", "abstain"):
            trace = old_trace
        else:
            try:
                result = run_agent(
                    case["question"], controller, retriever,
                    filters=case.get("filter"), max_searches=args.max_searches,
                    top_k=args.top_k, budget=args.budget,
                )
                trace = {"query_id": case["id"], "split": args.split, **result}
            except Exception as exc:
                trace = _error_trace(case, args.split, exc)
                stopped_for_provider = isinstance(exc, (ChatEndpointError, ChatTransportError))
        traces.append(trace)
        _write_json(traces_path, traces)

        if stopped_for_provider:
            for remaining in cases[index + 1:]:
                old_trace = previous.get(remaining["id"])
                if old_trace and old_trace.get("status") in ("answered", "abstain"):
                    traces.append(old_trace)
                else:
                    traces.append(_pending_trace(remaining, args.split))
            _write_json(traces_path, traces)
            break

    by_id = {trace["query_id"]: trace for trace in traces}
    traces = [by_id[case["id"]] for case in cases if case["id"] in by_id]
    _write_json(traces_path, traces)
    status_counts = {
        status: sum(trace["status"] == status for trace in traces)
        for status in ("answered", "abstain", "error", "pending")
    }
    run_status = "complete" if not status_counts["error"] and not status_counts["pending"] else "partial"
    metadata = {
        "split": args.split, "strategy": args.strategy,
        "retrieval": args.retrieval,
        "chat_model": args.model, "chat_base_url": args.base_url,
        "top_k": args.top_k, "context_budget_chars": args.budget,
        "max_searches": args.max_searches, "request_timeout_seconds": args.timeout,
        "case_count": len(cases), "run_status": run_status,
        "status_counts": status_counts,
        "resumed": args.resume,
        "note": "Gold answers and evidence were not passed to the agent.",
    }
    _write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("Saved cases and traces under " + str(output))


if __name__ == "__main__":
    main()