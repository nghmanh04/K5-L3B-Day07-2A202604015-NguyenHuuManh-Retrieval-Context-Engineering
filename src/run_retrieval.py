"""Compare Day 6 dense, TF-IDF, and hybrid retrieval on the Day 7 benchmark."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAY6_ROOT = ROOT.parent / "K5-L3B-Day06-NguyenHuuManh-2A202604015-DataIngestionEngineering"
sys.path.insert(0, str(ROOT / "starter"))
from baseline import TfidfIndex
from src.day6_loader import load_answerable_questions, load_day6_stack
from src.retrieval import HybridRetriever, TfidfRetriever, evaluate_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day6-root", type=Path, default=DEFAULT_DAY6_ROOT,
                        help="local Day 6 checkout containing starter/.env and starter/src")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data" / "day6_nist_questions.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--config", choices=(
        "all", "dense_day6", "tfidf", "hybrid_rrf",
        "dense_top5_context800", "hybrid_top3_context1200",
    ), default="all", help="run all configs or one frozen configuration")
    parser.add_argument("--strategy", default="C", help="ingested Day 6 vector strategy")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "retrieval")
    args = parser.parse_args()

    questions = load_answerable_questions(args.questions, args.split)
    _, _, dense, chunks = load_day6_stack(args.day6_root, strategy=args.strategy)
    lexical = TfidfRetriever(TfidfIndex(chunks))
    hybrid = HybridRetriever(dense, lexical)
    configs = [
        ("dense_day6", dense, 3, 800),
        ("tfidf", lexical, 3, 800),
        ("hybrid_rrf", hybrid, 3, 800),
        # Isolate the top-k effect while keeping the context budget fixed.
        ("dense_top5_context800", dense, 5, 800),
        # Isolate the context-budget effect while keeping retrieval top-k fixed.
        ("hybrid_top3_context1200", hybrid, 3, 1200),
    ]
    if args.config != "all":
        configs = [config for config in configs if config[0] == args.config]
    results = {}
    for name, retriever, top_k, budget in configs:
        results[name] = evaluate_retriever(
            retriever, questions, top_k=top_k, context_budget=budget,
        )

    output = args.output / args.split
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "split": args.split,
        "corpus_source": "Day 6 PostgreSQL/pgvector strategy " + args.strategy,
        "embedding_model": getattr(dense.embedder, "model_name", None),
        "chunk_count": len(chunks),
        "answerable_question_count": len(questions),
        "selected_config": None if args.config == "all" else args.config,
        "configurations": results,
        "hypotheses": {
            "dense_top5_context800": "More candidates may improve evidence retrieval while context budget stays fixed.",
            "hybrid_top3_context1200": "More context may preserve retrieved evidence while top-k stays fixed.",
        },
        "selection_note": (
            "Only the frozen dev-selected configuration was run on test."
            if args.split == "test" and args.config != "all"
            else "Choose using dev. Run test only after the configuration is frozen."
        ),
    }
    path = output / "retrieval_comparison.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {
        "hit1": data["hit1"],
        "evidence_hit3": data["evidence_hit3"],
        "context_evidence_rate": data["context_evidence_rate"],
        "mean_retrieval_ms": data["mean_retrieval_ms"],
    } for name, data in results.items()}, ensure_ascii=False, indent=2))
    print("Saved full traces to " + str(path))


if __name__ == "__main__":
    main()
