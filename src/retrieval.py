"""Composable Day 7 retrieval adapters: dense, TF-IDF, and hybrid RRF."""
import re
import unicodedata
from collections import defaultdict
from typing import Any, Callable, Protocol


class Searcher(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]: ...


def _matches_filters(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    metadata = chunk.get("metadata") or {}
    return all(chunk.get(key, metadata.get(key)) == value for key, value in filters.items())


def _validate_results(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if len(rows) > top_k:
        raise ValueError("retriever returned more than top_k results")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(cid, str) or not cid for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError("retrieval results need unique, non-empty chunk IDs")
    return rows


class DenseRetriever:
    """Adapt Day 6 embedding/vector-store functions to the common search contract.

    vector_search receives the query embedding and must return chunks containing
    id, text, source_id, source, page, metadata, and optionally score.
    """
    def __init__(self, embed_query: Callable[[str], Any],
                 vector_search: Callable[..., list[dict[str, Any]]]):
        self.embed_query = embed_query
        self.vector_search = vector_search

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        embedding = self.embed_query(query)
        rows = self.vector_search(embedding, filters=filters, top_k=top_k)
        rows = [dict(row, strategy="dense") for row in rows
                if _matches_filters(row, filters)]
        return _validate_results(rows, top_k)


class TfidfRetriever:
    """Expose the starter TfidfIndex through the common search contract."""
    def __init__(self, index: Any):
        self.index = index

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        # Retrieve the full lexical ranking before filtering so eligible chunks
        # below the unfiltered top_k are not accidentally discarded.
        rows = self.index.search(query, top_k=len(self.index.chunks))
        rows = [row for row in rows if _matches_filters(row, filters)][:top_k]
        return _validate_results(rows, top_k)


class HybridRetriever:
    """Fuse dense and lexical rankings with reciprocal rank fusion (RRF)."""
    def __init__(self, dense: Searcher, lexical: Searcher, *, rrf_k: int = 60):
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        dense_rows = self.dense.search(query, filters=filters, top_k=top_k)
        lexical_rows = self.lexical.search(query, filters=filters, top_k=top_k)
        fused: dict[str, dict[str, Any]] = {}
        scores: defaultdict[str, float] = defaultdict(float)
        ranks: dict[str, dict[str, int]] = defaultdict(dict)
        source_scores: dict[str, dict[str, Any]] = defaultdict(dict)

        for method, rows in (("dense", dense_rows), ("lexical", lexical_rows)):
            _validate_results(rows, top_k)
            for rank, row in enumerate(rows, start=1):
                chunk_id = row["id"]
                scores[chunk_id] += 1.0 / (self.rrf_k + rank)
                ranks[chunk_id][method] = rank
                source_scores[chunk_id][method] = row.get("score")
                # Prefer dense payload when a chunk appears in both lists.
                if chunk_id not in fused or method == "dense":
                    fused[chunk_id] = dict(row)

        ranked = []
        for chunk_id, row in fused.items():
            row.update(strategy="hybrid_rrf", score=scores[chunk_id],
                       source_ranks=ranks[chunk_id], source_scores=source_scores[chunk_id])
            ranked.append(row)
        ranked.sort(key=lambda row: (-row["score"], row["id"]))
        return _validate_results(ranked[:top_k], top_k)

def _canonical_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    value = " ".join(value.split())
    return re.sub(r"\s+([,.;:!?])", r"\1", value)


def _evidence_hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    return any(chunk.get("source_id") == item.get("source_id")
               and chunk.get("page") == item.get("page")
               and isinstance(item.get("evidence"), str)
               and _canonical_text(item["evidence"]) in _canonical_text(chunk.get("text", ""))
               for item in gold)


def _pack_context(rows: list[dict[str, Any]], budget: int) -> tuple[str, list[str]]:
    blocks: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        chunk_id = row["id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        block = "[" + chunk_id + " | " + str(row.get("source", "")) + " | page " + str(row.get("page", "")) + "]\n" + row.get("text", "")
        proposed = "\n\n".join(blocks + [block])
        if len(proposed) <= budget:
            blocks.append(block)
            ids.append(chunk_id)
    return "\n\n".join(blocks), ids


def evaluate_retriever(retriever: Searcher, questions: list[dict[str, Any]], *,
                       top_k: int = 3, context_budget: int = 800) -> dict[str, Any]:
    """Evaluate answerable benchmark questions without exposing gold to search.

    Each question must have a non-empty question string and gold evidence list.
    Gold labels are consulted only after search returns. Counts and rates follow
    the Day 7 benchmark definitions; context_budget is measured in characters.
    """
    import time

    if top_k < 3 or context_budget < 1:
        raise ValueError("top_k must be at least 3 and context_budget must be positive")
    traces: list[dict[str, Any]] = []
    for question in questions:
        gold = question.get("gold")
        query = question.get("question")
        if not isinstance(query, str) or not query.strip() or not isinstance(gold, list) or not gold:
            raise ValueError("each evaluated question needs non-empty question and gold")
        started = time.perf_counter()
        results = retriever.search(query, filters=question.get("filter"), top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        _validate_results(results, top_k)
        hit_flags = [_evidence_hit(row, gold) for row in results]
        context, context_ids = _pack_context(results, context_budget)
        context_hit = any(flag and row["id"] in context_ids
                          for row, flag in zip(results, hit_flags))
        traces.append({
            "query_id": question.get("id"),
            "question": query,
            "strategy": results[0].get("strategy") if results else retriever.__class__.__name__,
            "top_k": top_k,
            "gold": gold,
            "results": [dict(row, evidence_hit=flag)
                        for row, flag in zip(results, hit_flags)],
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": bool(hit_flags and hit_flags[0]),
            "evidence_hit3": any(hit_flags[:3]),
            "context_evidence_hit": context_hit,
        })

    n = len(traces)
    hit1 = sum(trace["hit1"] for trace in traces)
    hit3 = sum(trace["evidence_hit3"] for trace in traces)
    context_hits = sum(trace["context_evidence_hit"] for trace in traces)
    return {
        "n": n,
        "hit1": {"n": hit1, "N": n, "rate": hit1 / n if n else None},
        "evidence_hit3": {"n": hit3, "N": n, "rate": hit3 / n if n else None},
        "context_evidence_rate": {"n": context_hits, "N": n,
                                   "rate": context_hits / n if n else None},
        "mean_context_chars": (sum(t["context_chars"] for t in traces) / n) if n else None,
        "mean_retrieval_ms": (sum(t["elapsed_ms"] for t in traces) / n) if n else None,
        "traces": traces,
    }

class Day6PgVectorRetriever:
    """Adapt the Day 6 PgVectorStore + Embedder objects to Day 7 chunks."""
    def __init__(self, embedder: Any, store: Any, *, strategy: str = "C"):
        if not strategy:
            raise ValueError("Day 6 strategy must be non-empty")
        self.embedder = embedder
        self.store = store
        self.strategy = strategy

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip() or top_k < 1:
            raise ValueError("query must be non-empty and top_k must be positive")
        query_vector = self.embedder.embed_query(query)
        rows = self.store.search(
            strategy=self.strategy, query_vector=query_vector,
            top_k=top_k, filters=filters or {},
        )
        chunks = []
        for row in rows:
            chunk = {
                "id": row.get("chunk_id", row.get("id")),
                "text": row.get("text", row.get("chunk_text", "")),
                "source_id": row.get("document_id", row.get("source_id")),
                "source": row.get("source", ""),
                "page": row.get("page"),
                "metadata": row.get("metadata") or {},
                "score": row.get("score"),
                "strategy": "dense_day6_" + self.strategy,
            }
            chunks.append(chunk)
        return _validate_results(chunks, top_k)
