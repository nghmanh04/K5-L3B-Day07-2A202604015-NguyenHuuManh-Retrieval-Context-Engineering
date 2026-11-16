"""Load the user's Day 6 local embedder, pgvector store, and corpus chunks."""
from __future__ import annotations
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from src.retrieval import Day6PgVectorRetriever

DAY6_PACKAGE = "_day6_student_src"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _import_day6_modules(day6_root: Path):
    package_dir = day6_root / "starter" / "src"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"Day 6 source package not found: {package_dir}")
    if DAY6_PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            DAY6_PACKAGE, init_file, submodule_search_locations=[str(package_dir)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load Day 6 source package at {package_dir}")
        package = importlib.util.module_from_spec(spec)
        sys.modules[DAY6_PACKAGE] = package
        spec.loader.exec_module(package)
    embeddings = importlib.import_module(DAY6_PACKAGE + ".embeddings")
    store_module = importlib.import_module(DAY6_PACKAGE + ".store")
    return embeddings.MyEmbedder, store_module.PgVectorStore


def load_day6_stack(day6_root: Path, *, strategy: str = "C"):
    """Connect to an already-ingested Day 6 strategy; never silently re-ingest."""
    day6_root = day6_root.expanduser().resolve()
    if not day6_root.is_dir():
        raise FileNotFoundError(f"Day 6 project directory not found: {day6_root}")
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(day6_root / "starter" / ".env", override=False)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is missing. Set it in the Day 6 starter/.env or the environment."
        )
    import psycopg
    try:
        with psycopg.connect(database_url, connect_timeout=5) as connection:
            connection.execute("SELECT 1")
    except psycopg.OperationalError as exc:
        raise RuntimeError("Day 6 PostgreSQL is not reachable; start its database first.") from exc
    MyEmbedder, PgVectorStore = _import_day6_modules(day6_root)
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedder = MyEmbedder(model_name)
    store = PgVectorStore(database_url, embedder.dimensions)
    retriever = Day6PgVectorRetriever(embedder, store, strategy=strategy)
    chunks = load_day6_chunks(database_url, store.table_name, strategy)
    if not chunks:
        raise RuntimeError(
            f"No chunks found for Day 6 strategy {strategy!r}; run the Day 6 ingestion first."
        )
    return embedder, store, retriever, chunks


def load_day6_chunks(database_url: str, table_name: str, strategy: str) -> list[dict[str, Any]]:
    """Read text/metadata for the lexical baseline; embeddings remain in pgvector."""
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    statement = sql.SQL(
        "SELECT chunk_id, document_id, source, page, chunk_text, metadata "
        "FROM {} WHERE strategy = %s ORDER BY document_id, page, chunk_id"
    ).format(sql.Identifier(table_name))
    with psycopg.connect(database_url, row_factory=dict_row, connect_timeout=5) as connection:
        rows = connection.execute(statement, (strategy,)).fetchall()
    return [{
        "id": row["chunk_id"], "source_id": row["document_id"],
        "source": row["source"], "page": row["page"],
        "text": row["chunk_text"], "metadata": row["metadata"] or {},
    } for row in rows]


def convert_day6_question(question: dict[str, Any]) -> dict[str, Any]:
    gold = question.get("gold")
    if gold is None:
        pages = question.get("gold_pages", [question.get("gold_page")])
        gold = [{
            "source_id": question.get("gold_source_id"),
            "page": page,
            "evidence": question.get("evidence_quote"),
        } for page in pages if page is not None]
    result = {
        "id": question.get("id"), "split": question.get("split"),
        "question": question.get("question"),
        "expected_answer": question.get("expected_answer"), "gold": gold,
        "answerable": True,
    }
    if question.get("filter"):
        result["filter"] = question["filter"]
    return result


def load_answerable_questions(path: Path, split: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Day 6 benchmark must be a JSON array")
    questions = [convert_day6_question(item) for item in raw
                 if item.get("split") == split]
    if not questions:
        raise ValueError(f"No answerable questions found for split {split}")
    return questions


def load_unanswerable_cases(path: Path, split: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("unanswerable cases must be a JSON array")
    cases = [item for item in raw
             if item.get("split") == split and item.get("answerable") is False]
    return cases
