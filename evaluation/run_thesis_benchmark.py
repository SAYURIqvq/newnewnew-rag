#!/usr/bin/env python3
"""
Thesis benchmark: run Baseline vs Agentic on the same questions, export tables.

Usage:
  python evaluation/run_thesis_benchmark.py
  python evaluation/run_thesis_benchmark.py --limit 2
  python evaluation/run_thesis_benchmark.py --no-ragas
  python evaluation/run_thesis_benchmark.py --ragas-only

Outputs: results/thesis/experiment_table.csv, summary_table.md, case_studies/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

DEFAULT_DATASET = ROOT / "data/evaluation/thesis_test_dataset.json"
OUTPUT_DIR = ROOT / "results/thesis"


def _load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["test_cases"]


def _chunks_to_contexts(chunks, limit: int = 5) -> list[str]:
    return [c.text for c in chunks[:limit]]


def run_baseline(question: str, vector_store, embedder) -> dict:
    from src.baselines.naive_rag import NaiveRAG

    t0 = time.time()
    rag = NaiveRAG(vector_store=vector_store, embedder=embedder)
    state = rag.run(question)
    latency = time.time() - t0
    return {
        "answer": state.answer or "",
        "contexts": _chunks_to_contexts(state.chunks),
        "num_chunks": len(state.chunks),
        "latency_s": round(latency, 2),
    }


def run_agentic(question: str, vector_store, embedder) -> dict:
    from src.baselines.agentic_factory import create_agentic_workflow

    t0 = time.time()
    workflow = create_agentic_workflow()
    state = workflow.run(question)
    latency = time.time() - t0
    strategy = state.strategy
    if hasattr(strategy, "value"):
        strategy = strategy.value
    return {
        "answer": state.answer or "",
        "contexts": _chunks_to_contexts(state.chunks),
        "num_chunks": len(state.chunks),
        "latency_s": round(latency, 2),
        "strategy": str(strategy),
    }


def score_ragas(question: str, answer: str, contexts: list[str], ground_truth: str) -> dict:
    from src.evaluation.ragas_evaluator import RAGASEvaluator

    if not answer.strip():
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "overall": 0.0,
        }
    ev = RAGASEvaluator()
    return ev.evaluate_single_case(question, answer, contexts, ground_truth)


def _init_stores():
    from src.storage.chroma_store import ChromaVectorStore
    from src.ingestion.embedder import EmbeddingGenerator

    store = ChromaVectorStore(persist_directory=str(ROOT / "data/chroma_db"))
    stats = store.get_stats()
    if stats.get("total_vectors", 0) == 0:
        print("ERROR: ChromaDB is empty. Upload documents via Streamlit first.")
        sys.exit(1)
    embedder = EmbeddingGenerator()
    return store, embedder


def generate_answers(cases: list[dict], vector_store, embedder) -> list[dict]:
    rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        print(f"\n[{i}/{len(cases)}] {q[:60]}...")
        row = {
            "id": i,
            "question": q,
            "ground_truth": case.get("ground_truth", ""),
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", ""),
        }
        try:
            b = run_baseline(q, vector_store, embedder)
            row["baseline_answer"] = b["answer"]
            row["baseline_contexts"] = b["contexts"]
            row["baseline_chunks"] = b["num_chunks"]
            row["baseline_latency_s"] = b["latency_s"]
        except Exception as e:
            print(f"  Baseline failed: {e}")
            row["baseline_answer"] = ""
            row["baseline_error"] = str(e)

        try:
            a = run_agentic(q, vector_store, embedder)
            row["agentic_answer"] = a["answer"]
            row["agentic_contexts"] = a["contexts"]
            row["agentic_chunks"] = a["num_chunks"]
            row["agentic_latency_s"] = a["latency_s"]
            row["agentic_strategy"] = a.get("strategy", "")
        except Exception as e:
            print(f"  Agentic failed: {e}")
            row["agentic_answer"] = ""
            row["agentic_error"] = str(e)

        rows.append(row)
    return rows


def add_ragas_scores(rows: list[dict]) -> list[dict]:
    for i, row in enumerate(rows, 1):
        print(f"  RAGAS [{i}/{len(rows)}]...")
        q, gt = row["question"], row["ground_truth"]
        if row.get("baseline_answer"):
            bs = score_ragas(q, row["baseline_answer"], row.get("baseline_contexts", []), gt)
            for k, v in bs.items():
                row[f"baseline_{k}"] = round(v, 3)
        if row.get("agentic_answer"):
            ag = score_ragas(q, row["agentic_answer"], row.get("agentic_contexts", []), gt)
            for k, v in ag.items():
                row[f"agentic_{k}"] = round(v, 3)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    keys = [
        "id", "category", "difficulty", "question",
        "baseline_faithfulness", "baseline_answer_relevancy",
        "baseline_context_precision", "baseline_context_recall", "baseline_overall",
        "agentic_faithfulness", "agentic_answer_relevancy",
        "agentic_context_precision", "agentic_context_recall", "agentic_overall",
        "baseline_latency_s", "agentic_latency_s", "baseline_chunks", "agentic_chunks",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(rows: list[dict], path: Path) -> None:
    metrics = [
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevancy"),
        ("context_precision", "Context Precision"),
        ("context_recall", "Context Recall"),
        ("overall", "Overall"),
    ]

    def avg(prefix: str, key: str) -> float:
        vals = [r[f"{prefix}_{key}"] for r in rows if f"{prefix}_{key}" in r]
        return sum(vals) / len(vals) if vals else 0.0

    lines = [
        "# Experiment Summary (Baseline vs Agentic)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Questions: {len(rows)}",
        "",
        "| Metric | Baseline | Agentic | Delta (A-B) |",
        "|--------|----------|---------|-------------|",
    ]
    for key, label in metrics:
        b, a = avg("baseline", key), avg("agentic", key)
        d = a - b
        lines.append(f"| {label} | {b:.3f} | {a:.3f} | {d:+.3f} |")

    lat_b = sum(r.get("baseline_latency_s", 0) for r in rows) / max(len(rows), 1)
    lat_a = sum(r.get("agentic_latency_s", 0) for r in rows) / max(len(rows), 1)
    lines.extend([
        "",
        "| Avg latency (s) | Baseline | Agentic |",
        "|-----------------|----------|---------|",
        f"| Mean | {lat_b:.2f} | {lat_a:.2f} |",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_case_studies(rows: list[dict], out_dir: Path, n: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows[:n], 1):
        md = [
            f"# Case Study {i}",
            "",
            f"**Category:** {row.get('category', 'N/A')}  ",
            f"**Difficulty:** {row.get('difficulty', 'N/A')}",
            "",
            "## Question",
            "",
            row["question"],
            "",
            "## Baseline (Naive RAG)",
            "",
            row.get("baseline_answer", "(no answer)") or "(no answer)",
            "",
            "## Agentic RAG (Proposed)",
            "",
            row.get("agentic_answer", "(no answer)") or "(no answer)",
            "",
            "## RAGAS (if available)",
            "",
            f"- Baseline faithfulness: {row.get('baseline_faithfulness', 'N/A')}",
            f"- Agentic faithfulness: {row.get('agentic_faithfulness', 'N/A')}",
            "",
            "_Add Streamlit screenshots side-by-side in your thesis._",
        ]
        (out_dir / f"case_{i:02d}.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis benchmark: Baseline vs Agentic")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="JSON test set path",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    parser.add_argument("--no-ragas", action="store_true", help="Skip RAGAS scoring")
    parser.add_argument(
        "--ragas-only",
        action="store_true",
        help="Only RAGAS on existing results/thesis/answers.json",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    answers_path = args.output / "answers.json"

    if args.ragas_only:
        rows = json.loads(answers_path.read_text(encoding="utf-8"))
        rows = add_ragas_scores(rows)
        answers_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        cases = _load_cases(args.dataset)
        if args.limit > 0:
            cases = cases[: args.limit]
        print(f"Loaded {len(cases)} questions from {args.dataset}")
        vector_store, embedder = _init_stores()
        rows = generate_answers(cases, vector_store, embedder)
        answers_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_ragas and not args.ragas_only:
        print("\nRunning RAGAS scoring...")
        rows = add_ragas_scores(rows)
        answers_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    elif args.ragas_only:
        pass
    else:
        rows = json.loads(answers_path.read_text(encoding="utf-8"))

    write_csv(rows, args.output / "experiment_table.csv")
    write_summary(rows, args.output / "summary_table.md")
    write_case_studies(rows, args.output / "case_studies", n=min(3, len(rows)))

    print(f"\nDone. Outputs in {args.output}/")
    print("  - experiment_table.csv")
    print("  - summary_table.md")
    print("  - case_studies/case_01.md ...")
    print("  - answers.json")


if __name__ == "__main__":
    main()
