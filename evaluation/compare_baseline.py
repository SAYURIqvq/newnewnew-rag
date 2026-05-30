#!/usr/bin/env python3
"""
Compare Naive RAG (baseline) vs full Agentic RAG on the same question.

Usage:
  python evaluation/compare_baseline.py "What is machine learning?"
  python evaluation/compare_baseline.py --all   # run all rows in test_dataset.json

Requires: indexed documents in data/chroma_db, ANTHROPIC_AUTH_TOKEN in .env
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines.naive_rag import NaiveRAG
from src.baselines.agentic_factory import create_agentic_workflow


def run_compare(question: str) -> None:
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    print("\n--- Baseline: Naive RAG (vector only) ---")
    naive = NaiveRAG()
    naive_result = naive.run(question)
    print(f"Chunks: {len(naive_result.chunks)}")
    print(naive_result.answer or "(no answer)")

    print("\n--- Proposed: Agentic RAG (full workflow) ---")
    workflow = create_agentic_workflow()
    agentic_result = workflow.run(question)
    print(f"Strategy: {agentic_result.strategy} | Chunks: {len(agentic_result.chunks)}")
    print(agentic_result.answer or "(no answer)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Naive vs Agentic RAG")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all questions in data/evaluation/test_dataset.json",
    )
    args = parser.parse_args()

    if args.all:
        dataset_path = Path("data/evaluation/test_dataset.json")
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
        for case in data["test_cases"]:
            run_compare(case["question"])
        return

    question = args.question
    if not question:
        dataset_path = Path("data/evaluation/test_dataset.json")
        if dataset_path.exists():
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
            question = data["test_cases"][0]["question"]
        else:
            parser.error("Provide a question or use --all")

    run_compare(question)


if __name__ == "__main__":
    main()
