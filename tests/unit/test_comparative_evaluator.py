"""Regression tests for fair Baseline versus Agentic heuristic comparison."""

from src.evaluation.simple_evaluator import SimpleEvaluator
from src.models.chunk import Chunk


def test_comparative_score_excludes_agentic_only_critic_score():
    """A critic score must not change the shared Baseline/Agentic score."""
    evaluator = SimpleEvaluator()
    chunk = Chunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="Northstar approves controlled deployment with citations.",
    )
    answer = (
        "Northstar approves controlled deployment with citations and human review [1]. "
        "The document limits deployment to low-risk requests."
    )

    baseline = evaluator.evaluate_answer(
        "What does Northstar approve?",
        answer,
        [chunk],
        {"self_reflection": {"final_score": 0.0}},
    )
    agentic = evaluator.evaluate_answer(
        "What does Northstar approve?",
        answer,
        [chunk],
        {"self_reflection": {"final_score": 1.0}},
    )

    assert baseline["comparative_score"] == agentic["comparative_score"]
    assert baseline["overall"] < agentic["overall"]
