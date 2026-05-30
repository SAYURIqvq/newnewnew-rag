"""Tests for the final reliability gate."""

from src.agents.reliability_gate import ReliabilityGate
from src.models.agent_state import AgentState, Chunk


def test_reliability_gate_passes_grounded_answer():
    gate = ReliabilityGate()
    state = AgentState(
        query="What is Python?",
        chunks=[
            Chunk(
                text="Python is a programming language.",
                doc_id="doc",
                chunk_id="chunk-1",
                score=0.9,
                metadata={},
            )
        ],
        answer="Python is a programming language [1].",
        validation_score=0.8,
        critic_score=0.9,
    )

    result = gate.evaluate(state)

    assert result["passed"] is True
    assert result["citation_count"] == 1
    assert result["reasons"] == []


def test_reliability_gate_rewrites_unsafe_answer():
    gate = ReliabilityGate()
    state = AgentState(
        query="What is Python?",
        chunks=[
            Chunk(
                text="Python is a programming language.",
                doc_id="doc",
                chunk_id="chunk-1",
                score=0.9,
                metadata={},
            )
        ],
        answer="Python is a programming language.",
        validation_score=0.8,
        critic_score=0.9,
    )

    updated = gate.apply(state)

    assert updated.metadata["reliability_gate"]["passed"] is False
    assert "missing_citations" in updated.metadata["reliability_gate"]["reasons"]
    assert updated.answer.startswith("I cannot provide a reliable answer")


def test_reliability_gate_fallback_summary_can_recover_missing_citation():
    gate = ReliabilityGate()
    state = AgentState(
        query="What is Python?",
        chunks=[
            Chunk(
                text="Python is a programming language.",
                doc_id="doc",
                chunk_id="chunk-1",
                score=0.9,
                metadata={},
            )
        ],
        answer="Python is a programming language.",
        validation_score=0.8,
        critic_score=0.9,
    )

    state.answer = "Python is a programming language [1]."
    result = gate.evaluate(state)

    assert result["passed"] is True


def test_reliability_gate_treats_low_scores_as_warnings_for_cited_answers():
    gate = ReliabilityGate()
    state = AgentState(
        query="What is Python?",
        chunks=[
            Chunk(
                text="Python is a programming language.",
                doc_id="doc",
                chunk_id="chunk-1",
                score=0.9,
                metadata={},
            )
        ],
        answer="Python is a programming language [1].",
        validation_score=0.2,
        critic_score=0.4,
    )

    result = gate.evaluate(state)

    assert result["passed"] is True
    assert result["blocking_reasons"] == []
    assert "low_validation_score" in result["reasons"]
    assert "low_critic_score" in result["reasons"]


def test_reliability_gate_allows_honest_non_answer():
    gate = ReliabilityGate()
    state = AgentState(
        query="Unknown topic",
        chunks=[],
        answer="The provided documents do not contain information about Unknown topic.",
        validation_score=0.2,
        critic_score=0.2,
    )

    result = gate.evaluate(state)

    assert result["passed"] is True
    assert result["honest_non_answer"] is True
