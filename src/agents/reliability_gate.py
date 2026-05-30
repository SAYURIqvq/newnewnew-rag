"""
Reliability gate for final Agentic RAG answers.

The gate is intentionally lightweight and deterministic. RAGAS remains the
offline evaluation tool; this runtime gate catches obvious grounding failures
before an answer is returned to the UI or benchmark scripts.
"""

import re
from typing import Any, Dict, List

from src.models.agent_state import AgentState


class ReliabilityGate:
    """Final answer quality gate based on citations and prior agent scores."""

    def __init__(
        self,
        min_validation_score: float = 0.5,
        min_critic_score: float = 0.5,
    ):
        self.min_validation_score = min_validation_score
        self.min_critic_score = min_critic_score

    def evaluate(self, state: AgentState) -> Dict[str, Any]:
        """
        Evaluate whether the final answer is safe to return.

        Returns:
            Dict with passed flag, reasons, and citation diagnostics.
        """
        answer = state.answer or ""
        reasons: List[str] = []
        citation_numbers = self._extract_citations(answer)
        honest_non_answer = self._is_honest_non_answer(answer)

        if not answer.strip():
            reasons.append("empty_answer")

        if not state.chunks and not honest_non_answer:
            reasons.append("no_retrieved_context")

        validation_score = state.validation_score
        if validation_score is not None and validation_score < self.min_validation_score:
            reasons.append("low_validation_score")

        critic_score = state.critic_score
        if critic_score is not None and critic_score < self.min_critic_score:
            reasons.append("low_critic_score")

        invalid_citations = [
            num for num in citation_numbers
            if num < 1 or num > len(state.chunks)
        ]
        if invalid_citations:
            reasons.append("invalid_citations")

        if state.chunks and not citation_numbers and not honest_non_answer:
            reasons.append("missing_citations")

        blocking_reasons = [
            reason for reason in reasons
            if reason in {
                "empty_answer",
                "no_retrieved_context",
                "invalid_citations",
                "missing_citations",
            }
        ]
        passed = not blocking_reasons or honest_non_answer

        return {
            "passed": passed,
            "reasons": reasons,
            "blocking_reasons": blocking_reasons,
            "citation_count": len(set(citation_numbers)),
            "invalid_citations": invalid_citations,
            "honest_non_answer": honest_non_answer,
            "min_validation_score": self.min_validation_score,
            "min_critic_score": self.min_critic_score,
        }

    def apply(self, state: AgentState) -> AgentState:
        """Attach gate result to state metadata and revise unsafe answers."""
        result = self.evaluate(state)
        state.metadata["reliability_gate"] = result

        if not result["passed"]:
            state.answer = (
                "I cannot provide a reliable answer from the retrieved context. "
                f"Reliability gate failed: {', '.join(result['reasons'])}."
            )

        return state

    def _extract_citations(self, answer: str) -> List[int]:
        return [int(num) for num in re.findall(r"\[(\d+)\]", answer)]

    def _is_honest_non_answer(self, answer: str) -> bool:
        lowered = answer.lower()
        markers = [
            "do not contain information",
            "don't have enough information",
            "not enough information",
            "cannot answer",
            "not available in the provided",
            "not mentioned in the provided",
        ]
        return any(marker in lowered for marker in markers)
