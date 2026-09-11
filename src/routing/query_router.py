"""Local policy for selecting a fast or reliability-focused RAG path."""

from __future__ import annotations

from dataclasses import dataclass

from src.models.agent_state import Strategy


HIGH_RISK_TERMS = {
    "medical", "diagnosis", "legal advice", "lawsuit", "refund approval",
    "identity verification", "account closure",
}


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


def decide_route(query: str, strategy: Strategy | str | None) -> RouteDecision:
    """Route simple requests to baseline and complex requests to Agentic RAG."""
    lowered = query.lower()
    if any(term in lowered for term in HIGH_RISK_TERMS):
        return RouteDecision(
            route="refuse",
            reason="The local policy marks this as a high-risk request requiring human review.",
        )

    strategy_value = strategy.value if hasattr(strategy, "value") else str(strategy or "")
    if strategy_value.lower() == Strategy.SIMPLE.value:
        return RouteDecision(
            route="baseline",
            reason="Planner classified the question as simple, so the low-latency vector path is used.",
        )
    return RouteDecision(
        route="agentic",
        reason="Planner selected a multi-hop or graph strategy, so the reliability-focused workflow is used.",
    )
