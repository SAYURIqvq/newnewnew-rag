from types import SimpleNamespace

from src.models.agent_state import AgentState, Strategy
from src.orchestration.complete_workflow import CompleteAgenticRAGWorkflow


class _Logger:
    def info(self, _message):
        pass

    def warning(self, _message):
        pass


def _workflow_stub():
    return SimpleNamespace(logger=_Logger())


def test_multihop_requires_two_retrieval_rounds():
    state = AgentState(
        query="Compare evidence across sections",
        strategy=Strategy.MULTIHOP,
        retrieval_round=1,
        validation_status="PROCEED",
        validation_score=0.95,
    )

    decision = CompleteAgenticRAGWorkflow._should_retry_retrieval(
        _workflow_stub(),
        state,
    )

    assert decision == "retry"


def test_multihop_can_proceed_after_second_round():
    state = AgentState(
        query="Compare evidence across sections",
        strategy=Strategy.MULTIHOP,
        retrieval_round=2,
        validation_status="PROCEED",
        validation_score=0.95,
    )

    decision = CompleteAgenticRAGWorkflow._should_retry_retrieval(
        _workflow_stub(),
        state,
    )

    assert decision == "proceed"
