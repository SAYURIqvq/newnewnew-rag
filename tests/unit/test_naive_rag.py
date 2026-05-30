"""Unit tests for Naive RAG baseline (no API calls)."""

from unittest.mock import MagicMock, patch

import pytest

from src.baselines.naive_rag import NaiveRAG
from src.models.agent_state import AgentState
from src.models.chunk import Chunk


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search.return_value = [
        {
            "chunk_id": "c1",
            "text": "Machine learning is a subset of AI.",
            "score": 0.9,
            "chunk_type": "child",
            "metadata": {"filename": "test.pdf"},
        }
    ]
    return store


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.generate_query_embedding.return_value = [0.1] * 8
    return embedder


@pytest.fixture
def mock_writer():
    writer = MagicMock()

    def fake_run(state: AgentState) -> AgentState:
        state.answer = "ML is a subset of AI [1]."
        return state

    writer.run.side_effect = fake_run
    return writer


class TestNaiveRAG:
    def test_run_retrieves_and_generates(
        self, mock_vector_store, mock_embedder, mock_writer
    ):
        rag = NaiveRAG(
            vector_store=mock_vector_store,
            embedder=mock_embedder,
            writer=mock_writer,
        )
        result = rag.run("What is machine learning?")

        mock_embedder.generate_query_embedding.assert_called_once()
        mock_vector_store.search.assert_called_once_with(
            query_embedding=[0.1] * 8,
            top_k=5,
            return_parent=False,
        )
        mock_writer.run.assert_called_once()
        assert result.answer == "ML is a subset of AI [1]."
        assert len(result.chunks) == 1
        assert result.metadata["method"] == "naive_rag"

    def test_to_chunks_empty(self):
        assert NaiveRAG._to_chunks([]) == []

    def test_import_agentic_factory(self):
        from src.baselines.agentic_factory import create_agentic_workflow

        assert callable(create_agentic_workflow)
