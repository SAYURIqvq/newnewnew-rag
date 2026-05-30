"""
Traditional RAG baseline (minimal).

Pipeline: embed query → vector Top-K → Writer (single LLM call).
No Planner, hybrid retrieval, graph, validator, or critic.
"""

from typing import List, Optional

from src.models.agent_state import AgentState
from src.models.chunk import Chunk
from src.storage.chroma_store import ChromaVectorStore
from src.ingestion.embedder import EmbeddingGenerator
from src.agents.writer import WriterAgent
from src.llm.qwen import create_qwen_chat_model
from src.config import get_settings


class NaiveRAG:
    """
    Simple vector-only RAG baseline for thesis comparison.

    Example:
        >>> rag = NaiveRAG()
        >>> result = rag.run("What is machine learning?")
        >>> print(result.answer)
    """

    def __init__(
        self,
        top_k: int = 5,
        persist_directory: str = "data/chroma_db",
        vector_store: Optional[ChromaVectorStore] = None,
        embedder: Optional[EmbeddingGenerator] = None,
        writer: Optional[WriterAgent] = None,
    ):
        self.top_k = top_k
        self.vector_store = vector_store or ChromaVectorStore(
            persist_directory=persist_directory
        )
        self.embedder = embedder or EmbeddingGenerator()
        if writer is None:
            settings = get_settings()
            llm = create_qwen_chat_model(settings)
            writer = WriterAgent(llm=llm)
        self.writer = writer

    def run(self, query: str) -> AgentState:
        """Retrieve with vector search only, then generate answer once."""
        query_embedding = self.embedder.generate_query_embedding(query)
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            return_parent=False,
        )

        chunks = self._to_chunks(results)
        state = AgentState(query=query, chunks=chunks)
        state.metadata["method"] = "naive_rag"
        state.metadata["retrieval"] = "vector_only"
        return self.writer.run(state)

    @staticmethod
    def _to_chunks(results: List[dict]) -> List[Chunk]:
        chunks = []
        for result in results:
            meta = result.get("metadata") or {}
            chunks.append(
                Chunk(
                    text=result["text"],
                    doc_id=meta.get("doc_id", "unknown"),
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    chunk_type=result.get("chunk_type", "child"),
                    metadata={
                        "filename": meta.get("filename", "unknown"),
                        "source": "naive_vector",
                        **meta,
                    },
                )
            )
        return chunks
