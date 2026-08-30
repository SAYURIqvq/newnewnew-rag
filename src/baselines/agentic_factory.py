"""Build the full Agentic workflow (same stack as app.py)."""

from pathlib import Path
from typing import Optional

from src.config import get_settings
from src.llm.qwen import create_qwen_chat_model
from src.storage.chroma_store import ChromaVectorStore
from src.ingestion.embedder import EmbeddingGenerator
from src.graph.graph_builder import KnowledgeGraph
from src.agents.planner import PlannerAgent
from src.agents.query_decomposer import QueryDecomposer
from src.agents.retrieval_coordinator import RetrievalCoordinator
from src.agents.validator import ValidatorAgent
from src.agents.synthesis import SynthesisAgent
from src.agents.writer import WriterAgent
from src.agents.critic import CriticAgent
from src.orchestration.complete_workflow import CompleteAgenticRAGWorkflow
from src.retrieval.vector_search import VectorSearchAgent
from src.retrieval.keyword_search import KeywordSearchAgent
from src.retrieval.graph_search import GraphSearchAgent


def create_agentic_workflow(
    persist_directory: str = "data/chroma_db",
    graph_path: Optional[str] = None,
) -> CompleteAgenticRAGWorkflow:
    """Create CompleteAgenticRAGWorkflow using on-disk Chroma + optional graph."""
    settings = get_settings()
    planner_llm = create_qwen_chat_model(settings, model=settings.get_agent_model("planner"))
    decomposer_llm = create_qwen_chat_model(settings, model=settings.get_agent_model("decomposer"), max_tokens=1000)
    validator_llm = create_qwen_chat_model(settings, model=settings.get_agent_model("validator"))
    writer_llm = create_qwen_chat_model(settings, model=settings.get_agent_model("writer"))
    critic_llm = create_qwen_chat_model(settings, model=settings.get_agent_model("critic"))

    vector_store = ChromaVectorStore(persist_directory=persist_directory)
    embedder = EmbeddingGenerator()

    vector_agent = VectorSearchAgent(vector_store=vector_store, embedder=embedder)
    keyword_agent = KeywordSearchAgent(vector_store=vector_store)

    graph_agent = None
    if graph_path is None:
        graphs_dir = Path("data/graphs")
        if graphs_dir.exists():
            pkl_files = list(graphs_dir.glob("*_graph.pkl"))
            if pkl_files:
                graph_path = str(pkl_files[0])

    if graph_path and Path(graph_path).exists():
        kg = KnowledgeGraph()
        kg.load(graph_path)
        graph_agent = GraphSearchAgent(
            knowledge_graph=kg,
            vector_store=vector_store,
        )

    coordinator = RetrievalCoordinator(
        vector_agent=vector_agent,
        keyword_agent=keyword_agent,
        graph_agent=graph_agent,
    )

    return CompleteAgenticRAGWorkflow(
        planner=PlannerAgent(llm=planner_llm),
        decomposer=QueryDecomposer(llm=decomposer_llm),
        coordinator=coordinator,
        validator=ValidatorAgent(llm=validator_llm),
        synthesis=SynthesisAgent(),
        writer=WriterAgent(llm=writer_llm),
        critic=CriticAgent(llm=critic_llm, quality_threshold=0.7),
    )
