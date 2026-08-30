"""Background document preparation for independent comparison workspaces."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.graph.entity_extractor import EntityExtractor
from src.graph.graph_builder import KnowledgeGraph
from src.graph.relationship_extractor import RelationshipExtractor
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.hierarchical_chunker import HierarchicalChunker
from src.storage.chroma_store import ChromaVectorStore


def prepare_workspace(
    *,
    file_bytes: bytes,
    filename: str,
    mode: str,
    chunking_mode: str,
    embedder: Any,
) -> Dict[str, Any]:
    """Build one workspace without reading or mutating Streamlit state."""
    upload_dir = Path("data/uploads") / mode
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename
    file_path.write_bytes(file_bytes)

    loader = DocumentLoader()
    document = loader.load(str(file_path))
    file_ext = file_path.suffix.upper()
    metadata = {
        "filename": filename,
        "file_type": file_ext,
        "chunking_mode": chunking_mode,
        **document.metadata,
    }

    if chunking_mode == "hierarchical":
        chunker = HierarchicalChunker(
            parent_size=2000,
            child_size=500,
            child_overlap=50,
        )
        parent_chunks, child_chunks = chunker.chunk_text(
            text=document.text,
            doc_id=document.doc_id,
            metadata=metadata,
        )
    else:
        chunker = HierarchicalChunker(
            parent_size=500,
            child_size=500,
            child_overlap=50,
        )
        _, child_chunks = chunker.chunk_text(
            text=document.text,
            doc_id=document.doc_id,
            metadata=metadata,
        )
        parent_chunks = []

    if parent_chunks:
        parent_embeddings = embedder.generate([chunk.text for chunk in parent_chunks])
        for chunk, embedding in zip(parent_chunks, parent_embeddings):
            chunk.embedding = embedding

    if child_chunks:
        child_embeddings = embedder.generate([chunk.text for chunk in child_chunks])
        for chunk, embedding in zip(child_chunks, child_embeddings):
            chunk.embedding = embedding

    vector_store = ChromaVectorStore(
        persist_directory=f"data/chroma_db_{mode}"
    )
    vector_store.clear_all()
    vector_store.add_chunks(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        filename=filename,
    )

    knowledge_graph = None
    if mode == "agentic":
        entity_extractor = EntityExtractor()
        relationship_extractor = RelationshipExtractor()
        chunk_entities = {}
        chunk_relationships = {}

        for chunk in child_chunks:
            entities = entity_extractor.extract(chunk.text)
            chunk_entities[chunk.chunk_id] = entities
            chunk_relationships[chunk.chunk_id] = (
                relationship_extractor.extract_from_sentence(chunk.text, entities)
                if len(entities) >= 2
                else []
            )

        knowledge_graph = KnowledgeGraph()
        knowledge_graph.build_from_chunks(
            child_chunks,
            chunk_entities,
            chunk_relationships,
        )
        graph_path = Path("data/graphs") / f"{mode}_{filename}_graph.pkl"
        knowledge_graph.save(str(graph_path))

    document_record = {
        "name": filename,
        "path": str(file_path),
        "type": file_ext,
        "pages": loader.count_pages(str(file_path)),
        "chunks": len(child_chunks),
        "parents": len(parent_chunks),
        "chunking_mode": chunking_mode,
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return {
        "vector_store": vector_store,
        "knowledge_graph": knowledge_graph,
        "parent_chunks": parent_chunks,
        "child_chunks": child_chunks,
        "document_name": filename,
        "document_record": document_record,
        "ready": True,
    }
