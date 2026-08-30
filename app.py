"""
Streamlit Web Interface for Agentic RAG System
Phase 1 Day 10 - ChromaDB Persistent Storage
"""

import streamlit as st
import os
import re
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# ========== NEW IMPORTS (ADD) ==========
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.embedder import EmbeddingGenerator
from src.agents.writer import WriterAgent  # If AnswerGenerator is replaced

# ========== HIERARCHICAL CHUNKING ==========
from src.ingestion.hierarchical_chunker import HierarchicalChunker

# ========== DATA MODELS ==========
from src.models.chunk import Chunk
from src.models.agent_state import AgentState

# ========== STORAGE ==========
from src.storage.chroma_store import ChromaVectorStore

# ========== EVALUATION ==========
from src.evaluation.simple_evaluator import SimpleEvaluator

# ========== GRAPH (Optional) ==========
try:
    from src.graph.entity_extractor import EntityExtractor
    from src.graph.relationship_extractor import RelationshipExtractor
    from src.graph.graph_builder import KnowledgeGraph
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="Agentic RAG Demo",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_cached_embedder():
    """Create one embedding model per Streamlit process."""
    return EmbeddingGenerator()


st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.25rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1rem;
        color: #607D8B;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    .demo-footer {
        text-align: center;
        color: #78909C;
        font-size: 0.85rem;
        padding: 0.5rem 0;
        line-height: 1.5;
    }
    .mode-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        background: #1E3A5F;
        color: #E3F2FD;
    }
    .stButton>button[kind="primary"] {
        width: 100%;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def _llm_api_key_ok() -> bool:
    """True if a real OpenRouter / OpenAI-compatible API token is configured."""
    from src.config import get_settings

    key = (get_settings().anthropic_auth_token or "").strip()
    if not key:
        return False
    placeholders = {
        "your_openrouter_api_key_here",
        "your_deepseek_api_key_here",
        "your_dashscope_key_here",
        "changeme",
        "sk-xxx",
    }
    return key.lower() not in placeholders


def _rag_mode_short(mode=None) -> str:
    """Short mode label for metrics (avoids Streamlit truncation)."""
    mode = mode or st.session_state.get("rag_mode", "agentic")
    return "Agentic" if mode == "agentic" else "Baseline"


def _format_strategy_label(strategy) -> str:
    """Human-readable strategy without special Unicode chars."""
    if strategy is None:
        return "N/A"
    if hasattr(strategy, "value"):
        return str(strategy.value).replace("_", " ").title()
    text = str(strategy).strip()
    return text.replace("_", " ").title() if text else "N/A"


def _clean_agentic_answer(answer: str, max_words: int = 300) -> str:
    """Remove workflow narration and keep the final answer presentation-sized."""
    import re

    text = (answer or "").strip()
    boilerplate = [
        r"^Based on (?:the )?feedback,.*?\n\s*\n",
        r"^Here is (?:an|the) improved answer.*?\n\s*\n",
        r"^Based solely on the provided context,?\s*",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE | re.DOTALL)

    words = text.split()
    if len(words) <= max_words:
        return text

    shortened = " ".join(words[:max_words])
    sentence_end = max(
        shortened.rfind("."),
        shortened.rfind("!"),
        shortened.rfind("?"),
    )
    if sentence_end >= int(len(shortened) * 0.65):
        shortened = shortened[:sentence_end + 1]
    else:
        shortened = shortened.rstrip(",;:") + "."
    return shortened


SESSION_CACHE_PATH = Path("data/demo_session_cache.json")
EVALUATION_CACHE_PATH = Path("data/evaluation_results_cache.json")


def _json_cache_default(value):
    """Convert common numeric/model values without losing metric types."""
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def _save_demo_cache() -> None:
    """Persist user-visible demo state; runtime objects stay in memory/on disk."""
    payload = {
        "version": 1,
        "documents": st.session_state.get("documents", []),
        "comparison_results": st.session_state.get("comparison_results", {}),
        "evaluation_results": st.session_state.get("evaluation_results"),
        "chunking_mode": st.session_state.get("chunking_mode", "hierarchical"),
        "workspace_documents": {
            mode: resource.get("document_name")
            for mode, resource in st.session_state.get("workspace_resources", {}).items()
        },
    }
    SESSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = SESSION_CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_cache_default),
        encoding="utf-8",
    )
    temporary_path.replace(SESSION_CACHE_PATH)


def _save_evaluation_cache(evaluation_results: dict) -> None:
    """Persist evaluation independently from workspace/session updates."""
    EVALUATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = EVALUATION_CACHE_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            evaluation_results,
            ensure_ascii=False,
            indent=2,
            default=_json_cache_default,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(EVALUATION_CACHE_PATH)


def _clear_evaluation_cache() -> None:
    st.session_state.evaluation_results = None
    if EVALUATION_CACHE_PATH.exists():
        EVALUATION_CACHE_PATH.unlink()


def _restore_demo_cache() -> None:
    """Restore cached results and reconnect persistent workspace indexes."""
    if st.session_state.get("demo_cache_restored"):
        return
    st.session_state.demo_cache_restored = True
    if not SESSION_CACHE_PATH.exists():
        if EVALUATION_CACHE_PATH.exists():
            st.session_state.evaluation_results = json.loads(
                EVALUATION_CACHE_PATH.read_text(encoding="utf-8")
            )
            st.session_state.evaluation_results["restored"] = True
        return
    try:
        cached = json.loads(SESSION_CACHE_PATH.read_text(encoding="utf-8"))
        st.session_state.documents = cached.get("documents", [])
        st.session_state.comparison_results = cached.get(
            "comparison_results", {"baseline": None, "agentic": None}
        )
        st.session_state.evaluation_results = cached.get("evaluation_results")
        if EVALUATION_CACHE_PATH.exists():
            st.session_state.evaluation_results = json.loads(
                EVALUATION_CACHE_PATH.read_text(encoding="utf-8")
            )
        if st.session_state.evaluation_results:
            st.session_state.evaluation_results["restored"] = True
        st.session_state.chunking_mode = "hierarchical"
        for mode in ("baseline", "agentic"):
            index_path = Path(f"data/chroma_db_{mode}")
            document_name = cached.get("workspace_documents", {}).get(mode)
            if not document_name or not index_path.exists():
                continue
            store = ChromaVectorStore(persist_directory=str(index_path))
            if store.get_stats().get("total_vectors", 0) < 1:
                continue
            resource = st.session_state.workspace_resources[mode]
            resource.update({
                "vector_store": store,
                "document_name": document_name,
                "ready": True,
            })
            if mode == "agentic" and GRAPH_AVAILABLE:
                graph_path = Path("data/graphs") / f"agentic_{document_name}_graph.pkl"
                if graph_path.exists():
                    graph = KnowledgeGraph()
                    graph.load(str(graph_path))
                    resource["knowledge_graph"] = graph
        ready_resource = next(
            (resource for resource in st.session_state.workspace_resources.values()
             if resource.get("ready")),
            None,
        )
        if ready_resource:
            st.session_state.vector_store = ready_resource["vector_store"]
            st.session_state.knowledge_graph = ready_resource.get("knowledge_graph")
            st.session_state.rag_initialized = True
    except Exception as exc:
        print(f"Could not restore demo cache: {exc}")


def _clear_demo_state() -> None:
    """Clear cached UI results and both persistent workspaces."""
    for mode in ("baseline", "agentic"):
        resource = st.session_state.workspace_resources[mode]
        store = resource.get("vector_store")
        if store:
            try:
                store.clear_all()
            except Exception:
                pass
        shutil.rmtree(Path(f"data/chroma_db_{mode}"), ignore_errors=True)
        resource.update({
            "vector_store": None,
            "knowledge_graph": None,
            "parent_chunks": [],
            "child_chunks": [],
            "document_name": None,
            "ready": False,
        })
    shutil.rmtree(Path("data/graphs"), ignore_errors=True)
    if SESSION_CACHE_PATH.exists():
        SESSION_CACHE_PATH.unlink()
    if EVALUATION_CACHE_PATH.exists():
        EVALUATION_CACHE_PATH.unlink()
    st.session_state.documents = []
    st.session_state.messages = []
    st.session_state.comparison_results = {"baseline": None, "agentic": None}
    st.session_state.evaluation_results = None
    st.session_state.comparison_requested = False
    st.session_state.vector_store = None
    st.session_state.knowledge_graph = None
    st.session_state.parent_chunks = []
    st.session_state.child_chunks = []
    st.session_state.rag_initialized = False
    st.session_state.uploader_generation = st.session_state.get(
        "uploader_generation", 0
    ) + 1


def init_session_state():
    """Initialize session state variables."""
    
    # Chat messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'uploader_generation' not in st.session_state:
        st.session_state.uploader_generation = 0
    
    # Documents
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    
    # RAG components - HIERARCHICAL ← UPDATED
    if 'rag_initialized' not in st.session_state:
        st.session_state.rag_initialized = False
        st.session_state.vector_store = None
        st.session_state.embedder = None
        st.session_state.parent_chunks = []  # ← NEW
        st.session_state.child_chunks = []   # ← NEW
    
    # Processing status
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # The thesis specifies parent-child chunking as the system configuration.
    st.session_state.chunking_mode = 'hierarchical'
    
    # RAG components - FORCE REINITIALIZE
    if 'embedder' not in st.session_state or not hasattr(st.session_state.embedder, 'generate'):
        st.session_state.embedder = get_cached_embedder()
        print("✅ Embedder initialized from Streamlit cache")

    # Knowledge graph ← NEW
    if 'knowledge_graph' not in st.session_state:
        st.session_state.knowledge_graph = None

    # RAG mode for thesis demo: agentic (full) vs baseline (naive)
    if 'rag_mode' not in st.session_state:
        st.session_state.rag_mode = 'agentic'

    # Side-by-side comparison workspace
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = {
            'baseline': None,
            'agentic': None,
        }
    if 'comparison_requested' not in st.session_state:
        st.session_state.comparison_requested = False
    if 'evaluation_results' not in st.session_state:
        st.session_state.evaluation_results = None

    if 'workspace_resources' not in st.session_state:
        st.session_state.workspace_resources = {
            'baseline': {
                'vector_store': None,
                'knowledge_graph': None,
                'parent_chunks': [],
                'child_chunks': [],
                'document_name': None,
                'ready': False,
            },
            'agentic': {
                'vector_store': None,
                'knowledge_graph': None,
                'parent_chunks': [],
                'child_chunks': [],
                'document_name': None,
                'ready': False,
            },
        }

    if 'workspace_jobs' not in st.session_state:
        st.session_state.workspace_jobs = {
            'baseline': None,
            'agentic': None,
        }

    if 'workspace_executor' not in st.session_state:
        from concurrent.futures import ThreadPoolExecutor

        st.session_state.workspace_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rag-workspace",
        )

    _restore_demo_cache()

    # Remove stale entries created by the former automatic-index fallback.
    if any(
        doc.get("name") == "Preloaded ChromaDB index"
        for doc in st.session_state.documents
    ):
        if st.session_state.get("vector_store"):
            try:
                st.session_state.vector_store.clear_all()
            except Exception as e:
                print(f"Could not clear stale preloaded index: {e}")
        st.session_state.documents = []
        st.session_state.vector_store = None
        st.session_state.knowledge_graph = None
        st.session_state.parent_chunks = []
        st.session_state.child_chunks = []
        st.session_state.rag_initialized = False
        st.session_state.comparison_results = {
            'baseline': None,
            'agentic': None,
        }
        for workspace in st.session_state.workspace_resources.values():
            store = workspace.get('vector_store')
            if store:
                try:
                    store.clear_all()
                except Exception:
                    pass
            workspace.update({
                'vector_store': None,
                'knowledge_graph': None,
                'parent_chunks': [],
                'child_chunks': [],
                'document_name': None,
                'ready': False,
            })

def display_header():
    """Display app header."""
    query_count = sum(
        1
        for result in st.session_state.comparison_results.values()
        if result and (result.get("message") or {}).get("content")
    )

    col1, col2, col3, col4 = st.columns([3.2, 1, 1, 1.2])

    with col1:
        st.markdown(
            '<p class="main-header">Agentic RAG Document Q&amp;A</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="sub-header">Multi-agent retrieval-augmented generation with baseline comparison</p>',
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Docs", len(st.session_state.documents))
    with col3:
        st.metric("Queries", query_count)
    with col4:
        st.markdown("**Workspaces**")
        st.markdown(
            '<span class="mode-pill">2</span>',
            unsafe_allow_html=True,
        )


def sidebar():
    """Render sidebar with document upload and management."""
    
    with st.sidebar:
        st.markdown("### Documents & Settings")
        st.markdown("**Chunking**")
        st.caption("Hierarchical (Parent-Child)")
        st.info("📈 Parents: 2000 tokens | Children: 500 tokens")
        
        st.divider()

        # ============================================
        # FILE UPLOADER (EXISTING - KEPT)
        # ============================================
        st.markdown("**Select document**")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, Word (DOCX), or Text (TXT) files",
            label_visibility="collapsed",
            key=f"document_uploader_{st.session_state.uploader_generation}",
        )
        
        # Show file info if uploaded
        if uploaded_file is not None:
            file_size = uploaded_file.size / 1024  # KB
            file_type = Path(uploaded_file.name).suffix.upper().lstrip(".")
            st.caption(f"Selected: {uploaded_file.name} | {file_type} | {file_size:.1f} KB")
            
        st.divider()
        
        # ============================================
        # DOCUMENT LIST (UPDATED WITH HIERARCHICAL INFO)
        # ============================================
        st.markdown("**Indexed document**")
        
        if st.session_state.documents:
            for i, doc in enumerate(st.session_state.documents):
                # Get file icon based on type
                icon = {
                    '.PDF': '📕',
                    '.DOCX': '📘', 
                    '.TXT': '📄'
                }.get(doc.get('type', '.PDF'), '📄')
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.text(f"{icon} {doc['name']}")
                    
                    # Show chunking info (UPDATED)
                    mode = doc.get('chunking_mode', 'flat')
                    if mode == 'hierarchical':
                        st.caption(
                            f"🔺 Hierarchical • "
                            f"{doc.get('parents', 0)} parents • "
                            f"{doc['chunks']} children"
                        )
                    else:
                        st.caption(
                            f"📊 Flat • "
                            f"{doc.get('type', 'PDF')} • "
                            f"{doc['pages']} pages • "
                            f"{doc['chunks']} chunks"
                        )
                
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete document"):
                        delete_document(i)
                        try:
                            st.rerun()
                        except AttributeError:
                            st.experimental_rerun()
        else:
            st.info("No documents uploaded yet")
        
        st.divider()
        
        # ============================================
        # EXPORT CHAT (EXISTING - KEPT)
        # ============================================
        if st.session_state.messages:
            st.subheader("💾 Export")
            
            if st.button("📥 Download Chat History"):
                export_chat_history()
        
        st.divider()
        
        if not _llm_api_key_ok():
            st.error("Set OPENROUTER_API_KEY in `.env` and refresh.")
            st.divider()

        st.markdown("**Status**")
        
        if st.session_state.rag_initialized:
            st.success("System ready")
            try:
                stats = st.session_state.vector_store.get_stats()
                st.caption(f"Vectors indexed: {stats['total_vectors']:,}")
            except Exception:
                pass
        else:
            st.warning("Upload a document to begin")
        
        st.divider()
        with st.expander("Reset demo", expanded=False):
            st.caption("Clears documents, indexes, answers, graphs, and cached evaluation results.")
            if st.button(
                "Clear and Start Over",
                use_container_width=True,
                type="secondary",
                key="clear_complete_demo",
            ):
                _clear_demo_state()
                st.rerun()

    return uploaded_file

def export_chat_history():
    """Export chat history as text file."""
    
    chat_text = "# Chat History\n\n"
    chat_text += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    chat_text += "=" * 50 + "\n\n"
    
    for message in st.session_state.messages:
        role = "You" if message['role'] == 'user' else "Assistant"
        chat_text += f"{role}:\n{message['content']}\n\n"
        
        if 'citations' in message and message['citations']:
            chat_text += "Sources:\n"
            for citation in message['citations']:
                chat_text += f"- Source {citation['source_number']}: {citation['text_preview']}\n"
            chat_text += "\n"
        
        chat_text += "-" * 50 + "\n\n"
    
    # Download
    st.download_button(
        label="📄 Download as TXT",
        data=chat_text,
        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def process_uploaded_file(uploaded_file, processing_rag_mode=None):
    """Process uploaded document with ChromaDB persistent storage."""
    
    try:
        st.session_state.processing = True
        processing_rag_mode = (
            processing_rag_mode
            or st.session_state.get("rag_mode", "agentic")
        )
        workspace = st.session_state.workspace_resources[processing_rag_mode]
        workspace['ready'] = False
        st.session_state.comparison_results[processing_rag_mode] = None

        # Create upload directory
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_ext = file_path.suffix.upper()
        
        # Each workspace owns a separate persistent vector index.
        embedder = get_cached_embedder()
        from src.storage.chroma_store import ChromaVectorStore
        vector_store = ChromaVectorStore(
            persist_directory=f"data/chroma_db_{processing_rag_mode}"
        )
        vector_store.clear_all()
        print(f"🗑️ Cleared {processing_rag_mode} workspace index")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Initialize - USE CHROMADB ← UPDATED
        if processing_rag_mode == "baseline":
            status_text.text("🔧 Initializing Baseline vector RAG...")
        else:
            status_text.text("🔧 Initializing Agentic RAG components...")
        progress_bar.progress(10)
        
        # Step 2: Load document
        status_text.text(f"📄 Loading {file_ext} file...")
        progress_bar.progress(25)
        
        loader = DocumentLoader()
        doc = loader.load(str(file_path))
        text = doc.text
        
        # Step 3: Chunk - HIERARCHICAL OR FLAT
        status_text.text("✂️ Chunking text...")
        progress_bar.progress(40)
        
        # Prepare metadata
        chunk_metadata = {
            'filename': uploaded_file.name,
            'file_type': file_ext,
            'chunking_mode': st.session_state.chunking_mode, 
            **doc.metadata
        }
        
        if st.session_state.chunking_mode == 'hierarchical':
            # Hierarchical chunking
            chunker = HierarchicalChunker(
                parent_size=2000,
                child_size=500,
                child_overlap=50
            )
            parent_chunks, child_chunks = chunker.chunk_text(
                text=text,
                doc_id=doc.doc_id,      # ← PASS
                metadata=chunk_metadata          # ← PASS
            )
            
            status_text.text(
                f"✂️ Created {len(parent_chunks)} parents, "
                f"{len(child_chunks)} children..."
            )
        
        else:  # 'flat'
            # Flat mode: Use HierarchicalChunker with same size
            chunker = HierarchicalChunker(
                parent_size=500,
                child_size=500,
                child_overlap=50
            )
            
            parent_chunks, child_chunks = chunker.chunk_text(
                text=text,
                doc_id=doc.doc_id,      # ← PASS
                metadata=chunk_metadata          # ← PASS
            )
            
            # In flat mode, parent = child, so empty parents
            parent_chunks = []
            
            status_text.text(
                f"✂️ Created {len(child_chunks)} flat chunks..."
            )
        
        # Step 4: Embeddings
        total_chunks = len(parent_chunks) + len(child_chunks)
        status_text.text(f"🔢 Generating embeddings ({total_chunks} chunks)...")
        progress_bar.progress(60)
        
        # Embed parents
        if parent_chunks:
            for parent in parent_chunks:
                parent.embedding = embedder.generate([parent.text])[0]
        
        # Embed children
        for child in child_chunks:
            child.embedding = embedder.generate([child.text])[0]

        # ========== OPTIONAL AGENTIC KNOWLEDGE GRAPH ==========
        if processing_rag_mode == "baseline":
            status_text.text("📊 Preparing Baseline vector index...")
        else:
            status_text.text("🔨 Building Agentic knowledge graph...")
        progress_bar.progress(75)

        if processing_rag_mode == "agentic":
            try:
                from src.graph.entity_extractor import EntityExtractor
                from src.graph.relationship_extractor import RelationshipExtractor
                from src.graph.graph_builder import KnowledgeGraph

                print("🔨 Extracting entities and relationships...")
                entity_extractor = EntityExtractor()
                rel_extractor = RelationshipExtractor()
                chunk_entities = {}
                chunk_relationships = {}

                for i, chunk in enumerate(child_chunks):
                    entities = entity_extractor.extract(chunk.text)
                    chunk_entities[chunk.chunk_id] = entities
                    if len(entities) >= 2:
                        rels = rel_extractor.extract_from_sentence(chunk.text, entities)
                        chunk_relationships[chunk.chunk_id] = rels
                    else:
                        chunk_relationships[chunk.chunk_id] = []
                    if (i + 1) % 10 == 0:
                        print(f"   Processed {i+1}/{len(child_chunks)} chunks")

                print("🔨 Building knowledge graph structure...")
                kg = KnowledgeGraph()
                kg.build_from_chunks(child_chunks, chunk_entities, chunk_relationships)
                graph_path = f"data/graphs/{uploaded_file.name}_graph.pkl"
                kg.save(graph_path)
                knowledge_graph = kg
                print(f"✅ Knowledge graph built: {kg}")

            except Exception as e:
                print(f"⚠️  Graph building failed: {e}")
                import traceback
                traceback.print_exc()
                knowledge_graph = None
        else:
            knowledge_graph = None

        # ========== END GRAPH BUILDING ==========


        # Step 5: Store
        if processing_rag_mode == "baseline":
            status_text.text("💾 Storing Baseline vector index...")
        else:
            status_text.text("💾 Storing Agentic retrieval indexes...")
        progress_bar.progress(85)
        
        if st.session_state.chunking_mode == 'hierarchical':
            vector_store.add_chunks(
                parent_chunks=parent_chunks,
                child_chunks=child_chunks,
                filename=uploaded_file.name  # ← ADD THIS LINE
            )
        else:
            # For flat mode, use old add_chunks method
            for chunk in child_chunks:
                vector_store.chunks.append(chunk)
        
        # Step 6: Complete
        page_count = loader.count_pages(str(file_path))
        
        document_record = {
            'name': uploaded_file.name,
            'path': str(file_path),
            'type': file_ext,
            'pages': page_count,
            'chunks': len(child_chunks),
            'parents': len(parent_chunks) if parent_chunks else 0,
            'chunking_mode': st.session_state.chunking_mode,
            'uploaded_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.documents = [document_record]

        workspace.update({
            'vector_store': vector_store,
            'knowledge_graph': knowledge_graph,
            'parent_chunks': parent_chunks,
            'child_chunks': child_chunks,
            'document_name': uploaded_file.name,
            'ready': True,
        })

        # Keep legacy views working with the most recently prepared workspace.
        st.session_state.vector_store = vector_store
        st.session_state.knowledge_graph = knowledge_graph
        st.session_state.parent_chunks = parent_chunks
        st.session_state.child_chunks = child_chunks
        st.session_state.embedder = embedder
        st.session_state.rag_initialized = True
        
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        st.success(
            f"✅ {file_ext}: {uploaded_file.name} | "
            f"Workspace: {processing_rag_mode.upper()} | "
            f"Chunking: {st.session_state.chunking_mode.upper()}"
        )
        st.balloons()
        
        st.session_state.processing = False
        
        import time
        time.sleep(2)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state.processing = False


def delete_document(index):
    """Delete a document from the list."""
    
    try:
        doc = st.session_state.documents[index]
        
        # Local preloaded-index entries do not have a source file path.
        doc_path = doc.get('path')
        if doc_path and os.path.exists(doc_path):
            os.remove(doc_path)
        
        # Remove from list
        st.session_state.documents.pop(index)

        if not st.session_state.documents:
            for workspace in st.session_state.workspace_resources.values():
                store = workspace.get('vector_store')
                if store:
                    store.clear_all()
                workspace.update({
                    'vector_store': None,
                    'knowledge_graph': None,
                    'parent_chunks': [],
                    'child_chunks': [],
                    'document_name': None,
                    'ready': False,
                })
            st.session_state.vector_store = None
            st.session_state.knowledge_graph = None
            st.session_state.parent_chunks = []
            st.session_state.child_chunks = []
            st.session_state.rag_initialized = False
            st.session_state.comparison_results = {
                'baseline': None,
                'agentic': None,
            }
        
        st.success(f"Deleted: {doc['name']}")
        
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")


def display_chat_interface():
    """Display main chat interface."""
    
    st.header("💬 Chat with Your Documents")

    # Welcome message if no messages yet
    if not st.session_state.messages and st.session_state.documents:
        st.info("""
        👋 **Welcome!** Your documents are ready. Ask me anything!
        
        Try questions like:
        - "What is this document about?"
        - "Summarize the main points"
        - "What are the key findings?"
        """)
    elif not st.session_state.documents:
        st.warning("""
        📁 **No documents uploaded yet.**
        
        Please upload a PDF document using the sidebar to get started.
        """)

    # Check if there's a sample query to process
    if hasattr(st.session_state, 'sample_query') and st.session_state.sample_query:
        query = st.session_state.sample_query
        st.session_state.sample_query = None  # Clear it
        process_user_query(query)
        st.experimental_rerun()

    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display citations if available
                if "citations" in message and message["citations"]:
                    with st.expander("📚 View Sources"):
                        for citation in message["citations"]:
                            st.caption(f"**Source {citation['source_number']}** (Score: {citation['score']:.4f})")
                            if citation.get("retrieval_source"):
                                st.caption(f"Retrieval: {citation['retrieval_source']}")
                            if citation.get("graph_paths"):
                                st.caption("Graph reasoning paths:")
                                for path in citation["graph_paths"][:3]:
                                    st.code(path.get("description", ""), language="text")
                            st.text(citation['text_preview'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.rag_initialized:
            st.warning("Please upload a document first.")
        else:
            process_user_query(prompt)


def _append_assistant_response(
    query: str,
    result,
    start_time: float,
    rag_mode: str,
    workflow_metadata: dict,
    strategy_label: str,
    record_in_chat: bool = True,
):
    """Shared: citations, chat message, performance tracking."""
    import time

    citations = []
    for i, chunk in enumerate(result.chunks[:5], 1):
        citations.append({
            "source_number": i,
            "filename": chunk.metadata.get("filename", "unknown"),
            "chunk_type": chunk.metadata.get("chunk_type", "unknown"),
            "retrieval_source": chunk.metadata.get("source") or chunk.metadata.get("retrieval_method"),
            "graph_paths": chunk.metadata.get("graph_paths", []),
            "text_preview": chunk.text[:200],
            "score": chunk.score or 0.0,
        })

    latency = time.time() - start_time
    workflow_metadata = {
        **workflow_metadata,
        "latency_seconds": latency,
    }
    assistant_message = {
        "role": "assistant",
        "content": result.answer or "No answer generated. Check document index and API settings.",
        "citations": citations,
        "workflow_metadata": workflow_metadata,
    }
    if record_in_chat:
        st.session_state.messages.append(assistant_message)

    if "performance_tracker" not in st.session_state:
        from src.monitoring.performance_tracker import PerformanceTracker
        st.session_state.performance_tracker = PerformanceTracker()

    st.session_state.performance_tracker.track_query(
        query=query,
        latency=latency,
        chunks_retrieved=len(result.chunks),
        strategy=strategy_label,
        iterations=workflow_metadata.get("regenerations", 0),
        cache_hit=False,
    )

    print(f"⏱️  Total latency: {latency:.2f}s | mode={rag_mode}")
    print("=" * 60 + "\n")
    return assistant_message


def process_user_query(
    query: str,
    record_in_chat: bool = True,
    rag_mode_override: str | None = None,
):
    """Process a query in the selected or explicitly requested workflow."""
    import time
    start_time = time.time()
    
    from src.models.agent_state import AgentState
    from src.orchestration.complete_workflow import CompleteAgenticRAGWorkflow
    from src.agents.planner import PlannerAgent
    from src.agents.query_decomposer import QueryDecomposer
    from src.agents.retrieval_coordinator import RetrievalCoordinator
    from src.agents.validator import ValidatorAgent
    from src.agents.synthesis import SynthesisAgent
    from src.agents.writer import WriterAgent
    from src.agents.critic import CriticAgent
    from src.config import get_settings
    from src.llm.chat_model import create_chat_model
    
    rag_mode = rag_mode_override or st.session_state.get("rag_mode", "agentic")
    workspace = st.session_state.workspace_resources.get(rag_mode, {})
    vector_store = workspace.get("vector_store")
    knowledge_graph = workspace.get("knowledge_graph")

    print("\n" + "="*60)
    print(f"🔍 Processing query: {query}")
    print(f"   Mode: {rag_mode}")
    print("="*60)
    
    if not workspace.get("ready") or not vector_store:
        st.error(f"Process the document in the {rag_mode.title()} workspace first.")
        return

    if not _llm_api_key_ok():
        st.error(
            "Set a valid OPENROUTER_API_KEY in `.env`, then refresh and try again."
        )
        return
    
    # Add user message
    if record_in_chat:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

    # ========== BASELINE (Naive RAG) ==========
    if rag_mode == "baseline":
        from src.baselines.naive_rag import NaiveRAG

        with st.spinner("📊 Running Baseline (Naive RAG)..."):
            try:
                settings = get_settings()
                baseline_llm = create_chat_model(
                    settings,
                    model=settings.get_agent_model("writer"),
                    max_tokens=320,
                )
                naive = NaiveRAG(
                    vector_store=vector_store,
                    embedder=st.session_state.embedder,
                    writer=WriterAgent(llm=baseline_llm),
                )
                result = naive.run(query)
            except Exception as e:
                print(f"❌ Baseline failed: {e}")
                import traceback
                traceback.print_exc()
                st.error(f"Error: {e}")
                return

        return _append_assistant_response(
            query=query,
            result=result,
            start_time=start_time,
            rag_mode="baseline",
            workflow_metadata={
                "rag_mode": "baseline",
                "method": "naive_rag",
                "chunks_used": len(result.chunks),
            },
            strategy_label="naive_baseline",
            record_in_chat=record_in_chat,
        )
    
    # ========== INITIALIZE AGENTS (Agentic) ==========
    with st.spinner("⚙️ Initializing Agentic workflow..."):
        settings = get_settings()
        planner_llm = create_chat_model(settings, model=settings.get_agent_model("planner"), max_tokens=220)
        decomposer_llm = create_chat_model(settings, model=settings.get_agent_model("decomposer"), max_tokens=260)
        validator_llm = create_chat_model(settings, model=settings.get_agent_model("validator"), max_tokens=220)
        writer_llm = create_chat_model(
            settings,
            model=settings.get_agent_model("writer"),
            max_tokens=320,
            reasoning_effort="none",
        )
        critic_llm = create_chat_model(
            settings,
            model=settings.get_agent_model("critic"),
            max_tokens=300,
            reasoning_effort="none",
        )
        
        # Initialize all agents
        planner = PlannerAgent(llm=planner_llm)
        decomposer = QueryDecomposer(llm=decomposer_llm)
        
        # Retrieval coordinator needs swarm agents
        from src.retrieval.vector_search import VectorSearchAgent
        from src.retrieval.keyword_search import KeywordSearchAgent
        from src.retrieval.graph_search import GraphSearchAgent
        
        vector_agent = VectorSearchAgent(
            vector_store=vector_store,
            embedder=st.session_state.embedder
        )
        
        keyword_agent = KeywordSearchAgent(
            vector_store=vector_store
        )
        
        graph_agent = GraphSearchAgent(
            knowledge_graph=knowledge_graph,
            vector_store=vector_store
        ) if knowledge_graph else None
        
        coordinator = RetrievalCoordinator(
            vector_agent=vector_agent,
            keyword_agent=keyword_agent,
            graph_agent=graph_agent
        )
        
        validator = ValidatorAgent(llm=validator_llm)
        synthesis = SynthesisAgent()
        writer = WriterAgent(llm=writer_llm)
        critic = CriticAgent(
            llm=critic_llm,
            quality_threshold=0.7,
            max_iterations=1,
        )
        
        # Create complete workflow
        workflow = CompleteAgenticRAGWorkflow(
            planner=planner,
            decomposer=decomposer,
            coordinator=coordinator,
            validator=validator,
            synthesis=synthesis,
            writer=writer,
            critic=critic
        )
        
        print("✅ Workflow initialized")
    
    # ========== RUN WORKFLOW ==========
    with st.spinner("🤖 Running Agentic RAG workflow..."):
        try:
            # Single workflow call!
            result = workflow.run(query)
            result.answer = _clean_agentic_answer(result.answer, max_words=260)
            
            print(f"✅ Workflow complete!")
            print(f"   Strategy: {result.strategy}")
            complexity = result.complexity
            print(
                f"   Complexity: {complexity:.2f}"
                if complexity is not None
                else "   Complexity: N/A"
            )
            print(f"   Chunks: {len(result.chunks)}")
            print(f"   Retrieval rounds: {result.retrieval_round}")
            print(f"   Validation: {result.validation_status}")
            critic = result.critic_score
            print(
                f"   Critic score: {critic:.2f}"
                if critic is not None
                else "   Critic score: N/A"
            )
            print(f"   Regenerations: {result.metadata.get('regeneration_count', 0)}")
            
        except Exception as e:
            print(f"❌ Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            err_msg = f"Generation failed: {e}"
            st.error(err_msg)
            error_message = {
                "role": "assistant",
                "content": f"**Error:** {err_msg}\n\nTry **Baseline** mode or refresh the page.",
                "workflow_metadata": {"rag_mode": "agentic", "error": str(e)},
                "citations": [],
            }
            if record_in_chat:
                st.session_state.messages.append(error_message)
            return error_message
    
    # ========== PREPARE RESPONSE ==========
    strategy_val = (
        result.strategy.value
        if hasattr(result.strategy, "value")
        else str(result.strategy)
    )
    critic_decision = (
        result.critic_decision.value
        if result.critic_decision and hasattr(result.critic_decision, "value")
        else None
    )

    return _append_assistant_response(
        query=query,
        result=result,
        start_time=start_time,
        rag_mode="agentic",
        workflow_metadata={
            "rag_mode": "agentic",
            "complexity": result.complexity,
            "strategy": strategy_val,
            "retrieval_rounds": result.retrieval_round,
            "validation_score": result.validation_score,
            "critic_score": result.critic_score,
            "initial_critic_score": result.metadata.get("initial_critic_score"),
            "regenerations": result.metadata.get("regeneration_count", 0),
            "decision": critic_decision,
        },
        strategy_label=strategy_val,
        record_in_chat=record_in_chat,
    )


def _run_workspace_query(mode: str, query: str):
    """Run one mode without changing the selected single-chat workspace."""
    return process_user_query(
        query,
        record_in_chat=False,
        rag_mode_override=mode,
    )


def _store_comparison_result(mode: str, query: str, message):
    st.session_state.comparison_results[mode] = {
        "query": query,
        "message": message,
    }
    st.session_state.comparison_requested = False
    _save_demo_cache()


def _display_workspace_result(mode: str):
    saved = st.session_state.comparison_results.get(mode)
    if not saved:
        st.info("No result yet. Run both workflows or use this workspace only.")
        return

    message = saved.get("message") or {}
    metadata = message.get("workflow_metadata", {})
    citations = message.get("citations", [])
    latency = metadata.get("latency_seconds")
    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
        st.metric("Latency", f"{latency:.2f}s" if latency is not None else "N/A")
    with metric_b:
        st.metric("Sources", len(citations))
    with metric_c:
        if mode == "baseline":
            st.metric("Strategy", "Vector only")
        else:
            st.metric("Strategy", _format_strategy_label(metadata.get("strategy")))

    if mode == "agentic":
        detail_a, detail_b, detail_c, detail_d = st.columns(4)
        with detail_a:
            validation = metadata.get("validation_score")
            st.metric("Validation", f"{validation:.2f}" if validation is not None else "N/A")
        with detail_b:
            initial_critic = metadata.get("initial_critic_score")
            st.metric(
                "Initial critic",
                f"{initial_critic:.2f}" if initial_critic is not None else "N/A",
            )
        with detail_c:
            critic = metadata.get("critic_score")
            st.metric("Final critic", f"{critic:.2f}" if critic is not None else "N/A")
        with detail_d:
            st.metric("Retrieval rounds", metadata.get("retrieval_rounds", "N/A"))

    with st.expander("View question and full answer", expanded=False):
        st.markdown("**Question**")
        st.caption(saved.get("query", ""))
        st.markdown("**Answer**")
        st.markdown(message.get("content", "No answer generated."))

    if citations:
        with st.expander(f"View sources ({len(citations)})"):
            for citation in citations:
                source_number = citation.get("source_number", "-")
                filename = citation.get("filename", "unknown")
                chunk_type = citation.get("chunk_type", "unknown")
                score = citation.get("score", 0.0)
                st.caption(
                    f"[{source_number}] {filename} | {chunk_type} chunk | "
                    f"retrieval score {score:.2f}"
                )
                retrieval_source = citation.get("retrieval_source")
                if retrieval_source:
                    st.caption(f"Retrieval channel: {retrieval_source}")
                graph_paths = citation.get("graph_paths", [])
                if graph_paths:
                    st.caption("Graph reasoning paths")
                    for path in graph_paths[:3]:
                        st.code(path.get("description", ""), language="text")
                st.text(citation.get("text_preview", ""))


def display_knowledge_graph_summary():
    """Show Agentic graph status inside its owning workspace."""
    kg = st.session_state.workspace_resources["agentic"].get("knowledge_graph")
    if not kg:
        st.info(
            "Click **Process Document for Agentic** to build the knowledge graph."
        )
        return

    st.markdown("**Knowledge Graph**")
    node_col, edge_col = st.columns(2)
    with node_col:
        st.metric("Nodes", kg.graph.number_of_nodes())
    with edge_col:
        st.metric("Edges", kg.graph.number_of_edges())

    if kg.graph.number_of_nodes() > 0:
        top_entities = kg.get_top_entities(n=5, metric="degree")
        st.caption(
            "Top entities: "
            + ", ".join(f"{entity}: {int(degree)}" for entity, degree in top_entities)
        )


def _comparison_metrics(mode: str) -> dict:
    saved = st.session_state.comparison_results.get(mode) or {}
    message = saved.get("message") or {}
    metadata = message.get("workflow_metadata", {})
    citations = message.get("citations", [])
    answer = message.get("content", "")
    cited_numbers = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    invalid_citations = sorted({
        value for value in cited_numbers if value < 1 or value > len(citations)
    })
    latency = metadata.get("latency_seconds") or 0.0
    words = len(answer.split())
    rounds = metadata.get("retrieval_rounds", 1 if mode == "baseline" else 0)
    return {
        "question": saved.get("query", ""),
        "latency": latency,
        "sources": len(citations),
        "words": words,
        "strategy": (
            "Vector only"
            if mode == "baseline"
            else _format_strategy_label(metadata.get("strategy"))
        ),
        "validation": metadata.get("validation_score"),
        "critic": metadata.get("critic_score"),
        "rounds": rounds,
        "invalid_citations": invalid_citations,
        "citation_integrity": 100 if cited_numbers and not invalid_citations else 0,
    }


def display_automatic_comparison():
    """Render the requested result table and Pyecharts radar."""
    baseline_saved = st.session_state.comparison_results.get("baseline")
    agentic_saved = st.session_state.comparison_results.get("agentic")
    missing_results = [
        label
        for label, saved in [
            ("Baseline", baseline_saved),
            ("Agentic", agentic_saved),
        ]
        if not saved or not (saved.get("message") or {}).get("content")
    ]
    results_ready = not missing_results

    compare_clicked = st.button(
        "Compare Results",
        type="primary",
        use_container_width=True,
        key="compare_workspace_results",
    )
    if compare_clicked and results_ready:
        st.session_state.comparison_requested = True
    elif compare_clicked:
        st.session_state.comparison_requested = False
        st.warning(
            "Run the missing workspace result first: "
            + ", ".join(missing_results)
            + "."
        )

    if not results_ready:
        st.caption("A chart requires successful answers from both workspaces.")
        return

    if not st.session_state.comparison_requested:
        st.caption("Both results are ready. Click Compare Results to create the chart.")
        return

    baseline = _comparison_metrics("baseline")
    agentic = _comparison_metrics("agentic")

    st.markdown("### Live Run Comparison")
    if baseline["question"].strip() != agentic["question"].strip():
        st.warning(
            "The workspaces used different questions, so these results are not a "
            "direct comparison. Use the same question in both workspaces."
        )

    import pandas as pd

    table = pd.DataFrame([
        {
            "Workspace": "Baseline",
            "Latency": f"{baseline['latency']:.2f}s",
            "Sources": baseline["sources"],
            "Words": baseline["words"],
            "Strategy": baseline["strategy"],
            "Validation": "N/A",
            "Critic": "N/A",
            "Rounds": baseline["rounds"],
            "Citation check": (
                "Passed"
                if not baseline["invalid_citations"]
                else "Invalid: " + ", ".join(
                    f"[{value}]" for value in baseline["invalid_citations"]
                )
            ),
        },
        {
            "Workspace": "Agentic",
            "Latency": f"{agentic['latency']:.2f}s",
            "Sources": agentic["sources"],
            "Words": agentic["words"],
            "Strategy": agentic["strategy"],
            "Validation": (
                f"{agentic['validation']:.2f}"
                if agentic["validation"] is not None
                else "N/A"
            ),
            "Critic": (
                f"{agentic['critic']:.2f}"
                if agentic["critic"] is not None
                else "N/A"
            ),
            "Rounds": agentic["rounds"],
            "Citation check": (
                "Passed"
                if not agentic["invalid_citations"]
                else "Invalid: " + ", ".join(
                    f"[{value}]" for value in agentic["invalid_citations"]
                )
            ),
        },
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("### Current Result Charts")
    try:
        from pyecharts import options as opts
        from pyecharts.charts import Line, Radar

        max_sources = max(baseline["sources"], agentic["sources"], 1)
        baseline_reliability = [
            round(baseline["sources"] / max_sources * 100),
            baseline["citation_integrity"],
            min(100, int(baseline["rounds"] or 0) * 50),
            0,
            0,
        ]
        agentic_reliability = [
            round(agentic["sources"] / max_sources * 100),
            agentic["citation_integrity"],
            min(100, int(agentic["rounds"] or 0) * 50),
            round((agentic["validation"] or 0) * 100),
            round((agentic["critic"] or 0) * 100),
        ]

        reliability_chart = (
            Radar(init_opts=opts.InitOpts(width="100%", height="430px"))
            .add_schema(
                schema=[
                    opts.RadarIndicatorItem(name="Evidence coverage", max_=100),
                    opts.RadarIndicatorItem(name="Citation traceability", max_=100),
                    opts.RadarIndicatorItem(name="Retrieval depth", max_=100),
                    opts.RadarIndicatorItem(name="Validation", max_=100),
                    opts.RadarIndicatorItem(name="Critic review", max_=100),
                ],
                splitarea_opt=opts.SplitAreaOpts(is_show=True),
            )
            .add(
                "Baseline",
                [baseline_reliability],
                color="#315a8a",
                areastyle_opts=opts.AreaStyleOpts(opacity=0.16),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add(
                "Agentic",
                [agentic_reliability],
                color="#ef5350",
                areastyle_opts=opts.AreaStyleOpts(opacity=0.16),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Current Reliability Controls"),
                legend_opts=opts.LegendOpts(pos_top="8%"),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
        )

        latency_delta = agentic["latency"] - baseline["latency"]
        latency_max = max(10, round(max(baseline["latency"], agentic["latency"]) * 1.2))
        latency_chart = (
            Line(init_opts=opts.InitOpts(width="100%", height="430px"))
            .add_xaxis(["Baseline", "Agentic"])
            .add_yaxis(
                "Response time",
                [round(baseline["latency"], 2), round(agentic["latency"], 2)],
                color="#ef5350",
                symbol="circle",
                symbol_size=18,
                is_smooth=False,
                label_opts=opts.LabelOpts(
                    is_show=True,
                    position="top",
                    formatter="{c}s",
                    font_size=16,
                ),
                linestyle_opts=opts.LineStyleOpts(width=4),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Current Response Time",
                    subtitle=f"Agentic overhead: {latency_delta:+.2f}s",
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(is_show=True, trigger="axis"),
                yaxis_opts=opts.AxisOpts(
                    name="Seconds",
                    min_=0,
                    max_=latency_max,
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
                xaxis_opts=opts.AxisOpts(
                    axislabel_opts=opts.LabelOpts(font_size=14),
                ),
            )
        )

        reliability_col, latency_col = st.columns([1.35, 1], gap="large")
        with reliability_col:
            st.components.v1.html(reliability_chart.render_embed(), height=450)
        with latency_col:
            st.components.v1.html(latency_chart.render_embed(), height=450)
        st.caption(
            "Reliability indicators use only this run. Evidence coverage is "
            "normalized against the larger current source count; retrieval depth "
            "maps one round to 50 and two rounds to 100; Validation and Critic are "
            "the current Agentic scores. Zero for Baseline means that the control "
            "is not included, not that its answer accuracy is zero."
        )
    except Exception as exc:
        st.warning(f"Current result charts unavailable: {exc}")

    st.markdown("#### Hallucination Risk Controls")
    safeguards = pd.DataFrame([
        {
            "Control": "Evidence-backed citations",
            "Baseline": "Citation format check",
            "Agentic": "Citation format check",
        },
        {
            "Control": "Evidence validation",
            "Baseline": "Not included",
            "Agentic": (
                f"Included ({agentic['validation']:.2f})"
                if agentic["validation"] is not None else "Included"
            ),
        },
        {
            "Control": "Answer review",
            "Baseline": "Not included",
            "Agentic": (
                f"Included ({agentic['critic']:.2f})"
                if agentic["critic"] is not None else "Included"
            ),
        },
        {
            "Control": "Additional retrieval",
            "Baseline": f"{baseline['rounds']} round",
            "Agentic": f"{agentic['rounds']} rounds",
        },
    ])
    st.dataframe(safeguards, use_container_width=True, hide_index=True)
    st.caption(
        "These controls reduce unsupported claims, but they do not prove that "
        "hallucinations are impossible. The values above describe only the current "
        "Baseline and Agentic runs."
    )


def _start_workspace_job(
    mode: str,
    file_bytes: bytes,
    filename: str,
    chunking_mode: str,
):
    """Start one independent background preparation job."""
    current_job = st.session_state.workspace_jobs.get(mode)
    if current_job and not current_job.done():
        return

    from src.workspace_processor import prepare_workspace

    st.session_state.workspace_resources[mode]['ready'] = False
    st.session_state.workspace_resources[mode].pop('error', None)
    st.session_state.comparison_results[mode] = None
    cached_evaluation = st.session_state.get("evaluation_results") or {}
    evaluated_document = cached_evaluation.get("document_name")
    if evaluated_document and evaluated_document != filename:
        _clear_evaluation_cache()
    st.session_state.workspace_jobs[mode] = (
        st.session_state.workspace_executor.submit(
            prepare_workspace,
            file_bytes=file_bytes,
            filename=filename,
            mode=mode,
            chunking_mode=chunking_mode,
            embedder=st.session_state.embedder,
        )
    )


def _collect_workspace_jobs() -> bool:
    """Move completed background results into this Streamlit session."""
    changed = False
    for mode, future in list(st.session_state.workspace_jobs.items()):
        if not future or not future.done():
            continue

        changed = True
        st.session_state.workspace_jobs[mode] = None
        try:
            result = future.result()
            document_record = result.pop('document_record')
            st.session_state.workspace_resources[mode].update(result)
            st.session_state.documents = [document_record]

            # Keep legacy tabs aligned with the latest completed workspace.
            st.session_state.vector_store = result['vector_store']
            st.session_state.knowledge_graph = result['knowledge_graph']
            st.session_state.parent_chunks = result['parent_chunks']
            st.session_state.child_chunks = result['child_chunks']
            st.session_state.rag_initialized = True
            _save_demo_cache()
        except Exception as exc:
            st.session_state.workspace_resources[mode]['ready'] = False
            st.session_state.workspace_resources[mode]['error'] = str(exc)
    return changed


@st.fragment(run_every=1.0)
def display_workspace_job_status():
    """Poll background jobs without blocking either workspace button."""
    if _collect_workspace_jobs():
        st.rerun()

    running = [
        mode.title()
        for mode, future in st.session_state.workspace_jobs.items()
        if future and not future.done()
    ]
    if running:
        st.info(f"Preparing in background: {', '.join(running)}")


def display_comparison_workspace(uploaded_file=None):
    """Two independent document-processing and question-answer workspaces."""
    st.subheader("RAG Workspace Comparison")
    st.caption(
        "Run each workflow independently, then compare their evidence, "
        "quality checks, and response performance."
    )

    if uploaded_file is None and not st.session_state.documents:
        st.info("Choose a document in the sidebar, then prepare each workspace below.")

    baseline_job = st.session_state.workspace_jobs.get("baseline")
    agentic_job = st.session_state.workspace_jobs.get("agentic")
    baseline_processing = bool(baseline_job and not baseline_job.done())
    agentic_processing = bool(agentic_job and not agentic_job.done())

    baseline_col, agentic_col = st.columns(2, gap="large")

    with baseline_col:
        with st.container(border=True):
            st.markdown("### Baseline Workspace")
            st.caption("Vector retrieval → single LLM generation")
            st.markdown("**Document preparation**")
            st.caption(
                uploaded_file.name
                if uploaded_file is not None
                else "Choose a file in the sidebar."
            )
            if st.button(
                "Processing Baseline..." if baseline_processing else "Process Document for Baseline",
                use_container_width=True,
                key="process_baseline_workspace",
                disabled=uploaded_file is None or baseline_processing,
            ):
                _start_workspace_job(
                    "baseline",
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    st.session_state.chunking_mode,
                )
                st.rerun()
            if baseline_processing:
                st.info("Building vector index... You can prepare Agentic at the same time.")
            baseline_error = st.session_state.workspace_resources["baseline"].get("error")
            if baseline_error:
                st.error(f"Baseline preparation failed: {baseline_error}")
            baseline_ready = st.session_state.workspace_resources["baseline"]["ready"]
            if baseline_ready:
                st.success("Vector index ready")
            st.divider()
            baseline_question = st.text_area(
                "Baseline-only question",
                height=88,
                key="baseline_workspace_question",
            )
            if st.button(
                "Run Baseline Only",
                use_container_width=True,
                key="run_baseline_workspace",
                disabled=not baseline_ready,
            ):
                question = baseline_question.strip()
                if question:
                    message = _run_workspace_query("baseline", question)
                    _store_comparison_result("baseline", question, message)
                else:
                    st.warning("Enter a Baseline question first.")
            st.divider()
            _display_workspace_result("baseline")

    with agentic_col:
        with st.container(border=True):
            st.markdown("### Agentic Workspace")
            st.caption("Planning → hybrid retrieval → validation → review")
            st.markdown("**Document preparation**")
            st.caption(
                uploaded_file.name
                if uploaded_file is not None
                else "Choose a file in the sidebar."
            )
            if st.button(
                "Processing Agentic..." if agentic_processing else "Process Document for Agentic",
                type="primary",
                use_container_width=True,
                key="process_agentic_workspace",
                disabled=uploaded_file is None or agentic_processing,
            ):
                _start_workspace_job(
                    "agentic",
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    st.session_state.chunking_mode,
                )
                st.rerun()
            if agentic_processing:
                st.info("Building vector index and knowledge graph... Baseline remains available.")
            agentic_error = st.session_state.workspace_resources["agentic"].get("error")
            if agentic_error:
                st.error(f"Agentic preparation failed: {agentic_error}")
            if not agentic_processing:
                display_knowledge_graph_summary()
            st.divider()
            agentic_question = st.text_area(
                "Agentic-only question",
                height=88,
                key="agentic_workspace_question",
            )
            if st.button(
                "Run Agentic Only",
                use_container_width=True,
                key="run_agentic_workspace",
                disabled=not st.session_state.workspace_resources["agentic"]["ready"],
            ):
                question = agentic_question.strip()
                if question:
                    message = _run_workspace_query("agentic", question)
                    _store_comparison_result("agentic", question, message)
                else:
                    st.warning("Enter an Agentic question first.")
            st.divider()
            _display_workspace_result("agentic")

    st.divider()
    display_automatic_comparison()
    display_workspace_job_status()

def display_footer():
    """Display demo footer."""
    st.divider()
    st.markdown(
        f'<p class="demo-footer">'
        f'Agentic RAG Thesis Demo &nbsp;|&nbsp; '
        f'DeepSeek API, BGE, ChromaDB, LangGraph'
        f'</p>',
        unsafe_allow_html=True,
    )

def display_statistics():
    """Display system statistics with hierarchical info."""
    
    if st.session_state.documents and st.session_state.rag_initialized:
        st.subheader("📊 System Statistics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_chunks = sum(doc['chunks'] for doc in st.session_state.documents)
            st.metric("Total Chunks", total_chunks)
        
        with col2:
            total_parents = sum(doc.get('parents', 0) for doc in st.session_state.documents)
            st.metric("Parent Chunks", total_parents)
        
        with col3:
            queries_count = sum(
                1
                for result in st.session_state.comparison_results.values()
                if result and (result.get("message") or {}).get("content")
            )
            st.metric("Queries", queries_count)
        
        with col4:
            chunk_mode = st.session_state.chunking_mode
            chunk_label = "Hierarchical" if chunk_mode == "hierarchical" else "Flat"
            st.metric("Chunking", chunk_label)

        with col5:
            if st.session_state.chunking_mode == "hierarchical":
                context = "2k tok"
            else:
                context = "500 tok"
            st.metric("Context", context)
        
        st.divider()
def _display_cached_evaluation(evaluation_cache: dict) -> None:
    """Render a completed evaluation from live or restored state."""
    results = evaluation_cache["results"]
    workflow_rows = evaluation_cache["workflow_rows"]
    st.success(
        "Evaluation results restored from cache."
        if evaluation_cache.get("restored")
        else "Full Agentic Workflow Evaluation Complete!"
    )
    st.caption(
        f"Dataset: {evaluation_cache.get('dataset', 'Unknown')} | "
        f"Questions: {evaluation_cache.get('question_count', len(workflow_rows))} | "
        f"Completed: {evaluation_cache.get('completed_at', 'Unknown')}"
    )
    st.markdown("### Current Agentic Evaluation")
    st.caption(
        "These are heuristic scores for the current 8-question evaluation set, "
        "not dissertation benchmark accuracy."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = float(results["avg_overall"])
        color = "Green" if score >= 0.7 else "Amber" if score >= 0.5 else "Red"
        st.metric("Overall Heuristic Score", f"{score:.1%}", help=f"Status: {color}")
    with col2:
        st.metric("Citation Rate", f"{float(results['avg_citation_rate']):.1%}")
    with col3:
        st.metric("Context Usage", f"{float(results['avg_context_usage']):.1%}")
    with col4:
        st.metric(
            "Regeneration Rate",
            f"{float(results['improvement_rate']):.1%}",
            help="Percentage of answers that required an additional rewrite.",
        )

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Avg Quality Score", f"{float(results['avg_quality_score']):.1%}")
    with col6:
        st.metric("Avg Word Count", f"{float(results['avg_word_count']):.0f}")

    st.markdown("### Detailed Results")
    import pandas as pd

    rows = []
    for row_idx, score in enumerate(results["detailed_scores"]):
        workflow_row = workflow_rows[row_idx]
        rows.append({
            "Question": score["query"][:50] + "...",
            "Strategy": workflow_row["strategy"],
            "Heuristic": f"{float(score['overall']):.1%}",
            "Citations": "Pass" if score["has_citations"] else "Fail",
            "Validation": f"{float(workflow_row['validation_score']):.1%}",
            "Critic": f"{float(workflow_row['critic_score']):.1%}",
            "Reliable": "Pass" if workflow_row["reliability_passed"] else "Fail",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def display_evaluation_interface():
    """Display evaluation interface with custom evaluator."""
    
    st.subheader("System Evaluation")
    
    if not st.session_state.documents:
        st.info("Upload documents first to run evaluation")
        return
    
    st.markdown("""
    **Current evaluation checks:** citation presence, retrieved-context usage,
    answer completeness, validation, critic review, and regeneration.
    """)
    
    # Load test questions
    import json
    from pathlib import Path

    candidate_files = [
        Path("data/test_questions.json"),
        Path("data/evaluation/thesis_test_dataset.json"),
        Path("data/evaluation/test_dataset.json"),
        Path("tests/fixtures/test_dataset.json"),
    ]
    test_file = next((path for path in candidate_files if path.exists()), None)

    if test_file is None:
        st.warning("⚠️ No evaluation dataset found")
        st.info(
            "Expected one of: data/test_questions.json, "
            "data/evaluation/thesis_test_dataset.json, "
            "data/evaluation/test_dataset.json, tests/fixtures/test_dataset.json"
        )
        return

    with open(test_file, 'r') as f:
        test_data = json.load(f)

    if "questions" in test_data:
        questions = test_data.get("questions", [])
    elif "test_cases" in test_data:
        questions = [
            case.get("question", "")
            for case in test_data.get("test_cases", [])
            if case.get("question")
        ]
    else:
        questions = []

    if not questions:
        st.warning(f"No questions found in test file: {test_file}")
        return

    st.info(f"📝 Loaded {len(questions)} test questions from `{test_file}`")
    
    # Show sample questions
    with st.expander("👁️ View Test Questions"):
        for i, q in enumerate(questions, 1):
            st.write(f"{i}. {q}")
    
    # Run evaluation button
    if st.button("🚀 Run Evaluation", type="primary"):
        from src.evaluation.simple_evaluator import SimpleEvaluator
        from src.config import get_settings
        from src.llm.chat_model import create_chat_model
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

        if not _llm_api_key_ok():
            st.error(
                "Set a valid OPENROUTER_API_KEY in `.env`, then refresh and try again."
            )
            return

        evaluator = SimpleEvaluator()

        with st.spinner("Initializing complete Agentic RAG workflow..."):
            settings = get_settings()
            planner_llm = create_chat_model(settings, model=settings.get_agent_model("planner"))
            decomposer_llm = create_chat_model(settings, model=settings.get_agent_model("decomposer"), max_tokens=1000)
            validator_llm = create_chat_model(settings, model=settings.get_agent_model("validator"))
            writer_llm = create_chat_model(settings, model=settings.get_agent_model("writer"))
            critic_llm = create_chat_model(settings, model=settings.get_agent_model("critic"))

            vector_agent = VectorSearchAgent(
                vector_store=st.session_state.vector_store,
                embedder=st.session_state.embedder,
            )
            keyword_agent = KeywordSearchAgent(
                vector_store=st.session_state.vector_store,
            )
            graph_agent = (
                GraphSearchAgent(
                    knowledge_graph=st.session_state.knowledge_graph,
                    vector_store=st.session_state.vector_store,
                )
                if st.session_state.knowledge_graph
                else None
            )

            coordinator = RetrievalCoordinator(
                vector_agent=vector_agent,
                keyword_agent=keyword_agent,
                graph_agent=graph_agent,
            )

            workflow = CompleteAgenticRAGWorkflow(
                planner=PlannerAgent(llm=planner_llm),
                decomposer=QueryDecomposer(llm=decomposer_llm),
                coordinator=coordinator,
                validator=ValidatorAgent(llm=validator_llm),
                synthesis=SynthesisAgent(),
                writer=WriterAgent(llm=writer_llm),
                critic=CriticAgent(llm=critic_llm, quality_threshold=0.7),
            )

        # Process each question
        all_answers = []
        all_chunks_list = []
        all_metadata = []
        all_workflow_rows = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, question in enumerate(questions):
            status_text.text(f"Running full Agentic workflow {i+1}/{len(questions)}...")

            result = workflow.run(question)

            # Store results
            all_answers.append(result.answer or "")
            all_chunks_list.append(result.chunks)
            all_metadata.append({
                "self_reflection": {
                    "iterations": result.metadata.get("regeneration_count", 0),
                    "final_score": result.critic_score or 0.0,
                    "final_decision": (
                        result.critic_decision.value
                        if result.critic_decision and hasattr(result.critic_decision, "value")
                        else str(result.critic_decision)
                    ),
                    "improved": result.metadata.get("regeneration_count", 0) > 0,
                }
            })
            all_workflow_rows.append({
                "strategy": (
                    result.strategy.value
                    if hasattr(result.strategy, "value")
                    else str(result.strategy)
                ),
                "retrieval_rounds": result.retrieval_round,
                "validation_score": result.validation_score or 0.0,
                "critic_score": result.critic_score or 0.0,
                "reliability_passed": result.metadata.get("reliability_gate", {}).get("passed"),
                "chunk_count": len(result.chunks),
            })
            
            progress_bar.progress((i + 1) / len(questions))
        
        status_text.empty()
        progress_bar.empty()
        
        # Evaluate
        results = evaluator.evaluate_batch(
            questions, all_answers, all_chunks_list, all_metadata
        )
        
        st.session_state.evaluation_results = {
            "results": results,
            "workflow_rows": all_workflow_rows,
            "dataset": str(test_file),
            "document_name": (
                st.session_state.documents[0].get("name")
                if st.session_state.documents else None
            ),
            "question_count": len(questions),
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _save_evaluation_cache(st.session_state.evaluation_results)
        _save_demo_cache()

    evaluation_cache = st.session_state.get("evaluation_results")
    if evaluation_cache:
        _display_cached_evaluation(evaluation_cache)

def display_document_preview():
    """Show preview of uploaded documents."""
    
    if st.session_state.documents:
        st.subheader("👁️ Document Preview")
        
        with st.expander("📄 View Document Details", expanded=False):
            # Select document to preview
            doc_names = [doc['name'] for doc in st.session_state.documents]
            selected_doc = st.selectbox(
                "Select document to preview",
                doc_names,
                key="doc_preview_selector"
            )
            
            # Find selected document
            doc = next((d for d in st.session_state.documents if d['name'] == selected_doc), None)
            
            if doc:
                # Document metadata
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Type", doc.get('type', 'PDF'))
                
                with col2:
                    st.metric("Parent Chunks", doc.get('parents', 0))
                
                with col3:
                    st.metric("Child Chunks", doc['chunks'])
                
                with col4:
                    st.metric("Chunking", "Parent-Child")
                
                # Show chunking info
                if doc.get('chunking_mode') == 'hierarchical':
                    st.info(f"🔺 Hierarchical: {doc.get('parents', 0)} parents, {doc['chunks']} children")
                else:
                    st.info(f"📊 Flat: {doc['chunks']} chunks")
                
                st.caption(f"📅 Uploaded: {doc.get('uploaded_at', 'Unknown')}")
                
                # Load and show preview (optional - can be slow)
                if st.button("📖 Load Content Preview", key=f"preview_{selected_doc}"):
                    try:
                        from src.ingestion.document_loader import DocumentLoader
                        loader = DocumentLoader()
                        loaded_document = loader.load(doc['path'])
                        text = loaded_document.text
                        
                        # Show first 1000 characters
                        preview_text = text[:1000]
                        if len(text) > 1000:
                            preview_text += "..."
                        
                        st.text_area(
                            "Content Preview (first 1000 chars)",
                            preview_text,
                            height=200,
                            key=f"preview_text_{selected_doc}"
                        )
                        
                        # Statistics
                        st.caption(f"Total: {len(text):,} chars | {len(text.split()):,} words")
                        
                    except Exception as e:
                        st.error(f"Error loading preview: {e}")

        display_chunk_structure()


def display_chunk_structure():
    """Inspect the actual parent-child records stored in ChromaDB."""
    workspace = next(
        (
            st.session_state.workspace_resources[mode]
            for mode in ("agentic", "baseline")
            if st.session_state.workspace_resources[mode].get("ready")
        ),
        None,
    )
    store = (workspace or {}).get("vector_store")
    if not store:
        return

    with st.expander("Parent-Child Internal Structure", expanded=False):
        try:
            parent_data = store.parent_collection.get(
                include=["documents", "metadatas"]
            )
            child_data = store.child_collection.get(
                include=["documents", "metadatas"]
            )
            parents = [
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                }
                for chunk_id, text, metadata in zip(
                    parent_data.get("ids", []),
                    parent_data.get("documents", []),
                    parent_data.get("metadatas", []),
                )
            ]
            children = [
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                }
                for chunk_id, text, metadata in zip(
                    child_data.get("ids", []),
                    child_data.get("documents", []),
                    child_data.get("metadatas", []),
                )
            ]
            if not parents:
                st.info("No parent chunks are stored for this document.")
                return

            st.caption(
                f"Stored hierarchy: {len(parents)} parent chunk(s) and "
                f"{len(children)} child chunk(s)."
            )
            parent_ids = [parent["id"] for parent in parents]
            selected_parent_id = st.selectbox(
                "Select a parent chunk",
                parent_ids,
                format_func=lambda value: f"Parent {parent_ids.index(value) + 1}: {value}",
                key="chunk_structure_parent",
            )
            parent = next(item for item in parents if item["id"] == selected_parent_id)
            linked_children = [
                child
                for child in children
                if child["metadata"].get("parent_id") == selected_parent_id
            ]

            parent_a, parent_b = st.columns(2)
            with parent_a:
                st.metric("Parent tokens", parent["metadata"].get("token_count", "N/A"))
            with parent_b:
                st.metric("Linked children", len(linked_children))
            st.code(f"Parent ID: {selected_parent_id}", language="text")
            st.text_area(
                "Parent text",
                parent["text"],
                height=180,
                disabled=True,
                key=f"parent_text_{selected_parent_id}",
            )

            st.markdown("**Children linked to this parent**")
            for index, child in enumerate(linked_children, 1):
                metadata = child["metadata"]
                with st.container(border=True):
                    child_a, child_b = st.columns([3, 1])
                    with child_a:
                        st.code(
                            f"Child {index} ID: {child['id']}\n"
                            f"Parent ID: {metadata.get('parent_id', 'N/A')}",
                            language="text",
                        )
                    with child_b:
                        st.metric("Tokens", metadata.get("token_count", "N/A"))
                    st.text_area(
                        f"Child {index} text",
                        child["text"],
                        height=130,
                        disabled=True,
                        key=f"child_text_{child['id']}",
                    )
        except Exception as exc:
            st.error(f"Could not load the stored chunk structure: {exc}")
                        
def display_chat_messages():
    """Display chat message history."""
    
    if not st.session_state.messages:
        if st.session_state.get("rag_mode", "agentic") == "baseline":
            st.markdown(
                "#### Welcome\n\n"
                "1. **Upload** a document in the sidebar and wait for indexing  \n"
                "2. Choose **Agentic RAG** or **Baseline** for comparison  \n"
                "3. Ask a question below\n\n"
                "**Current pipeline:** vector retrieval → single LLM generation\n\n"
                "**Not used:** BM25, graph retrieval, planning, validation, "
                "criticism, reliability gate"
            )
        else:
            st.markdown(
                "#### Welcome\n\n"
                "1. **Upload** a document in the sidebar and wait for indexing  \n"
                "2. Choose **Agentic RAG** or **Baseline** for comparison  \n"
                "3. Ask a question below\n\n"
                "**Proposed system:** multi-agent orchestration, hybrid retrieval, "
                "self-reflection, GraphRAG"
            )
        return
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("workflow_metadata"):
                meta = message["workflow_metadata"]
                if meta.get("rag_mode") == "baseline":
                    st.caption("Baseline | naive vector RAG")
                else:
                    strategy = _format_strategy_label(meta.get("strategy"))
                    st.caption(f"Agentic | strategy: {strategy}")

            st.markdown(message["content"])
            
            # Show self-reflection info (NEW)
            if message["role"] == "assistant" and "self_reflection" in message:
                reflection = message["self_reflection"]
                
                # Create compact info box
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    iterations = reflection.get("iterations", 0)
                    emoji = "🔄" if iterations > 0 else "✅"
                    st.metric("Iterations", f"{emoji} {iterations}")
                
                with col2:
                    score = reflection.get("final_score", 0.0)
                    color = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
                    st.metric("Quality Score", f"{color} {score:.2f}")
                
                with col3:
                    improved = reflection.get("improved", False)
                    st.metric("Improved", "✅ Yes" if improved else "➖ No")
                
                with col4:
                    decision = reflection.get("decision", "unknown")
                    st.metric("Status", decision.upper())
            
            # Show citations with chunk type
            if message["role"] == "assistant" and "citations" in message:
                if message["citations"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for citation in message["citations"]:
                            # Extract chunk info
                            chunk_type = citation.get('chunk_type', 'unknown')
                            chunk_emoji = "📄" if chunk_type == "parent" else "📝"
                            
                            st.caption(
                                f"**[{citation['source_number']}] {citation.get('filename', 'unknown')}** "
                                f"{chunk_emoji} ({chunk_type} chunk, Relevance: {citation['score']:.2%})"
                            )
                            st.text(citation['text_preview'])        

def display_chat_input():
    """Display the single-mode input inside the Chat workspace."""
    
    # Only show if documents uploaded
    if not st.session_state.documents:
        st.info("Upload a document in the sidebar to start.")
        return
    
    # Handle sample query (if triggered from sidebar)
    if 'sample_query' in st.session_state and st.session_state.sample_query:
        query = st.session_state.sample_query
        st.session_state.sample_query = None  # Clear it
        process_user_query(query)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
        return
    
    # Chat input (must be at root level, not in tabs/columns/expander)
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not _llm_api_key_ok():
            st.error("Configure OPENROUTER_API_KEY in `.env` and refresh the page.")
            return
        process_user_query(prompt)
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()


def display_single_chat_workspace(uploaded_file=None):
    """Render the independently selected single-mode chat workspace."""
    mode = st.selectbox(
        "Chat workspace",
        options=["agentic", "baseline"],
        format_func=lambda value: {
            "agentic": "Agentic workspace (proposed)",
            "baseline": "Baseline workspace (naive RAG)",
        }[value],
        key="rag_mode",
    )

    process_label = (
        "Process Document for Agentic"
        if mode == "agentic"
        else "Process Document for Baseline"
    )
    if st.button(
        process_label,
        type="primary",
        key="process_single_chat_workspace",
        disabled=uploaded_file is None,
    ):
        process_uploaded_file(uploaded_file, processing_rag_mode=mode)

    if uploaded_file is None:
        st.caption("Choose a document in the sidebar before processing.")

    if mode == "agentic" and st.session_state.rag_initialized:
        with st.expander("Knowledge Graph", expanded=False):
            display_knowledge_graph_summary()
    elif mode == "baseline" and st.session_state.rag_initialized:
        st.info(
            "Baseline uses vector retrieval only. Knowledge graph and agent "
            "quality checks are disabled."
        )

    if st.session_state.documents:
        st.markdown("**Sample questions**")
        sample_cols = st.columns(2)
        sample_questions = [
            "What is this document about?",
            "Summarize the main points",
            "What are the key findings?",
            "Give me specific details",
        ]
        for index, question in enumerate(sample_questions):
            with sample_cols[index % 2]:
                if st.button(question, key=f"sample_{question}"):
                    st.session_state.sample_query = question

    st.divider()
    display_chat_messages()
    display_chat_input()


def main():
    """Main application."""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Sidebar
    uploaded_file = sidebar()
   
    # Main content tabs
    tab_compare, tab_eval, tab_stats = st.tabs(
        ["Compare", "Evaluation", "Statistics"]
    )

    with tab_compare:
        display_comparison_workspace(uploaded_file)

    with tab_eval:
        # Evaluation interface
        display_evaluation_interface()
    
    with tab_stats:
        # Statistics and preview
        display_statistics()
        display_document_preview()

    # Footer
    display_footer()


if __name__ == "__main__":
    main()
