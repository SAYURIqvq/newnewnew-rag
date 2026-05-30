# Agentic RAG System

**A Hierarchical Multi-Agent Framework for Reliable RAG with Self-Reflection and Graph-Based Reasoning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-230+-brightgreen.svg)]()

An intelligent document Q&A system built on a hierarchical multi-agent architecture with self-reflection, graph-based reasoning, and adaptive query strategies.

---

## What Makes This Different

**Traditional RAG:**
```
Query → Retrieve chunks → Generate answer

- Fixed pipeline
- No quality checks on retrieval or generation
- Cannot reason about relationships between entities
- Hallucinations go undetected
```

**This system:**
```
Query → Planner analyses complexity
      → Adaptive retrieval (vector + keyword in parallel)
      → Validator checks retrieval quality, retries if needed
      → Synthesis deduplicates and ranks across sources
      → Writer generates answer grounded in context
      → Critic reviews quality, triggers regeneration if needed
      → Reliability Gate performs final grounding check
      → Answer with full citation trail

- Adaptive decisions based on query complexity
- Self-reflection with automatic quality correction
- Relationship reasoning via knowledge graph
- Runtime reliability gate catches grounding failures
- Honest when information is not available
```

---

## Key Features

### Multi-Agent Architecture (10 agents, 3 hierarchical levels)

**Strategic Layer (L1)**
- **Planner** — Analyses query complexity (0.0–1.0), selects strategy (Simple / Multi-hop / Graph)

**Tactical Layer (L2)**
- **Query Decomposer** — Breaks complex multi-aspect questions into focused sub-queries
- **Retrieval Coordinator** — Spawns and manages swarm agents in parallel
- **Validator** — Quality gate; checks chunk sufficiency, triggers re-retrieval if needed
- **Synthesis** — Deduplicates across retrieval methods, applies hybrid scoring
- **Writer** — Generates answers grounded strictly in context with inline citations
- **Critic** — Reviews quality on 5 dimensions, regenerates with feedback if needed

**Operational Layer (L3 — Swarm, runs in parallel)**
- **Vector Agent** — Semantic search via BGE embeddings
- **Keyword Agent** — Exact-match BM25 scoring
- **Graph Agent** — Relationship reasoning via knowledge-graph path finding

**Reliability Gate** — Final deterministic check for citations, context grounding, and prior agent scores before the answer reaches the user.

---

### Self-Reflection Loop

Two-stage quality check before an answer reaches the user:

**Stage 1 — Retrieval (Validator)**
```
Chunks retrieved → score relevance, coverage, confidence
  score ≥ 0.7 → proceed
  score < 0.7 → re-retrieve (max 2 retries)
```

**Stage 2 — Generation (Critic)**
```
Answer generated → score on 5 dimensions
  Accuracy 30% | Completeness 25% | Citations 15% | Clarity 15% | Relevance 15%
  score ≥ threshold → approve
  score < threshold → regenerate with feedback (max 3 iterations)
```

---

### Graph-Based Reasoning

Builds searchable knowledge graphs directly from uploaded documents:

- Entity extraction (spaCy NER)
- Relationship extraction — co-occurrence, dependency parsing, pattern matching
- Graph construction (NetworkX)
- Path finding for relationship queries

```
"How does X relate to Y?"
→ Path found: X --[relation]--> Z --[relation]--> Y
→ Retrieves chunks along the path
→ Answer includes graph reasoning evidence
```

---

### Adaptive Strategy Selection

```
Simple query   (complexity < 0.3)   → Vector search → direct generation
Complex query  (complexity 0.3–0.7) → Decompose → parallel retrieval → synthesis
Relationship   (complexity > 0.7)   → Graph path finding → entity retrieval
```

---

## Architecture

```
                          ┌──────────────────┐
                          │  USER (Streamlit) │
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │   PLANNER (L1)    │
                          │ complexity→strategy│
                          └────────┬─────────┘
                                   ▼
                          ┌──────────────────┐
                          │ QUERY DECOMPOSER  │
                          └────────┬─────────┘
                                   ▼
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
         ┌──────────┐      ┌──────────┐      ┌──────────┐
         │  Vector  │      │ Keyword  │      │  Graph   │  L3 Swarm
         │  Agent   │      │  Agent   │      │  Agent   │  (parallel)
         └────┬─────┘      └────┬─────┘      └────┬─────┘
              └────────────────┼─────────────────┘
                               ▼
                     ┌──────────────────┐
                     │  VALIDATOR (L2)   │◄── retry (max 2)
                     │  quality gate     │
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │  SYNTHESIS (L2)   │
                     │ dedupe+hybrid rank│
                     └────────┬─────────┘
                              ▼
                     ┌──────────────────┐
                     │   WRITER (L2)     │◄──────────┐
                     │ generate+citations│           │
                     └────────┬─────────┘           │
                              ▼                      │
                     ┌──────────────────┐           │
                     │   CRITIC (L2)     │           │
                     │ review→regenerate?│──┐        │ regenerate
                     └────────┬─────────┘  │        │ (max 3)
                              ▼             │        │
                     ┌──────────────────┐  │        │
                     │ RELIABILITY GATE  │  │        │
                     │ grounding check   │  │        │
                     └────────┬─────────┘  │        │
                              ▼             │        │
                     ┌──────────────────┐  │        │
                     │ Answer+Citations  │  │        │
                     └──────────────────┘  │        │
                                           │        │
              ┌────────────────────────────┘        │
              │  Two feedback loops:                │
              │  1. Validator → back to Retrieval ◄─┘
              │  2. Writer ↔ Critic self-reflection
              └─────────────────────────────────────
```

---

## Quick Start

### Prerequisites

Python 3.12+, Git, DeepSeek-compatible API key

### Installation

```bash
git clone https://github.com/SAYURIqvq/newnewnew-rag.git
cd newnewnew-rag

python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_md

cp .env.example .env
# Fill in:
#   ANTHROPIC_AUTH_TOKEN=your_deepseek_api_key
#   ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
#   EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

### Run

```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Usage

### 1. Upload a Document

Supports PDF, DOCX, TXT. Processing is fully automatic:
text extraction → hierarchical chunking → embeddings → vector store → entity extraction → knowledge graph → BM25 index.

### 2. Ask Questions

```
Simple:        "What is machine learning?"
               → fast path

Relationship:  "How does TensorFlow relate to neural networks?"
               → graph reasoning

Complex:       "Compare supervised and unsupervised learning"
               → multi-hop decomposition
```

### 3. Read the Answer

Every answer includes inline citations (`[1]`, `[2]`, …) tracing each claim to its source chunk. When information is not available, the system says so explicitly.

---

## Technology Stack

| Layer | Technology | Role |
|-------|-----------|------|
| LLM | DeepSeek via Anthropic-compatible API | Generation, validation, critique |
| Embeddings | BGE-large (`BAAI/bge-large-en-v1.5`) | Semantic vectors |
| Orchestration | LangChain + LangGraph | Agent wiring, state-machine workflows |
| Vector DB | ChromaDB | Persistent vector storage |
| Graph | NetworkX | Knowledge graph + path finding |
| NLP | spaCy `en_core_web_md` | NER, dependency parsing |
| Evaluation | RAGAS + custom SimpleEvaluator | Answer quality scoring |
| Frontend | Streamlit | Web interface |

---

## Project Structure

```
newnewnew-rag/
├── app.py                              # Streamlit entry point
├── src/
│   ├── agents/                         # Agent implementations
│   │   ├── planner.py                  # L1 — strategy selection
│   │   ├── query_decomposer.py         # L2 — multi-hop decomposition
│   │   ├── retrieval_coordinator.py    # L2 — swarm orchestration
│   │   ├── validator.py                # L2 — retrieval quality gate
│   │   ├── synthesis.py                # L2 — dedupe + ranking
│   │   ├── writer.py                   # L2 — answer generation
│   │   ├── critic.py                   # L2 — answer review
│   │   ├── reliability_gate.py         # Final grounding check
│   │   ├── self_reflection.py          # Self-reflection agent
│   │   ├── graph_search_agent.py       # Graph-based search agent
│   │   ├── graph_traversal_agent.py    # Graph traversal logic
│   │   └── retrieval/                  # Legacy retrieval agents (test only)
│   ├── retrieval/                      # Production retrieval
│   │   ├── vector_search.py            # BGE semantic search
│   │   ├── keyword_search.py           # BM25 keyword search
│   │   ├── graph_search.py             # Knowledge graph search
│   │   ├── graph_retrieval.py          # Graph-enhanced retrieval
│   │   └── bm25_index.py               # BM25 index builder
│   ├── graph/                          # GraphRAG pipeline
│   │   ├── entity_extractor.py
│   │   ├── relationship_extractor.py
│   │   └── graph_builder.py
│   ├── ingestion/                      # Document processing
│   │   ├── document_loader.py
│   │   ├── hierarchical_chunker.py
│   │   └── embedder.py
│   ├── storage/                        # Persistence
│   │   ├── chroma_store.py
│   │   └── database.py
│   ├── evaluation/                     # Quality measurement
│   │   ├── ragas_evaluator.py          # RAGAS (DeepSeek + BGE)
│   │   └── simple_evaluator.py         # Lightweight rule-based metrics
│   ├── orchestration/                  # LangGraph workflows
│   │   ├── complete_workflow.py        # Full 7-node pipeline
│   │   ├── langgraph_workflow.py
│   │   └── multihop_handler.py
│   ├── models/                         # Pydantic data models
│   └── config.py                       # Centralised settings
├── tests/
│   ├── unit/                           # Isolated component tests
│   ├── integration/                    # Multi-component flow tests
│   ├── evaluation/                     # RAGAS pipeline tests
│   └── e2e/                            # End-to-end workflow tests
├── docs/                               # Architecture & evaluation reports
├── data/                               # Runtime data (.gitignore'd)
│   ├── chroma_db/
│   ├── evaluation/
│   └── uploads/
├── evaluation/                         # Benchmark scripts
│   └── run_thesis_benchmark.py
├── .env.example
└── requirements.txt
```

---

## Testing

```bash
# Full unit test suite — 230+ tests
pytest tests/unit/ -v

# By layer
pytest tests/unit/                                     # Unit tests
pytest tests/integration/                              # Integration tests
pytest tests/e2e/                                      # End-to-end tests

# Reliability gate tests
pytest tests/unit/test_reliability_gate.py -v          # 5 tests

# RAGAS pipeline — mocked, no API cost
pytest tests/evaluation/test_ragas_evaluation.py -v

# Thesis benchmark (needs indexed docs + API key)
python evaluation/run_thesis_benchmark.py --limit 2
```

---

## Evaluation

The system uses a custom `SimpleEvaluator` for lightweight runtime metrics:

| Metric | What it measures |
|--------|-----------------|
| Citation Rate | Whether answers include inline citations to source chunks |
| Context Usage | Whether retrieved chunks are actually used in the answer |
| Answer Quality | Substantial, complete, and grounded responses |
| Self-Reflection | Improvement through critic-triggered regeneration |

RAGAS (Faithfulness, Answer Relevancy, Context Precision/Recall) is available offline for detailed analysis via `src/evaluation/ragas_evaluator.py`.

---

## Documentation

| Doc | What it covers |
|-----|----------------|
| [RAGAS Evaluation Report](docs/RAGAS_EVALUATION_REPORT.md) | Methodology, scores, production-gate logic |
| [Thesis Architecture](docs/THESIS_ARCHITECTURE.md) | Agent hierarchy, data flow, component interaction |
| [Ablation Report](docs/ABLATION_REPORT.md) | Quantified impact of each subsystem |
| [User Guide](docs/USER_GUIDE.md) | End-user how-to |

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgments

- **DeepSeek** — LLM API
- **BAAI BGE** — Embedding model
- **Microsoft Research** — GraphRAG methodology
- **LangChain / LangGraph** — Orchestration framework
- **RAGAS** — Evaluation framework
