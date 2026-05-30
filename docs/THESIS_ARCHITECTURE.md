# Thesis Architecture & Innovation Points

Copy diagrams to [Mermaid Live Editor](https://mermaid.live) and export as PNG for your thesis.

---

## Figure 1 — System architecture

```mermaid
flowchart TB
  subgraph UI["Presentation Layer"]
    ST[Streamlit Web UI]
  end

  subgraph Orchestration["Agent Orchestration (LangGraph)"]
    P[Planner]
    D[Query Decomposer]
    RC[Retrieval Coordinator]
    V[Validator]
    SY[Synthesis]
    W[Writer]
    CR[Critic]
  end

  subgraph Retrieval["Retrieval Layer"]
    VS[Vector Search]
    BM[BM25 Keyword]
    GR[Graph Search]
  end

  subgraph Storage["Storage Layer"]
    CH[(ChromaDB)]
    KG[(Knowledge Graph)]
  end

  subgraph Models["Models"]
    LLM[DeepSeek LLM]
    EMB[BGE Embeddings]
  end

  ST --> P --> D --> RC
  RC --> VS & BM & GR
  VS & BM --> CH
  GR --> KG
  RC --> V --> SY --> W --> CR
  W --> CR
  CR -->|regenerate| W
  V -->|retry| RC
  VS --> EMB
  W & P & V & CR --> LLM
  ST --> ST
```

---

## Figure 2 — Baseline vs proposed pipeline

```mermaid
flowchart LR
  subgraph B["Baseline (comparison)"]
    Q1[Query] --> E1[Embed]
    E1 --> R1[Vector Top-K]
    R1 --> G1[Single LLM call]
    G1 --> A1[Answer]
  end

  subgraph A["Agentic RAG (proposed)"]
    Q2[Query] --> P2[Planner]
    P2 --> R2[Parallel retrieval]
    R2 --> V2[Validator]
    V2 --> S2[Synthesis]
    S2 --> W2[Writer]
    W2 --> C2[Critic]
    C2 --> A2[Answer]
  end
```

---

## Innovation points (thesis text)

### 1. Hierarchical multi-agent orchestration

Unlike fixed pipelines, the system selects strategies (simple / multi-hop / graph) via a **Planner** and decomposes complex queries before retrieval.

### 2. Hybrid retrieval swarm

**Vector**, **BM25**, and **graph** retrieval run in parallel; **Synthesis** fuses and deduplicates evidence before generation.

### 3. Dual quality control

- **Validator**: retrieval sufficiency, optional re-retrieval.  
- **Critic**: answer quality loop with regeneration.

### 4. GraphRAG for relational questions

Entity–relation paths support questions that pure vector search cannot answer reliably.

### 5. Reproducible baseline comparison

The same index and LLM power a **Naive RAG** baseline, enabling fair comparison in experiments (RAGAS metrics + case studies).

---

## Suggested figure captions (English)

- **Figure 4-1** Overall architecture of the proposed Agentic RAG system.  
- **Figure 4-2** Comparison between Baseline naive RAG and the proposed multi-agent pipeline.  
- **Table 5-1** Average RAGAS scores on the thesis test set (Baseline vs Agentic).

Translate captions to Chinese if your thesis is written in Chinese.
