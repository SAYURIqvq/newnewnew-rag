# Thesis Experiment Guide

How to produce: **experiment table**, **case studies**, and **architecture figures** for your thesis.

---

## Prerequisites

1. Documents indexed in `data/chroma_db` (upload via Streamlit first).
2. `.env` contains valid `ANTHROPIC_AUTH_TOKEN`.
3. BM25 index built (optional but helps Agentic): `python build_bm25_index.py`

---

## Step 1 — Build your question set (15–30 min)

Edit `data/evaluation/thesis_test_dataset.json`:

- Use **10–20 questions** about **your uploaded documents** (not only generic ML questions).
- Mix categories: `definition`, `explanation`, `comparison`, `relationship`, `process`.
- Each row needs `question` and `ground_truth` (reference answer for RAGAS).

Example:

```json
{
  "question": "What is the main topic of the uploaded report?",
  "ground_truth": "One sentence from your document that answers this.",
  "difficulty": "easy",
  "category": "definition"
}
```

---

## Step 2 — Run the benchmark (generates the experiment table)

```bash
cd /path/to/agentic-rag-system

# Full run: Baseline + Agentic answers + RAGAS scores (uses API, ~2–5 min per question)
python evaluation/run_thesis_benchmark.py

# Quick test on 2 questions only
python evaluation/run_thesis_benchmark.py --limit 2

# Answers only (no RAGAS) — fast, add RAGAS later
python evaluation/run_thesis_benchmark.py --no-ragas

# Re-score saved answers with RAGAS
python evaluation/run_thesis_benchmark.py --ragas-only
```

### Outputs (under `results/thesis/`)

| File | Use in thesis |
|------|----------------|
| `experiment_table.csv` | Main Table: per-question Baseline vs Agentic metrics |
| `summary_table.md` | Table 2: average Faithfulness, Relevancy, etc. |
| `case_studies/case_01.md` … | Case study text (screenshot + paste) |
| `answers.json` | Raw answers for re-run / audit |

Copy `summary_table.md` into Word/LaTeX as **Table 5-1**.

---

## Step 3 — Case study screenshots (30 min)

Pick **3 questions** from `case_studies/` where Agentic clearly wins (or Baseline fails).

For each question:

1. Streamlit → sidebar → **Baseline** → ask question → screenshot answer.
2. Same question → **Agentic RAG** → screenshot answer.
3. Paste into thesis as **Figure 5-x** with caption:  
   *"Comparison on Q3: Baseline (left) vs proposed Agentic RAG (right)."*

Optional: export chat from UI or copy from `case_studies/case_XX.md`.

---

## Step 4 — Architecture & innovation (copy into thesis)

Use `docs/THESIS_ARCHITECTURE.md`:

- Figure: system architecture (Mermaid → export PNG via https://mermaid.live)
- Figure: Baseline vs Agentic pipeline
- Section: innovation bullets (English or translate to Chinese)

---

## Suggested thesis chapter structure

### Chapter 5 Experiments

1. **Setup** — hardware, DeepSeek, BGE, ChromaDB, dataset size, N questions.
2. **Metrics** — Faithfulness, Answer Relevancy, Context Precision/Recall (RAGAS).
3. **Main results** — paste `summary_table.md`.
4. **Per-category analysis** — group CSV by `category` column.
5. **Case studies** — 2–3 figures from Streamlit.
6. **Discussion** — Agentic slower but higher faithfulness; Baseline enough for simple factual QA.

---

## Minimal acceptable experiment (if short on time)

- **8 questions** in `thesis_test_dataset.json`
- Run benchmark once → one summary table
- **2 case study screenshots**
- **1 architecture diagram**

This is enough for many undergraduate theses.
