"""
RAGAS-based evaluation for RAG system.
Override default OpenAI models with the project's OpenRouter LLM
and BGE embedding model.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.embeddings import Embeddings
from src.config import get_settings
from src.llm.chat_model import create_chat_model

class BGELargeEmbeddings(Embeddings):
    """
    Custom Embeddings class untuk RAGAS.
    Pakai Sentence-Transformers (BGE-large) sama seperti retrieval kita.
    """

    def __init__(self):
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        _os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import SentenceTransformer
        settings = get_settings()
        self.model = settings.embedding_model
        self.client = SentenceTransformer(self.model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed list of documents."""
        if not texts:
            return []
        embs = self.client.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed single query."""
        emb = self.client.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return emb.tolist()

# ========== RAGAS EVALUATOR ==========

class RAGASEvaluator:
    """
    Evaluate RAG system using RAGAS framework.
    Uses OpenRouter (LLM) + BGE-large (embeddings).
    """

    def __init__(self, model: str = None):
        settings = get_settings()
        if not settings.anthropic_auth_token:
            raise ValueError("OPENROUTER_API_KEY not found in .env")

        model = model or settings.llm_model
        llm = create_chat_model(settings, model=model, temperature=0.0)
        self.llm = LangchainLLMWrapper(llm)

        # Override Embeddings → BGE-large (Sentence-Transformers)
        self.embeddings = LangchainEmbeddingsWrapper(BGELargeEmbeddings())

        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall
        ]

        print("📊 RAGAS Evaluator initialized")
        print(f"   LLM: OpenRouter ({model})")
        print(f"   Embeddings: BGE-large ({self.embeddings})")
        print(f"   Metrics: {len(self.metrics)}")

    def evaluate_rag_system(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Evaluate RAG system performance."""
        print(f"\n📊 Evaluating {len(questions)} test cases...")

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }

        dataset = Dataset.from_dict(data)

        print("   Running RAGAS evaluation with OpenRouter + BGE-large...")

        # ← Pass BOTH llm and embeddings
        results = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings  # ← KEY FIX
        )

        scores = {
            'answer_relevancy': float(results['answer_relevancy']),
            'faithfulness': float(results['faithfulness']),
            'context_precision': float(results['context_precision']),
            'context_recall': float(results['context_recall']),
        }
        scores['overall'] = sum(scores.values()) / len(scores)

        print(f"\n   ✅ Evaluation complete!")
        print(f"      Answer Relevancy:  {scores['answer_relevancy']:.3f}")
        print(f"      Faithfulness:      {scores['faithfulness']:.3f}")
        print(f"      Context Precision: {scores['context_precision']:.3f}")
        print(f"      Context Recall:    {scores['context_recall']:.3f}")
        print(f"      Overall Score:     {scores['overall']:.3f}")

        return scores

    def evaluate_single_case(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """Evaluate single Q&A case."""
        return self.evaluate_rag_system(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth]
        )

    def load_test_dataset(self, filepath: str) -> List[Dict[str, Any]]:
        """Load test dataset dari JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['test_cases']
    
    HONESTY_PHRASES = [
        "do not contain information",
        "not found in",
        "not available in",
        "does not mention",
        "no information about",
        "not covered in",
    ]

    def check_production_gate(self, scores: Dict[str, float], answer: str = "") -> Dict[str, Any]:
        """
        Production gate with honesty detection.

        Logic:
        - Faithfulness >= 0.5       → Hard gate
        - Overall >= 0.7            → Soft gate
        - Honest non-answer detected → Relevancy tidak di-penalize
        """
        FAITHFULNESS_THRESHOLD = 0.5
        OVERALL_THRESHOLD = 0.7

        passed = True
        reasons = []

        # Detect apakah answer honest tentang missing info
        is_honest_non_answer = any(
            phrase in answer.lower()
            for phrase in self.HONESTY_PHRASES
        )

        # Hard gate: Faithfulness
        if scores["faithfulness"] < FAITHFULNESS_THRESHOLD:
            passed = False
            reasons.append(
                f"❌ Faithfulness too low: {scores['faithfulness']:.3f} "
                f"(min: {FAITHFULNESS_THRESHOLD}) — possible hallucination"
            )

        # Soft gate: Overall
        # Kalau honest non-answer → skip overall check
        if scores["overall"] < OVERALL_THRESHOLD and not is_honest_non_answer:
            passed = False
            reasons.append(
                f"❌ Overall too low: {scores['overall']:.3f} "
                f"(min: {OVERALL_THRESHOLD})"
            )

        # Build reasons
        if is_honest_non_answer:
            reasons.append(
                "ℹ️ Honest non-answer detected — "
                "model correctly flagged missing info"
            )

        if passed:
            reasons.append("✅ All checks passed")

        return {
            "passed": passed,
            "reasons": reasons,
            "scores": scores,
            "is_honest_non_answer": is_honest_non_answer
        }
