"""Evaluation module for RAG system quality assessment."""

try:
    from .ragas_evaluator import RAGASEvaluator
    HAS_RAGAS = True
except Exception:
    HAS_RAGAS = False
    RAGASEvaluator = None  # type: ignore

from .simple_evaluator import SimpleEvaluator

__all__ = ['SimpleEvaluator']
if HAS_RAGAS:
    __all__.append('RAGASEvaluator')