"""
Simple Custom Evaluator - Week 5 Day 2
No external dependencies (no RAGAS needed).
"""

from typing import List, Dict, Any
import re


class SimpleEvaluator:
    """
    Custom evaluator for RAG system quality.
    
    Metrics:
    1. Citation Rate - Does answer have citations?
    2. Context Usage - Are chunks used in answer?
    3. Answer Completeness - Is answer substantial?
    4. Self-Reflection Rate - How often improved?
    """
    
    def __init__(self):
        self.results = []
    
    def evaluate_answer(
        self,
        query: str,
        answer: str,
        chunks: List[Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate single answer.
        
        Args:
            query: User question
            answer: Generated answer
            chunks: Retrieved chunks
            metadata: Self-reflection metadata
        
        Returns:
            Dictionary with scores
        """
        scores = {}
        
        # 1. Citation Rate (has citations?)
        citations = re.findall(r'\[(\d+)\]', answer)
        scores['has_citations'] = 1.0 if len(citations) > 0 else 0.0
        scores['citation_count'] = len(set(citations))
        
        # 2. Answer Length (substantial enough for concise RAG answers?)
        word_count = len(answer.split())
        scores['word_count'] = word_count
        if word_count >= 45:
            scores['is_substantial'] = 1.0
        elif word_count >= 25:
            scores['is_substantial'] = 0.7
        elif word_count >= 15:
            scores['is_substantial'] = 0.4
        else:
            scores['is_substantial'] = 0.0
        
        # 3. Context Usage (chunks mentioned in answer?)
        cited_chunk_numbers = {
            int(citation)
            for citation in citations
            if citation.isdigit() and 1 <= int(citation) <= len(chunks)
        }
        cited_chunks_used = len(cited_chunk_numbers)

        chunks_used = 0
        answer_terms = self._content_terms(answer)
        for chunk in chunks[:5]:  # Check top 5
            chunk_terms = self._content_terms(chunk.text)
            overlap = answer_terms & chunk_terms
            if len(overlap) >= 2:
                chunks_used += 1
        
        chunks_used = max(chunks_used, cited_chunks_used)
        denominator = min(len(chunks), 5)
        scores['chunks_used'] = chunks_used
        scores['context_usage_rate'] = chunks_used / denominator if denominator else 0.0
        
        # 4. Self-Reflection Metrics
        # Accept either the full workflow metadata or the self_reflection sub-dict.
        reflection = metadata.get('self_reflection', metadata)
        scores['iterations'] = reflection.get('iterations', 0)
        scores['final_score'] = reflection.get('final_score') or 0.0
        scores['was_improved'] = 1.0 if reflection.get('improved', False) else 0.0
        
        # 5. Overall Quality Score (weighted average)
        scores['overall'] = (
            scores['has_citations'] * 0.25 +
            scores['is_substantial'] * 0.25 +
            scores['context_usage_rate'] * 0.25 +
            scores['final_score'] * 0.25
        )
        
        return scores

    def _content_terms(self, text: str) -> set[str]:
        stopwords = {
            "about", "above", "after", "again", "against", "also", "because",
            "before", "being", "between", "could", "document", "documents",
            "does", "from", "have", "into", "more", "other", "provided",
            "should", "some", "such", "than", "that", "their", "there",
            "these", "this", "through", "uploaded", "using", "were", "what",
            "when", "where", "which", "while", "with", "would",
        }
        terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
        return {term for term in terms if term not in stopwords}
    
    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        chunks_list: List[List[Any]],
        metadata_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate multiple Q&A pairs.
        
        Returns:
            Aggregated statistics
        """
        all_scores = []
        
        for q, a, chunks, meta in zip(questions, answers, chunks_list, metadata_list):
            scores = self.evaluate_answer(q, a, chunks, meta)
            scores['query'] = q  # ✅ ADD THIS LINE
            all_scores.append(scores)
        
        # Aggregate
        aggregated = {
            'total_evaluated': len(all_scores),
            'avg_citation_rate': sum(s['has_citations'] for s in all_scores) / len(all_scores),
            'avg_word_count': sum(s['word_count'] for s in all_scores) / len(all_scores),
            'avg_context_usage': sum(s['context_usage_rate'] for s in all_scores) / len(all_scores),
            'avg_quality_score': sum(s['final_score'] for s in all_scores) / len(all_scores),
            'improvement_rate': sum(s['was_improved'] for s in all_scores) / len(all_scores),
            'avg_overall': sum(s['overall'] for s in all_scores) / len(all_scores),
            'detailed_scores': all_scores
        }
        
        return aggregated
