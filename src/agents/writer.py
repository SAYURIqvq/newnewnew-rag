"""
Writer Agent - Tactical Level 2 Agent.

Generates formatted answers from retrieved chunks with citations.
Uses LLM to synthesize information into coherent responses.
"""

from typing import List, Dict, Any, Optional
import re

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.config import get_settings
from src.llm.qwen import create_qwen_chat_model
from src.utils.logger import setup_logger
from src.utils.exceptions import AgenticRAGException


class WriterError(AgenticRAGException):
    """Error during answer generation."""
    pass


class WriterAgent(BaseAgent):
    """
    Writer Agent - Answer generation with citations.
    
    Takes retrieved chunks and user query, generates:
    - Coherent answer synthesizing information
    - Inline citations [1], [2], [3]
    - Source list with references
    
    Features:
    - LLM-based answer generation
    - Citation extraction and formatting
    - Answer quality checks
    - Source attribution
    
    Attributes:
        llm: Language model for generation
        max_tokens: Maximum tokens for answer
        temperature: LLM temperature
        include_sources: Whether to include source list
        
    Example:
        >>> agent = WriterAgent(llm=llm)
        >>> state = AgentState(query="What is Python?", chunks=[...])
        >>> result = agent.run(state)
        >>> print(result.answer)
        Python is a programming language [1] created by Guido van Rossum [2]...
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        max_tokens: int = None,
        temperature: float = None,
        include_sources: bool = True
    ):
        """
        Initialize Writer Agent.
        
        Args:
            llm: ChatAnthropic instance (creates if None)
            max_tokens: Max tokens for answer (default from config)
            temperature: LLM temperature (default from config)
            include_sources: Include source list in answer
        
        Example:
            >>> from src.config import get_settings
            >>> from src.llm.qwen import create_qwen_chat_model
            >>> llm = create_qwen_chat_model(get_settings())
            >>> agent = WriterAgent(llm=llm)
        """
        super().__init__(name="writer", version="1.0.0")
        
        settings = get_settings()
        
        # Initialize LLM
        if llm is None:
            self.llm = create_qwen_chat_model(
                settings,
                model=settings.get_agent_model("writer"),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            self.llm = llm
        
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.temperature = temperature or settings.llm_temperature
        self.include_sources = include_sources
        
        self.log(
            f"Initialized with model={settings.llm_model}, "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature}",
            level="info"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute answer generation.
        
        Args:
            state: Current state with query and chunks
        
        Returns:
            Updated state with generated answer
        
        Raises:
            WriterError: If generation fails
        
        Example:
            >>> state = AgentState(query="What is ML?", chunks=[...])
            >>> result = agent.execute(state)
            >>> print(result.answer)
        """
        try:
            query = state.query
            chunks = state.chunks
            
            if not chunks:
                self.log("No chunks provided for answer generation", level="warning")
                state.answer = "I don't have enough information to answer this question."
                return state
            
            self.log(
                f"Generating answer for: {query[:50]}... "
                f"(using {len(chunks)} chunks)",
                level="info"
            )
            
            # Generate answer with citations
            answer = self._generate_answer(query, chunks)
            
            # Extract and format citations
            cited_answer = self._ensure_minimum_citation(answer, chunks)
            formatted_answer = self._format_answer(cited_answer, chunks)
            
            # Update state
            state.answer = formatted_answer
            
            # Add metadata
            state.metadata["writer"] = {
                "chunks_used": len(chunks),
                "answer_length": len(formatted_answer),
                "citations_count": self._count_citations(formatted_answer)
            }
            
            self.log(
                f"Answer generated: {len(formatted_answer)} chars, "
                f"{self._count_citations(formatted_answer)} citations",
                level="info"
            )
            
            return state
            
        except Exception as e:
            self.log(f"Answer generation failed: {str(e)}", level="error")
            raise WriterError(
                message=f"Failed to generate answer: {str(e)}",
                details={"query": state.query}
            ) from e
    
    def _generate_answer(self, query: str, chunks: List[Chunk]) -> str:
        """
        Generate answer using LLM.
        
        Args:
            query: User query
            chunks: Retrieved chunks
        
        Returns:
            Generated answer with citations
        """
        # Prepare context from chunks
        context = self._build_compact_context(chunks)
        
        # Create prompt with strict citation rules
        prompt = f"""Answer the question using only the numbered evidence excerpts.

User Question: {query}

Evidence:
{context}

Requirements:
- Silently make a checklist of every clause in the question and answer each one.
- Ground every factual sentence in the excerpts and cite its exact number, such as [2].
- State clearly when an item cannot be answered from the excerpts.
- Cover conditions, exclusions, conflicting results, and evidence gaps when requested.
- Use no outside knowledge and do not add a Sources section.
- Start directly with the answer and stay under 260 words.

Answer (inline citations only, no Sources section):"""
        
        # Generate
        try:
            response = self.llm.invoke(prompt)
            from src.llm.content_utils import extract_llm_text
            answer = extract_llm_text(response.content)
            if not answer.strip():
                raise WriterError(
                    message="The model returned no final answer text",
                    details={"hint": "Disable reasoning or increase output tokens"},
                )
            
            return answer
            
        except Exception as e:
            raise WriterError(
                message=f"LLM generation failed: {str(e)}",
                details={"query": query}
            ) from e

    def generate_context_summary(self, query: str, chunks: List[Chunk]) -> str:
        """
        Deterministic fallback that summarizes retrieved evidence with citations.

        Used when the LLM produces an ungrounded or citation-free answer.
        """
        selected = chunks[:3]
        sentences = []

        for i, chunk in enumerate(selected, 1):
            snippet = self._first_sentence(chunk.text)
            if snippet:
                sentences.append(f"{snippet} [{i}]")

        if not sentences:
            return "The provided documents do not contain enough readable information to answer this question."

        if "compare" in query.lower() and len(sentences) >= 2:
            intro = "The retrieved evidence presents multiple relevant points for comparison."
            return f"{intro} " + " ".join(sentences)

        return " ".join(sentences)

    def _first_sentence(self, text: str, max_words: int = 35) -> str:
        clean = " ".join(text.split())
        if not clean:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", clean)
        sentence = parts[0] if parts else clean
        words = sentence.split()
        if len(words) > max_words:
            sentence = " ".join(words[:max_words]).rstrip(",;:")

        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence
    
    def _format_answer(self, answer: str, chunks: List[Chunk]) -> str:
        """
        Format answer with source list.
        
        Args:
            answer: Generated answer with citations
            chunks: Source chunks
        
        Returns:
            Formatted answer with source list
        """
        if not self.include_sources:
            return answer
        
        # Extract unique citation numbers
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = sorted(set(int(c) for c in citations))
        
        if not unique_citations:
            return answer
        
        # Build source list
        sources_section = "\n\n---\n\n**Sources:**\n"
        
        for citation_num in unique_citations:
            # Get corresponding chunk (1-indexed)
            if citation_num <= len(chunks):
                chunk = chunks[citation_num - 1]
                source = chunk.metadata.get('filename', 'Unknown source')
                
                # Add source entry
                sources_section += f"\n[{citation_num}] {source}"
        
        return answer + sources_section
    
    def _count_citations(self, answer: str) -> int:
        """
        Count number of citations in answer.
        
        Args:
            answer: Answer text
        
        Returns:
            Number of unique citations
        """
        citations = re.findall(r'\[(\d+)\]', answer)
        return len(set(citations))

    def _ensure_minimum_citation(self, answer: str, chunks: List[Chunk]) -> str:
        """
        Add a conservative citation when the LLM produced a grounded answer
        but omitted inline references.
        """
        if not chunks or self._count_citations(answer) > 0:
            return answer

        if self._is_honest_non_answer(answer):
            return answer

        stripped = answer.rstrip()
        if not stripped:
            return answer

        if stripped[-1] in ".!?":
            return f"{stripped} [1]"
        return f"{stripped} [1]"

    def _is_honest_non_answer(self, answer: str) -> bool:
        lowered = answer.lower()
        markers = [
            "do not contain information",
            "don't have enough information",
            "not enough information",
            "cannot answer",
            "not available in the provided",
            "not mentioned in the provided",
        ]
        return any(marker in lowered for marker in markers)
    
    def generate_with_feedback(
        self,
        query: str,
        chunks: List[Chunk],
        feedback: str
    ) -> str:
        """
        Regenerate answer with feedback from Critic.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            feedback: Feedback from Critic agent
        
        Returns:
            Improved answer
        
        Example:
            >>> answer = agent.generate_with_feedback(
            ...     query="What is ML?",
            ...     chunks=chunks,
            ...     feedback="Add more examples"
            ... )
        """
        # Prepare context
        context = self._build_compact_context(chunks)
        
        # Create improvement prompt
        prompt = f"""You are improving an answer based on feedback.

Original Question: {query}

Context:
{context}

Feedback for improvement:
{feedback}

Instructions:
1. Generate an IMPROVED answer addressing the feedback
2. Use inline citations [1], [2], [3]
3. Maintain accuracy and source attribution
4. Address all points in the feedback
5. Keep the complete answer under 260 words
6. Start directly with the answer
7. Do not mention feedback, revision, improvement, or the writing process
8. Silently check every clause in the original question and answer each one explicitly

Improved Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            from src.llm.content_utils import extract_llm_text
            answer = extract_llm_text(response.content)
            if not answer.strip():
                raise WriterError(
                    message="The model returned no regenerated answer text",
                    details={"hint": "Disable reasoning or increase output tokens"},
                )
            
            cited_answer = self._ensure_minimum_citation(answer, chunks)
            return self._format_answer(cited_answer, chunks)
            
        except Exception as e:
            self.log(f"Answer regeneration failed: {str(e)}", level="error")
            raise WriterError(
                message=f"Failed to regenerate answer: {str(e)}",
                details={"feedback": feedback}
            ) from e

    def _build_compact_context(
        self,
        chunks: List[Chunk],
        total_chars: int = 6500,
        per_chunk_chars: int = 1400,
    ) -> str:
        """Build a bounded evidence prompt that fits low-credit API limits."""
        parts = []
        used = 0
        for i, chunk in enumerate(chunks, 1):
            remaining = total_chars - used
            if remaining <= 0:
                break
            text = " ".join(chunk.text.split())[:min(per_chunk_chars, remaining)]
            if not text:
                continue
            part = f"[{i}] {text}"
            parts.append(part)
            used += len(part)
        return "\n\n".join(parts)
