"""
Retrieval Coordinator Agent - Tactical Level 2 Agent.

Manages parallel retrieval swarm (Vector, Keyword, Graph agents).
Coordinates retrieval, aggregates results, and deduplicates chunks.

Swarm Pattern:
- Spawns 3 retrieval agents in parallel
- Each agent uses different retrieval method
- Aggregates all results
- Deduplicates by content similarity
- Returns top-k unique chunks
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from collections import defaultdict
import hashlib

from src.agents.base_agent import BaseAgent
from src.models.agent_state import AgentState, Chunk
from src.utils.exceptions import AgentExecutionError, RetrievalError
from src.config import get_settings


class RetrievalCoordinator(BaseAgent):
    """
    Retrieval Coordinator - Manages parallel retrieval swarm.
    
    Spawns multiple retrieval agents to search for relevant chunks
    using different methods (vector, keyword, graph). Aggregates
    and deduplicates results.
    
    Attributes:
        vector_agent: Vector search agent (semantic)
        keyword_agent: Keyword search agent (BM25)
        graph_agent: Graph search agent (relationships)
        top_k: Number of chunks to return
        parallel: Whether to execute agents in parallel
        
    Example:
        >>> coordinator = RetrievalCoordinator(
        ...     vector_agent=vector_agent,
        ...     keyword_agent=keyword_agent,
        ...     graph_agent=graph_agent
        ... )
        >>> 
        >>> state = AgentState(query="What is Python?")
        >>> result = coordinator.run(state)
        >>> 
        >>> print(len(result.chunks))  # 10 (top_k)
        >>> print(result.retrieval_round)  # 0 or incremented
    """
    
    def __init__(
        self,
        vector_agent: BaseAgent = None,
        keyword_agent: BaseAgent = None,
        graph_agent: BaseAgent = None,
        top_k: int = None,
        parallel: bool = None
    ):
        """
        Initialize Retrieval Coordinator.
        
        Args:
            vector_agent: Vector search agent instance
            keyword_agent: Keyword search agent instance
            graph_agent: Graph search agent instance
            top_k: Number of top chunks to return (default: from config)
            parallel: Execute agents in parallel (default: from config)
        
        Example:
            >>> coordinator = RetrievalCoordinator(
            ...     vector_agent=VectorAgent(),
            ...     keyword_agent=KeywordAgent(),
            ...     graph_agent=GraphAgent(),
            ...     top_k=15
            ... )
        """
        super().__init__(name="retrieval_coordinator", version="1.0.0")
        
        self.vector_agent = vector_agent
        self.keyword_agent = keyword_agent
        self.graph_agent = graph_agent
        
        # Load settings from config
        settings = get_settings()
        self.top_k = top_k if top_k is not None else settings.retrieval_top_k
        self.parallel = (
            parallel if parallel is not None else settings.parallel_retrieval
        )
        
        self.log(
            f"Initialized with top_k={self.top_k}, parallel={self.parallel}",
            level="debug"
        )
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute retrieval coordination: spawn swarm and aggregate results.
        
        Args:
            state: Current agent state with query
        
        Returns:
            Updated state with chunks
        
        Raises:
            RetrievalError: If retrieval fails
        
        Example:
            >>> state = AgentState(query="What is machine learning?")
            >>> result = coordinator.execute(state)
            >>> print(len(result.chunks))  # Top-k chunks
        """
        try:
            query = state.query
            retrieval_queries = state.sub_queries or [query]
            current_round = state.retrieval_round
            
            self.log(
                f"Starting retrieval round {current_round} for query: {query[:50]}...",
                level="info"
            )
            
            # Step 1: Spawn retrieval swarm
            all_results = []
            query_metadata = []
            for retrieval_query in retrieval_queries:
                query_results, swarm_metadata = self._spawn_swarm_with_metadata(retrieval_query)
                all_results.extend(query_results)
                query_metadata.append({
                    "query": retrieval_query,
                    **swarm_metadata,
                })
            
            self.log(
                f"Retrieved {len(all_results)} total chunks from swarm",
                level="info"
            )
            
            # Step 2: Deduplicate
            unique_chunks = self._deduplicate(all_results)
            
            self.log(
                f"Deduplication: {len(all_results)} → {len(unique_chunks)} unique chunks",
                level="info"
            )
            
            # Step 3: Select top-k
            top_chunks = self._select_top_k(unique_chunks, self.top_k)
            
            self.log(
                f"Selected top {len(top_chunks)} chunks",
                level="info"
            )
            
            # Step 4: Update state
            state.chunks = top_chunks
            state.retrieval_round = current_round + 1
            
            # Step 5: Add metadata
            state.metadata["retrieval_coordinator"] = {
                "round": current_round,
                "total_retrieved": len(all_results),
                "unique_chunks": len(unique_chunks),
                "final_chunks": len(top_chunks),
                "parallel": self.parallel,
                "query_count": len(retrieval_queries),
                "query_breakdown": query_metadata,
            }
            
            return state
            
        except Exception as e:
            self.log(f"Retrieval coordination failed: {str(e)}", level="error")
            raise RetrievalError(
                retrieval_type="coordination",
                message=f"Failed to coordinate retrieval: {str(e)}",
                details={"query": state.query, "round": state.retrieval_round}
            ) from e
    
    def _spawn_swarm(self, query: str) -> List[Chunk]:
        """Spawn retrieval swarm (private method)."""
        chunks, _ = self._spawn_swarm_with_metadata(query)
        return chunks

    def _spawn_swarm_with_metadata(self, query: str) -> tuple[List[Chunk], Dict[str, Any]]:
        """Spawn retrieval swarm and return chunks plus execution metadata."""
        
        self.log(f"Spawning retrieval swarm for: {query}")
        
        # Collect available agents
        agents = []
        
        if self.vector_agent:
            agents.append(('vector', self.vector_agent))
        
        if self.keyword_agent:
            agents.append(('keyword', self.keyword_agent))
        
        if self.graph_agent:  # ← Just check if exists!
            agents.append(('graph', self.graph_agent))
            self.log("Graph search agent included in swarm")
        else:
            self.log("Graph search unavailable", level="warning")
        
        metadata = {
            "agents_requested": [name for name, _ in agents],
            "agent_counts": {},
            "agent_errors": {},
            "execution_mode": "parallel" if self.parallel else "sequential",
        }

        if self.parallel and len(agents) > 1:
            all_results = self._execute_swarm_parallel(agents, query, metadata)
        else:
            all_results = self._execute_swarm_sequential(agents, query, metadata)

        self.log(f"Swarm complete: {len(all_results)} chunks from {len(agents)} agents")

        return all_results, metadata

    def _execute_swarm_sequential(
        self,
        agents: List[tuple[str, BaseAgent]],
        query: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Execute retrieval agents one by one."""
        all_results = []

        for agent_name, agent in agents:
            self.log(f"Executing {agent_name} agent...")
            try:
                results = self._search_agent(agent, query)
                metadata["agent_counts"][agent_name] = len(results)
                self.log(f"{agent_name}: {len(results)} chunks")
                all_results.extend(results)
            except Exception as e:
                metadata["agent_errors"][agent_name] = str(e)
                metadata["agent_counts"][agent_name] = 0
                self.log(f"{agent_name} failed: {e}", level="error")

        return all_results

    def _execute_swarm_parallel(
        self,
        agents: List[tuple[str, BaseAgent]],
        query: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Execute retrieval agents concurrently using worker threads."""
        all_results = []
        max_workers = min(len(agents), 3)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._search_agent, agent, query): agent_name
                for agent_name, agent in agents
            }

            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    results = future.result()
                    metadata["agent_counts"][agent_name] = len(results)
                    self.log(f"{agent_name}: {len(results)} chunks")
                    all_results.extend(results)
                except Exception as e:
                    metadata["agent_errors"][agent_name] = str(e)
                    metadata["agent_counts"][agent_name] = 0
                    self.log(f"{agent_name} failed: {e}", level="error")

        return all_results

    def _search_agent(self, agent: BaseAgent, query: str) -> List[Chunk]:
        """Run a retrieval agent through its preferred search interface."""
        if hasattr(agent, "search_async"):
            return agent.search_async(query, top_k=self.top_k)

        temp_state = AgentState(query=query)
        result_state = agent.run(temp_state)
        return result_state.chunks
    
    
    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Remove duplicate chunks based on content similarity.
        
        Uses text hashing to identify duplicates. Keeps chunk with
        highest score when duplicates found.
        
        Args:
            chunks: List of chunks (may contain duplicates)
        
        Returns:
            List of unique chunks
        
        Example:
            >>> duplicates = [chunk1, chunk1_copy, chunk2]
            >>> unique = coordinator._deduplicate(duplicates)
            >>> print(len(unique))  # 2
        """
        if not chunks:
            return []
        
        # Group by content hash
        hash_groups = defaultdict(list)
        
        for chunk in chunks:
            content_hash = self._hash_content(chunk.text)
            hash_groups[content_hash].append(chunk)
        
        # Keep best chunk from each group
        unique_chunks = []
        for group in hash_groups.values():
            # Sort by score (descending)
            sorted_group = sorted(
                group,
                key=lambda c: c.score if c.score is not None else 0.0,
                reverse=True
            )
            # Keep highest scored
            unique_chunks.append(sorted_group[0])
        
        return unique_chunks
    
    def _hash_content(self, text: str) -> str:
        """
        Generate hash for content similarity.
        
        Uses MD5 hash of normalized text.
        
        Args:
            text: Chunk text
        
        Returns:
            Content hash string
        """
        # Normalize text (lowercase, strip whitespace)
        normalized = text.lower().strip()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Generate hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _select_top_k(self, chunks: List[Chunk], k: int) -> List[Chunk]:
        """
        Select top-k chunks by score.
        
        Args:
            chunks: List of chunks
            k: Number to select
        
        Returns:
            Top-k chunks sorted by score (descending)
        
        Example:
            >>> top_10 = coordinator._select_top_k(chunks, 10)
            >>> print(len(top_10))  # 10
            >>> print(top_10[0].score >= top_10[-1].score)  # True
        """
        if not chunks:
            return []
        
        # Sort by calibrated score. BM25 scores are unbounded while vector
        # scores are normalized, so raw sorting lets keyword retrieval dominate.
        sorted_chunks = sorted(
            chunks,
            key=self._calibrated_score,
            reverse=True
        )
        
        # Return top-k
        return sorted_chunks[:k]

    def _calibrated_score(self, chunk: Chunk) -> float:
        score = chunk.score if chunk.score is not None else 0.0
        source = chunk.metadata.get("source", "unknown")

        if source == "document_overview":
            return 1.0
        if source == "keyword":
            return min(score / 15.0, 0.65)
        if source == "graph":
            return min(score, 0.8)
        return score
    
    def retrieve_with_details(self, query: str) -> Dict[str, Any]:
        """
        Detailed retrieval for debugging/analysis.
        
        Returns breakdown of retrieval from each agent and
        deduplication statistics.
        
        Args:
            query: User query string
        
        Returns:
            Dictionary with detailed retrieval info
        
        Example:
            >>> details = coordinator.retrieve_with_details("What is X?")
            >>> print(details["vector_count"])
            >>> print(details["dedup_stats"])
        """
        # Execute retrieval
        all_chunks = self._spawn_swarm(query)
        
        # Count by source (if agents tag chunks)
        source_counts = defaultdict(int)
        for chunk in all_chunks:
            source = chunk.metadata.get("source", "unknown")
            source_counts[source] += 1
        
        # Deduplicate
        unique_chunks = self._deduplicate(all_chunks)
        top_chunks = self._select_top_k(unique_chunks, self.top_k)
        
        return {
            "query": query,
            "total_retrieved": len(all_chunks),
            "source_counts": dict(source_counts),
            "unique_chunks": len(unique_chunks),
            "duplicates_removed": len(all_chunks) - len(unique_chunks),
            "final_chunks": len(top_chunks),
            "top_k": self.top_k,
            "parallel": self.parallel,
            "chunks": top_chunks
        }
