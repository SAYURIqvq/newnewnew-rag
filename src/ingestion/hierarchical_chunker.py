"""
Hierarchical Chunking System
Parent chunks: 2000 tokens (full context)
Child chunks: 500 tokens (searchable units)

This provides flexibility:
- Search in child chunks (fast, precise)
- Retrieve parent chunks (full context)
"""

from src.models.chunk import Chunk
from typing import List, Dict, Any, Tuple
import re
import tiktoken

class HierarchicalChunker:
    """
    Create hierarchical chunks with parent-child relationships.
    
    Strategy:
    1. Create large parent chunks (2000 tokens)
    2. Split each parent into smaller child chunks (500 tokens)
    3. Maintain relationships between parents and children
    """
    
    def __init__(
        self,
        parent_size: int = 2000,
        child_size: int = 500,
        child_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize hierarchical chunker.
        
        Args:
            parent_size: Parent chunk size in tokens
            child_size: Child chunk size in tokens
            child_overlap: Overlap between child chunks
            encoding_name: Tokenizer encoding
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def chunk_text(
        self, 
        text: str,
        doc_id: str = "unknown",  # ← ADD
        metadata: Dict[str, Any] = None  # ← ADD
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Create hierarchical chunks from text.
        
        Args:
            text: Input text to chunk
            
        Returns:
            Tuple of (parent_chunks, child_chunks)
        """
        print(f"\n📝 Creating hierarchical chunks...")
        print(f"   Parent size: {self.parent_size} tokens")
        print(f"   Child size: {self.child_size} tokens")
        print(f"   Child overlap: {self.child_overlap} tokens")
        
        sections = self._split_structured_sections(text)
        total_tokens = sum(len(self.encoding.encode(section["text"])) for section in sections)
        
        print(f"   Total tokens: {total_tokens:,}")
        
        parent_chunks = []
        child_chunks = []
        
        # Create parent chunks
        parent_num = 0
        document_offset = 0
        for section in sections:
            section_tokens = self.encoding.encode(section["text"])
            section_metadata = {
                **(metadata or {}),
                "section_path": section["section_path"],
                "page": section["page"],
                "content_type": section["content_type"],
            }
            start_idx = 0
            while start_idx < len(section_tokens):
                end_idx = min(start_idx + self.parent_size, len(section_tokens))
                parent_tokens = section_tokens[start_idx:end_idx]
                parent_text = self.encoding.decode(parent_tokens)
                parent_id = f"parent_{parent_num}"
                absolute_start = document_offset + start_idx
                absolute_end = document_offset + end_idx
                parent = Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    doc_id=doc_id,
                    tokens=parent_tokens,
                    token_count=len(parent_tokens),
                    start_idx=absolute_start,
                    end_idx=absolute_end,
                    chunk_type='parent',
                    children_ids=[],
                    metadata=section_metadata,
                )
                children = self._create_children(
                    parent_tokens=parent_tokens,
                    parent_id=parent_id,
                    parent_start_idx=absolute_start,
                    doc_id=doc_id,
                    metadata=section_metadata,
                )
                parent.children_ids = [child.chunk_id for child in children]
                parent_chunks.append(parent)
                child_chunks.extend(children)
                parent_num += 1
                start_idx = end_idx
            document_offset += len(section_tokens)
        
        print(f"✅ Created {len(parent_chunks)} parent chunks")
        print(f"✅ Created {len(child_chunks)} child chunks")
        print(f"   Average children per parent: {len(child_chunks)/len(parent_chunks):.1f}")
        
        return parent_chunks, child_chunks

    def _split_structured_sections(self, text: str) -> List[Dict[str, Any]]:
        """Split loader markers into section-aware source units.

        Markers are emitted by ``DocumentLoader`` for PDF pages, DOCX heading
        styles, and DOCX tables. Plain text without markers remains one section.
        """
        marker = re.compile(r"^\[\[(PAGE|SECTION|TABLE):(.*?)\]\]$", re.MULTILINE)
        sections: List[Dict[str, Any]] = []
        current = {"section_path": "Document", "page": 1, "content_type": "text"}
        cursor = 0

        def append_content(content: str) -> None:
            cleaned = content.strip()
            if cleaned:
                sections.append({**current, "text": cleaned})

        for match in marker.finditer(text):
            append_content(text[cursor:match.start()])
            kind, value = match.group(1), match.group(2).strip()
            if kind == "PAGE":
                current["page"] = int(value) if value.isdigit() else current["page"]
                current["content_type"] = "text"
            elif kind == "SECTION":
                current["section_path"] = value or "Document"
                current["content_type"] = "text"
            else:
                current["section_path"] = value or "Table"
                current["content_type"] = "table"
            cursor = match.end()
        append_content(text[cursor:])
        return sections or [{"text": text.strip(), "section_path": "Document", "page": 1, "content_type": "text"}]
    
    def _create_children(
        self,
        parent_tokens: List[int],
        parent_id: str,
        parent_start_idx: int,
        doc_id: str = "unknown",          # ← ADD
        metadata: Dict[str, Any] = None   # ← ADD
    ) -> List[Chunk]:
        """
        Create child chunks from a parent chunk.
        
        Args:
            parent_tokens: Parent's token list
            parent_id: Parent chunk ID
            parent_start_idx: Parent's start index in full text
            
        Returns:
            List of child chunks
        """
        children = []
        child_num = 0
        start_idx = 0
        
        while start_idx < len(parent_tokens):
            # Child chunk boundaries
            end_idx = min(start_idx + self.child_size, len(parent_tokens))
            child_tokens = parent_tokens[start_idx:end_idx]
            child_text = self.encoding.decode(child_tokens)
            
            child_id = f"{parent_id}_child_{child_num}"
            
            # Create child chunk
            child = Chunk(
                chunk_id=child_id,
                text=child_text,
                doc_id=doc_id,                      # ← USE doc_id
                tokens=child_tokens,
                token_count=len(child_tokens),
                start_idx=parent_start_idx + start_idx,
                end_idx=parent_start_idx + end_idx,
                chunk_type='child',
                parent_id=parent_id,
                metadata=metadata or {}              # ← USE metadata
            )
            
            children.append(child)
            child_num += 1
            
            # Move to next child with overlap
            start_idx += self.child_size - self.child_overlap
        
        return children
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def get_parent_context(
        self,
        child_chunk: Chunk,
        parent_chunks: List[Chunk]
    ) -> Chunk:
        """
        Get parent chunk for a child chunk.
        
        Args:
            child_chunk: Child chunk
            parent_chunks: List of all parent chunks
            
        Returns:
            Parent chunk or None
        """
        if child_chunk.chunk_type != 'child':
            return child_chunk
        
        for parent in parent_chunks:
            if parent.chunk_id == child_chunk.parent_id:
                return parent
        
        return None
