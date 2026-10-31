import hashlib
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    parent_id: Optional[str] = None
    content: str
    is_parent: bool = False
    token_count_approx: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HierarchicalChunker:
    """
    Splits documents into large 'Parent' context blocks and smaller 'Child' search chunks.
    Child chunks are embedded for maximum vector search precision, while Parent chunks
    provide the full surrounding context to the LLM during generation.
    """

    def __init__(self, parent_chunk_size: int = 1200, child_chunk_size: int = 300, overlap: int = 50):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.overlap = overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter matching punctuation followed by whitespace
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def chunk_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> List[DocumentChunk]:
        metadata = metadata or {}
        sentences = self._split_into_sentences(text)
        all_chunks: List[DocumentChunk] = []

        # 1. Create Parent chunks
        parent_blocks: List[str] = []
        current_parent: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > self.parent_chunk_size and current_parent:
                parent_blocks.append(" ".join(current_parent))
                current_parent = [sent]
                current_len = sent_len
            else:
                current_parent.append(sent)
                current_len += sent_len

        if current_parent:
            parent_blocks.append(" ".join(current_parent))

        # 2. For each parent block, create child chunks and link them
        for p_idx, p_text in enumerate(parent_blocks):
            p_hash = hashlib.sha256(f"{document_id}_parent_{p_idx}".encode()).hexdigest()[:16]
            parent_chunk_id = f"{document_id}_p{p_idx}_{p_hash}"

            parent_chunk = DocumentChunk(
                chunk_id=parent_chunk_id,
                document_id=document_id,
                parent_id=None,
                content=p_text,
                is_parent=True,
                token_count_approx=len(p_text.split()),
                metadata={**metadata, "parent_index": p_idx}
            )
            all_chunks.append(parent_chunk)

            # Child chunking within parent
            p_sentences = self._split_into_sentences(p_text)
            current_child: List[str] = []
            c_len = 0
            c_idx = 0

            for sent in p_sentences:
                sent_len = len(sent.split())
                if c_len + sent_len > self.child_chunk_size and current_child:
                    child_text = " ".join(current_child)
                    c_hash = hashlib.sha256(f"{parent_chunk_id}_child_{c_idx}".encode()).hexdigest()[:16]
                    all_chunks.append(DocumentChunk(
                        chunk_id=f"{parent_chunk_id}_c{c_idx}_{c_hash}",
                        document_id=document_id,
                        parent_id=parent_chunk_id,
                        content=child_text,
                        is_parent=False,
                        token_count_approx=len(child_text.split()),
                        metadata={**metadata, "parent_index": p_idx, "child_index": c_idx}
                    ))
                    c_idx += 1
                    current_child = [sent]
                    c_len = sent_len
                else:
                    current_child.append(sent)
                    c_len += sent_len

            if current_child:
                child_text = " ".join(current_child)
                c_hash = hashlib.sha256(f"{parent_chunk_id}_child_{c_idx}".encode()).hexdigest()[:16]
                all_chunks.append(DocumentChunk(
                    chunk_id=f"{parent_chunk_id}_c{c_idx}_{c_hash}",
                    document_id=document_id,
                    parent_id=parent_chunk_id,
                    content=child_text,
                    is_parent=False,
                    token_count_approx=len(child_text.split()),
                    metadata={**metadata, "parent_index": p_idx, "child_index": c_idx}
                ))

        return all_chunks
