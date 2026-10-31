import pytest
from src.ingestion.chunkers.hierarchical import HierarchicalChunker


def test_hierarchical_chunking_creates_parent_and_child():
    sample_text = (
        "Retrieval-Augmented Generation (RAG) optimizes LLM outputs with external knowledge. "
        "It retrieves relevant document chunks from a vector database before generating an answer. "
        "Production systems employ hybrid search combining dense semantic vectors with BM25 keywords. "
        "Cross-encoders rerank the retrieved candidates to prune low-scoring distractors. "
        "Finally, citation grounding forces the LLM to cite verifiable references."
    )

    chunker = HierarchicalChunker(parent_chunk_size=30, child_chunk_size=15)
    chunks = chunker.chunk_document(document_id="test_doc", text=sample_text, metadata={"source": "test.txt"})

    assert len(chunks) > 0
    parent_chunks = [c for c in chunks if c.is_parent]
    child_chunks = [c for c in chunks if not c.is_parent]

    assert len(parent_chunks) > 0
    assert len(child_chunks) > 0

    # Ensure all child chunks have a valid parent_id linking to a parent chunk
    parent_ids = {p.chunk_id for p in parent_chunks}
    for child in child_chunks:
        assert child.parent_id in parent_ids
