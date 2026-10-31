from src.retrieval.hybrid import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_order():
    dense_results = [
        {"chunk_id": "c1", "content": "Text 1"},
        {"chunk_id": "c2", "content": "Text 2"},
        {"chunk_id": "c3", "content": "Text 3"},
    ]
    sparse_results = [
        {"chunk_id": "c2", "content": "Text 2"},
        {"chunk_id": "c1", "content": "Text 1"},
        {"chunk_id": "c4", "content": "Text 4"},
    ]

    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=60)

    assert len(fused) == 4
    # c1 and c2 should have higher scores because they appear in both result lists
    top_two_ids = {fused[0]["chunk_id"], fused[1]["chunk_id"]}
    assert "c1" in top_two_ids
    assert "c2" in top_two_ids
    assert fused[0]["rrf_score"] > fused[2]["rrf_score"]
