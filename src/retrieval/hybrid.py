import math
from typing import List, Dict, Any
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_result_lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple rankings (e.g. dense vector search and sparse BM25 search)
    using Reciprocal Rank Fusion:
    RRF_score(d) = sum(1 / (k + rank_i(d)))
    """
    rrf_scores = defaultdict(float)
    doc_lookup: Dict[str, Dict[str, Any]] = {}

    for results in ranked_result_lists:
        for rank, item in enumerate(results, start=1):
            doc_id = item["chunk_id"]
            rrf_scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in doc_lookup:
                doc_lookup[doc_id] = item

    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_doc_ids:
        merged_item = dict(doc_lookup[doc_id])
        merged_item["rrf_score"] = rrf_scores[doc_id]
        fused_results.append(merged_item)

    return fused_results
