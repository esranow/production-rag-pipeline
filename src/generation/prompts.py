from typing import List, Dict, Any

RAG_SYSTEM_PROMPT = """You are an expert AI assistant that answers questions accurately using ONLY the provided verified context chunks.

GUIDELINES:
1. Strict Grounding: Rely strictly on the information in the Context. Do not introduce outside assumptions, extrapolations, or unsubstantiated claims.
2. Inline Citations: Whenever you state a fact derived from a context chunk, cite it immediately using bracketed numbers like [1], [2], corresponding to the Chunk number.
3. Insufficient Context Refusal: If the provided context does not contain enough evidence to answer the question, explicitly state:
   "I do not have sufficient information in the provided context to answer this question."
4. Structure: Present your answer clearly using concise paragraphs or bullet points where appropriate.
"""


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Builds a formatted context block with enumerated citations."""
    context_str = ""
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        doc_info = chunk.get("metadata", {}).get("source", chunk.get("document_id", "Unknown Document"))
        page_info = chunk.get("metadata", {}).get("page", "")
        source_label = f"{doc_info} (page {page_info})" if page_info else doc_info

        context_str += f"\n--- [Chunk {idx}] (Source: {source_label}) ---\n"
        context_str += chunk.get("content", "").strip() + "\n"

    return f"""Context:
{context_str}

User Question:
{query}

Answer with inline citations (e.g., [1], [2]):"""
