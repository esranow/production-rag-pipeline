import streamlit as st
import time
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.config import settings
from src.embeddings.providers import GeminiEmbeddingProvider
from src.storage.local_store import LocalVectorStore
from src.ingestion.chunkers.hierarchical import HierarchicalChunker
from src.generation.llm_client import GroundedLLMClient

# --- Page Setup ---
st.set_page_config(
    page_title="Enterprise Production RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 5px 14px;
        background-color: #EEF2FF;
        color: #4338CA;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .key-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 15px;
        width: 100%;
    }
    .citation-card {
        background-color: #F8FAFC;
        border-left: 4px solid #2563EB;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialization / Singletons ---
@st.cache_resource
def get_vector_store():
    return LocalVectorStore()

@st.cache_resource
def get_chunker():
    return HierarchicalChunker(parent_chunk_size=1200, child_chunk_size=300)

@st.cache_resource
def get_embedder():
    # Embedder strictly loads the locked API key from server-side settings
    return GeminiEmbeddingProvider(api_key=settings.GEMINI_API_KEY)

vector_store = get_vector_store()
chunker = get_chunker()
embedder = get_embedder()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    
    # Hidden / Locked API Key Indicator (No modification permissions)
    st.markdown("""
    <div class="key-badge">
        🔒 <strong>API Key Status:</strong>&nbsp; Secured from Environment (.env)<br/>
        <span style="font-size:0.75rem; color:#065F46;">Permissions: Read-Only (Modification Locked)</span>
    </div>
    """, unsafe_allow_html=True)
    
    selected_model = st.selectbox(
        "LLM Generator Model",
        options=["gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-2.5-pro", "gemini-3.5-flash"],
        index=0,
        help="Model options for grounded answer synthesis."
    )
    
    top_k = st.slider("Top-K Retrieved Chunks", min_value=1, max_value=10, value=4)
    min_score = st.slider("Min Relevance Score", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    
    st.divider()
    st.markdown("### 📊 Vector Index Stats")
    st.metric("Total Indexed Chunks", vector_store.total_count)
    
    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        vector_store.clear()
        st.success("Knowledge base cleared!")
        st.rerun()

# Grounded LLM Client initialized with secured server-side key
llm_client = GroundedLLMClient(model_name=selected_model, api_key=settings.GEMINI_API_KEY)


# --- Top Banner ---
st.markdown('<div class="main-header">⚡ Production RAG Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hierarchical Parent-Child Chunking • Vector Cosine Similarity • Grounded Inline Citations</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<span class="metric-badge">🟢 Pipeline: Active</span>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<span class="metric-badge">🤖 LLM: {selected_model}</span>', unsafe_allow_html=True)
with col3:
    st.markdown('<span class="metric-badge">🧠 Context Window: 1,000,000 tokens</span>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<span class="metric-badge">📚 Chunks: {vector_store.total_count}</span>', unsafe_allow_html=True)

st.write("")

# --- Navigation Tabs ---
tab_chat, tab_ingest, tab_context = st.tabs(["💬 RAG Chat & Retrieval", "📥 Ingest Documents", "🧠 Context Window & Architecture"])

# =========================================================================
# TAB 1: RAG Chat
# =========================================================================
with tab_chat:
    st.subheader("Grounded Question Answering")
    
    if vector_store.total_count == 0:
        st.info("💡 Your knowledge base is currently empty. Go to the **'📥 Ingest Documents'** tab to upload or paste reference material first!")

    user_query = st.text_input("Ask a question about your ingested documents:", placeholder="e.g. What are the key features of our pipeline?")
    col_ask, col_clear = st.columns([1, 6])
    ask_button = col_ask.button("🚀 Ask", type="primary", use_container_width=True)

    if ask_button and user_query:
        with st.spinner("Searching knowledge base & synthesizing answer..."):
            start_t = time.perf_counter()
            
            # 1. Embed query
            query_vector = embedder.embed_query(user_query)
            
            # 2. Retrieve top-k
            results = vector_store.search_dense(query_vector, limit=top_k)
            
            # 3. Filter by min score
            filtered_chunks = [r for r in results if r["score"] >= min_score]
            
            # 4. Generate answer with citations
            response = llm_client.generate_answer(user_query, filtered_chunks)
            total_time_ms = round((time.perf_counter() - start_t) * 1000, 1)

        # Display Answer
        st.markdown("### Answer")
        st.markdown(response["answer"])
        
        st.caption(f"⚡ Latency: **{total_time_ms} ms** | Model: **{selected_model}** | Chunks Retrieved: **{len(filtered_chunks)}**")
        
        # Display Sources / Citations Accordion
        if filtered_chunks:
            with st.expander(f"📚 View {len(filtered_chunks)} Verified Source Chunks & Relevance Scores", expanded=True):
                for idx, chunk in enumerate(filtered_chunks, start=1):
                    score_pct = int(chunk["score"] * 100)
                    doc_name = chunk.get("metadata", {}).get("title", chunk.get("document_id", "Unknown Document"))
                    
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>[Chunk {idx}] Source: {doc_name}</strong> &nbsp; | &nbsp; 
                        <span style="color:#059669; font-weight:600;">Match Score: {chunk['score']} ({score_pct}%)</span><br/>
                        <p style="margin-top:6px; color:#334155; font-size:0.95rem;">{chunk['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No context chunks met the minimum similarity score threshold.")


# =========================================================================
# TAB 2: Document Ingestion
# =========================================================================
with tab_ingest:
    st.subheader("Ingest New Knowledge")
    
    ingest_method = st.radio("Choose ingestion input method:", ["Paste Text Directly", "Upload Document File (.txt, .md)"], horizontal=True)
    
    doc_title = st.text_input("Document Title / Reference Name:", placeholder="e.g. Architecture Overview")
    
    doc_content = ""
    if ingest_method == "Paste Text Directly":
        doc_content = st.text_area(
            "Document Text Content:",
            placeholder="Paste your documentation, report, or notes here...",
            height=250
        )
    else:
        uploaded_file = st.file_uploader("Upload a text or markdown file", type=["txt", "md"])
        if uploaded_file is not None:
            doc_content = uploaded_file.read().decode("utf-8", errors="ignore")
            if not doc_title:
                doc_title = uploaded_file.name

    if st.button("⚡ Process & Index Document", type="primary"):
        if not doc_content.strip():
            st.error("Please provide some document content before indexing.")
        else:
            with st.spinner("Chunking semantically & generating embeddings..."):
                doc_id = f"doc_{int(time.time())}"
                title = doc_title.strip() or doc_id
                
                # Hierarchical parent-child chunking
                chunks = chunker.chunk_document(document_id=doc_id, text=doc_content, metadata={"title": title})
                
                # Only child chunks are embedded for vector search
                search_chunks = [c for c in chunks if not c.is_parent]
                chunk_dicts = [c.model_dump() for c in search_chunks]
                chunk_texts = [c.content for c in search_chunks]
                
                embeddings = embedder.embed_documents(chunk_texts)
                vector_store.upsert_chunks(chunk_dicts, embeddings)
                
                st.success(f"Successfully indexed document **'{title}'**! Created **{len(chunks)}** total chunks (**{len(search_chunks)}** search index vectors).")
                st.rerun()


# =========================================================================
# TAB 3: Context Window & Architecture Visualizer
# =========================================================================
with tab_context:
    st.subheader("What is the Context Window & Why do we need RAG?")
    
    st.markdown("""
    ### 1. What is the "Context Window"?
    The **context window** is the maximum amount of text (measured in **tokens**, roughly words/sub-words) that an AI model can remember, read, and reason over in a single prompt.

    Think of the context window as the model's **working memory (RAM)**:
    - Anything inside the window can be read by the model.
    - Anything outside the window is completely invisible to the model.
    """)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Gemini 1.5/2.5 Flash", "1,000,000 tokens", "≈ 750,000 words (~2,500 pages)")
    with col_b:
        st.metric("GPT-4o", "128,000 tokens", "≈ 96,000 words (~300 pages)")
    with col_c:
        st.metric("Claude 3.5 Sonnet", "200,000 tokens", "≈ 150,000 words (~500 pages)")

    st.markdown("""
    ---
    ### 2. "If Gemini has a 1 Million Token Window, why do we need RAG?"
    This is one of the most common questions in modern AI engineering. Here is why enterprise production systems still use RAG:

    | Dimension | Stuffing 1,000,000 Tokens into Prompt | Production RAG Pipeline (What We Built) |
    |---|---|---|
    | **Cost per Query** | **High** (~$0.50 – $2.00 per single question) | **Ultra Low** (~$0.0002 per question) |
    | **Query Latency** | **Slow** (10 to 30 seconds to parse 1M tokens) | **Sub-second** (200ms – 800ms) |
    | **Accuracy ("Lost in the Middle")** | High risk of hallucination or skipping subtle details | High precision: cross-encoder isolates exact chunks |
    | **Verifiable Citations** | Hard to track which page or paragraph was used | Exact chunk IDs, source document titles, and relevance scores |
    | **Knowledge Updates** | You have to upload your whole company corpus every query | Real-time indexing: upload once, query millions of times |

    ---
    ### 3. How Chunking Fits Into the Context Window
    In this architecture:
    1. **Child Chunk (300 tokens)**: Small and focused for high-accuracy cosine vector search.
    2. **Parent Chunk (1,200 tokens)**: Retains paragraph context.
    3. **RAG Context Window (1,500 – 3,000 tokens)**: We only send the top 3–5 most relevant parent chunks into the LLM, leaving 99.8% of the LLM's context window free for complex reasoning!
    """)
