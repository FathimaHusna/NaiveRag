import streamlit as st
import os
import re
import pandas as pd
from naive_rag import NaiveRAG, load_text_files

# Page Configuration
st.set_page_config(
    page_title="Naive RAG Failure Modes",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E1E1E;
    }
    .failure-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ff4b4b;
        background-color: #ffecec;
        margin-bottom: 20px;
    }
    .success-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        background-color: #e8f5e9;
        margin-bottom: 20px;
    }
    .chunk-box {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: white;
    }
    .highlight {
        background-color: #ffffcc;
        padding: 2px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def dynamic_analysis(query, answer, retrieved_chunks):
    insights = []
    
    # 1. Retrieval Score Analysis
    if not retrieved_chunks:
        return ["🔴 **Critical Failure:** No documents were retrieved."]
    
    top_score = retrieved_chunks[0][1]
    if top_score < 0.4:
        insights.append(f"⚠️ **Weak Retrieval Signal:** Top match score is low ({top_score:.2f}). The system is struggling to find relevant content.")
    elif top_score > 0.7:
        insights.append(f"✅ **Strong Retrieval Signal:** Top match score is high ({top_score:.2f}).")

    # 2. Key Term Coverage
    # Simple stopword removal
    stopwords = {"is", "the", "in", "at", "which", "who", "what", "where", "when", "does", "that", "and", "or", "to", "of", "a", "an"}
    query_terms = set(re.findall(r"\w+", query.lower())) - stopwords
    
    if query_terms:
        top_chunk_text = retrieved_chunks[0][0].text.lower()
        found_terms = {t for t in query_terms if t in top_chunk_text}
        missing_terms = query_terms - found_terms
        coverage_pct = len(found_terms) / len(query_terms)
        
        if coverage_pct < 0.5:
            insights.append(f"📉 **Low Keyword Coverage:** Only {int(coverage_pct*100)}% of query keywords found in top chunk. Missing: *{', '.join(missing_terms)}*.")
            if "replaced" in query_terms or "group" in query_terms:
                 insights.append("🧩 **Potential Reasoning Gap:** Key relationships might be split across chunks (Multi-Hop failure).")
        else:
            insights.append(f"🎯 **Good Keyword Coverage:** {int(coverage_pct*100)}% of query keywords present in context.")

    # 3. Answer Provenance (Hallucination Check)
    if answer:
        in_context = any(answer.lower() in c[0].text.lower() for c in retrieved_chunks)
        if not in_context:
            insights.append("👻 **Potential Hallucination:** The answer string was NOT found exactly in the retrieved chunks.")
        else:
             insights.append("🔗 **Grounded Answer:** Answer found verbatim in the text.")
             
    if not insights:
        insights.append("✨ Analysis complete. No specific anomalies detected.")

    return insights

# Title
st.title("📉 The Physics of Failure: Naive RAG Demo")
st.markdown("Interact with the ICC T20 World Cup 2026 dataset to observe common RAG failure modes.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Configuration")
    chunk_size = st.slider("Chunk Size (words)", min_value=50, max_value=300, value=100)
    overlap = st.slider("Overlap (words)", min_value=0, max_value=50, value=20)
    top_k = st.slider("Top-K Retrieval", min_value=1, max_value=10, value=3)
    
    st.divider()
    
    st.header("🧪 Select Scenario")
    scenario = st.radio(
        "Choose a Failure Mode:",
        [
            "Multi-Hop Reasoning Gap",
            "Extraction Error / Partial Context",
            "Custom Query"
        ]
    )

    if st.button("Re-Index Knowledge Base"):
        st.session_state.rag_system = None
        st.experimental_rerun()

# Initialize RAG System
if "rag_system" not in st.session_state or st.session_state.rag_system is None:
    with st.spinner("Building Index..."):
        try:
            rag = NaiveRAG()
            docs = load_text_files("data")
            rag.build_index(docs, chunk_size_words=chunk_size, overlap_words=overlap)
            st.session_state.rag_system = rag
            st.session_state.docs = docs
            st.success("Index Built Successfully!")
        except Exception as e:
            st.error(f"Error building index: {e}")
            st.stop()

rag = st.session_state.rag_system

# Main Content Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📚 Knowledge Base")
    with st.expander("View Source Document (icc_world_cup_2026.txt)", expanded=False):
        st.text(st.session_state.docs.get("icc_world_cup_2026.txt", "File not found"))

    st.subheader("🔍 Query Interface")
    
    default_query = ""
    
    if scenario == "Multi-Hop Reasoning Gap":
        default_query = "Which group is the team that replaced Bangladesh in?"
    elif scenario == "Extraction Error / Partial Context":
        default_query = "Who is hosting the ICC Men’s T20 World Cup 2026 and when does it start?"
    
    query = st.text_input("Enter your question:", value=default_query)
    
    if st.button("Run RAG Pipeline", type="primary"):
        with st.spinner("Retrieving and Generating..."):
            # 1. Retrieve
            retrieved_chunks = rag.retrieve(query, top_k=top_k)
            
            # 2. Generate
            # Using the same logic as the CLI script
            filtered = [(c, s) for c, s in retrieved_chunks if s >= 0.0] # show all positive matches
            answer, report = rag.generate_extractive_answer(query, filtered)
            
            # 3. Dynamic Analysis
            insights = dynamic_analysis(query, answer, retrieved_chunks)
            
            st.session_state.last_result = {
                "query": query,
                "retrieved": retrieved_chunks,
                "answer": answer,
                "insights": insights
            }

with col2:
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        
        st.subheader("🤖 System Output")
        
        # Display Answer
        st.markdown(f"**Generated Answer:**")
        if res['answer']:
            st.info(f"📄 {res['answer']}")
        else:
            st.warning("No answer could be extracted.")
            
        # Analysis of Failure (if applicable)
        if res['insights']:
            with st.expander("🧐 Dynamic Analysis", expanded=True):
                for insight in res['insights']:
                    st.markdown(insight)

        st.divider()
        
        # Display Retrieved Context
        st.subheader(f"🧩 Top-{top_k} Retrieved Chunks")
        
        for i, (chunk, score) in enumerate(res['retrieved']):
            score_color = "green" if score > 0.4 else "orange" if score > 0.25 else "red"
            border_style = "2px solid #4CAF50" if i == 0 else "1px solid #ddd"
            
            st.markdown(f"""
            <div class="chunk-box" style="border-left: 5px solid {score_color};">
                <div style="display:flex; justify-content:space-between;">
                    <strong>Chunk #{chunk.chunk_id}</strong>
                    <span style="color:{score_color}; font-weight:bold;">Score: {score:.3f}</span>
                </div>
                <hr style="margin: 5px 0;">
                <p style="font-family:monospace; font-size:0.9em;">{chunk.text}</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.caption("Lower scores indicate weaker semantic similarity to the query.")

if __name__ == "__main__":
    pass
