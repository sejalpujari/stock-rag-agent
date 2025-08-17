import streamlit as st
from langchain_core.messages import HumanMessage
from rag import rag_agent   # Import your RAG agent from rag.py

st.set_page_config(page_title="📊 Stock Market RAG Agent", page_icon="📈", layout="wide")

st.title("📊 Stock Market Performance 2024 - RAG Agent")
st.markdown("Ask me questions about **Stock Market Performance (2024)** based on the loaded PDF.")

# Sidebar
with st.sidebar:
    st.subheader("ℹ️ About")
    st.write("This app uses **Groq LLM + HuggingFace embeddings + ChromaDB**.")
    st.write("It retrieves context from the `Stock_Market_Performance_2024.pdf` document.")

# Input box
user_input = st.text_area("💬 Ask a question:", placeholder="e.g. How did tech stocks perform in Q1 2024?", height=100)

if st.button("🔍 Get Answer", use_container_width=True):
    if user_input.strip():
        with st.spinner("Thinking..."):
            # Prepare messages and run RAG agent
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})

            st.subheader("✅ Answer")
            st.write(result["messages"][-1].content)

    else:
        st.warning("⚠️ Please enter a question.")

# Footer
st.markdown("---")
st.caption("Built with LangGraph,LangChain, ChromaDB, Groq, HuggingFace, and Streamlit 🚀")
