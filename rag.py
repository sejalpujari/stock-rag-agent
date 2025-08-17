import os
from dotenv import load_dotenv
import streamlit as st
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages

# Groq
from langchain_groq import ChatGroq

# Embeddings + Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# PDF loader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

# --------------------
# Load API Key
# --------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# --------------------
# LLM
# --------------------

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"])
# --------------------
# Embeddings + Vector DB
# --------------------
persist_directory = "./persist"
collection_name = "stock_market"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ensure persistence folder exists
os.makedirs(persist_directory, exist_ok=True)

# Load PDF once and persist to DB
pdf_path = "Stock_Market_Performance_2024.pdf"
if os.path.exists(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pages_split = text_splitter.split_documents(pages)

    if not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        vectorstore = Chroma.from_documents(
            documents=pages_split,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
else:
    st.error("PDF file not found. Please upload `Stock_Market_Performance_2024.pdf` to the repo.")

# Reconnect to Chroma DB
vectorstore = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# --------------------
# Retriever Tool
# --------------------
@tool
def retriever_tool(query: str) -> str:
    """Searches Stock Market PDF via retriever."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Doc {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

tools = [retriever_tool]
llm = llm.bind_tools(tools)
tools_dict = {t.name: t for t in tools}

# --------------------
# Agent State
# --------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 
based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions.
Always cite the document parts you use.
"""

# --------------------
# Agent Nodes
# --------------------
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        if t['name'] not in tools_dict:
            result = "Incorrect Tool Name."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

# --------------------
# Graph
# --------------------
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()

# --------------------
# Streamlit UI
# --------------------
st.title("📊 Stock Market RAG Agent (Groq + Chroma)")

query = st.text_input("Ask a question about the Stock Market PDF:")
if st.button("Submit") and query:
    with st.spinner("Thinking..."):
        messages = [HumanMessage(content=query)]
        result = rag_agent.invoke({"messages": messages})
        st.success(result['messages'][-1].content)
