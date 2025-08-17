from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages

# Groq + Ollama imports
from langchain_groq import ChatGroq
# from langchain_ollama.embeddings import OllamaEmbeddings

# Document loaders & vector DB
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

# --------------------
# 1. LLM (Groq model)
# --------------------
llm = ChatGroq(
    model="llama3-8b-8192",  # You can change to "llama-3.1-70b" or another available Groq model
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")  # expects GROQ_API_KEY in .env
)

# --------------------
# 2. Embeddings (Ollama)
# --------------------
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --------------------
# 3. PDF Load
# --------------------
pdf_path = "Stock_Market_Performance_2024.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)
pages = pdf_loader.load()
print(f"PDF has been loaded and has {len(pages)} pages")

# --------------------
# 4. Chunking
# --------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

pages_split = text_splitter.split_documents(pages)

persist_directory = r"C:\Users\Admin\Documents\agents"
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

# --------------------
# 5. Chroma VectorDB
# --------------------
vectorstore = Chroma.from_documents(
    documents=pages_split,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)
print(f"Created ChromaDB vector store!")

# --------------------
# 6. Retriever
# --------------------
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """Searches Stock Market PDF via retriever."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Doc {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])


tools = [retriever_tool]
llm = llm.bind_tools(tools)

# --------------------
# 7. Agent State
# --------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data.
Please always cite the specific parts of the documents you use in your answers.
"""


tools_dict = {our_tool.name: our_tool for our_tool in tools}

# --------------------
# 8. LLM Agent Node
# --------------------
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
    message = llm.invoke(messages)
    return {'messages': [message]}

# --------------------
# 9. Retriever Node
# --------------------
def take_action(state: AgentState) -> AgentState:
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query')}")
        if t['name'] not in tools_dict:
            result = "Incorrect Tool Name."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    return {'messages': results}

# --------------------
# 10. Graph Definition
# --------------------
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)

graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")

rag_agent = graph.compile()

# --------------------
# 11. Running Agent
# --------------------

def running_agent():
    print("\n=== RAG AGENT (Groq) ===")
    while True:
        user_input = input("\nAsk your question (or 'exit'): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        messages = [HumanMessage(content=user_input)]
        result = rag_agent.invoke({"messages": messages})
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Same persist directory & collection name you used
persist_directory = r"C:\Users\Admin\Documents\agents"
collection_name = "stock_market"

# Load embeddings (must match what you used when creating DB)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Reconnect to existing Chroma DB
vectorstore = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embeddings
)

print("\n=== Chroma DB Content ===")
docs = vectorstore.get()   # get all stored docs

# docs is a dictionary with ids, embeddings, metadatas, documents
print(f"Total documents stored: {len(docs['documents'])}\n")

# Show a few entries
for i in range(min(2, len(docs['documents']))):   # show only first 3 chunks
    print(f"ID: {docs['ids'][i]}")
    print(f"Metadata: {docs['metadatas'][i]}")
    print(f"Document: {docs['documents'][i][:200]}...")  # print first 200 chars
    print("-"*50)


# Run only when executing rag.py directly
if __name__ == "__main__":
    running_agent()

