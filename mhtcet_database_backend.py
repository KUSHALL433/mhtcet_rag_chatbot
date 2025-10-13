from langgraph.graph import StateGraph,START,END,MessagesState
from langchain_huggingface import HuggingFaceEndpointEmbeddings,ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import TypedDict,Annotated,Sequence
from langchain_core.messages import BaseMessage,HumanMessage,ToolMessage,SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import os

load_dotenv()
HUGGINGFACE_API=os.getenv("HUGGINGFACE_API_KEY")

conn=sqlite3.connect(database='mhtcet_rag.db',check_same_thread=False)
checkpointer=SqliteSaver(conn=conn)

model=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)
model_em = "sentence-transformers/all-mpnet-base-v2"

embedding_model = HuggingFaceEndpointEmbeddings(
    model=model_em,
    huggingfacehub_api_token=HUGGINGFACE_API
)

INDEX_DIR = "mhtcet_index"
PDF_PATH = "MHTCET Pdf latest.pdf"

if os.path.exists(INDEX_DIR):
    print("✅ Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(INDEX_DIR, embedding_model, allow_dangerous_deserialization=True)
else:
    print("⚙️ Building FAISS index from PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_split = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=docs_split, embedding=embedding_model)
    vectorstore.save_local(INDEX_DIR)
    print("✅ FAISS index created and saved successfully!")

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={'k':3})

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools=[retrieve]

tool_nodes=ToolNode(tools=tools)

model_with_tools=model.bind_tools(tools)



def call_llm(state: MessagesState) -> MessagesState:
    """Function to call the LLM with the current state."""
    messages = state['messages']
    if messages and isinstance(messages[-1],ToolMessage):
        context=messages[-1].content
        system_prompt = f"""
            You are an intelligent AI assistant who answers questions about MHTCET Exam based on the PDF document loaded into your knowledge base.
            Use the retriever tool available to answer questions about the MHTCET exam. You can make multiple calls if needed.
            Summarize the context from retriever tool in best way covering all the points from results.Provide the final answer ONLY in summary of retrieved documents context.
            If you need to look up some information before asking a follow up question, you are allowed to do that! You can remeber user specified information.
            If no relevant information is found give a suitable reply to user.
            If user query is not related to MHTCET exam then ask user to keep query ONLY MHTCET related.
            
            Context:\n\n{context}
            """
        system_prompt=SystemMessage(content=system_prompt)
        user_message=next((m for m in reversed(messages) if isinstance(m,HumanMessage)),None)
        prompt=[system_prompt,user_message] if user_message else [system_prompt]
        final_response=model.invoke(prompt)
        return {'messages':[final_response]}
    
    response = model_with_tools.invoke(messages)
    return {'messages': [response]}


graph = StateGraph(MessagesState)
graph.add_node("llm", call_llm)
graph.add_node("tool_node", tool_nodes)

graph.add_conditional_edges(
    "llm",
    tools_condition,
    {'tools': "tool_node", '__end__': END}
)
graph.add_edge("tool_node", "llm")
graph.set_entry_point("llm")


# rag_agent
rag_agent = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
        
    return list(all_threads)


