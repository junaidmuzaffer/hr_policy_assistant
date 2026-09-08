import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 HR Policy Assistant (Powered by Groq)")
st.write("Ask any question regarding leave policy, working hours, benefits, etc.")

# Check for Groq API key in .env or Streamlit Secrets
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to your .env file or Streamlit secrets.")
    st.stop()

@st.cache_resource
def initialize_vector_store():
    # 1. Load PDFs from docs folder
    if not os.path.exists("docs") or not os.listdir("docs"):
        st.warning("Please add at least one PDF file into the 'docs/' directory.")
        st.stop()
        
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    
    # 2. Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # 3. Create free local embeddings (no API key needed for embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 4. Store chunks in vector store
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# Initialize Vector Store
with st.spinner("Processing HR Documents..."):
    vectorstore = initialize_vector_store()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# System prompt definition
system_prompt = (
    "You are an assistant for answering employee questions about HR policies.\n"
    "Use the following pieces of retrieved context to answer the question.\n"
    "If you don't know the answer, say that you don't know.\n"
    "Keep the answer concise and professional.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Initialize LLM using Groq API
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0
)

# Build RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about HR policy..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_query})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
