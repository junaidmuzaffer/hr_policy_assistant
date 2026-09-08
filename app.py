import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="HR Policy Assistant", page_icon="🏢")
st.title("🏢 HR Policy Assistant (Powered by Groq)")

# Retrieve API Key
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please configure your GROQ_API_KEY in .env or Streamlit Secrets.")
    st.stop()

# Helper function to process uploaded files
def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary directory to let LangChain load it
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load based on file extension
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
            documents.extend(loader.load())
            
        os.remove(tmp_path) # Clean up temp file

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Generate embeddings and store in Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

# Sidebar for file uploads
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload HR Policy files (PDF or DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )

if not uploaded_files:
    st.info("Please upload at least one PDF or DOCX file in the sidebar to begin.")
    st.stop()

# Build vector store from uploaded files
with st.spinner("Processing uploaded documents..."):
    vectorstore = process_uploaded_files(uploaded_files)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for answering employee questions about HR policies.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise and professional.

Context:
{context}

Question: {question}"""
)

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0
)

# LCEL RAG Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat Interface
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
            answer = rag_chain.invoke(user_query)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
   
