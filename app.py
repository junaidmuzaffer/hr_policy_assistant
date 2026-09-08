import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
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
st.write("Ask any question regarding leave policy, working hours, benefits, etc.")

# Check API Key
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Please add your GROQ_API_KEY to your .env file or Streamlit secrets.")
    st.stop()

@st.cache_resource
def initialize_vector_store():
    if not os.path.exists("docs") or not os.listdir("docs"):
        st.warning("Please add at least one PDF file into the 'docs/' directory.")
        st.stop()
        
    loader = PyPDFDirectoryLoader("docs")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    return vectorstore

with st.spinner("Processing HR Documents..."):
    vectorstore = initialize_vector_store()

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Helper function to format documents as string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt Definition
prompt = ChatPromptTemplate.from_template(
    """You are an assistant for answering employee questions about HR policies.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know.
Keep the answer concise and professional.

Context:
{context}

Question: {question}"""
)

# LLM Initialization
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
    temperature=0
)

# Modern LCEL RAG Chain (No legacy imports required)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

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
            answer = rag_chain.invoke(user_query)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer}) 
