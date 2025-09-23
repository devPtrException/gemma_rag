import os
from urllib import response
import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


st.title("RAG Gemma 2.0")


llm = ChatGroq(
    # groq_api_key=
    GROQ_API_KEY,
    model="Gemma-7b-it",
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context.
    if you find no relevant answers in the context,
    say "I don't know".
    <context>
    {context}
    <context>
    Questions: {input}

    """
)

document_chain = create_stuff_documents_chain(lln, prompt)
retriever = st.session_state.vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt = st.text_input("Your prompt here")

if prompt:
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": prompt})
    print("response time :", time.process_time() - start_time)
    st.write(response["answer"])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("......................")


def vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        st.session_state.final_documents = (
            st.session_state.text_splitter.split_documents(
                st.session_state.docs,
            )
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_docume,
            st.session_state.embeddings,
        )


promptt = "What do you want to ask from the documents?"


if st.button("Create Vector Store"):
    vector_embeddings()
    st.write("VectorStore is ready")
