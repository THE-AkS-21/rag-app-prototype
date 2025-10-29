# src/logic.py

import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import Optional, List
from langchain_core.documents import Document

# Define a type hint for the chain
# A Runnable takes a string (question) and outputs a string (answer)
from langchain_core.runnables.base import Runnable


def create_rag_chain(youtube_url: str, hf_api_token: str) -> Optional[Runnable]:
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using LCEL.

    Args:
        youtube_url (str): The URL of the YouTube video.
        hf_api_token (str): The Hugging Face API token.

    Returns:
        Optional[Runnable]: A runnable chain object, or None if setup fails.
    """
    try:
        # 1. Load Transcript
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()
        if not docs:
            st.error("Could not load transcript. The video may have transcripts disabled.")
            return None

        # 2. Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever()

        # 4. Define the LLM
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_api_token,
            temperature=0.7,
        )

        # 5. Define the Prompt Template
        template = """
        You are a helpful AI assistant. Answer the question based only on the following context from a video transcript:
        {context}

        Question: {question}

        Answer:
        """
        prompt = PromptTemplate.from_template(template)

        # Helper function to format retrieved documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # 6. Create the RAG Chain using LCEL
        rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        return rag_chain

    except Exception as e:
        st.error(f"An error occurred during chain creation: {e}")
        return None


def get_answer(chain: Runnable, query: str) -> str:
    """
    Invokes the RAG chain with a query to get an answer.

    Args:
        chain (Runnable): The configured LCEL RAG chain.
        query (str): The user's question.

    Returns:
        str: The answer from the LLM.
    """
    if not chain:
        return "The processing chain is not initialized."

    # The invoke method now runs the entire chain: retrieval -> prompt -> llm -> output
    return chain.invoke(query)