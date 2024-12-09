import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
import sys
import os
from time import sleep 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
# Load environment variables
load_dotenv()
###################################################################################################################################
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=GROQ_API_KEY
)
###################################################################################################################################
loader = PyPDFLoader("Artificial Intelligence for Career Guidance – Current Requirements and Prospects for the Future.pdf")
pages = loader.load()
chunk_size =500
chunk_overlap = 100
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
text_chunks=r_splitter.split_documents(pages)
# # download the embeddings to use to represent text chunks in a vector space, using the pre-trained model "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectordb.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)










st.header("chat_with_data")

