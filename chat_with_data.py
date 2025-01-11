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
    model="mixtral-8x7b-32768",
    api_key=GROQ_API_KEY
)
###################################################################################################################################
loader = PyPDFLoader("EUAS_Tuition and service fees_2024-2025-2.pdf")
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
from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
#sentences = ['search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten']
#embeddings = model.encode(sentences)
#print(embeddings)

#embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5")
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



st.header("Chat with Your Data")


query=st.text_input("Write Your Query Here")
if st.button("Submit"):
    res=rag_chain.invoke(query)
    
    st.write(res)



