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
st.header("chat_with_data")

