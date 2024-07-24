from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from pinecone import Pinecone, PodSpec, ServerlessSpec
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import time
import pillow_heif
import unstructured_inference
import unstructured_pytesseract
import pytesseract

# Ensure tesseract is found
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USE_SERVERLESS = os.getenv('USE_SERVERLESS', 'true').lower() == 'true'

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def doc_preprocessing():
    loader = DirectoryLoader(
        '/content/data/',
        glob='**/*.pdf',     # only the PDFs
        show_progress=True
    )
    docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(docs)
    return docs_split

@st.cache_resource
def embedding_db():
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if USE_SERVERLESS:
        spec = ServerlessSpec(cloud='aws', region='us-west-2')
    else:
        # Ensure PINECONE_ENV is set for non-serverless configurations
        if not PINECONE_ENV:
            raise ValueError("PINECONE_ENV must be set when not using serverless configuration.")
        spec = PodSpec(environment=PINECONE_ENV)

    # Define the index name
    index_name = "stre"

    # Check if index already exists
    if index_name not in pc.list_indexes().names():
        # If it does not exist, create index
        pc.create_index(
            name=index_name,
            dimension=1536,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=spec
        )
        # Wait for index to be initialized
        time.sleep(1)

    # Connect to index
    index = pc.Index(index_name)
    index.describe_index_stats()

    # We use the OpenAI embedding model
    embeddings = OpenAIEmbeddings()
    docs_split = doc_preprocessing()
    doc_db = LangchainPinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name=index_name
    )
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result

def main():
    st.title("Question and Answering App powered by LLM and Pinecone")

    text_input = st.text_input("Ask your query...") 
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
