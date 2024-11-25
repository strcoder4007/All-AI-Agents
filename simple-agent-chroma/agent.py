import os
from dotenv import load_dotenv
import streamlit as st
from langchain_ollama import OllamaLLM
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import fitz
import time
import numpy as np

# Load environment variables
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("first")

# Streamlit app title
st.title("Search from Docs")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text

def chunk_text(text, max_length=256):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk) + ' ' + word) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    # Add the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# File uploader in Streamlit
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract text from the PDF
    text = extract_text_from_pdf(f"temp_{uploaded_file.name}")
    
    # Chunk the text
    chunks = chunk_text(text)

    data = [{"id": f"doc_{i}", "text": c} for i, c in enumerate(chunks)]

    embeddings = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[d['text'] for d in data],
        parameters={"input_type": "passage", "truncate": "END"}
    )

    records = []
    for d, e in zip(data, embeddings):
        records.append({
            "id": d['id'],
            "values": e['values'],
            "metadata": {'text': d['text']}
        })

    # Upsert the records into the index
    index.upsert(
        vectors=records,
        namespace="default"
    )

    time.sleep(10)  # Wait for the upserted vectors to be indexed

    print(index.describe_index_stats())
        



# llm = OllamaLLM(
#   model="llama3.1:8b",
#   base_url="http://localhost:11434"
# )
