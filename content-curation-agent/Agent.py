import os
from dotenv import load_dotenv
import pinecone
from langchain_ollama import OllamaLLM
import fitz  # PyMuPDF for PDF processing
import json
import streamlit as st
import requests

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = pinecone.Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment="us-east-1-aws"
)


# Connect to the index
pinecone_client = pc.Index('first')

llm = OllamaLLM(
    model="llama3.1:8b",
    base_url="http://localhost:11434"
)

# Streamlit app title
st.title("Search from Docs")

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text

def chunk_text(text, max_length=256):
    """Chunk text into smaller segments."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(' '.join(current_chunk) + ' ' + word) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def embed_text(texts, input_type="passage"):
    return pc.inference.embed(
        model="multilingual-e5-large",
        inputs=texts,
        parameters={
            "input_type": input_type
        }
    )

def upsert_records(index, records):
    """Upsert records into the Pinecone index."""
    try:
        index.upsert(vectors=records)
        st.success("Records upserted successfully.")
    except Exception as e:
        st.error(f"Failed to upsert records: {e}")

# Handle file upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the PDF
    text = extract_text_from_pdf("temp.pdf")

    os.remove("temp.pdf")
    
    # Chunk the text
    chunks = chunk_text(text)

    # Generate embeddings for each chunk
    embeddings = embed_text(chunks)

    # Create records with ids and values
    records = [{"id": f"chunk_{i}", "values": embedding["values"]} for i, embedding in enumerate(embeddings)]

    # Upsert records into the Pinecone index
    upsert_records(pinecone_client, records)

# Handle query submission
query = st.text_input("Enter your query")
submit_button = st.button("Submit Query")

if submit_button and query.strip():
    try:
        # Convert the query into a numerical vector that Pinecone can search with
        query_embedding = embed_text([query], input_type="query")[0]

        # Search the index for the three most similar vectors
        results = pinecone_client.query(
            namespace="default",
            vector=query_embedding["values"],
            top_k=3,
            include_values=False,
            include_metadata=True
        )

        # Extract document chunks from the search results
        document_chunks = [item["metadata"]["chunk"] for item in results["matches"]]

        # Combine document chunks into a single context string
        context = "\n".join(document_chunks)

        # Prepare the data payload
        data = {
            "prompt": query,
            "context": context
        }

        # Send the request to the LLM using Ollama's API
        response = requests.post(f"{llm.base_url}/api/predict", json=data, headers={"Content-Type": "application/json"})

        # Check if the request was successful
        if response.status_code == 200:
            # Display the generated text from the LLM
            st.write(response.json().get("generated_text"))
        else:
            # Display the error message if the request failed
            st.error(f"Error: {response.status_code} - {response.text}")

    except Exception as e:
        st.error(f"An error occurred: {e}")