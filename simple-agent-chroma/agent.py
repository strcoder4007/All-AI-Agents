import os
from dotenv import load_dotenv

import streamlit as st
# from langchain import VectorDBQA
from langchain_ollama import OllamaLLM
import pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

# Load the desired embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
# pc = pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index("first")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(len(doc)):
        text += doc.load_page(page_num).get_text()
    return text

def create_embeddings(text, model):
    sentences = text.split('.')
    embeddings = []
    for sentence in sentences:
        embedding = model.encode(sentence.strip(), convert_to_tensor=False)
        embeddings.append(embedding)
    return embeddings

# Streamlit app
st.title("PDF to Pinecone Vector Store")

llm = OllamaLLM(
  model="llama3.1:8b",
  base_url="http://localhost:11434"
)

# # Set up the RAG system
# rag_system = VectorDBQA(
#     vector_db=index,
#     llm=llm,
# )

# def retrieve_and_generate(query):
#     response = rag_system.run(query)
#     return response

# Streamlit Frontend
st.title("AI Multi-Agent Chatbot")

# Initialize the session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Create a text input field for user queries
user_query = st.text_input("Ask a question", value=st.session_state.query, on_change=lambda: st.session_state.update(query=user_query))

# # Button to trigger the RAG system
# if st.button("Get Answer"):
#     with st.spinner("Generating response..."):
#         answer = retrieve_and_generate(embedding_model.encode(user_query))
#         st.session_state.query = user_query  # Update session state with current query
#         st.write(f"Answer: {answer}")