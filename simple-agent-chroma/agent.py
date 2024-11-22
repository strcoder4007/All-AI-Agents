import os
from dotenv import load_dotenv

import streamlit as st
from langchain import VectorDBQA, LangChain
from langchain_ollama import OllamaLLM
import pinecone


load_dotenv()

# import tensorflow as tf
# import tensorflow_hub as hub

# # Load the USE model
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# embed = hub.load(module_url)

# def get_embeddings(texts):
#     return embed(texts).numpy()


# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="your_environment")

# Assume you have a Pinecone index named 'my_index'
index = pinecone.Index("my_index")

# Initialize Ollama with your local model
llm = OllamaLLM(
  model="llama3.1:8b",
  base_url="http://localhost:11434"
)

# Set up the RAG system
rag_system = VectorDBQA(
    vector_db=index,
    llm=llm,
)

def retrieve_and_generate(query):
    response = rag_system.run(query)
    return response

# Streamlit Frontend
st.title("AI Multi-Agent Chatbot")

# Initialize the session state for query
if 'query' not in st.session_state:
    st.session_state.query = ""

# Create a text input field for user queries
user_query = st.text_input("Ask a question", value=st.session_state.query, on_change=lambda: st.session_state.update(query=user_query))

# Button to trigger the RAG system
if st.button("Get Answer"):
    with st.spinner("Generating response..."):
        answer = retrieve_and_generate(user_query)
        st.session_state.query = user_query  # Update session state with current query
        st.write(f"Answer: {answer}")