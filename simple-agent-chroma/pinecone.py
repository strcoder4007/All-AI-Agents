from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "my_index"

pc.create_index(
    name=index_name,
    dimension=2, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)