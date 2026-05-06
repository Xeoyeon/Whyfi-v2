import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
client = QdrantClient(url=qdrant_url)

VECTOR_SIZE = 1024 # BAAI/bge-m3's output size

def init_collections():
    # 1. 'Finance Dictionary' collection
    dict_collection ="finance_dictionary"
    if not client.collection_exists(collection_name=dict_collection):
        client.create_collection(
            collection_name=dict_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        ),
        print(f"{dict_collection} collection created.")
    else:
        print(f"{dict_collection} collection already exists.")
    
    # 2. 'Market News' collection
    news_collection = "market_news"
    if not client.collection_exists(collection_name=news_collection):
        client.create_collection(
            collection_name=news_collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        ),
        print(f"{news_collection} collection created.")
    else:
        print(f"{news_collection} collection already exists.")

if __name__ == "__main__":
    print("Initializing Qdrant collections...")
    init_collections()
    print("Initialization complete.")
