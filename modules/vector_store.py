import chromadb
from chromadb.config import Settings

def init_vector_store():
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("real_docs")
