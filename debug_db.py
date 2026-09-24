# debug_db.py
import chromadb
from chromadb.utils import embedding_functions
from backend.config import CHROMA_DB_PATH, FDA_COLLECTION

client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
ef         = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name="all-MiniLM-L6-v2")
collection = client.get_collection(FDA_COLLECTION, embedding_function=ef)

# Print first 10 metadata entries
results = collection.get(limit=10, include=["metadatas"])
for m in results["metadatas"]:
    print(m)