# debug_db2.py — run after a few minutes of ingestion
import chromadb
from chromadb.utils import embedding_functions
from backend.config import CHROMA_DB_PATH, FDA_COLLECTION

client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
ef         = embedding_functions.SentenceTransformerEmbeddingFunction(
                 model_name="all-MiniLM-L6-v2")
collection = client.get_collection(FDA_COLLECTION, embedding_function=ef)

results = collection.get(limit=20, include=["metadatas"])
names   = set(m["drug_name"] for m in results["metadatas"])
print("Drug names in DB:")
for n in sorted(names):
    print(f"  {n}")