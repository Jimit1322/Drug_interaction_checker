# test_db.py  — run from project root
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
             model_name="all-MiniLM-L6-v2")

fda_col    = client.get_collection("fda_drug_labels",    embedding_function=ef)
pubmed_col = client.get_collection("pubmed_abstracts",   embedding_function=ef)

print(f"FDA chunks    : {fda_col.count()}")
print(f"PubMed chunks : {pubmed_col.count()}")

# Test search
results = fda_col.query(
    query_texts=["warfarin ibuprofen interaction bleeding"],
    n_results=3
)
print("\nTop 3 search results:")
for i, (doc, meta) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0]
)):
    print(f"\n[{i+1}] Drug: {meta['drug_name']} | Section: {meta['section']}")
    print(f"     {doc[:200]}...")