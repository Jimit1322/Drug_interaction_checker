"""
Retrieves relevant documents from ChromaDB for a given query.
Searches both FDA and PubMed collections and merges results.
"""

import chromadb
from chromadb.utils import embedding_functions
from backend.config import (
    CHROMA_DB_PATH, FDA_COLLECTION,
    PUBMED_COLLECTION, TOP_K
)

# ── Setup ─────────────────────────────────────────────────────
_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
_ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
              model_name="all-MiniLM-L6-v2")

_fda_col    = _client.get_collection(FDA_COLLECTION,    embedding_function=_ef)
_pubmed_col = _client.get_collection(PUBMED_COLLECTION, embedding_function=_ef)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Main retrieval function.
    
    1. Searches FDA collection   → top_k results
    2. Searches PubMed collection → top_k results
    3. Merges and re-ranks by relevance score
    4. Returns top_k combined results

    Each result dict has:
        text, source, drug_name/title, section/pmid,
        score (lower = more relevant in ChromaDB)
    """
    fda_results    = _search_fda(query,    top_k)
    pubmed_results = _search_pubmed(query, top_k)

    # Merge both sources
    all_results = fda_results + pubmed_results

    # Sort by relevance score (ChromaDB returns L2 distance — lower is better)
    all_results.sort(key=lambda x: x["score"])

    # Return top_k after merging
    return all_results[:top_k]


def retrieve_by_drugs(drug1: str, drug2: str,
                      top_k: int = TOP_K) -> list[dict]:
    """
    Targeted retrieval when user specifies two drug names.
    Searches with multiple query angles for better coverage.
    """
    queries = [
        f"{drug1} {drug2} interaction",
        f"{drug1} drug interaction warnings",
        f"{drug2} drug interaction warnings",
        f"{drug1} {drug2} contraindicated",
        f"{drug1} {drug2} adverse effects",
    ]

    seen_ids = set()
    all_docs = []

    for q in queries:
        results = retrieve(q, top_k=4)
        for doc in results:
            doc_id = doc.get("id", doc["text"][:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    # Re-rank: prioritize docs mentioning both drug names
    all_docs = _boost_relevant(all_docs, drug1, drug2)

    return all_docs[:top_k]


def _search_fda(query: str, top_k: int) -> list[dict]:
    """Search FDA ChromaDB collection."""
    try:
        results = _fda_col.query(
            query_texts = [query],
            n_results   = top_k,
            include     = ["documents", "metadatas", "distances"]
        )
        return _format_results(results, "FDA DailyMed")
    except Exception as e:
        print(f"FDA search error: {e}")
        return []


def _search_pubmed(query: str, top_k: int) -> list[dict]:
    """Search PubMed ChromaDB collection."""
    try:
        results = _pubmed_col.query(
            query_texts = [query],
            n_results   = top_k,
            include     = ["documents", "metadatas", "distances"]
        )
        return _format_results(results, "PubMed")
    except Exception as e:
        print(f"PubMed search error: {e}")
        return []


def _format_results(raw_results: dict, source: str) -> list[dict]:
    """Converts raw ChromaDB output into clean list of dicts."""
    formatted = []

    docs      = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metadatas, distances):
        entry = {
            "text":   doc,
            "source": source,
            "score":  dist,
        }
        # Add source-specific fields
        if source == "FDA DailyMed":
            entry["drug_name"] = meta.get("drug_name",    "Unknown")
            entry["section"]   = meta.get("section",      "Unknown")
            entry["set_id"]    = meta.get("set_id",       "")
            entry["id"]        = f"fda_{meta.get('set_id','')}_{doc[:20]}"
        else:
            entry["title"]   = meta.get("title",   "Unknown")
            entry["pmid"]    = meta.get("pmid",    "Unknown")
            entry["year"]    = meta.get("year",    "Unknown")
            entry["authors"] = meta.get("authors", "Unknown")
            entry["journal"] = meta.get("journal", "Unknown")
            entry["url"]     = meta.get("url",     "")
            entry["id"]      = f"pubmed_{meta.get('pmid','')}"

        formatted.append(entry)

    return formatted


def _boost_relevant(docs: list[dict],
                    drug1: str, drug2: str) -> list[dict]:
    """
    Re-ranks docs by boosting those that mention both drug names.
    Simple but effective — docs mentioning both drugs are most relevant.
    """
    d1 = drug1.lower()
    d2 = drug2.lower()

    def boost_score(doc):
        text  = doc["text"].lower()
        score = doc["score"]
        # Reduce score (= increase relevance) if drug names appear
        if d1 in text and d2 in text:
            score *= 0.5    # both drugs mentioned — most relevant
        elif d1 in text or d2 in text:
            score *= 0.75   # one drug mentioned — somewhat relevant
        return score

    return sorted(docs, key=boost_score)