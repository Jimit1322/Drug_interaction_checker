"""
Main RAG pipeline — ties retriever + Gemini together.
This is the core of the project.

Flow:
  User query
    → understand query (extract drug names)
    → retrieve relevant docs from ChromaDB
    → send docs + query to Gemini
    → return structured response
"""

import re
from backend.rag.retriever    import retrieve, retrieve_by_drugs
from backend.llm.gemini_client import query_gemini
from backend.mcp.mcp_client import needs_live_fetch, fetch_via_mcp_fda, fetch_via_mcp_pubmed


def run_query(user_question: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Query: {user_question}")

    drugs_found = extract_drug_names(user_question)
    print(f"Drugs detected: {drugs_found}")

    # Step 1 — Local retrieval
    if len(drugs_found) >= 2:
        docs = retrieve_by_drugs(drugs_found[0], drugs_found[1])
    else:
        docs = retrieve(user_question)

    print(f"Retrieved {len(docs)} documents from local DB")

    # Step 2 — MCP fallback if not enough local docs
    if needs_live_fetch(docs, min_docs=3):
        print("⚡ Not enough local docs — fetching live via MCP...")

        if drugs_found:
            mcp_result = fetch_via_mcp_fda(drugs_found[0])
            print(f"MCP FDA result: {mcp_result[:100]}...")

        if len(drugs_found) >= 2:
            mcp_result = fetch_via_mcp_pubmed(drugs_found[0], drugs_found[1])
            print(f"MCP PubMed result: {mcp_result[:100]}...")

        # Re-retrieve after MCP populated ChromaDB
        if len(drugs_found) >= 2:
            docs = retrieve_by_drugs(drugs_found[0], drugs_found[1])
        else:
            docs = retrieve(user_question)

        print(f"Retrieved {len(docs)} documents after MCP fetch")

    # Step 3 — Generate answer
    answer  = query_gemini(user_question, docs)
    sources = format_sources(docs)

    return {
        "answer":      answer,
        "sources":     sources,
        "drugs_found": drugs_found,
        "doc_count":   len(docs),
    }

def extract_drug_names(text: str) -> list[str]:
    """
    Extracts drug names from user question.
    
    Simple approach: look for capitalized words or words
    after "take", "taking", "between", "and", "with".
    For production: use a medical NER model (spaCy + scispacy).
    """
    text_lower = text.lower()

    # Common patterns:
    # "can I take X with Y"
    # "interaction between X and Y"
    # "is X safe with Y"
    # "X and Y together"

    patterns = [
        r"take\s+(\w+)\s+(?:with|and)\s+(\w+)",
        r"between\s+(\w+)\s+and\s+(\w+)",
        r"(\w+)\s+(?:with|and)\s+(\w+)\s+(?:together|interaction|safe)",
        r"interaction.*?(\w+).*?(?:with|and).*?(\w+)",
        r"(\w+)\s+and\s+(\w+)\s+interaction",
    ]

    # Stop words to filter out
   # backend/rag/pipeline.py — add to STOP_WORDS in extract_drug_names
    STOP_WORDS = {
    "take", "taking", "can", "i", "is", "it", "safe",
    "with", "and", "the", "a", "an", "to", "for",
    "my", "use", "using", "drug", "medicine", "medication",
    "between", "together", "dangerous", "harmful", "okay",
    "are", "there", "any", "what", "how", "does",
    "patient", "context", "old", "year", "age",  # ← add these
    "high", "low", "blood", "pressure", "diabetes",
}

    found = []

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            if isinstance(match, tuple):
                for m in match:
                    if m not in STOP_WORDS and len(m) > 2:
                        if m not in found:
                            found.append(m)
            elif match not in STOP_WORDS and len(match) > 2:
                if match not in found:
                    found.append(match)

    # Also extract capitalized words from original text
    # (drug names are often capitalized: Warfarin, Ibuprofen)
    cap_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
    for word in cap_words:
        w = word.lower()
        if w not in STOP_WORDS and w not in found:
            found.append(w)

    return found[:3]   # max 3 drugs


def format_sources(docs: list[dict]) -> list[dict]:
    """Formats retrieved docs into clean source citations."""
    sources = []

    for doc in docs:
        if doc["source"] == "FDA DailyMed":
            sources.append({
                "type":    "FDA Drug Label",
                "name":    doc.get("drug_name", "Unknown"),
                "section": doc.get("section",   "Unknown"),
                "url":     f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?labeltype=all&query={doc.get('drug_name','')}",
                "icon":    "🏛️",
            })
        else:
            sources.append({
                "type":    "PubMed Research",
                "name":    doc.get("title", "Unknown")[:80],
                "pmid":    doc.get("pmid",  "Unknown"),
                "year":    doc.get("year",  "Unknown"),
                "url":     doc.get("url",   ""),
                "icon":    "🔬",
            })

    return sources