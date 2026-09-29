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
from backend.rag.brand_resolver import resolve_brand_to_generic


def run_query(user_question: str) -> dict:
    print(f"\n{'='*50}")
    print(f"Query: {user_question}")

    # Step 1 — Extract drug names
    drugs_found = extract_drug_names(user_question)
    print(f"Drugs detected: {drugs_found}")

    # Step 2 — Resolve brand → generic
    resolved_drugs = []
    brand_map      = {}

    for drug in drugs_found:
        generics = resolve_brand_to_generic(drug)
        for g in generics:
            if g not in resolved_drugs:
                resolved_drugs.append(g)
            brand_map[g] = drug

    print(f"Resolved generics: {resolved_drugs}")

    # Step 3 — Retrieve
    if len(resolved_drugs) >= 2:
        docs = retrieve_by_drugs(resolved_drugs[0], resolved_drugs[1])
    elif len(resolved_drugs) == 1:
        docs = retrieve(resolved_drugs[0] + " drug interaction warnings")
    else:
        docs = retrieve(user_question)

    print(f"Retrieved {len(docs)} docs from local DB")

    # Step 4 — Check if docs are actually relevant
    # (not just any 8 docs — do they mention our drugs?)
    def docs_mention_drugs(docs, drug_list):
        combined_text = " ".join(d["text"].lower() for d in docs)
        return any(drug.lower() in combined_text for drug in drug_list)
    
    def all_drugs_covered(docs, drug_list):
        """Returns list of drugs NOT found in retrieved docs."""
        combined_text = " ".join(d["text"].lower() for d in docs)
        return [drug for drug in drug_list
                if drug.lower() not in combined_text]
    missing_drugs = all_drugs_covered(docs, resolved_drugs)
    print(f"Missing drugs in docs: {missing_drugs}")
    
    relevant = docs_mention_drugs(docs, resolved_drugs)
    print(f"Docs relevant to query: {relevant}")

    if missing_drugs:
        print(f"⚡ Fetching missing drugs via MCP: {missing_drugs}")

        for drug in missing_drugs[:3]:
            mcp_result = fetch_via_mcp_fda(drug)
            print(f"  FDA MCP '{drug}': {mcp_result[:60]}...")

        # PubMed for the pair
        if len(resolved_drugs) >= 2:
            fetch_via_mcp_pubmed(resolved_drugs[0], resolved_drugs[1])

        # Re-retrieve after new docs added
        if len(resolved_drugs) >= 2:
            docs = retrieve_by_drugs(resolved_drugs[0], resolved_drugs[1])
        elif resolved_drugs:
            docs = retrieve(resolved_drugs[0] + " drug interaction")

        print(f"Retrieved {len(docs)} docs after MCP fetch")
    # Step 6 — Build enriched question
    enriched_question = user_question
    if brand_map:
        notes = ", ".join([
            f"{orig} contains {gen}"
            for gen, orig in brand_map.items()
            if orig.lower() != gen.lower()
        ])
        if notes:
            enriched_question += f"\n\nNote for context: {notes}"

    # Step 7 — Generate answer
    answer  = query_gemini(enriched_question, docs)
    sources = format_sources(docs)

    return {
        "answer":         answer,
        "sources":        sources,
        "drugs_found":    drugs_found,
        "resolved_drugs": resolved_drugs,
        "doc_count":      len(docs),
    }
    
    
def extract_drug_names(text: str) -> list[str]:
    STOP_WORDS = {
        "take", "taking", "can", "i", "is", "it", "safe",
        "with", "and", "the", "a", "an", "to", "for",
        "my", "use", "using", "drug", "medicine", "medication",
        "between", "together", "dangerous", "harmful", "okay",
        "are", "there", "any", "what", "how", "does",
        "patient", "context", "old", "year", "age",
        "high", "low", "blood", "pressure", "diabetes",
    }

    found = []
    text_lower = text.lower()

    # ── Pattern 1: "take X and Y", "X with Y" ─────────────────
    import re
    patterns = [
        r"take\s+([\w\s\-]+?)\s+(?:with|and)\s+([\w\s\-]+?)(?:\?|$|\.|\,)",
        r"between\s+([\w\s\-]+?)\s+and\s+([\w\s\-]+?)(?:\?|$|\.|\,)",
        r"([\w\s\-]+?)\s+(?:with|and)\s+([\w\s\-]+?)\s+(?:together|interaction|safe)(?:\?|$|\.|\,)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            for m in match:
                m = m.strip()
                if m and m not in STOP_WORDS and len(m) > 2:
                    # Keep multi-word brand names intact
                    if m not in found:
                        found.append(m)

    # ── Pattern 2: Capitalized words (brand names) ─────────────
    # Look for sequences of capitalized words: "Montair LC"
    cap_pattern = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b', text)
    for phrase in cap_pattern:
        p = phrase.strip().lower()
        words = p.split()
        if all(w not in STOP_WORDS for w in words) and len(p) > 2:
            if p not in found:
                found.append(p)

    # Remove duplicates and substrings
    # e.g. if we have both "montair" and "montair lc", keep "montair lc"
    final = []
    for drug in found:
        # skip if a longer version already exists
        if not any(drug != other and drug in other for other in found):
            final.append(drug)

    return final[:3]


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