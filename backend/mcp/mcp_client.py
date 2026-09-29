"""
Direct MCP client — calls ingestion functions directly
instead of subprocess (more reliable).
"""

import time
import requests
from backend.ingestion.fda_loader    import (
    fetch_drug_label_xml, parse_drug_label,
    store_chunks, get_chroma_collection
)
from backend.ingestion.pubmed_loader import (
    search_pubmed, fetch_abstracts,
    store_papers, get_chroma_collection as get_pubmed_collection
)


def needs_live_fetch(docs: list[dict], min_docs: int = 3) -> bool:
    return len(docs) < min_docs


def fetch_via_mcp_fda(drug_name: str) -> str:
    """Fetch FDA label for a drug and store in ChromaDB directly."""
    print(f"  [MCP FDA] Fetching: {drug_name}")

    search_url = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    try:
        r    = requests.get(search_url,
                            params={"drug_name": drug_name, "pagesize": 3},
                            timeout=15)
        data = r.json()
        spls = data.get("data", [])
    except Exception as e:
        return f"FDA search failed: {e}"

    if not spls:
        return f"No FDA data found for '{drug_name}'"

    collection   = get_chroma_collection()
    total_stored = 0

    for spl in spls[:2]:
        set_id = spl.get("setid", "")
        if not set_id:
            continue

        xml_text = fetch_drug_label_xml(set_id)
        if not xml_text:
            continue

        chunks = parse_drug_label(xml_text, set_id)
        if chunks:
            store_chunks(collection, chunks)
            total_stored += len(chunks)

        time.sleep(0.2)

    return f"Stored {total_stored} chunks for '{drug_name}'"


def fetch_via_mcp_pubmed(drug1: str, drug2: str = "") -> str:
    query = f"{drug1} {drug2} drug interaction".strip()
    print(f"  [MCP PubMed] Searching: '{query}'")

    try:
        pmids = search_pubmed(query, max_results=10)

        # Also search reverse order
        if drug2:
            pmids2 = search_pubmed(
                f"{drug2} {drug1} interaction safety", max_results=5)
            pmids  = list(dict.fromkeys(pmids + pmids2))  # deduplicate

        if not pmids:
            return f"No PubMed results for '{query}'"

        papers = fetch_abstracts(pmids)
        if papers:
            collection = get_pubmed_collection()
            store_papers(collection, papers)
            return f"Stored {len(papers)} papers for '{query}'"

        return "No abstracts found"

    except Exception as e:
        return f"PubMed fetch failed: {e}"