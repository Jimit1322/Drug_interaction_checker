"""
Fetches drug interaction research papers from PubMed API.
PubMed is free — no API key needed for basic use.

We search for: "drug interaction [drug_name]"
and store the abstracts in ChromaDB.
"""

import time
import requests
import chromadb
from tqdm import tqdm
from xml.etree import ElementTree as ET
from chromadb.utils import embedding_functions
from backend.config import CHROMA_DB_PATH, PUBMED_COLLECTION

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

SEARCH_TERMS = [
    "drug drug interaction",
    "adverse drug reaction",
    "drug interaction warfarin",
    "drug interaction aspirin",
    "drug interaction metformin",
    "drug interaction statins",
    "drug interaction antibiotics",
    "drug interaction antidepressants",
    "drug interaction blood pressure",
    "drug interaction diabetes medication",
    "pharmacokinetic drug interaction",
    "drug interaction mechanism review",
    "dangerous drug combination",
    "contraindicated drug combination",
]


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=PUBMED_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"description": "PubMed drug interaction research papers"}
    )
    return collection


def search_pubmed(query: str, max_results: int = 50) -> list[str]:
    """
    Step 1: Search PubMed for paper IDs matching a query.
    Returns list of PubMed IDs (PMIDs).
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    try:
        r = requests.get(f"{BASE_URL}/esearch.fcgi",
                         params=params, timeout=20)
        r.raise_for_status()
        data  = r.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
        return pmids
    except Exception as e:
        print(f"  Search error for '{query}': {e}")
        return []


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """
    Step 2: Fetch full abstracts for a list of PMIDs.
    Returns list of paper dicts with title, abstract, authors, year.
    """
    if not pmids:
        return []

    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    try:
        r = requests.get(f"{BASE_URL}/efetch.fcgi",
                         params=params, timeout=30)
        r.raise_for_status()
        return parse_pubmed_xml(r.text)
    except Exception as e:
        print(f"  Fetch error: {e}")
        return []


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parses PubMed XML response into structured dicts."""
    papers = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    for article in root.findall(".//PubmedArticle"):
        try:
            # PMID
            pmid_el = article.find(".//PMID")
            pmid    = pmid_el.text if pmid_el is not None else "unknown"

            # Title
            title_el = article.find(".//ArticleTitle")
            title    = title_el.text if title_el is not None else ""
            if title is None:
                title = ""

            # Abstract — can have multiple sections
            abstract_parts = []
            for ab in article.findall(".//AbstractText"):
                label = ab.get("Label", "")
                text  = ab.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
            abstract = " ".join(abstract_parts).strip()

            # Year
            year_el = article.find(".//PubDate/Year")
            year    = year_el.text if year_el is not None else "unknown"

            # Authors
            authors = []
            for author in article.findall(".//Author")[:3]:  # first 3 only
                last  = author.findtext("LastName",  "")
                first = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{last} {first}".strip())
            author_str = ", ".join(authors)

            # Journal
            journal_el = article.find(".//Journal/Title")
            journal    = journal_el.text if journal_el is not None else ""

            # Only keep papers with actual abstracts
            if abstract and len(abstract) > 100:
                papers.append({
                    "pmid":     pmid,
                    "title":    title,
                    "abstract": abstract,
                    "year":     year,
                    "authors":  author_str,
                    "journal":  journal,
                })

        except Exception:
            continue

    return papers


def store_papers(collection, papers: list[dict]):
    """Stores paper abstracts in ChromaDB."""
    if not papers:
        return

    ids        = [f"pubmed_{p['pmid']}" for p in papers]
    documents  = [f"{p['title']}\n\n{p['abstract']}" for p in papers]
    metadatas  = [{
        "pmid":    p["pmid"],
        "title":   p["title"][:500],
        "year":    p["year"],
        "authors": p["authors"][:300],
        "journal": p["journal"][:200],
        "source":  "PubMed",
        "url":     f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/",
    } for p in papers]

    collection.upsert(
        ids       = ids,
        documents = documents,
        metadatas = metadatas,
    )


def run_pubmed_ingestion(max_per_term: int = 50):
    """
    Full pipeline:
    1. Search PubMed for each term in SEARCH_TERMS
    2. Fetch abstracts for found papers
    3. Store in ChromaDB
    """
    print("=" * 60)
    print("PubMed Ingestion")
    print("=" * 60)

    collection   = get_chroma_collection()
    existing     = collection.count()
    print(f"ChromaDB already has {existing} PubMed chunks.")

    total_stored = 0

    for term in tqdm(SEARCH_TERMS, desc="Search terms"):
        print(f"\nSearching: '{term}'")

        pmids = search_pubmed(term, max_results=max_per_term)
        if not pmids:
            continue

        print(f"  Found {len(pmids)} papers")

        # Filter out already stored
        new_pmids = []
        for pmid in pmids:
            try:
                check = collection.get(ids=[f"pubmed_{pmid}"])
                if not check["ids"]:
                    new_pmids.append(pmid)
            except:
                new_pmids.append(pmid)

        if not new_pmids:
            print(f"  All already stored, skipping")
            continue

        # Fetch in batches of 20
        for i in range(0, len(new_pmids), 20):
            batch   = new_pmids[i:i+20]
            papers  = fetch_abstracts(batch)
            store_papers(collection, papers)
            total_stored += len(papers)
            time.sleep(0.5)   # PubMed rate limit: 3 req/sec without API key

    print("\n" + "=" * 60)
    print(f"PubMed ingestion complete!")
    print(f"  Papers stored   : {total_stored}")
    print(f"  ChromaDB size   : {collection.count()} chunks")
    print("=" * 60)