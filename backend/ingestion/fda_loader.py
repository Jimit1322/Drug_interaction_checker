"""
Downloads FDA drug labels from DailyMed API and stores them in ChromaDB.

DailyMed is the official FDA drug label database.
API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

What we download:
  - Drug name
  - Active ingredients
  - Drug interactions section (most important)
  - Warnings section
  - Indications (what it's used for)
"""

import os
import json
import time
import requests
import chromadb
from tqdm import tqdm
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from backend.config import CHROMA_DB_PATH, FDA_DATA_PATH, FDA_COLLECTION


# ── ChromaDB setup ────────────────────────────────────────────
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Using sentence-transformers (free, no API key needed)
    # Model: all-MiniLM-L6-v2 — fast, good for medical text
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=FDA_COLLECTION,
        embedding_function=embedding_fn,
        metadata={"description": "FDA DailyMed drug labels"}
    )
    return collection


# ── Step 1: Get list of all drug SPL IDs from DailyMed ───────
def fetch_drug_list(limit=500):
    """
    DailyMed API returns all drug labels paginated.
    We fetch the first `limit` drugs — enough for a demo.
    For production: loop through all pages (~80,000 drugs).
    """
    print(f"Fetching list of {limit} drugs from DailyMed...")

    url    = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    drugs  = []
    page   = 1
    per_pg = 100   # max allowed by API

    while len(drugs) < limit:
        params = {"page": page, "pagesize": per_pg}
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()

            items = data.get("data", [])
            if not items:
                break

            drugs.extend(items)
            print(f"  Page {page}: got {len(items)} drugs "
                  f"(total so far: {len(drugs)})")

            page += 1
            time.sleep(0.3)    # be polite to the API

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

    return drugs[:limit]


# ── Step 2: Download XML label for one drug ───────────────────
def fetch_drug_label_xml(set_id: str) -> str | None:
    """
    Downloads the full SPL (Structured Product Label) XML for one drug.
    set_id is the unique identifier from DailyMed.
    """
    url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.text
    except Exception as e:
        return None


# ── Step 3: Parse XML → extract useful sections ───────────────
def parse_drug_label(xml_text: str, set_id: str) -> list[dict]:
    """
    Parses FDA SPL XML and extracts key sections.
    
    Returns list of chunks — we split by section so each chunk
    is focused on one topic (interactions, warnings, etc.)
    This gives better RAG retrieval than one giant document.
    """
    soup = BeautifulSoup(xml_text, "lxml-xml")
    chunks = []

    # ── Extract drug name ──────────────────────────────────────
    drug_name = "Unknown"
    name_tag  = soup.find("name")
    if name_tag:
        drug_name = name_tag.get_text(strip=True)

    # ── Extract manufacturer ───────────────────────────────────
    manufacturer = "Unknown"
    mfr_tag = soup.find("manufacturerOrganization")
    if mfr_tag:
        name_el = mfr_tag.find("name")
        if name_el:
            manufacturer = name_el.get_text(strip=True)

    # ── Extract active ingredients ─────────────────────────────
    ingredients = []
    for ingr in soup.find_all("activeIngredient"):
        n = ingr.find("name")
        if n:
            ingredients.append(n.get_text(strip=True))

    # ── Section codes we care about ────────────────────────────
    # These are standard FDA section codes in SPL format
    SECTION_CODES = {
        "34073-7":  "DRUG INTERACTIONS",
        "34071-1":  "WARNINGS",
        "34084-4":  "ADVERSE REACTIONS",
        "34067-9":  "INDICATIONS AND USAGE",
        "34068-7":  "DOSAGE AND ADMINISTRATION",
        "34070-3":  "CONTRAINDICATIONS",
        "43685-7":  "WARNINGS AND PRECAUTIONS",
    }

    # ── Extract each section ───────────────────────────────────
    for code, section_name in SECTION_CODES.items():
        # SPL uses <code code="34073-7"> inside <section>
        section_tag = soup.find("code", {"code": code})

        if not section_tag:
            continue

        # The section's parent holds the actual text
        parent = section_tag.find_parent("section")
        if not parent:
            continue

        # Extract all text from this section
        text = parent.get_text(separator=" ", strip=True)

        # Skip very short or empty sections
        if len(text) < 50:
            continue

        # Chunk text if very long (ChromaDB works best under 1000 chars)
        text_chunks = chunk_text(text, max_chars=900, overlap=100)

        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "id":           f"{set_id}_{code}_{i}",
                "text":         chunk,
                "drug_name":    drug_name,
                "section":      section_name,
                "ingredients":  ", ".join(ingredients[:5]),
                "manufacturer": manufacturer,
                "set_id":       set_id,
                "source":       "FDA DailyMed",
            })

    # ── If no structured sections found, use raw text ──────────
    if not chunks:
        raw = soup.get_text(separator=" ", strip=True)[:2000]
        if raw:
            chunks.append({
                "id":           f"{set_id}_raw_0",
                "text":         raw,
                "drug_name":    drug_name,
                "section":      "GENERAL",
                "ingredients":  ", ".join(ingredients[:5]),
                "manufacturer": manufacturer,
                "set_id":       set_id,
                "source":       "FDA DailyMed",
            })

    return chunks


def chunk_text(text: str, max_chars: int = 900, overlap: int = 100) -> list[str]:
    """
    Splits long text into overlapping chunks.
    Overlap ensures context isn't lost at chunk boundaries.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start  = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + max_chars // 2:
                end = last_period + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return [c for c in chunks if len(c) > 30]


# ── Step 4: Store chunks in ChromaDB ─────────────────────────
def store_chunks(collection, chunks: list[dict]):
    """Batch upsert chunks into ChromaDB."""
    if not chunks:
        return

    ids        = [c["id"]   for c in chunks]
    documents  = [c["text"] for c in chunks]
    metadatas  = [{
        "drug_name":    c["drug_name"],
        "section":      c["section"],
        "ingredients":  c["ingredients"],
        "manufacturer": c["manufacturer"],
        "set_id":       c["set_id"],
        "source":       c["source"],
    } for c in chunks]

    # Upsert = insert if new, update if exists
    collection.upsert(
        ids       = ids,
        documents = documents,
        metadatas = metadatas,
    )


# ── Step 5: Save raw XML locally (for debugging) ──────────────
def save_xml_locally(xml_text: str, set_id: str):
    os.makedirs(FDA_DATA_PATH, exist_ok=True)
    path = os.path.join(FDA_DATA_PATH, f"{set_id}.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(xml_text)


# ── Main ingestion function ───────────────────────────────────
def run_fda_ingestion(limit: int = 500, save_xml: bool = False):
    """
    Full pipeline:
    1. Fetch list of drugs from DailyMed
    2. For each drug: download XML label
    3. Parse XML → extract sections
    4. Store in ChromaDB
    """
    print("=" * 60)
    print("FDA DailyMed Ingestion")
    print("=" * 60)

    collection = get_chroma_collection()
    existing   = collection.count()
    print(f"ChromaDB already has {existing} chunks.")

    drugs = fetch_drug_list(limit=limit)
    print(f"\nProcessing {len(drugs)} drugs...\n")

    success  = 0
    skipped  = 0
    failed   = 0
    total_chunks = 0

    for drug in tqdm(drugs):
        set_id = drug.get("setid", "")
        if not set_id:
            skipped += 1
            continue

        # Skip if already in ChromaDB (resume-safe)
        try:
            existing_check = collection.get(
                where={"set_id": set_id},
                limit=1
            )
            if existing_check["ids"]:
                skipped += 1
                continue
        except:
            pass

        # Download XML
        xml_text = fetch_drug_label_xml(set_id)
        if not xml_text:
            failed += 1
            continue

        if save_xml:
            save_xml_locally(xml_text, set_id)

        # Parse and store
        chunks = parse_drug_label(xml_text, set_id)
        if chunks:
            store_chunks(collection, chunks)
            total_chunks += len(chunks)
            success += 1

        # Polite delay
        time.sleep(0.2)

    print("\n" + "=" * 60)
    print(f"Ingestion complete!")
    print(f"  Drugs processed : {success}")
    print(f"  Skipped (exist) : {skipped}")
    print(f"  Failed          : {failed}")
    print(f"  Total chunks    : {total_chunks}")
    print(f"  ChromaDB size   : {collection.count()} chunks")
    print("=" * 60)