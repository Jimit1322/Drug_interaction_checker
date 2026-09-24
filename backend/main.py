"""
FastAPI backend — serves the RAG pipeline as REST API.
Frontend will call these endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.rag.pipeline import run_query

app = FastAPI(
    title       = "Drug Interaction Checker API",
    description = "AI-powered drug interaction checker using RAG + Gemini",
    version     = "1.0.0"
)

# Allow frontend (Next.js on port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response models ─────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    patient_context: str = ""   # optional: age, conditions, other meds


class Source(BaseModel):
    type:    str
    name:    str
    url:     str  = ""
    section: str  = ""
    pmid:    str  = ""
    year:    str  = ""
    icon:    str  = ""


class QueryResponse(BaseModel):
    answer:       str
    sources:      list[Source]
    drugs_found:  list[str]
    doc_count:    int
    severity:     str          # extracted from answer


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status":  "running",
        "message": "Drug Interaction Checker API",
        "docs":    "/docs"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query_drug_interaction(request: QueryRequest):
    """
    Main endpoint — takes a natural language question,
    runs RAG pipeline, returns answer with sources.
    
    Example request:
    {
        "question": "Can I take Warfarin and Ibuprofen together?",
        "patient_context": "65 year old with diabetes"
    }
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > 500:
        raise HTTPException(status_code=400, detail="Question too long (max 500 chars)")

    # Add patient context to question if provided
    full_question = request.question
    if request.patient_context.strip():
        full_question += f"\n\nPatient context: {request.patient_context}"

    try:
        result = run_query(full_question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extract severity from answer text
    severity = extract_severity(result["answer"])

    return QueryResponse(
        answer      = result["answer"],
        sources     = [Source(**s) for s in result["sources"]],
        drugs_found = result["drugs_found"],
        doc_count   = result["doc_count"],
        severity    = severity,
    )


# backend/main.py — replace the /drugs/search endpoint
@app.get("/drugs/search")
async def search_drugs(q: str):
    if not q or len(q) < 2:
        return {"drugs": []}
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        from backend.config import CHROMA_DB_PATH, FDA_COLLECTION

        client     = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ef         = embedding_functions.SentenceTransformerEmbeddingFunction(
                         model_name="all-MiniLM-L6-v2")
        collection = client.get_collection(FDA_COLLECTION, embedding_function=ef)

        # Get all metadata without where filter
        results = collection.get(
            limit   = 500,
            include = ["metadatas"]
        )

        drug_names = set()
        q_lower    = q.lower()
        for meta in results["metadatas"]:
            name = meta.get("drug_name", "")
            # Filter: must start with query, reasonable length, no company names
            if (name.lower().startswith(q_lower)
                    and len(name) < 40
                    and "LLC" not in name
                    and "Inc" not in name
                    and "Corp" not in name
                    and "Pharma" not in name):
                drug_names.add(name.title())

        return {"drugs": sorted(list(drug_names))[:10]}

    except Exception as e:
        return {"drugs": [], "error": str(e)}
@app.get("/stats")
async def get_stats():
    """Returns knowledge base statistics."""
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        from backend.config import (CHROMA_DB_PATH, FDA_COLLECTION,
                                    PUBMED_COLLECTION)

        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        ef     = embedding_functions.SentenceTransformerEmbeddingFunction(
                     model_name="all-MiniLM-L6-v2")

        fda_col    = client.get_collection(FDA_COLLECTION,    embedding_function=ef)
        pubmed_col = client.get_collection(PUBMED_COLLECTION, embedding_function=ef)

        return {
            "fda_chunks":    fda_col.count(),
            "pubmed_chunks": pubmed_col.count(),
            "total_chunks":  fda_col.count() + pubmed_col.count(),
            "status":        "ready"
        }
    except Exception as e:
        return {"error": str(e), "status": "error"}


# ── Helper ────────────────────────────────────────────────────
def extract_severity(answer_text: str) -> str:
    """Extracts severity level from Gemini's response."""
    text = answer_text.upper()
    if "SEVERITY: SEVERE"   in text: return "SEVERE"
    if "SEVERITY: MODERATE" in text: return "MODERATE"
    if "SEVERITY: MILD"     in text: return "MILD"
    return "UNKNOWN"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)