# Drug Interaction Checker

An AI-powered clinical assistant that answers drug interaction queries using LLM + RAG + MCP — grounded in 10,000+ FDA drug labels and 600+ PubMed abstracts with inline citations.

> For educational purposes only. Always consult a pharmacist or doctor.

---

## Features

- Natural language queries — "Can I take Montair LC with Paracetamol?"
- Brand name resolution — Montair LC -> montelukast + levocetirizine
- Severity classification — MILD / MODERATE / SEVERE / NO KNOWN INTERACTION
- Cited answers — every response links to FDA labels and PubMed papers
- Live data fetch via MCP — unknown drugs fetched from FDA/PubMed on demand
- Model fallback chain — Gemini 3.8 -> 3.5 -> 3.1-lite, no 503 errors
- Response caching — same query answered instantly without hitting API

---

## Architecture

```
Frontend (Next.js :3000)
    |
    | HTTP POST /query
    v
Backend (FastAPI :8000)
    |
    |-- Brand Resolver (RxNorm + local dict)
    |-- RAG Pipeline (pipeline.py)
    |-- Retriever (ChromaDB vector search)
    |
    |-- [if docs missing] MCP Servers
    |       |-- FDA MCP Server (live FDA fetch)
    |       |-- PubMed MCP Server (live paper fetch)
    |
    |-- ChromaDB
    |       |-- fda_drug_labels     (10,275 chunks)
    |       |-- pubmed_abstracts    (566 chunks)
    |
    v
Gemini API (LLM Layer)
    gemini-3.8-flash -> gemini-3.5-flash -> gemini-3.1-flash-lite
    + In-memory response cache
```

Full diagram: architecture.drawio (open at app.diagrams.net)

---

## Tech Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| Frontend    | Next.js 14, Tailwind CSS, TypeScript|
| Backend     | FastAPI, Python 3.13                |
| Vector DB   | ChromaDB                            |
| Embeddings  | all-MiniLM-L6-v2 (local, free)     |
| LLM         | Google Gemini API                   |
| MCP         | Custom FDA + PubMed MCP servers     |
| Data        | FDA DailyMed API, PubMed eUtils     |
| Brand Res.  | RxNorm API + Indian brands dict     |

---

## Project Structure

```
drug-interaction-checker/
|
|-- backend/
|   |-- main.py                   # FastAPI app + endpoints
|   |-- config.py                 # All config variables
|   |-- ingestion/
|   |   |-- fda_loader.py         # Download + parse FDA XML
|   |   |-- pubmed_loader.py      # Fetch PubMed abstracts
|   |   +-- run_ingestion.py      # Master ingestion script
|   |-- rag/
|   |   |-- pipeline.py           # Core RAG orchestration
|   |   |-- retriever.py          # ChromaDB vector search
|   |   +-- brand_resolver.py     # Brand to generic mapping
|   |-- mcp/
|   |   |-- fda_server.py         # MCP: live FDA fetch
|   |   |-- pubmed_server.py      # MCP: live PubMed fetch
|   |   +-- mcp_client.py         # Direct ingestion client
|   +-- llm/
|       +-- gemini_client.py      # Gemini + fallback + cache
|
|-- frontend/
|   +-- app/
|       +-- page.tsx              # Chat UI
|
|-- chroma_db/                    # Vector store (auto-created)
|-- architecture.drawio           # System diagram
|-- .env                          # API keys (never commit)
|-- .env.example
|-- requirements.txt
+-- README.md
```

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Google AI Studio API key (free at aistudio.google.com)

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/drug-interaction-checker
cd drug-interaction-checker

# Backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend && npm install && cd ..
```

### 2. Environment Variables

```bash
cp .env.example .env
```

Edit .env:
```
GEMINI_API_KEY=AIzaSy_your_key_here
GROQ_API_KEY=gsk_your_key_here
CHROMA_DB_PATH=./chroma_db
FDA_DATA_PATH=./data/fda
PUBMED_DATA_PATH=./data/pubmed
```

### 3. Build Knowledge Base

```bash
# Quick test (5 min)
PYTHONPATH=. python -m backend.ingestion.run_ingestion --fda-limit 50 --pubmed-limit 20

# Full ingestion — run overnight
PYTHONPATH=. python -m backend.ingestion.run_ingestion --fda-limit 10000 --pubmed-limit 100
```

### 4. Run

```bash
# Terminal 1 — Backend
PYTHONPATH=. uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open http://localhost:3000

---

## API Endpoints

| Method | Endpoint              | Description               |
|--------|-----------------------|---------------------------|
| GET    | /health               | Health check              |
| GET    | /stats                | Knowledge base stats      |
| GET    | /drugs/search?q=name  | Drug name autocomplete    |
| POST   | /query                | Drug interaction query    |

### Query Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can I take Warfarin and Ibuprofen together?",
    "patient_context": "65 year old with hypertension"
  }'
```

---

## Supported Brand Names

| Brand          | Generic                           |
|----------------|-----------------------------------|
| Montair LC     | montelukast + levocetirizine      |
| Crocin / Dolo  | paracetamol                       |
| Combiflam      | ibuprofen + paracetamol           |
| Augmentin      | amoxicillin + clavulanate         |
| Ecosprin       | aspirin                           |
| Glycomet       | metformin                         |
| Pan / Pantocid | pantoprazole                      |
| Tylenol        | acetaminophen                     |
| Advil / Motrin | ibuprofen                         |

Add more in backend/rag/brand_resolver.py -> INDIAN_BRANDS dict.

---

## How It Works

```
User: "Can I take Montair LC with Paracetamol?"
    1. Brand Resolution
       Montair LC -> [montelukast, levocetirizine]
       Paracetamol -> [acetaminophen]

    2. ChromaDB Vector Search
       Top 8 chunks from FDA + PubMed

    3. Relevance Check
       Missing drugs? -> trigger MCP live fetch

    4. MCP Live Fetch (if needed)
       FDA: fetch montelukast label -> store in ChromaDB
       PubMed: fetch interaction papers

    5. Gemini Generation
       Send docs + question -> cited answer
       Fallback: 3.8 -> 3.5 -> 3.1-lite

    6. Response with severity + citations
```

---

## Tests

```bash
PYTHONPATH=. python test_db.py       # ChromaDB
PYTHONPATH=. python test_gemini.py   # Gemini API
PYTHONPATH=. python test_rag.py      # Full RAG pipeline
PYTHONPATH=. python test_mcp.py      # MCP servers
```

---

## Deployment

### Backend (Railway / Render)
```
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

### Frontend (Vercel)
```bash
cd frontend && vercel --prod
```

Update NEXT_PUBLIC_API_URL env var to deployed backend URL.

---

## Limitations

- Drug coverage depends on ingestion limit (500 drugs = ~3% of FDA database)
- Indian brand resolution uses local dictionary — RxNorm doesn't cover all Indian brands
- Gemini free tier has rate limits — handled by automatic fallback
- Not for clinical use — always verify with a licensed pharmacist

---

## References

- FDA DailyMed: https://dailymed.nlm.nih.gov
- PubMed eUtils: https://www.ncbi.nlm.nih.gov/home/develop/api/
- RxNorm API: https://rxnav.nlm.nih.gov
- ChromaDB: https://www.trychroma.com
- Google Gemini API: https://ai.google.dev

---

## License

MIT License

Built by Jimit Sankhesara — MNIT Jaipur