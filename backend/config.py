import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
FDA_DATA_PATH  = os.getenv("FDA_DATA_PATH",  "./data/fda")
PUBMED_DATA_PATH = os.getenv("PUBMED_DATA_PATH", "./data/pubmed")

# ChromaDB collection names
FDA_COLLECTION    = "fda_drug_labels"
PUBMED_COLLECTION = "pubmed_abstracts"

# RAG settings
TOP_K = 8

# Gemini model
GEMINI_MODEL = "gemini-3.6-flash"     # free tier model