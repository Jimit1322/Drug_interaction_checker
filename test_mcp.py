# test_mcp.py
from backend.rag.pipeline import run_query

# This drug is likely NOT in our 500-drug local DB
# MCP should kick in and fetch it live
result = run_query(
    "Can I take Apixaban and Clarithromycin together?"
)

print("\n" + "="*60)
print("ANSWER:")
print(result["answer"])
print(f"\nDrugs  : {result['drugs_found']}")
print(f"Sources: {result['doc_count']} docs used")