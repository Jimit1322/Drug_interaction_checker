# test_rag.py — run from project root
from backend.rag.pipeline import run_query

# Test 1 — classic dangerous combination
result = run_query("Can I take Warfarin and Ibuprofen together?")
print("\n" + "="*60)
print("ANSWER:")
print(result["answer"])
print("\nSOURCES USED:")
for s in result["sources"]:
    print(f"  {s['icon']} [{s['type']}] {s['name']}")
print(f"\nDrugs detected : {result['drugs_found']}")
print(f"Docs retrieved : {result['doc_count']}")