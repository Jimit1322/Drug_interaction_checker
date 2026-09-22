"""
Master script — runs both FDA and PubMed ingestion.
Run this once to build your knowledge base.
After this, ChromaDB is populated and ready for RAG queries.

Usage:
    python -m backend.ingestion.run_ingestion
    python -m backend.ingestion.run_ingestion --fda-limit 200 --pubmed-limit 30
"""

import argparse
from backend.ingestion.fda_loader    import run_fda_ingestion
from backend.ingestion.pubmed_loader import run_pubmed_ingestion


def main():
    parser = argparse.ArgumentParser(description="Build drug interaction knowledge base")
    parser.add_argument("--fda-limit",    type=int, default=500,
                        help="Number of FDA drug labels to ingest (default: 500)")
    parser.add_argument("--pubmed-limit", type=int, default=50,
                        help="PubMed papers per search term (default: 50)")
    parser.add_argument("--skip-fda",     action="store_true",
                        help="Skip FDA ingestion")
    parser.add_argument("--skip-pubmed",  action="store_true",
                        help="Skip PubMed ingestion")
    parser.add_argument("--save-xml",     action="store_true",
                        help="Save raw FDA XML files locally")
    args = parser.parse_args()

    print("\n🔬 Drug Interaction Checker — Knowledge Base Builder")
    print("=" * 60)

    if not args.skip_fda:
        run_fda_ingestion(
            limit    = args.fda_limit,
            save_xml = args.save_xml,
        )

    if not args.skip_pubmed:
        run_pubmed_ingestion(
            max_per_term = args.pubmed_limit,
        )

    print("\n✅ Knowledge base ready!")
    print("Next step: run Phase 2 (RAG pipeline)")


if __name__ == "__main__":
    main()