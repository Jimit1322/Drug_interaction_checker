"""
MCP Server for live PubMed fetching.
Fetches latest research papers for drugs not in local DB.
"""

import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from xml.etree import ElementTree as ET
from backend.ingestion.pubmed_loader import (
    parse_pubmed_xml, store_papers, get_chroma_collection
)

app    = Server("pubmed-research-server")
BASE   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name        = "search_pubmed_interactions",
            description = "Search PubMed for research papers about drug interactions. "
                          "Use when more evidence is needed beyond FDA labels.",
            inputSchema = {
                "type":       "object",
                "properties": {
                    "drug1": {
                        "type":        "string",
                        "description": "First drug name"
                    },
                    "drug2": {
                        "type":        "string",
                        "description": "Second drug name (optional)",
                        "default":     ""
                    },
                    "max_results": {
                        "type":        "integer",
                        "description": "Max papers to fetch (default 10)",
                        "default":     10
                    }
                },
                "required": ["drug1"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_pubmed_interactions":
        return await search_pubmed_interactions(
            arguments["drug1"],
            arguments.get("drug2", ""),
            arguments.get("max_results", 10)
        )
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_pubmed_interactions(drug1: str, drug2: str,
                                     max_results: int) -> list[TextContent]:
    """Searches PubMed and stores results in ChromaDB."""

    query = f"{drug1} {drug2} drug interaction".strip()
    print(f"[MCP PubMed] Searching: '{query}'")

    # Step 1: Search for PMIDs
    search_params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            r     = await client.get(f"{BASE}/esearch.fcgi",
                                     params=search_params)
            data  = r.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            return [TextContent(type="text",
                                text=f"PubMed search failed: {str(e)}")]

    if not pmids:
        return [TextContent(type="text",
                            text=f"No PubMed papers found for '{query}'")]

    # Step 2: Fetch abstracts
    fetch_params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r      = await client.get(f"{BASE}/efetch.fcgi",
                                      params=fetch_params)
            papers = parse_pubmed_xml(r.text)
        except Exception as e:
            return [TextContent(type="text",
                                text=f"PubMed fetch failed: {str(e)}")]

    # Step 3: Store in ChromaDB
    collection = get_chroma_collection()
    store_papers(collection, papers)

    # Step 4: Return summary
    summaries = []
    for p in papers[:5]:
        summaries.append(
            f"• {p['title'][:80]}...\n"
            f"  {p['authors']} ({p['year']}) — PMID {p['pmid']}"
        )

    result = (f"Found {len(papers)} papers for '{query}':\n\n"
              + "\n".join(summaries))

    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream,
                      app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())