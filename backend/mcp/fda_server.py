"""
MCP Server for live FDA DailyMed fetching.
When a drug is not found in ChromaDB, this server
fetches it live from FDA and stores it for future queries.
"""

import time
import asyncio
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from bs4 import BeautifulSoup
from backend.ingestion.fda_loader import (
    parse_drug_label, store_chunks, get_chroma_collection
)

app = Server("fda-drug-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name        = "search_fda_drug",
            description = "Search FDA DailyMed for a specific drug label by name. "
                          "Use this when the drug is not found in local knowledge base.",
            inputSchema = {
                "type":       "object",
                "properties": {
                    "drug_name": {
                        "type":        "string",
                        "description": "Name of the drug to search for (e.g. 'warfarin', 'metformin')"
                    }
                },
                "required": ["drug_name"]
            }
        ),
        Tool(
            name        = "fetch_drug_interactions",
            description = "Fetch the drug interactions section specifically for a drug from FDA.",
            inputSchema = {
                "type":       "object",
                "properties": {
                    "drug_name": {
                        "type":        "string",
                        "description": "Drug name to fetch interactions for"
                    }
                },
                "required": ["drug_name"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_fda_drug":
        return await search_fda_drug(arguments["drug_name"])
    elif name == "fetch_drug_interactions":
        return await fetch_drug_interactions(arguments["drug_name"])
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def search_fda_drug(drug_name: str) -> list[TextContent]:
    """
    Searches DailyMed for a drug and stores results in ChromaDB.
    Returns summary of what was found.
    """
    print(f"[MCP FDA] Searching for: {drug_name}")

    url    = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    params = {"drug_name": drug_name, "pagesize": 5}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r    = await client.get(url, params=params)
            data = r.json()
            spls = data.get("data", [])
        except Exception as e:
            return [TextContent(type="text",
                                text=f"FDA search failed: {str(e)}")]

    if not spls:
        return [TextContent(type="text",
                            text=f"No FDA labels found for '{drug_name}'.")]

    # Fetch and store top 3 results
    collection   = get_chroma_collection()
    stored_count = 0
    summaries    = []

    for spl in spls[:3]:
        set_id = spl.get("setid", "")
        if not set_id:
            continue

        xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                xml_r    = await client.get(xml_url)
                xml_text = xml_r.text
            except:
                continue

        chunks = parse_drug_label(xml_text, set_id)
        if chunks:
            store_chunks(collection, chunks)
            stored_count += len(chunks)
            summaries.append(
                f"• {chunks[0]['drug_name']} "
                f"({len(chunks)} sections stored)"
            )

        await asyncio.sleep(0.3)

    result = (f"Found and stored FDA data for '{drug_name}':\n"
              + "\n".join(summaries)
              + f"\nTotal chunks added: {stored_count}")

    return [TextContent(type="text", text=result)]


async def fetch_drug_interactions(drug_name: str) -> list[TextContent]:
    """
    Specifically fetches just the drug interactions section.
    Faster than full label fetch.
    """
    print(f"[MCP FDA] Fetching interactions for: {drug_name}")

    # Search for the drug
    url    = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json"
    params = {"drug_name": drug_name, "pagesize": 1}

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r    = await client.get(url, params=params)
            data = r.json()
            spls = data.get("data", [])
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    if not spls:
        return [TextContent(type="text",
                            text=f"No data found for '{drug_name}'")]

    set_id  = spls[0].get("setid", "")
    xml_url = f"https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{set_id}.xml"

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            xml_r    = await client.get(xml_url)
            xml_text = xml_r.text
        except Exception as e:
            return [TextContent(type="text", text=f"Fetch error: {str(e)}")]

    # Parse just the interactions section
    soup    = BeautifulSoup(xml_text, "lxml-xml")
    section = soup.find("code", {"code": "34073-7"})  # DRUG INTERACTIONS code

    if not section:
        return [TextContent(type="text",
                            text=f"No drug interactions section found for '{drug_name}'")]

    parent = section.find_parent("section")
    text   = parent.get_text(separator=" ", strip=True)[:2000] if parent else ""

    return [TextContent(
        type = "text",
        text = f"FDA Drug Interactions for {drug_name}:\n\n{text}"
    )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream,
                      app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())