"""
MCP client — called by the RAG pipeline when local search
returns too few results.
Decides whether to call FDA or PubMed MCP server.
"""

import subprocess
import json
from backend.rag.retriever import retrieve


def needs_live_fetch(docs: list[dict], min_docs: int = 3) -> bool:
    """
    Returns True if we don't have enough local docs
    and need to fetch live data via MCP.
    """
    return len(docs) < min_docs


def fetch_via_mcp_fda(drug_name: str) -> str:
    """
    Calls the FDA MCP server as a subprocess.
    Returns the text response.
    """
    request = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params":  {
            "name":      "search_fda_drug",
            "arguments": {"drug_name": drug_name}
        }
    }

    try:
        result = subprocess.run(
            ["python", "-m", "backend.mcp.fda_server"],
            input   = json.dumps(request),
            capture_output = True,
            text    = True,
            timeout = 60
        )
        response = json.loads(result.stdout)
        return response.get("result", {}).get("content", [{}])[0].get("text", "")
    except Exception as e:
        return f"MCP FDA fetch failed: {str(e)}"


def fetch_via_mcp_pubmed(drug1: str, drug2: str = "") -> str:
    """Calls PubMed MCP server for live paper fetch."""
    request = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "tools/call",
        "params":  {
            "name":      "search_pubmed_interactions",
            "arguments": {
                "drug1":       drug1,
                "drug2":       drug2,
                "max_results": 10
            }
        }
    }

    try:
        result = subprocess.run(
            ["python", "-m", "backend.mcp.pubmed_server"],
            input   = json.dumps(request),
            capture_output = True,
            text    = True,
            timeout = 60
        )
        response = json.loads(result.stdout)
        return response.get("result", {}).get("content", [{}])[0].get("text", "")
    except Exception as e:
        return f"MCP PubMed fetch failed: {str(e)}"