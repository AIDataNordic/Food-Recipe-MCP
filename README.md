> [!WARNING]
> 🚧 **Server under maintenance** — The hosted endpoint (`mcp.aidatanorge.no/recipes-mcp`) is temporarily unavailable. The server can still be run locally by following the [Quickstart](#quickstart) instructions. We'll update this notice when the service is back online.

# AIDataNordic — Food Recipe MCP
A production-grade semantic search server for food recipes — built for AI agents using the Model Context Protocol (MCP). Search across 53,000+ recipes with hybrid dense + sparse retrieval and cross-encoder reranking.

[![smithery badge](https://smithery.ai/badge/kontakt-qy0g/Food-Recipe-MCP)](https://smithery.ai/servers/kontakt-qy0g/Food-Recipe-MCP)

---

## What This Is
A self-hosted MCP server exposing a recipe dataset through semantic search. Agents can query by natural language, filter by diet, difficulty, time, and servings — and get back ranked, structured recipe data including ingredients, instructions, and nutrition.

Designed for autonomous machine-to-machine consumption via FastMCP 3.2 over HTTP.

---

## Architecture
```
Query (natural language)
        ↓
  Dense embedding          Sparse embedding
  (e5-large-v2)            (BM25 / fastembed)
        ↓                        ↓
       Qdrant — Hybrid Fusion (RRF)
                    ↓
          Cross-encoder reranking
          (mmarco-mMiniLMv2-L12-H384-v1)
                    ↓
          Structured JSON response
                    ↓
           MCP tool / AI agent
```

---

## Data Coverage

| Field | Details |
| --- | --- |
| Total recipes | 53,000+ |
| Fields | title, description, ingredients, instructions, nutrition, rating, difficulty, diet, total_time, servings |
| Diet tags | vegetarian, vegan, gluten-free, dairy-free |
| Difficulty | easy, medium, hard |
| Languages | English, Norwegian |

---

## Technical Stack

**Search**
* Dense embeddings: `intfloat/e5-large-v2` (1024 dim)
* Sparse embeddings: `Qdrant/bm25` via fastembed
* Fusion: Reciprocal Rank Fusion (RRF) in Qdrant
* Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

**Serving**
* FastMCP 3.2 over HTTP (`/mcp` endpoint)
* Compatible with Claude, LangChain, and any MCP-capable agent

**Infrastructure**
* Ubuntu Server 24 LTS, self-hosted
* Qdrant vector database (self-hosted)

---

## MCP Tools

### `search_filings`

Semantic search over Nordic company filings, press releases and macroeconomic summaries.

```python
search_filings(
    query="Nordea net interest margin outlook 2025",
    report_type="quarterly_report",  # annual_report | quarterly_report | press_release | macro_summary
    country="SE",                    # NO | SE | DK | FI
    ticker="NDA",                    # optional — filter by company ticker
    fiscal_year=2025,                # optional — filter by year
    sector="energy",                 # optional — seafood | energy | shipping
    limit=10                         # default 5, max 20
)
# Returns semantically ranked text chunks with rerank_score, hybrid_score, vector_score,
# company, ticker, country, fiscal_year, report_type, filing_date and full text.
```

---

### `get_company_info`

Look up a company in the official business registry.

```python
get_company_info(
    identifier="923609016",  # org/CVR/business ID
    country="NO"             # NO (Brønnøysund) | DK (CVR) | FI (PRH)
)
# Returns company name, status and registered address.
```

---

### `parse_pdf_to_text`

Download a PDF from a URL and extract all text, page by page.

```python
parse_pdf_to_text(
    pdf_url="https://example.com/annual_report_2024.pdf"
)
# Returns extracted text with page separators.
# Useful for reading report attachments not indexed in the main database.
```

---

### `ping`

Connectivity test.

```python
ping(name="world")
# Returns: "Hello world! Nordic MCP server is running."
```

---

## Connect

**Live endpoint:**
```
https://recipes.aidatanorge.no/mcp
```

### Claude Code (terminal)
```bash
claude mcp add food-recipes --transport http https://recipes.aidatanorge.no/mcp
```

### Claude Desktop
Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "food-recipes": {
      "url": "https://recipes.aidatanorge.no/mcp"
    }
  }
}
```

### FastMCP (Python)
```python
import fastmcp, asyncio

async def main():
    async with fastmcp.Client("https://recipes.aidatanorge.no/mcp") as client:
        result = await client.call_tool("search_recipes", {
            "query": "quick chicken pasta",
            "max_minutes": 30,
            "limit": 3
        })
        for recipe in result.structured_content["result"]:
            print(recipe["title"], "-", recipe["total_time"], "min")

asyncio.run(main())
```

### Via Smithery
Available on [Smithery](https://smithery.ai/servers/kontakt-qy0g/Food-Recipe-MCP) as an alternative connection method:
```bash
npx -y @smithery/cli@latest mcp add kontakt-qy0g/Food-Recipe-MCP
```

---

## Self-hosting

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
python recipe_mcp_server.py
```

Server starts at `http://localhost:8004/mcp`

---

## Live Demo

Try the search interface at [recipes.aidatanorge.no](https://recipes.aidatanorge.no)

---

## Files

| File | Description |
| --- | --- |
| `recipe_mcp_server.py` | FastMCP server with hybrid search |
| `recipe_ingest.py` | Ingest pipeline — food.com recipes |
| `tine_recipe_ingest.py` | Ingest pipeline — tine.no recipes (Norwegian) |
| `mcp_client.py` | Example Python client |
| `requirements.txt` | Python dependencies |

---
