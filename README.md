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

## MCP Tool

```python
search_recipes(
    query="quick chicken pasta",       # required — natural language, English or Norwegian
    diet="vegetarian",                 # optional: vegetarian, vegan, gluten-free, dairy-free
    difficulty="easy",                 # optional: easy, medium, hard
    max_minutes=30,                    # optional: maximum total time in minutes
    servings=4,                        # optional: included in query context, not used for filtering
    limit=5                            # optional: number of results (default 5)
)
# Returns semantically ranked recipes with ingredients, instructions, nutrition, and ratings
# Norwegian and English queries are both supported natively — no translation step
```

### Example response
```json
[
  {
    "rerank_score": 7.96,
    "title": "quick and easy chicken pasta salad",
    "description": "great use for left-over chicken.",
    "total_time": 25,
    "difficulty": "medium",
    "diet": [],
    "main_ingredient": "chicken",
    "ingredients": ["cooked chicken", "pasta shells", "tomatoes", "italian dressing"],
    "instructions": ["combine ingredients", "pour dressing", "chill 1 hour"],
    "nutrition": {"calories": 424, "protein_g": 26, "carbs_g": 38.5, "fat_g": 19.5},
    "rating": 4.8,
    "rating_count": 5
  }
]
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
