# AIDataNordic — Food Recipe MCP

A production-grade semantic search server for food recipes — built for AI agents using the Model Context Protocol (MCP). Search across 50,000+ recipes with hybrid dense + sparse retrieval and cross-encoder reranking.

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

| Field            | Details                                      |
|------------------|----------------------------------------------|
| Total recipes    | 50,000+                                      |
| Source           | food.com and others                          |
| Fields           | title, description, ingredients, instructions, nutrition, rating, difficulty, diet, total_time, servings |
| Diet tags        | vegetarian, vegan, gluten-free, dairy-free   |
| Difficulty       | easy, medium, hard                           |

---

## Technical Stack

**Search**
- Dense embeddings: `intfloat/e5-large-v2` (1024 dim)
- Sparse embeddings: `Qdrant/bm25` via fastembed
- Fusion: Reciprocal Rank Fusion (RRF) in Qdrant
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

**Serving**
- FastMCP 3.2 over HTTP (`/mcp` endpoint)
- Compatible with Claude, LangChain, and any MCP-capable agent

**Infrastructure**
- Ubuntu Server 24 LTS, self-hosted
- Qdrant vector database (self-hosted)

---

## MCP Tool

```python
search_recipes(
    query="quick chicken pasta",       # required — natural language
    diet="vegetarian",                 # optional: vegetarian, vegan, gluten-free, dairy-free
    difficulty="easy",                 # optional: easy, medium, hard
    max_minutes=30,                    # optional: maximum total time in minutes
    servings=4,                        # optional: number of servings
    limit=5                            # optional: number of results (default 5)
)
# Returns semantically ranked recipes with ingredients, instructions, nutrition, and ratings
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

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the server

```bash
python recipe_mcp_server.py
```

Server starts at `http://localhost:8004/mcp`

### 3. Connect with FastMCP client

```python
import fastmcp, asyncio

async def main():
    async with fastmcp.Client("http://localhost:8004/mcp") as client:
        result = await client.call_tool("search_recipes", {
            "query": "quick chicken pasta",
            "max_minutes": 30,
            "limit": 3
        })
        for recipe in result.structured_content["result"]:
            print(recipe["title"], "-", recipe["total_time"], "min")

asyncio.run(main())
```

### 4. Connect with Claude Desktop

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

---

## Live Demo

Try the search interface at [recipes.aidatanorge.no](https://recipes.aidatanorge.no)

---

## Files

| File | Description |
|------|-------------|
| `recipe_mcp_server.py` | FastMCP server with hybrid search |
| `mcp_client.py` | Example Python client |
| `requirements.txt` | Python dependencies |

---

*Built and operated as part of [AIDataNordic](https://github.com/AIDataNordic) — self-hosted AI data infrastructure for autonomous agents.*
