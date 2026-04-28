# Food Recipe MCP

<!-- mcp-name: io.github.AIDataNordic/food-recipe-mcp -->

Semantic search over 50,000+ food recipes — built for AI agents and LLMs. Two-stage hybrid retrieval (dense + sparse BM25, fused via RRF) with cross-encoder reranking. Supports natural language queries in Norwegian and English.

**Live endpoint:** `https://recipes.aidatanorge.no/mcp`  
**Transport:** `streamable-http`

---

## Connect

Add to your MCP client config:

```json
{
  "mcpServers": {
    "food-recipe": {
      "type": "streamable-http",
      "url": "https://recipes.aidatanorge.no/mcp"
    }
  }
}
```

Or with Claude Code:
```bash
claude mcp add --transport http food-recipe https://recipes.aidatanorge.no/mcp
```

---

## MCP Tools

### `search_recipes`

Semantic search over 50,000+ recipes from Food.com with hybrid retrieval and reranking.

```python
search_recipes(
    query="quick Italian pasta for weeknight dinner",
    diet="vegetarian",      # vegetarian | vegan | gluten-free | dairy-free | low-carb | keto | paleo
    max_minutes=30,         # maximum total cooking time in minutes
    difficulty="easy",      # easy | medium | hard
    limit=5                 # default 5, max 20
)
# Returns: rerank_score, rrf_score, title, description, total_time, difficulty,
#          diet, main_ingredient, servings, ingredients, instructions, nutrition,
#          rating, rating_count, source, recipe_id
```

**Query examples:**
- `"Swedish meatballs with gravy"`
- `"healthy high-protein chicken bowl"`
- `"easy chocolate cake for beginners"`
- `"traditional Norwegian kjøttkaker"`
- `"hurtig pasta med kylling"`

**Search pipeline:** Dense embedding (`intfloat/e5-large-v2`, 1024d) + sparse BM25, fused via Reciprocal Rank Fusion (RRF), reranked by `mmarco-mMiniLMv2-L12-H384-v1`.

### `ping`

```python
ping(name="world")
# Returns: "Hello world! Recipe MCP server is running."
```

---

## Data

- **Source:** Food.com (~50,000 recipes)
- **Coverage:** Wide range of cuisines, meal types, and cooking styles
- **Nutritional data:** calories, fat, protein, carbohydrates, sodium, fiber, sugar per serving
- **Ratings:** user rating + rating count per recipe
- **Languages:** English and Norwegian supported natively in queries

---

## Architecture

```
Food.com recipes → Python ingest → Qdrant (recipe_data_v2 collection)
                                         ↓
                              Hybrid search (dense e5-large-v2 + sparse BM25)
                                         ↓
                              RRF fusion + cross-encoder reranking
                                         ↓
                              FastMCP 3.2 → MCP clients / AI agents
```

---

## Technical Stack

- **Embeddings:** `intfloat/e5-large-v2` (1024d dense) + `Qdrant/bm25` (sparse)
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- **Vector DB:** Qdrant (self-hosted)
- **Server:** FastMCP 3.2 over HTTP
- **Infrastructure:** Ubuntu Server 24 LTS, Cloudflare Tunnel

---

## License

MIT
