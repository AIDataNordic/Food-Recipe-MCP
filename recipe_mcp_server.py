"""
Recipe MCP Server - fastmcp
Semantic search over recipe data with hybrid search (dense + sparse) and cross-encoder reranking.
Uses e5-large-v2 embeddings (1024 dim) and BM25 sparse vectors.
Queries are automatically translated to English before search.
"""

import os
import logging
import torch
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    Prefetch, FusionQuery, Fusion, SparseVector
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from deep_translator import GoogleTranslator

# --- Logging ---
_log = logging.getLogger("recipe_mcp")
_log.setLevel(logging.INFO)
_fh = logging.FileHandler(os.path.expanduser("~/logs/recipe_mcp_server.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
_log.addHandler(_fh)

# --- Configuration ---
COLLECTION_NAME = "recipe_data_v2"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
RERANK_FETCH = 20
EMBED_MODEL = "intfloat/e5-large-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
SPARSE_MODEL = "Qdrant/bm25"

print("Loading dense embedding model (e5-large-v2)...")
_dense_model = SentenceTransformer(EMBED_MODEL, device="cpu")
_dense_model.max_seq_length = 512
print("Dense model loaded.")

print("Loading sparse embedding model (BM25)...")
_sparse_model = SparseTextEmbedding(SPARSE_MODEL)
print("Sparse model loaded.")

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL, device="cpu")
print("Reranker loaded.")

_qdrant = QdrantClient(host=QDRANT_HOST, port=6333)

mcp = FastMCP("recipe-mcp")


def _translate_to_english(text: str) -> str:
    """Translate query to English if needed. Returns original text on failure."""
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        if translated and translated != text:
            _log.info(f"translated query='{text}' → '{translated}'")
            return translated
        return text
    except Exception as e:
        _log.warning(f"Translation failed for query='{text}': {e}")
        return text


@mcp.tool()
async def ping(name: str = "world") -> str:
    """Simple connectivity test. Returns a greeting to confirm the server is running."""
    return f"Hello {name}! Recipe MCP server is running."


@mcp.tool()
async def search_recipes(
    query: str,
    diet: str = "",
    max_minutes: int = 0,
    difficulty: str = "",
    servings: int = 0,
    limit: int = 5,
) -> list[dict]:
    """Search a database of recipes using hybrid semantic search (dense + sparse) with reranking.

    The database contains ~50,000 recipes from Food.com covering a wide
    range of cuisines, meal types, and cooking styles. Recipes include
    nutritional information, difficulty ratings, and user ratings.

    Use natural language in the query to describe what you are looking
    for — cuisine, style, main ingredient, occasion, or mood all work
    well. Queries in any language are supported and will be automatically
    translated to English before search. Examples:
        'quick Italian pasta for weeknight dinner'
        'Swedish meatballs with gravy'
        'healthy high-protein chicken bowl'
        'easy chocolate cake for beginners'
        'something with salmon and lemon'
        'Indian curry chicken'
        'traditional Norwegian kjøttkaker'
        'hurtig pasta med kylling'
        'enkel sjokoladekake'

    Args:
        query:       What you are looking for — describe the dish, cuisine,
                     main ingredient, cooking style or mood freely.
                     Any language is supported.
        diet:        Optional — filter by dietary requirement:
                         'vegetarian', 'vegan', 'gluten-free',
                         'dairy-free', 'low-carb', 'keto', 'paleo'
        max_minutes: Optional — maximum total time in minutes, e.g. 30
        difficulty:  Optional — 'easy', 'medium' or 'hard'
        servings:    Optional — not used for filtering (servings vary),
                     but include in query for scaling context,
                     e.g. 'pasta dish for 6 people'
        limit:       Number of results to return after reranking
                     (default 5, max 20)

    Returns:
        List of recipes ranked by relevance. Each result includes
        rerank_score, rrf_score (hybrid fusion), title, total_time,
        difficulty, diet labels, ingredients, instructions, nutrition,
        rating, and source URL context.
    """
    import time
    t0 = time.time()
    limit = min(limit, 20)

    # Translate query to English before embedding
    original_query = query
    query = _translate_to_english(query)

    with torch.no_grad():
        # Dense vector with e5-large query prefix
        dense_vec = _dense_model.encode(f"query: {query}", normalize_embeddings=True).tolist()

        # Sparse vector (BM25)
        sparse_embedding = list(_sparse_model.embed([query]))[0]
        sparse_vec = SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist()
        )

    # Build filters
    conditions = []
    if diet:
        conditions.append(FieldCondition(key="diet", match=MatchAny(any=[diet.lower()])))
    if difficulty:
        conditions.append(FieldCondition(key="difficulty", match=MatchValue(value=difficulty.lower())))
    if max_minutes > 0:
        conditions.append(FieldCondition(key="total_time", range=Range(lte=max_minutes)))

    query_filter = Filter(must=conditions) if conditions else None

    fetch_limit = max(RERANK_FETCH, limit * 4)

    # Hybrid search with RRF fusion
    results = _qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=fetch_limit),
            Prefetch(query=sparse_vec, using="sparse", limit=fetch_limit),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        query_filter=query_filter,
        limit=fetch_limit,
        with_payload=True,
    )

    if not results.points:
        _log.info(f"search_recipes query='{original_query}' translated='{query}' results=0 elapsed={time.time()-t0:.2f}s")
        return []

    candidates = results.points
    pairs = [(query, p.payload.get("text", "")) for p in candidates]
    with torch.no_grad():
        rerank_scores = _reranker.predict(pairs)

    ranked = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)

    elapsed = time.time() - t0
    _log.info(
        f"search_recipes query='{original_query}' translated='{query}' "
        f"results={min(limit, len(ranked))} elapsed={elapsed:.2f}s"
    )

    output = []
    for rerank_score, point in ranked[:limit]:
        p = point.payload
        output.append({
            "rerank_score": round(float(rerank_score), 4),
            "rrf_score": round(point.score, 4),
            "title": p.get("title"),
            "description": p.get("description"),
            "total_time": p.get("total_time"),
            "difficulty": p.get("difficulty"),
            "diet": p.get("diet"),
            "main_ingredient": p.get("main_ingredient"),
            "unit_system": p.get("unit_system"),
            "servings": p.get("servings"),
            "ingredients": p.get("ingredients_raw"),
            "instructions": p.get("instructions"),
            "nutrition": p.get("nutrition"),
            "rating": p.get("rating"),
            "rating_count": p.get("rating_count"),
            "source": p.get("source"),
            "recipe_id": p.get("recipe_id"),
        })

    return output


# --- DEMO ENDEPUNKT ---
from starlette.responses import HTMLResponse

with open(os.path.join(os.path.dirname(__file__), "recipes_demo.html"), "r") as f:
    DEMO_HTML = f.read()

@mcp.custom_route("/", methods=["GET"])
async def recipes_demo_endpoint(request):
    """Demo side for oppskriftssøk"""
    return HTMLResponse(content=DEMO_HTML)

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        port = int(os.getenv("MCP_PORT", 8004))
        print(f"→ Demo available at http://0.0.0.0:{port}/")
        print(f"→ Starting Recipe MCP server at http://0.0.0.0:{port}/mcp")
        mcp.run(transport="http", host="0.0.0.0", port=port, path="/mcp")
