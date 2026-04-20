"""
Nordic MCP Server - fastmcp v3
Semantic search over Nordic company filings, press releases and macroeconomic
summaries, with hybrid dense+sparse retrieval and cross-encoder reranking.
"""

import os
import logging
import time
import json
import torch
import httpx
from datetime import datetime
from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue,
    Prefetch, FusionQuery, Fusion, SparseVector,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding
from fastapi.responses import HTMLResponse

# --- Configuration ---
COLLECTION_NAME  = "nordic_company_data"
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
RERANK_FETCH     = 20
RERANK_MODEL     = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print("Loading embedding model...")
_model = SentenceTransformer("intfloat/e5-large-v2", device="cpu")
_model.max_seq_length = 512
print("Embedding model loaded.")

print("Loading sparse model...")
_sparse_model = SparseTextEmbedding("Qdrant/bm25")
print("Sparse model loaded.")

print("Loading reranker...")
_reranker = CrossEncoder(RERANK_MODEL, device="cpu")
print("Reranker loaded.")

_qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# --- Logging ---
_log = logging.getLogger("mcp")
_log.setLevel(logging.INFO)
os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)
_fh = logging.FileHandler(os.path.expanduser("~/logs/mcp_server.log"))
_fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
_log.addHandler(_fh)

mcp = FastMCP("nordic-public-data-mcp")


@mcp.tool()
async def ping(name: str = "world") -> str:
    """Simple connectivity test. Returns a greeting to confirm the server is running."""
    _log.info(f'ping name="{name}"')
    return f"Hello {name}! Nordic MCP server is running."


@mcp.tool()
async def get_company_info(orgnr: str) -> dict:
    """Look up a Norwegian company in the Brønnøysund Register (Enhetsregisteret).

    Args:
        orgnr: Norwegian organisation number without hyphens, e.g. 923609016.

    Returns:
        Dict with company name, status and registered business address.
    """
    url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{orgnr}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return {
                "name":    data.get("navn"),
                "status":  data.get("status"),
                "address": data.get("forretningsadresse", {}).get("adresse"),
            }
        except Exception as e:
            return {"error": str(e)}


@mcp.tool()
async def search_filings(
    query: str,
    ticker: str = "",
    fiscal_year: int = 0,
    report_type: str = "",
    sector: str = "",
    country: str = "",
    limit: int = 5,
) -> list[dict]:
    """Search the Nordic financial database for company filings, press releases
    and macroeconomic summaries.

    The database contains:

    COMPANY FILINGS
      Annual and quarterly reports (IR PDFs):
        SalMar (SALM), Mowi (MOWI), Lerøy (LSG), Grieg Seafood (GSF),
        Austevoll (AUSS), Bakkafrost (BAKKA), Aker BP (AKRBP), Odfjell (ODF)

      SEC EDGAR filings (Form 20-F annual reports and 6-K current reports):
        Equinor (EQNR), Höegh Autoliners (HSHP), Okeanis Eco Tankers (ECO),
        BW LPG (BWLP), Flex LNG (FLNG), Hafnia (HAFN), Cadeler (CDLR),
        Scorpio Tankers (STNG), SFL Corporation (SFL), Golden Ocean (GOGL),
        Frontline (FRO), Golar LNG (GLNG), Nordic American Tankers (NAT),
        Atlas Corp (ATCO)

    PRESS RELEASES
      GlobeNewswire RSS (continuous, hourly updates Mon–Fri):
        Norwegian, Swedish, Danish and Finnish listed companies (NO/SE/DK/FI)

    MACROECONOMIC SUMMARIES
      Quarterly macro summaries covering key indicators per country:
        Norway (NO):  policy rate, FX rates, CPI, house prices, credit growth,
                      electricity price, salmon price, GDP components
        Sweden (SE):  policy rate, house price index, household credit
        Denmark (DK): policy rate, house price index, household loans,
                      electricity price
        Finland (FI): house price index, household debt-to-income ratio,
                      electricity price
      Use report_type='macro_summary' and country='NO'/'SE'/'DK'/'FI' to filter.
      Use fiscal_year and a quarter reference in your query, e.g.
      "Norwegian housing market Q1 2024".

    Args:
        query:       What you are looking for, e.g. 'salmon price Q3',
                     'fleet utilization', 'dividend policy',
                     'Norwegian housing market 2024 Q1',
                     'Swedish policy rate inflation 2023'
        ticker:      Optional — filter by company ticker, e.g. 'SALM', 'EQNR'
        fiscal_year: Optional — filter by year, e.g. 2024
        report_type: Optional — one of:
                         'annual_report'     – Nordic IR annual reports (PDF)
                         'quarterly_report'  – Quarterly/interim reports (PDF)
                         'annual_report_20f' – SEC Form 20-F
                         '6k'                – SEC Form 6-K
                         'press_release'     – GlobeNewswire press releases
                         'macro_summary'     – Quarterly macroeconomic summaries
        sector:      Optional — filter by sector:
                         'seafood'   – seafood companies
                         'energy'    – energy / oil & gas
                         'shipping'  – shipping companies
        country:     Optional — filter by country code: 'NO', 'SE', 'DK' or 'FI'
        limit:       Number of results after reranking (default 5, max 20)

    Returns:
        List of relevant text excerpts with metadata, reranked by relevance.
        Each result includes rerank_score, vector_score, company, ticker,
        country, fiscal_year, report_type, period and the full text chunk.
    """
    limit = min(limit, 20)
    _t0 = time.time()

    # e5-large-v2 requires "query:"-prefix at search time
    e5_query = f"query: {query}"

    with torch.no_grad():
        dense_vec = _model.encode(e5_query, normalize_embeddings=True).tolist()

    sparse_result = list(_sparse_model.embed([query]))[0]
    sparse_vec = SparseVector(
        indices=sparse_result.indices.tolist(),
        values=sparse_result.values.tolist(),
    )

    conditions = []
    if ticker:
        conditions.append(
            FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))
        )
    if fiscal_year:
        conditions.append(
            FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year))
        )
    if report_type:
        conditions.append(
            FieldCondition(key="report_type", match=MatchValue(value=report_type))
        )
    if sector:
        conditions.append(
            FieldCondition(key="sector", match=MatchValue(value=sector.lower()))
        )
    if country:
        conditions.append(
            FieldCondition(key="country", match=MatchValue(value=country.upper()))
        )

    query_filter = Filter(must=conditions) if conditions else None

    fetch_limit = max(RERANK_FETCH, limit * 4)
    try:
        results = _qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vec,   using="dense",  limit=fetch_limit),
                Prefetch(query=sparse_vec,  using="sparse", limit=fetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=query_filter,
            limit=fetch_limit,
            with_payload=True,
        )
    except Exception as e:
        _log.exception(f"Qdrant query failed: {e}")
        return []

    if not results.points:
        return []

    candidates = results.points
    pairs = [(query, p.payload.get("text", "")) for p in candidates]
    with torch.no_grad():
        rerank_scores = _reranker.predict(pairs)

    ranked = sorted(
        zip(rerank_scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    output = []
    for rerank_score, point in ranked[:limit]:
        p = point.payload
        output.append({
            "rerank_score":  round(float(rerank_score), 4),
            "vector_score":  round(point.score, 4),
            "company":       p.get("company_name"),
            "ticker":        p.get("ticker"),
            "sector":        p.get("sector"),
            "country":       p.get("country"),
            "fiscal_year":   p.get("fiscal_year"),
            "report_type":   p.get("report_type"),
            "period":        p.get("period") or p.get("period_ending"),
            "filing_date":   p.get("filing_date") or p.get("published_date"),
            "text":          p.get("text"),
            "chunk_index":   p.get("chunk_index"),
            "total_chunks":  p.get("total_chunks"),
        })

    elapsed = round(time.time() - _t0, 3)
    _log.info(f'search_filings query="{query}" ticker="{ticker}" report_type="{report_type}" country="{country}" results={len(output)} elapsed={elapsed}s')
    return output


@mcp.tool()
async def parse_pdf_to_text(pdf_url: str) -> str:
    """Download a PDF from a URL and extract all text as a single string, page by page.
    
    This is useful for agents that need to read report attachments, press releases,
    or any PDF content that is not directly searchable in the main database.
    
    Args:
        pdf_url: Direct URL to the PDF file (e.g. https://example.com/report.pdf)
    
    Returns:
        All text from the PDF with page separators, or an error message.
    """
    import aiohttp
    import fitz  # PyMuPDF
    
    _log.info(f'parse_pdf_to_text url="{pdf_url}"')
    pdf_document = None
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(pdf_url) as response:
                if response.status != 200:
                    error_msg = f"Download failed: HTTP {response.status}"
                    _log.error(f'parse_pdf_to_text {error_msg}')
                    return error_msg
                
                pdf_bytes = await response.read()
        
        if b"%PDF" not in pdf_bytes[:10]:
            error_msg = "URL did not return a PDF (got HTML or other content)"
            _log.error(f'parse_pdf_to_text {error_msg}')
            return error_msg
        
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        all_text = []
        num_pages = len(pdf_document)
        
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            text = page.get_text()
            if text.strip():
                all_text.append(f"--- Page {page_num + 1} ---\n{text}")
            else:
                all_text.append(f"--- Page {page_num + 1} (no extractable text) ---")
        
        if not all_text:
            return "PDF contains no extractable text (may be a scanned image PDF)."
        
        result = "\n\n".join(all_text)
        _log.info(f'parse_pdf_to_text success url="{pdf_url}" pages={num_pages} chars={len(result)}')
        return result
    
    except aiohttp.ClientError as e:
        error_msg = f"Network error downloading PDF: {str(e)}"
        _log.error(f'parse_pdf_to_text {error_msg}')
        return error_msg
    except Exception as e:
        error_msg = f"PDF parsing error: {str(e)}"
        _log.error(f'parse_pdf_to_text {error_msg}')
        return error_msg
    finally:
        if pdf_document is not None:
            pdf_document.close()


# --- DEMO ENDEPUNKT (nettleser-demo) ---
DEMO_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Nordisk Finanssøk - Demo</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        input, button { padding: 10px; font-size: 16px; }
        input { width: 70%; }
        button { cursor: pointer; background: #0066cc; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0052a3; }
        .result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #f9f9f9; }
        .result strong { color: #0066cc; }
        .loading { color: #666; font-style: italic; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>🔍 Nordisk Finanssøk</h1>
    <p>Søk i 130 000+ nordiske finansdokumenter, pressemeldinger og makrodata</p>
    
    <input type="text" id="query" placeholder="F.eks. 'Equinor dividend' eller 'norsk boligpris Q3 2024'" style="width: 70%">
    <button onclick="search()">Søk</button>
    
    <div id="results"></div>

    <script>
        const MCP_URL = window.location.origin + '/mcp';
        
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading">🔍 Søker etter "' + query + '"...</div>';
            
            try {
                const sessionRes = await fetch(MCP_URL, {
                    method: 'GET',
                    headers: { 'Accept': 'application/json, text/event-stream' }
                });
                const sessionId = sessionRes.headers.get('mcp-session-id');
                
                if (!sessionId) throw new Error('Kunne ikke opprette session');
                
                await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({
                        jsonrpc: "2.0", id: 1, method: "initialize",
                        params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "web-demo", version: "1.0" } }
                    })
                });
                
                await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" })
                });
                
                const searchRes = await fetch(MCP_URL, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json, text/event-stream',
                        'mcp-session-id': sessionId
                    },
                    body: JSON.stringify({
                        jsonrpc: "2.0", id: 2, method: "tools/call",
                        params: { name: "search_filings", arguments: { query: query, limit: 5 } }
                    })
                });
                
                const text = await searchRes.text();
                const lines = text.split('\\n');
                let data = null;
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        data = JSON.parse(line.substring(6));
                        break;
                    }
                }
                
                if (data?.error) {
                    throw new Error(data.error.message || 'Ukjent feil');
                }
                
                if (data?.result?.content) {
                    const results = JSON.parse(data.result.content[0].text);
                    displayResults(results);
                } else {
                    resultsDiv.innerHTML = '<div class="error">Ingen resultater funnet</div>';
                }
            } catch (error) {
                resultsDiv.innerHTML = '<div class="error">❌ Feil: ' + error.message + '</div>';
                console.error(error);
            }
        }
        
        function displayResults(results) {
            const container = document.getElementById('results');
            if (!results.length) {
                container.innerHTML = '<div class="error">Ingen resultater</div>';
                return;
            }
            
            container.innerHTML = results.map(r => `
                <div class="result">
                    <strong>${escapeHtml(r.company || 'Ukjent')}</strong> 
                    ${r.ticker ? '(' + r.ticker + ')' : ''} 
                    ${r.report_type ? '- ' + r.report_type.replace('_', ' ') : ''}<br>
                    <small>📊 Relevans: ${r.rerank_score.toFixed(2)} | 📅 ${r.fiscal_year || 'Ukjent år'}${r.country ? ' | 🌍 ' + r.country : ''}</small>
                    <p>${escapeHtml(r.text.substring(0, 400))}${r.text.length > 400 ? '...' : ''}</p>
                </div>
            `).join('');
        }
        
        function escapeHtml(text) {
            if (!text) return '';
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') search();
        });
    </script>
</body>
</html>'''


@mcp.custom_route("/demo", methods=["GET"])
async def demo_endpoint(request):
    """Demo side for nettleser"""
    return HTMLResponse(content=DEMO_HTML)


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", 8003))
    print(f"→ Starting MCP server at http://0.0.0.0:{port}/mcp")
    print(f"→ Demo available at http://0.0.0.0:{port}/demo")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
