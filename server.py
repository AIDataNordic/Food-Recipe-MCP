from fastapi import FastAPI
import uvicorn
import os
from fastmcp import FastMCP

app = FastAPI(title="Norsk MCP Server")

# Opprett MCP-server
mcp = FastMCP("norsk-offentlig-data-mcp")

@mcp.tool()
async def hello_mcp(name: str = "verden") -> str:
    """En enkel hilsen for å teste at MCP fungerer."""
    return f"Hei {name}! Dette er din norske MCP-server på mini-PC-en."

@mcp.tool()
async def get_company_info(orgnr: str) -> dict:
    """Henter grunnleggende info om et norsk selskap fra Brønnøysundregistrene.
    Bruk orgnr uten bindestrek, f.eks. 123456789."""
    import httpx
    url = f"https://data.brreg.no/enhetsregisteret/api/enheter/{orgnr}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return {
                "navn": data.get("navn"),
                "organisasjonsform": data.get("organisasjonsform", {}).get("beskrivelse"),
                "forretningsadresse": data.get("forretningsadresse", {}).get("adresse"),
                "status": data.get("status"),
                "oppstartsdato": data.get("registreringsdato"),
                "hjemmeside": data.get("hjemmeside")
            }
        except Exception as e:
            return {"error": str(e), "message": "Kunne ikke hente data. Sjekk orgnr."}

# Mount MCP på /mcp
app.mount("/mcp", mcp)

if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", 8001))
    print(f"Starter stabil MCP-server på http://0.0.0.0:{port}/mcp")
    print("Trykk Ctrl+C for å stoppe serveren.")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
