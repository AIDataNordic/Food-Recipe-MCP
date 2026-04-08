#!/usr/bin/env python3
"""
MCP-klient for recipe_mcp_server
Bruker FastMCP sin innebygde klient
"""
import asyncio
import fastmcp

SERVER_URL = "http://localhost:8004/mcp"

async def main():
    async with fastmcp.Client(SERVER_URL) as client:

        # List tilgjengelige verktøy
        tools = await client.list_tools()
        print("\n📦 Tilgjengelige verktøy:")
        for tool in tools:
            print(f"  - {tool.name}")

        # Ping-test
        print("\n🏓 Ping-test:")
        ping = await client.call_tool("ping", {"name": "verden"})
        print(f"  {ping.data}")

        # Søk etter oppskrifter
        print("\n🔍 Søker etter 'quick chicken pasta' (maks 30 min)...")
        result = await client.call_tool("search_recipes", {
            "query": "quick chicken pasta",
            "max_minutes": 30,
            "limit": 3
        })

        recipes = result.structured_content.get("result", [])
        print("\n📋 Resultater:")
        for recipe in recipes:
            print(f"\n  🍽  {recipe['title'].title()}")
            print(f"      ⏱  {recipe['total_time']} min  |  ⭐ {recipe['rating']}  |  {recipe['difficulty']}")
            print(f"      {recipe['description'][:80]}...")

if __name__ == "__main__":
    asyncio.run(main())
