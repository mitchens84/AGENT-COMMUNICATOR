from tavily import TavilyClient
from typing import Dict, Any
from langchain.tools import StructuredTool

def perform_tavily_search(query: str) -> list[Dict[str, Any]]:
    client = TavilyClient()
    try:
        search_results = client.search(query=query, search_depth="advanced", max_results=5)
        return search_results
    except Exception as e:
        print(f"Error performing Tavily search: {e}")
        return []

tavily_tool = StructuredTool.from_function(
    func=perform_tavily_search,
    name="tavily_search",
    description="Searches the web using Tavily API"
)
