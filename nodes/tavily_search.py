import logging
import requests
from typing import Dict, Any
from langchain_core.messages import AIMessage
import os

logger = logging.getLogger(__name__)

def perform_tavily_search(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Entering perform_tavily_search with state: {state}")
    structured_query = state.get("structured_query", "")
    messages = state.get("messages", [])

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("TAVILY_API_KEY is not set")
        return {
            "messages": messages + [AIMessage(content="Search results: Error: Tavily API key is not set.")],
            "search_results": [],
            "structured_query": structured_query,
            "next": "end_convo"
        }

    try:
        # Truncate the query if it's too long
        truncated_query = structured_query[:400] if len(structured_query) > 400 else structured_query
        logger.debug(f"Sending query to Tavily (truncated to 400 chars): {truncated_query}")

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "api_key": api_key,
            "query": truncated_query,
            "search_depth": "advanced",
            "max_results": 5
        }

        response = requests.post("https://api.tavily.com/search", json=payload, headers=headers)
        response.raise_for_status()

        search_results = response.json()
        logger.debug(f"Tavily search results: {search_results}")

        # Formulate a response based on the search results
        response_text = "Here are some relevant findings based on your query:\n\n"
        for result in search_results.get("results", []):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            response_text += f"- {title}: {url}\n"

        if len(structured_query) > 400:
            response_text += "\n(Note: The original query was truncated to 400 characters for the search.)"

    except requests.RequestException as e:
        logger.error(f"Error in Tavily search: {e}")
        error_message = f"An error occurred during the search: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_message += f"\nResponse: {e.response.text}"
        search_results = []
        response_text = error_message

    logger.debug(f"Generated response: {response_text}")

    return {
        "messages": messages + [AIMessage(content="Search results: " + response_text)],
        "search_results": search_results,
        "structured_query": structured_query,
        "next": "end_convo"
    }
