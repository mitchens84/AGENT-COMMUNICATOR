import warnings
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")

import os
import traceback
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from nodes.requirements_gatherer import end_convo, gather_requirements
from nodes.tavily_search import perform_tavily_search
from utils.telegram_bot import TelegramBot
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'NODES/.env'))

def test_env_variables():
    required_vars = [
        "REQUIREMENTS_GATHERER_BOT_TOKEN",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error("The following required environment variables are missing:")
        for var in missing_vars:
            logger.error(f"- {var}")
        return False
    else:
        logger.info("All required environment variables are set.")
        return True

# LangSmith setup
client = Client()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

class State(TypedDict):
    messages: Annotated[List[BaseMessage], "The conversation history"]
    structured_query: Annotated[str, "The structured research query"]
    search_results: Annotated[list[dict], "The search results from Tavily"]

# Create the graph
graph = StateGraph(State)

# Add nodes
graph.add_node("requirements_gatherer", gather_requirements)
graph.add_node("tavily_search", perform_tavily_search)
graph.add_node("end_convo", end_convo)

# Add edges
graph.add_edge("requirements_gatherer", "tavily_search")
graph.add_edge("tavily_search", "end_convo")
graph.add_edge("end_convo", END)

# Add start edge
graph.set_entry_point("requirements_gatherer")

# Set up checkpointing
memory = MemorySaver()

# Compile the graph
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    if test_env_variables():
        try:
            bot_token = os.getenv("REQUIREMENTS_GATHERER_BOT_TOKEN")
            if not bot_token:
                raise ValueError("REQUIREMENTS_GATHERER_BOT_TOKEN is not set")
            bot = TelegramBot(bot_token, app)
            bot.run()
        except Exception as e:
            error_message = f"An error occurred: {e}\n\nTraceback:\n{traceback.format_exc()}"
            logger.error(error_message)
            client.create_run(
                project_name=os.getenv("LANGCHAIN_PROJECT", "default"),
                name="Error Log",
                inputs={"error": error_message},
                run_type="error_log",
                error=error_message
            )
    else:
        logger.error("Please set all required environment variables before running the bot.")
