import warnings
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langsmith import Client
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from nodes.requirements_gatherer import end_convo, gather_requirements
from nodes.tavily_search import tavily_tool
from utils.telegram_bot import TelegramBot

# Load .env from the parent directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'NODES/.env'))

def test_env_variables():
    required_vars = [
        "REQUIREMENTS_GATHERER_BOT_TOKEN",  # Updated this line
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "LANGSMITH_API_KEY",
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        print("The following required environment variables are missing:")
        for var in missing_vars:
            print(f"- {var}")
        return False
    else:
        print("All required environment variables are set.")
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
graph.add_node("tavily_search", ToolNode([tavily_tool]))
graph.add_node("end_convo", end_convo)

# Add edges
graph.add_edge("requirements_gatherer", "tavily_search")
graph.add_edge("tavily_search", "end_convo")

# Add start edge
graph.set_entry_point("requirements_gatherer")


# Set up checkpointing
memory = MemorySaver()

# Compile the graph
app = graph.compile(checkpointer=memory)

# Set up and run the Telegram bot
bot = TelegramBot(os.getenv("REQUIREMENTS_GATHERER_BOT_TOKEN"), app)

if __name__ == "__main__":
    if test_env_variables():
        try:
            bot_token = os.getenv("REQUIREMENTS_GATHERER_BOT_TOKEN")
            if not bot_token:
                raise ValueError("REQUIREMENTS_GATHERER_BOT_TOKEN is not set")
            bot = TelegramBot(bot_token, app)
            bot.run()
        except Exception as e:
            client.log_error(error=str(e), project=os.getenv("LANGCHAIN_PROJECT"))
            print(f"An error occurred: {e}")
    else:
        print("Please set all required environment variables before running the bot.")
