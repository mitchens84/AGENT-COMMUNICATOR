import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from langgraph.graph import END
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

def gather_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Entering gather_requirements with state: {state}")
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that gathers requirements for research queries. Generate a concise query of no more than 400 characters."),
        ("human", "Please provide a research query."),
        ("ai", "Certainly! I'd be happy to help you formulate a concise research query. Could you please provide me with some information about the topic you'd like to research?"),
        ("human", "{input}"),
        ("ai", "Based on your input, I'll formulate a structured research query of no more than 400 characters. Here's what I've come up with:\n\n")
    ])

    llm = ChatOpenAI(model_name="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    last_message = messages[-1] if messages else ""
    if isinstance(last_message, (HumanMessage, AIMessage)):
        last_message_content = last_message.content
    else:
        last_message_content = str(last_message)

    logger.debug(f"Last message content: {last_message_content}")

    result = chain.invoke({"input": last_message_content})
    logger.debug(f"Chain result: {result}")

    return {
        "messages": messages + [AIMessage(content="Refined query: " + result)],
        "structured_query": result,
        "next": "tavily_search"
    }

def end_convo(state: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Entering end_convo with state: {state}")
    return {
        "messages": state.get("messages", []),
        "structured_query": state.get("structured_query", ""),
        "search_results": state.get("search_results", []),
        "next": END
    }
