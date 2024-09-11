from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any
from langgraph.graph import END

def gather_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant that gathers requirements for research queries."),
        ("human", "Please provide a research query."),
        ("ai", "Certainly! I'd be happy to help you formulate a research query. Could you please provide me with some information about the topic you'd like to research?"),
        ("human", "{input}"),
        ("ai", "Based on your input, I'll formulate a structured research query. Here's what I've come up with:\n\n{output}")
    ])

    llm = ChatOpenAI(model_name="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()

    last_message = messages[-1].content if messages else ""
    result = chain.invoke({"input": last_message})

    return {
        "messages": messages + [result],
        "structured_query": result,
        "next": "tavily_search"
    }

def end_convo(state: Dict[str, Any]) -> Dict[str, Any]:
    return {"next": END}
