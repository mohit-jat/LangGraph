import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# Load .env
load_dotenv()


# OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# State
class State(TypedDict):
    topic: str
    information: str
    summary: str


# Node 1
def research(state: State):

    response = llm.invoke(
        f"Give me simple and important information about {state['topic']}"
    )

    return {
        "information": response.content
    }


# Node 2
def summarize(state: State):

    response = llm.invoke(
        f"Summarize this information in simple points:\n\n"
        f"{state['information']}"
    )

    return {
        "summary": response.content
    }


# Create graph
graph = StateGraph(State)


# Add nodes
graph.add_node("research", research)
graph.add_node("summarize", summarize)


# Sequential flow
graph.add_edge(START, "research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", END)


# Compile
app = graph.compile()


# Run workflow
result = app.invoke({
    "topic": "Artificial Intelligence",
    "information": "",
    "summary": ""
})


print("\n===== FINAL ANSWER =====")
print(result["summary"])