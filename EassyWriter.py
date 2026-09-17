# Imports
import os
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient
from typing import TypedDict, List
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# 1. Streamlit configuration must be the very first Streamlit command
st.set_page_config(page_title="Essay Writer Agent", page_icon="📝", layout="wide")

# Initialize Tavily Client
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Define Model
model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0.69
)

# Define Agent State
class EassyWriterAgent(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int

class Queries(BaseModel):
    queries: List[str]

# Prompts
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays. Generate the best essay possible for the user's request and the initial outline. If the user provides critique, respond with a revised version of your previous attempts. Utilize all the information below as needed: \n\n------\n\n{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. Generate critique and recommendations for the user's submission. Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can be used when writing the following essay. Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can be used when making any requested revisions (as outlined below). Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""

JSON_QUERIES_INSTRUCTION = "\n\nRespond only with a valid JSON object of the form {\"queries\": [\"...\", \"...\"]}. Do not include any other text, explanation, or essay content."

# Define Nodes
def plan_node(state: EassyWriterAgent):
    messages = [
        SystemMessage(content=PLAN_PROMPT),
        HumanMessage(content=state["task"])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}

def research_plan_node(state: EassyWriterAgent):
    queries = model.with_structured_output(Queries, method="json_mode").invoke(
        [
            SystemMessage(content=RESEARCH_PLAN_PROMPT + JSON_QUERIES_INSTRUCTION),
            HumanMessage(content=f"Generate search queries to research background information for an essay on this topic: {state['task']}")
        ]
    )
    content = state.get('content', []) or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=1)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: EassyWriterAgent):
    content = "\n\n".join(state.get('content', []) or [])
    user_message = HumanMessage(
        content=f"{state['task']} \n\n Here is my plan: \n\n {state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: EassyWriterAgent):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state["draft"])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}

def research_critique_node(state: EassyWriterAgent):
    queries = model.with_structured_output(Queries, method="json_mode").invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT + JSON_QUERIES_INSTRUCTION),
            HumanMessage(content=state["critique"])
        ]
    )
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=1)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state: EassyWriterAgent):
    if state['revision_number'] > state['max_revisions']:
        return END
    return "reflect"

# Build the Graph
graph = StateGraph(EassyWriterAgent)

graph.add_node("plan", plan_node)
graph.add_node("research_plan", research_plan_node)
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_node("research_critique", research_critique_node)

graph.set_entry_point("plan")
graph.add_conditional_edges(
    'generate',
    should_continue,
    {END: END, "reflect": "reflect"}
)
graph.add_edge("plan", "research_plan")
graph.add_edge("research_plan", "generate")
graph.add_edge("reflect", "research_critique")
graph.add_edge("research_critique", "generate")

agent = graph.compile()

# ---------------------------------------------------------------------------
# Streamlit UI & Execution
# ---------------------------------------------------------------------------

st.title("📝 Essay Writer Agent")
st.caption("A LangGraph agent that plans → researches → drafts → reflects → revises an essay.")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "qwen/qwen3-32b"],
        index=0,
    )
    max_revisions = st.slider("Max revisions", min_value=0, max_value=5, value=2)
    st.divider()

task = st.text_area(
    "Essay topic / prompt",
    height=100,
    placeholder="e.g. What is the difference between LangChain and LangGraph?",
)

run = st.button("Write Essay", type="primary")

if run:
    full_state = {
        "task": task,
        "max_revisions": max_revisions,
        "revision_number": 1,
        "content": [],
    }

    node_labels = {
        "plan": "📋 Planning the essay outline",
        "research_plan": "🔍 Researching the outline",
        "generate": "✍️ Writing a draft",
        "reflect": "🧑‍🏫 Critiquing the draft",
        "research_critique": "🔍 Researching the critique",
    }

    status = st.status("Running the agent…", expanded=True)
    try:
        with status:
            for update in agent.stream(full_state, {"recursion_limit": 50}, stream_mode="updates"):
                node_name, partial = next(iter(update.items()))
                full_state.update(partial)
                st.write(node_labels.get(node_name, node_name) + " — done")
        status.update(label="Essay complete!", state="complete")
    except Exception as e:
        status.update(label="Failed", state="error")
        st.error(f"Agent run failed: {e}")
        st.stop()

    st.session_state["result"] = full_state

if "result" in st.session_state:
    result = st.session_state["result"]

    tab_essay, tab_plan, tab_critique, tab_sources = st.tabs(
        ["Final Essay", "Outline", "Latest Critique", "Research Sources"]
    )

    with tab_essay:
        st.subheader(f"Revision {result.get('revision_number', 1) - 1}")
        st.markdown(result.get("draft", "_No draft yet._"))
        st.download_button(
            "Download essay (.md)",
            data=result.get("draft", ""),
            file_name="essay.md",
            mime="text/markdown",
        )

    with tab_plan:
        st.markdown(result.get("plan", "_No outline yet._"))

    with tab_critique:
        st.markdown(result.get("critique", "_No critique generated (essay finished within revision limit)._"))

    with tab_sources:
        sources = result.get("content", [])
        if not sources:
            st.write("_No research content gathered._")
        for i, chunk in enumerate(sources, start=1):
            with st.expander(f"Source snippet {i}"):
                st.write(chunk)