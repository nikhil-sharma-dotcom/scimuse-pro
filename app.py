import os
from dotenv import load_dotenv

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import operator
import re
from typing import TypedDict, List, Annotated, Optional
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import faiss
from pypdf import PdfReader
import networkx as nx
import matplotlib.pyplot as plt

import streamlit as st
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langgraph.graph import StateGraph, END


load_dotenv()
st.set_page_config(page_title="SciMuse Pro: Autonomous Lab", layout="wide")
st.sidebar.title("API Configuration")

GEMINI_KEY = os.getenv("GEMINI_API_KEY") or st.sidebar.text_input(
    "Gemini API Key", type="password"
)

if not GEMINI_KEY:
    import streamlit as st
    st.warning("Please provide your Gemini API key in the .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_KEY,
    temperature=0.3
)

emb_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_KEY
)



class ResearchState(TypedDict):
    topic: str
    grounding_data: str
    hypothesis: str
    analysis_code: str
    analysis_output: str
    visual_data: Optional[bytes]
    review_feedback: Annotated[List[str], operator.add]
    iteration_count: int
    is_novel: Annotated[bool, operator.or_]
    citations: List[str]
    retrieved_context: str
    novelty_delta: str
    semantic_novelty: float
    review_score: int
    reflection_notes: Annotated[List[str], operator.add]
    falsification_report: str
    failure_modes: List[str]
    





def extract_text_from_files(files):
    texts = []
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    continue
        elif file.name.endswith(".txt"):
            texts.append(file.read().decode("utf-8", errors="ignore"))
    return texts


def chunk_text_with_source(text, source, chunk_size=600, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append({"text": text[start:end], "source": source})
        start += chunk_size - overlap
    return chunks


@st.cache_resource
def build_cached_vector_store(chunks):
    embeddings = [emb_model.embed_query(c["text"]) for c in chunks]
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks


def retrieve_with_citations(query, index, chunks, k=4):
    qvec = np.array([emb_model.embed_query(query)]).astype("float32")
    _, ids = index.search(qvec, k)

    texts, cites = [], set()
    for i in ids[0]:
        if i == -1:
            continue
        texts.append(chunks[i]["text"])
        cites.add(chunks[i]["source"])

    return "\n".join(texts), list(cites)
def compute_semantic_novelty(hypothesis: str, context: str) -> float:
    if not context.strip():
        return 1.0  # maximally novel if no prior context

    h = np.array(emb_model.embed_query(hypothesis))
    c = np.array(emb_model.embed_query(context))
    return float(np.linalg.norm(h - c))

# =========================================================
# 4. AGENT NODES
# =========================================================

def grounding_node(state: ResearchState):
    st.write("🔍 Grounding via arXiv...")
    try:
        encoded = urllib.parse.quote(state["topic"])
        url = f"http://export.arxiv.org/api/query?search_query=all:{encoded}&max_results=3"

        req = urllib.request.Request(
            url, headers={"User-Agent": "SciMuse-Local/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            xml_data = r.read().decode("utf-8")

        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        results = []
        for e in root.findall("atom:entry", ns):
            t = e.find("atom:title", ns).text.strip()
            s = e.find("atom:summary", ns).text.strip()
            results.append(f"{t}\n{s}")

        if not results:
            return {"grounding_data": "No relevant arXiv papers found for this topic."}
        return {"grounding_data": "\n---\n".join(results)}

    except Exception as e:
        return {"grounding_data": f"arXiv error: {e}"}


def scientist_node(state: ResearchState):
    st.write("🧠 Scientist reasoning...")

    retrieved_context = state.get("retrieved_context", "")
    citations = state.get("citations", [])
    
    prompt = f"""
GLOBAL SCIENTIFIC BASELINE (arXiv):
{state['grounding_data']}

USER-SELECTED DOCUMENTS (must stay consistent with these):
{retrieved_context or "No user documents provided."}

TASK:
Propose a novel, testable hypothesis that:
- Extends the arXiv baseline
- Is CONSISTENT with the uploaded documents
- Explicitly addresses reviewer feedback
- Does NOT contradict cited sources
- Is under 150 words

Previous reviewer feedback:
{'; '.join(state.get('review_feedback', [])) or "None"}
"""
    res = llm.invoke(prompt)

    return {
        "hypothesis": res.content,
        "retrieved_context": retrieved_context,
        "citations": citations
    }


def analyst_visualizer_node(state: ResearchState):
    st.write("💻 Analyst executing simulation...")

    code_prompt = f"""
Write Python code to simulate or test:

{state['hypothesis']}

Rules:
- numpy + matplotlib
- save plot as research_plot.png
- print numeric output
"""

    raw = llm.invoke(code_prompt).content
    clean = raw
    if "```" in raw:
        clean = re.sub(
    r"```(python)?(.*?)```",
    lambda m: m.group(2),
    raw,
    flags=re.S | re.I
).strip()


    clean = raw.strip()
    output, plot_bytes = "", None

    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.py"
            p.write_text(clean)

            r = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = r.stdout or r.stderr

            img = Path(td) / "research_plot.png"
            if img.exists():
                plot_bytes = img.read_bytes()

    except Exception as e:
        output = f"Execution failed: {e}"

    return {
        "analysis_code": clean,
        "analysis_output": output,
        "visual_data": plot_bytes
    }


def reviewer_node(state: ResearchState):
    st.write("⚖️ Reviewer evaluating novelty...")

    prompt = f"""
Hypothesis:
{state['hypothesis']}

Analysis:
{state['analysis_output']}

Give SCORE 1-10 and critique.
"""

    # 1️⃣ LLM peer review (subjective)
    res = llm.invoke(prompt).content

    m = re.search(r"score\s*[:\-]?\s*(\d+)", res, re.I)
    score = int(m.group(1)) if m else 0

    # 2️⃣ System-side semantic novelty (objective)
    semantic_novelty = compute_semantic_novelty(
        state["hypothesis"],
        state.get("retrieved_context", "")
    )

    # 3️⃣ Final decision (controller logic)
    is_novel = score >= 8 and semantic_novelty > 0.7

    return {
        "review_feedback": [res],
        "iteration_count": state["iteration_count"] + 1,
        "review_score": score,
        "semantic_novelty": semantic_novelty,
        "is_novel": is_novel
    }

def reflection_node(state: ResearchState):
    prompt = f"""
Previous feedback:
{state.get('review_feedback', [])}

Current hypothesis:
{state['hypothesis']}

Summarize:
- Improvements made
- Remaining weaknesses
"""
    res = llm.invoke(prompt)
    return {"reflection_notes": [res.content]}


def falsifier_node(state: ResearchState):
    st.write("🧪 Falsification: stress-testing hypothesis...")

    prompt = f"""
You are a skeptical peer reviewer.

Hypothesis:
{state['hypothesis']}

Analysis Results:
{state['analysis_output']}

Tasks:
1. List 3 strong counter-arguments or failure modes
2. Identify assumptions that may not hold
3. Propose at least one experiment that could falsify this hypothesis

Format STRICTLY as:
FAILURE_MODES:
- Simulation assumptions may oversimplify real systems
- Literature retrieval limited to uploaded scope
- Embedding-based novelty may miss conceptual overlap
FALSIFICATION_EXPERIMENT:
...
Be critical, precise, and scientific.
"""

    res = llm.invoke(prompt).content
    in_section = False
    failure_modes = []
    for line in res.splitlines():
        if line.strip().startswith("FAILURE_MODES"):
            in_section = True
            continue
        if line.strip().startswith("FALSIFICATION_EXPERIMENT"):
            break
        if in_section and line.strip().startswith("-"):
            failure_modes.append(line.strip()[1:].strip())
    
    updated_is_novel = state["is_novel"]  # start with previous novelty
    if len(failure_modes) >= 4:
        updated_is_novel = False

    return {
        "falsification_report": res,
        "failure_modes": failure_modes,
        "is_novel": updated_is_novel
    }


def novelty_delta_node(state: ResearchState):
    prompt = f"""
Compare hypothesis with literature.

Hypothesis:
{state['hypothesis']}

Citations:
{state['citations']}
"""
    res = llm.invoke(prompt)
    return {"novelty_delta": res.content}



def router(state):
    if state["is_novel"] or state["iteration_count"] >= 3:
        return "end"
    return "refine"


builder = StateGraph(ResearchState)
builder.add_node("grounding", grounding_node)
builder.add_node("scientist", scientist_node)
builder.add_node("analyst", analyst_visualizer_node)
builder.add_node("reviewer", reviewer_node)
builder.add_node("reflection", reflection_node)
builder.add_node("falsifier", falsifier_node)
builder.add_node("novelty_delta", novelty_delta_node)


builder.set_entry_point("grounding")
builder.add_edge("grounding", "scientist")
builder.add_edge("scientist", "analyst")
builder.add_edge("analyst", "reviewer")

builder.add_edge("reflection", "scientist")

builder.add_conditional_edges(
    "reviewer",
    router,
    {
        "refine": "reflection",
        "end": "novelty_delta"
    }
)
builder.add_edge("reviewer", "falsifier")
builder.add_edge("falsifier", "novelty_delta")

builder.add_edge("novelty_delta", END)
agent = builder.compile()




def build_knowledge_graph(text):
    prompt = f"""
Extract at most 10 factual relationships.

Rules:
- Do NOT invent entities
- Use only entities explicitly present
- Format strictly:
ENTITY1 -> RELATION -> ENTITY2

Text:
{text}
"""
    res = llm.invoke(prompt).content
    G = nx.DiGraph()

    for line in res.splitlines():
        parts = [p.strip() for p in line.split("->")]
        if len(parts) == 3:
            G.add_edge(parts[0], parts[2], label=parts[1])
    return G


def draw_kg(G):
    if not G.nodes:
        st.warning("Empty KG")
        return
    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, ax=ax)
    nx.draw_networkx_edge_labels(
        G, pos, nx.get_edge_attributes(G, "label"), ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)


def generate_markdown_report(state: ResearchState):
    return f"""
# SciMuse Research Report

## Topic
{state['topic']}

## Hypothesis
{state['hypothesis']}

## Novelty Delta
{state['novelty_delta']}

## Analysis Output
{state['analysis_output']}

## Citations
""" + "\n".join(f"- {c}" for c in state["citations"])




st.title("🔬 SciMuse Pro")

uploaded_files = st.file_uploader(
    "Upload PDFs / TXT", type=["pdf", "txt"], accept_multiple_files=True
)

topic = st.text_input(
    "Research Topic",
    "Carbon capture using metal-organic frameworks"
)

if st.button("🚀 Run"):
    retrieved_context, citations = "", []

    if uploaded_files:
        texts = extract_text_from_files(uploaded_files)
        chunks = []
        for t, f in zip(texts, uploaded_files):
            chunks.extend(chunk_text_with_source(t, f.name))

        if chunks:
            idx, store = build_cached_vector_store(chunks)
            retrieved_context, citations = retrieve_with_citations(
                topic, idx, store
            )

    init_state = {
        "topic": topic,
        "grounding_data": "",
        "hypothesis": "",
        "analysis_code": "",
        "analysis_output": "",
        "visual_data": None,
        "review_feedback": [],
        "iteration_count": 0,
        "is_novel": False,
        "citations": citations,
        "retrieved_context": retrieved_context,
        "novelty_delta": ""
    }

    final = agent.invoke(init_state)

    st.subheader("Hypothesis")
    st.info(final["hypothesis"])
    st.subheader("Iteration Reflections")
    for r in final["reflection_notes"]:
        st.write("•", r)
    
    st.subheader("⚠️ Failure Modes")
    if final.get("failure_modes"):
        for fm in final["failure_modes"]:
            st.write(f"- {fm}")
    else:
        st.info("No explicit failure modes identified.")
    st.subheader("🧪 Falsification Report")
    st.warning(final.get("falsification_report", "N/A"))


    st.subheader("Novelty Delta")
    st.info(final["novelty_delta"])

    st.subheader("Knowledge Graph")
    draw_kg(build_knowledge_graph(final["hypothesis"]))

    st.download_button(
        "Download Report",
        generate_markdown_report(final),
        "scimuse_report.md"
    )
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Iterations", final["iteration_count"])
    with col2:
        st.metric("Novel?", "Yes" if final["is_novel"] else "No")
    st.metric("Semantic Novelty", f"{final['semantic_novelty']:.3f}")
    st.metric("Reviewer Score", final["review_score"])
