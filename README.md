# 🔬 SciMuse Pro — Autonomous Research Agent
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge.svg)](https://scimuse-pro-hbnqmfwynhswqu9yaja4om.streamlit.app/)

SciMuse Pro is an autonomous scientific research system that generates, validates, and stress-tests research hypotheses using multi-agent reasoning. Unlike standard RAG systems, it doesn't just retrieve—it iterates through a scientific loop to ensure novelty and technical rigor.

## 🚀 [Live Demo Link](https://scimuse-pro-hbnqmfwynhswqu9yaja4om.streamlit.app/)

---

## 🧠 System Architecture
SciMuse Pro uses **LangGraph** to orchestrate a team of specialized agents. This ensures the research process follows a logical, iterative flow rather than a linear retrieval.



### 🤖 The Agent Team:
* **Grounding Agent:** Fetches arXiv literature and processes user PDFs to establish a baseline.
* **Scientist Agent:** Generates testable, high-impact hypotheses based on the grounding data.
* **Analyst Agent:** Writes and executes Python simulation code in a local sandbox to test the hypothesis.
* **Reviewer Agent:** Scores novelty (1–10) and identifies semantic "deltas" vs. existing work.
* **Falsifier Agent:** Conducts adversarial reviews to find failure modes and stress-test the logic.

---

## 🛠️ Tech Stack
* **Core Model:** Google Gemini 1.5 Flash
* **Orchestration:** LangGraph & LangChain
* **Vector Database:** FAISS (for high-speed semantic search)
* **Embeddings:** Google Gemini-001
* **Visualizations:** NetworkX (Knowledge Graphs) & Matplotlib (Simulation plots)
* **Interface:** Streamlit

---

## 🧪 Key Capabilities
* **Autonomous Research Loop:** Iterative reflection that terminates only when a novelty threshold is met.
* **Local Code Execution:** Automatically generates simulation scripts, runs them in a subprocess, and captures output/plots.
* **Knowledge Graph Synthesis:** Extracts entity relationships to map out complex scientific domains visually.
* **Exportable Artifacts:** Generates full Markdown research reports including citations and falsification analysis.

---

## ⚙️ Quick Start (Local Setup)

1. **Clone & Enter:**
   ```bash
   git clone [https://github.com/nikhil-sharma-dotcom/scimuse-pro.git](https://github.com/nikhil-sharma-dotcom/scimuse-pro.git)
   cd scimuse-pro
Install Dependencies:

Bash

pip install -r requirements.txt
Set API Key: Create a .env file and add: GEMINI_API_KEY=your_key_here

Launch App:

Bash

streamlit run app.py
👤 Author
Valluri Naga Jishnu Nikhil Sharma B.Tech CSE | AI & Autonomous Systems Specialist LinkedIn | GitHub

Disclaimer: This tool is designed for research ideation and pre-proposal drafting. Always review auto-generated simulation code before use.
