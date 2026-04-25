# ⚡ Eightfold AI: Enterprise Research Agent

[![Deployed on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/codeAIDrafter/Eightfold-Research-Agent)
[![Built with Chainlit](https://img.shields.io/badge/Built_with-Chainlit-F18D77?logo=chainlit)](https://github.com/Chainlit/chainlit)
[![Powered by Groq](https://img.shields.io/badge/Powered_by-Groq-black)](https://groq.com/)

**Live Demo:** [Launch the Agent on Hugging Face](https://huggingface.co/spaces/codeAIDrafter/Eightfold-Research-Agent)

## 📌 Overview
The **Enterprise Research Agent** is an autonomous AI assistant designed to move beyond basic data-entry and actively synthesize strategic Account Plans for enterprise sales teams. 

Rather than simply summarizing Wikipedia, this agent leverages an internal **Semantic Router** and a **Sales Strategist Prompt Engine** to search the web, analyze real-time market signals, and generate highly targeted Value Propositions that connect a company's recent news to the user's specific sales goals.

## ✨ Key Features
* 🎙️ **Voice-to-Text Integration:** Native support for real-time microphone streaming and transcription via Groq's Whisper-large-v3.
* 🧠 **Agentic Synthesis:** Dynamically deduces a company's Strategic Priorities and Pain Points based on live news and competitor data.
* 🔀 **Semantic Context Routing:** Intelligently manages conversation state, knowing exactly when to pivot to a new company and when to hold context.
* 🛡️ **Plausibility Guardrails:** Autonomously pauses execution to ask the user clarifying questions if it encounters conflicting web data.
* 💾 **In-Memory Markdown Export:** Instantly download a beautifully formatted `.md` Executive Summary of your Account Plan.
* 📝 **Targeted Plan Modifications:** Users can dynamically update specific sections of a generated account plan through natural conversation (e.g., "Change the target executives to just the CTO") without resetting the entire document.
---

## 🏗️ Architecture & Engineering Decisions

Building a robust, stateful LLM agent requires solving several complex Natural Language Processing (NLP) and pipeline challenges. Here is how the core architecture addresses them:

### 1. Semantic Routing & The "Comparative Entity" Trap
A naive chatbot will wipe its memory if a user asks, *"How does Microsoft compare to Apple?"* because it detects the word "Apple". 
* **The Solution:** The `LLMEngine` acts as an initial control layer. It utilizes a strict classification prompt to differentiate between a **Context Switch** (abandoning the current plan) and a **Comparative Inquiry** (researching a competitor to strengthen the current plan). The underlying state machine is only modified if a true `NEW_COMPANY` intent is triggered.

### 2. Graceful API Fallbacks
Network requests fail, and APIs hit rate limits. 
* **The Solution:** The `ResearchTool` implements a graceful fallback mechanism. It attempts to pull Verified FAQs, Organic Results, and News from **SerpApi**. If a quota error occurs, it silently catches the exception and falls back to **DuckDuckGo's native search library (`DDGS`)**. By ensuring both APIs return an identical `List[Dict]` schema, the LLM parser never crashes, and the user never notices the interruption.

### 3. Agentic Synthesis vs. Data Retrieval
A standard RAG pipeline extracts facts. An Agent synthesizes strategy.
* **The Solution:** The data extraction pipeline is governed by a set of strict rules. If the user goal is "selling cloud security" and the news search reveals "Expansion into European markets," the LLM is explicitly instructed to synthesize a derived pain point (e.g., *"GDPR compliance latency"*). Furthermore, the LLM is permitted to utilize baseline enterprise knowledge to fill in `market_revenue` and `competitors` if search results are sparse, preventing the "Data Starvation" trap.

### 4. Plausibility Guardrails & Conflict Resolution
LLMs are prone to hallucinating when fed contradictory web data.
* **The Solution:** The extraction schema includes an `open_questions` array. If the LLM detects conflicting data (e.g., two different revenue reports), it flags the conflict, pushes a question to the state queue, and places the agent in a `WAIT_FOR_USER` state. The agent refuses to finalize the Account Plan until the human operator resolves the discrepancy.

---

## 👥 Handling Diverse User Personas
The system is explicitly engineered to maintain conversational quality across various interaction styles:
* **The Efficient User:** Can bypass chatty greetings by instantly providing a company and goal (e.g., "Netflix. Goal: Sell video compression."). The agent instantly routes to `NEW_COMPANY` and generates the plan.
* **The Confused User:** If a user types "I don't know what to do", the Semantic Router classifies the intent as `CONFUSED_USER` and gracefully guides them on how to construct a valid prompt.
* **The Chatty / Off-Topic User:** If a user asks a `GENERAL_QUESTION` (e.g., "What is the weather?"), the router intercepts the prompt and politely redirects the conversation back to the active Account Plan.
* **The Edge Case User:** By strictly extracting entities and validating data against the `AccountPlanState` schema, the agent prevents malformed inputs or impossible requests from crashing the extraction pipeline.

---

## ⚖️ Engineering Trade-offs

* **Stateless Backend vs. Databases:** To ensure a clean, fast prototype for demonstration purposes, I opted for a stateless backend architecture rather than integrating a PostgreSQL or Redis cache. Session data is held in RAM during the conversation. To persist data, users utilize the "Export" command to generate a local `.md` file. In a V2 production environment, state would be serialized to Redis via Chainlit's Data Layer.
* **In-Memory File Generation:** The file download and audio processing pipelines bypass the server's hard drive entirely. By utilizing `io.BytesIO()` and explicit MIME-type declarations, audio chunks are stitched and Markdown files are encoded purely in RAM. This prevents I/O bottlenecks and race conditions common in concurrent cloud deployments.

---

## 💻 Tech Stack
* **Frontend UI:** Chainlit (React-based Python framework)
* **LLM Engine:** Groq (Llama-3-70b-8192)
* **Voice-to-Text:** Groq (Whisper-large-v3)
* **Search Tools:** SerpApi, DuckDuckGo Search
* **Deployment:** Docker, Hugging Face Spaces

---

## 🚀 Local Installation & Quick Start

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/Eightfold-Research-Agent.git](https://github.com/YOUR_USERNAME/Eightfold-Research-Agent.git)
cd Eightfold-Research-Agent
```

**2. Create a virtual environment and install dependencies**

```Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Configure Environment Variables**

Create a .env file in the root directory and add your API keys:

```Code
GROQ_API_KEY=your_groq_api_key_here
SERPAPI_KEY=your_serpapi_key_here
```

**4. Run the Application**

```Bash
chainlit run app.py -w
```

The application will automatically open in your browser at http://localhost:8000.