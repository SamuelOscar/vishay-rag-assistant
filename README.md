# Vishay Intertechnology RAG Research Assistant

A conversational equity research assistant powered by **Retrieval-Augmented Generation (RAG)**.
Ask natural language questions about Vishay Intertechnology (NYSE: VSH) and get grounded, source-cited answers drawn directly from SEC filings, earnings reports, investor presentations, and press releases.

Built as the capstone project for the **Agentic AI Essentials Certification Program**.

---

## What It Does

This assistant lets you query a custom knowledge base of Vishay financial documents using plain English. Instead of searching through hundreds of pages of 10-K filings manually, you can ask:

- *"What was Vishay's gross margin in 2024?"*
- *"What risks did management highlight in the most recent 10-K?"*
- *"How is Vishay positioning itself in the automotive electronics market?"*
- *"What did the Q4 2025 earnings call reveal about inventory trends?"*

The assistant retrieves the most relevant document chunks, feeds them to an LLM with a custom equity research prompt, and returns a precise, cited answer — all without hallucinating data.

---

## System Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         main.py  (CLI)              │
│  - Conversation loop                │
│  - Input/output formatting          │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         qa_chain.py                 │
│  - ConversationalRetrievalChain     │
│  - Custom equity research prompt    │
│  - Session memory (last 5 turns)    │
└──────────┬──────────────────────────┘
           │                    │
           ▼                    ▼
┌─────────────────┐   ┌──────────────────┐
│  retriever.py   │   │  GPT-4o-mini     │
│  - FAISS index  │   │  (OpenAI LLM)    │
│  - Top-5 chunks │   │  temperature=0   │
└────────┬────────┘   └──────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│         ingest.py  (run once)       │
│  - Load PDFs + HTML files           │
│  - Clean & extract metadata         │
│  - Chunk (1000 tokens / 150 overlap)│
│  - Embed → FAISS index on disk      │
└─────────────────────────────────────┘
```

---

## Knowledge Base

The assistant is grounded in the following Vishay Intertechnology documents:

| Document Type | Years Covered |
|---|---|
| Annual Reports (Form 10-K) | 2018, 2020, 2021, 2022, 2023, 2024, 2025 |
| Proxy Statements | 2024, 2025 |
| Quarterly Earnings Releases | Q3 2025, Q4 2025 |
| Investor Day Presentation | 2024 |
| EDGAR SEC Filings | 2025–2026 |
| Press Releases | 2025–2026 |

---

## Project Structure

```
vishay-rag-assistant/
├── data/
│   └── raw/                     # All source documents (PDFs, HTML)
├── vectorstore/
│   └── faiss_index/             # Auto-generated after running ingest.py
├── src/
│   ├── ingest.py                # Step 1: Load, chunk, and embed documents
│   ├── retriever.py             # Step 2: Load FAISS index + build retriever
│   ├── qa_chain.py              # Step 3: Build QA chain with memory + prompt
│   └── main.py                  # Step 4: Interactive CLI
├── notebooks/
│   └── experimentation.ipynb   # Exploratory prototyping notebook
├── .env                         # Your API keys (never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/vishay-rag-assistant.git
cd vishay-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Your OpenAI API Key

Create a `.env` file in the root of the project:

```
OPENAI_API_KEY=sk-your-key-here
```

> You can get a key at https://platform.openai.com/api-keys

### 5. Ingest Documents (Run Once)

This reads all PDFs and HTML files in `data/raw/`, splits them into chunks, embeds them, and saves the FAISS index to disk.

```bash
python src/ingest.py
```

Expected output:
```
Loading documents...
Loaded 312 documents
Splitting into chunks...
Created 1847 chunks
Creating embeddings...
Saving FAISS index...
Ingestion complete!
```

> Note: This step costs a small amount in OpenAI API credits (embedding calls).

### 6. Run the Assistant

```bash
python src/main.py
```

---

## Example Session

```
╔══════════════════════════════════════════════════════════╗
║       Vishay Intertechnology  |  RAG Research Assistant  ║
╚══════════════════════════════════════════════════════════╝

✅ Vector store loaded successfully.

🔍 Your question: What was Vishay's revenue in fiscal year 2024?

💬 Answer:
   According to the 2024 Annual Report (Form 10-K), Vishay Intertechnology
   reported total net revenues of approximately $3.03 billion for fiscal
   year 2024, a decline from $3.39 billion in fiscal year 2023, reflecting
   weaker demand across most end markets and ongoing inventory corrections
   by customers.

📄 Sources used:
   • 2024 Form 10-K (Investor Relations).pdf
   • 2023 Vishay Intertechnology Inc. Form 10-K.pdf

─────────────────────────────────────────────────────────────

🔍 Your question: What segment drove the most revenue?

💬 Answer:
   Following up on the 2024 figures: the Resistors & Inductors segment
   was the largest contributor to Vishay's revenue, followed by
   Capacitors and Semiconductors (MOSFETs and Diodes)...

📄 Sources used:
   • 2024 Form 10-K (Investor Relations).pdf
```

---

## Design Decisions

**Why FAISS over Chroma?** FAISS is lightweight, runs fully locally (no server needed), and is well-suited for a fixed document set that doesn't change frequently.

**Why `temperature=0`?** Financial Q&A demands precision. A temperature of 0 makes the LLM deterministic and less likely to "improvise" numbers.

**Why session memory with a window of 5?** This lets you ask natural follow-up questions ("What about the prior year?", "And the margins?") without losing context, while keeping the prompt size manageable.

**Why a custom prompt?** The default LangChain QA prompt is generic. The custom equity research prompt instructs the model to cite document types and years, use financial terminology, and explicitly refuse to hallucinate.

---

## Optional Enhancements (Planned)

- [ ] Add a Streamlit web UI for a richer UX
- [ ] Expand metadata (segment tags, quarter labels) for filtered retrieval
- [ ] Add logging to track query history to a file
- [ ] Support multi-hop reasoning with LangGraph ReAct agent

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | FAISS (via `langchain-community`) |
| Orchestration | LangChain `ConversationalRetrievalChain` |
| Memory | `ConversationBufferWindowMemory` (k=5) |
| Document Loaders | `PyPDFLoader`, `UnstructuredHTMLLoader` |
| Interface | Python CLI |

---

## Author

**Samuel Nkpado**
Equity Research Externship | Agentic AI Essentials Certification Program
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## License

This project is for educational purposes. All Vishay Intertechnology documents used are publicly available SEC filings and investor relations materials.
