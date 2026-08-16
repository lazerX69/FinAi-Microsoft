# FinAi -- Local Financial Literacy RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.59-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5-orange?style=flat-square)](https://www.trychroma.com/)
[![Foundry Local](https://img.shields.io/badge/Microsoft-Foundry%20Local-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://aka.ms/foundry-local)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**A fully offline, source-grounded financial literacy Q&A assistant powered by Retrieval-Augmented Generation (RAG).**

[Features](#-features) | [Architecture](#-architecture) | [Requirements](#-requirements) | [Setup](#-setup-from-scratch) | [Usage](#-usage) | [Testing](#-testing) | [Project Structure](#-project-structure)

---

## Overview

**FinAi** is a Python application that answers financial literacy questions using a fully local Retrieval-Augmented Generation (RAG) architecture. It runs entirely on your machine -- no cloud API keys, no internet connection needed after the initial model download.

The system only generates answers from passages retrieved from its local knowledge base. If not enough relevant sources are found, the LLM is never called and no information is fabricated -- making it safe, transparent, and hallucination-resistant by design.

> **Tech Stack:** Python | Microsoft Foundry Local SDK | Qwen 3.5 2B | multilingual-e5-large | ChromaDB | Streamlit | pytest

---

## Features

| Feature | Description |
|---|---|
| **Fully Offline** | Runs 100% locally via Microsoft Foundry Local after first setup |
| **RAG Architecture** | Retrieves relevant passages before generating any answer |
| **Multilingual** | Supports Turkish and English (UI toggle included) |
| **Vector Database** | ChromaDB persistent store with 1024-dim multilingual embeddings |
| **Hybrid Retrieval** | Semantic search with similarity score filtering |
| **Quality Control** | LLM output is checked; falls back to source-based answer if needed |
| **Out-of-Scope Guard** | Safely refuses off-topic questions -- no hallucination |
| **Document Ingestion** | Supports TXT and PDF files in the documents/ folder |
| **Source Transparency** | Every answer shows source passages and similarity scores |
| **Streamlit UI** | Clean, dark-themed, responsive web interface |

---

## Architecture

```
User Question
      |
      v
Query Embedding  (multilingual-e5-large)
      |
      v
Hybrid Retrieval  (ChromaDB semantic search)
      |
      v
Source Quality Check  (min similarity score: 0.55)
      |
      v
Context Assembly  (top-K retrieved passages)
      |
      v
Foundry Local LLM  (Qwen 3.5 2B -- fully on-device)
      |
      v
Answer Quality Control
      |
      v
LLM Answer  OR  Deterministic Source-Based Fallback
      |
      v
Final Answer + Source Citations
```

---

## Requirements

### System Requirements

| Component | Requirement |
|---|---|
| **OS** | Windows 10 / 11 (64-bit) |
| **Python** | 3.10 or higher |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **Disk Space** | ~6 GB (model weights + embeddings + ChromaDB) |
| **Internet** | Required only for first-time model and embedding download |

### Software Requirements

- **Python 3.10+** -- https://www.python.org/downloads/
- **Microsoft Foundry Local** -- installed via `winget` (see setup below)
- **Git** -- https://git-scm.com/ (for cloning the repo)

### Python Dependencies

All Python packages are listed in `requirements.txt`:

```
foundry-local-sdk==1.2.3
openai==2.45.0
chromadb==1.5.9
sentence-transformers==5.6.0
pypdf==6.14.2
streamlit==1.59.1
python-dotenv==1.2.2
pytest==9.1.1
pytest-cov==7.1.0
```

---

## Setup From Scratch

Follow these steps **in order**. All commands are run in **Windows PowerShell**.

### Step 1 -- Clone the Repository

```powershell
git clone https://github.com/<your-username>/FinAi.git
cd FinAi
```

### Step 2 -- Install Microsoft Foundry Local

Foundry Local is the on-device inference engine that runs the LLM.

```powershell
winget install Microsoft.FoundryLocal
```

After installation, **restart PowerShell**, then verify:

```powershell
foundry --version
foundry model list
```

> If `winget` is not available on your system, download Foundry Local directly from: https://aka.ms/foundry-local

### Step 3 -- Create a Python Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt confirming the environment is active.

### Step 4 -- Install Python Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> This step downloads the `sentence-transformers` embedding model (~1.1 GB) from Hugging Face on first run. Subsequent runs use the local cache.

### Step 5 -- Configure Environment Variables (Optional)

Copy the example env file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` if needed:

```env
# Optional: Hugging Face token for higher download rate limits
HF_TOKEN=

PYTHONUTF8=1
TRANSFORMERS_VERBOSITY=warning
```

> `HF_TOKEN` is **not required** for the app to work. It only helps avoid rate-limiting when downloading the embedding model for the first time.

### Step 6 -- Add Your Documents

Place any `.txt` or text-layer `.pdf` files into the `documents/` folder. A sample financial literacy encyclopedia is already included.

```
documents/
+-- finansal_kavramlar_ansiklopedisi.txt   (included sample)
+-- your_document.pdf                       (add your own files here)
```

### Step 7 -- Index Your Documents (Build the Vector Database)

```powershell
python -m src.ingest
```

This chunks, embeds, and stores all documents into the local ChromaDB vector database under `data/chroma/`.

Verify the chunk count after indexing:

```powershell
python -c "from src.vector_store import VectorStore; vs = VectorStore(); print(vs.count())"
```

> Expected output: a number around `540` for the included sample document.

### Step 8 -- Run the App

```powershell
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Usage

### Web Interface

After running `streamlit run app.py`:

1. **Ask a question** -- Type a financial literacy question in the chat input at the bottom
2. **Browse sources** -- Expand "Sources used" to see which passages were retrieved
3. **Adjust settings** -- Use the sidebar sliders to tune retrieval sensitivity
4. **Switch language** -- Click the language toggle button in the sidebar to switch between Turkish and English
5. **Use examples** -- Click any example question in the sidebar to pre-fill it

### Command Line Tools

You can also test individual components directly from the terminal:

**Test retrieval only:**
```powershell
python -m src.retriever "What is compound interest?"
```

**Test retrieval with full context display:**
```powershell
python -m src.retriever "What is compound interest?" --show-context
```

**Test the Foundry Local LLM directly:**
```powershell
python -m src.foundry_client "What is compound interest?"
```

**Test the full RAG pipeline:**
```powershell
python -m src.rag "What is a mutual fund?" --use-llm
```

**Test out-of-scope rejection:**
```powershell
python -m src.rag "How do I grow crops on Mars?" --use-llm
# Expected: system returns no answer because no relevant sources are found
```

---

## Configuration Reference

All settings live in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `qwen3.5-2b-text` | Foundry Local LLM model alias |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence-transformer embedding model |
| `CHUNK_SIZE` | `900` | Characters per document chunk |
| `CHUNK_OVERLAP` | `140` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of top sources retrieved per query |
| `DEFAULT_MIN_SCORE` | `0.55` | Minimum cosine similarity to accept a source |
| `MAX_CONTEXT_CHARACTERS` | `5000` | Max characters passed to the LLM as context |
| `PREFER_DETERMINISTIC_ANSWERS` | `True` | Use source-based fallback by default |
| `ENABLE_LLM_GENERATION` | `False` | Enable LLM only when explicitly requested |

> **Important:** If you change `EMBEDDING_MODEL`, you **must** re-run `python -m src.ingest` to rebuild the vector database. The new model likely has different embedding dimensions and the existing index will be incompatible.

---

## Testing

### Run All Unit Tests

```powershell
python -m pytest -m "not integration"
```

### Run All Tests Including Integration Tests

> Requires a populated ChromaDB collection and Foundry Local running.

```powershell
python -m pytest
```

### Run With Coverage Report

```powershell
python -m pytest --cov=src --cov-report=term-missing
```

---

## Project Structure

```
FinAi/
|
+-- app.py                    # Streamlit web application (entry point)
|
+-- src/
|   +-- config.py             # Central configuration (models, paths, thresholds)
|   +-- embedding.py          # Sentence-transformer embedding wrapper
|   +-- vector_store.py       # ChromaDB collection management
|   +-- ingest.py             # Document chunking and indexing pipeline
|   +-- retriever.py          # Hybrid semantic retrieval
|   +-- foundry_client.py     # Microsoft Foundry Local LLM client
|   +-- rag.py                # Full RAG pipeline orchestration
|
+-- documents/                # Place your TXT/PDF knowledge base files here
|   +-- finansal_kavramlar_ansiklopedisi.txt
|
+-- data/                     # Auto-generated at runtime (not committed to git)
|   +-- chroma/               # ChromaDB persistent vector store
|   +-- model_cache/          # Embedding model cache
|
+-- tests/                    # pytest test suite
|
+-- requirements.txt          # Python dependencies
+-- pyproject.toml            # Project metadata and pytest configuration
+-- .env.example              # Environment variable template
+-- .gitignore                # Git ignore rules
+-- LICENSE                   # MIT License
```

---

## Privacy and Data

FinAi is designed with a **local-first, privacy-first** approach:

- Documents are indexed and stored **locally** in ChromaDB -- never uploaded anywhere
- Answer generation runs **on-device** via Foundry Local -- no cloud API calls are made
- The embedding model (`multilingual-e5-large`) is downloaded once from Hugging Face on first run, then cached locally
- Foundry Local downloads the LLM (`qwen3.5-2b-text`) on first use -- after that it runs fully offline

---

## Common Issues and Fixes

**`foundry` command not found after install:**
> Close and reopen PowerShell after installing Foundry Local via winget. The PATH environment variable needs to refresh.

**Hugging Face warning "You are sending unauthenticated requests to the HF Hub":**
> This is informational only, not a functional error. The embedding model will still download successfully. Set `HF_TOKEN` in your `.env` file to suppress it.

**"ChromaDB collection is empty" error in the UI:**
> You have not indexed any documents yet. Run `python -m src.ingest` before starting the app.

**Slow first response:**
> On the first query, Foundry Local loads the LLM model weights into memory. This takes 10-30 seconds. Subsequent queries in the same session are much faster.

**`Set-ExecutionPolicy` error when activating venv:**
> Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` before the activation command.

**Embedding model re-downloads on every run:**
> Make sure your `data/model_cache/` directory exists and is writable. The model cache path is controlled by `MODEL_CACHE_DIR` in `src/config.py`.

---

## Financial Disclaimer

FinAi is developed for **educational and informational purposes only**.

The answers provided by this application are **not** personalized investment, credit, tax, legal, insurance, or retirement advice. Always consult a qualified financial professional for decisions specific to your situation.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Built as an internship project submission for ZAID IYAD J.J.DWEEKAT // Microsoft Turkiye.
