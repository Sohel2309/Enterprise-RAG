# Enterprise RAG Platform

> Production-grade Retrieval-Augmented Generation system with hybrid retrieval, intelligent reranking, conversation memory, and RAGAS evaluation.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Visit-blue?style=flat)](https://sohel2309-enterprise-rag.hf.space)
[![GitHub](https://img.shields.io/badge/Code-Repository-black?style=flat)](https://github.com/Sohel2309/Enterprise-RAG)

## Overview

Enterprise RAG Platform combines **BM25 sparse retrieval + BGE-large dense embeddings** with **Cohere cross-encoder reranking** to deliver accurate, context-aware answers from uploaded documents.

The platform supports multi-turn conversation memory, real-time PDF/DOCX ingestion, hybrid retrieval, and automated RAGAS evaluation.

## 📊 Key Results

| Metric | Score |
|---|---:|
| Faithfulness | **0.842** |
| Answer Relevancy | **0.900** ⭐ |
| Context Precision | **0.877** |
| Context Recall | **0.721** |
| Indexed Chunks | **7,536** |
| Evaluated QA Pairs | **45** |

## 🏗️ Architecture

```text
Documents (PDF/DOCX)
        ↓
Parsing + Chunking
        ↓
BGE-large Embeddings (Redis-cached)
        ↓
        ├── FAISS (dense search)
        └── BM25 (sparse search)
                ↓
        Reciprocal Rank Fusion (RRF)
                ↓
        Cohere Cross-Encoder Reranking
                ↓
        LLM Generation (Groq gpt-oss-120b)
        + Conversation Memory
                ↓
            Answer + Sources
```

### Retrieval Pipeline

- **BM25** sparse retrieval
- **BGE-large** dense embeddings
- **Reciprocal Rank Fusion (RRF)** to combine both
- **Cohere cross-encoder reranking** on top candidates
- Top retrieved results are passed to the generation pipeline

### Generation

- **openai/gpt-oss-120b** through the Groq API
- Multi-turn conversation memory
- Automatic session management
- Context truncation for token safety
- Retry/fallback handling

### Evaluation

- **RAGAS-style LLM-as-judge evaluation**
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

## 🚀 Tech Stack

**Backend:** FastAPI
**Frontend:** Streamlit
**Retrieval:** BM25 + FAISS + RRF
**Embeddings:** BAAI/bge-large (local, Redis-cached)
**Reranking:** Cohere
**LLM:** openai/gpt-oss-120b via Groq
**Evaluation:** RAGAS-style LLM-as-judge
**Deployment:** Docker · Hugging Face Spaces

## ⚡ Quick Start

### 1. Clone the repo

This repo uses **Git LFS** for pre-built index files (`data/processed/*.bin`, `*.pkl`). Without LFS, those files download as small pointer files and the app starts with an empty index.

```bash
git lfs install
git clone https://github.com/Sohel2309/Enterprise-RAG.git
cd Enterprise-RAG
git lfs pull
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_key          # required — generation & evaluation
COHERE_API_KEY=your_cohere_key      # optional — reranking falls back to
                                     # rank-order without it
REDIS_URL=redis://localhost:6379    # optional — embedding cache; app runs
                                     # fine without Redis, just uncached
```

Get a free `GROQ_API_KEY` at [console.groq.com](https://console.groq.com) and a free `COHERE_API_KEY` at [dashboard.cohere.com](https://dashboard.cohere.com).

### 4. (Optional) Start Redis for embedding caching

```bash
docker compose up -d
```

### 5. Run the app

```bash
uvicorn api.rag_api:app --reload --port 8000   # Terminal 1 — backend
streamlit run ui/streamlit_app.py              # Terminal 2 — frontend
```

Then visit:

```text
http://localhost:8501
```

### Or run with Docker

The app also ships as a single container (this is how the live demo is deployed) that runs the FastAPI backend and Streamlit frontend together:

```bash
docker build -t enterprise-rag .
docker run -p 7860:7860 --env-file .env enterprise-rag
```

Then visit `http://localhost:7860`.

## 📁 Project Structure

```text
Enterprise-RAG/
├── api/                 # FastAPI backend
├── ingestion/           # PDF/DOCX parsing and chunking
├── retrieval/           # BM25 + FAISS + reranking
├── generation/          # LLM client and prompt templates
├── evaluation/          # RAGAS-style evaluation
├── ui/                  # Streamlit application
├── scripts/             # Data ingestion and evaluation scripts
├── app.py               # Combined FastAPI + Streamlit entrypoint (Docker/HF Spaces)
├── Dockerfile
├── docker-compose.yml    # Local Redis for embedding cache
└── data/
    └── processed/        # Pre-built indexes (Git LFS)
```

## 🎯 Features

- ✅ Hybrid BM25 + dense vector retrieval
- ✅ Reciprocal Rank Fusion
- ✅ Cohere intelligent reranking
- ✅ Multi-turn conversation memory
- ✅ Real-time PDF/DOCX document upload
- ✅ Automatic document indexing
- ✅ RAGAS evaluation dashboard
- ✅ Interactive Streamlit interface
- ✅ FastAPI backend
- ✅ Session management
- ✅ Pre-indexed document collection (50 arXiv papers, ~7,500 chunks)

## 📈 Benchmark

Fixed-size chunking outperformed sentence-based chunking by **17% on context precision** in the project benchmark.

The evaluation dataset was generated from **50 ArXiv papers** with a larger QA set during experimentation. The currently reported production evaluation contains **45 valid samples**.

## 🔍 Evaluation Methodology & Limitations

The metrics reported above come from an **LLM-as-judge evaluation**, not a human-labeled benchmark.

The evaluation uses `openai/gpt-oss-120b` against a held-out QA set using the same retrieval configuration as production:

```text
BM25 + Dense Retrieval
        ↓
RRF Fusion
        ↓
Cohere Reranking
        ↓
LLM Generation
        ↓
RAGAS Evaluation
```

The current reported run contains **45 valid samples**.

### Limitations

**Self-judging bias:**
The same model family (`openai/gpt-oss-120b`) is used for generation and evaluation. Therefore, the scores should be interpreted as internal project-level evidence rather than an independent industry benchmark.

**Sample size:**
45 samples provide directional evidence of system performance but are not sufficient for statistically rigorous benchmarking.

The purpose of the evaluation is to demonstrate that retrieval and generation quality were **measured, analyzed, and iterated on**, rather than assumed to work.

## 🔗 Links

- **Live Demo:** https://sohel2309-enterprise-rag.hf.space
- **GitHub:** https://github.com/Sohel2309/Enterprise-RAG

---

**MIT License**
