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

### Retrieval Pipeline

- **BM25** sparse retrieval
- **BGE-large** dense embeddings
- **Reciprocal Rank Fusion (RRF)**
- **Cohere cross-encoder reranking**
- Top retrieved results are passed to the generation pipeline

### Generation

- **openai/gpt-oss-120b** through Groq API
- Multi-turn conversation memory
- Automatic session management
- Context truncation for token safety
- Retry/fallback handling

### Evaluation

- **RAGAS** evaluation
- Faithfulness
- Answer relevancy
- Context precision
- Context recall

## 🚀 Tech Stack

**Backend:** FastAPI  
**Frontend:** Streamlit  
**Retrieval:** BM25 + FAISS + RRF  
**Embeddings:** BAAI/bge-large  
**Reranking:** Cohere  
**LLM:** openai/gpt-oss-120b via Groq  
**Evaluation:** RAGAS  
**Deployment:** Hugging Face Spaces

## ⚡ Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Run Backend

```bash
uvicorn api.rag_api:app --reload --port 8000
```

### Run Streamlit

```bash
streamlit run ui/streamlit_app.py
```

Then visit:

```text
http://localhost:8501
```

## 📁 Project Structure

```text
Enterprise-RAG/
├── api/                 # FastAPI backend
├── ingestion/           # PDF/DOCX parsing and chunking
├── retrieval/           # BM25 + FAISS + reranking
├── generation/          # LLM client and prompt templates
├── evaluation/          # RAGAS evaluation
├── ui/                  # Streamlit application
├── scripts/             # Data ingestion and evaluation scripts
└── data/
    └── processed/       # Processed evaluation/index metadata
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
- ✅ Pre-indexed document collection

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