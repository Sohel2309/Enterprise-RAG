# Enterprise RAG Platform



> Production-grade retrieval-augmented generation system with hybrid retrieval, intelligent reranking, and conversation memory.



[![Live Demo](https://img.shields.io/badge/Live\_Demo-Visit-blue?style=flat)](https://sohel2309-enterprise-rag.hf.space)

[![GitHub](https://img.shields.io/badge/Code-Repository-black?style=flat)](https://github.com/Sohel2309/Enterprise-RAG)



## Overview



Enterprise RAG Platform combines BM25 sparse + BGE dense embeddings with Cohere reranking to deliver accurate, context-aware answers. Features conversation memory, real-time document upload, and automated RAGAS evaluation.



## 📊 Key Results



| Metric | Score |

|--------|-------|

| Answer Relevancy | \*\*0.936\*\* ⭐ |

| Context Precision | 0.636 |

| Indexed Chunks | 7,536 |

| Evaluated QA Pairs | 355 |



## 🏗️ Architecture



**Retrieval Pipeline:**

- BM25 (sparse) + BGE-large embeddings (dense) with RRF fusion

- Cohere cross-encoder reranking (top-60 → top-10)

- DeBERTa-v3 NLI hallucination detection per-chunk



**Generation:**

- Groq Llama 3.3 70B with 3-turn conversation memory

- Automatic session management + real-time document ingestion



## 🚀 Tech Stack



FastAPI · Streamlit · FAISS · Qdrant · BGE Embeddings · Cohere Reranking · Groq LLM · RAGAS Evaluation · HuggingFace Spaces



## ⚡ Quick Start



```bash

# Install

pip install -r requirements.txt



# Run locally

uvicorn api.rag_api:app --reload --port 8000  # Terminal 1

streamlit run ui/streamlit_app.py             # Terminal 2



# Visit http://localhost:8501

```



## 📁 Project Structure

├── api/                    # FastAPI backend



├── ingestion/              # PDF/DOCX parsing + 5 chunking strategies



├── retrieval/              # BM25 + FAISS + Cohere reranking



├── generation/             # Groq LLM client



├── evaluation/             # RAGAS + hallucination detection



├── ui/                     # Streamlit dashboard (3 tabs)



└── data/processed/         # Pre-indexed chunks (FAISS + BM25)



## 🎯 Features



✅ Hybrid retrieval (BM25 + dense embeddings)  

✅ Intelligent reranking with Cohere  

✅ Multi-turn conversation memory  

✅ Real-time PDF/DOCX upload  

✅ Automated hallucination detection  

✅ Interactive evaluation dashboard  



## 📈 Benchmark



Fixed-size chunking outperformed sentence-based by \*\*17%\*\* on context precision. Evaluated on 50 ArXiv papers with 355 QA pairs.



## 🔗 Links



- **Live Demo:** https://sohel2309-enterprise-rag.hf.space

- **Code:** https://github.com/Sohel2309/Enterprise-RAG



---



\*\*MIT License\*\* | Last Updated: June 2026

