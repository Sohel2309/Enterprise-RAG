\---

title: Enterprise RAG Platform

emoji: 🔍

colorFrom: blue

colorTo: indigo

sdk: docker

app\_port: 7860

pinned: false

\---



\# Enterprise RAG Platform



Production RAG system with hybrid retrieval, RAGAS evaluation, and conversation memory.



\## Live Demo



&#x20;  🔴 \*\*\[Try the Live Demo](https://sohel2309-enterprise-rag.hf.space)\*\*



&#x20;  Ask questions about RAG, retrieval, and machine learning research.

&#x20;  Searching across 7,536 indexed chunks from 50 ArXiv papers.



\## Benchmark Results

| Metric | Score |

|---|---|

| Answer Relevancy | 0.936 |

| Context Precision | 0.636 |

| Context Recall | 0.362 |

| Faithfulness | 0.233 |



\## Architecture

\- Hybrid retrieval: BM25 + BGE-large embeddings + RRF fusion

\- Cohere cross-encoder reranking

\- Conversation memory (last 3 turns)

\- RAGAS automated evaluation

\- NLI hallucination detection (DeBERTa-v3)



\## Tech Stack

Python 3.11 · FastAPI · Streamlit · FAISS · Qdrant ·

Llama 3.3 70B · Groq · RAGAS · Docker

