import os
import json
import time
import pickle
import numpy as np
import faiss as faiss_lib
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")

from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import CachedEmbedder
from retrieval.vector_store import FAISSStore, QdrantStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from generation.llm_client import RAGGenerator
from evaluation.hallucination_detector import HallucinationDetector

app = FastAPI(
    title='Enterprise RAG API',
    version='2.0.0',
    description='Production RAG with conversation memory and document upload'
)

# ── Global state ──────────────────────────────────────────────────────────────
embedder = None
faiss_store = None
bm25 = None
hybrid = None
generator = None
detector = None

conversation_store: dict = {}


def load_components():
    global embedder, faiss_store, bm25, hybrid, generator, detector

    print('Loading components...')
    embedder = CachedEmbedder()

    faiss_store = FAISSStore(dim=1024)
    try:
        faiss_store.index = faiss_lib.read_index('data/processed/faiss_index.bin')
        with open('data/processed/faiss_metadata.json') as f:
            faiss_store.metadata = json.load(f)
        print(f'FAISS: {faiss_store.index.ntotal} vectors loaded')
    except FileNotFoundError:
        print('WARNING: FAISS index not found. Upload documents to get started.')

    try:
        with open('data/processed/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
        print(f'BM25: {len(bm25.corpus)} documents loaded')
    except FileNotFoundError:
        bm25 = BM25Retriever()
        print('WARNING: BM25 index not found.')

    hybrid = HybridRetriever(faiss_store, bm25, embedder, use_reranker=True)
    generator = RAGGenerator(provider='groq')
    detector = HallucinationDetector()
    print('All components loaded.')


load_components()


def rebuild_index():
    global faiss_store, bm25, hybrid
    try:
        faiss_store.index = faiss_lib.read_index('data/processed/faiss_index.bin')
        with open('data/processed/faiss_metadata.json') as f:
            faiss_store.metadata = json.load(f)
        with open('data/processed/bm25_index.pkl', 'rb') as f:
            bm25 = pickle.load(f)
        hybrid = HybridRetriever(faiss_store, bm25, embedder, use_reranker=True)
        print(f'Index rebuilt: {faiss_store.index.ntotal} vectors')
        return True
    except Exception as e:
        print(f'Index rebuild failed: {e}')
        return False


# ── Pydantic Models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str = 'default'
    detect_hallucinations: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float
    session_id: str
    turn_number: int
    hallucination_report: Optional[dict] = None


class ClearSessionRequest(BaseModel):
    session_id: str


class UploadResponse(BaseModel):
    success: bool
    message: str
    chunks_added: int
    total_vectors: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/query', response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail='Question cannot be empty')

    if faiss_store.index.ntotal == 0:
        raise HTTPException(
            status_code=503,
            detail='No documents indexed. Please upload documents first.'
        )

    start = time.time()

    session_id = request.session_id or 'default'
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    history = conversation_store[session_id]
    turn_number = len(history) + 1

    try:
        chunks = hybrid.retrieve(
            request.question,
            top_k_retrieve=50,
            top_k_rerank=request.top_k
        )

        if not chunks:
            return QueryResponse(
                answer='No relevant documents found for your question.',
                sources=[],
                latency_ms=round((time.time() - start) * 1000, 2),
                session_id=session_id,
                turn_number=turn_number
            )

        context = _format_context_with_history(chunks, history, request.question)
        answer = _generate_with_history(request.question, context, history)

        if not answer or len(answer.strip()) < 5:
            answer = 'I was unable to generate a response. Please try again.'

        conversation_store[session_id].append({
            'question': request.question,
            'answer': answer,
            'timestamp': time.time()
        })

        if len(conversation_store[session_id]) > 10:
            conversation_store[session_id] = conversation_store[session_id][-10:]

        # ── FIXED: Pass chunk texts as list, not joined string ────────────────
        hal_report = None
        if request.detect_hallucinations:
            context_chunks = [c.get('text', '') for c in chunks if c.get('text')]
            hal_report = detector.detect(answer, context_chunks)

        return QueryResponse(
            answer=answer,
            sources=chunks,
            latency_ms=round((time.time() - start) * 1000, 2),
            session_id=session_id,
            turn_number=turn_number,
            hallucination_report=hal_report
        )

    except Exception as e:
        print(f'Query error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


def _format_context_with_history(chunks, history, current_question):
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f'[Source {i + 1}]\n{chunk["text"]}')
    return '\n\n---\n\n'.join(context_parts)


def _generate_with_history(question: str, context: str, history: list) -> str:
    from generation.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
    import time as time_module

    if len(context) > 5000:
        context = context[:5000] + '\n\n[Context truncated]'

    messages = [{'role': 'system', 'content': RAG_SYSTEM_PROMPT}]

    recent_history = history[-3:] if len(history) > 3 else history
    for turn in recent_history:
        messages.append({'role': 'user', 'content': turn['question']})
        messages.append({'role': 'assistant', 'content': turn['answer']})

    user_message = RAG_USER_TEMPLATE.format(
        context=context,
        question=question
    )
    messages.append({'role': 'user', 'content': user_message})

    for attempt in range(3):
        try:
            response = generator.client.chat.completions.create(
                model=generator.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f'Generation attempt {attempt + 1} failed: {e}')
            if attempt < 2:
                time_module.sleep(2 * (attempt + 1))

    return 'I was unable to generate a response at this time. Please try again.'


@app.post('/upload', response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ''
    if not filename.lower().endswith(('.pdf', '.docx', '.doc')):
        raise HTTPException(
            status_code=400,
            detail='Only PDF and DOCX files are supported'
        )

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail='File too large. Maximum size is 20MB.'
        )

    upload_dir = Path('data/uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / filename

    try:
        with open(file_path, 'wb') as f:
            f.write(contents)

        parser = DocumentParser()
        parsed = parser.parse(str(file_path))

        if not parsed.text.strip():
            raise ValueError('Could not extract text from document')

        chunker = DocumentChunker()
        chunks = chunker.chunk(
            parsed.text,
            parsed.metadata,
            strategy='recursive',
            chunk_size=512,
            chunk_overlap=100
        )

        if not chunks:
            raise ValueError('No chunks created from document')

        texts = [c.text for c in chunks]
        batch_size = 32
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.embed(batch)
            all_embeddings.extend(batch_embeddings)

        vecs = np.array(all_embeddings, dtype='float32')
        faiss_lib.normalize_L2(vecs)
        faiss_store.index.add(vecs)
        faiss_store.metadata.extend([
            {'text': c.text, 'metadata': c.metadata}
            for c in chunks
        ])

        new_corpus = [
            {'text': c.text, 'metadata': c.metadata}
            for c in chunks
        ]
        bm25.corpus.extend(new_corpus)
        bm25.index(bm25.corpus)

        qdrant = QdrantStore(collection='documents')
        try:
            existing = [
                c.name for c in qdrant.client.get_collections().collections
            ]
            if 'documents' not in existing:
                qdrant.create_collection(dim=1024)
            qdrant.add(all_embeddings, chunks)
        except Exception as e:
            print(f'Qdrant update failed (non-critical): {e}')

        os.makedirs('data/processed', exist_ok=True)
        faiss_lib.write_index(
            faiss_store.index,
            'data/processed/faiss_index.bin'
        )
        with open('data/processed/faiss_metadata.json', 'w') as f:
            json.dump(faiss_store.metadata, f)
        with open('data/processed/bm25_index.pkl', 'wb') as f:
            pickle.dump(bm25, f)

        return UploadResponse(
            success=True,
            message=f'Successfully indexed {filename}',
            chunks_added=len(chunks),
            total_vectors=faiss_store.index.ntotal
        )

    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        print(f'Upload error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/clear_session')
async def clear_session(request: ClearSessionRequest):
    if request.session_id in conversation_store:
        conversation_store.pop(request.session_id)
    return {
        'success': True,
        'message': f'Session {request.session_id} cleared'
    }


@app.get('/session/{session_id}')
async def get_session(session_id: str):
    history = conversation_store.get(session_id, [])
    return {
        'session_id': session_id,
        'turns': len(history),
        'history': history
    }


@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'components': {
            'faiss_vectors': faiss_store.index.ntotal if faiss_store else 0,
            'bm25_documents': len(bm25.corpus) if bm25 else 0,
            'generator': 'groq/llama-3.3-70b-versatile',
            'embedder': 'BAAI/bge-large-en-v1.5',
            'active_sessions': len(conversation_store)
        }
    }


@app.get('/')
async def root():
    return {
        'message': 'Enterprise RAG API v2.0',
        'docs': '/docs',
        'health': '/health'
    }