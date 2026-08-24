from typing import List, Dict
from retrieval.reranker import CohereReranker


class HybridRetriever:
    def __init__(self, dense_store, bm25_retriever, embedder, use_reranker: bool = True):
        self.dense = dense_store
        self.bm25 = bm25_retriever
        self.embedder = embedder
        self.reranker = CohereReranker() if use_reranker else None

    def _reciprocal_rank_fusion(self, dense_results: List[Dict],
                                bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        scores: dict = {}
        texts: dict = {}
        for rank, doc in enumerate(dense_results):
            key = doc['text'][:100]
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + k)
            texts[key] = doc
        for rank, doc in enumerate(bm25_results):
            key = doc['text'][:100]
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + k)
            texts[key] = doc
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [texts[key] for key, _ in ranked]

    def retrieve(self, query: str, top_k_retrieve: int = 50,
                 top_k_rerank: int = 5) -> List[Dict]:
        # Dense retrieval
        query_vec = self.embedder.embed([query])[0]
        dense_results = self.dense.search(query_vec, top_k=top_k_retrieve)

        # BM25 sparse retrieval
        bm25_results = self.bm25.search(query, top_k=top_k_retrieve)

        # Hybrid fusion
        fused = self._reciprocal_rank_fusion(dense_results, bm25_results)

        if not fused:
            return []

        # Reranking (with graceful fallback)
        if self.reranker:
            candidates = [r['text'] for r in fused[:top_k_retrieve]]
            reranked = self.reranker.rerank(query, candidates, top_n=top_k_rerank)
            # Merge metadata back
            text_to_meta = {r['text'][:100]: r.get('metadata', {}) for r in fused}
            for r in reranked:
                r['metadata'] = text_to_meta.get(r['text'][:100], {})
            return reranked
        else:
            return fused[:top_k_rerank]