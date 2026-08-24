import os
import time
import cohere
from typing import List, Dict


class CohereReranker:
    def __init__(self):
        self.client = cohere.Client(os.getenv('COHERE_API_KEY', ''))
        self.model = 'rerank-english-v3.0'
        self._last_call = 0
        self._min_interval = 6.5  # Respect 10 calls/minute free tier

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Dict]:
        if not documents:
            return []
        if not os.getenv('COHERE_API_KEY', ''):
            # No key — return documents as-is with dummy scores
            return [{'text': d, 'relevance_score': 1.0 - (i * 0.1)}
                    for i, d in enumerate(documents[:top_n])]
        # Rate limit protection
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        try:
            response = self.client.rerank(
                query=query,
                documents=documents[:50],  # Max 50 candidates
                top_n=top_n,
                model=self.model
            )
            self._last_call = time.time()
            return [
                {'text': documents[r.index], 'relevance_score': r.relevance_score}
                for r in response.results
            ]
        except Exception as e:
            print(f'Cohere rerank failed: {e} — returning top-{top_n} by position')
            return [{'text': d, 'relevance_score': 1.0 - (i * 0.1)}
                    for i, d in enumerate(documents[:top_n])]