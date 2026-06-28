import os
import time
import pandas as pd
from groq import Groq
from typing import List, Dict


class RAGASEvaluator:
    def __init__(self):
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        self.model = 'llama-3.3-70b-versatile'
        self._call_count = 0

    def _llm_score(self, prompt: str, retries: int = 3) -> float:
        for attempt in range(retries):
            try:
                # Rate limit: Groq free tier allows ~30 RPM on 70B
                if self._call_count > 0 and self._call_count % 25 == 0:
                    time.sleep(60)  # Pause every 25 calls

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=0.0,
                    max_tokens=10,  # We only need a number
                    timeout=15
                )
                self._call_count += 1
                text = response.choices[0].message.content.strip()

                # Extract first float-like substring
                import re
                matches = re.findall(r'\d+\.?\d*', text)
                if matches:
                    score = float(matches[0])
                    return min(max(score, 0.0), 1.0)
                return 0.5

            except Exception as e:
                print(f'LLM score attempt {attempt + 1} failed: {e}')
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
        return 0.5  # Default on complete failure

    def _score_faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Rate how faithful this answer is to the context.
Faithful means: every claim in the answer is supported by the context.
Score from 0.0 (completely unfaithful) to 1.0 (completely faithful).

Context: {context[:400]}
Answer: {answer[:300]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_answer_relevancy(self, answer: str, question: str) -> float:
        prompt = f"""Rate how relevant this answer is to the question.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

Question: {question}
Answer: {answer[:300]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_context_precision(self, question: str, context: str) -> float:
        prompt = f"""Rate how precisely the context contains information needed to answer the question.
Score from 0.0 (context is useless) to 1.0 (context perfectly answers the question).

Question: {question}
Context: {context[:400]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def _score_context_recall(self, ground_truth: str, context: str) -> float:
        prompt = f"""Rate how well the context covers the ground truth answer.
Score from 0.0 (context misses everything) to 1.0 (context covers everything).

Ground Truth: {ground_truth[:300]}
Context: {context[:400]}

Reply with only a number between 0 and 1:"""
        return self._llm_score(prompt)

    def evaluate_single(self, qa: Dict) -> Dict:
        question = qa.get('question', '')
        answer = qa.get('answer', '')
        contexts = qa.get('contexts', [])
        ground_truth = qa.get('ground_truth', '')

        context_str = '\n'.join(contexts[:3])[:600]

        return {
            'question': question,
            'faithfulness': self._score_faithfulness(answer, context_str),
            'answer_relevancy': self._score_answer_relevancy(answer, question),
            'context_precision': self._score_context_precision(question, context_str),
            'context_recall': self._score_context_recall(ground_truth, context_str),
        }

    def evaluate(self, qa_pairs: List[Dict]) -> pd.DataFrame:
        results = []
        for i, qa in enumerate(qa_pairs):
            print(f'Evaluating {i + 1}/{len(qa_pairs)}...', end='\r')
            try:
                scores = self.evaluate_single(qa)
                results.append(scores)
            except Exception as e:
                print(f'\nFailed sample {i + 1}: {e}')
                results.append({
                    'question': qa.get('question', ''),
                    'faithfulness': 0.5,
                    'answer_relevancy': 0.5,
                    'context_precision': 0.5,
                    'context_recall': 0.5,
                })
        return pd.DataFrame(results)

    def print_summary(self, df: pd.DataFrame):
        print('\n=== RAGAS Evaluation Summary ===')
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
            if metric in df.columns:
                print(f'{metric:25s}: {df[metric].mean():.3f}')