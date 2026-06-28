import re
from typing import List, Dict, Union


class HallucinationDetector:
    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from transformers import pipeline
                self._model = pipeline(
                    'text-classification',
                    model='cross-encoder/nli-deberta-v3-base',
                    device=-1
                )
                print('Hallucination detector loaded.')
            except Exception as e:
                print(f'Could not load hallucination detector: {e}')
                self._model = 'unavailable'

    def _split_into_claims(self, text: str) -> List[str]:
        try:
            import nltk
            for resource in ['punkt', 'punkt_tab']:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    nltk.download(resource, quiet=True)
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [
                s.strip() + '.'
                for s in text.split('.')
                if len(s.strip()) > 20
            ]
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def _is_refusal(self, answer: str) -> bool:
        """
        Only flag genuine refusals — answers where the system
        explicitly says it cannot answer. NOT answers that
        cite sources or say 'according to context'.
        """
        answer_lower = answer.lower()

        # These are definitive refusal phrases
        hard_refusals = [
            'i cannot find sufficient information',
            'i cannot answer this question',
            'not enough information in the provided',
            'the provided documents do not contain',
            'no information available in the',
        ]
        return any(phrase in answer_lower for phrase in hard_refusals)

    def _check_claim_against_chunks(
        self, claim: str, context_chunks: List[str]
    ) -> tuple:
        best_entail_score = 0.0
        best_label = 'neutral'
        best_score = 0.0

        for chunk in context_chunks:
            try:
                chunk_text = chunk[:400]
                input_text = f'{chunk_text} [SEP] {claim}'
                prediction = self._model(input_text)[0]

                label = prediction['label'].lower()
                score = prediction['score']

                if label == 'entailment':
                    if score > best_entail_score:
                        best_entail_score = score
                        best_label = label
                        best_score = score
                elif label == 'contradiction' and best_entail_score == 0:
                    best_label = label
                    best_score = score
                elif label == 'neutral' and best_label == 'neutral':
                    if score > best_score:
                        best_score = score

            except Exception:
                continue

        is_supported = best_entail_score >= 0.4
        return is_supported, best_entail_score, best_label

    def detect(self, answer: str, context: Union[str, List[str]]) -> Dict:
        default_result = {
            'claims': [],
            'total_claims': 0,
            'supported_claims': 0,
            'hallucination_rate': 0.0,
            'is_hallucinated': False,
            'status': 'not_run'
        }

        if not answer or not context:
            return default_result

        # Only skip genuine refusals
        if self._is_refusal(answer):
            return {
                **default_result,
                'hallucination_rate': 0.0,
                'is_hallucinated': False,
                'status': 'refusal_answer'
            }

        if len(answer.strip()) < 30:
            return {
                **default_result,
                'hallucination_rate': 0.0,
                'is_hallucinated': False,
                'status': 'answer_too_short'
            }

        self._load_model()

        if self._model == 'unavailable':
            return {**default_result, 'status': 'model_unavailable'}

        try:
            # Build context chunks list
            if isinstance(context, list):
                context_chunks = [c for c in context if c.strip()]
            else:
                words = context.split()
                context_chunks = []
                chunk_size = 80
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk:
                        context_chunks.append(chunk)

            if not context_chunks:
                return {**default_result, 'status': 'no_context'}

            claims = self._split_into_claims(answer)
            if not claims:
                return {**default_result, 'status': 'no_claims'}

            results = []
            for claim in claims:
                try:
                    is_supported, score, label = \
                        self._check_claim_against_chunks(
                            claim, context_chunks
                        )
                    results.append({
                        'claim': claim,
                        'label': label,
                        'score': round(score, 3),
                        'supported': is_supported
                    })
                except Exception:
                    results.append({
                        'claim': claim,
                        'label': 'error',
                        'score': 0.0,
                        'supported': True
                    })

            supported = sum(1 for r in results if r['supported'])
            total = len(results)
            hal_rate = 1 - (supported / total) if total > 0 else 0.0

            return {
                'claims': results,
                'total_claims': total,
                'supported_claims': supported,
                'hallucination_rate': round(hal_rate, 3),
                'is_hallucinated': hal_rate > 0.5,
                'status': 'ok'
            }

        except Exception as e:
            print(f'Hallucination detection error: {e}')
            return {**default_result, 'status': f'error: {str(e)}'}