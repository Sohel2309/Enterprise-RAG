import os
import time
from groq import Groq
from typing import List, Dict
from generation.prompt_templates import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE


class RAGGenerator:
    def __init__(self, provider: str = 'groq'):
        self.provider = provider
        self.client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        # Use 70B for better quality — still free on Groq
        self.model = 'llama-3.3-70b-versatile'
        self._retry_attempts = 3
        self._retry_delay = 2

    def _format_context(self, retrieved_chunks: List[Dict]) -> str:
        if not retrieved_chunks:
            return 'No context available.'
        return '\n\n---\n\n'.join([
            f'[Source {i + 1}]\n{chunk["text"]}'
            for i, chunk in enumerate(retrieved_chunks)
        ])

    def generate(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        context = self._format_context(retrieved_chunks)

        # Truncate context to avoid token limits
        if len(context) > 6000:
            context = context[:6000] + '\n\n[Context truncated]'

        user_message = RAG_USER_TEMPLATE.format(
            context=context,
            question=question
        )

        for attempt in range(self._retry_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': RAG_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_message}
                    ],
                    temperature=0.0,
                    max_tokens=1024,
                    timeout=30
                )
                answer = response.choices[0].message.content

                # Validate answer is not empty or malformed
                if not answer or len(answer.strip()) < 5:
                    raise ValueError('Empty or too-short response from LLM')

                return {
                    'answer': answer.strip(),
                    'context': context,
                    'question': question
                }

            except Exception as e:
                print(f'Generation attempt {attempt + 1} failed: {e}')
                if attempt < self._retry_attempts - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    # Final fallback — never crash the API
                    return {
                        'answer': 'I was unable to generate a response at this time. Please try again.',
                        'context': context,
                        'question': question,
                        'error': str(e)
                    }