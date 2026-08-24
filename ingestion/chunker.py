import re
import nltk
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Chunk:
    text: str
    metadata: dict
    strategy: str


class DocumentChunker:
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model

    def chunk(self, text: str, metadata: dict, strategy: str,
              chunk_size: int = 512, chunk_overlap: int = 50) -> List[Chunk]:
        strategies = {
            'fixed': self._fixed_size,
            'recursive': self._recursive,
            'sentence': self._sentence_based,
            'semantic': self._semantic,
            'sliding_window': self._sliding_window,
        }
        if strategy not in strategies:
            raise ValueError(f'Unknown strategy: {strategy}')
        texts = strategies[strategy](text, chunk_size, chunk_overlap)
        texts = [t for t in texts if t.strip()]  # Remove empty chunks
        return [Chunk(
            text=t,
            metadata={**metadata, 'chunk_index': i, 'chunk_total': len(texts)},
            strategy=strategy
        ) for i, t in enumerate(texts)]

    def _fixed_size(self, text, size, overlap):
        splitter = CharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, separator='\n')
        return splitter.split_text(text)

    def _recursive(self, text, size, overlap):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap,
            separators=['\n\n', '\n', '.', ' ', ''])
        return splitter.split_text(text)

    def _sentence_based(self, text, size, overlap):
        # Download both punkt variants to handle all NLTK versions
        for resource in ['punkt', 'punkt_tab']:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:
                    pass
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            # Fallback: split by period if NLTK fails
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

        chunks, current, count = [], '', 0
        for s in sentences:
            if count + len(s) > size and current:
                chunks.append(current.strip())
                current, count = s, len(s)
            else:
                current += ' ' + s
                count += len(s)
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def _semantic(self, text, size, overlap):
        if not self.embedding_model:
            raise ValueError('Semantic chunking requires an embedding model')
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            splitter = SemanticChunker(
                self.embedding_model,
                breakpoint_threshold_type='percentile'
            )
            return splitter.split_text(text)
        except Exception as e:
            print(f'Semantic chunking failed: {e}, falling back to recursive')
            return self._recursive(text, size, overlap)

    def _sliding_window(self, text, size, overlap):
        words = text.split()
        step = max(1, size - overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk = ' '.join(words[i:i + size])
            if chunk:
                chunks.append(chunk)
        return chunks