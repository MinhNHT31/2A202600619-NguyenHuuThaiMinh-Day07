from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split text into sentences using delimiters: ". ", "! ", "? ", ".\n"
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = (parts[i] + parts[i+1]).strip()
            if sentence:
                sentences.append(sentence)
        if len(parts) % 2 == 1:
            last_part = parts[-1].strip()
            if last_part:
                sentences.append(last_part)

        # Group sentences into chunks of max_sentences_per_chunk
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.max_sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Find the first separator that exists in current_text
        sep = None
        sep_idx = -1
        for idx, s in enumerate(remaining_separators):
            if s == "":
                sep = s
                sep_idx = idx
                break
            if s in current_text:
                sep = s
                sep_idx = idx
                break

        if sep is None:
            # If no separator was found or remaining_separators is empty, divide current_text into slices of chunk_size
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        # Split current_text by sep
        if sep == "":
            splits = list(current_text)
        else:
            splits = current_text.split(sep)

        # Recursively split splits that are too long
        next_separators = remaining_separators[sep_idx + 1 :]
        processed_splits = []
        for s in splits:
            if len(s) > self.chunk_size:
                processed_splits.extend(self._split(s, next_separators))
            else:
                processed_splits.append(s)

        # Merge splits back together up to chunk_size
        chunks = []
        current_chunk = []
        current_len = 0

        for s in processed_splits:
            if not s:
                continue
            sep_len = len(sep) if current_chunk else 0
            if current_len + sep_len + len(s) <= self.chunk_size:
                current_chunk.append(s)
                current_len += sep_len + len(s)
            else:
                if current_chunk:
                    chunks.append(sep.join(current_chunk))
                current_chunk = [s]
                current_len = len(s)

        if current_chunk:
            chunks.append(sep.join(current_chunk))

        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(vec_a, vec_b)) / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_size_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=max(0, chunk_size // 10))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        fixed_chunks = fixed_size_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        def get_stats(chunks):
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0.0
            return {
                "count": count,
                "avg_length": avg_length,
                "chunks": chunks
            }

        return {
            "fixed_size": get_stats(fixed_chunks),
            "by_sentences": get_stats(sentence_chunks),
            "recursive": get_stats(recursive_chunks)
        }
