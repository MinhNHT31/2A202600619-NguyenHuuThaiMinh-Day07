from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class HashingTFIDFEmbedder:
    """Lightweight, zero-dependency term frequency-inverse document frequency embedder with feature hashing."""

    def __init__(self, dim: int = 1024, data_path: str = "/media/minhnht31/data/vinuni/2A2026-NguyenHuuThaiMinh-Day07/data/data_md") -> None:
        self.dim = dim
        self._backend_name = "hashing-tfidf"
        self.idf = [1.0] * dim

        try:
            p = Path(data_path)
            if p.exists() and p.is_dir():
                docs = []
                for fpath in p.glob("*.md"):
                    docs.append(fpath.read_text(encoding="utf-8"))
                
                if docs:
                    num_docs = len(docs)
                    df = [0] * dim
                    for doc in docs:
                        indices = self._get_unique_indices(doc)
                        for idx in indices:
                            df[idx] += 1
                    
                    self.idf = []
                    for count in df:
                        if count > 0:
                            self.idf.append(math.log(1.0 + (num_docs / count)))
                        else:
                            self.idf.append(1.0)
        except Exception:
            self.idf = [1.0] * dim

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        return re.findall(r'[a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+', text)

    def _get_unique_indices(self, text: str) -> set[int]:
        tokens = self._tokenize(text)
        indices = set()
        for token in tokens:
            digest = hashlib.md5(token.encode('utf-8')).hexdigest()
            idx = int(digest, 16) % self.dim
            indices.add(idx)
        return indices

    def __call__(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        vector = [0.0] * self.dim
        for token in tokens:
            digest = hashlib.md5(token.encode('utf-8')).hexdigest()
            idx = int(digest, 16) % self.dim
            vector[idx] += 1.0
        
        for idx in range(self.dim):
            vector[idx] *= self.idf[idx]

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return [0.0] * self.dim
        return [v / norm for v in vector]



class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


class OpenAIEmbedder:
    """OpenAI embeddings API-backed embedder."""

    def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
        from openai import OpenAI

        self.model_name = model_name
        self._backend_name = model_name
        self.client = OpenAI()

    def __call__(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model_name, input=text)
        return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()
