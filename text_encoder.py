import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False


class SimpleTextEncoder:
    """
    Very small, dependency-free encoder used as a fallback when PyTorch or
    sentence-transformers backends are unavailable. It creates deterministic
    token embeddings and averages them per document, then L2-normalizes.
    """

    def __init__(self, dimension: int = 384, random_seed: int = 13):
        self.dimension = dimension
        self.random_state = np.random.RandomState(random_seed)
        self.token_to_vector: dict[str, np.ndarray] = {}

    def _vector_for_token(self, token: str) -> np.ndarray:
        vec = self.token_to_vector.get(token)
        if vec is None:
            # Deterministic per token using a seed derived from the token
            seed = abs(hash(token)) % (2**32)
            local_rng = np.random.RandomState(seed)
            vec = local_rng.normal(0.0, 1.0, size=self.dimension).astype(np.float32)
            self.token_to_vector[token] = vec
        return vec

    def encode(self, texts, convert_to_numpy: bool = True, normalize_embeddings: bool = True):
        embeddings = []
        for text in texts:
            tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
            if not tokens:
                doc_vec = np.zeros(self.dimension, dtype=np.float32)
            else:
                vecs = [self._vector_for_token(tok) for tok in tokens[:2048]]
                doc_vec = np.mean(vecs, axis=0).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(doc_vec) + 1e-12
                doc_vec = (doc_vec / norm).astype(np.float32)
            embeddings.append(doc_vec)
        result = np.stack(embeddings, axis=0)
        if convert_to_numpy:
            return result
        return result.tolist()


def get_text_encoder():
    """Return a real SentenceTransformer when available, otherwise fallback."""
    if _HAS_ST:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            # If model loading fails (e.g., PyTorch missing), use fallback
            return SimpleTextEncoder()
    return SimpleTextEncoder() 