import os
import json
from typing import List, Dict, Tuple, Optional

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False

import numpy as np
import requests
# from sentence_transformers import SentenceTransformer, util
import re
from text_encoder import get_text_encoder


class RAGEngine:
    def __init__(self):
        self.encoder = get_text_encoder()
        self.index = None
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray | None = None
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        
        # Retrieval configuration
        self.min_cosine: float = 0.25
        self.max_context_docs: int = 6
        self.use_mmr: bool = True
        self.mmr_lambda: float = 0.65
        # Backend availability
        self.use_faiss: bool = _HAS_FAISS

    def build_index(self, chunks: List[Dict]):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.embeddings = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = int(self.embeddings.shape[1])
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.embeddings)
        else:
            self.index = None  # NumPy fallback will be used

    def _encode(self, texts: List[str]) -> np.ndarray:
        embs = self.encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs

    def _mmr(self, query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float) -> List[int]:
        selected: List[int] = []
        candidates = list(range(len(doc_vecs)))
        if not candidates:
            return selected
        # Precompute sims
        q_sims = (doc_vecs @ query_vec.T).reshape(-1)
        # Pick the best first
        best = int(np.argmax(q_sims))
        selected.append(best)
        candidates.remove(best)
        while len(selected) < k and candidates:
            max_score = -1e9
            max_idx = None
            for i in candidates:
                diversity = max((doc_vecs[i] @ doc_vecs[j].T) for j in selected)
                mmr_score = lambda_mult * q_sims[i] - (1 - lambda_mult) * diversity
                if mmr_score > max_score:
                    max_score = mmr_score
                    max_idx = i
            selected.append(max_idx)
            candidates.remove(max_idx)
        return selected

    def _search_numpy(self, query_vec: np.ndarray, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) using pure NumPy inner product search.
        Assumes self.embeddings are L2-normalized so IP == cosine.
        """
        if self.embeddings is None or self.embeddings.size == 0:
            return np.array([]), np.array([])
        sims = (self.embeddings @ query_vec.T).reshape(-1)
        n = sims.shape[0]
        k = min(top_n, n)
        if k <= 0:
            return np.array([]), np.array([])
        # Use argpartition for efficiency, then sort top-k
        top_idx_part = np.argpartition(-sims, k - 1)[:k]
        top_sorted_local = np.argsort(-sims[top_idx_part])
        top_indices = top_idx_part[top_sorted_local]
        top_scores = sims[top_indices]
        return top_scores, top_indices

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self._encode([query])  # shape (1, d)
        top_n = min(self.max_context_docs * 3, len(self.chunks))
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_vec, top_n)
            scores = scores[0]
            indices = indices[0]
        else:
            scores, indices = self._search_numpy(query_vec[0], top_n)
        # Filter by cosine threshold first
        prelim: List[Tuple[int, float]] = [
            (int(idx), float(score))
            for score, idx in zip(scores, indices)
            if 0 <= int(idx) < len(self.chunks) and float(score) >= self.min_cosine
        ]
        if not prelim:
            prelim = [
                (int(idx), float(score))
                for score, idx in zip(scores, indices)
                if 0 <= int(idx) < len(self.chunks)
            ][:k]
            if not prelim:
                return []
        # Optional MMR diversification on the filtered set
        chosen = prelim
        if self.use_mmr:
            doc_vecs = self.embeddings[[i for i, _ in prelim]]
            sel_indices = self._mmr(query_vec[0], doc_vecs, k=min(k, len(prelim)), lambda_mult=self.mmr_lambda)
            chosen = [prelim[i] for i in sel_indices]
        else:
            chosen = sorted(prelim, key=lambda x: x[1], reverse=True)[:k]

        relevant_chunks: List[Dict] = []
        for idx, score in chosen[:k]:
            chunk = self.chunks[idx].copy()
            chunk["relevance_score"] = float(score)
            # Add a short quote for grounding
            quote = chunk["text"][:240].strip()
            chunk["quote"] = quote
            relevant_chunks.append(chunk)
        return relevant_chunks

    def debug_search(self, query: str, top_n: int = 10) -> Dict:
        if self.embeddings is None or len(self.chunks) == 0:
            return {"top": [], "total_chunks": len(self.chunks), "min_cosine": self.min_cosine}
        query_vec = self._encode([query])
        n = min(top_n, len(self.chunks))
        if self.use_faiss and self.index is not None:
            scores, indices = self.index.search(query_vec, n)
            scores = scores[0]
            indices = indices[0]
        else:
            scores, indices = self._search_numpy(query_vec[0], n)
        top = [
            {
                "idx": int(idx),
                "score": float(score),
                "filename": self.chunks[int(idx)].get("filename"),
                "page": self.chunks[int(idx)].get("page"),
                "quote": (self.chunks[int(idx)].get("text") or "")[:180],
            }
            for score, idx in zip(scores, indices)
            if 0 <= int(idx) < len(self.chunks)
        ]
        return {"top": top, "total_chunks": len(self.chunks), "min_cosine": self.min_cosine}

    def _build_context(self, context_chunks: List[Dict]) -> str:
        return "\n\n".join(
            [
                f"Source {i+1} — {c['filename']} (Page {c['page']}):\n{c['text']}"
                for i, c in enumerate(context_chunks[: self.max_context_docs])
            ]
        )

    def generate_answer_with_perplexity(self, query: str, context_chunks: List[Dict], history: Optional[List[Dict]] = None) -> str:
        if not self.perplexity_api_key:
            return "Please provide a Perplexity API key to generate answers."
        if not context_chunks:
            return "I couldn't find relevant content in the provided PDFs for this query."
        context_text = self._build_context(context_chunks)
        history = history or []
        history_text = "\n".join(
            [
                f"{('User' if m.get('role') == 'user' else 'Assistant')}: {m.get('content','').strip()}"
                for m in history
                if (m.get('role') in ('user', 'assistant') and str(m.get('content', '')).strip())
            ]
        ).strip()

        prompt = (
            "You are a careful assistant that must answer ONLY using the quoted sources provided below. "
            "Do not add facts that are not present in the sources. If the sources do not contain the answer, say: "
            '"I could not find this in the provided PDF context."\n\n'
            + (f"Conversation so far (for resolving references like 'it/that/this'):\n{history_text}\n\n" if history_text else "")
            + f"Sources (verbatim from the uploaded PDFs):\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Instructions:\n"
            "- Answer concisely and factually using only the sources above.\n"
            "- When you state a fact, add (Source X, Page Y).\n"
            "- If multiple sources agree, cite all briefly.\n"
            "- If the term appears in the sources (e.g., 'Super AI'), include it exactly as written.\n"
            "- If the answer is not present, explicitly say you could not find it."
        )
        headers = {"Authorization": f"Bearer {self.perplexity_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "sonar-pro",
            "messages": [
                {"role": "system", "content": "Answer strictly from the provided sources; refuse otherwise."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 800,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
        try:
            resp = requests.post(self.perplexity_url, json=payload, headers=headers, timeout=45)
            if resp.status_code != 200:
                return f"Error: API {resp.status_code}: {resp.text}"
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating answer: {e}"

    def generate_fallback_answer(self, query: str, context_chunks: List[Dict]) -> str:
        if not context_chunks:
            return "I couldn't find relevant content in the provided PDFs for this query."
        # Simple extractive style: pick sentences that contain query key phrases
        query_terms = [w for w in re.split(r"[^A-Za-z0-9]+", query.lower()) if len(w) > 2]
        best: Dict | None = None
        best_score = -1
        for ch in context_chunks:
            text_l = ch["text"].lower()
            score = sum(text_l.count(t) for t in query_terms)
            if score > best_score:
                best_score = score
                best = ch
        if not best:
            return "I couldn't find relevant content in the provided PDFs for this query."
        # Extract up to 3 sentences that contain query terms
        sentences = re.split(r"(?<=[.!?])\s+", best["text"])[:20]
        chosen: List[str] = []
        for s in sentences:
            s_l = s.lower()
            if any(t in s_l for t in query_terms):
                chosen.append(s.strip())
            if len(chosen) >= 3:
                break
        if not chosen:
            chosen = sentences[:2]
        answer = " ".join(chosen).strip()
        return f"{answer} (Source 1, Page {best['page']} — {best['filename']})"

    def query(self, query: str, history: Optional[List[Dict]] = None) -> Dict:
        relevant = self.retrieve_relevant_chunks(query, k=5)
        if self.perplexity_api_key:
            answer = self.generate_answer_with_perplexity(query, relevant, history=history)
        else:
            answer = self.generate_fallback_answer(query, relevant)
        # Preserve the exact context ordering used to build the prompt so that
        # "(Source N, Page X)" citations in the answer can be mapped back.
        context_sources = [
            {
                "index": i + 1,
                "text": c["text"],
                "page": c["page"],
                "filename": c["filename"],
                "relevance_score": c.get("relevance_score", 0.0),
                "quote": c.get("quote", c["text"][:200].strip()),
            }
            for i, c in enumerate(relevant[: self.max_context_docs])
        ]
        sources = [
            {
                "text": c["text"],
                "page": c["page"],
                "filename": c["filename"],
                "relevance_score": c["relevance_score"],
                "quote": c.get("quote", c["text"][:200].strip()),
            }
            for c in relevant[:3]
        ]
        return {"answer": answer, "sources": sources, "context_sources": context_sources, "query": query}
