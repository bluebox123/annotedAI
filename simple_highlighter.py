import os
import re
from typing import Dict, Optional, Tuple, List

import numpy as np
from pdf_processor import PDFProcessor
import time
import fitz  

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None  
    st_util = None  


class SimpleHighlighter:
    def __init__(self) -> None:
        self.processor = PDFProcessor()
        self._encoder = None
        
        self.category_colors: Dict[str, Tuple[float, float, float]] = {
            "Main Answer": (1.0, 1.0, 0.6),       # Soft yellow (readable)
            "Context": (0.75, 0.95, 0.75),        # Soft green
            "Advanced Topic": (0.75, 0.85, 1.0),   # Soft blue
            "Supporting info": (1.0, 0.9, 0.7),    # Soft orange
            "Definition": (1.0, 0.85, 0.95),       # Soft magenta
        }

    def _get_encoder(self):
        if self._encoder is None and SentenceTransformer is not None:
            try:
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._encoder = None
        return self._encoder

    def _parse_main_page_from_answer(self, rag_response: Dict) -> Tuple[Optional[str], Optional[int]]:
        answer_text = (rag_response or {}).get("answer", "") or ""
        context_sources = (rag_response or {}).get("context_sources", []) or []
        sources = (rag_response or {}).get("sources", []) or []

        by_source_index: Dict[int, Dict] = {int(c.get("index", 0)): c for c in context_sources if isinstance(c.get("index", None), int) or str(c.get("index", "")).isdigit()}

        citation_pattern = re.compile(r"\(\s*Source\s+(\d+)\s*,\s*Page\s+(\d+)\s*\)")
        citations = citation_pattern.findall(answer_text)

        if citations:
            try:
                first_src_idx = int(citations[0][0])
                cited_page = int(citations[0][1])
                ctx = by_source_index.get(first_src_idx)
                if ctx:
                    return ctx.get("filename"), cited_page
            except Exception:
                pass

        if sources:
            return sources[0].get("filename"), int(sources[0].get("page", 1))

        return None, None

    def print_related_info(self, query: str, rag_response: Dict, filename_to_path: Dict[str, str]) -> None:
        query = query or (rag_response or {}).get("query", "") or ""
        answer = (rag_response or {}).get("answer", "") or ""

        print("=== Question ===")
        print(query.strip())
        print()

        print("=== Answer (from RAG) ===")
        print(answer.strip())
        print()

        main_filename, main_page = self._parse_main_page_from_answer(rag_response)
        if not main_filename or not main_page:
            print("No main page could be determined from citations or sources.")
            return

        pdf_path = filename_to_path.get(main_filename)
        if not pdf_path or not os.path.exists(pdf_path):
            print(f"PDF path not found for: {main_filename}")
            return

        try:
            page_text_map = self.processor.extract_text_with_pages(pdf_path)
            page_text = page_text_map.get(int(main_page), "")
        except Exception as e:
            print(f"Failed to extract page text: {e}")
            return

        print(f"=== Main Page Content — {main_filename} (Page {main_page}) ===")
        print(page_text.strip())

        classifications = self._classify_text_by_sentences(page_text, answer)
        self._print_classifications(classifications, label=f"{main_filename} — Page {main_page}")

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        raw = re.split(r"(?<=[.!?])\s+|\n+|\u2022|\-\s+", text)
        sentences: List[str] = []
        for s in raw:
            s = (s or "").strip()
            if not s:
                continue
            if len(s) < 4:
                continue
            sentences.append(s)
        return sentences

    def _calc_similarity(self, a: List[str], b: List[str]) -> np.ndarray:
        encoder = self._get_encoder()
        if encoder is None or st_util is None:
            return np.zeros((len(a), len(b)))
        a_emb = encoder.encode(a, convert_to_numpy=True, normalize_embeddings=True)
        b_emb = encoder.encode(b, convert_to_numpy=True, normalize_embeddings=True)
        return np.clip(a_emb @ b_emb.T, -1.0, 1.0)

    def _classify_text_by_sentences(self, text: str, answer_text: str) -> Dict[str, List[str]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return {"Main Answer": [], "Context": [], "Advanced Topic": [], "Supporting info": [], "Definition": []}

        answer_text = (answer_text or "").strip()
        sim_to_answer = self._calc_similarity(sentences, [answer_text])[:, 0] if answer_text else np.zeros(len(sentences))

        def_score = np.array([
            1.0 if re.search(
                r"(?:"
                r"\b(?:is|are)\s+(?:a|an|the)\b|"
                r"\brefers?\s+to\b|"
                r"\bdefined\s+as\b|\bdefinition\s+of\b|"
                r"\bknown\s+as\b|\bmeans\b|\bmeaning\s+of\b|"
                r"\bis\s+called\b|\bare\s+called\b|"
                r"\bis\s+termed\b|\bare\s+termed\b"
                r")",
                s,
                re.IGNORECASE,
            ) else 0.0
            for s in sentences
        ], dtype=float)

        supp_score = np.array([
            (0.7 if re.search(r"\b(for example|e\.g\.|such as|because|therefore|due to|since)\b", s, re.IGNORECASE) else 0.0)
            + (0.3 if re.search(r"\d", s) else 0.0)
            for s in sentences
        ], dtype=float)

        adv_keywords = r"algorithm|complexit|architecture|framework|deriv|matrix|gradient|theorem|proof|optimization|Bayes|Markov|transformer|kernel|vector|manifold|eigen|regulariz|convergence|hyperparameter|distribution|probabilit|statistical|inference|latent|embedding"
        adv_score = np.array([
            (0.8 if re.search(adv_keywords, s, re.IGNORECASE) else 0.0)
            + (0.2 if (sum(1 for w in re.findall(r"[A-Za-z]+", s) if len(w) >= 10) >= 2) else 0.0)
            for s in sentences
        ], dtype=float)

        context_cues = r"background|history|overview|context|introduction|in general|initially|later|previously|trend|motivation|goal|purpose"
        ctx_score = np.array([
            (0.6 if re.search(context_cues, s, re.IGNORECASE) else 0.0)
            + (0.4 if 0.25 <= sim_to_answer[i] < 0.55 else 0.0)
            for i, s in enumerate(sentences)
        ], dtype=float)

        main_score = sim_to_answer.copy()
        if len(sentences) > 0 and np.max(main_score) <= 0.2:
            main_score = main_score * 0.0

        scores = np.vstack([
            main_score,         # Main Answer
            ctx_score,          # Context
            adv_score,          # Advanced Topic
            supp_score,         # Supporting info
            def_score,          # Definition
        ])

        labels = ["Main Answer", "Context", "Advanced Topic", "Supporting info", "Definition"]


        for i in range(len(sentences)):
            if def_score[i] >= 1.0:
                scores[:, i] = 0.0
                scores[labels.index("Definition"), i] = 1.0


        assignments = np.argmax(scores, axis=0)

        buckets: Dict[str, List[str]] = {k: [] for k in labels}
        for i, label_idx in enumerate(assignments):
            buckets[labels[label_idx]].append(sentences[i])


        for k in buckets:
            uniq = []
            seen = set()
            for s in buckets[k]:
                norm = s.strip()
                if norm and norm not in seen:
                    seen.add(norm)
                    uniq.append(norm if len(norm) <= 500 else (norm[:500] + "…"))
            buckets[k] = uniq
        return buckets

    def _print_classifications(self, classifications: Dict[str, List[str]], label: str) -> None:
        print(f"=== Classified Snippets — {label} ===")
        order = ["Main Answer", "Context", "Advanced Topic", "Supporting info", "Definition"]
        for cat in order:
            vals = classifications.get(cat, [])
            print(f"[{cat}] ({len(vals)} snippet(s))")
            for s in vals[:20]:
                print(f"- {s}")
            print()

    def highlight_multiple_simple(self, pdf_path: str, targets, answer_text: str, out_stub: Optional[str] = None, max_highlights: int = 3):
        """Highlight only the specific target snippets, not all classified sentences."""
        try:
            label = out_stub or os.path.splitext(os.path.basename(pdf_path))[0]
            targets = targets or []
            if not targets:
                return None

            doc = fitz.open(pdf_path)
            try:
                total_marks = 0
                # Only highlight the specific targets (max_highlights == max rectangles)
                for idx, (p0, snippet) in enumerate(targets):
                    if total_marks >= max_highlights:
                        break
                    if not snippet:
                        continue
                    
                    try:
                        page = doc[p0]
                        # Find coordinates for this specific snippet
                        coords = self.processor.find_text_coordinates(pdf_path, p0, snippet)
                        if not coords:
                            continue
                        
                        color = self.category_colors.get("Main Answer", (1.0, 0.95, 0.2))
                        
                        for rect_dict in coords:
                            if total_marks >= max_highlights:
                                break
                            r = fitz.Rect(rect_dict["x0"], rect_dict["y0"], rect_dict["x1"], rect_dict["y1"])
                            # Expand slightly so highlight doesn't touch text
                            r = r + (-1, -1, 1, 1)
                            page.draw_rect(
                                r,
                                fill=color,
                                fill_opacity=0.20,
                                color=color,
                                stroke_opacity=0.50,
                                stroke_width=0.5,
                                overlay=True
                            )
                            total_marks += 1
                    except Exception as e:
                        print(f"Error highlighting on page {p0}: {e}")
                        continue

                os.makedirs("highlighted_pdfs", exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join("highlighted_pdfs", f"highlighted_{label}_{timestamp}.pdf")
                doc.save(out_path)
            finally:
                doc.close()

            return out_path if os.path.exists(out_path) else None
        except Exception as e:
            print(f"Failed to generate highlighted PDF for {pdf_path}: {e}")
            return None
