import os
import re
import json
import time
from typing import Dict, Optional, Tuple, List

import requests
import fitz  # PyMuPDF

from pdf_processor import PDFProcessor


class PerplexityHighlighter:
    def __init__(self) -> None:
        self.processor = PDFProcessor()
        self.perplexity_api_key: Optional[str] = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_url: str = "https://api.perplexity.ai/chat/completions"
        self.model: str = "sonar-pro"

        self.category_colors: Dict[str, Tuple[float, float, float]] = {
            "Main Answer": (1.0, 1.0, 0.6),       # Soft yellow (readable)
            "Context": (0.75, 0.95, 0.75),        # Soft green
            "Advanced Topic": (0.75, 0.85, 1.0),   # Soft blue
            "Supporting info": (1.0, 0.9, 0.7),    # Soft orange
            "Definition": (1.0, 0.85, 0.95),       # Soft magenta
        }

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

    def _classify_text_by_sentences_llm(self, text: str, answer_text: str) -> Dict[str, List[str]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return {"Main Answer": [], "Context": [], "Advanced Topic": [], "Supporting info": [], "Definition": []}


        max_sentences = 120
        sentences = sentences[:max_sentences]

        if not self.perplexity_api_key:

            return {"Main Answer": [], "Context": [], "Advanced Topic": [], "Supporting info": [], "Definition": []}

        categories = [
            "Main Answer",
            "Context",
            "Advanced Topic",
            "Supporting info",
            "Definition",
        ]


        instructions = (
            "You are given a list of sentences extracted from a PDF page and the final answer text. "
            "For each sentence, assign exactly one of the following labels: \n"
            "- 'Main Answer': Directly supports or states the core answer\n"
            "- 'Context': Background or setup that helps understand the answer\n"
            "- 'Advanced Topic': Technical or advanced details (jargon, equations, deeper theory)\n"
            "- 'Supporting info': Examples, evidence, or reasoning that support the answer\n"
            "- 'Definition': A formal definition (phrases like 'is a', 'defined as', 'refers to')\n\n"
            "Rules:\n"
            "- Use ONLY the provided sentences. Do NOT rewrite or paraphrase them.\n"
            "- Each sentence must appear in exactly one category.\n"
            "- Prefer 'Definition' when a sentence defines a term.\n"
            "- Prefer 'Main Answer' for sentences that best match the provided answer text.\n"
            "- Return strictly JSON with keys exactly: 'Main Answer', 'Context', 'Advanced Topic', 'Supporting info', 'Definition'.\n"
            "- Values must be arrays of the ORIGINAL sentences that belong to that category.\n"
        )

        user_payload = {
            "answer_text": (answer_text or "").strip(),
            "sentences": sentences,
            "categories": categories,
            "format": "Return ONLY JSON with the exact keys and arrays of the original sentences."
        }

        headers = {"Authorization": f"Bearer {self.perplexity_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            "max_tokens": 1000,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }

        try:
            resp = requests.post(self.perplexity_url, json=payload, headers=headers, timeout=60)
            if resp.status_code != 200:
                return {"Main Answer": [], "Context": [], "Advanced Topic": [], "Supporting info": [], "Definition": []}
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            json_text = content

            if not json_text.strip().startswith("{"):
                match = re.search(r"\{[\s\S]*\}", json_text)
                if match:
                    json_text = match.group(0)
            parsed = json.loads(json_text)

            result: Dict[str, List[str]] = {k: [] for k in categories}
            for k in categories:
                vals = parsed.get(k, [])
                clean_vals: List[str] = []
                if isinstance(vals, list):
                    for s in vals:
                        s2 = (s or "").strip()
                        if s2:
                            clean_vals.append(s2)
                result[k] = clean_vals
            return result
        except Exception:
            return {"Main Answer": [], "Context": [], "Advanced Topic": [], "Supporting info": [], "Definition": []}

    def highlight_multiple_perplexity(self, pdf_path: str, targets, answer_text: str, out_stub: Optional[str] = None, max_highlights: int = 3):
        """Highlight only the specific target snippets using LLM classification."""
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
            print(f"Failed to generate Perplexity-highlighted PDF for {pdf_path}: {e}")
            return None 