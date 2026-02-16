import PyPDF2
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple


class PDFProcessor:
    def __init__(self):
        # Target characters per chunk; tune for retrieval quality
        self.chunk_size: int = 700
        self.overlap: int = 120

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Normalize spaces and non-breaking spaces
        text = text.replace("\u00a0", " ")
        # De-hyphenate line breaks like "super-\nintelligence" -> "superintelligence"
        text = re.sub(r"-\n\s*", "", text)
        # Preserve bullets, headings, punctuation; collapse excessive newlines
        text = re.sub(r"\n{2,}", "\n", text)
        # Collapse excessive spaces but keep newlines (used for chunking)
        text = re.sub(r"[\t ]{2,}", " ", text)
        return text.strip()

    def _extract_page_text_pymupdf(self, pdf_path: str, page_index: int) -> str:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_index]
            blocks = page.get_text("blocks") or []
            # Sort by y then x to preserve visual order
            blocks.sort(key=lambda b: (round(b[1]), round(b[0])))
            lines: List[str] = []
            for b in blocks:
                try:
                    x0, y0, x1, y1, text, *_ = b
                except ValueError:
                    if len(b) >= 5:
                        text = b[4]
                    else:
                        continue
                if not text:
                    continue
                # Clean within-block text but keep internal newlines for bullets
                cleaned = self._normalize_text(text)
                if cleaned:
                    lines.append(cleaned)
            page_text = "\n".join(lines)
            return self._normalize_text(page_text)
        finally:
            doc.close()

    def _extract_text_with_pymupdf(self, pdf_path: str) -> Dict[int, str]:
        page_texts: Dict[int, str] = {}
        doc = fitz.open(pdf_path)
        try:
            for i in range(len(doc)):
                page_texts[i + 1] = self._extract_page_text_pymupdf(pdf_path, i)
        finally:
            doc.close()
        return page_texts

    def _extract_text_with_pypdf2(self, pdf_path: str) -> Dict[int, str]:
        page_texts: Dict[int, str] = {}
        try:
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for idx, page in enumerate(reader.pages):
                    raw = page.extract_text() or ""
                    page_texts[idx + 1] = self._normalize_text(raw)
        except Exception as e:
            print(f"PyPDF2 extraction error: {e}")
        return page_texts

    def extract_text_with_pages(self, pdf_path: str) -> Dict[int, str]:
        try:
            page_texts = self._extract_text_with_pymupdf(pdf_path)
        except Exception as e:
            print(f"PyMuPDF extraction error: {e}; falling back to PyPDF2")
            page_texts = self._extract_text_with_pypdf2(pdf_path)

        # If any page is suspiciously empty, try to fill from PyPDF2 for that page only
        if not page_texts:
            return self._extract_text_with_pypdf2(pdf_path)

        fallback_texts = None
        for page_num, text in list(page_texts.items()):
            if len((text or "").strip()) < 15:
                if fallback_texts is None:
                    fallback_texts = self._extract_text_with_pypdf2(pdf_path)
                if page_num in fallback_texts and len(fallback_texts[page_num].strip()) > len(text.strip()):
                    page_texts[page_num] = fallback_texts[page_num]
        return page_texts

    def _split_lines_for_chunking(self, page_text: str) -> List[str]:
        if not page_text:
            return []
        # Keep bullet markers and short headings as their own units
        raw_lines = [ln.strip() for ln in page_text.splitlines()]
        lines: List[str] = []
        for ln in raw_lines:
            if not ln:
                continue
            # If line is very long, further split on sentence boundaries to aid chunking
            if len(ln) > 400:
                parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", ln) if p.strip()]
                lines.extend(parts)
            else:
                lines.append(ln)
        return lines

    def create_chunks(self, page_texts: Dict[int, str], filename: str) -> List[Dict]:
        chunks: List[Dict] = []
        for page_num, text in page_texts.items():
            if not text or len(text.strip()) < 10:
                continue
            lines = self._split_lines_for_chunking(text)
            if not lines:
                continue
            buffer = ""
            start_idx = 0
            for i, line in enumerate(lines):
                candidate = (buffer + (" " if buffer else "") + line).strip()
                if len(candidate) >= self.chunk_size:
                    if buffer:
                        chunks.append({
                            "text": buffer.strip(),
                            "page": page_num,
                            "filename": filename,
                            "chunk_id": len(chunks)
                        })
                    # start a new buffer with overlap from the tail of previous buffer
                    if self.overlap > 0 and buffer:
                        tail = buffer[-self.overlap :]
                        buffer = (tail + " " + line).strip()
                    else:
                        buffer = line
                else:
                    buffer = candidate

            if buffer:
                chunks.append({
                    "text": buffer.strip(),
                    "page": page_num,
                    "filename": filename,
                    "chunk_id": len(chunks)
                })
        return chunks

    def extract_and_chunk(self, pdf_path: str, filename: str) -> List[Dict]:
        page_texts = self.extract_text_with_pages(pdf_path)
        chunks = self.create_chunks(page_texts, filename)
        return chunks

    def find_text_coordinates(self, pdf_path: str, page_num: int, search_text: str) -> List[Dict]:
        coordinates: List[Dict] = []
        try:
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            snippet_raw = (search_text or "").strip()
            snippet_raw = re.sub(r"\s+", " ", snippet_raw)

            def _normalize_for_search(s: str) -> str:
                s = (s or "").strip()
                s = s.replace("\u00a0", " ")
                s = re.sub(r"\s+", " ", s)
                # Normalize common quotes/dashes that differ between extractors
                s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
                s = s.replace("–", "-").replace("—", "-")
                return s.strip()

            snippet = _normalize_for_search(snippet_raw)

            # Prefer a short exact search first
            if len(snippet) > 120:
                snippet = snippet[:120]

            def _add_rects(phrase: str):
                phrase = _normalize_for_search(phrase)
                if not phrase:
                    return
                for rect in page.search_for(phrase):
                    coordinates.append({
                        "x0": rect.x0,
                        "y0": rect.y0,
                        "x1": rect.x1,
                        "y1": rect.y1,
                    })

            if snippet:
                _add_rects(snippet)
                # Also try a punctuation-stripped variant (PDF text often drops punctuation)
                if not coordinates:
                    _add_rects(re.sub(r"[^A-Za-z0-9 ]+", " ", snippet))

            # If nothing found, try shorter windows (PDF extraction often changes spacing/ligatures)
            if not coordinates and snippet:
                words = [w for w in re.split(r"\s+", snippet) if w]
                # Try windows from 14 down to 3 words
                for win in (14, 12, 10, 9, 8, 7, 6, 5, 4, 3):
                    if len(words) < win:
                        continue
                    for i in range(0, len(words) - win + 1):
                        phrase = " ".join(words[i : i + win]).strip()
                        if len(phrase) < 10:
                            continue
                        _add_rects(phrase)
                        # stop early once we have a few hits (thorough but bounded)
                        if len(coordinates) >= 18:
                            break
                    if len(coordinates) >= 18:
                        break
            doc.close()
        except Exception as e:
            print(f"Error finding text coordinates: {e}")
        return coordinates
