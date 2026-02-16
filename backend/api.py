from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Optional
import tempfile
import os
import shutil
from pathlib import Path
import uvicorn
import sys
import uuid
import json

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Add parent directory to Python path to import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from the project root (.env)
if load_dotenv is not None:
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv(os.path.join(project_root, ".env"))
    except Exception:
        pass

from pdf_processor import PDFProcessor
from rag_engine import RAGEngine
from simple_highlighter import SimpleHighlighter
from perplexity_highlighter import PerplexityHighlighter

app = FastAPI(title="RAG PDF Highlighter API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for uploaded PDFs and their processed data
pdf_storage: Dict[str, dict] = {}
rag_engine: Optional[RAGEngine] = None

# Track generated highlighted previews so the UI always loads the correct file
preview_storage: Dict[str, dict] = {}

# Ensure temp directory exists
TEMP_DIR = Path(tempfile.gettempdir()) / "rag_highlighter"
TEMP_DIR.mkdir(exist_ok=True)


@app.post("/api/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    """Upload one or more PDF files"""
    global rag_engine
    
    processor = PDFProcessor()
    uploaded_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            continue
            
        # Save file to temp location
        file_path = TEMP_DIR / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process PDF
        chunks = processor.extract_and_chunk(str(file_path), file.filename)
        
        pdf_storage[file.filename] = {
            'path': str(file_path),
            'chunks': chunks,
            'filename': file.filename
        }
        uploaded_files.append({
            'filename': file.filename,
            'chunks_count': len(chunks)
        })
    
    # Rebuild RAG index with all chunks
    all_chunks = []
    for pdf_data in pdf_storage.values():
        all_chunks.extend(pdf_data['chunks'])
    
    if all_chunks:
        rag_engine = RAGEngine()
        rag_engine.build_index(all_chunks)
    
    return {
        'success': True,
        'files': uploaded_files,
        'total_chunks': len(all_chunks)
    }


@app.post("/api/ask")
async def ask_question(
    question: str = Form(...),
    highlight_engine: str = Form("keyword"),  # "keyword" or "perplexity"
    restrict_context: bool = Form(True),
    history: str = Form("")
):
    """Ask a question and get answer with sources and multi-source previews"""
    global rag_engine
    
    if not rag_engine:
        raise HTTPException(status_code=400, detail="No PDFs uploaded yet")
    
    # Parse chat history (JSON array of {role, content})
    history_msgs = []
    if history:
        try:
            parsed = json.loads(history)
            if isinstance(parsed, list):
                history_msgs = parsed[-10:]
        except Exception:
            history_msgs = []

    # Get answer
    response = rag_engine.query(question, history=history_msgs)
    
    # IMPORTANT: Perplexity citations use the exact ordering of context sources.
    # Use context_sources as the authoritative list so Source N / Page Y in the
    # answer maps directly to what we display.
    sources = response.get('context_sources', []) or response.get('sources', [])
    previews = []
    
    for idx, source in enumerate(sources[:3]):  # Limit to top 3
        preview = await generate_highlighted_preview(
            source,
            response.get('answer', ''),
            highlight_engine,
            max_highlights=22  # allow more rectangles across multiple snippets
        )
        if preview:
            preview['source_index'] = idx
            preview['relevance_score'] = source.get('relevance_score', 0)
            previews.append(preview)
    
    # Set active preview to the highest relevance one
    active_preview = previews[0] if previews else None
    
    return {
        'answer': response.get('answer', ''),
        'sources': sources,
        'context_sources': response.get('context_sources', []),
        'previews': previews,
        'active_preview': active_preview
    }


async def generate_highlighted_preview(source: dict, answer: str, engine: str, max_highlights: int = 12):
    """Generate a highlighted PDF preview - highlight a few answer-focused snippets."""
    filename = source.get('filename')
    page = source.get('page', 1)
    text = source.get('text', '')
    quote = source.get('quote', '')
    
    if not filename or filename not in pdf_storage:
        return None
    
    pdf_path = pdf_storage[filename]['path']
    
    snippets = _select_top_snippets(text, answer, fallback=quote)
    if not snippets:
        return None

    # Create highlighted PDF from multiple snippets (same page)
    targets = [(page - 1, s[:280].strip()) for s in snippets if s.strip()]
    out_stub = f"preview_{filename.replace('.pdf', '')}"
    
    try:
        if engine == "perplexity":
            highlighter = PerplexityHighlighter()
            highlighted_path = highlighter.highlight_multiple_perplexity(
                pdf_path, targets, answer, out_stub=out_stub, max_highlights=max_highlights
            )
        else:
            highlighter = SimpleHighlighter()
            highlighted_path = highlighter.highlight_multiple_simple(
                pdf_path, targets, answer, out_stub=out_stub, max_highlights=max_highlights
            )
        
        if highlighted_path and os.path.exists(highlighted_path):
            preview_id = uuid.uuid4().hex
            preview_storage[preview_id] = {
                "path": highlighted_path,
                "page": int(page),
                "source_filename": filename,
            }
            return {
                'preview_id': preview_id,
                'page': int(page),
                'source_filename': filename,
                'snippet': (snippets[0][:100] + '...') if len(snippets[0]) > 100 else snippets[0]
            }
    except Exception as e:
        print(f"Highlight generation failed: {e}")
    
    return None


def _select_top_snippets(text: str, answer: str, fallback: str = "") -> List[str]:
    """Pick a few short phrases/sentences that best align with the answer."""
    import re
    fallback = (fallback or "").strip()
    if fallback:
        return [fallback]
    if not text:
        return []
    if not answer:
        return [text[:220]]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return [text[:220]]

    # keywords from answer
    ans_words = [w.lower() for w in re.findall(r"\b\w{4,}\b", answer)]
    ans_set = set(ans_words)
    if not ans_set:
        return [sentences[0][:220]]

    scored = []
    for s in sentences[:80]:
        s_words = set(w.lower() for w in re.findall(r"\b\w{4,}\b", s))
        overlap = len(s_words & ans_set)
        if overlap <= 0:
            continue
        score = overlap * 10 - max(0, len(s) - 260) * 0.02
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [s for _sc, s in scored[:6]]
    if top:
        return top

    # fallback: produce a few keyword windows from the chunk text
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return [text[:220]]
    windows: List[str] = []
    win = 26
    for i in range(0, min(len(words), 120), win):
        w = " ".join(words[i : i + win]).strip()
        if len(w) >= 40:
            windows.append(w)
        if len(windows) >= 4:
            break
    return windows or [text[:220]]


@app.get("/api/preview/{preview_id}")
async def get_preview(preview_id: str):
    """Get the exact highlighted PDF generated for a specific answer."""
    meta = preview_storage.get(preview_id)
    if not meta:
        raise HTTPException(status_code=404, detail="File not found")
    file_path = meta.get("path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.get("/api/files")
async def list_files():
    """List uploaded files"""
    return {
        'files': [
            {
                'filename': data['filename'],
                'chunks': len(data['chunks'])
            }
            for data in pdf_storage.values()
        ]
    }


@app.delete("/api/files")
async def clear_files():
    """Clear all uploaded files"""
    global pdf_storage, rag_engine, preview_storage
    
    # Clean up temp files
    for data in pdf_storage.values():
        try:
            if os.path.exists(data['path']):
                os.remove(data['path'])
        except:
            pass
    
    pdf_storage = {}
    rag_engine = None
    preview_storage = {}
    
    return {'success': True}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
