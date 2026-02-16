# RAG PDF Highlighter - React + FastAPI Version

A professional, minimal UI for asking questions about PDFs with AI-powered highlighting.

## Architecture

- **Backend**: FastAPI (Python) - handles PDF processing, RAG, and highlighting
- **Frontend**: React + Vite + Tailwind CSS - modern minimal UI

## Quick Start

### Option 1: Run Both Servers (Recommended)

Simply double-click `start.bat` or run in PowerShell:

```powershell
.\start.ps1
```

This will:
1. Start FastAPI backend on http://localhost:8000
2. Start React dev server on http://localhost:5173
3. Open your browser automatically

### Option 2: Manual Start

**Terminal 1 - Backend:**
```powershell
cd backend
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## Features

✅ **Professional Minimal UI** - Clean, modern interface inspired by Google Flow Labs  
✅ **PDF Upload** - Drag & drop multiple PDFs  
✅ **Smart Q&A** - Ask questions about your documents  
✅ **Source Citations** - See exactly where answers come from  
✅ **PDF Preview Pane** - Right-side panel with auto-highlighting  
✅ **Two Highlight Engines**:
   - Keyword-based (fast, local)
   - Perplexity LLM (AI-powered, requires API key)
✅ **Smooth Animations** - Framer Motion transitions throughout  

## API Endpoints

- `POST /api/upload` - Upload PDF files
- `POST /api/ask` - Ask a question
- `GET /api/files` - List uploaded files
- `DELETE /api/files` - Clear all files
- `GET /api/preview/{filename}` - Get highlighted PDF

## Environment Variables

Create a `.env` file in the root:

```env
PERPLEXITY_API_KEY=your_key_here
```

Only needed if using Perplexity highlighting engine.

## Tech Stack

**Backend:**
- FastAPI
- PyPDF2, PyMuPDF (PDF processing)
- sentence-transformers (embeddings)
- Custom RAG engine

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Framer Motion (animations)
- react-dropzone (file upload)
- react-pdf (PDF viewer)
- Lucide React (icons)
- Axios (HTTP client)

## Project Structure

```
annotedAI/
├── backend/
│   └── api.py              # FastAPI backend
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── api.js         # API service
│   │   ├── App.jsx        # Main app
│   │   └── ...
│   ├── package.json
│   └── vite.config.js
├── start.bat              # Windows starter
├── start.ps1              # PowerShell starter
└── README.md
```

## Troubleshooting

**Port already in use?**
- Backend: Change port in `backend/api.py` (line with `uvicorn.run`)
- Frontend: Change port in `frontend/vite.config.js`

**PDF preview not showing?**
- Make sure pop-ups aren't blocked
- Check browser console for CORS errors
- Verify backend is running on port 8000

**Frontend build errors?**
```powershell
cd frontend
rm -rf node_modules
npm install
npm run dev
```

## Development

To add new features:
1. Backend logic → `backend/api.py`
2. UI components → `frontend/src/components/`
3. API calls → `frontend/src/api.js`

The frontend proxy is configured in `vite.config.js` to forward `/api` requests to the backend.
