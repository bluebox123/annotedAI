# AnnotedAI

Live site: https://annoted-ai.vercel.app/

## Overview

AnnotedAI is a PDF question-answering and highlighting application. You can upload one or more PDFs, ask questions in natural language, and get answers with source-aware previews where relevant passages are highlighted directly in the PDF.

The repository contains:

- A **React + Vite** frontend for a modern web UI
- A **FastAPI** backend that performs PDF processing, retrieval-augmented generation (RAG), and PDF highlighting
- A **Streamlit** app (`main.py`) used for experimentation and additional modes (including numerical modes)

## Key Features

- Upload and manage multiple PDFs
- Ask questions across uploaded documents
- Source-aware answers (with extracted citations when available)
- Highlighted PDF previews for the top relevant sources
- Multiple highlighting engines:
  - Keyword-based (local)
  - Perplexity LLM (optional; requires an API key)

## Tech Stack

### Frontend

- React 18
- Vite
- Tailwind CSS
- Axios
- react-pdf

### Backend

- FastAPI
- Uvicorn
- PyMuPDF and PyPDF2 for PDF processing
- sentence-transformers for embeddings
- Custom RAG engine

## Using the Hosted App

1. Open https://annoted-ai.vercel.app/
2. Upload one or more PDF files.
3. Ask a question.
4. Review the answer, sources, and highlighted PDF preview.

## Run Locally (React + FastAPI)

### Prerequisites

- Node.js 18+ (recommended)
- Python 3.10+ (recommended)

### 1) Backend setup

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

(Optional) Create a `.env` file in the repository root:

```env
PERPLEXITY_API_KEY=your_key_here
# Optional: comma-separated list of origins
CORS_ORIGINS=http://localhost:5173
```

Start the API:

```powershell
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

API docs will be available at:

- http://localhost:8000/docs

### 2) Frontend setup

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

- http://localhost:5173

### 3) One-command startup (Windows)

You can also use the helper scripts:

- `start.bat`
- `start.ps1`

These start the FastAPI backend on `http://localhost:8000` and the frontend on `http://localhost:5173`.

## API Endpoints (Backend)

- `POST /api/upload` Upload one or more PDFs
- `POST /api/ask` Ask a question (supports `highlight_engine`)
- `GET /api/files` List uploaded files
- `DELETE /api/files` Clear all files
- `GET /api/preview/{filename}` Fetch a highlighted PDF preview

## Environment Variables

- `PERPLEXITY_API_KEY`
  - Required only if you select the Perplexity highlighting engine.
- `CORS_ORIGINS`
  - Optional comma-separated allowlist for frontend origins.
  - Defaults include `http://localhost:5173`.

## Project Structure

```text
annotedAI/
  backend/
    api.py
  frontend/
    src/
  main.py
  requirements.txt
  start.bat
  start.ps1
```

## Notes

- Uploaded PDFs are stored in a temporary directory on the machine running the backend.
- The `models/` folder and `.env` are intentionally excluded from git.

## License

No license file is currently included in this repository. If you plan to open source this project, consider adding a LICENSE file before publishing.
