@echo off
echo Starting RAG PDF Highlighter...
echo.

REM Start backend
echo [1/2] Starting FastAPI backend on http://localhost:8000...
start "Backend" cmd /k "cd backend && python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend
echo [2/2] Starting React frontend on http://localhost:5173...
cd frontend
start "Frontend" cmd /k "npm install && npm run dev"

echo.
echo ============================================
echo Application started!
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:5173
echo ============================================
echo.
echo Close this window to stop both servers.
pause
