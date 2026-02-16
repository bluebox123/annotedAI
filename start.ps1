# RAG PDF Highlighter Startup Script
Write-Host "Starting RAG PDF Highlighter..." -ForegroundColor Green
Write-Host ""

# Function to check if a port is in use
function Test-PortInUse {
    param($Port)
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    return $connection -ne $null
}

# Check ports
if (Test-PortInUse -Port 8000) {
    Write-Host "WARNING: Port 8000 is already in use. Backend may fail to start." -ForegroundColor Yellow
}
if (Test-PortInUse -Port 5173) {
    Write-Host "WARNING: Port 5173 is already in use. Frontend may fail to start." -ForegroundColor Yellow
}

Write-Host "[1/3] Installing frontend dependencies..." -ForegroundColor Cyan
Set-Location frontend
npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: npm install failed" -ForegroundColor Red
    exit 1
}
Set-Location ..

Write-Host "[2/3] Starting FastAPI backend on http://localhost:8000..." -ForegroundColor Cyan
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD\backend
    python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
}

# Wait for backend to start
Write-Host "      Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host "[3/3] Starting React frontend on http://localhost:5173..." -ForegroundColor Cyan
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD\frontend
    npm run dev
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Application Started Successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host ""

# Keep script running and show output
try {
    while ($true) {
        $backendOutput = Receive-Job -Job $backendJob
        $frontendOutput = Receive-Job -Job $frontendJob
        
        if ($backendOutput) {
            Write-Host "[BACKEND] $backendOutput" -ForegroundColor Blue
        }
        if ($frontendOutput) {
            Write-Host "[FRONTEND] $frontendOutput" -ForegroundColor Magenta
        }
        
        Start-Sleep -Milliseconds 100
    }
} finally {
    Write-Host "`nStopping servers..." -ForegroundColor Yellow
    Stop-Job -Job $backendJob, $frontendJob
    Remove-Job -Job $backendJob, $frontendJob
    Write-Host "Servers stopped." -ForegroundColor Green
}
