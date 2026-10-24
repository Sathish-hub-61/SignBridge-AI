# SignBridge AI Dev Launch Script (PowerShell)
# Starts backend and frontend development servers concurrently.

$BackendPort = 8000
$FrontendPort = 5173
$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Starting SignBridge AI Local Dev Servers       " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Start Backend Server
Write-Host "[1/2] Starting FastAPI Backend on port $BackendPort..." -ForegroundColor Yellow
$BackendCmd = "cd '$RepoRoot\backend'; python -m uvicorn app.main:app --reload --port $BackendPort"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd

# 2. Start Frontend Server
Write-Host "[2/2] Starting Vite Frontend on port $FrontendPort..." -ForegroundColor Yellow
$FrontendCmd = "cd '$RepoRoot\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd

Write-Host "`nServers launched in separate terminal windows." -ForegroundColor Green
Write-Host "Backend API:  http://127.0.0.1:$BackendPort/health" -ForegroundColor Green
Write-Host "Frontend App: http://localhost:$FrontendPort" -ForegroundColor Green
