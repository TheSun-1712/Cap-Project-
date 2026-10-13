# Launch Toothbrush Quality Inspection AI Dashboard
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Launching Toothbrush Quality Inspection AI Dashboard..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

& ".\.venv\Scripts\streamlit.exe" run app.py
