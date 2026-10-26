@echo off
title Toothbrush Automated Optical Inspection Dashboard
echo ========================================================
echo   Launching Toothbrush Quality Inspection AI Dashboard...
echo ========================================================
cd /d "%~dp0"
call .\.venv\Scripts\activate.bat
streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause
