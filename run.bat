@echo off
echo ========================================
echo SHL Assessment Recommender API (LLM + RAG)
echo ========================================
echo.

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

if not exist data\catalog.json (
    echo Downloading catalog...
    mkdir data 2>nul
    python -c "import requests; r=requests.get('https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json'); open('data/catalog.json','wb').write(r.content)"
)

echo.
echo Starting API server...
echo API will be available at: http://127.0.0.1:8001
echo Health check: http://127.0.0.1:8001/health
echo.
echo Press CTRL+C to stop the server
echo.

uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload