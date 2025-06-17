@echo off
echo 🏠 Starting LeaseLens - AI Lease Assistant
echo =========================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo ⚠️ Virtual environment not found. Creating one...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/update dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check if .env exists
if not exist ".env" (
    echo ⚠️ .env file not found. Creating from template...
    copy .env.example .env
    echo ✅ Please edit .env file with your API keys
)

REM Start the application
echo 🚀 Starting LeaseLens application...
echo    Access at: http://localhost:8501
echo    Press Ctrl+C to stop
echo.

streamlit run streamlit_app.py

pause
