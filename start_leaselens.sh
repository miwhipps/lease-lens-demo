#!/bin/bash
# Quick start script for LeaseLens

echo "🏠 Starting LeaseLens - AI Lease Assistant"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Creating from template..."
    cp .env.example .env
    echo "✅ Please edit .env file with your API keys"
fi

# Start the application
echo "🚀 Starting LeaseLens application..."
echo "   Access at: http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

streamlit run streamlit_app.py
