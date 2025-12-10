#!/bin/bash
# Quick start script for KonkaniVani deployment

echo "🚀 Starting KonkaniVani NLP Pipeline"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run Streamlit
echo "🌐 Starting Streamlit app..."
echo "   Open http://localhost:8501 in your browser"
echo ""
streamlit run app.py
