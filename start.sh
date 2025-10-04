#!/bin/bash

# Exoplanet Hunter AI - Quick Start Script
# NASA Space Apps Challenge 2025

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║          EXOPLANET HUNTER AI - NASA SPACE APPS 2025                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed!"
    echo ""
fi

# Check if model exists
if [ ! -f "checkpoints/best.pth" ]; then
    echo "⚠️  Warning: Model checkpoint not found!"
    echo "   Run 'python training.py' to train the model first."
    echo ""
fi

# Start the Flask application
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    STARTING WEB APPLICATION                          ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Starting Flask server..."
echo "📱 Open your browser and navigate to:"
echo ""
echo "   🏠 Home:        http://localhost:5000"
echo "   🎯 Predict:     http://localhost:5000/predict"
echo "   📊 Metrics:     http://localhost:5000/metrics"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo ""

python app.py
