#!/bin/bash
# Setup test environment using uv (fast Python package installer)
# This script installs dependencies 10-100x faster than pip

set -e

echo "🚀 Setting up test environment with uv..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv (fast Python package installer)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "✅ uv installed successfully"
    echo ""
fi

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Create virtual environment
echo "🔧 Creating virtual environment..."
uv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies with uv (this is fast!)..."
uv pip install -r requirements.txt
uv pip install pytest pytest-asyncio pytest-cov
echo "✅ All dependencies installed"
echo ""

echo "🎉 Setup complete!"
echo ""
echo "To run tests:"
echo "  cd backend"
echo "  source .venv/bin/activate"
echo "  pytest test/unit_tests/ -v"
echo ""
echo "To run with coverage:"
echo "  pytest test/unit_tests/ --cov=backend/app --cov-report=html"
echo ""
echo "To run RAGAS integration tests:"
echo "  pytest test/integration_tests/test_ragas_evaluation.py -v -s"
