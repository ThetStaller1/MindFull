#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script directory
cd "$SCRIPT_DIR"

# Set up Python environment if needed
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Creating and activating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# Install email-validator if not already installed
pip install email-validator

# Run FastAPI app
echo "Starting MindFull API server..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload 