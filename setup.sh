#! /bin/bash

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Activating existing environment."
    source venv/bin/activate
else
    echo "Creating new virtual environment..."
    python -m venv venv
    source venv/bin/activate
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installation complete."
