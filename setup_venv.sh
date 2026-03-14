#!/bin/bash
# Setup script for creating a virtual environment with all dependencies

set -e  # Exit on error

echo "=========================================="
echo "Setting up Virtual Environment"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.7"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.7 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# Create virtual environment
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠ Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install the package in development mode
echo "Installing mongol-ml-autocomplete package..."
pip install -e . --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt --quiet
    echo "✓ Requirements installed"
else
    echo "⚠ requirements.txt not found, installing basic dependencies..."
    pip install torch jupyter ipykernel notebook Pillow ipython matplotlib --quiet
fi

# Install Jupyter kernel for this virtual environment
echo "Installing Jupyter kernel..."
python -m ipykernel install --user --name=mongol-ml-autocomplete --display-name="Python (Mongol ML Autocomplete)"
echo "✓ Jupyter kernel installed"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Notebook, run:"
echo "  jupyter notebook"
echo ""
echo "Or to start JupyterLab, run:"
echo "  jupyter lab"
echo ""
echo "The kernel 'Python (Mongol ML Autocomplete)' will be available"
echo "in Jupyter for use with this environment."
echo ""
