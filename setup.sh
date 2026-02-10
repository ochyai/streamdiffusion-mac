#!/bin/bash
# StreamDiffusion for Mac — Environment Setup Script
# Tested on: macOS 14+ (Sonoma/Sequoia) with Apple Silicon (M1/M2/M3/M4)
set -e

echo "============================================"
echo " StreamDiffusion for Mac — Setup"
echo "============================================"
echo ""

# --- Check macOS and Apple Silicon ---
if [[ "$(uname)" != "Darwin" ]]; then
    echo "ERROR: This script is for macOS only."
    exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "WARNING: Apple Silicon (arm64) is required for MPS/CoreML acceleration."
    echo "  Detected: $ARCH"
    echo "  The pipeline may not work correctly on Intel Macs."
fi

OS_VERSION=$(sw_vers -productVersion)
echo "macOS version: $OS_VERSION"
echo "Architecture: $ARCH"
echo ""

# --- Check Python ---
PYTHON=""
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "ERROR: Python 3 not found."
    echo "  Install Python 3.9+ via:"
    echo "    brew install python@3.11"
    echo "  or download from https://www.python.org/"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"

# Check Python version >= 3.9
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 9 ]]; then
    echo "ERROR: Python 3.9+ is required (found $PY_VERSION)"
    exit 1
fi
echo ""

# --- Create virtual environment ---
VENV_DIR="$(cd "$(dirname "$0")" && pwd)/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "  To recreate, delete it first: rm -rf $VENV_DIR"
else
    echo "Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
    echo "  Created."
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo ""

# --- Install dependencies ---
echo "Installing Python dependencies..."
pip install --upgrade pip

# PyTorch with MPS support (nightly or stable with Metal)
echo ""
echo "[1/5] Installing PyTorch..."
pip install torch torchvision

# Hugging Face diffusers + transformers
echo ""
echo "[2/5] Installing diffusers and transformers..."
pip install diffusers transformers accelerate

# Core ML Tools
echo ""
echo "[3/5] Installing coremltools..."
pip install coremltools

# OpenCV for camera
echo ""
echo "[4/5] Installing OpenCV..."
pip install opencv-python

# NumPy
echo ""
echo "[5/5] Installing NumPy..."
pip install numpy

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Convert models to CoreML:"
echo "     python scripts/convert_models.py"
echo ""
echo "  2. Run the camera pipeline:"
echo "     python camera.py --prompt 'oil painting style'"
echo ""
