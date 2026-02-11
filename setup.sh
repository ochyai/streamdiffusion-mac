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
# Note: Python 3.13+ is not supported by coremltools. Use Python 3.10-3.12.
PYTHON=""
for py in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$py" &>/dev/null; then
        PY_MINOR_VER=$("$py" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")
        PY_MAJOR_VER=$("$py" -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
        if [[ "$PY_MAJOR_VER" -eq 3 && "$PY_MINOR_VER" -ge 9 && "$PY_MINOR_VER" -le 12 ]]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [[ -z "$PYTHON" ]]; then
    echo "ERROR: Python 3.9-3.12 is required (coremltools does not support 3.13+)."
    echo "  Install via: brew install python@3.12"
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"
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

# PyTorch with MPS support (pinned for coremltools compatibility)
echo ""
echo "[1/5] Installing PyTorch..."
pip install 'torch>=2.4.0,<2.5.0' 'torchvision>=0.19.0,<0.20.0'

# Hugging Face diffusers + transformers
echo ""
echo "[2/5] Installing diffusers and transformers..."
pip install 'diffusers>=0.30.0,<0.31.0' transformers accelerate

# Core ML Tools (pinned to 8.x for compatibility)
echo ""
echo "[3/5] Installing coremltools..."
pip install 'coremltools>=8.1,<9.0'

# OpenCV for camera
echo ""
echo "[4/5] Installing OpenCV..."
pip install opencv-python

# NumPy
echo ""
echo "[5/5] Installing NumPy..."
pip install numpy

# --- Patch coremltools _cast bug (numpy scalar conversion) ---
echo ""
echo "[Patch] Applying coremltools fix for numpy scalar conversion..."
python -c "
import importlib, pathlib, re
spec = importlib.util.find_spec('coremltools')
if spec and spec.submodule_search_locations:
    ops = pathlib.Path(spec.submodule_search_locations[0]) / 'converters/mil/frontend/torch/ops.py'
    if ops.exists():
        src = ops.read_text()
        old = 'if not isinstance(x.val, dtype):\n            res = mb.const(val=dtype(x.val), name=node.name)'
        new = '''if not isinstance(x.val, dtype):
            val = x.val
            if hasattr(val, 'item'):
                val = val.item()
            res = mb.const(val=dtype(val), name=node.name)'''
        if 'val = x.val' in src:
            print('  Patch already applied.')
        elif old in src:
            ops.write_text(src.replace(old, new))
            print('  Patch applied.')
        else:
            print('  WARNING: Could not find target code to patch.')
"

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
