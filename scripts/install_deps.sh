#!/usr/bin/env bash
# =============================================================================
# AssetForge — Complete Dependency Installer
#
# Installs all required dependencies for ALL pipelines in one shot.
# Run this once after cloning the repo.
#
# Usage:
#   bash scripts/install_deps.sh
#
# Options:
#   --help             Show this help
#   --python PYTHON    Python executable (default: python3)
#   --cuda             Force CUDA dependencies
#   --no-trellis       Skip TRELLIS 2 installation (large ~8GB)
#   --no-hunyuan       Skip Hunyuan3D-2 installation
#   --verbose          Verbose pip output
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python3}"
VERBOSE=""
INSTALL_TRELLIS=true
INSTALL_HUNYUAN=true

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help) head -20 "$0"; exit 0 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --verbose) VERBOSE="--verbose"; shift ;;
        --no-trellis) INSTALL_TRELLIS=false; shift ;;
        --no-hunyuan) INSTALL_HUNYUAN=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════╗"
echo "║     AssetForge — Dependency Installer        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Python: $($PYTHON --version 2>&1)"
echo "Pip:    $($PYTHON -m pip --version 2>&1)"

# ── Step 0: Upgrade pip ─────────────────────────────────────────────────────
echo ""
echo "━━━ Step 0: Upgrading pip ─────────────────────"
$PYTHON -m pip install --upgrade pip $VERBOSE

# ── Step 0b: Fix torchaudio version ─────────────────────────────────────────
echo ""
echo "━━━ Step 0b: Fix torchaudio version ────────────"
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
if [ -n "$TORCH_VER" ]; then
    # Pin torchaudio to match torch version (avoids ABI mismatch)
    $PYTHON -m pip install --upgrade "torchaudio==$TORCH_VER" $VERBOSE
fi

# ── Step 1: Core dependencies ──────────────────────────────────────────────
echo ""
echo "━━━ Step 1: Core dependencies ──────────────────"
$PYTHON -m pip install \
    torch torchvision torchaudio \
    typer \
    pillow \
    numpy \
    omegaconf \
    einops \
    trimesh \
    scikit-image \
    imageio \
    imageio-ffmpeg \
    rembg \
    huggingface_hub \
    transformers>=4.48 \
    xatlas \
    gradio \
    $VERBOSE

# ── Step 2: Fix numpy ABI ──────────────────────────────────────────────────
echo ""
echo "━━━ Step 2: Fix numpy ABI compatibility ────────"
NUMPY_VER=$($PYTHON -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "latest")
$PYTHON -m pip install --upgrade --force-reinstall "numpy==$NUMPY_VER" $VERBOSE

# ── Step 3: Install Rust/PyO3 extension ────────────────────────────────────
echo ""
echo "━━━ Step 3: Building Rust PyO3 extension ───────"
cd "$REPO_DIR"
if command -v cargo &>/dev/null; then
    $PYTHON -m pip install -e ".[dev]" $VERBOSE
else
    echo "⚠  cargo not found — skipping Rust build. Install from: https://rustup.rs"
    $PYTHON -m pip install -e . $VERBOSE
fi

# ── Step 4: Optional — TRELLIS 2 ──────────────────────────────────────────
if [ "$INSTALL_TRELLIS" = true ]; then
    echo ""
    echo "━━━ Step 4: Installing TRELLIS 2 ──────────────"
    echo "     (~8GB model, may take a while)"
    $PYTHON -m pip install trellis2 $VERBOSE
fi

# ── Step 5: Optional — Hunyuan3D-2 ────────────────────────────────────────
if [ "$INSTALL_HUNYUAN" = true ]; then
    echo ""
    echo "━━━ Step 5: Installing Hunyuan3D-2 ────────────"
    $PYTHON -m pip install hy3dgen $VERBOSE
fi

# ── Step 6: Verify installation ────────────────────────────────────────────
echo ""
echo "━━━ Step 6: Verifying installation ──────────────"
$PYTHON -c "
import sys
missing = []
# Core
for mod in ['torch', 'typer', 'PIL', 'numpy', 'trimesh', 'rembg', 'huggingface_hub', 'omegaconf', 'einops', 'imageio']:
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f'✗ Missing core packages: {missing}')
    sys.exit(1)
print('✓ Core packages OK')

# GPU check
import torch
if torch.cuda.is_available():
    print(f'✓ GPU available: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
else:
    print('⚠ No GPU detected — running on CPU (slow)')

# Backends
for name, pkg, imp in [
    ('TripoSR', None, 'tsr'),
    ('TRELLIS 2', 'trellis2', 'trellis2'),
    ('Hunyuan3D-2', 'hy3dgen', 'hy3dgen'),
]:
    try:
        __import__(imp)
        print(f'✓ {name} installed')
    except ImportError:
        print(f'  {name}: not installed (auto-setup at runtime)')
"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  Installation complete!                      ║"
echo "║                                              ║"
echo "║  Try:  assetforge 3d-gen input.png           ║"
echo "╚══════════════════════════════════════════════╝"
