#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# AssetForge — One-Click Install & Setup
# ─────────────────────────────────────────────────────────────────────
# This script installs all dependencies, builds native extensions,
# downloads model weights, and verifies the installation.
#
# Usage:
#   bash scripts/install.sh                # Full install
#   bash scripts/install.sh --minimal      # Core only (no ML models)
#   bash scripts/install.sh --no-rust      # Skip Rust build
#   bash scripts/install.sh --help         # Show all options
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Config ──────────────────────────────────────────────────────────
INSTALL_ML=true
BUILD_RUST=true
DOWNLOAD_MODELS=true
RUN_TESTS=true
VERBOSE=false

# ── Parse flags ─────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --minimal)  INSTALL_ML=false; DOWNLOAD_MODELS=false ;;
    --no-rust)  BUILD_RUST=false ;;
    --no-tests) RUN_TESTS=false ;;
    --no-models) DOWNLOAD_MODELS=false ;;
    --verbose)  VERBOSE=true ;;
    --help)
      echo "Usage: bash scripts/install.sh [options]"
      echo ""
      echo "Options:"
      echo "  --minimal     Install core only (skip ML models, skip downloads)"
      echo "  --no-rust     Skip Rust/PyO3 extension build"
      echo "  --no-tests    Skip post-install test run"
      echo "  --no-models   Skip model weight downloads"
      echo "  --verbose     Show full command output"
      echo "  --help        Show this help"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: bash scripts/install.sh [--minimal] [--no-rust] [--no-tests] [--no-models] [--verbose]"
      exit 1
      ;;
  esac
done

# ── Utilities ───────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

step()   { echo -e "\n${BLUE}━━━ ${1} ━━━${NC}"; }
info()   { echo -e "  ${CYAN}→${NC} $1"; }
ok()     { echo -e "  ${GREEN}✓${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()   { echo -e "  ${RED}✗${NC} $1"; }
header() { echo -e "\n${BOLD}${1}${NC}"; }

# Track elapsed time
START_TIME=$(date +%s)

# ── Banner ──────────────────────────────────────────────────────────
clear 2>/dev/null || true
echo ""
echo -e "${CYAN}  ╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}  ║${NC}          ${BOLD}AssetForge — Install & Setup${NC}            ${CYAN}║${NC}"
echo -e "${CYAN}  ║${NC}   Real-world images → game-ready assets             ${CYAN}║${NC}"
echo -e "${CYAN}  ╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. System Dependencies ──────────────────────────────────────────
step "1/7  System Dependencies"

install_system_pkgs() {
  local pkgs=()
  # Build tools
  command -v cmake  &>/dev/null || pkgs+=("cmake")
  command -v make   &>/dev/null || pkgs+=("make")
  command -v g++    &>/dev/null || pkgs+=("g++")
  command -v git    &>/dev/null || pkgs+=("git")
  command -v curl   &>/dev/null || pkgs+=("curl")
  command -v wget   &>/dev/null || pkgs+=("wget")

  # Python dev headers (needed for some native packages)
  python3-config --includes &>/dev/null || pkgs+=("python3-dev")

  # Image I/O libs
  dpkg -l libpng-dev &>/dev/null 2>&1 || pkgs+=("libpng-dev")
  dpkg -l libjpeg-dev &>/dev/null 2>&1 || pkgs+=("libjpeg-dev")
  dpkg -l libwebp-dev &>/dev/null 2>&1 || pkgs+=("libwebp-dev")

  if [ ${#pkgs[@]} -gt 0 ]; then
    info "Installing: ${pkgs[*]}"
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq "${pkgs[@]}"
      ok "System packages installed"
    else
      warn "No apt-get found — install manually: ${pkgs[*]}"
    fi
  else
    ok "All system packages already present"
  fi
}

install_system_pkgs

# Check Blender
if command -v blender &>/dev/null; then
  ok "Blender: $(blender --version 2>&1 | head -1)"
else
  warn "Blender not found. Install it for retopology/export features:"
  info "  sudo snap install blender --classic"
  info "  # or: https://www.blender.org/download/"
fi

# ── 2. Python Dependencies (Core) ───────────────────────────────────
step "2/7  Python Core Dependencies"

cd "$REPO_DIR"

# Install core dependencies from pyproject.toml
# We install in-editable mode which pulls the [project.dependencies]
info "Installing AssetForge core package (editable)..."
if $VERBOSE; then
  pip install -e . 2>&1
else
  pip install -e . -q 2>&1
fi
ok "Core package installed"

# Install additional runtime deps (not in pyproject.toml but needed)
info "Ensuring additional runtime packages..."
EXTRA_DEPS=(
  "huggingface_hub>=0.20"
  "trimesh>=4.0"
  "scipy>=1.10"
)
for dep in "${EXTRA_DEPS[@]}"; do
  pip install "$dep" -q 2>&1
done
ok "Runtime dependencies satisfied"

# ── 3. Rust / PyO3 Extension ────────────────────────────────────────
step "3/7  Rust PyO3 Native Extension"

build_rust_ext() {
  if ! command -v cargo &>/dev/null; then
    warn "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    ok "Rust installed: $(rustc --version)"
  else
    ok "Rust: $(rustc --version)"
  fi

  info "Building PyO3 extension (this may take a minute)..."
  cd "$REPO_DIR"
  if $VERBOSE; then
    maturin build --release 2>&1
  else
    maturin build --release -q 2>&1
  fi

  # Find and install the built wheel
  WHEEL=$(ls target/wheels/assetforge_pyo3-*.whl 2>/dev/null | sort -V | tail -1)
  if [ -n "$WHEEL" ]; then
    pip install "$WHEEL" --force-reinstall -q 2>&1
    ok "PyO3 extension built and installed"
  else
    # Try maturin develop if build didn't produce a wheel
    info "Trying maturin develop..."
    maturin develop --release -q 2>&1
    ok "PyO3 extension built via maturin develop"
  fi

  # Verify the extension is importable
  python3 -c "import assetforge_pyo3; print(f'  PyO3 version: {assetforge_pyo3.__version__}')" 2>&1
  ok "PyO3 extension verified"
}

if $BUILD_RUST; then
  build_rust_ext
else
  warn "Skipping Rust build (--no-rust)"
fi

# ── 4. ML Backends (Optional) ───────────────────────────────────────
step "4/7  ML Backends"

install_ml_backend() {
  local name="$1"
  local pkg="$2"
  local import_check="$3"

  info "Installing $name..."
  if pip install "$pkg" -q 2>&1; then
    if python3 -c "import $import_check" 2>/dev/null; then
      ok "$name ready"
    else
      warn "$name installed but import failed (may need additional deps)"
    fi
  else
    warn "$name install failed (may not be available on this platform)"
  fi
}

if $INSTALL_ML; then
  # rembg (2D background removal)
  install_ml_backend "Rembg (2D background removal)" "rembg>=0.7.0" "rembg"

  # SAM 2 (optional segmentation backend)
  # Install from source (not on PyPI)
  info "SAM 2 (2D segmentation from facebookresearch/sam2)..."
  if python3 -c "import sam2" 2>/dev/null; then
    ok "SAM 2 already installed"
  else
    info "Cloning SAM 2 from facebookresearch/sam2..."
    local sam2_dir="/tmp/opencode/sam2"
    if [ ! -d "$sam2_dir" ]; then
      git clone https://github.com/facebookresearch/sam2.git --depth=1 "$sam2_dir"
    fi
    SAM2_BUILD_CUDA=0 pip install -e "$sam2_dir" -q
    if python3 -c "import sam2" 2>/dev/null; then
      ok "SAM 2 installed"
    else
      warn "SAM 2 install failed. Try manually:"
      info "  pip install 'git+https://github.com/facebookresearch/sam2.git'"
    fi
  fi

  # TripoSR (3D from single image)
  info "TripoSR (3D from single image)..."
  if python3 -c "import tsr" 2>/dev/null; then
    ok "TripoSR already installed"
  else
    info "Cloning TripoSR inference code..."
    local repo_dir="$HOME/.cache/assetforge/triposr_repo"
    mkdir -p "$(dirname "$repo_dir")"
    if git clone --depth=1 https://github.com/VAST-AI-Research/TripoSR.git "$repo_dir" 2>/dev/null; then
      ok "TripoSR repo cloned"
      info "Setting up dependencies..."
      pip install omegaconf einops trimesh imageio imageio-ffmpeg -q 2>&1
      # Add repo to Python path via .pth file
      python -c "import sys; sys.path.insert(0, '$repo_dir'); open('/tmp/triposr_path.txt', 'w').write('$repo_dir')" 2>/dev/null || true
    else
      warn "Could not clone TripoSR. Install manually:"
      info "  git clone https://github.com/VAST-AI-Research/TripoSR.git"
    fi
  fi

  # Hunyuan3D-2
  info "Hunyuan3D-2 (text/image to 3D)..."
  if python3 -c "import hy3dgen" 2>/dev/null; then
    ok "Hunyuan3D-2 already installed"
  else
    warn "Not on PyPI. Install from Tencent/Hunyuan3D-2 on GitHub."
  fi

  # TRELLIS 2
  info "TRELLIS 2 (structured 3D generation)..."
  if python3 -c "import trellis2" 2>/dev/null; then
    ok "TRELLIS 2 already installed"
  else
    warn "Not on PyPI. Install from microsoft/TRELLIS on GitHub."
  fi
else
  warn "Skipping ML backend installation (--minimal)"
fi

# ── 5. Photogrammetry Tools ─────────────────────────────────────────
step "5/7  Photogrammetry Tools"

install_photogrammetry() {
  # COLMAP
  if command -v colmap &>/dev/null; then
    ok "COLMAP found: $(colmap help 2>&1 | head -1 || echo 'colmap available')"
  else
    info "Installing COLMAP..."
    if command -v apt-get &>/dev/null; then
      sudo apt-get install -y -qq colmap 2>&1 && ok "COLMAP installed" || warn "COLMAP install failed"
    else
      warn "Install COLMAP manually: sudo apt-get install colmap"
    fi
  fi

  # OpenMVG — needs source build
  if command -v openMVG_main_SfM &>/dev/null; then
    ok "OpenMVG found"
  else
    warn "OpenMVG not in package repos. Build from source:"
    info "  git clone --recursive https://github.com/openMVG/openMVG.git"
    info "  cd openMVG && mkdir build && cd build"
    info "  cmake -DCMAKE_BUILD_TYPE=Release .."
    info "  make -j\$(nproc) && sudo make install"
  fi

  # OpenMVS — needs source build
  if command -v DensifyPointCloud &>/dev/null; then
    ok "OpenMVS found"
  else
    warn "OpenMVS not in package repos. Build from source:"
    info "  git clone --recursive https://github.com/cdcseacave/openMVS.git"
    info "  cd openMVS && mkdir build && cd build"
    info "  cmake -DCMAKE_BUILD_TYPE=Release .."
    info "  make -j\$(nproc) && sudo make install"
  fi
}

install_photogrammetry

# ── 6. Download Model Weights ───────────────────────────────────────
step "6/7  Model Weights"

download_models() {
  local cache_dir="${ASSETFORGE_CACHE_DIR:-$HOME/.cache/assetforge/models}"
  mkdir -p "$cache_dir"
  info "Cache: $cache_dir"

  # SAM 2 (if available)
  if python3 -c "import sam2" 2>/dev/null; then
    info "Downloading SAM 2 weights..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam2-hiera-large', 'sam2_hiera_large.pt', cache_dir='$cache_dir')
print('  ✓ Done')
" 2>/dev/null && ok "SAM 2 weights cached" || warn "SAM 2 download skipped"
  fi

  # TripoSR
  info "Downloading TripoSR weights..."
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('stabilityai/TripoSR', 'model.ckpt', cache_dir='$cache_dir')
print('  ✓ Done')
" 2>/dev/null && ok "TripoSR weights cached" || warn "TripoSR download skipped (not available)"

  # Rembg — will auto-download on first use
  if python3 -c "import rembg" 2>/dev/null; then
    info "Warming Rembg cache (first inference auto-downloads model)..."
    python3 -c "
import rembg
sess = rembg.new_session()
print('  ✓ Rembg session ready')
" 2>/dev/null && ok "Rembg model cached" || warn "Rembg model download skipped"
  fi

  # OpenCV DNN face detection model (for character features)
  info "Pre-caching OpenCV DNN face detection model..."
  python3 -c "
import urllib.request, os
model_dir = os.path.expanduser('~/.cache/assetforge/models')
os.makedirs(model_dir, exist_ok=True)
proto = os.path.join(model_dir, 'deploy.prototxt')
model = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
if not os.path.exists(model):
    print('  ○ Downloading face detector prototxt...')
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
        proto
    )
    print('  ○ Downloading face detector model (~10 MB)...')
    urllib.request.urlretrieve(
        'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel',
        model
    )
    print('  ✓ Face detection model cached')
else:
    print('  ✓ Already cached')
" 2>/dev/null && ok "Face detection model cached" || warn "Face detection model download skipped"

  ok "Model downloads complete"
}

if $DOWNLOAD_MODELS; then
  download_models
else
  warn "Skipping model downloads (--no-models or --minimal)"
fi

# ── 7. Verify Installation ──────────────────────────────────────────
step "7/7  Verification"

# Global verification status
VERIFY_PASS=true

check_import() {
  if python3 -c "import $1" 2>/dev/null; then
    echo -e "    ${GREEN}✓${NC} $1"
  else
    echo -e "    ${RED}✗${NC} $1"
    VERIFY_PASS=false
  fi
}

echo ""
header "  Import Check:"
for mod_name in "assetforge" "assetforge_pyo3" "PIL" "cv2" "numpy" "typer" "rich" "huggingface_hub" "trimesh" "scipy"; do
  check_import "$mod_name"
done

echo ""
header "  CLI Check:"
if python3 -m assetforge --help &>/dev/null; then
  echo -e "    ${GREEN}✓${NC} assetforge CLI works"
else
  echo -e "    ${RED}✗${NC} assetforge CLI failed"
  VERIFY_PASS=false
fi

echo ""
header "  GPU Check:"
gpu_info=$(python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'CUDA available: {name} ({gb:.1f} GB)')
else:
    print('CUDA not available (running on CPU)')
" 2>&1) && echo -e "    ${GREEN}✓${NC} $gpu_info" || echo -e "    ${YELLOW}⚠${NC} Could not detect GPU"

echo ""
header "  Blender Check:"
if command -v blender &>/dev/null; then
  bver=$(blender --version 2>&1 | head -1)
  echo -e "    ${GREEN}✓${NC} $bver"
else
  echo -e "    ${YELLOW}⚠${NC} Blender not installed"
fi

# Run tests
if $RUN_TESTS; then
  echo ""
  header "  Test Suite:"
  cd "$REPO_DIR"
  if python3 -m pytest tests/ -v --tb=short 2>&1; then
    echo -e "    ${GREEN}✓${NC} All tests passed"
  else
    echo -e "    ${RED}✗${NC} Some tests failed"
    VERIFY_PASS=false
  fi
else
  echo ""
  warn "Skipping test suite (--no-tests)"
fi

echo ""
if $VERIFY_PASS; then
  ok "Installation verified successfully!"
else
  warn "Installation partially complete — some checks failed"
fi

# ── Summary ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}  ╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}  ║${NC}           ${BOLD}Install Complete${NC}  (${ELAPSED}s)              ${CYAN}║${NC}"
echo -e "${CYAN}  ╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo -e "    ${GREEN}assetforge 2d --help${NC}        — 2D sprite pipeline"
echo -e "    ${GREEN}assetforge retopo --help${NC}     — Mesh retopology"
echo -e "    ${GREEN}assetforge info${NC}              — System info"
echo -e "    ${GREEN}python3 scripts/demo.py${NC}      — End-to-end demo"
echo -e "    ${GREEN}python3 scripts/benchmark.py${NC} — Run benchmarks"
echo ""
echo -e "  ${BOLD}Output Examples:${NC}"
echo -e "    ${GREEN}assetforge 2d input.png output/ --sizes 64 128 256${NC}"
echo -e "    ${GREEN}assetforge retopo model.obj output.obj --ratio 0.3${NC}"
echo ""
echo -e "  ${BOLD}Next Steps:${NC}"
echo "    - Run 'python3 scripts/demo.py' for a full showcase"
echo "    - Run 'assetforge info' to verify GPU/CUDA"
echo "    - Install additional models for 3D pipelines (see docs)"
echo ""

# Return exit code based on verification
if $VERIFY_PASS; then
  exit 0
else
  exit 1
fi
