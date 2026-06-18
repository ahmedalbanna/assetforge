# Installation Guide

## Quick Install (Recommended)

```bash
git clone https://github.com/assetforge/assetforge.git
cd assetforge
bash scripts/install.sh
```

This installs everything: Python deps, Rust extensions, ML backends,
and downloads model weights. Run with `--minimal` for core only:

```bash
bash scripts/install.sh --minimal
```

## Manual Installation

### 1. System Dependencies

```bash
# Build tools
sudo apt-get install -y cmake make g++ git curl wget python3-dev

# Image libraries
sudo apt-get install -y libpng-dev libjpeg-dev libwebp-dev

# Blender (for retopology)
sudo snap install blender --classic
# Or: https://www.bilender.org/download/

# COLMAP (for photogrammetry + 3DGS)
sudo apt-get install -y colmap
```

### 2. Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install core package
pip install -e .

# Or with all optional ML dependencies
pip install -e ".[all]"
```

### 3. Rust Extension

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# Build PyO3 extension
maturin develop --release
```

### 4. Verify Installation

```bash
python3 -m assetforge info
python3 -m pytest tests/ -v
```

## Optional Backends

### 2D Background Removal

```bash
# Rembg (fast, 1 GB VRAM)
pip install rembg

# SAM 2 (higher quality, 3 GB VRAM)
pip install git+https://github.com/facebookresearch/sam2.git
```

### 3D Generative AI

```bash
# TripoSR — auto-cloned on first use
# No manual install needed; the backend handles it.

# Hunyuan3D-2
pip install hy3dgen

# TRELLIS
pip install trellis2
```

### Photogrammetry

```bash
# OpenMVG (build from source)
git clone --recursive https://github.com/openMVG/openMVG.git
cd openMVG && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install

# OpenMVS (build from source)
git clone --recursive https://github.com/cdcseacave/openMVS.git
cd openMVS && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc) && sudo make install
```

### Gaussian Splatting

```bash
# 3DGS trainer
pip install gaussian-splatting

# Mesh extraction
pip install milo    # MILo
pip install sugar   # SuGaR
```

## Docker (Coming Soon)

A Docker image with all dependencies pre-installed is planned.
For now, use the install script on a clean Ubuntu 22.04+ system.

## Troubleshooting

**`maturin build` fails:**
- Ensure Rust is installed: `rustc --version`
- Install Python headers: `sudo apt install python3-dev`
- Try: `maturin develop --release -j 1`

**Blender not found:**
- Install: `sudo snap install blender --classic`
- Or download from blender.org and add to PATH

**CUDA out of memory:**
- Use `--bg-remover rembg` instead of `sam2`
- Use `--model triposr` (lowest VRAM)
- Reduce texture resolution: `--tex-resolution 1024`
- Close other GPU processes: `nvidia-smi`

**GLTF export fails:**
- Workaround: use OBJ format instead (`--format obj`)
- This is a known conda environment conflict with Blender's ctypes
