<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Rust-PyO3-orange?logo=rust" alt="Rust PyO3">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/CUDA-12.8-brightgreen?logo=nvidia" alt="CUDA">
</p>

<h1 align="center">🎮 AssetForge</h1>
<p align="center"><strong>Convert real-world images into game-ready digital assets.</strong></p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#pipelines">Pipelines</a> •
  <a href="#cli-reference">CLI Reference</a> •
  <a href="#web-ui">Web UI</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#development">Development</a>
</p>

---

**AssetForge** is an open-source pipeline tool that transforms photos into game-ready assets using a hybrid Python + Rust architecture and state-of-the-art ML models.

| Asset Type | Input | Pipeline |
|-----------|-------|----------|
| 🖼️ **2D Sprites** | Single photo | Background removal → Edge cleanup → SVG vectorization → Multi-size export |
| 🏛️ **3D Models (Scan)** | Multi-angle photos | SfM → MVS → Mesh → Texture → Retopology |
| 🧊 **3D Models (AI)** | Single image | TripoSR / Hunyuan3D-2 / TRELLIS → Cleanup → Texture → Export |
| ✨ **Gaussian Splats** | Photos / Video | COLMAP → 3DGS → Mesh extraction → SPZ compression |

---

## Quick Start

```bash
# One-click install (recommended)
bash scripts/install.sh

# Or manually
pip install -e ".[all]"
maturin develop --release  # Build Rust extensions

# Launch the web UI
python3 scripts/ui.py

# Or use the CLI
assetforge 2d input.png -o ./sprites --sheet --vectorize
assetforge retopo model.obj optimized.obj --ratio 0.3 --quads
assetforge info
```

---

## Pipelines

### 1. 2D Sprite Pipeline
Extract clean, game-ready sprites from photographs. Supports **outline** (silhouette) and **multi-color** vectorization.

```
Input Image → Background Removal → Edge Cleanup → Rasterize → SVG Export
                    ↓                                       ↓
       SAM2 or Rembg (with model      Optional feathering   Outline or Color
       selection & alpha matting)                           vectorization (16-32 colors)
```

```bash
# Basic usage
assetforge 2d character.png

# Full options
assetforge 2d character.png \
  -o ./sprites \
  --bg-remover rembg \
  --bg-model isnet-general-use \
  --alpha-matting \
  --sprite-sizes 32,64,128,256 \
  --sheet \
  --vectorize \
  --vectorize-mode color \
  --num-colors 24 \
  --simplify-tolerance 1.5 \
  --feather 1
```

**Output:**
- `sprite_{size}x{size}.png` — RGBA sprites at each requested size
- `spritesheet.png` + `spritesheet.json` — Packed sprite sheet (optional)
- `sprite.svg` — SVG vector paths (optional, outline or multi-color)

---

### 2. 3D Photogrammetry Pipeline
Reconstruct 3D models from multi-angle photographs.

```
Photo Set → OpenMVG → OpenMVS → Blender → Game Mesh
               │           │          │
         Feature       MVS Mesh   Retopology
         Extraction    + Texture  + Export
```

```bash
assetforge 3d-scan ./photos/ \
  -o ./model \
  --feature-type sift \
  --mesh-quality high \
  --retopology \
  --target-faces 15000 \
  --format glb
```

**External tools required:**
- [OpenMVG](https://github.com/openMVG/openMVG) — Structure from Motion
- [OpenMVS](https://github.com/cdcseacave/openMVS) — Multi-View Stereo
- [Blender](https://www.blender.org/) — Retopology & export

---

### 3. 3D Generative AI Pipeline
Generate 3D meshes from a single image using state-of-the-art AI.

```
Single Image → AI Model → Mesh Cleanup → Texture → Game Mesh
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
 TripoSR       Hunyuan3D-2      TRELLIS
 (3.5 GB)       (6 GB)          (8 GB)
```

```bash
# Auto-select best model for your GPU
assetforge 3d-gen concept_art.png -o ./model

# Specific model
assetforge 3d-gen concept_art.png \
  -o ./model \
  --model trellis \
  --pbr \
  --tex-resolution 2048 \
  --retopology \
  --format glb
```

**Model auto-selection (based on VRAM):**

| VRAM | Model | Quality | Features |
|------|-------|---------|----------|
| < 4 GB | TripoSR | ⭐⭐⭐ | Basic mesh |
| 4-8 GB | Hunyuan3D-2 | ⭐⭐⭐⭐ | Mesh + PBR textures |
| > 8 GB | TRELLIS | ⭐⭐⭐⭐⭐ | Mesh + PBR + Splats |

---

### 4. Gaussian Splatting Pipeline
Train 3D Gaussian Splatting for real-time rendering.

```
Photos/Video → COLMAP → 3DGS → Mesh Extraction → SPZ → GLTF
                   │              │                    │
             Camera Poses     MILo/SuGaR         KHR_gaussian_
                                                 splatting ext
```

```bash
# From photos
assetforge 3dgs ./photos/ -o ./splats

# From video with mesh extraction
assetforge 3dgs video.mp4 \
  -o ./splats \
  --extract-mesh \
  --mesh-method milo \
  --compress \
  --quality high
```

---

## Web UI

For an interactive experience, launch the Gradio web interface:

```bash
python3 scripts/ui.py
# → http://localhost:7860
python3 scripts/ui.py --share
# → https://xxxxx.gradio.live (public link)
```

**Four tabs provide easy access to all pipelines:**

| Tab | Features |
|-----|----------|
| 🎨 **2D Sprites** | Upload image → choose settings → preview sprites + download SVG/sheet |
| 🧊 **3D Generation** | Upload concept art → select AI model → 3D viewer + download |
| 🔧 **Retopology** | Upload mesh → set parameters → optimized mesh preview + download |
| ℹ️ **System Info** | GPU, CUDA, versions, installed backends, cached models |

---

## CLI Reference

### Global Options

| Flag | Description |
|------|-------------|
| `--verbose`, `-v` | Enable verbose logging |
| `--quiet`, `-q` | Suppress non-error output |
| `--version` | Show version |
| `--help` | Show help |

### Commands

#### `assetforge 2d`
Extract a 2D sprite from an image.

```
Usage: assetforge 2d [OPTIONS] INPUT

Arguments:
  INPUT  Input image path (JPG/PNG)

Background Removal Options:
  --bg-remover TEXT        Background removal: 'sam2' or 'rembg'  [default: rembg]
  --bg-model TEXT          Rembg model: u2net, u2netp, u2net_human_seg,
                           isnet-general-use, isnet-anime, silueta  [default: auto]
  --alpha-matting          Enable alpha matting for smoother edges (Rembg only)
  --alpha-matting-fg INT   Alpha matting foreground threshold  [default: 240]
  --alpha-matting-bg INT   Alpha matting background threshold  [default: 10]
  --alpha-matting-erode INT  Alpha matting erode size  [default: 10]

Sprite Options:
  -o, --output TEXT        Output directory  [default: ./output]
  --sprite-sizes TEXT      Comma-separated output sizes  [default: 64,128,256]
  --sheet                  Pack sprites into a sprite sheet
  --feather INTEGER        Edge feather radius in pixels  [default: 0]

Vectorization Options:
  --vectorize              Also output SVG vector paths
  --vectorize-mode TEXT    Vectorization mode: 'outline' or 'color'  [default: outline]
  --num-colors INTEGER     Number of colors for color mode (4-64)  [default: 24]
  --simplify-tolerance FLOAT  Path simplification tolerance in pixels  [default: 1.5]
  --min-area INTEGER       Minimum contour area in pixels  [default: 10]
```

#### `assetforge 3d-scan`
Reconstruct 3D model from multi-angle photos.

```
Usage: assetforge 3d-scan [OPTIONS] INPUT

Arguments:
  INPUT  Directory of multi-angle photos

Options:
  -o, --output TEXT          Output directory  [default: ./output]
  --feature-type TEXT        Feature extractor: 'sift' or 'orb'  [default: sift]
  --mesh-quality TEXT        Mesh quality: 'low', 'medium', 'high'  [default: high]
  --retopology               Apply quad remeshing for game use
  --target-faces INTEGER     Target triangle count  [default: 15000]
  --format TEXT              Output format: 'glb', 'obj', 'fbx'  [default: glb]
```

#### `assetforge 3d-gen`
Generate a 3D model from a single image using AI.

```
Usage: assetforge 3d-gen [OPTIONS] INPUT

Arguments:
  INPUT  Single input image (concept art, reference)

Options:
  -o, --output TEXT            Output directory  [default: ./output]
  --model TEXT                 AI model: 'auto', 'triposr', 'hunyuan3d-2', 'trellis'  [default: auto]
  --retopology                 Optimize mesh for game use
  --target-faces INTEGER       Target triangle count  [default: 20000]
  --pbr                        Generate PBR material maps
  --tex-resolution INTEGER     Texture resolution  [default: 2048]
  --format TEXT                Output format: 'glb', 'obj', 'fbx'  [default: glb]
```

#### `assetforge 3dgs`
Train 3D Gaussian Splatting from photos.

```
Usage: assetforge 3dgs [OPTIONS] INPUT

Arguments:
  INPUT  Photo directory or video file

Options:
  -o, --output TEXT            Output directory  [default: ./output]
  --extract-mesh               Extract surface mesh from splats
  --mesh-method TEXT           Mesh extraction: 'milo' or 'sugar'  [default: milo]
  --compress                   Compress using SPZ format
  --quality TEXT               Training quality: 'low', 'medium', 'high'  [default: medium]
  --max-splats INTEGER         Maximum number of Gaussians  [default: 1000000]
```

#### `assetforge retopo`
Optimize a mesh for game engines.

```
Usage: assetforge retopo [OPTIONS] INPUT OUTPUT

Arguments:
  INPUT   Input mesh file (OBJ/GLB/FBX)
  OUTPUT  Output mesh file (OBJ/GLB)

Options:
  --target-faces, -f INTEGER   Target triangle count  [default: 15000]
  --ratio, -r FLOAT             Decimation ratio (0.0-1.0)  [default: 0.5]
  --quads, -q                   Apply quad remeshing after decimation
  --format TEXT                 Output format: 'obj', 'glb', 'fbx'  [default: obj]
```

#### `assetforge compress`
Compress Gaussian splats to SPZ format.

```
Usage: assetforge compress [OPTIONS] INPUT OUTPUT

Arguments:
  INPUT   Input PLY file from 3DGS training
  OUTPUT  Output .spz compressed file

Options:
  --quality, -q INTEGER RANGE  Compression quality (0-100)  [default: 80]
```

#### `assetforge info`
Show system information.

```
Usage: assetforge info [OPTIONS]

Displays GPU model, VRAM, CUDA version, PyTorch version,
and status of all installed backends.
```

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  2D Pipeline     │    │  3D Pipelines    │    │  Gaussian Splat  │
│  Python + Rust   │    │ Python+Blender   │    │  Python + Rust   │
│  (PyO3 bridge)   │    │ (subprocess)     │    │  (PyO3 bridge)   │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Rust Core (PyO3)       │
                    │   Vectorization          │
                    │   Mesh Optimization      │
                    │   SPZ Compression        │
                    │   GLTF Export            │
                    └─────────────────────────┘
```

**Hybrid design: Python for ML orchestration + Rust for hot paths.**

| Layer | Technology | Role |
|-------|-----------|------|
| **CLI** | Typer (Python) | Command parsing, help, argument validation |
| **Pipelines** | Python | Stage orchestration, error handling, progress |
| **Backends** | Python | ML model wrappers, external tool subprocess calls |
| **Native** | Rust (PyO3) | Vectorization, mesh ops, splat compression, GLTF export |
| **External** | Blender, OpenMVG, OpenMVS, COLMAP | Rendering, photogrammetry, pose estimation |

### Project Structure

```
assetforge/
├── src/assetforge/          # Python package
│   ├── cli.py               # Typer CLI (7 commands)
│   ├── config.py            # Configuration management
│   ├── pipelines/           # Pipeline orchestrators
│   │   ├── base.py          # BasePipeline with stage timing
│   │   ├── pipeline_2d.py   # 2D sprite pipeline (with color vectorization)
│   │   ├── pipeline_3d_photo.py  # Photogrammetry pipeline
│   │   ├── pipeline_3d_gen.py    # Generative AI pipeline
│   │   └── pipeline_3dgs.py     # Gaussian splatting pipeline
│   ├── backends/            # External tool wrappers
│   │   ├── segmentation.py  # SAM2 + Rembg (model selection, alpha matting)
│   │   ├── vectorize.py     # Color vectorization (Python/OpenCV)
│   │   ├── blender.py       # Blender headless + scripts/
│   │   ├── openmvg.py       # OpenMVG CLI wrapper
│   │   ├── openmvs.py       # OpenMVS CLI wrapper
│   │   ├── triposr.py       # TripoSR auto-setup + inference
│   │   ├── hunyuan3d.py     # Hunyuan3D-2 wrapper
│   │   ├── trellis.py       # TRELLIS wrapper
│   │   ├── gaussian_splat.py # COLMAP + 3DGS + SuGaR/MILo
│   │   └── instant_meshes.py # Instant Meshes wrapper
│   ├── models/registry.py   # Model cache management
│   └── utils/               # GPU detection, I/O, logging
├── rust/                    # Rust workspace
│   ├── assetforge-core/     # Core algorithms
│   │   ├── src/vectorize.rs     # Bitmap→SVG tracing (outline)
│   │   ├── src/mesh_utils.rs    # Mesh simplification
│   │   ├── src/splat_compress.rs # SPZ compression
│   │   └── src/image_ops.rs     # Image processing
│   └── assetforge-pyo3/    # Python bindings
│       └── src/lib.rs       # 5 exported functions
├── scripts/                 # Utility scripts
│   ├── install.sh           # One-click install
│   ├── ui.py                # Gradio web interface
│   ├── demo.py              # End-to-end demo
│   ├── benchmark.py         # Performance benchmarks
│   └── download_models.sh   # Model weight caching
├── tests/                   # Test suite (62+ tests)
│   ├── test_vectorize.py    # Vectorization tests (31 tests)
│   ├── test_pipeline_2d.py
│   ├── test_pipeline_3d_photo.py
│   ├── test_pipeline_3d_gen.py
│   ├── test_pipeline_3dgs.py
│   └── fixtures/            # Generated test assets
└── examples/                # Sample assets
    ├── sample_2d_input.png
    ├── sample_mesh.obj
    └── sample_photo_set/    # Multi-view test images
```

---

## Backend Setup

### Required (core pipelines)

| Tool | Install | Required by |
|------|---------|-------------|
| Python 3.10+ | `apt install python3` | All pipelines |
| Blender 3.6+ | `snap install blender --classic` | Retopology, export |

### Optional (ML pipelines)

| Tool | Install | VRAM | Pipeline |
|------|---------|------|----------|
| Rembg | `pip install rembg` | 1 GB | 2D sprites (models: u2net, isnet-general-use, isnet-anime, u2net_human_seg, silueta, sam) |
| SAM 2 | `pip install git+https://github.com/facebookresearch/sam2.git` | 3 GB | 2D sprites |
| TripoSR | Auto-cloned by backend | 3.5 GB | 3D generative |
| Hunyuan3D-2 | `pip install hy3dgen` | 6 GB | 3D generative |
| TRELLIS | `pip install trellis2` | 8 GB | 3D generative |

### Optional (photogrammetry & splatting)

| Tool | Install | Pipeline |
|------|---------|----------|
| COLMAP | `apt install colmap` | 3D photogrammetry, 3DGS |
| OpenMVG | Build from source | 3D photogrammetry |
| OpenMVS | Build from source | 3D photogrammetry |
| 3DGS | `pip install gaussian-splatting` | Gaussian splatting |
| SuGaR | `pip install sugar` | Mesh from splats |
| MILo | `pip install milo` | Mesh from splats |

---

## Development

```bash
# Clone and install
git clone https://github.com/assetforge/assetforge.git
cd assetforge
pip install -e ".[all]"

# Build Rust extensions
maturin develop --release

# Verify installation
python3 -m assetforge info

# Run tests
python3 -m pytest tests/ -v

# Run Rust tests
cd rust && cargo test

# Launch UI
python3 scripts/ui.py

# Run demo
python3 scripts/demo.py

# Run benchmarks
python3 scripts/benchmark.py
```

### Test Coverage

```
62 Python tests + 9 Rust tests — all passing
```

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_base_pipeline.py` | 3 | Stage results, output dirs, timing |
| `test_pipeline_2d.py` | 9 | Basic 2D, spritesheet, invalid input, edge cleanup, color vectorization, alpha matting, bg model |
| `test_vectorize.py` | 31 | Color quantization, contour extraction, layer separation, SVG generation, edge cases |
| `test_pipeline_3d_photo.py` | 5 | Init, retopology, missing tools, output dirs, invalid input |
| `test_pipeline_3d_gen.py` | 5 | Init, options, missing model, unknown model, invalid input |
| `test_pipeline_3dgs.py` | 5 | Init, options, missing COLMAP, video input, invalid input |
| `test_utils.py` | 4 | Output dirs, input type detection, image listing |
| Rust core | 9 | Vectorization, mesh utils, image ops, splat compression |

---

## Requirements

### Minimum
- Python 3.10+
- 4 GB RAM

### Recommended for ML
- NVIDIA GPU with 4+ GB VRAM (CUDA 11.8+)
- 16 GB RAM
- Blender 3.6+

---

## Model Support

| Model | Pipeline | VRAM | Quality | Package |
|-------|----------|------|---------|---------|
| SAM 2 | 2D background removal | 3 GB | ★★★★★ | `facebookresearch/sam2` |
| Rembg (U²-Net) | 2D background removal | 1 GB | ★★★☆☆ | `rembg` (model: `u2net`) |
| Rembg (IS-Net) | 2D background removal | 1 GB | ★★★★☆ | `rembg` (model: `isnet-general-use`) |
| Rembg (IS-Net Anime) | 2D background removal | 1 GB | ★★★★☆ | `rembg` (model: `isnet-anime`) |
| Color Vectorization | 2D SVG export | CPU | ★★★★★ | Built-in (OpenCV + Python) |
| TripoSR | 3D from single image | 3.5 GB | ★★★☆☆ | Auto-cloned from GitHub |
| Hunyuan3D-2 | 3D from image + PBR | 6 GB | ★★★★☆ | `hy3dgen` |
| TRELLIS | 3D from image + PBR + Splats | 8 GB | ★★★★★ | `trellis2` |

---

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

Built on the shoulders of giants: OpenMVG, OpenMVS, COLMAP, SAM 2, Rembg, TripoSR,
Hunyuan3D-2, TRELLIS, 3D Gaussian Splatting, SuGaR, MILo, Blender, Instant Meshes,
Gradio, Typer, PyO3, and the open-source ML community.
