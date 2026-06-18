# AssetForge 🎮

**Convert real-world images into game-ready digital assets.**

AssetForge is an open-source CLI pipeline tool that transforms photos into:
- **2D Sprites** — With background removal, vectorization, and sprite sheet packing
- **3D Models** — From multi-angle photos (photogrammetry) or single images (AI)
- **Gaussian Splats** — Real-time 3D representation for cutting-edge games

Built with a **hybrid Python + Rust** architecture for both rapid prototyping and high performance.

---

## Quick Start

```bash
# Install
pip install assetforge

# Extract a 2D sprite
assetforge 2d character_photo.jpg -o ./sprites

# Generate 3D from a single concept art image
assetforge 3d-gen concept_art.png -o ./model

# Reconstruct 3D from multi-angle photos
assetforge 3d-scan ./photos/ -o ./statue

# Train Gaussian splats
assetforge 3dgs ./photos/ -o ./splats --compress
```

---

## Pipelines

### 2D Sprite Pipeline
```
Input Image → Background Removal (SAM2/Rembg) → Edge Cleanup → Vectorization → Sprite Export
```

### 3D Photogrammetry Pipeline
```
Multi-Angle Photos → OpenMVG (SfM) → OpenMVS (MVS) → Mesh → Texture → Retopology → GLTF
```

### 3D Generative AI Pipeline
```
Single Image → TripoSR / Hunyuan3D-2 / TRELLIS → Mesh Cleanup → Retopology → Texture Bake → GLTF
```

### Gaussian Splatting Pipeline
```
Photos/Video → COLMAP Poses → 3DGS Training → Mesh Extraction (MILo/SuGaR) → SPZ Compression → GLTF
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
                    │   Vectorization, Mesh    │
                    │   Compression, Export    │
                    └─────────────────────────┘
```

- **Python**: Orchestration, ML inference, CLI framework (Typer)
- **Rust (PyO3)**: Image processing, vectorization, splat compression, mesh ops
- **External (subprocess)**: Blender headless, OpenMVG, OpenMVS, COLMAP

---

## Requirements

### Minimum
- Python 3.10+
- 4 GB RAM

### Recommended for ML
- NVIDIA GPU with 4+ GB VRAM (CUDA 11.8+)
- 16 GB RAM
- Blender 3.6+ (for retopology)

### System Dependencies (optional)
```bash
# For photogrammetry
sudo apt install openmvg openmvs

# For retopology
sudo apt install blender

# For Gaussian splatting
# Install COLMAP: https://github.com/colmap/colmap
```

---

## Model Support

| Model | Pipeline | VRAM | Quality |
|-------|----------|------|---------|
| SAM 2 | 2D Background Removal | 3 GB | ★★★★★ |
| Rembg | 2D Background Removal | 1 GB | ★★★☆☆ |
| TripoSR | 3D from Image | 3.5 GB | ★★★☆☆ |
| Hunyuan3D-2 | 3D from Image + PBR | 6 GB | ★★★★☆ |
| TRELLIS 2 | 3D from Image + PBR + Splats | 8 GB | ★★★★★ |

---

## Development

```bash
# Clone and install in development mode
git clone https://github.com/assetforge/assetforge.git
cd assetforge
pip install -e ".[all]"

# Build Rust extensions
maturin develop --release

# Run tests
pytest

# Build Rust tests
cd rust && cargo test
```

---

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Built on the shoulders of giants: OpenMVG, OpenMVS, COLMAP, SAM 2, Rembg, TripoSR,
Hunyuan3D-2, TRELLIS, 3D Gaussian Splatting, SuGaR, MILo, Blender, Instant Meshes.
