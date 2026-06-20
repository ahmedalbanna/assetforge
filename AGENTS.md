# AssetForge — Agent Instructions

## Build & Setup

```bash
pip install -e ".[rembg]"        # minimal (no Rust); add [all] for full ML deps
maturin develop --release         # Rust PyO3 extension — required, not optional
bash scripts/install.sh           # does both + model downloads (--minimal → no ML)
```

- Rust module: `assetforge_pyo3` — 5 functions: `vectorize_bitmap`, `compress_splats`, `rasterize_sprite`, `optimize_mesh`, `export_splat_gltf`.
- Rust source: `rust/assetforge-core/` + `rust/assetforge-pyo3/`.

## Tests

```bash
pytest tests/ -v                     # 140 tests (all fast, no model downloads)
cd rust && cargo test                # Rust tests
python tests/generate_fixtures.py    # regenerate test fixtures
```

- Use `tmp_path`, synthetic images — no external models.
- No CI, no pre-commit, no lint/formatter config.

## CLI

Entry: `assetforge` (Typer). 7 commands:

| Command | Purpose |
|---------|---------|
| `assetforge 2d` | Sprite extraction + vectorization |
| `assetforge 3d-scan` | Photogrammetry from multi-angle photos |
| `assetforge 3d-gen` | AI 3D generation from single image |
| `assetforge 3dgs` | Gaussian splatting from photos/video |
| `assetforge retopo` | Mesh decimation + quad remeshing |
| `assetforge compress` | SPZ compression for Gaussian splats |
| `assetforge info` | GPU/CUDA/model backend detection |

Key flags:
- `assetforge 2d --quality {fast,balanced,high,ultra,auto}` — AI preset replaces 10+ individual params
- `assetforge 2d --vectorize` — output SVG; `--vectorize-mode {outline,color}`
- `assetforge 2d --character-features` — face-aware vectorization (eyes, mouth, line art)
- `assetforge 3d-gen --model auto` — VRAM-based model selection: `<4GB`→TripoSR, `4-8GB`→Hunyuan3D-2, `>8GB`→TRELLIS

## Web UI

```bash
python3 scripts/ui.py              # http://localhost:7860
python3 scripts/ui.py --share       # Gradio public link
```

## Env

| Variable | Effect |
|----------|--------|
| `ASSETFORGE_CACHE_DIR` | Model/cache root (`~/.cache/assetforge`) |
| `ASSETFORCE_OUTPUT_DIR` | Default output directory |
| `ASSETFORGE_GPU_DEVICE` | CUDA device index (default 0) |
| `ASSETFORGE_NO_CUDA` | `1`/`true` → force CPU |
| `ASSETFORGE_BLENDER_PATH` | Blender binary path (default `blender`) |

## Architecture

```
src/assetforge/
├── cli.py                     # Typer entry (7 commands)
├── backends/
│   ├── segmentation.py        # SAM2 + Rembg background removal
│   ├── vectorize.py           # Color quantization + SVG generation (gap-free)
│   ├── smart_defaults.py      # Quality presets + image analysis + auto-config
│   ├── character_features.py  # Face/eye/mouth detection for enhanced SVG
│   ├── blender.py             # Headless Blender subprocess
│   ├── blender_scripts/       # Blender Python scripts run headless
│   ├── gaussian_splat.py      # COLMAP + 3DGS
│   ├── openmvg.py / openmvs.py # Photogrammetry CLIs
│   ├── triposr.py / hunyuan3d.py / trellis.py  # 3D gen models
│   └── instant_meshes.py      # Retopology
├── pipelines/
│   ├── base.py                # run_stage() / StageResult lifecycle
│   ├── pipeline_2d.py         # Sprite extraction pipeline
│   ├── pipeline_3d_gen.py     # AI 3D generation pipeline
│   ├── pipeline_3d_photo.py   # Photogrammetry pipeline
│   └── pipeline_3dgs.py       # Gaussian splatting pipeline
├── models/registry.py         # HF Hub model cache
└── utils/                     # gpu.py, io.py, logging.py
```

- Every pipeline extends `BasePipeline` → `self.run_stage(name, fn)`.
- Allowed inputs: image (≤500 MB), mesh (obj/glb/gltf/fbx/ply/stl).
- `Pipeline2D` supports `quality=` param — delegates to `smart_defaults.resolve_2d_config()` for AI-driven defaults (auto-model, auto-alpha, auto-vectorize params based on image analysis).
- SVG vectorization is now **gap-free**: nearest-color Lab assignment + MORPH_OPEN removed + gap-fill safety net (<1% coverage loss on real images).

## Dependencies

- **pip**: `typer`, `click`, `opencv-python`, `pillow`, `numpy`, `rich`
- **Optional**: `rembg` (2D); `sam2` (2D); `triposr`, `hy3dgen`, `trellis2` (3D gen); `character` (face detection)
- **External tools** (subprocess): Blender 3.6+, COLMAP, OpenMVG, OpenMVS, `instant-meshes`
