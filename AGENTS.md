# AssetForge — Agent Instructions

## Build & Setup

```bash
# First install — do NOT reorder
pip install -e ".[all]"         # Python package (editable)
maturin develop --release       # Rust PyO3 extension

# Alternative: install.sh does both plus model downloads
bash scripts/install.sh
# --minimal  → core only, no ML; --no-rust → skip Rust build
```

- Rust extensions are required for all pipelines — `maturin develop --release` is **not optional**.
- The PyO3 module is named `assetforge_pyo3` (importable in Python after build).
- Rust workspace: `rust/assetforge-core/` (algorithms) + `rust/assetforge-pyo3/` (bindings).

## Tests

```bash
pytest tests/ -v               # 62+ Python tests
cd rust && cargo test           # 9 Rust tests
pytest tests/ -m "not slow"    # Skip model-download-dependent tests
```

- Tests use `tmp_path` fixtures and synthetic images — no external models required by default.
- `@pytest.mark.slow` markers exist for tests that download model checkpoints.
- Fixtures live in `tests/fixtures/` — regenerate with `python tests/generate_fixtures.py`.
- No CI workflows, no pre-commit hooks, no lint/formatter config exist in this repo.

## CLI

Entry point: `assetforge` (Typer). Invocation: `python -m assetforge` or the installed script.

| Command | Purpose |
|---------|---------|
| `assetforge 2d` | Sprite extraction (background removal → vectorization → export) |
| `assetforge 3d-scan` | Photogrammetry from multi-angle photos |
| `assetforge 3d-gen` | AI 3D generation from single image |
| `assetforge 3dgs` | Gaussian splatting from photos/video |
| `assetforge retopo` | Mesh decimation + quad remeshing |
| `assetforge compress` | SPZ compression for Gaussian splats |
| `assetforge info` | GPU/CUDA/model backend detection |

- `--bg-remover` defaults to `rembg` in CLI (not `sam2` despite config default).
- `assetforge 3d-gen --model auto` selects model based on VRAM: `<4GB`→TripoSR, `4-8GB`→Hunyuan3D-2, `>8GB`→TRELLIS.

## Web UI

```bash
python3 scripts/ui.py        # http://localhost:7860
python3 scripts/ui.py --share  # Gradio public link
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `ASSETFORGE_CACHE_DIR` | Model/cache root (`~/.cache/assetforge`) |
| `ASSETFORGE_OUTPUT_DIR` | Default output directory |
| `ASSETFORGE_GPU_DEVICE` | CUDA device index (default 0) |
| `ASSETFORGE_NO_CUDA` | Set to `1`/`true` to force CPU |
| `ASSETFORGE_BLENDER_PATH` | Blender binary path (default `blender`) |

## Models

- Auto-downloaded to `~/.cache/assetforge/models/` on first use.
- Pre-cache: `bash scripts/download_models.sh`.
- SAM 2 clones from `facebookresearch/sam2` into `/tmp/opencode/sam2/` during `install.sh`.
- Rembg models auto-download on first `rembg.new_session()`.
- TripoSR code is cloned to `~/.cache/assetforge/triposr_repo/` at install time.

## Architecture

```
src/assetforge/
├── cli.py                     # Typer entry (7 commands)
├── config.py                  # ASSETFORGE_* env var overrides
├── pipelines/                 # BasePipeline → stage lifecycle
│   ├── base.py                #   run_stage(), StageResult dataclass
│   ├── pipeline_2d.py
│   ├── pipeline_3d_photo.py
│   ├── pipeline_3d_gen.py
│   └── pipeline_3dgs.py
├── backends/                  # External tool wrappers
│   ├── segmentation.py        #   SAM2 + Rembg
│   ├── blender.py             #   Headless Blender subprocess
│   ├── openmvg/mvs.py         #   Photogrammetry CLIs
│   ├── triposr/hunyuan3d/trellis.py  # 3D gen models
│   ├── gaussian_splat.py      #   COLMAP + 3DGS + SuGaR/MILo
│   ├── character_features.py  #   Face/eye/mouth detection
│   └── blender_scripts/       #   Blender Python scripts run headless
├── models/registry.py         # HF Hub model cache management
└── utils/                     # gpu.py, io.py, logging.py
```

- Every pipeline extends `BasePipeline` and returns `list[StageResult]`.
- Stages use `self.run_stage(name, fn)` → auto-timed, error-wrapped.
- Allowed inputs: image (png/jpg/webp/bmp/tiff) ≤ 500 MB, mesh (obj/glb/gltf/fbx/ply/stl).

## Rust → Python bridge

Five functions exported via `assetforge_pyo3`:
`vectorize_bitmap`, `compress_splats`, `rasterize_sprite`, `optimize_mesh`, `export_splat_gltf`.

## Dependencies

- **Required**: `typer`, `click`, `opencv-python`, `pillow`, `numpy`, `rich`
- **Optional extras**: `sam2`, `rembg` (2D); `triposr`, `hy3dgen` (`hunyuan3d`), `trellis2` (3D gen); `character` (face detection)
- **External tools** (subprocess, not pip): Blender 3.6+, COLMAP, OpenMVG, OpenMVS
- `all` extra installs everything: `pip install -e ".[all]"` then manually install sam2/triposr/hy3dgen/trellis2 (some are GitHub-only).
