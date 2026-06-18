# Architecture

## Design Philosophy

AssetForge uses a **hybrid Python + Rust + External Tools** architecture:

| Layer | Technology | Why |
|-------|-----------|-----|
| **CLI / UI** | Python (Typer, Gradio) | Rapid development, rich ecosystem |
| **Pipeline orchestration** | Python | Flexibility, ML integration |
| **Hot paths** | Rust (PyO3) | Performance-critical algorithms |
| **Heavy compute** | External (subprocess) | Battle-tested tools (Blender, COLMAP) |

## Python Package Structure

```
src/assetforge/
├── cli.py               # Entry point — 7 Typer commands
├── config.py            # Global config (paths, defaults)
├── pipelines/           # Pipeline orchestrators
│   ├── base.py          # BasePipeline: stage lifecycle, timing, errors
│   ├── pipeline_2d.py   # 2D sprite stages
│   ├── pipeline_3d_photo.py  # Photogrammetry stages
│   ├── pipeline_3d_gen.py    # Generative AI stages
│   └── pipeline_3dgs.py     # Gaussian splatting stages
├── backends/            # Tool wrappers (standardized interface)
│   ├── segmentation.py  # SAM2 + Rembg
│   ├── blender.py       # Blender headless
│   ├── openmvg.py       # OpenMVG CLI
│   ├── openmvs.py       # OpenMVS CLI
│   ├── triposr.py       # TripoSR auto-setup + inference
│   ├── hunyuan3d.py     # Hunyuan3D-2
│   ├── trellis.py       # TRELLIS
│   ├── gaussian_splat.py # COLMAP + 3DGS + SuGaR/MILo
│   └── instant_meshes.py # Instant Meshes
├── models/registry.py   # HF model cache, auto-download
└── utils/               # GPU detection, I/O helpers, logging
```

## Pipeline Lifecycle

Every pipeline extends `BasePipeline` and follows a standardized lifecycle:

```
1. __init__()      → Validate inputs, create output dirs
2. run()           → Execute stages sequentially
   ├── run_stage() → Call stage function with timing
   │   ├── success → Record result + output paths
   │   └── failure → Record error, continue pipeline
   └── summary()   → Print pass/fail summary
3. Return stages[] → List of StageResult objects
```

Each `StageResult` contains:
- `name`: Stage identifier
- `success`: Boolean
- `duration_s`: Wall-clock time
- `output_paths`: List of produced files
- `error`: Error message (if failed)
- `metadata`: Dict for extensible data

## Rust PyO3 Bridge

The Rust extension (`assetforge_pyo3`) exposes 5 functions to Python:

| Function | Input | Output | Used by |
|----------|-------|--------|---------|
| `vectorize_bitmap()` | Binary pixel buffer | SVG string | 2D pipeline |
| `compress_splats()` | Gaussian arrays | SPZ bytes | 3DGS pipeline, `compress` command |
| `rasterize_sprite()` | SVG content | RGBA pixels | 2D pipeline (fallback) |
| `optimize_mesh()` | Vertex/face arrays | Simplified mesh | Retopo pipeline |
| `export_splat_gltf()` | PLY/SPZ file | GLB file | 3DGS pipeline |

### Rust Workspace

```
rust/
├── Cargo.toml                    # Workspace root
├── assetforge-core/              # Core algorithms (no Python deps)
│   ├── src/vectorize.rs          # Potrace-style bitmap tracing
│   ├── src/mesh_utils.rs         # Mesh simplification, OBJ parse/export
│   ├── src/splat_compress.rs     # SPZ: quantization + zstd compression
│   └── src/image_ops.rs         # Resize, bounds detection
└── assetforge-pyo3/             # PyO3 bindings
    └── src/lib.rs                # 5 #[pyfunction] exports
```

### Performance

The Rust core handles operations that are performance-critical or
would be awkward in Python:
- **Vectorization**: Bitmap tracing uses a modified Potrace algorithm
- **Splat compression**: Quantizes Gaussian parameters, compresses with zstd
- **Mesh simplification**: Edge collapse with quadric error metrics
- **GLTF export**: Binary GLB container with KHR_gaussian_splatting extension

## External Tool Integration

### Blender (Headless)

Blender is used for mesh processing (decimate, remesh, cleanup, bake, export).
It runs as a subprocess with environment sanitization to avoid conda conflicts:

```python
# blender.py sanitizes the environment
subprocess.run([
    "blender", "--background", "--python", script_path, "--",
    input_path, output_path, ...
], env={k: v for k, v in os.environ.items()
       if not k.startswith("PYTHON") and "LD_" not in k})
```

Each Blender operation is a standalone Python script in
`backends/blender_scripts/`:

| Script | Function | Description |
|--------|----------|-------------|
| `blender_decimate.py` | `apply_decimate()` | Collapse modifier, ratio-based |
| `blender_remesh.py` | `apply_remesh()` | Quad remesh to target face count |
| `blender_cleanup.py` | `run_cleanup()` | Remove doubles, fill holes, smooth |
| `blender_bake.py` | `bake_textures()` | Cycles PBR baking (albedo, normal, roughness, metallic) |
| `blender_export.py` | `export_format()` | Export to OBJ/GLB/FBX/STL |

### ML Model Auto-Setup

For models without pip packages (TripoSR), the backend auto-clones
the repository, creates missing module fallbacks (e.g., `torchmcubes`
replaced with trimesh-based marching cubes), and downloads weights:

```python
def _ensure_repo(self):
    # 1. Check if `tsr` is importable
    # 2. If not, clone VAST-AI-Research/TripoSR to ~/.cache/assetforge/
    # 3. Add to sys.path
    # 4. Create torchmcubes fallback if missing
    # 5. Download model.ckpt + config.yaml from HuggingFace
```

## File Formats

| Format | Used for | Pipeline |
|--------|----------|----------|
| PNG | Input/output images | 2D, 3D generative |
| JPG | Input photos | Photogrammetry, 3DGS |
| SVG | Vector sprites | 2D |
| JSON | Sprite sheet metadata | 2D |
| OBJ | Mesh exchange | All 3D pipelines |
| GLB/GLTF | Game-ready 3D | All 3D pipelines |
| FBX | Game engine import | Retopo |
| PLY | Gaussian splat data | 3DGS |
| SPZ | Compressed splats | 3DGS |
