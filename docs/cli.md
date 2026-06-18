# CLI Reference

## Global Options

These flags apply to all commands:

| Flag | Description |
|------|-------------|
| `--verbose`, `-v` | Enable verbose/debug logging |
| `--quiet`, `-q` | Suppress all non-error output |
| `--version` | Print version and exit |
| `--install-completion` | Install shell completion |
| `--show-completion` | Show shell completion script |
| `--help` | Show help message and exit |

---

## `assetforge 2d` — 2D Sprite Pipeline

Extract a clean, game-ready 2D sprite from a photograph.

### Usage

```bash
assetforge 2d [OPTIONS] INPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | Input image (PNG/JPG) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | `PATH` | `./output` | Output directory |
| `--bg-remover` | `str` | `rembg` | Background removal model: `sam2` or `rembg` |
| `--sprite-sizes` | `str` | `64,128,256` | Comma-separated output sprite sizes |
| `--sheet` | `flag` | `False` | Pack sprites into a spritesheet |
| `--vectorize` | `flag` | `False` | Generate SVG vector paths |
| `--feather` | `int` | `0` | Edge feather radius in pixels |

### Examples

```bash
# Simple sprite extraction
assetforge 2d character.png

# Full pipeline with all options
assetforge 2d character_photo.jpg \
  --output ./sprites \
  --bg-remover sam2 \
  --sprite-sizes 32,64,128,256 \
  --sheet \
  --vectorize \
  --feather 2

# Quick batch-style (just one size)
assetforge 2d icon.png --sprite-sizes 128
```

### Output Files

- `stages/01_foreground.png` — Background removed
- `stages/02_cleaned.png` — After edge cleanup
- `sprite_{size}x{size}.png` — Final sprites at each size
- `output.svg` — SVG vector paths (if `--vectorize`)
- `spritesheet.png` + `spritesheet.json` — Sprite sheet (if `--sheet`)

---

## `assetforge 3d-scan` — 3D Photogrammetry

Reconstruct a 3D model from multi-angle photographs.

### Usage

```bash
assetforge 3d-scan [OPTIONS] INPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | Directory of multi-angle photos |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | `PATH` | `./output` | Output directory |
| `--feature-type` | `str` | `sift` | Feature extractor: `sift` or `orb` |
| `--mesh-quality` | `str` | `high` | Quality: `low`, `medium`, `high` |
| `--retopology` | `flag` | `False` | Apply quad remeshing |
| `--target-faces` | `int` | `15000` | Target triangle count |
| `--format` | `str` | `glb` | Output: `glb`, `obj`, `fbx` |

### Requirements

Requires external tools: **OpenMVG**, **OpenMVS**, **Blender**.
Install them first (see [installation guide](installation.md)).

### Examples

```bash
# Basic reconstruction
assetforge 3d-scan ./photos/

# Game-ready model with optimization
assetforge 3d-scan ./photos/ \
  --retopology \
  --target-faces 10000 \
  --format fbx

# Draft quality for quick preview
assetforge 3d-scan ./photos/ \
  --mesh-quality low \
  --feature-type orb
```

---

## `assetforge 3d-gen` — 3D Generative AI

Generate a 3D model from a single image using AI.

### Usage

```bash
assetforge 3d-gen [OPTIONS] INPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | Input image (concept art, reference photo) |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | `PATH` | `./output` | Output directory |
| `--model` | `str` | `auto` | AI model: `auto`, `triposr`, `hunyuan3d-2`, `trellis` |
| `--retopology` | `flag` | `False` | Optimize mesh for game use |
| `--target-faces` | `int` | `20000` | Target triangle count |
| `--pbr` | `flag` | `False` | Generate PBR material maps |
| `--tex-resolution` | `int` | `2048` | Texture resolution |
| `--format` | `str` | `glb` | Output: `glb`, `obj`, `fbx` |

### Model Selection

When `--model auto` (default), the best model is selected based on VRAM:

| VRAM | Selected Model |
|------|---------------|
| < 4 GB | TripoSR |
| 4-8 GB | Hunyuan3D-2 |
| > 8 GB | TRELLIS |

### Examples

```bash
# Auto-select model
assetforge 3d-gen concept_art.png

# High quality with PBR textures
assetforge 3d-gen ref_image.png \
  --model trellis \
  --pbr \
  --tex-resolution 4096 \
  --retopology

# Fast draft
assetforge 3d-gen sketch.png \
  --model triposr \
  --format obj
```

---

## `assetforge 3dgs` — Gaussian Splatting

Train 3D Gaussian Splatting from photos or video.

### Usage

```bash
assetforge 3dgs [OPTIONS] INPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | Photo directory or video file |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-o, --output` | `PATH` | `./output` | Output directory |
| `--extract-mesh` | `flag` | `False` | Extract surface mesh from splats |
| `--mesh-method` | `str` | `milo` | Extraction: `milo` or `sugar` |
| `--compress` | `flag` | `False` | Compress using SPZ format |
| `--quality` | `str` | `medium` | Training quality: `low`, `medium`, `high` |
| `--max-splats` | `int` | `1000000` | Maximum Gaussian count |

### Examples

```bash
# From photos
assetforge 3dgs ./photos/ --compress

# From video with mesh extraction
assetforge 3dgs scan.mp4 \
  --extract-mesh \
  --mesh-method sugar \
  --quality high

# Draft splats (quick test)
assetforge 3dgs ./photos/ --quality low --max-splats 100000
```

---

## `assetforge retopo` — Mesh Retopology

Optimize a mesh for game engines — decimate, remesh, and export.

### Usage

```bash
assetforge retopo [OPTIONS] INPUT OUTPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | Input mesh (OBJ/GLB/FBX) |
| `OUTPUT` | `PATH` | Output mesh file |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-f, --target-faces` | `int` | `15000` | Target triangle count |
| `-r, --ratio` | `float` | `0.5` | Decimation ratio (0.0-1.0) |
| `-q, --quads` | `flag` | `False` | Apply quad remeshing |
| `--format` | `str` | `obj` | Output: `obj`, `glb`, `fbx` |

### Examples

```bash
# Simple decimation
assetforge retopo high_poly.obj game_ready.obj --ratio 0.3

# With quad remeshing for animation
assetforge retopo scan.obj character.obj --quads --target-faces 5000

# Export as GLTF
assetforge retopo model.obj model.glb --format glb
```

---

## `assetforge compress` — Splat Compression

Compress Gaussian splats to SPZ format.

### Usage

```bash
assetforge compress [OPTIONS] INPUT OUTPUT
```

### Arguments

| Arg | Type | Description |
|-----|------|-------------|
| `INPUT` | `PATH` | PLY file from 3DGS training |
| `OUTPUT` | `PATH` | Output `.spz` file |

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-q, --quality` | `int` | `80` | Compression quality (0-100) |

### Example

```bash
assetforge compress splat.ply splat.spz --quality 85
```

---

## `assetforge info` — System Information

Display system info: GPU, versions, installed backends.

### Usage

```bash
assetforge info
```

### Example Output

```
╭──────────────────────────────────────────╮
│           AssetForge System Info          │
├──────────────────────────────────────────┤
│ GPU:      Tesla T4 (15.6 GB VRAM)        │
│ CUDA:     12.8                           │
│ PyTorch:  2.8.0                          │
│ Python:   3.12.11                        │
├──────────────────────────────────────────┤
│ Backends:                                │
│  ✓ Blender 4.0.2                         │
│  ✓ Rembg 2.0.76                          │
│  ✓ COLMAP                                │
│  ✗ OpenMVG — not found                   │
│  ✗ OpenMVS — not found                   │
│  ✗ Instant Meshes — not found            │
╰──────────────────────────────────────────╯
```

---

## Shell Completion

Install shell completion for TAB autocomplete:

```bash
# Bash
assetforge --install-completion bash

# Zsh
assetforge --install-completion zsh

# Fish
assetforge --install-completion fish
```
