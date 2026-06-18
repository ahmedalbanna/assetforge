# Web UI Guide

AssetForge includes a **Gradio web interface** for visually interacting with all
pipelines without using the command line.

## Launch

```bash
# Basic (http://localhost:7860)
python3 scripts/ui.py

# Custom port
python3 scripts/ui.py --port 8080

# Public share link (for remote access / sharing)
python3 scripts/ui.py --share

# Listen on all interfaces (for local network access)
python3 scripts/ui.py --host 0.0.0.0
```

## Tabs

### 🎨 2D Sprites
Upload an image and configure:
- **Background Remover**: `rembg` (fast, low VRAM) or `sam2` (higher quality, needs GPU)
- **Sprite Sizes**: Comma-separated pixel sizes (e.g., `64,128,256`)
- **Pack Spritesheet**: Combine all sprites into a single row
- **Generate SVG**: Export vector paths
- **Edge Feather**: Smooth the sprite border (0 = sharp, 5 = very soft)

After running, preview sprites in the gallery and download SVG + spritesheet.

### 🧊 3D Generation
Upload a concept art image and select an AI model:
- **auto**: Best model for your GPU VRAM
- **triposr**: Fast, low VRAM, basic mesh
- **hunyuan3d-2**: Mesh + PBR textures (requires 6 GB)
- **trellis**: Highest quality mesh + PBR + splats (requires 8 GB)
- **PBR Textures**: Enable for physically-based rendering materials

The generated model appears in the 3D viewer (click + drag to orbit).

### 🔧 Retopology
Upload a mesh file and configure:
- **Target Faces**: Desired triangle count
- **Decimation Ratio**: 0.01 = max reduction, 1.0 = no change
- **Quad Remeshing**: Convert triangles to quads for better deformation
- **Output Format**: OBJ, GLB, or FBX

### ℹ️ System Info
Auto-refreshing dashboard showing:
- GPU model, VRAM, CUDA version
- Python, PyTorch, Gradio, AssetForge versions
- Installed backend status (✅/⬜)
- Cached model weights

## Troubleshooting

**UI won't start:**
- Ensure Gradio is installed: `pip install gradio`
- Check port availability: `lsof -i :7860`

**Pipeline fails silently:**
- Check the terminal output for error messages
- Open browser developer console (F12) for network errors

**3D viewer shows nothing:**
- Ensure the model was generated successfully (check status message)
- Try a smaller model or different format
