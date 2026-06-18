#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# AssetForge — Model Download Script
# Downloads all required ML models to the cache directory.
# ─────────────────────────────────────────────────────
set -euo pipefail

CACHE_DIR="${ASSETFORGE_CACHE_DIR:-$HOME/.cache/assetforge/models}"
mkdir -p "$CACHE_DIR"

echo "=== AssetForge Model Download ==="
echo "Cache directory: $CACHE_DIR"
echo ""

# 1. SAM 2 (segmentation)
echo "[1/5] Downloading SAM 2 (facebook/sam2-hiera-large)..."
if [ -f "$CACHE_DIR/sam2_hiera_large.pt" ]; then
    echo "  ✓ Already cached"
else
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('facebook/sam2-hiera-large', 'sam2_hiera_large.pt', cache_dir='$CACHE_DIR')
print('  ✓ Downloaded')
" 2>/dev/null || echo "  ⚠ SKIP (run: pip install huggingface_hub && python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('facebook/sam2-hiera-large', 'sam2_hiera_large.pt', cache_dir='$CACHE_DIR')\")"
fi

# 2. Rembg (U²-Net) — auto-downloaded on first use
echo "[2/5] Rembg (U²-Net) — auto-downloaded on first use"
python -c "
import rembg
rembg.new_session()
print('  ✓ Ready')
" 2>/dev/null || echo "  ⚠ SKIP (run: pip install rembg)"

# 3. TripoSR
echo "[3/5] Downloading TripoSR (stabilityai/TripoSR)..."
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('stabilityai/TripoSR', 'model.ckpt', cache_dir='$CACHE_DIR')
print('  ✓ Model weights cached')
hf_hub_download('stabilityai/TripoSR', 'config.yaml', cache_dir='$CACHE_DIR')
print('  ✓ Config cached')
" 2>/dev/null || echo "  ⚠ SKIP (could not download TripoSR weights)"

echo "[3b/5] TripoSR inference code..."
if python -c "import tsr" 2>/dev/null; then
    echo "  ✓ Already installed"
else
    echo "  ○ Cloning TripoSR inference code..."
    git clone --depth=1 https://github.com/VAST-AI-Research/TripoSR.git "$CACHE_DIR/triposr_repo" 2>/dev/null && \
    echo "  ✓ Repo cloned" || echo "  ⚠ SKIP (could not clone repo)"
fi

# 4. Hunyuan3D-2
echo "[4/5] Downloading Hunyuan3D-2 (Tencent/Hunyuan3D-2)..."
if [ -d "$CACHE_DIR/hunyuan3d-2" ]; then
    echo "  ✓ Already cached"
else
    echo "  ⚠ SKIP — Large model. Download manually:"
    echo "     huggingface-cli download Tencent/Hunyuan3D-2 --local-dir $CACHE_DIR/hunyuan3d-2"
fi

# 5. TRELLIS 2
echo "[5/5] TRELLIS 2 (microsoft/TRELLIS) — auto-downloaded on first use"
python -c "
print('  ✓ Will download on first pipeline run')
"  2>/dev/null

echo ""
echo "=== Download complete ==="
echo ""
echo "To verify models:"
echo "  assetforge info"
