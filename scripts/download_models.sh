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

# 2. Rembg models (U²-Net, IS-Net, etc.) — pre-cache common variants
echo "[2/5] Rembg models — pre-caching common variants..."
python -c "
import rembg
print('  ○ Pre-caching default (u2net)...')
rembg.new_session('u2net')
print('  ○ Pre-caching isnet-general-use...')
rembg.new_session('isnet-general-use')
print('  ○ Pre-caching u2net_human_seg...')
rembg.new_session('u2net_human_seg')
print('  ✓ Rembg models cached')
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

# 6. OpenCV DNN face detection (for character features)
echo "[6/6] Pre-caching OpenCV DNN face detection model..."
python -c "
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
" 2>/dev/null || echo "  ⚠ SKIP (could not download face detection model)"

echo ""
echo "=== Download complete ==="
echo ""
echo "To verify models:"
echo "  assetforge info"
