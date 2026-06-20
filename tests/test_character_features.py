"""
Tests for character feature detection and feature-aware vectorization.

Tests cover:
- Face detection on synthetic face-like images
- Eye and mouth region refinement
- Feature mask generation
- Bounding box utilities
- Character-aware vectorization config
- Line art extraction
- Procedural eye highlights
- SVG output with character features
- Edge cases (no face, transparent, tiny images)
"""

import pytest
import numpy as np
import cv2

from assetforge.backends.character_features import (
    CharacterFeatures,
    DetectionMethod,
    _detect_faces,
    _detect_faces_dnn,
    _detect_faces_haar,
    _refine_eye_regions,
    _refine_mouth_region,
    _generate_face_mask,
    _generate_eye_mask,
    _generate_mouth_mask,
    _bbox_iou,
    detect_character_features,
    enhance_feature_masks,
    visualize_features,
)
from assetforge.backends.vectorize import (
    ColorVectorizationConfig,
    _validate_config,
    vectorize_color,
    vectorize_color_character_aware,
    _extract_line_art,
    _contours_to_svg_strokes,
    _add_eye_highlight_paths,
    _vectorize_feature_region,
)


# ── Helper Functions ──────────────────────────────────────────────


def _make_face_like_image(
    h: int = 120,
    w: int = 100,
) -> np.ndarray:
    """Create a synthetic RGB image with face-like features.

    Draws an elliptical face with dark eye spots and a mouth line
    for testing detection algorithms.
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # Light gray background

    # Face ellipse (slightly darker)
    center = (w // 2, h // 2)
    axes = (w // 3, h // 3)
    cv2.ellipse(img, center, axes, 0, 0, 360, (220, 200, 180), -1)

    # Left eye (dark spot)
    eye_y = h // 2 - h // 8
    eye_x1 = w // 2 - w // 6
    cv2.circle(img, (eye_x1, eye_y), max(2, w // 20), (50, 50, 50), -1)

    # Right eye (dark spot)
    eye_x2 = w // 2 + w // 6
    cv2.circle(img, (eye_x2, eye_y), max(2, w // 20), (50, 50, 50), -1)

    # Mouth (horizontal line/ellipse)
    mouth_y = h // 2 + h // 6
    cv2.ellipse(
        img,
        (w // 2, mouth_y),
        (w // 8, h // 20),
        0,
        0,
        180,
        (80, 60, 60),
        -1,
    )

    return img


def _make_test_image_rgba(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a synthetic RGBA image with colored shapes."""
    img = np.zeros((h, w, 4), dtype=np.uint8)

    # Red square in top-left
    img[8:24, 8:24] = [255, 0, 0, 255]

    # Green square in center
    img[20:36, 20:36] = [0, 255, 0, 255]

    # Blue square in bottom-right
    img[32:56, 32:56] = [0, 0, 255, 255]

    # Semi-transparent yellow
    img[40:56, 8:24] = [255, 255, 0, 128]

    return img


# ── CharacterFeatures Dataclass ──────────────────────────────────


def test_character_features_defaults():
    """Test CharacterFeatures default values."""
    cf = CharacterFeatures()
    assert cf.face_detected is False
    assert cf.face_bbox is None
    assert cf.feature_masks == {}
    assert cf.face_landmarks is None
    assert cf.face_confidence == 0.0
    assert cf.num_faces == 0
    assert cf.eye_regions == []
    assert cf.mouth_region is None


def test_character_features_has_feature():
    """Test has_feature method."""
    cf = CharacterFeatures()
    assert cf.has_feature("face") is False

    cf.feature_masks["face"] = np.zeros((10, 10), dtype=np.uint8)
    assert cf.has_feature("face") is False  # All zeros

    cf.feature_masks["face"][3:7, 3:7] = 255
    assert cf.has_feature("face") is True


def test_character_features_get_feature_bbox():
    """Test get_feature_bbox method."""
    cf = CharacterFeatures()
    assert cf.get_feature_bbox("face") is None

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:80] = 255
    cf.feature_masks["face"] = mask

    bbox = cf.get_feature_bbox("face")
    assert bbox is not None
    x, y, w, h = bbox
    assert x == 30
    assert y == 20
    assert w == 50
    assert h == 40


# ── Bounding Box IoU ─────────────────────────────────────────────


def test_bbox_iou():
    """Test bounding box IoU calculation."""
    bbox1 = (10, 10, 20, 20)  # (x, y, w, h)
    bbox2 = (10, 10, 20, 20)  # Same
    assert _bbox_iou(bbox1, bbox2) == pytest.approx(1.0)

    bbox2 = (30, 10, 20, 20)  # Adjacent, no overlap
    assert _bbox_iou(bbox1, bbox2) == pytest.approx(0.0)

    bbox2 = (10, 10, 10, 10)  # Smaller, fully inside (area 100)
    # Box1 area=400, Box2 area=100, intersection=100, union=400
    # IoU = 100/400 = 0.25
    assert _bbox_iou(bbox1, bbox2) == pytest.approx(0.25)

    bbox2 = (15, 15, 20, 20)  # Partial overlap
    iou = _bbox_iou(bbox1, bbox2)
    assert 0 < iou < 1.0


# ── Mask Generation ──────────────────────────────────────────────


def test_generate_face_mask():
    """Test elliptical face mask generation."""
    mask = _generate_face_mask((100, 100), (10, 20, 80, 60))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert mask.max() == 255  # Has white pixels
    # Center should be filled
    assert mask[50, 50] > 0


def test_generate_eye_mask():
    """Test elliptical eye mask generation."""
    mask = _generate_eye_mask((100, 100), (30, 40, 20, 10))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Center of eye should be filled
    assert mask[45, 40] > 0


def test_generate_mouth_mask():
    """Test elliptical mouth mask generation."""
    mask = _generate_mouth_mask((100, 100), (35, 60, 30, 10))
    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    # Center of mouth should be filled
    assert mask[65, 50] > 0


# ── Face Detection ───────────────────────────────────────────────


def test_detect_faces_dnn_on_face():
    """Test DNN face detection on a face-like image."""
    img = _make_face_like_image(200, 160)
    faces = _detect_faces_dnn(img)
    # Should detect at least one face
    assert len(faces) >= 1
    for face in faces:
        assert "bbox" in face
        assert "confidence" in face
        x, y, w, h = face["bbox"]
        assert w > 0
        assert h > 0
        assert face["confidence"] >= 0.4


def test_detect_faces_dnn_on_random():
    """Test DNN face detection on random noise (no face)."""
    np.random.seed(42)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    faces = _detect_faces_dnn(img, min_confidence=0.5)
    # Random noise should not produce face detections above threshold
    assert len(faces) == 0


def test_detect_faces_haar_on_face():
    """Test Haar cascade face detection on a face-like image."""
    img = _make_face_like_image(200, 160)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _detect_faces_haar(gray)
    # May or may not detect — but should not crash
    assert isinstance(faces, list)


def test_detect_faces_auto():
    """Test auto face detection (DNN first, fallback to Haar)."""
    img = _make_face_like_image(200, 160)
    faces = _detect_faces(img, method=DetectionMethod.AUTO)
    # Should detect at least one face
    assert len(faces) >= 1


def test_detect_faces_heuristic():
    """Test heuristic detection (full image as face)."""
    img = np.ones((100, 80, 3), dtype=np.uint8) * 200
    faces = _detect_faces(img, method=DetectionMethod.HEURISTIC)
    assert len(faces) == 1
    assert faces[0]["bbox"] == (0, 0, 80, 100)


# ── Eye Region Refinement ────────────────────────────────────────


def test_refine_eye_regions_on_face():
    """Test eye region detection on a face-like image."""
    img = _make_face_like_image(200, 160)
    face_bbox = (30, 40, 100, 120)
    eyes = _refine_eye_regions(img, face_bbox)
    assert len(eyes) >= 1  # At least one eye should be found
    for ex, ey, ew, eh in eyes:
        assert ew > 0
        assert eh > 0
        # Eyes should be in upper half of face
        assert ey < face_bbox[1] + face_bbox[3] // 2


def test_refine_eye_regions_no_face():
    """Test eye refinement on blank image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 240
    eyes = _refine_eye_regions(img, (10, 10, 30, 30))
    assert isinstance(eyes, list)


# ── Mouth Region Refinement ──────────────────────────────────────


def test_refine_mouth_region_on_face():
    """Test mouth region detection on a face-like image."""
    img = _make_face_like_image(200, 160)
    face_bbox = (30, 40, 100, 120)
    mouth = _refine_mouth_region(img, face_bbox)
    # Mouth may or may not be detected, but should not crash
    if mouth is not None:
        mx, my, mw, mh = mouth
        assert mw > 0
        assert mh > 0
        # Mouth should be in lower half of face
        assert my >= face_bbox[1] + face_bbox[3] // 2


def test_refine_mouth_region_no_face():
    """Test mouth refinement on blank image."""
    img = np.ones((100, 100, 3), dtype=np.uint8) * 240
    mouth = _refine_mouth_region(img, (10, 10, 20, 20))
    # Small face may not produce a mouth, but should not crash
    assert mouth is None or len(mouth) == 4


# ── Full Feature Detection Pipeline ──────────────────────────────


def test_detect_character_features_on_face():
    """Test full feature detection pipeline on face-like image."""
    img = _make_face_like_image(200, 160)
    features = detect_character_features(
        img,
        method=DetectionMethod.AUTO,
        min_face_size=30,
        min_confidence=0.3,
    )
    # Should detect a face
    assert features.face_detected
    assert features.face_bbox is not None
    x, y, w, h = features.face_bbox
    assert w > 20
    assert h > 20
    # Should have face mask
    assert features.has_feature("face")
    # May have eyes
    if features.eye_regions:
        assert len(features.eye_regions) >= 1


def test_detect_character_features_no_face():
    """Test feature detection on image with no face."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    features = detect_character_features(
        img,
        method=DetectionMethod.DNN,
        min_face_size=30,
        min_confidence=0.8,
    )
    # High threshold should give no detections
    # But it's possible random noise produces a false positive,
    # so just check we don't crash
    assert isinstance(features, CharacterFeatures)


def test_detect_character_features_heuristic():
    """Test feature detection with heuristic method."""
    img = _make_face_like_image(100, 80)
    features = detect_character_features(
        img,
        method=DetectionMethod.HEURISTIC,
    )
    assert features.face_detected
    assert features.face_bbox == (0, 0, 80, 100)


def test_detect_character_features_tiny_image():
    """Test feature detection on very small image."""
    img = _make_face_like_image(20, 16)
    features = detect_character_features(
        img,
        method=DetectionMethod.AUTO,
        min_face_size=30,  # Larger than image
    )
    assert features.face_detected is False


def test_detect_character_features_grayscale():
    """Test feature detection on grayscale image."""
    gray = np.ones((100, 100), dtype=np.uint8) * 200
    features = detect_character_features(gray)
    # Should handle gracefully
    assert isinstance(features, CharacterFeatures)


def test_detect_character_features_rgba():
    """Test feature detection on RGBA image."""
    rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    rgba[:, :, :3] = _make_face_like_image(100, 100)
    rgba[:, :, 3] = 255
    features = detect_character_features(rgba)
    # Should handle RGBA by using RGB channels
    assert isinstance(features, CharacterFeatures)


# ── Enhance Feature Masks ────────────────────────────────────────


def test_enhance_feature_masks():
    """Test dilation of feature masks."""
    img = _make_face_like_image(100, 80)
    features = detect_character_features(img, method=DetectionMethod.HEURISTIC)
    original_area = features.feature_masks["face"].sum()

    enhanced = enhance_feature_masks(img, features, dilation_kernel=5)
    enhanced_area = enhanced.feature_masks["face"].sum()

    # Dilation should increase mask area
    assert enhanced_area >= original_area


# ── Visualize Features ───────────────────────────────────────────


def test_visualize_features():
    """Test feature visualization does not crash."""
    img = _make_face_like_image(100, 80)
    features = detect_character_features(img, method=DetectionMethod.HEURISTIC)
    vis = visualize_features(img, features)
    assert vis.shape[:2] == img.shape[:2]
    assert vis.dtype == np.uint8


# ── Config Validation ────────────────────────────────────────────


def test_config_character_features_defaults():
    """Test default character feature config is valid."""
    config = ColorVectorizationConfig(
        enable_character_features=True,
    )
    _validate_config(config)  # Should not raise


def test_config_character_features_invalid_feature_simplify():
    """Test invalid feature_simplify_tolerance raises."""
    config = ColorVectorizationConfig(
        enable_character_features=True,
        feature_simplify_tolerance=10.0,
    )
    with pytest.raises(ValueError, match="feature_simplify_tolerance"):
        _validate_config(config)


def test_config_character_features_invalid_detail_level():
    """Test invalid semantic_detail_level raises."""
    config = ColorVectorizationConfig(
        enable_character_features=True,
        semantic_detail_level="ultra",
    )
    with pytest.raises(ValueError, match="semantic_detail_level"):
        _validate_config(config)


def test_config_character_features_invalid_line_art_threshold():
    """Test invalid line_art_threshold raises."""
    config = ColorVectorizationConfig(
        enable_character_features=True,
        line_art_threshold=5,
    )
    with pytest.raises(ValueError, match="line_art_threshold"):
        _validate_config(config)


# ── Line Art Extraction ──────────────────────────────────────────


def test_extract_line_art_basic():
    """Test line art extraction on simple image."""
    # Create an image with a clear edge
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    img[10:40, 10:40] = [0, 0, 0]  # Dark square on white

    contours, edge_mask = _extract_line_art(img, threshold1=20, threshold2=60)
    assert isinstance(contours, list)
    assert edge_mask.shape == (50, 50)
    assert edge_mask.dtype == np.uint8
    # Should detect edges around the square
    assert edge_mask.sum() > 0, "Edge mask should contain edge pixels"
    # Should find at least 1 contour (the square outline)
    assert len(contours) >= 1, "Should find at least the square contour"


def test_extract_line_art_blank():
    """Test line art extraction on blank image."""
    img = np.ones((50, 50, 3), dtype=np.uint8) * 255
    contours, edge_mask = _extract_line_art(img, threshold1=100, threshold2=200)
    assert isinstance(contours, list)
    assert edge_mask.shape == (50, 50)


def test_extract_line_art_rgba():
    """Test line art extraction on RGBA image."""
    rgba = np.zeros((50, 50, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200
    rgba[10:40, 10:40, :3] = 50
    rgba[:, :, 3] = 255

    contours, edge_mask = _extract_line_art(rgba, threshold1=20, threshold2=60)
    assert isinstance(contours, list)
    assert edge_mask.shape == (50, 50)


# ── Stroke Paths ──────────────────────────────────────────────────


def test_contours_to_svg_strokes():
    """Test contour to SVG stroke conversion."""
    contour = np.array([[[0, 0]], [[10, 0]], [[10, 10]], [[0, 10]]], dtype=np.int32)
    svg = _contours_to_svg_strokes([contour], stroke_color="black", stroke_width=1.0)
    assert "M" in svg
    assert "L" in svg
    assert 'stroke="black"' in svg
    assert 'stroke-width="1.0"' in svg
    assert 'fill="none"' in svg
    # Should NOT have Z (open path)
    assert "Z" not in svg


def test_contours_to_svg_strokes_empty():
    """Test empty stroke list."""
    svg = _contours_to_svg_strokes([])
    assert svg == ""


# ── Eye Highlights ────────────────────────────────────────────────


def test_add_eye_highlight_paths():
    """Test procedural eye highlight generation."""
    eye_regions = [(30, 40, 20, 10)]
    elements = _add_eye_highlight_paths(
        eye_regions,
        highlight_color="white",
        highlight_radius_ratio=0.25,
    )
    assert len(elements) >= 1  # At least main highlight
    assert "<circle" in elements[0]
    assert 'fill="white"' in elements[0]

    # With larger eyes, should also have secondary highlight
    eye_regions_large = [(30, 40, 40, 30)]
    elements_large = _add_eye_highlight_paths(
        eye_regions_large,
        highlight_color="white",
    )
    assert len(elements_large) >= 2  # Main + secondary


def test_add_eye_highlight_paths_empty():
    """Test eye highlights with no eyes."""
    elements = _add_eye_highlight_paths([])
    assert elements == []


def test_add_eye_highlight_paths_tiny():
    """Test eye highlights with tiny eye regions (should still produce something when large enough)."""
    # Size (5, 4) is above the min threshold (4, 4)
    elements = _add_eye_highlight_paths([(0, 0, 5, 4)])
    assert len(elements) >= 1  # At least main highlight
    # Size (3, 2) is below threshold, should produce no highlights
    elements_tiny = _add_eye_highlight_paths([(0, 0, 3, 2)])
    assert elements_tiny == []


# ── Feature Region Vectorization ─────────────────────────────────


def test_vectorize_feature_region():
    """Test vectorization of a specific feature region."""
    img = _make_test_image_rgba(64, 64)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:36, 20:36] = 255  # Center square region

    config = ColorVectorizationConfig(
        num_colors=8,
        simplify_tolerance=1.0,
        min_area=5,
        feature_simplify_tolerance=0.5,
        feature_min_area=2,
    )

    layers = _vectorize_feature_region(img, mask, config)
    assert isinstance(layers, list)
    for layer in layers:
        assert "color" in layer
        assert "contours" in layer
        assert "area" in layer
        assert "feature_layer" in layer
        assert layer["feature_layer"] is True


# ── Character-Aware Vectorization ────────────────────────────────


def test_vectorize_color_character_aware_fallback():
    """Test character-aware vectorization falls back gracefully when no face."""
    img = _make_test_image_rgba(64, 64)
    config = ColorVectorizationConfig(
        num_colors=8,
        enable_character_features=True,
        simplify_tolerance=1.0,
        min_area=5,
    )
    svg = vectorize_color_character_aware(img, config)
    assert "<?xml" in svg
    assert "<svg" in svg
    assert "</svg>" in svg
    # Should still produce valid SVG content
    assert 'fill="' in svg or "M " in svg


def test_vectorize_color_character_aware_with_heuristic_face():
    """Test character-aware vectorization with heuristic face detection."""
    img = _make_face_like_image(100, 80)
    # Add alpha channel
    rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = 255

    config = ColorVectorizationConfig(
        num_colors=8,
        enable_character_features=True,
        simplify_tolerance=1.0,
        min_area=5,
        feature_simplify_tolerance=0.5,
        feature_min_area=2,
        preserve_line_art=True,
        enable_eye_highlights=True,
    )
    svg = vectorize_color_character_aware(rgba, config)
    assert "<?xml" in svg
    assert "<svg" in svg
    assert "</svg>" in svg


def test_vectorize_color_with_character_features_enabled():
    """Test vectorize_color delegates to character-aware when enabled."""
    img = _make_test_image_rgba(64, 64)
    config = ColorVectorizationConfig(
        num_colors=8,
        enable_character_features=True,
        simplify_tolerance=1.0,
        min_area=5,
    )
    svg = vectorize_color(img, config)
    assert "<?xml" in svg
    assert "<svg" in svg
    assert "</svg>" in svg


def test_vectorize_color_character_aware_line_art_disabled():
    """Test character-aware vectorization with line art disabled."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200
    rgba[20:44, 20:44, :3] = 100
    rgba[:, :, 3] = 255

    config = ColorVectorizationConfig(
        num_colors=8,
        enable_character_features=True,
        preserve_line_art=False,
        enable_eye_highlights=False,
    )
    svg = vectorize_color_character_aware(rgba, config)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_vectorize_color_character_aware_high_detail():
    """Test character-aware vectorization with high detail level."""
    img = _make_face_like_image(100, 80)
    rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = 255

    config = ColorVectorizationConfig(
        num_colors=16,
        enable_character_features=True,
        simplify_tolerance=0.8,
        min_area=5,
        feature_simplify_tolerance=0.3,
        feature_min_area=1,
        semantic_detail_level="high",
    )
    svg = vectorize_color_character_aware(rgba, config)
    assert "<svg" in svg
    assert "</svg>" in svg
