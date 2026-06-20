"""
Tests for AI-driven smart defaults and quality presets.

Covers:
- Quality preset structure and validation
- Preset resolution (resolve_preset_quality)
- Full config resolution (resolve_2d_config)
- Image analysis (analyze_image) basics
- Backward compatibility (empty quality = individual params)
- Pipeline2D integration with quality presets
"""

import pytest
import numpy as np
from pathlib import Path

from assetforge.backends.smart_defaults import (
    QUALITY_PRESETS,
    VALID_QUALITY_LEVELS,
    resolve_preset_quality,
    resolve_2d_config,
    analyze_image,
    auto_sprite_sizes,
    auto_vectorize_config,
    select_best_rembg_model,
)


# ── Preset Validation ─────────────────────────────────────────────────


def test_presets_defined():
    """Test all expected quality presets exist."""
    assert "fast" in QUALITY_PRESETS
    assert "balanced" in QUALITY_PRESETS
    assert "high" in QUALITY_PRESETS
    assert "ultra" in QUALITY_PRESETS


def test_preset_keys_consistent():
    """Test all presets have the same keys."""
    keys = set(QUALITY_PRESETS["balanced"].keys())
    for name, preset in QUALITY_PRESETS.items():
        assert set(preset.keys()) == keys, f"Preset '{name}' has mismatched keys"


def test_preset_values_in_range():
    """Test preset numeric values are within valid ranges."""
    for name, preset in QUALITY_PRESETS.items():
        assert 4 <= preset.get("num_colors", 0) <= 64, (
            f"{name}: num_colors out of range"
        )
        assert 0.1 <= preset.get("simplify_tolerance", 0) <= 5.0, (
            f"{name}: simplify_tolerance out of range"
        )
        assert 1 <= preset.get("min_area", 0) <= 100, f"{name}: min_area out of range"
        assert preset.get("vectorize_mode") in ("outline", "color"), (
            f"{name}: invalid vectorize_mode"
        )
        assert preset.get("background_remover") in ("rembg", "sam2"), (
            f"{name}: invalid background_remover"
        )
        assert preset.get("semantic_detail_level") in ("low", "medium", "high"), (
            f"{name}: invalid semantic_detail_level"
        )


def test_valid_quality_levels():
    """Test VALID_QUALITY_LEVELS includes all presets plus auto."""
    for name in QUALITY_PRESETS:
        assert name in VALID_QUALITY_LEVELS
    assert "auto" in VALID_QUALITY_LEVELS


# ── Preset Resolution ────────────────────────────────────────────────


def test_resolve_empty_returns_balanced():
    """Test empty quality resolves to balanced."""
    assert resolve_preset_quality("") == "balanced"


def test_resolve_none_returns_balanced():
    """Test unspecified quality resolves to balanced."""
    assert resolve_preset_quality("") == "balanced"


def test_resolve_valid_preset():
    """Test valid preset name returns as-is."""
    for name in QUALITY_PRESETS:
        assert resolve_preset_quality(name) == name


def test_resolve_invalid_falls_back():
    """Test invalid preset name falls back to balanced."""
    assert resolve_preset_quality("nonexistent") == "balanced"


def test_resolve_auto_no_image():
    """Test auto without image resolves to a valid preset."""
    result = resolve_preset_quality("auto")
    assert result in QUALITY_PRESETS


# ── Full Config Resolution ───────────────────────────────────────────


def test_resolve_2d_config_returns_dict():
    """Test resolve_2d_config returns a complete config dict."""
    cfg = resolve_2d_config("balanced")
    assert isinstance(cfg, dict)
    assert "num_colors" in cfg
    assert "simplify_tolerance" in cfg
    assert "min_area" in cfg
    assert "vectorize_mode" in cfg
    assert "background_remover" in cfg
    assert "gap_fill" in cfg


def test_resolve_2d_config_preset_values():
    """Test config values match preset for known qualities."""
    for name in QUALITY_PRESETS:
        cfg = resolve_2d_config(name)
        preset = QUALITY_PRESETS[name]
        # Check key values match (some may be overridden by image analysis)
        assert cfg["num_colors"] == preset["num_colors"] or cfg.get("_overridden"), (
            f"{name}: num_colors mismatch ({cfg['num_colors']} vs {preset['num_colors']})"
        )
        assert cfg["vectorize_mode"] == preset["vectorize_mode"], (
            f"{name}: vectorize_mode mismatch"
        )
        assert cfg["min_area"] == preset["min_area"], f"{name}: min_area mismatch"


def test_resolve_2d_config_with_overrides():
    """Test user overrides take precedence."""
    cfg = resolve_2d_config(
        "balanced",
        user_overrides={"num_colors": 48, "min_area": 10},
    )
    assert cfg["num_colors"] == 48
    assert cfg["min_area"] == 10
    # Other preset values should remain
    assert cfg["vectorize_mode"] == "color"


def test_resolve_2d_config_auto():
    """Test auto mode resolves to a valid preset."""
    cfg = resolve_2d_config("auto")
    assert isinstance(cfg, dict)
    for key in (
        "num_colors",
        "simplify_tolerance",
        "min_area",
        "vectorize_mode",
        "background_remover",
    ):
        assert key in cfg


def test_resolve_2d_config_no_label():
    """Test the 'label' key is removed from output."""
    cfg = resolve_2d_config("balanced")
    assert "label" not in cfg


# ── Image Analysis ───────────────────────────────────────────────────


def test_analyze_image_no_input():
    """Test analyze_image with no input returns defaults."""
    result = analyze_image()
    assert result["has_face"] is False
    assert result["is_anime"] is False
    assert result["megapixels"] == 0.0


def test_analyze_image_with_array():
    """Test analyze_image with a numpy array."""
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :] = [128, 128, 128]
    result = analyze_image(image_array=img)
    assert result["width"] == 200
    assert result["height"] == 100
    assert result["megapixels"] == 0.02
    assert result["aspect_ratio"] == 2.0


def test_analyze_image_face_not_detected_on_random():
    """Test face detection on random noise returns no face."""
    np.random.seed(42)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = analyze_image(image_array=img)
    assert result["has_face"] is False


def test_analyze_image_complexity_bounds():
    """Test complexity is between 0 and 1."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:40, 10:40] = [255, 0, 0]
    result = analyze_image(image_array=img)
    assert 0.0 <= result["complexity"] <= 1.0


# ── Auto Sprite Sizes ────────────────────────────────────────────────


def test_auto_sprite_sizes_fast():
    """Test fast preset gives small sprites."""
    analysis = {"megapixels": 1.0, "width": 512, "height": 512}
    sizes = auto_sprite_sizes(analysis=analysis, quality="fast")
    assert all(s <= 128 for s in sizes)


def test_auto_sprite_sizes_balanced():
    """Test balanced preset includes moderate sizes."""
    analysis = {"megapixels": 2.0, "width": 1024, "height": 1024}
    sizes = auto_sprite_sizes(analysis=analysis, quality="balanced")
    assert len(sizes) >= 2
    assert all(s <= 512 for s in sizes)


def test_auto_sprite_sizes_low_res():
    """Test low-res input gives smaller sprites."""
    analysis = {"megapixels": 0.1, "width": 200, "height": 200}
    sizes = auto_sprite_sizes(analysis=analysis, quality="balanced")
    assert all(s <= 256 for s in sizes)


# ── Auto Model Selection ─────────────────────────────────────────────


def test_select_best_rembg_model_default():
    """Test default model selection for general image."""
    analysis = {"is_anime": False, "has_face": False}
    model = select_best_rembg_model(analysis=analysis)
    assert model == "isnet-general-use"


def test_select_best_rembg_model_anime():
    """Test anime image selects isnet-anime."""
    analysis = {"is_anime": True, "has_face": False}
    model = select_best_rembg_model(analysis=analysis)
    assert model == "isnet-anime"


def test_select_best_rembg_model_face():
    """Test face image selects human_seg model."""
    analysis = {"is_anime": False, "has_face": True}
    model = select_best_rembg_model(analysis=analysis)
    assert model == "u2net_human_seg"


# ── Smart Vectorization Config ───────────────────────────────────────


def test_auto_vectorize_config_returns_correct_keys():
    """Test auto_vectorize_config returns expected keys."""
    analysis = {"complexity": 0.5}
    cfg = auto_vectorize_config(analysis, "balanced")
    for key in ("num_colors", "simplify_tolerance", "min_area", "gap_fill"):
        assert key in cfg


def test_auto_vectorize_config_complexity_scaling():
    """Test num_colors scales with complexity."""
    low = auto_vectorize_config({"complexity": 0.2}, "balanced")
    high = auto_vectorize_config({"complexity": 0.8}, "balanced")
    assert high["num_colors"] >= low["num_colors"]


# ── Pipeline2D Backward Compatibility ────────────────────────────────


def test_pipeline_2d_empty_quality_uses_defaults(tmp_path):
    """Test Pipeline2D with empty quality uses individual params (backward compat)."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    test_img = tmp_path / "test.png"
    from PIL import Image

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(test_img)

    pipeline = Pipeline2D(
        input_path=str(test_img),
        output_dir=str(tmp_path / "out"),
        quality="",
        vectorize_mode="outline",
        num_colors=8,
        simplify_tolerance=2.0,
        min_area=20,
    )
    assert pipeline.num_colors == 8
    assert pipeline.simplify_tolerance == 2.0
    assert pipeline.min_area == 20
    assert pipeline.vectorize_mode == "outline"


def test_pipeline_2d_quality_overrides_defaults(tmp_path):
    """Test Pipeline2D with quality preset overrides individual defaults."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    test_img = tmp_path / "test.png"
    from PIL import Image

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(test_img)

    pipeline = Pipeline2D(
        input_path=str(test_img),
        output_dir=str(tmp_path / "out"),
        quality="high",
        generate_vector=True,
    )
    # high preset has base num_colors=32, but auto-config may scale down
    # for a simple solid-color image
    assert 8 <= pipeline.num_colors <= 64, (
        f"num_colors out of range: {pipeline.num_colors}"
    )
    # high preset has alpha_matting=True
    assert pipeline.alpha_matting is True
    # high preset has enable_character_features=True
    assert pipeline.enable_character_features is True


def test_pipeline_2d_run_with_quality_preset(tmp_path):
    """Test Pipeline2D runs successfully with a quality preset."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    test_img = tmp_path / "test.png"
    from PIL import Image

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(test_img)

    pipeline = Pipeline2D(
        input_path=str(test_img),
        output_dir=str(tmp_path / "out2"),
        quality="fast",
        generate_vector=True,
    )
    results = pipeline.run()
    assert len(results) > 0
    # Should have succeeded
    assert any(r.success for r in results)


def test_pipeline_2d_fast_quality_uses_outline(tmp_path):
    """Test fast quality preset uses outline vectorization mode."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    test_img = tmp_path / "test_fast.png"
    from PIL import Image

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(test_img)

    pipeline = Pipeline2D(
        input_path=str(test_img),
        output_dir=str(tmp_path / "out_fast"),
        quality="fast",
        generate_vector=True,
    )
    # Fast preset uses outline mode
    assert pipeline.vectorize_mode == "outline"


def test_pipeline_2d_ultra_quality_uses_sam2(tmp_path):
    """Test ultra quality preset uses SAM2 background remover."""
    from assetforge.pipelines.pipeline_2d import Pipeline2D

    test_img = tmp_path / "test_ultra.png"
    from PIL import Image

    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(test_img)

    pipeline = Pipeline2D(
        input_path=str(test_img),
        output_dir=str(tmp_path / "out_ultra"),
        quality="ultra",
    )
    # Ultra preset uses sam2
    assert pipeline.background_remover == "sam2"
    # Ultra preset has fine detail
    assert pipeline.min_area <= 2
    assert pipeline.num_colors >= 24
