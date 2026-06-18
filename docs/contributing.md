# Contributing Guide

## Development Setup

```bash
git clone https://github.com/assetforge/assetforge.git
cd assetforge
pip install -e ".[all]"
pip install pytest pytest-cov mypy
maturin develop --release
```

## Code Style

- **Python**: Follow PEP 8. Use type hints for all public APIs.
- **Rust**: Follow standard Rust formatting (`rustfmt`).
- **Documentation**: Docstrings for all public classes/methods.

Run linting before committing:

```bash
# Python
ruff check src/ tests/
mypy src/assetforge/

# Rust
cd rust && cargo clippy && cargo fmt --check
```

## Testing

```bash
# Run all Python tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_pipeline_2d.py -v

# Run with coverage
python3 -m pytest tests/ --cov=src/assetforge/

# Run Rust tests
cd rust && cargo test

# Run specific Rust test
cd rust && cargo test vectorize
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use `tmp_path` fixtures for temporary files
- Test error paths, not just happy paths
- For pipeline tests, verify stage result structure

Example:
```python
def test_pipeline_initialization(self, tmp_path):
    pipeline = Pipeline2D(
        input_path=str(tmp_path / "test.png"),
        output_dir=str(tmp_path / "output"),
    )
    assert pipeline.output_dir.exists()
```

## Adding a New Pipeline

1. Create `src/assetforge/pipelines/pipeline_your_name.py`
2. Extend `BasePipeline` and implement `run()` with stage methods
3. Add backend wrapper in `src/assetforge/backends/` if needed
4. Register CLI command in `src/assetforge/cli.py`
5. Add tests in `tests/test_pipeline_your_name.py`

## Adding a New Backend

1. Create `src/assetforge/backends/your_backend.py`
2. Implement a class with the backend's public API
3. Use `subprocess` for external tools, direct imports for Python packages
4. Provide clear error messages with install instructions
5. Import and use in the appropriate pipeline

## Building the Rust Extension

```bash
# Development build (fast iteration)
maturin develop --release

# Production wheel
maturin build --release

# The wheel will be in target/wheels/
pip install target/wheels/assetforge_pyo3-*.whl --force-reinstall
```

## Release Process

1. Update version in `pyproject.toml` and `src/assetforge/__init__.py`
2. Run full test suite: `python3 -m pytest tests/ -v && cd rust && cargo test`
3. Build Python wheel: `maturin build --release`
4. Build Rust wheel: `cd rust && cargo build --release`
5. Update `CHANGELOG.md`
6. Tag release: `git tag v0.1.0 && git push --tags`
7. Publish to PyPI: `twine upload dist/*`

## Project Roadmap

### Completed ✅
- [x] Core Python package + Rust workspace
- [x] 2D sprite pipeline with Rembg/SAM2 + Rust vectorization
- [x] 3D photogrammetry pipeline (OpenMVG + OpenMVS + Blender)
- [x] 3D generative AI pipeline (TripoSR + Hunyuan3D-2 + TRELLIS)
- [x] Gaussian splatting pipeline (COLMAP + 3DGS + SuGaR/MILo)
- [x] SPZ compression + GLTF KHR_gaussian_splatting export
- [x] Blender headless integration (decimate, remesh, cleanup, bake, export)
- [x] Web UI with Gradio
- [x] One-click install script
- [x] Comprehensive test suite (26 Python + 9 Rust)

### Future Ideas
- [ ] Docker image with all dependencies
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Real-time webcam → asset pipeline
- [ ] Batch processing multiple images
- [ ] Model training/fine-tuning integration
- [ ] Plugin system for custom backends
- [ ] Publish to PyPI
