"""
Tests for the base pipeline infrastructure.
"""

from pathlib import Path
from assetforge.pipelines.base import BasePipeline, StageResult


def test_stage_result_defaults():
    """Test StageResult default values."""
    result = StageResult(name="test", success=True, duration_s=1.5)
    assert result.name == "test"
    assert result.success is True
    assert result.duration_s == 1.5
    assert result.output_paths == []
    assert result.error is None
    assert result.metadata == {}


def test_base_pipeline_output_dir(tmp_path: Path):
    """Test output directory creation."""

    class TestPipeline(BasePipeline):
        def run(self):
            return self.stages

    pipeline = TestPipeline(output_dir=str(tmp_path / "test_out"), name="test")
    assert pipeline.output_dir.exists()
    assert pipeline.name == "test"


def test_base_pipeline_stage_timing(tmp_path: Path):
    """Test stage timing and success tracking."""

    class TestPipeline(BasePipeline):
        def run(self):
            self.run_stage("success_stage", lambda: True)
            self.run_stage(
                "fail_stage", lambda: (_ for _ in ()).throw(Exception("test error"))
            )
            return self.stages

    pipeline = TestPipeline(output_dir=str(tmp_path / "test"), name="test")
    pipeline.run()

    assert len(pipeline.stages) == 2
    assert pipeline.stages[0].success is True
    assert pipeline.stages[0].name == "success_stage"
    assert pipeline.stages[1].success is False
    assert pipeline.stages[1].error == "test error"
    assert pipeline.stages[1].duration_s >= 0


def test_validate_input_dir(tmp_path: Path):
    """Test validate_input_dir function."""
    from assetforge.pipelines.base import validate_input_dir

    # Valid directory
    d = tmp_path / "valid_dir"
    d.mkdir()
    result = validate_input_dir(str(d), check_exists=True)
    assert result == d

    # Non-existent directory with check_exists=False (should not raise)
    result = validate_input_dir(str(tmp_path / "nonexistent"), check_exists=False)
    assert str(result).endswith("nonexistent")

    # Non-existent directory with check_exists=True (should raise)
    import pytest

    with pytest.raises(FileNotFoundError):
        validate_input_dir(str(tmp_path / "missing"), check_exists=True)
