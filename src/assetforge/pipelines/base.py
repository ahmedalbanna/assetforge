"""
Base pipeline class providing common infrastructure.

All pipelines inherit from this base class which provides:
- Stage sequencing and timing
- Output directory management
- Logging
- Error handling
"""

import logging
import time
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("assetforge")


@dataclass
class StageResult:
    """Result of a single pipeline stage."""

    name: str
    success: bool
    duration_s: float
    output_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BasePipeline(ABC):
    """Abstract base class for all asset pipelines."""

    def __init__(self, output_dir: str | Path, name: str = "pipeline"):
        self.name = name
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: list[StageResult] = []

    @abstractmethod
    def run(self) -> list[StageResult]:
        """Execute all stages of the pipeline."""
        ...

    def run_stage(self, name: str, fn, *args, **kwargs) -> StageResult:
        """Run a single pipeline stage with timing and error handling."""
        logger.info(f"[{self.name}] Stage: {name}...")
        start = time.time()

        try:
            result = fn(*args, **kwargs)
            duration = time.time() - start

            stage_result = StageResult(
                name=name,
                success=True,
                duration_s=round(duration, 2),
                output_paths=result
                if isinstance(result, list)
                else [result]
                if result
                else [],
            )
            logger.info(f"[{self.name}] ✓ {name} ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            logger.error(f"[{self.name}] ✗ {name} failed after {duration:.2f}s: {e}")
            stage_result = StageResult(
                name=name,
                success=False,
                duration_s=round(duration, 2),
                error=str(e),
            )

        self.stages.append(stage_result)
        return stage_result

    def summary(self) -> None:
        """Print a summary of all stages."""
        total_time = sum(s.duration_s for s in self.stages)
        successes = sum(1 for s in self.stages if s.success)
        failures = sum(1 for s in self.stages if not s.success)

        logger.info("=" * 50)
        logger.info(f"Pipeline '{self.name}' complete")
        logger.info(f"  Stages: {successes} succeeded, {failures} failed")
        logger.info(f"  Total time: {total_time:.2f}s")

        if failures > 0:
            for s in self.stages:
                if not s.success:
                    logger.warning(f"  Failed: {s.name} — {s.error}")

        logger.info(f"  Output: {self.output_dir}")
        logger.info("=" * 50)
