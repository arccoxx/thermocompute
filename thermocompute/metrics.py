from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch


@dataclass
class ExperimentResult:
    """Serializable experiment result."""

    name: str
    metrics: dict[str, Any]

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def moment_summary(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().float().reshape(-1)
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "min": float(flat.min().cpu()),
        "max": float(flat.max().cpu()),
    }


def physical_advantage_ratio(classical_steps: int, thermodynamic_physical_time: float, dt: float) -> float:
    """Compare sequential classical update count to normalized physical evolution steps.

    This is a dimensionless emulator metric. It is not a silicon benchmark.
    """

    denom = max(thermodynamic_physical_time / dt, 1e-12)
    return float(classical_steps / denom)
