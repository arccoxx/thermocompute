from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from thermocompute import (
    ThermodynamicCNNClassifier,
    ThermodynamicNeuronConfig,
    fit_cnn_classifier_end_to_end,
    make_toy_cnn_data,
    set_seed,
)


class ClassicalTinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 2, channels: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, n_classes),
        )

    @property
    def physical_time(self) -> float:
        return 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-light thermodynamic CNN experiment.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train-steps", type=int, default=80)
    args = parser.parse_args()

    if args.device == "cuda":
        raise RuntimeError("This experiment is intentionally CPU-safe while external GPU training is active.")
    device = torch.device("cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(4242)
    generator = torch.Generator(device=device).manual_seed(4242)
    train_x, train_y = make_toy_cnn_data(96, noise=0.04, device=device, generator=generator)
    eval_x, eval_y = make_toy_cnn_data(64, noise=0.04, device=device, generator=generator)

    set_seed(1001)
    classical = ClassicalTinyCNN().to(device)
    classical_result = fit_cnn_classifier_end_to_end(
        classical,
        train_x,
        train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        n_steps=args.train_steps,
        learning_rate=1e-2,
    )

    set_seed(1001)
    thermo = ThermodynamicCNNClassifier(
        1,
        2,
        conv_channels=8,
        thermo_channels=24,
        neuron_config=ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0),
        memory_efficient_chunk_size=12,
    ).to(device)
    thermo_result = fit_cnn_classifier_end_to_end(
        thermo,
        train_x,
        train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        n_steps=args.train_steps,
        learning_rate=1e-2,
    )

    payload = {
        "name": "cnn_experiment",
        "device": str(device),
        "task": "vertical-vs-horizontal 8x8 bar classification",
        "train_steps": args.train_steps,
        "models": {
            "classical_tiny_cnn": _row(classical_result, physical_time=classical.physical_time),
            "thermodynamic_cnn": _row(thermo_result, physical_time=thermo.physical_time),
        },
        "claim_boundary": (
            "This is a CPU-light coverage experiment showing CNN compatibility. "
            "It is not a production computer-vision benchmark."
        ),
    }
    out = args.outdir / "cnn_experiment.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _row(result: Any, *, physical_time: float) -> dict[str, float | int]:
    return {
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "initial_accuracy": result.initial_accuracy,
        "final_accuracy": result.final_accuracy,
        "n_steps": result.n_steps,
        "fit_wall_ms": result.fit_wall_ms,
        "modeled_physical_time": physical_time,
    }


if __name__ == "__main__":
    raise SystemExit(main())
