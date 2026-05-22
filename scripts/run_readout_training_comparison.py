from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from thermocompute import (
    ThermodynamicCNNClassifier,
    ThermodynamicFlowVelocity,
    ThermodynamicNeuronConfig,
    fit_cnn_classifier_end_to_end,
    fit_cnn_readout_ridge,
    fit_flow_matching_end_to_end,
    fit_flow_matching_readout_ridge,
    make_mog2d,
    make_toy_cnn_data,
    rbf_mmd2,
    sample_flow,
    set_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-light fast readout vs end-to-end training comparison.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--flow-steps", type=int, default=96)
    parser.add_argument("--cnn-steps", type=int, default=80)
    args = parser.parse_args()

    if args.device == "cuda":
        raise RuntimeError("This comparison is intentionally CPU-safe while external GPU training is active.")
    device = torch.device("cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": "readout_training_comparison",
        "device": str(device),
        "flow_matching": _run_flow_comparison(device, args.flow_steps),
        "cnn": _run_cnn_comparison(device, args.cnn_steps),
        "claim_boundary": (
            "Readout ridge is a fast frozen-feature training path. End-to-end is the no-ridge inductive baseline. "
            "This CPU artifact tests training mechanics and relative cost; it is not a production benchmark."
        ),
    }
    out = args.outdir / "readout_training_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _run_flow_comparison(device: torch.device, n_steps: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(8181)
    train = make_mog2d(384, device=device, generator=generator)
    reference = make_mog2d(128, device=device, generator=generator)
    neuron = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    ridge_train_generator = torch.Generator(device=device).manual_seed(9101)
    ridge_sample_generator = torch.Generator(device=device).manual_seed(9102)
    end_to_end_train_generator = torch.Generator(device=device).manual_seed(9101)
    end_to_end_sample_generator = torch.Generator(device=device).manual_seed(9102)

    set_seed(9001)
    ridge_model = ThermodynamicFlowVelocity(
        2,
        embed_dim=16,
        thermo_hidden_dim=48,
        time_features=4,
        neuron_config=neuron,
        memory_efficient_chunk_size=16,
    ).to(device)
    ridge = fit_flow_matching_readout_ridge(
        ridge_model,
        train,
        ridge=1e-2,
        n_pairs=512,
        eval_pairs=64,
        generator=ridge_train_generator,
    )
    ridge_sample = sample_flow(ridge_model, 128, n_flow_steps=1, generator=ridge_sample_generator, device=device)

    set_seed(9001)
    end_to_end_model = ThermodynamicFlowVelocity(
        2,
        embed_dim=16,
        thermo_hidden_dim=48,
        time_features=4,
        neuron_config=neuron,
        memory_efficient_chunk_size=16,
    ).to(device)
    end_to_end = fit_flow_matching_end_to_end(
        end_to_end_model,
        train,
        n_steps=n_steps,
        batch_size=64,
        learning_rate=2e-3,
        generator=end_to_end_train_generator,
    )
    end_to_end_sample = sample_flow(
        end_to_end_model,
        128,
        n_flow_steps=1,
        generator=end_to_end_sample_generator,
        device=device,
    )

    return {
        "task": "2D eight-mode Gaussian mixture flow matching",
        "readout_ridge": {
            **ridge.__dict__,
            "one_step_mmd2": float(rbf_mmd2(ridge_sample.samples, reference).detach().cpu()),
            "one_step_wall_ms": ridge_sample.wall_ms,
            "one_step_physical_time": ridge_sample.modeled_physical_time,
        },
        "end_to_end_no_ridge": {
            **end_to_end.__dict__,
            "one_step_mmd2": float(rbf_mmd2(end_to_end_sample.samples, reference).detach().cpu()),
            "one_step_wall_ms": end_to_end_sample.wall_ms,
            "one_step_physical_time": end_to_end_sample.modeled_physical_time,
        },
    }


def _run_cnn_comparison(device: torch.device, n_steps: int) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(4242)
    train_x, train_y = make_toy_cnn_data(96, noise=0.04, device=device, generator=generator)
    eval_x, eval_y = make_toy_cnn_data(64, noise=0.04, device=device, generator=generator)
    neuron = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)

    set_seed(1001)
    ridge_model = ThermodynamicCNNClassifier(
        1,
        2,
        conv_channels=8,
        thermo_channels=24,
        neuron_config=neuron,
        memory_efficient_chunk_size=12,
    ).to(device)
    ridge = fit_cnn_readout_ridge(
        ridge_model,
        train_x,
        train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        ridge=1e-2,
    )

    set_seed(1001)
    end_to_end_model = ThermodynamicCNNClassifier(
        1,
        2,
        conv_channels=8,
        thermo_channels=24,
        neuron_config=neuron,
        memory_efficient_chunk_size=12,
    ).to(device)
    end_to_end = fit_cnn_classifier_end_to_end(
        end_to_end_model,
        train_x,
        train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        n_steps=n_steps,
        learning_rate=1e-2,
    )

    return {
        "task": "vertical-vs-horizontal 8x8 bar classification",
        "readout_ridge": ridge.__dict__,
        "end_to_end_no_ridge": end_to_end.__dict__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
