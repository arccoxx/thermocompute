from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from thermocompute import (
    FlowVelocityMLP,
    ThermodynamicFlowVelocity,
    ThermodynamicNeuronConfig,
    fit_flow_matching_end_to_end,
    flow_speedup_vs_diffusion,
    make_mog2d,
    rbf_mmd2,
    sample_flow,
    set_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-light thermodynamic flow-matching diffusion experiment.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train-steps", type=int, default=96)
    parser.add_argument("--sample-count", type=int, default=128)
    args = parser.parse_args()

    if args.device == "cuda":
        raise RuntimeError("This experiment is intentionally CPU-safe by default; use CPU while external GPU training is active.")
    device = torch.device("cpu")
    set_seed(31415)
    generator = torch.Generator(device=device).manual_seed(31415)
    args.outdir.mkdir(parents=True, exist_ok=True)

    train = make_mog2d(384, device=device, generator=generator)
    reference = make_mog2d(args.sample_count, device=device, generator=generator)
    neuron = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    classical = FlowVelocityMLP(2, hidden_dim=48).to(device)
    thermo = ThermodynamicFlowVelocity(
        2,
        embed_dim=16,
        thermo_hidden_dim=48,
        neuron_config=neuron,
        memory_efficient_chunk_size=16,
    ).to(device)

    # Give both models the same deterministic random seed before training;
    # their architectures differ, but the experiment is reproducible.
    set_seed(2718)
    classical_fit = fit_flow_matching_end_to_end(
        classical,
        train,
        n_steps=args.train_steps,
        batch_size=64,
        learning_rate=2e-3,
        generator=generator,
    )
    set_seed(2718)
    thermo_fit = fit_flow_matching_end_to_end(
        thermo,
        train,
        n_steps=args.train_steps,
        batch_size=64,
        learning_rate=2e-3,
        generator=generator,
    )

    flow_steps = [1, 2, 4, 8]
    rows: list[dict[str, Any]] = []
    rows.extend(_sample_rows("classical_flow_mlp", classical, reference, flow_steps, args.sample_count, generator))
    rows.extend(_sample_rows("thermodynamic_flow", thermo, reference, flow_steps, args.sample_count, generator))
    payload = {
        "name": "flow_matching_experiment",
        "device": str(device),
        "data": "2D eight-mode Gaussian mixture",
        "train_steps": args.train_steps,
        "sample_count": args.sample_count,
        "fits": {
            "classical_flow_mlp": classical_fit.__dict__,
            "thermodynamic_flow": thermo_fit.__dict__,
        },
        "samples": rows,
        "best_by_model": _best_by_model(rows),
        "claim_boundary": (
            "Flow matching reduces neural evaluation count versus long iterative diffusion samplers. "
            "This toy CPU experiment is a lightweight feasibility check, not a production image/audio diffusion benchmark."
        ),
    }
    out = args.outdir / "flow_matching_experiment.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _sample_rows(
    name: str,
    model: torch.nn.Module,
    reference: torch.Tensor,
    flow_steps: list[int],
    sample_count: int,
    generator: torch.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for steps in flow_steps:
        result = sample_flow(model, sample_count, n_flow_steps=steps, generator=generator, device=torch.device("cpu"))
        mmd = float(rbf_mmd2(result.samples, reference, bandwidth=1.0).detach().cpu())
        rows.append(
            {
                "model": name,
                "flow_steps": steps,
                "mmd2_to_reference": mmd,
                "wall_ms": result.wall_ms,
                "function_evaluations": result.function_evaluations,
                "modeled_physical_time": result.modeled_physical_time,
                "eval_speedup_vs_50_step_diffusion": flow_speedup_vs_diffusion(50, steps),
                "eval_speedup_vs_100_step_diffusion": flow_speedup_vs_diffusion(100, steps),
            }
        )
    return rows


def _best_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        model = str(row["model"])
        if model not in best or float(row["mmd2_to_reference"]) < float(best[model]["mmd2_to_reference"]):
            best[model] = copy.deepcopy(row)
    return best


if __name__ == "__main__":
    raise SystemExit(main())
