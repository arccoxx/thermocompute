from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import DeviceConfig, ThermodynamicNeuronConfig, ThermodynamicTransformerConfig, set_seed
from .integration import ThermodynamicFFN
from .metrics import ExperimentResult
from .neurons import ThermodynamicNeuronLayer
from .training import fit_transformer_end_to_end_cold, fit_transformer_end_to_end_parallel_tempering
from .transformer import ThermodynamicTransformerLayer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _param_bytes(module: nn.Module) -> int:
    return int(sum(p.numel() * p.element_size() for p in module.parameters()))


def _time_forward(fn, device: torch.device, repeats: int = 3) -> tuple[float, Any]:
    timings: list[float] = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(torch.tensor(timings).median()), value


def _target_tokens(x: Tensor) -> Tensor:
    return torch.sin(x + 0.35 * torch.roll(x, shifts=1, dims=1)) + 0.1 * torch.cos(torch.roll(x, shifts=-1, dims=1))


def _fit_linear_readout(features: Tensor, base: Tensor, targets: Tensor, ridge: float = 1e-2) -> Tensor:
    x = features.reshape(-1, features.shape[-1])
    y = (targets - base).reshape(-1, targets.shape[-1])
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=x.device, dtype=x.dtype) * ridge
    reg[-1, -1] = 0.0
    return torch.linalg.solve(gram + reg, design.T @ y)


class ClassicalFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        base = x
        return torch.nn.functional.gelu(self.in_proj(self.norm(x))), base

    def forward(self, x: Tensor) -> Tensor:
        features, base = self.features(x)
        return base + self.out_proj(features)


def benchmark_thermoneuron_width(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(101)
    device, dtype = cfg.device, cfg.dtype
    batch = 256 if device.type == "cuda" else 64
    in_features = 32
    widths = [32, 64, 128, 256, 512, 1024, 2048]
    x = torch.randn(batch, in_features, device=device, dtype=dtype)
    rows: list[dict[str, Any]] = []
    for width in widths:
        layer = ThermodynamicNeuronLayer(in_features, width, t_f=0.2, dt=0.04, n_replicas=2, output="mean").to(device=device, dtype=dtype)
        wall_ms, result = _time_forward(lambda: layer(x, return_info=True), device)
        y, info = result
        rows.append(
            {
                "out_features": width,
                "physical_time": info.physical_time,
                "wall_ms_median": wall_ms,
                "parameter_count": _param_count(layer),
                "parameter_bytes": _param_bytes(layer),
                "output_std": float(y.std(unbiased=False).detach().cpu()),
                "output_nan_fraction": float(torch.isnan(y).float().mean().detach().cpu()),
            }
        )
    physical_times = [r["physical_time"] for r in rows]
    return ExperimentResult(
        "thermoneuron_width_scaling",
        {
            "device": str(device),
            "rows": rows,
            "physical_time_range": max(physical_times) - min(physical_times),
        },
    )


def benchmark_transformer_vs_classical(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(103)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 32
    seq_len = 16 if device.type == "cuda" else 8
    train_batch = 96 if device.type == "cuda" else 24
    eval_batch = 48 if device.type == "cuda" else 16
    widths = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = _target_tokens(x_train)
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = _target_tokens(x_eval)
    rows: list[dict[str, Any]] = []
    for width in widths:
        thermo = ThermodynamicTransformerLayer(
            embed_dim,
            4,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=2,
            thermo_output="mean",
        ).to(device=device, dtype=dtype)
        classical = ClassicalFFN(embed_dim, width).to(device=device, dtype=dtype)

        with torch.no_grad():
            thermo_features, thermo_base, _ = thermo.thermodynamic_features(x_train, return_base=True)
            thermo_solution = _fit_linear_readout(thermo_features, thermo_base, y_train)
            thermo.out_proj.weight.copy_(thermo_solution[:-1].T)
            thermo.out_proj.bias.copy_(thermo_solution[-1])
            c_features, c_base = classical.features(x_train)
            classical_solution = _fit_linear_readout(c_features, c_base, y_train)
            classical.out_proj.weight.copy_(classical_solution[:-1].T)
            classical.out_proj.bias.copy_(classical_solution[-1])

            thermo_wall, thermo_result = _time_forward(lambda: thermo(x_eval, return_info=True), device)
            thermo_pred, thermo_info = thermo_result
            classical_wall, classical_pred = _time_forward(lambda: classical(x_eval), device)
            thermo_loss = torch.mean((thermo_pred - y_eval).square())
            classical_loss = torch.mean((classical_pred - y_eval).square())

        tokens = eval_batch * seq_len
        classical_flop_proxy = 2 * tokens * embed_dim * width + 2 * tokens * width * embed_dim
        rows.append(
            {
                "width": width,
                "thermo_physical_time": thermo_info.physical_time,
                "thermo_wall_ms_median": thermo_wall,
                "classical_wall_ms_median": classical_wall,
                "thermo_param_count": _param_count(thermo),
                "classical_param_count": _param_count(classical),
                "thermo_param_bytes": _param_bytes(thermo),
                "classical_param_bytes": _param_bytes(classical),
                "classical_flop_proxy": int(classical_flop_proxy),
                "thermo_eval_mse": float(thermo_loss.detach().cpu()),
                "classical_eval_mse": float(classical_loss.detach().cpu()),
            }
        )
    physical_times = [r["thermo_physical_time"] for r in rows]
    return ExperimentResult(
        "transformer_vs_classical_width_benchmark",
        {
            "device": str(device),
            "rows": rows,
            "physical_time_range": max(physical_times) - min(physical_times),
            "classical_flop_proxy_increases": rows[-1]["classical_flop_proxy"] > rows[0]["classical_flop_proxy"],
        },
    )


def benchmark_training_comparison(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(107)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 8
    train_batch = 32 if device.type == "cuda" else 12
    eval_batch = 24 if device.type == "cuda" else 8
    seq_len = 6
    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = 0.5 * torch.sin(x_train + 0.25 * torch.roll(x_train, shifts=1, dims=1))
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = 0.5 * torch.sin(x_eval + 0.25 * torch.roll(x_eval, shifts=1, dims=1))
    config = ThermodynamicTransformerConfig(
        embed_dim=embed_dim,
        num_heads=2,
        thermo_hidden_dim=32,
        neuron=ThermodynamicNeuronConfig(t_f=0.1, dt=0.05, n_replicas=2, output="mean", temperature=0.05),
        residual_scale=0.7,
    )
    cold = config.build_layer().to(device=device, dtype=dtype)
    pt = config.build_layer().to(device=device, dtype=dtype)
    pt.load_state_dict(cold.state_dict())
    cold_result = fit_transformer_end_to_end_cold(
        cold,
        x_train,
        y_train,
        eval_inputs=x_eval,
        eval_targets=y_eval,
        n_steps=24,
        learning_rate=4e-3,
    )
    pt_result = fit_transformer_end_to_end_parallel_tempering(
        pt,
        x_train,
        y_train,
        eval_inputs=x_eval,
        eval_targets=y_eval,
        n_tempering_replicas=4,
        n_tempering_steps=24,
        learning_rate=4e-3,
        noise_scale=2e-3,
        swap_interval=3,
    )
    return ExperimentResult(
        "training_cold_vs_parallel_tempered",
        {
            "device": str(device),
            "cold": cold_result.__dict__,
            "parallel_tempered": pt_result.__dict__,
            "training_not_constant_time": True,
        },
    )


def run_research_benchmarks(outdir: str | Path = "artifacts", device_config: DeviceConfig | None = None) -> list[ExperimentResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results = [
        benchmark_thermoneuron_width(device_config),
        benchmark_transformer_vs_classical(device_config),
        benchmark_training_comparison(device_config),
    ]
    for result in results:
        result.to_json(out / f"{result.name}.json")
    _try_plot(results, out)
    return results


def _try_plot(results: list[ExperimentResult], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    comparison = next((r for r in results if r.name == "transformer_vs_classical_width_benchmark"), None)
    if comparison is not None:
        rows = comparison.metrics["rows"]
        widths = [r["width"] for r in rows]
        physical = [r["thermo_physical_time"] for r in rows]
        flops = [r["classical_flop_proxy"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(widths, physical, marker="o", label="thermo physical time")
        ax1.set_xlabel("width")
        ax1.set_ylabel("modeled physical time")
        ax1.set_xscale("log", base=2)
        ax2 = ax1.twinx()
        ax2.plot(widths, flops, color="tab:red", marker="s", label="classical FLOP proxy")
        ax2.set_ylabel("classical FLOP proxy")
        ax2.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out / "fixed_time_advantage.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(widths, [r["thermo_eval_mse"] for r in rows], marker="o", label="thermo")
        ax.plot(widths, [r["classical_eval_mse"] for r in rows], marker="s", label="classical")
        ax.set_xlabel("width")
        ax.set_ylabel("eval MSE after matched readout fit")
        ax.set_xscale("log", base=2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "accuracy_loss_vs_width.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cfg = DeviceConfig.auto(prefer_cuda=not args.cpu)
    results = run_research_benchmarks(args.outdir, cfg)
    print(json.dumps([{"name": r.name, "metrics": r.metrics} for r in results], indent=2))


if __name__ == "__main__":
    main()
