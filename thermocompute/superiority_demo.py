from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .config import DeviceConfig, set_seed
from .metrics import ExperimentResult
from .transformer import ThermodynamicTransformerLayer


class ClassicalDenseFFN(nn.Module):
    """Conventional dense transformer FFN baseline."""

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return torch.nn.functional.gelu(self.in_proj(self.norm(x))), x

    def forward(self, x: Tensor) -> Tensor:
        features, base = self.features(x)
        return base + self.out_proj(features)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _time_forward(fn, device: torch.device, repeats: int = 5) -> tuple[float, Any]:
    timings: list[float] = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(torch.tensor(timings).median()), value


def _target_fn(x: Tensor) -> Tensor:
    prev_token = torch.roll(x, shifts=1, dims=1)
    next_token = torch.roll(x, shifts=-1, dims=1)
    return torch.sin(x + 0.45 * prev_token) + 0.15 * torch.cos(next_token - x)


def _fit_readout(features: Tensor, base: Tensor, targets: Tensor, ridge: float = 1e-2) -> Tensor:
    x = features.reshape(-1, features.shape[-1])
    y = (targets - base).reshape(-1, targets.shape[-1])
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=x.device, dtype=x.dtype) * ridge
    reg[-1, -1] = 0.0
    return torch.linalg.solve(gram + reg, design.T @ y)


def _dense_ffn_flop_proxy(batch: int, seq_len: int, embed_dim: int, width: int) -> int:
    token_count = batch * seq_len
    return int(4 * token_count * embed_dim * width)


def _attention_flop_proxy(batch: int, seq_len: int, embed_dim: int) -> int:
    return int(2 * batch * seq_len * seq_len * embed_dim)


def run_superiority_demo(
    outdir: str | Path = "artifacts",
    device_config: DeviceConfig | None = None,
) -> ExperimentResult:
    """Run the flagship constant-physical-time superiority demo.

    The demo compares modeled thermodynamic FFN physical time against a
    conventional dense digital FFN work proxy. It is deliberately explicit:
    PyTorch wall time is reported, but the headline advantage is modeled
    physical latency under a parallel thermodynamic substrate.
    """

    cfg = device_config or DeviceConfig.auto()
    set_seed(211)
    device, dtype = cfg.device, cfg.dtype
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    embed_dim = 32
    seq_len = 16 if device.type == "cuda" else 8
    train_batch = 96 if device.type == "cuda" else 24
    eval_batch = 48 if device.type == "cuda" else 16
    measured_widths = [64, 128, 256, 512, 1024, 2048, 4096]
    projected_widths = [8192, 16384, 32768, 65536]
    t_f = 0.2
    dt = 0.04

    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = _target_fn(x_train)
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = _target_fn(x_eval)

    measured: list[dict[str, Any]] = []
    attention_work = _attention_flop_proxy(eval_batch, seq_len, embed_dim)
    min_digital_work = None

    for width in measured_widths:
        thermo = ThermodynamicTransformerLayer(
            embed_dim,
            4,
            thermo_hidden_dim=width,
            t_f=t_f,
            dt=dt,
            n_replicas=2,
            thermo_output="mean",
        ).to(device=device, dtype=dtype)
        classical = ClassicalDenseFFN(embed_dim, width).to(device=device, dtype=dtype)

        with torch.no_grad():
            thermo_features, thermo_base, _ = thermo.thermodynamic_features(x_train, return_base=True)
            thermo_solution = _fit_readout(thermo_features, thermo_base, y_train, ridge=1e-2)
            thermo.out_proj.weight.copy_(thermo_solution[:-1].T)
            thermo.out_proj.bias.copy_(thermo_solution[-1])

            classical_features, classical_base = classical.features(x_train)
            classical_solution = _fit_readout(classical_features, classical_base, y_train, ridge=1e-2)
            classical.out_proj.weight.copy_(classical_solution[:-1].T)
            classical.out_proj.bias.copy_(classical_solution[-1])

            thermo_wall, thermo_result = _time_forward(lambda: thermo(x_eval, return_info=True), device)
            thermo_pred, thermo_info = thermo_result
            classical_wall, classical_pred = _time_forward(lambda: classical(x_eval), device)
            thermo_loss = torch.mean((thermo_pred - y_eval).square())
            classical_loss = torch.mean((classical_pred - y_eval).square())

        ffn_work = _dense_ffn_flop_proxy(eval_batch, seq_len, embed_dim, width)
        digital_work = attention_work + ffn_work
        min_digital_work = digital_work if min_digital_work is None else min(min_digital_work, digital_work)
        measured.append(
            {
                "width": width,
                "thermo_physical_time": thermo_info.physical_time,
                "thermo_wall_ms_median": thermo_wall,
                "classical_wall_ms_median": classical_wall,
                "attention_flop_proxy": attention_work,
                "classical_ffn_flop_proxy": ffn_work,
                "classical_total_flop_proxy": digital_work,
                "thermo_param_count": _param_count(thermo),
                "classical_param_count": _param_count(classical),
                "thermo_eval_mse": float(thermo_loss.detach().cpu()),
                "classical_eval_mse": float(classical_loss.detach().cpu()),
                "digital_work_growth_vs_min_width": 1.0,
                "thermo_physical_growth_vs_min_width": 1.0,
                "modeled_latency_advantage_factor": 1.0,
            }
        )

    assert min_digital_work is not None
    base_physical = measured[0]["thermo_physical_time"]
    for row in measured:
        digital_growth = row["classical_total_flop_proxy"] / float(min_digital_work)
        physical_growth = row["thermo_physical_time"] / base_physical
        row["digital_work_growth_vs_min_width"] = digital_growth
        row["thermo_physical_growth_vs_min_width"] = physical_growth
        row["modeled_latency_advantage_factor"] = digital_growth / max(physical_growth, 1e-12)

    projected: list[dict[str, Any]] = []
    for width in projected_widths:
        ffn_work = _dense_ffn_flop_proxy(eval_batch, seq_len, embed_dim, width)
        digital_work = attention_work + ffn_work
        digital_growth = digital_work / float(min_digital_work)
        projected.append(
            {
                "width": width,
                "thermo_physical_time": t_f,
                "attention_flop_proxy": attention_work,
                "classical_ffn_flop_proxy": ffn_work,
                "classical_total_flop_proxy": digital_work,
                "digital_work_growth_vs_min_width": digital_growth,
                "thermo_physical_growth_vs_min_width": 1.0,
                "modeled_latency_advantage_factor": digital_growth,
            }
        )

    all_rows = measured + projected
    result = ExperimentResult(
        "superiority_demo",
        {
            "device": str(device),
            "claim": "Modeled thermodynamic physical inference time stays fixed as width grows; dense digital FFN work grows with width.",
            "state_of_art_context": {
                "attention": "FlashAttention-style kernels reduce memory traffic and improve wall time for exact attention, but they do not remove the width-dependent dense FFN work inside transformer blocks.",
                "dense_ffn_lower_bound": "A dense digital FFN must touch width-dependent weights and activations unless it changes the model class with sparsity, low-rank structure, quantization, or approximation.",
                "thermodynamic_model": "The thermodynamic layer maps extra width to extra parallel physical units and reads them after the same fixed t_f window.",
                "demo_scope": "This is an inference-scaling demonstration, not a training-speed benchmark and not a hardware measurement.",
            },
            "not_claimed": [
                "PyTorch wall time is faster than all state-of-the-art kernels.",
                "Training is constant time.",
                "Real chip speedup is proven without hardware.",
            ],
            "references": {
                "flash_attention": "https://arxiv.org/abs/2205.14135",
                "flash_attention_3": "https://arxiv.org/abs/2407.08608",
                "ddpm_iterative_sampling": "https://arxiv.org/abs/2006.11239",
                "parallel_tempering_review": "https://pubs.rsc.org/en/content/articlepdf/2005/cp/b509983h",
            },
            "measured_rows": measured,
            "projected_rows": projected,
            "max_measured_modeled_latency_advantage_factor": max(r["modeled_latency_advantage_factor"] for r in measured),
            "max_projected_modeled_latency_advantage_factor": max(r["modeled_latency_advantage_factor"] for r in projected),
            "physical_time_range_measured": max(r["thermo_physical_time"] for r in measured) - min(r["thermo_physical_time"] for r in measured),
        },
    )
    result.to_json(out / "superiority_demo.json")
    _write_summary(result, out)
    _try_plot(result, out)
    return result


def _write_summary(result: ExperimentResult, out: Path) -> None:
    measured = result.metrics["measured_rows"]
    projected = result.metrics["projected_rows"]
    lines = [
        "# Thermocompute Superiority Demo",
        "",
        result.metrics["claim"],
        "",
        "This demo compares modeled thermodynamic physical latency against a dense digital FFN work proxy.",
        "It reports PyTorch wall time separately and does not claim emulator wall time beats optimized kernels.",
        "",
        "## State-of-the-Art Context",
        "",
        "- Optimized attention kernels can make exact attention much faster, but transformer FFNs still carry width-dependent dense work.",
        "- Dense digital FFNs must touch width-dependent weights and activations unless the model class changes through sparsity, low-rank structure, quantization, or approximation.",
        "- The thermodynamic model maps extra width to extra parallel physical units and reads them after the same fixed observation window.",
        "- This is an inference-scaling demo, not a training-speed benchmark and not a silicon measurement.",
        "",
        "## Measured Widths",
        "",
        "| width | thermo physical time | digital work growth | modeled advantage | thermo MSE | classical MSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in measured:
        lines.append(
            f"| {row['width']} | {row['thermo_physical_time']:.3f} | "
            f"{row['digital_work_growth_vs_min_width']:.1f}x | "
            f"{row['modeled_latency_advantage_factor']:.1f}x | "
            f"{row['thermo_eval_mse']:.4f} | {row['classical_eval_mse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Projected Widths",
            "",
            "| width | thermo physical time | digital work growth | modeled advantage |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in projected:
        lines.append(
            f"| {row['width']} | {row['thermo_physical_time']:.3f} | "
            f"{row['digital_work_growth_vs_min_width']:.1f}x | "
            f"{row['modeled_latency_advantage_factor']:.1f}x |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Strongly supported: fixed modeled physical time as width grows.",
            "- Strongly supported: dense digital FFN work grows with width.",
            "- Not claimed: PyTorch emulator wall time is faster than all production kernels.",
            "- Not claimed: training is constant time.",
        ]
    )
    (out / "superiority_demo.md").write_text("\n".join(lines), encoding="utf-8")


def _try_plot(result: ExperimentResult, out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    measured = result.metrics["measured_rows"]
    projected = result.metrics["projected_rows"]
    all_rows = measured + projected
    widths = [r["width"] for r in all_rows]
    physical = [r["thermo_physical_time"] for r in all_rows]
    digital_growth = [r["digital_work_growth_vs_min_width"] for r in all_rows]
    advantage = [r["modeled_latency_advantage_factor"] for r in all_rows]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(widths, physical, marker="o", label="thermo physical time")
    ax1.set_xlabel("thermodynamic FFN width")
    ax1.set_ylabel("modeled physical time")
    ax1.set_xscale("log", base=2)
    ax2 = ax1.twinx()
    ax2.plot(widths, digital_growth, color="tab:red", marker="s", label="digital work growth")
    ax2.set_ylabel("digital work growth vs min width")
    ax2.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "superiority_latency_advantage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(widths, advantage, marker="o")
    ax.set_xlabel("thermodynamic FFN width")
    ax.set_ylabel("modeled latency advantage factor")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out / "superiority_advantage_factor.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["width"] for r in measured], [r["thermo_eval_mse"] for r in measured], marker="o", label="thermo")
    ax.plot([r["width"] for r in measured], [r["classical_eval_mse"] for r in measured], marker="s", label="classical")
    ax.set_xlabel("width")
    ax.set_ylabel("eval MSE after matched readout fit")
    ax.set_xscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "superiority_loss_vs_width.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cfg = DeviceConfig.auto(prefer_cuda=not args.cpu)
    result = run_superiority_demo(args.outdir, cfg)
    print(json.dumps({"name": result.name, "metrics": result.metrics}, indent=2))


if __name__ == "__main__":
    main()
