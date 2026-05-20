from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import Tensor
import torch.nn.functional as F

from thermocompute import (
    QuantizationConfig,
    QuantizedThermodynamicFFN,
    ThermodynamicFFN,
    ThermodynamicNeuronConfig,
    available_numeric_formats,
    estimate_classical_ffn_memory,
    estimate_quantized_thermo_ffn_memory,
    estimate_thermo_ffn_memory,
    set_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare standard, mixed-precision, and low-bit thermodynamic FFN training."
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.steps <= 0 or args.repeats <= 0:
        raise ValueError("steps and repeats must be positive")

    device = _choose_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    methods = _methods(device)

    rows: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        rows.extend(_run_repeat(repeat, methods, args.steps, device))

    summary = _summarize(rows)
    payload = {
        "name": "precision_training_comparison",
        "device": str(device),
        "steps": args.steps,
        "repeats": args.repeats,
        "methods": [method["name"] for method in methods],
        "rows": rows,
        "summary": summary,
        "memory_estimates": _memory_estimates(),
        "claim_boundary": (
            "mixed precision uses autocast-style lower precision forward math with fp32 master parameters; "
            "quantized methods use fake-quantized forward passes with straight-through gradients, not native packed int4 training kernels."
        ),
    }
    out = args.outdir / "precision_training_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _run_repeat(
    repeat: int,
    methods: list[dict[str, Any]],
    steps: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    set_seed(7000 + repeat)
    embed_dim = 8
    hidden_dim = 24
    batch = 8
    seq_len = 4
    neuron = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    x_train = torch.randn(batch, seq_len, embed_dim, device=device)
    x_eval = torch.randn(batch, seq_len, embed_dim, device=device)
    y_train = _target(x_train)
    y_eval = _target(x_eval)
    base = ThermodynamicFFN(embed_dim, hidden_dim, neuron_config=neuron, memory_efficient_chunk_size=11).to(device)
    base_state = copy.deepcopy(base.state_dict())

    rows = []
    for method in methods:
        set_seed(9000 + repeat)
        model = _build_model(method, embed_dim, hidden_dim, neuron, base_state, device)
        row = _fit_one(model, method, x_train, y_train, x_eval, y_eval, steps)
        row["repeat"] = repeat
        rows.append(row)
    return rows


def _methods(device: torch.device) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = [
        {"name": "standard_fp32", "kind": "standard"},
    ]
    if _autocast_supported(device, torch.bfloat16):
        methods.append({"name": "mixed_autocast_bf16", "kind": "mixed", "autocast_dtype": torch.bfloat16})
    if _autocast_supported(device, torch.float16):
        methods.append({"name": "mixed_autocast_fp16", "kind": "mixed", "autocast_dtype": torch.float16})
    formats = ["fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "int8", "int4", "int2", "binary"]
    available = set(available_numeric_formats())
    for fmt in formats:
        if fmt in available:
            methods.append({"name": f"quant_{fmt}", "kind": "quantized", "format": fmt})
    return methods


def _build_model(
    method: dict[str, Any],
    embed_dim: int,
    hidden_dim: int,
    neuron: ThermodynamicNeuronConfig,
    base_state: dict[str, Tensor],
    device: torch.device,
) -> torch.nn.Module:
    if method["kind"] == "quantized":
        model = QuantizedThermodynamicFFN(
            embed_dim,
            hidden_dim,
            quantization=QuantizationConfig(
                format=method["format"],
                compute_dtype=torch.float32,
                per_channel=True,
            ),
            neuron_config=neuron,
            memory_efficient_chunk_size=11,
        ).to(device)
    else:
        model = ThermodynamicFFN(embed_dim, hidden_dim, neuron_config=neuron, memory_efficient_chunk_size=11).to(device)
    model.load_state_dict(base_state)
    return model


def _fit_one(
    model: torch.nn.Module,
    method: dict[str, Any],
    x_train: Tensor,
    y_train: Tensor,
    x_eval: Tensor,
    y_eval: Tensor,
    steps: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    context = _autocast_context(x_train.device, method.get("autocast_dtype"))
    model.train()
    initial_train = _loss(model, x_train, y_train, context)
    initial_eval = _loss(model, x_eval, y_eval, context)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with context:
            pred = model(x_train)
            loss = F.mse_loss(_as_tensor(pred).float(), y_train.float())
        loss.backward()
        optimizer.step()
    wall_ms = (time.perf_counter() - start) * 1000.0
    model.eval()
    final_train = _loss(model, x_train, y_train, context)
    final_eval = _loss(model, x_eval, y_eval, context)
    return {
        "method": method["name"],
        "kind": method["kind"],
        "format": method.get("format", str(method.get("autocast_dtype", "fp32")).replace("torch.", "")),
        "initial_train_loss": initial_train,
        "final_train_loss": final_train,
        "train_loss_delta": initial_train - final_train,
        "initial_eval_loss": initial_eval,
        "final_eval_loss": final_eval,
        "eval_loss_delta": initial_eval - final_eval,
        "wall_ms": wall_ms,
        "finite": bool(math.isfinite(final_train) and math.isfinite(final_eval)),
    }


def _loss(model: torch.nn.Module, x: Tensor, y: Tensor, context: Any) -> float:
    with torch.no_grad():
        with context:
            pred = model(x)
            loss = F.mse_loss(_as_tensor(pred).float(), y.float())
    return float(loss.detach().cpu())


def _target(x: Tensor) -> Tensor:
    return 0.55 * torch.sin(1.7 * x) + 0.25 * torch.cos(0.8 * x) + 0.12 * x.square().tanh()


def _summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    methods = sorted({row["method"] for row in rows})
    summary: dict[str, dict[str, float]] = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        final_eval = [float(row["final_eval_loss"]) for row in subset]
        final_train = [float(row["final_train_loss"]) for row in subset]
        wall = [float(row["wall_ms"]) for row in subset]
        train_delta = [float(row["train_loss_delta"]) for row in subset]
        summary[method] = {
            "final_eval_loss_mean": mean(final_eval),
            "final_eval_loss_std": pstdev(final_eval) if len(final_eval) > 1 else 0.0,
            "final_train_loss_mean": mean(final_train),
            "train_loss_delta_mean": mean(train_delta),
            "wall_ms_mean": mean(wall),
        }
    fp32 = summary.get("standard_fp32")
    if fp32 is not None:
        fp32_loss = fp32["final_eval_loss_mean"]
        for stats in summary.values():
            stats["eval_loss_ratio_vs_standard_fp32"] = stats["final_eval_loss_mean"] / fp32_loss
    return summary


def _memory_estimates() -> dict[str, dict[str, float | int | str]]:
    input_dim = 8
    hidden_dim = 24
    batch_tokens = 32
    classical = estimate_classical_ffn_memory(input_dim, hidden_dim, batch_tokens=batch_tokens, dtype_bytes=4)
    thermo = estimate_thermo_ffn_memory(input_dim, hidden_dim, batch_tokens=batch_tokens, dtype_bytes=4, chunk_size=11)
    estimates: dict[str, dict[str, float | int | str]] = {
        "standard_fp32": {
            "format": "fp32",
            "peak_bytes": classical.peak_bytes,
            "parameter_bytes": classical.parameter_bytes,
        },
        "thermo_fp32_chunked": {
            "format": "fp32",
            "peak_bytes": thermo.peak_bytes,
            "parameter_bytes": thermo.parameter_bytes,
        },
    }
    for fmt in ("fp16", "bf16", "int8", "int4", "int2", "binary"):
        estimate = estimate_quantized_thermo_ffn_memory(
            input_dim,
            hidden_dim,
            batch_tokens=batch_tokens,
            parameter_format=fmt,
            state_format="fp16",
            output_format="fp16",
            chunk_size=11,
        )
        estimates[f"quant_{fmt}"] = {
            "format": fmt,
            "peak_bytes": estimate.peak_bytes,
            "parameter_bits": estimate.parameter_bits,
        }
    return estimates


def _autocast_supported(device: torch.device, dtype: torch.dtype) -> bool:
    try:
        x = torch.ones(2, 2, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype):
            y = x @ x
        return torch.isfinite(y).all().item()
    except (RuntimeError, TypeError):
        return False


def _autocast_context(device: torch.device, dtype: torch.dtype | None) -> Any:
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _as_tensor(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _choose_device(requested: str) -> torch.device:
    if requested == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        torch.cuda.empty_cache()
        free_bytes, _ = torch.cuda.mem_get_info()
        if requested == "cuda" and free_bytes < 512 * 1024**2:
            raise RuntimeError("requested CUDA but less than 512 MiB is free")
        if requested == "auto" and free_bytes < 1536 * 1024**2:
            return torch.device("cpu")
    except RuntimeError:
        if requested == "cuda":
            raise
        return torch.device("cpu")
    return torch.device("cuda")


if __name__ == "__main__":
    raise SystemExit(main())
