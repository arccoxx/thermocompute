from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from thermocompute import (
    QuantizationConfig,
    QuantizedThermodynamicFFN,
    ThermodynamicNeuronConfig,
    available_numeric_formats,
    estimate_quantized_thermo_ffn_memory,
    fit_quantized_ffn_mse,
    quantize_tensor,
    set_seed,
)


DEFAULT_FORMATS = ["fp32", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2", "int8", "int4", "int2", "binary"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tiny low-precision thermodynamic FFN experiments.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    set_seed(2026)
    device = _choose_device(args.device)
    formats = [name for name in DEFAULT_FORMATS if name in available_numeric_formats()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    x_probe = torch.linspace(-2.0, 2.0, 65, device=device, dtype=torch.float32, requires_grad=True)
    quantization_checks = []
    for name in formats:
        start = time.perf_counter()
        q = quantize_tensor(x_probe, QuantizationConfig(format=name, compute_dtype=torch.float32))
        q.square().mean().backward(retain_graph=True)
        quantization_checks.append(
            {
                "format": name,
                "max_abs_error": float((q - x_probe.detach()).abs().max().detach().cpu()),
                "finite": bool(torch.isfinite(q).all().detach().cpu()),
                "wall_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
    x_probe.grad = None

    inputs = torch.randn(4, 3, 6, device=device)
    targets = torch.sin(inputs)
    neuron = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    training_checks = []
    for name in formats:
        set_seed(100 + len(training_checks))
        model = QuantizedThermodynamicFFN(
            6,
            16,
            quantization=QuantizationConfig(format=name, compute_dtype=torch.float32, per_channel=True),
            neuron_config=neuron,
            memory_efficient_chunk_size=7,
        ).to(device)
        with torch.no_grad():
            y, info = model(inputs, return_info=True)
            forward_loss = float(F.mse_loss(y, targets).detach().cpu())
        result = fit_quantized_ffn_mse(model, inputs, targets, n_steps=args.steps, learning_rate=4e-3)
        training_checks.append(
            {
                "format": name,
                "forward_loss": forward_loss,
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "loss_delta": result.initial_loss - result.final_loss,
                "fit_wall_ms": result.fit_wall_ms,
                "physical_time": info.physical_time,
            }
        )

    memory = {
        "int8": _memory_row("int8"),
        "int4": _memory_row("int4"),
        "binary": _memory_row("binary"),
    }
    payload = {
        "name": "precision_experiments",
        "device": str(device),
        "formats": formats,
        "quantization_checks": quantization_checks,
        "training_checks": training_checks,
        "memory": memory,
    }
    out = args.outdir / "precision_experiments.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _memory_row(format_name: str) -> dict[str, int | str]:
    estimate = estimate_quantized_thermo_ffn_memory(
        64,
        4096,
        batch_tokens=128,
        parameter_format=format_name,
        state_format="fp16",
        output_format="fp16",
        chunk_size=256,
    )
    return {
        "parameter_format": estimate.parameter_format,
        "parameter_bits": estimate.parameter_bits,
        "state_bits": estimate.state_bits,
        "peak_bytes": estimate.peak_bytes,
    }


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
