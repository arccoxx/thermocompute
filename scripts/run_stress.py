from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from thermocompute import (
    DeviceConfig,
    ThermodynamicFFN,
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerLayer,
    estimate_classical_ffn_memory,
    estimate_thermo_ffn_memory,
    fit_transformer_end_to_end_cold,
    set_seed,
)


def main() -> int:
    cfg = DeviceConfig.auto()
    device, dtype = cfg.device, cfg.dtype
    set_seed(911)
    metrics: dict[str, object] = {"device": str(device)}

    _stress_deterministic_equivalence(metrics, device, dtype)
    _stress_wide_no_replica_chunked_inference(metrics, device, dtype)
    _stress_chunked_cold_training(metrics, device, dtype)
    _stress_memory_laws(metrics)
    _stress_invalid_chunk_sizes(metrics)

    print(json.dumps({"name": "stress_checks", "metrics": metrics}, indent=2))
    return 0


def _stress_deterministic_equivalence(metrics: dict[str, object], device: torch.device, dtype: torch.dtype) -> None:
    x = torch.randn(3, 5, 16, device=device, dtype=dtype)
    config = ThermodynamicNeuronConfig(t_f=0.12, dt=0.04, temperature=0.0)
    full = ThermodynamicFFN(16, 96, neuron_config=config).to(device=device, dtype=dtype)
    chunked = ThermodynamicFFN(16, 96, neuron_config=config, memory_efficient_chunk_size=17).to(device=device, dtype=dtype)
    chunked.load_state_dict(full.state_dict())
    with torch.no_grad():
        y_full = full(x)
        y_chunked = chunked(x)
    max_diff = torch.max(torch.abs(y_full - y_chunked)).detach().cpu().item()
    if max_diff > 2e-5:
        raise AssertionError(f"chunked deterministic FFN mismatch: {max_diff}")
    metrics["deterministic_chunked_ffn_max_abs_diff"] = max_diff


def _stress_wide_no_replica_chunked_inference(
    metrics: dict[str, object],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    width = 8192 if device.type == "cuda" else 1024
    chunk = 512 if device.type == "cuda" else 128
    batch = 4 if device.type == "cuda" else 2
    seq_len = 16 if device.type == "cuda" else 8
    embed_dim = 64
    layer = ThermodynamicTransformerLayer(
        embed_dim,
        4,
        thermo_hidden_dim=width,
        t_f=0.12,
        dt=0.04,
        memory_efficient_chunk_size=chunk,
    ).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, embed_dim, device=device, dtype=dtype)
    _sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        y, info = layer(x, return_info=True)
    _sync(device)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if y.shape != x.shape:
        raise AssertionError(f"wide chunked output shape mismatch: {tuple(y.shape)} != {tuple(x.shape)}")
    if not torch.isfinite(y).all():
        raise AssertionError("wide chunked inference produced non-finite output")
    if layer.thermo_ff.n_replicas != 1 or layer.thermo_ff.tempering:
        raise AssertionError("stress path must stay no-replica and no-tempering")
    metrics["wide_chunked_width"] = width
    metrics["wide_chunked_chunk_size"] = chunk
    metrics["wide_chunked_wall_ms"] = wall_ms
    metrics["wide_chunked_physical_time"] = info.physical_time
    metrics["wide_chunked_output_std"] = float(y.std().detach().cpu())


def _stress_chunked_cold_training(metrics: dict[str, object], device: torch.device, dtype: torch.dtype) -> None:
    x = torch.randn(8, 4, 12, device=device, dtype=dtype)
    target = torch.zeros_like(x)
    layer = ThermodynamicTransformerLayer(
        12,
        3,
        thermo_hidden_dim=80,
        t_f=0.1,
        dt=0.05,
        temperature=0.0,
        memory_efficient_chunk_size=19,
    ).to(device=device, dtype=dtype)
    result = fit_transformer_end_to_end_cold(
        layer,
        x,
        target,
        n_steps=6,
        learning_rate=3e-3,
    )
    if result.memory_replicas != 1:
        raise AssertionError("cold stress training used more than one memory replica")
    if result.final_train_loss >= result.initial_train_loss:
        raise AssertionError("chunked cold training did not reduce loss")
    metrics["chunked_cold_initial_loss"] = result.initial_train_loss
    metrics["chunked_cold_final_loss"] = result.final_train_loss
    metrics["chunked_cold_wall_ms"] = result.fit_wall_ms


def _stress_memory_laws(metrics: dict[str, object]) -> None:
    classical = estimate_classical_ffn_memory(4096, 500_000, batch_tokens=2048, dtype_bytes=2)
    thermo_full = estimate_thermo_ffn_memory(
        4096,
        500_000,
        batch_tokens=2048,
        dtype_bytes=2,
        replicas=1,
    )
    thermo_chunked = estimate_thermo_ffn_memory(
        4096,
        500_000,
        batch_tokens=2048,
        dtype_bytes=2,
        replicas=1,
        chunk_size=8192,
    )
    if thermo_chunked.parameter_bytes != thermo_full.parameter_bytes:
        raise AssertionError("chunking should not change parameter memory")
    if thermo_chunked.state_bytes >= thermo_full.state_bytes:
        raise AssertionError("chunking should reduce thermodynamic state memory")
    metrics["memory_law_classical_peak_gb"] = classical.peak_bytes / 1e9
    metrics["memory_law_thermo_full_peak_gb"] = thermo_full.peak_bytes / 1e9
    metrics["memory_law_thermo_chunked_peak_gb"] = thermo_chunked.peak_bytes / 1e9
    metrics["memory_law_chunked_state_reduction"] = thermo_full.state_bytes / thermo_chunked.state_bytes


def _stress_invalid_chunk_sizes(metrics: dict[str, object]) -> None:
    failures = 0
    try:
        ThermodynamicFFN(8, 16, memory_efficient_chunk_size=0)
    except ValueError:
        failures += 1
    try:
        ThermodynamicTransformerLayer(8, 2, thermo_hidden_dim=16, memory_efficient_chunk_size=-1)
    except ValueError:
        failures += 1
    layer = ThermodynamicTransformerLayer(8, 2, thermo_hidden_dim=16)
    try:
        layer(torch.randn(1, 2, 8), chunk_size=0)
    except ValueError:
        failures += 1
    if failures != 3:
        raise AssertionError(f"expected 3 invalid chunk failures, got {failures}")
    metrics["invalid_chunk_checks"] = failures


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


if __name__ == "__main__":
    raise SystemExit(main())
