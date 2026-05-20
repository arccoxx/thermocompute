from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from thermocompute import (
    DeviceConfig,
    DistributionAdapter,
    DistributionSampler,
    QuantizationConfig,
    QuantizedThermodynamicFFN,
    ThermodynamicFFN,
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerLayer,
    estimate_classical_ffn_memory,
    estimate_quantized_thermo_ffn_memory,
    estimate_thermo_ffn_memory,
    fit_quantized_ffn_mse,
    quantize_tensor,
    fit_transformer_end_to_end_cold,
    set_seed,
)


def main() -> int:
    cfg = _safe_device_config()
    device, dtype = cfg.device, cfg.dtype
    set_seed(911)
    metrics: dict[str, object] = {"device": str(device)}

    _stress_deterministic_equivalence(metrics, device, dtype)
    _stress_wide_no_replica_chunked_inference(metrics, device, dtype)
    _stress_chunked_cold_training(metrics, device, dtype)
    _stress_memory_laws(metrics)
    _stress_distributions(metrics, device, dtype)
    _stress_low_precision(metrics)
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


def _stress_distributions(metrics: dict[str, object], device: torch.device, dtype: torch.dtype) -> None:
    families = {
        "normal": {"loc": torch.zeros(4, device=device, dtype=dtype), "scale": torch.ones(4, device=device, dtype=dtype)},
        "beta": {
            "concentration1": torch.ones(4, device=device, dtype=dtype) * 2.0,
            "concentration0": torch.ones(4, device=device, dtype=dtype) * 3.0,
        },
        "gamma": {
            "concentration": torch.ones(4, device=device, dtype=dtype) * 2.0,
            "rate": torch.ones(4, device=device, dtype=dtype),
        },
        "poisson": {"rate": torch.ones(4, device=device, dtype=dtype) * 3.0},
        "categorical": {"logits": torch.zeros(4, 5, device=device, dtype=dtype)},
        "student_t": {
            "df": torch.ones(4, device=device, dtype=dtype) * 5.0,
            "loc": torch.zeros(4, device=device, dtype=dtype),
            "scale": torch.ones(4, device=device, dtype=dtype),
        },
    }
    checked = 0
    for name, params in families.items():
        sampler = DistributionSampler(name, **params)
        samples = sampler.sample(32)
        log_prob = sampler.log_prob(samples)
        if samples.shape[0] != 32:
            raise AssertionError(f"{name} stress sample has wrong leading dimension")
        if not torch.isfinite(log_prob.float()).all():
            raise AssertionError(f"{name} stress log_prob has non-finite values")
        checked += 1
    adapter = DistributionAdapter(torch.distributions.Laplace(torch.zeros(3, device=device), torch.ones(3, device=device)))
    adapted = adapter.sample(16)
    if adapted.shape != (16, 3):
        raise AssertionError("custom distribution adapter returned wrong shape")
    metrics["distribution_families_checked"] = checked
    metrics["distribution_adapter_checked"] = True


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


def _stress_low_precision(metrics: dict[str, object]) -> None:
    device = torch.device("cpu")
    x = torch.linspace(-1.0, 1.0, 33, device=device, requires_grad=True)
    checked = 0
    for name in ("fp32", "fp16", "bf16", "int8", "int4", "int2", "binary"):
        q = quantize_tensor(x, QuantizationConfig(format=name, compute_dtype=torch.float32))
        if q.shape != x.shape or not torch.isfinite(q).all():
            raise AssertionError(f"low precision quantization failed for {name}")
        checked += 1
    q4 = quantize_tensor(x, QuantizationConfig(format="int4", compute_dtype=torch.float32))
    q4.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("int4 STE gradient failed")

    config = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    model = QuantizedThermodynamicFFN(
        4,
        12,
        quantization=QuantizationConfig(format="int4", compute_dtype=torch.float32, per_channel=True),
        neuron_config=config,
        memory_efficient_chunk_size=5,
    )
    inputs = torch.randn(4, 2, 4)
    targets = torch.zeros_like(inputs)
    result = fit_quantized_ffn_mse(model, inputs, targets, n_steps=6, learning_rate=5e-3)
    if result.final_loss >= result.initial_loss:
        raise AssertionError("quantized int4 FFN training did not reduce loss")
    estimate = estimate_quantized_thermo_ffn_memory(
        4,
        1024,
        batch_tokens=8,
        parameter_format="int4",
        state_format="fp16",
        chunk_size=64,
    )
    metrics["low_precision_formats_checked"] = checked
    metrics["low_precision_int4_initial_loss"] = result.initial_loss
    metrics["low_precision_int4_final_loss"] = result.final_loss
    metrics["low_precision_int4_peak_bytes"] = estimate.peak_bytes


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _safe_device_config() -> DeviceConfig:
    if torch.cuda.is_available():
        smi = _nvidia_smi_memory()
        if smi is not None:
            used_mb, total_mb, utilization = smi
            if total_mb > 0 and (used_mb / total_mb > 0.75 or utilization > 90):
                return DeviceConfig.auto(prefer_cuda=False)
        try:
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info()
            if free_bytes >= 1536 * 1024**2:
                return DeviceConfig.auto()
        except RuntimeError:
            pass
    return DeviceConfig.auto(prefer_cuda=False)


def _nvidia_smi_memory() -> tuple[float, float, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
