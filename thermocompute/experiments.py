from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .config import DeviceConfig, set_seed
from .metrics import ExperimentResult
from .neurons import ThermodynamicMLP, ThermodynamicNeuronLayer, quartic_potential
from .primitives import BinaryPBit, CategoricalPDIT, IsingEnergy, PMODE, PMOG
from .training import (
    fit_transformer_end_to_end_cold,
    fit_transformer_end_to_end_parallel_tempering,
    fit_transformer_readout_parallel_tempering,
    fit_transformer_readout_ridge,
)
from .transformer import ThermodynamicSelfAttention, ThermodynamicTransformerLayer


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def smoke_checks(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(7)
    device, dtype = cfg.device, cfg.dtype

    pbit = BinaryPBit(beta=2.0).to(device)
    control = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype)
    p = pbit.probabilities(control)
    draws = pbit.sample(control)

    pdit = CategoricalPDIT(beta=1.0).to(device)
    logits = torch.tensor([[0.0, 1.0, -1.0]], device=device, dtype=dtype)
    cat = pdit.sample(logits)

    pmode = PMODE().to(device)
    samples = pmode.sample(torch.zeros(16, device=device, dtype=dtype), 0.5, n_samples=8, t_total=5e-7)

    pmog = PMOG(3).to(device)
    ms, modes = pmog.sample(
        torch.tensor([0.0, 1.0, -0.5], device=device, dtype=dtype),
        torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype),
        torch.tensor([0.2, 0.3, 0.4], device=device, dtype=dtype),
        n_samples=32,
        t_total=5e-7,
    )

    layer = ThermodynamicNeuronLayer(2, 4, t_f=0.2, dt=0.05, n_replicas=2, tempering=True).to(device)
    y, info = layer(torch.randn(5, 2, device=device, dtype=dtype), return_info=True)
    transformer = ThermodynamicTransformerLayer(
        12,
        3,
        thermo_hidden_dim=48,
        attention_mode="pdit",
        n_attention_samples=2,
        attention_t_f=0.05,
        t_f=0.1,
        dt=0.05,
    ).to(device=device, dtype=dtype)
    tx, tinfo = transformer(torch.randn(2, 4, 12, device=device, dtype=dtype), causal=True, return_info=True)

    return ExperimentResult(
        name="smoke_checks",
        metrics={
            "device": str(device),
            "pbit_probabilities": [float(v) for v in p.detach().cpu()],
            "pbit_draw_shape": list(draws.shape),
            "pdit_sample": int(cat.detach().cpu().reshape(-1)[0]),
            "pmode_shape": list(samples.shape),
            "pmog_shape": list(ms.shape),
            "pmog_mode_shape": list(modes.shape),
            "layer_output_shape": list(y.shape),
            "layer_physical_time": info.physical_time,
            "tempering_swap_attempts": info.swap_attempts,
            "transformer_output_shape": list(tx.shape),
            "transformer_physical_time": tinfo.physical_time,
            "transformer_attention_mode": tinfo.attention_mode,
        },
    )


def proof_of_concept_checks(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(11)
    device, dtype = cfg.device, cfg.dtype

    pbit = BinaryPBit(beta=1.0).to(device)
    voltages = torch.linspace(-2, 2, 9, device=device, dtype=dtype)
    draws = torch.stack([pbit.sample(voltages) for _ in range(4000)])
    pbit_mae = (draws.float().mean(dim=0) - torch.sigmoid(voltages)).abs().mean()

    pmode = PMODE().to(device)
    mu = torch.tensor([0.75], device=device, dtype=dtype)
    sigma = torch.tensor([0.4], device=device, dtype=dtype)
    ou = pmode.sample(mu, sigma, n_samples=5000, t_total=1e-4, burnin=2e-6)
    pmode_mean_err = (ou.mean() - mu.squeeze()).abs()
    pmode_std_err = (ou.std(unbiased=False) - sigma.squeeze()).abs()

    pmog = PMOG(3).to(device)
    logits = torch.tensor([0.2, 1.1, -0.4], device=device, dtype=dtype)
    means = torch.tensor([-1.5, 0.25, 2.0], device=device, dtype=dtype)
    scales = torch.tensor([0.2, 0.35, 0.25], device=device, dtype=dtype)
    samples, modes = pmog.sample(logits, means, scales, n_samples=6000, t_total=6e-5)
    empirical_weights = torch.bincount(modes.reshape(-1).cpu(), minlength=3).float()
    empirical_weights = empirical_weights / empirical_weights.sum()
    target_weights = torch.softmax(logits.cpu(), dim=-1)
    pmog_weight_l1 = (empirical_weights - target_weights).abs().sum()
    pmog_mean_err = (samples.mean().cpu() - (target_weights * means.cpu()).sum()).abs()

    layer = ThermodynamicNeuronLayer(
        1,
        1,
        j4=1.25,
        temperature=0.05,
        t_f=1.0,
        dt=0.025,
        n_replicas=512,
        output="mean",
    ).to(device)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    currents = torch.linspace(-2.0, 2.0, 25, device=device, dtype=dtype).unsqueeze(-1)
    activations = layer(currents).squeeze(-1)
    monotonic_fraction = (activations[1:] >= activations[:-1]).float().mean()
    activation_corr = torch.corrcoef(torch.stack([currents.squeeze(-1).float(), activations.float()]))[0, 1]

    return ExperimentResult(
        name="proof_of_concept",
        metrics={
            "device": str(device),
            "pbit_probability_mae": float(pbit_mae.detach().cpu()),
            "pmode_mean_abs_error": float(pmode_mean_err.detach().cpu()),
            "pmode_std_abs_error": float(pmode_std_err.detach().cpu()),
            "pmog_weight_l1": float(pmog_weight_l1),
            "pmog_mean_abs_error": float(pmog_mean_err),
            "thermo_activation_monotonic_fraction": float(monotonic_fraction.detach().cpu()),
            "thermo_activation_input_corr": float(activation_corr.detach().cpu()),
        },
    )


def experiment_scaling(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(13)
    device, dtype = cfg.device, cfg.dtype
    widths = [8, 16, 32, 64, 128, 256]
    batch = 512 if device.type == "cuda" else 128
    rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        warm = ThermodynamicMLP([16, 32, 32, 8], t_f=0.4, dt=0.04, n_replicas=4, tempering=True).to(device=device, dtype=dtype)
        warm_x = torch.randn(batch, 16, device=device, dtype=dtype)
        for _ in range(5):
            warm(warm_x)
        _sync(device)

    for width in widths:
        model = ThermodynamicMLP([16, width, width, 8], t_f=0.4, dt=0.04, n_replicas=4, tempering=True).to(device=device, dtype=dtype)
        x = torch.randn(batch, 16, device=device, dtype=dtype)
        for _ in range(3):
            model(x)
        _sync(device)
        timings = []
        y = None
        info = None
        for _ in range(5):
            start = time.perf_counter()
            y, info = model(x, return_info=True)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        timings_t = torch.tensor(timings)
        assert y is not None and info is not None
        rows.append(
            {
                "width": width,
                "batch": batch,
                "wall_ms_median": float(timings_t.median()),
                "wall_ms_min": float(timings_t.min()),
                "wall_ms_max": float(timings_t.max()),
                "physical_time": info.physical_time,
                "swap_acceptance": info.swap_acceptance,
                "output_std": float(y.std(unbiased=False).detach().cpu()),
            }
        )

    physical_times = [r["physical_time"] for r in rows]
    return ExperimentResult(
        name="width_scaling_fixed_physical_time",
        metrics={
            "device": str(device),
            "rows": rows,
            "physical_time_min": min(physical_times),
            "physical_time_max": max(physical_times),
            "physical_time_range": max(physical_times) - min(physical_times),
        },
    )


def experiment_tempering_escape(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(17)
    device, dtype = cfg.device, cfg.dtype

    def run(n_replicas: int, tempering: bool) -> tuple[float, float, float, float]:
        batch = 8192 if device.type == "cuda" else 2048
        dt = 0.01
        steps = 160
        j2 = torch.tensor(-2.0, device=device, dtype=dtype)
        j3 = torch.tensor(0.0, device=device, dtype=dtype)
        j4 = torch.tensor(1.0, device=device, dtype=dtype)
        temps = torch.linspace(0.08, 5.0, n_replicas, device=device, dtype=dtype).view(n_replicas, 1)
        beta = 1.0 / temps
        state = torch.full((n_replicas, batch), -math.sqrt(2.0), device=device, dtype=dtype)
        attempts = 0
        accepts = 0
        for step in range(steps):
            force = -(j2 * state + j4 * state.pow(3))
            force = force.clamp(-80.0, 80.0)
            noise = torch.randn_like(state)
            state = state + dt * force + torch.sqrt(2.0 * temps * dt) * noise
            state = torch.nan_to_num(state, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
            if tempering and n_replicas > 1 and (step + 1) % 4 == 0:
                for start in (0, 1):
                    for r in range(start, n_replicas - 1, 2):
                        e_a = quartic_potential(state[r], torch.zeros_like(state[r]), j2, j3, j4)
                        e_b = quartic_potential(state[r + 1], torch.zeros_like(state[r + 1]), j2, j3, j4)
                        log_alpha = (beta[r] - beta[r + 1]) * (e_a - e_b)
                        accept = torch.log(torch.rand_like(log_alpha).clamp_min(1e-12)) < log_alpha.clamp_max(80.0)
                        old = state[r].clone()
                        state[r] = torch.where(accept, state[r + 1], state[r])
                        state[r + 1] = torch.where(accept, old, state[r + 1])
                        attempts += accept.numel()
                        accepts += int(accept.sum().detach().cpu())
        cold = state[0]
        right_fraction = float((cold > 0).float().mean().detach().cpu())
        well_coverage = 1.0 - abs(right_fraction - 0.5) * 2.0
        energy = quartic_potential(cold, torch.zeros_like(cold), j2, j3, j4).mean()
        swap_acceptance = float(accepts / attempts) if attempts else 0.0
        return right_fraction, well_coverage, float(energy.detach().cpu()), swap_acceptance

    no_temp_right, no_temp_coverage, no_temp_energy, _ = run(1, False)
    temp_right, temp_coverage, temp_energy, temp_accept = run(8, True)
    return ExperimentResult(
        name="parallel_tempering_double_well_escape",
        metrics={
            "device": str(device),
            "no_tempering_right_well_fraction": no_temp_right,
            "tempering_right_well_fraction": temp_right,
            "no_tempering_well_coverage": no_temp_coverage,
            "tempering_well_coverage": temp_coverage,
            "no_tempering_mean_energy": no_temp_energy,
            "tempering_mean_energy": temp_energy,
            "tempering_swap_acceptance": temp_accept,
            "well_coverage_improvement": temp_coverage - no_temp_coverage,
            "energy_improvement": no_temp_energy - temp_energy,
        },
    )


def experiment_pmog_density(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(19)
    device, dtype = cfg.device, cfg.dtype
    pmog = PMOG(4, beta=1.0).to(device)
    logits = torch.tensor([-0.2, 1.0, 0.4, -1.2], device=device, dtype=dtype)
    means = torch.tensor([-2.0, -0.35, 1.1, 2.6], device=device, dtype=dtype)
    scales = torch.tensor([0.25, 0.18, 0.35, 0.22], device=device, dtype=dtype)
    samples, modes = pmog.sample(logits, means, scales, n_samples=20000, t_total=2e-4, switch_rate=3e5)
    target_w = torch.softmax(logits.cpu(), dim=-1)
    empirical_w = torch.bincount(modes.reshape(-1).cpu(), minlength=4).float()
    empirical_w /= empirical_w.sum()
    target_mean = float((target_w * means.cpu()).sum())
    target_var = float((target_w * (scales.cpu().square() + means.cpu().square())).sum() - target_mean**2)
    empirical_mean = float(samples.mean().detach().cpu())
    empirical_var = float(samples.var(unbiased=False).detach().cpu())
    return ExperimentResult(
        name="pmog_multimodal_fidelity",
        metrics={
            "device": str(device),
            "target_weights": [float(v) for v in target_w],
            "empirical_weights": [float(v) for v in empirical_w],
            "weight_l1": float((target_w - empirical_w).abs().sum()),
            "target_mean": target_mean,
            "empirical_mean": empirical_mean,
            "mean_abs_error": abs(target_mean - empirical_mean),
            "target_var": target_var,
            "empirical_var": empirical_var,
            "var_abs_error": abs(target_var - empirical_var),
        },
    )


def experiment_transformer_width_scaling(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(23)
    device, dtype = cfg.device, cfg.dtype
    widths = [32, 64, 128, 256, 512, 1024]
    batch = 48 if device.type == "cuda" else 12
    seq_len = 24 if device.type == "cuda" else 12
    embed_dim = 32
    rows: list[dict[str, Any]] = []

    if device.type == "cuda":
        warm = ThermodynamicTransformerLayer(
            embed_dim,
            4,
            thermo_hidden_dim=128,
            t_f=0.2,
            dt=0.04,
            n_replicas=2,
        ).to(device=device, dtype=dtype)
        warm_x = torch.randn(batch, seq_len, embed_dim, device=device, dtype=dtype)
        for _ in range(4):
            warm(warm_x)
        _sync(device)

    for width in widths:
        layer = ThermodynamicTransformerLayer(
            embed_dim,
            4,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=2,
        ).to(device=device, dtype=dtype)
        x = torch.randn(batch, seq_len, embed_dim, device=device, dtype=dtype)
        for _ in range(2):
            layer(x)
        _sync(device)
        timings = []
        y = None
        info = None
        for _ in range(5):
            start = time.perf_counter()
            y, info = layer(x, return_info=True)
            _sync(device)
            timings.append((time.perf_counter() - start) * 1000.0)
        assert y is not None and info is not None
        timings_t = torch.tensor(timings)
        rows.append(
            {
                "thermo_hidden_dim": width,
                "batch": batch,
                "seq_len": seq_len,
                "embed_dim": embed_dim,
                "wall_ms_median": float(timings_t.median()),
                "wall_ms_min": float(timings_t.min()),
                "wall_ms_max": float(timings_t.max()),
                "physical_time": info.physical_time,
                "feedforward_physical_time": info.feedforward_physical_time,
                "output_std": float(y.std(unbiased=False).detach().cpu()),
            }
        )

    physical_times = [r["physical_time"] for r in rows]
    return ExperimentResult(
        name="transformer_width_constant_physical_time",
        metrics={
            "device": str(device),
            "rows": rows,
            "physical_time_min": min(physical_times),
            "physical_time_max": max(physical_times),
            "physical_time_range": max(physical_times) - min(physical_times),
        },
    )


def experiment_pdit_attention_convergence(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(29)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 32
    heads = 4
    batch = 32 if device.type == "cuda" else 8
    seq_len = 16
    x = torch.randn(batch, seq_len, embed_dim, device=device, dtype=dtype)

    teacher = ThermodynamicSelfAttention(embed_dim, heads, mode="softmax").to(device=device, dtype=dtype)
    with torch.no_grad():
        target = teacher(x)

    rows: list[dict[str, Any]] = []
    for n_samples in [1, 2, 4, 8, 16, 32]:
        sampler = ThermodynamicSelfAttention(
            embed_dim,
            heads,
            mode="pdit",
            n_attention_samples=n_samples,
            beta=1.0,
            attention_t_f=0.05,
        ).to(device=device, dtype=dtype)
        sampler.load_state_dict(teacher.state_dict())
        errors = []
        for _ in range(8):
            with torch.no_grad():
                sampled = sampler(x)
                errors.append(float(torch.mean((sampled - target).square()).detach().cpu()))
        rows.append(
            {
                "attention_samples": n_samples,
                "mse_to_softmax_mean": float(torch.tensor(errors).mean()),
                "mse_to_softmax_std": float(torch.tensor(errors).std(unbiased=False)),
                "attention_physical_time": sampler.physical_time,
            }
        )

    return ExperimentResult(
        name="pdit_attention_sampling_convergence",
        metrics={
            "device": str(device),
            "rows": rows,
            "best_mse": min(r["mse_to_softmax_mean"] for r in rows),
            "mse_improvement_1_to_32": rows[0]["mse_to_softmax_mean"] - rows[-1]["mse_to_softmax_mean"],
        },
    )


def experiment_transformer_readout_alignment(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(31)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 16
    heads = 4
    train_batch = 384 if device.type == "cuda" else 64
    eval_batch = 64 if device.type == "cuda" else 16
    seq_len = 8

    def target_fn(x: torch.Tensor) -> torch.Tensor:
        left = torch.roll(x, shifts=1, dims=1)
        right = torch.roll(x, shifts=-1, dims=1)
        return torch.sin(x + 0.35 * left) + 0.2 * torch.cos(right - x)

    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = target_fn(x_train)
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = target_fn(x_eval)
    rows: list[dict[str, Any]] = []

    for width in [32, 64, 128, 256, 512]:
        layer = ThermodynamicTransformerLayer(
            embed_dim,
            heads,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=4,
            thermo_output="mean",
            residual_scale=1.0,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            layer.thermo_ff.temperatures.fill_(0.08)
            before = torch.mean((layer(x_train) - y_train).square())
        fit = fit_transformer_readout_ridge(
            layer,
            x_train,
            y_train,
            ridge=1e-1,
            feature_repeats=2,
        )
        with torch.no_grad():
            features, base, _ = layer.thermodynamic_features(x_eval, return_base=True)
            pred_eval = base + layer.residual_scale * layer.out_proj(features)
            eval_mse = torch.mean((pred_eval - y_eval).square())
        rows.append(
            {
                "thermo_hidden_dim": width,
                "pre_fit_train_mse": float(before.detach().cpu()),
                "post_fit_train_mse": fit.train_mse,
                "eval_mse_single_shot": float(eval_mse.detach().cpu()),
                "fit_wall_ms": fit.fit_wall_ms,
                "fit_physical_time": fit.physical_time,
                "inference_physical_time": layer.physical_time,
                "feature_repeats": fit.feature_repeats,
            }
        )

    return ExperimentResult(
        name="thermodynamic_readout_alignment",
        metrics={
            "device": str(device),
            "method": "Thermodynamic Readout Alignment (closed-form ridge on fixed-time thermodynamic features)",
            "rows": rows,
            "best_train_mse": min(r["post_fit_train_mse"] for r in rows),
            "best_eval_mse_single_shot": min(r["eval_mse_single_shot"] for r in rows),
        },
    )


def experiment_parallel_tempered_mask_alignment(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(37)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 16
    heads = 4
    train_batch = 128 if device.type == "cuda" else 32
    eval_batch = 64 if device.type == "cuda" else 16
    seq_len = 8

    def target_fn(x: torch.Tensor) -> torch.Tensor:
        prev_token = torch.roll(x, shifts=1, dims=1)
        next_token = torch.roll(x, shifts=-1, dims=1)
        return torch.tanh(x + 0.45 * prev_token) + 0.15 * torch.sin(2.0 * next_token)

    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = target_fn(x_train)
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = target_fn(x_eval)
    rows: list[dict[str, Any]] = []

    for width in [64, 128, 256]:
        base = ThermodynamicTransformerLayer(
            embed_dim,
            heads,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=4,
            thermo_output="mean",
            residual_scale=1.0,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            base.thermo_ff.temperatures.fill_(0.08)

        dense = ThermodynamicTransformerLayer(
            embed_dim,
            heads,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=4,
            thermo_output="mean",
            residual_scale=1.0,
        ).to(device=device, dtype=dtype)
        dense.load_state_dict(base.state_dict())
        dense_fit = fit_transformer_readout_ridge(
            dense,
            x_train,
            y_train,
            ridge=1e-1,
            feature_repeats=2,
        )
        with torch.no_grad():
            dense_features, dense_base, _ = dense.thermodynamic_features(x_eval, return_base=True)
            dense_eval = torch.mean((dense_base + dense.out_proj(dense_features) - y_eval).square())

        sparse = ThermodynamicTransformerLayer(
            embed_dim,
            heads,
            thermo_hidden_dim=width,
            t_f=0.2,
            dt=0.04,
            n_replicas=4,
            thermo_output="mean",
            residual_scale=1.0,
        ).to(device=device, dtype=dtype)
        sparse.load_state_dict(base.state_dict())
        sparse_fit = fit_transformer_readout_parallel_tempering(
            sparse,
            x_train,
            y_train,
            ridge=1e-1,
            feature_repeats=2,
            keep_fraction=0.35,
            sparsity_penalty=1e-3,
            n_tempering_replicas=6,
            n_tempering_steps=18,
            flips_per_step=max(2, width // 64),
            min_temperature=0.02,
            max_temperature=0.8,
        )
        with torch.no_grad():
            sparse_features, sparse_base, _ = sparse.thermodynamic_features(x_eval, return_base=True)
            sparse_eval = torch.mean((sparse_base + sparse.out_proj(sparse_features) - y_eval).square())

        rows.append(
            {
                "thermo_hidden_dim": width,
                "dense_train_mse": dense_fit.train_mse,
                "dense_eval_mse_single_shot": float(dense_eval.detach().cpu()),
                "dense_fit_wall_ms": dense_fit.fit_wall_ms,
                "sparse_train_mse": sparse_fit.train_mse,
                "sparse_eval_mse_single_shot": float(sparse_eval.detach().cpu()),
                "sparse_fit_wall_ms": sparse_fit.fit_wall_ms,
                "selected_features": sparse_fit.selected_features,
                "selected_fraction": sparse_fit.selected_fraction,
                "swap_acceptance": sparse_fit.swap_acceptance,
                "swap_attempts": sparse_fit.swap_attempts,
                "inference_physical_time": sparse.physical_time,
                "fit_physical_time": sparse_fit.physical_time,
            }
        )

    return ExperimentResult(
        name="parallel_tempered_mask_alignment",
        metrics={
            "device": str(device),
            "method": "Parallel Tempered Mask Alignment (PT over sparse thermodynamic reservoir feature masks)",
            "rows": rows,
            "best_sparse_train_mse": min(r["sparse_train_mse"] for r in rows),
            "best_sparse_eval_mse_single_shot": min(r["sparse_eval_mse_single_shot"] for r in rows),
        },
    )


def experiment_end_to_end_parallel_tempered_training(device_config: DeviceConfig | None = None) -> ExperimentResult:
    cfg = device_config or DeviceConfig.auto()
    set_seed(41)
    device, dtype = cfg.device, cfg.dtype
    embed_dim = 8
    heads = 2
    seq_len = 6
    train_batch = 48 if device.type == "cuda" else 16
    eval_batch = 48 if device.type == "cuda" else 16

    def target_fn(x: torch.Tensor) -> torch.Tensor:
        prev_token = torch.roll(x, shifts=1, dims=1)
        return 0.55 * torch.sin(x + 0.4 * prev_token)

    x_train = torch.randn(train_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_train = target_fn(x_train)
    x_eval = torch.randn(eval_batch, seq_len, embed_dim, device=device, dtype=dtype)
    y_eval = target_fn(x_eval)

    base = ThermodynamicTransformerLayer(
        embed_dim,
        heads,
        thermo_hidden_dim=32,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
        residual_scale=0.7,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        base.thermo_ff.temperatures.fill_(0.01)

    cold = ThermodynamicTransformerLayer(
        embed_dim,
        heads,
        thermo_hidden_dim=32,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
        residual_scale=0.7,
    ).to(device=device, dtype=dtype)
    cold.load_state_dict(base.state_dict())
    cold_fit = fit_transformer_end_to_end_cold(
        cold,
        x_train,
        y_train,
        eval_inputs=x_eval,
        eval_targets=y_eval,
        n_steps=42,
        learning_rate=4e-3,
        grad_clip=1.0,
    )

    pt = ThermodynamicTransformerLayer(
        embed_dim,
        heads,
        thermo_hidden_dim=32,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
        residual_scale=0.7,
    ).to(device=device, dtype=dtype)
    pt.load_state_dict(base.state_dict())
    pt_fit = fit_transformer_end_to_end_parallel_tempering(
        pt,
        x_train,
        y_train,
        eval_inputs=x_eval,
        eval_targets=y_eval,
        n_tempering_replicas=5,
        n_tempering_steps=42,
        learning_rate=4e-3,
        noise_scale=2e-3,
        min_temperature=0.03,
        max_temperature=0.8,
        swap_interval=3,
        grad_clip=1.0,
    )

    return ExperimentResult(
        name="end_to_end_parallel_tempered_training",
        metrics={
            "device": str(device),
            "method": "End-to-end inductive transformer training with parallel-tempered parameter replicas; no ridge solve",
            "cold_single_replica": {
                "initial_train_loss": cold_fit.initial_train_loss,
                "final_train_loss": cold_fit.final_train_loss,
                "final_eval_loss": cold_fit.final_eval_loss,
                "fit_wall_ms": cold_fit.fit_wall_ms,
                "physical_time_per_forward": cold_fit.physical_time_per_forward,
                "memory_replicas": cold_fit.memory_replicas,
            },
            "parallel_tempered": {
                "initial_train_loss": pt_fit.initial_train_loss,
                "final_train_loss": pt_fit.final_train_loss,
                "final_eval_loss": pt_fit.final_eval_loss,
                "fit_wall_ms": pt_fit.fit_wall_ms,
                "physical_time_per_forward": pt_fit.physical_time_per_forward,
                "n_tempering_replicas": pt_fit.n_tempering_replicas,
                "swap_attempts": pt_fit.swap_attempts,
                "swap_acceptance": pt_fit.swap_acceptance,
                "best_replica_temperature": pt_fit.best_replica_temperature,
            },
            "pt_eval_delta_vs_cold": (
                float(pt_fit.final_eval_loss - cold_fit.final_eval_loss)
                if pt_fit.final_eval_loss is not None and cold_fit.final_eval_loss is not None
                else None
            ),
        },
    )


def run_all_experiments(outdir: str | Path = "artifacts", device_config: DeviceConfig | None = None) -> list[ExperimentResult]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    results = [
        experiment_scaling(device_config),
        experiment_tempering_escape(device_config),
        experiment_pmog_density(device_config),
        experiment_transformer_width_scaling(device_config),
        experiment_pdit_attention_convergence(device_config),
        experiment_transformer_readout_alignment(device_config),
        experiment_parallel_tempered_mask_alignment(device_config),
        experiment_end_to_end_parallel_tempered_training(device_config),
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

    scaling = next((r for r in results if r.name == "width_scaling_fixed_physical_time"), None)
    if scaling is not None:
        rows = scaling.metrics["rows"]
        widths = [r["width"] for r in rows]
        wall = [r["wall_ms_median"] for r in rows]
        phys = [r["physical_time"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(widths, wall, marker="o", label="wall time (ms)")
        ax1.set_xlabel("width")
        ax1.set_ylabel("PyTorch wall time (ms)")
        ax2 = ax1.twinx()
        ax2.plot(widths, phys, color="tab:orange", marker="s", label="physical time")
        ax2.set_ylabel("emulated physical time")
        ax1.set_xscale("log", base=2)
        fig.tight_layout()
        fig.savefig(out / "width_scaling_fixed_physical_time.png", dpi=160)
        plt.close(fig)

    pmog = next((r for r in results if r.name == "pmog_multimodal_fidelity"), None)
    if pmog is not None:
        x = torch.arange(len(pmog.metrics["target_weights"]))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar((x - 0.18).numpy(), pmog.metrics["target_weights"], width=0.35, label="target")
        ax.bar((x + 0.18).numpy(), pmog.metrics["empirical_weights"], width=0.35, label="empirical")
        ax.set_xlabel("component")
        ax.set_ylabel("weight")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "pmog_multimodal_fidelity.png", dpi=160)
        plt.close(fig)

    transformer_scaling = next((r for r in results if r.name == "transformer_width_constant_physical_time"), None)
    if transformer_scaling is not None:
        rows = transformer_scaling.metrics["rows"]
        widths = [r["thermo_hidden_dim"] for r in rows]
        wall = [r["wall_ms_median"] for r in rows]
        phys = [r["physical_time"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(widths, wall, marker="o", label="wall time (ms)")
        ax1.set_xlabel("thermodynamic FFN width")
        ax1.set_ylabel("PyTorch wall time (ms)")
        ax1.set_xscale("log", base=2)
        ax2 = ax1.twinx()
        ax2.plot(widths, phys, color="tab:orange", marker="s", label="physical time")
        ax2.set_ylabel("emulated physical time")
        fig.tight_layout()
        fig.savefig(out / "transformer_width_constant_physical_time.png", dpi=160)
        plt.close(fig)

    pdit_attention = next((r for r in results if r.name == "pdit_attention_sampling_convergence"), None)
    if pdit_attention is not None:
        rows = pdit_attention.metrics["rows"]
        samples = [r["attention_samples"] for r in rows]
        mse = [r["mse_to_softmax_mean"] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(samples, mse, marker="o")
        ax.set_xlabel("PDIT attention samples")
        ax.set_ylabel("MSE to softmax attention")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out / "pdit_attention_sampling_convergence.png", dpi=160)
        plt.close(fig)

    readout = next((r for r in results if r.name == "thermodynamic_readout_alignment"), None)
    if readout is not None:
        rows = readout.metrics["rows"]
        widths = [r["thermo_hidden_dim"] for r in rows]
        train = [r["post_fit_train_mse"] for r in rows]
        eval_mse = [r["eval_mse_single_shot"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(widths, train, marker="o", label="train")
        ax.plot(widths, eval_mse, marker="s", label="eval single-shot")
        ax.set_xlabel("thermodynamic reservoir width")
        ax.set_ylabel("MSE after one ridge solve")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "thermodynamic_readout_alignment.png", dpi=160)
        plt.close(fig)

    pt_mask = next((r for r in results if r.name == "parallel_tempered_mask_alignment"), None)
    if pt_mask is not None:
        rows = pt_mask.metrics["rows"]
        widths = [r["thermo_hidden_dim"] for r in rows]
        dense = [r["dense_eval_mse_single_shot"] for r in rows]
        sparse = [r["sparse_eval_mse_single_shot"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(widths, dense, marker="o", label="dense ridge eval")
        ax1.plot(widths, sparse, marker="s", label="PT sparse eval")
        ax1.set_xlabel("thermodynamic reservoir width")
        ax1.set_ylabel("eval MSE")
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax2 = ax1.twinx()
        ax2.plot(widths, [r["selected_fraction"] for r in rows], color="tab:green", marker="^", label="selected fraction")
        ax2.set_ylabel("selected feature fraction")
        ax1.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out / "parallel_tempered_mask_alignment.png", dpi=160)
        plt.close(fig)

    e2e_pt = next((r for r in results if r.name == "end_to_end_parallel_tempered_training"), None)
    if e2e_pt is not None:
        labels = ["cold", "parallel tempered"]
        train = [
            e2e_pt.metrics["cold_single_replica"]["final_train_loss"],
            e2e_pt.metrics["parallel_tempered"]["final_train_loss"],
        ]
        eval_loss = [
            e2e_pt.metrics["cold_single_replica"]["final_eval_loss"],
            e2e_pt.metrics["parallel_tempered"]["final_eval_loss"],
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        x = torch.arange(2)
        ax.bar((x - 0.18).numpy(), train, width=0.35, label="train")
        ax.bar((x + 0.18).numpy(), eval_loss, width=0.35, label="eval")
        ax.set_xticks(x.numpy(), labels)
        ax.set_ylabel("MSE after end-to-end training")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "end_to_end_parallel_tempered_training.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="artifacts")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    cfg = DeviceConfig.auto(prefer_cuda=not args.cpu)
    results = run_all_experiments(args.outdir, cfg)
    print(json.dumps([{"name": r.name, "metrics": r.metrics} for r in results], indent=2))


if __name__ == "__main__":
    main()
