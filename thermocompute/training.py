from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Optional

import torch
from torch import Tensor

from .transformer import ThermodynamicTransformerLayer


@dataclass(frozen=True)
class ReadoutFitResult:
    """Metrics from one-shot thermodynamic readout fitting."""

    method: str
    ridge: float
    feature_repeats: int
    fit_wall_ms: float
    train_mse: float
    physical_time: float
    n_examples: int
    feature_dim: int
    target_dim: int


@dataclass(frozen=True)
class ParallelTemperedMaskFitResult:
    """Metrics from parallel-tempered sparse reservoir readout fitting."""

    method: str
    ridge: float
    feature_repeats: int
    fit_wall_ms: float
    train_mse: float
    best_energy: float
    physical_time: float
    n_examples: int
    feature_dim: int
    target_dim: int
    selected_features: int
    selected_fraction: float
    n_tempering_replicas: int
    n_tempering_steps: int
    swap_attempts: int
    swap_acceptance: float


@dataclass(frozen=True)
class ParallelTemperedEndToEndFitResult:
    """Metrics from end-to-end parallel-tempered inductive training."""

    method: str
    fit_wall_ms: float
    initial_train_loss: float
    final_train_loss: float
    final_eval_loss: float | None
    physical_time_per_forward: float
    n_tempering_replicas: int
    n_tempering_steps: int
    learning_rate: float
    noise_scale: float
    swap_attempts: int
    swap_acceptance: float
    best_replica_temperature: float


@dataclass(frozen=True)
class ColdEndToEndFitResult:
    """Metrics from ordinary single-replica end-to-end inductive training."""

    method: str
    fit_wall_ms: float
    initial_train_loss: float
    final_train_loss: float
    final_eval_loss: float | None
    physical_time_per_forward: float
    n_steps: int
    learning_rate: float
    memory_replicas: int


@torch.no_grad()
def fit_transformer_readout_ridge(
    layer: ThermodynamicTransformerLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    ridge: float = 1e-3,
    feature_repeats: int = 1,
    attn_mask: Optional[Tensor] = None,
    causal: bool = False,
    generator: Optional[torch.Generator] = None,
) -> ReadoutFitResult:
    """Fit a thermodynamic transformer readout with one ridge solve.

    This is Thermodynamic Readout Alignment (TRA): keep the stochastic
    fixed-time thermodynamic core as a parallel reservoir, collect its features,
    and solve only the final projection in closed form. It is a fast analogue
    to backpropagation for cases where the thermodynamic substrate is used as a
    rich feature generator.
    """

    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must have shape [batch, seq_len, dim]")
    if inputs.shape[:2] != targets.shape[:2]:
        raise ValueError("inputs and targets must share batch and sequence dimensions")
    if targets.shape[-1] != layer.embed_dim:
        raise ValueError("targets last dimension must match layer embed_dim")
    if feature_repeats <= 0:
        raise ValueError("feature_repeats must be positive")

    start = time.perf_counter()
    feature_accum = None
    base = None
    info = None
    for _ in range(feature_repeats):
        features, base, info = layer.thermodynamic_features(
            inputs,
            attn_mask=attn_mask,
            causal=causal,
            generator=generator,
            return_base=True,
        )
        feature_accum = features if feature_accum is None else feature_accum + features
    assert feature_accum is not None and base is not None and info is not None
    features = feature_accum / float(feature_repeats)

    x = features.reshape(-1, layer.thermo_hidden_dim)
    y = ((targets - base) / layer.residual_scale).reshape(-1, layer.embed_dim)
    ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
    design = torch.cat([x, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = design.T @ y
    solution = torch.linalg.solve(gram + reg, rhs)

    weight = solution[:-1].T.contiguous()
    bias = solution[-1].contiguous()
    layer.out_proj.weight.copy_(weight)
    layer.out_proj.bias.copy_(bias)

    pred = base + layer.residual_scale * layer.out_proj(features)
    mse = torch.mean((pred - targets).square())
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    return ReadoutFitResult(
        method="thermodynamic_readout_alignment",
        ridge=float(ridge),
        feature_repeats=int(feature_repeats),
        fit_wall_ms=float(fit_wall_ms),
        train_mse=float(mse.detach().cpu()),
        physical_time=layer.physical_time * feature_repeats,
        n_examples=int(x.shape[0]),
        feature_dim=int(layer.thermo_hidden_dim),
        target_dim=int(layer.embed_dim),
    )


def fit_transformer_end_to_end_cold(
    layer: ThermodynamicTransformerLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    eval_inputs: Optional[Tensor] = None,
    eval_targets: Optional[Tensor] = None,
    n_steps: int = 40,
    learning_rate: float = 2e-3,
    grad_clip: float = 1.0,
    causal: bool = False,
) -> ColdEndToEndFitResult:
    """Standard single-replica end-to-end training for a thermodynamic transformer.

    This is the simplest inductive baseline: train all differentiable
    parameters directly with gradient descent, with no ridge solve, no replica
    ladder, and no tempering swaps.
    """

    _validate_transformer_readout_inputs(layer, inputs, targets)
    if eval_inputs is not None or eval_targets is not None:
        if eval_inputs is None or eval_targets is None:
            raise ValueError("eval_inputs and eval_targets must be provided together")
        _validate_transformer_readout_inputs(layer, eval_inputs, eval_targets)
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    start = time.perf_counter()
    was_training = layer.training
    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_train_loss = float(_mse_model_loss(layer, inputs, targets, causal).detach().cpu())

    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = _mse_model_loss(layer, inputs, targets, causal)
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(layer.parameters(), grad_clip)
        optimizer.step()
        with torch.no_grad():
            _stabilize_thermodynamic_params(layer)

    with torch.no_grad():
        final_train_loss = float(_mse_model_loss(layer, inputs, targets, causal).detach().cpu())
        final_eval_loss = None
        if eval_inputs is not None and eval_targets is not None:
            layer.eval()
            final_eval_loss = float(_mse_model_loss(layer, eval_inputs, eval_targets, causal).detach().cpu())
        if was_training:
            layer.train()
        else:
            layer.eval()

    return ColdEndToEndFitResult(
        method="cold_end_to_end_training",
        fit_wall_ms=float((time.perf_counter() - start) * 1000.0),
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        physical_time_per_forward=layer.physical_time,
        n_steps=int(n_steps),
        learning_rate=float(learning_rate),
        memory_replicas=1,
    )


def _mse_model_loss(model: ThermodynamicTransformerLayer, inputs: Tensor, targets: Tensor, causal: bool) -> Tensor:
    pred = model(inputs, causal=causal)
    return torch.mean((pred - targets).square())


@torch.no_grad()
def _stabilize_thermodynamic_params(model: ThermodynamicTransformerLayer) -> None:
    model.thermo_ff.j2.clamp_(-5.0, 5.0)
    model.thermo_ff.j3.clamp_(-5.0, 5.0)
    model.thermo_ff.j4.clamp_(0.05, 8.0)


def fit_transformer_end_to_end_parallel_tempering(
    layer: ThermodynamicTransformerLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    eval_inputs: Optional[Tensor] = None,
    eval_targets: Optional[Tensor] = None,
    n_tempering_replicas: int = 4,
    n_tempering_steps: int = 40,
    learning_rate: float = 2e-3,
    noise_scale: float = 1e-3,
    min_temperature: float = 0.05,
    max_temperature: float = 1.0,
    swap_interval: int = 4,
    grad_clip: float = 1.0,
    causal: bool = False,
) -> ParallelTemperedEndToEndFitResult:
    """End-to-end inductive transformer training with parallel tempering.

    This method does not solve a ridge system. It keeps several full copies of
    the thermodynamic transformer layer, trains each copy by gradient descent,
    injects temperature-scaled Langevin parameter noise, and periodically swaps
    whole parameter states between adjacent temperatures by a Metropolis rule.

    The best final replica is loaded back into `layer`.
    """

    _validate_transformer_readout_inputs(layer, inputs, targets)
    if eval_inputs is not None or eval_targets is not None:
        if eval_inputs is None or eval_targets is None:
            raise ValueError("eval_inputs and eval_targets must be provided together")
        _validate_transformer_readout_inputs(layer, eval_inputs, eval_targets)
    if n_tempering_replicas <= 0:
        raise ValueError("n_tempering_replicas must be positive")
    if n_tempering_steps <= 0:
        raise ValueError("n_tempering_steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if min_temperature <= 0.0 or max_temperature <= 0.0 or max_temperature < min_temperature:
        raise ValueError("temperature ladder must be positive and ordered")

    start = time.perf_counter()
    layer_was_training = layer.training
    layer.train()
    with torch.no_grad():
        initial_train_loss = float(_mse_model_loss(layer, inputs, targets, causal).detach().cpu())

    replicas = [copy.deepcopy(layer).to(device=inputs.device, dtype=inputs.dtype) for _ in range(n_tempering_replicas)]
    for replica in replicas:
        replica.train()
    optimizers = [torch.optim.Adam(replica.parameters(), lr=learning_rate) for replica in replicas]
    temperatures = torch.logspace(
        math.log10(min_temperature),
        math.log10(max_temperature),
        n_tempering_replicas,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    swap_attempts = 0
    swap_accepts = 0
    latest_losses = torch.full((n_tempering_replicas,), initial_train_loss, device=inputs.device, dtype=inputs.dtype)

    for step in range(n_tempering_steps):
        for r, replica in enumerate(replicas):
            optimizers[r].zero_grad(set_to_none=True)
            loss = _mse_model_loss(replica, inputs, targets, causal)
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(replica.parameters(), grad_clip)
            optimizers[r].step()
            with torch.no_grad():
                _stabilize_thermodynamic_params(replica)
                if noise_scale > 0.0:
                    sigma = math.sqrt(2.0 * learning_rate * float(temperatures[r].detach().cpu())) * noise_scale
                    for param in replica.parameters():
                        if param.requires_grad and param.dtype.is_floating_point:
                            param.add_(sigma * torch.randn_like(param))
                latest_losses[r] = _mse_model_loss(replica, inputs, targets, causal).detach()

        if n_tempering_replicas > 1 and (step + 1) % max(1, swap_interval) == 0:
            for start_pair in (0, 1):
                for r in range(start_pair, n_tempering_replicas - 1, 2):
                    delta = (1.0 / temperatures[r] - 1.0 / temperatures[r + 1]) * (latest_losses[r] - latest_losses[r + 1])
                    swap = bool(torch.log(torch.rand((), device=inputs.device, dtype=inputs.dtype).clamp_min(1e-12)) < delta.clamp_max(80.0))
                    swap_attempts += 1
                    if swap:
                        swap_accepts += 1
                        replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                        optimizers[r], optimizers[r + 1] = optimizers[r + 1], optimizers[r]
                        latest_losses[[r, r + 1]] = latest_losses[[r + 1, r]]

    with torch.no_grad():
        final_losses = torch.stack([_mse_model_loss(replica, inputs, targets, causal).detach() for replica in replicas])
        best_idx = int(torch.argmin(final_losses).detach().cpu())
        layer.load_state_dict(replicas[best_idx].state_dict())
        final_train_loss = float(final_losses[best_idx].detach().cpu())
        final_eval_loss = None
        if eval_inputs is not None and eval_targets is not None:
            layer.eval()
            final_eval_loss = float(_mse_model_loss(layer, eval_inputs, eval_targets, causal).detach().cpu())
        if layer_was_training:
            layer.train()
        else:
            layer.eval()

    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    return ParallelTemperedEndToEndFitResult(
        method="parallel_tempered_end_to_end_training",
        fit_wall_ms=float(fit_wall_ms),
        initial_train_loss=initial_train_loss,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        physical_time_per_forward=layer.physical_time,
        n_tempering_replicas=int(n_tempering_replicas),
        n_tempering_steps=int(n_tempering_steps),
        learning_rate=float(learning_rate),
        noise_scale=float(noise_scale),
        swap_attempts=int(swap_attempts),
        swap_acceptance=float(swap_accepts / swap_attempts) if swap_attempts else 0.0,
        best_replica_temperature=float(temperatures[best_idx].detach().cpu()),
    )


def _validate_transformer_readout_inputs(layer: ThermodynamicTransformerLayer, inputs: Tensor, targets: Tensor) -> None:
    if inputs.ndim != 3 or targets.ndim != 3:
        raise ValueError("inputs and targets must have shape [batch, seq_len, dim]")
    if inputs.shape[:2] != targets.shape[:2]:
        raise ValueError("inputs and targets must share batch and sequence dimensions")
    if targets.shape[-1] != layer.embed_dim:
        raise ValueError("targets last dimension must match layer embed_dim")


def _collect_transformer_features(
    layer: ThermodynamicTransformerLayer,
    inputs: Tensor,
    *,
    feature_repeats: int,
    attn_mask: Optional[Tensor],
    causal: bool,
    generator: Optional[torch.Generator],
) -> tuple[Tensor, Tensor]:
    if feature_repeats <= 0:
        raise ValueError("feature_repeats must be positive")
    feature_accum = None
    base = None
    for _ in range(feature_repeats):
        features, base, _ = layer.thermodynamic_features(
            inputs,
            attn_mask=attn_mask,
            causal=causal,
            generator=generator,
            return_base=True,
        )
        feature_accum = features if feature_accum is None else feature_accum + features
    assert feature_accum is not None and base is not None
    return feature_accum / float(feature_repeats), base


def _ridge_solution_for_mask(x: Tensor, y: Tensor, mask: Tensor, ridge: float) -> tuple[Tensor, Tensor, Tensor]:
    mask = mask.to(device=x.device, dtype=torch.bool)
    if not bool(mask.any()):
        mask = mask.clone()
        mask[0] = True
    x_sel = x[:, mask]
    ones = torch.ones(x_sel.shape[0], 1, device=x.device, dtype=x.dtype)
    design = torch.cat([x_sel, ones], dim=1)
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = design.T @ y
    solution = torch.linalg.solve(gram + reg, rhs)
    pred = design @ solution
    mse = torch.mean((pred - y).square())
    return solution, pred, mse


@torch.no_grad()
def fit_transformer_readout_parallel_tempering(
    layer: ThermodynamicTransformerLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    ridge: float = 1e-2,
    feature_repeats: int = 1,
    keep_fraction: float = 0.5,
    sparsity_penalty: float = 1e-3,
    n_tempering_replicas: int = 6,
    n_tempering_steps: int = 32,
    flips_per_step: int = 2,
    min_temperature: float = 0.02,
    max_temperature: float = 1.0,
    attn_mask: Optional[Tensor] = None,
    causal: bool = False,
    generator: Optional[torch.Generator] = None,
) -> ParallelTemperedMaskFitResult:
    """Train a sparse thermodynamic readout with parallel-tempered mask search.

    Parallel Tempered Mask Alignment (PTMA) searches the binary feature-mask
    space of a thermodynamic transformer reservoir. Each replica carries a mask
    over reservoir features. Local moves flip feature selectors, Metropolis
    acceptance uses ridge-readout loss plus a sparsity energy, and adjacent
    temperatures swap masks to cross bad sparse subsets.

    The best mask is finally programmed into `layer.out_proj` as a sparse
    readout. This is a second training analogue: the nonconvex discrete
    selection problem is handled by parallel tempering, while inference still
    uses one fixed thermodynamic evolution window.
    """

    _validate_transformer_readout_inputs(layer, inputs, targets)
    if not (0.0 < keep_fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")
    if n_tempering_replicas < 2:
        raise ValueError("n_tempering_replicas must be at least 2")
    if n_tempering_steps <= 0:
        raise ValueError("n_tempering_steps must be positive")
    if flips_per_step <= 0:
        raise ValueError("flips_per_step must be positive")
    if min_temperature <= 0.0 or max_temperature <= 0.0 or max_temperature < min_temperature:
        raise ValueError("temperature ladder must be positive and ordered")

    start = time.perf_counter()
    features, base = _collect_transformer_features(
        layer,
        inputs,
        feature_repeats=feature_repeats,
        attn_mask=attn_mask,
        causal=causal,
        generator=generator,
    )
    x = features.reshape(-1, layer.thermo_hidden_dim)
    y = ((targets - base) / layer.residual_scale).reshape(-1, layer.embed_dim)
    feature_dim = x.shape[1]
    keep_count = max(1, min(feature_dim, int(round(feature_dim * keep_fraction))))

    masks = torch.zeros(n_tempering_replicas, feature_dim, device=x.device, dtype=torch.bool)
    scores = torch.rand(n_tempering_replicas, feature_dim, device=x.device, dtype=x.dtype, generator=generator)
    topk = torch.topk(scores, keep_count, dim=1).indices
    masks.scatter_(1, topk, True)
    temperatures = torch.logspace(
        math.log10(min_temperature),
        math.log10(max_temperature),
        n_tempering_replicas,
        device=x.device,
        dtype=x.dtype,
    )

    solutions: list[Tensor] = []
    mses = torch.empty(n_tempering_replicas, device=x.device, dtype=x.dtype)
    energies = torch.empty_like(mses)
    for r in range(n_tempering_replicas):
        solution, _, mse = _ridge_solution_for_mask(x, y, masks[r], ridge)
        solutions.append(solution)
        mses[r] = mse
        energies[r] = mse + float(sparsity_penalty) * masks[r].float().mean()

    best_idx = int(torch.argmin(energies).detach().cpu())
    best_mask = masks[best_idx].clone()
    best_solution = solutions[best_idx].clone()
    best_mse = mses[best_idx].clone()
    best_energy = energies[best_idx].clone()
    swap_attempts = 0
    swap_accepts = 0

    for _ in range(n_tempering_steps):
        for r in range(n_tempering_replicas):
            proposal = masks[r].clone()
            flip_idx = torch.randint(0, feature_dim, (flips_per_step,), device=x.device, generator=generator)
            proposal[flip_idx] = ~proposal[flip_idx]
            if not bool(proposal.any()):
                proposal[torch.randint(0, feature_dim, (1,), device=x.device, generator=generator)] = True
            solution, _, mse = _ridge_solution_for_mask(x, y, proposal, ridge)
            energy = mse + float(sparsity_penalty) * proposal.float().mean()
            log_accept = -(energy - energies[r]) / temperatures[r]
            accept = bool(torch.log(torch.rand((), device=x.device, dtype=x.dtype, generator=generator).clamp_min(1e-12)) < log_accept.clamp_max(80.0))
            if accept:
                masks[r] = proposal
                solutions[r] = solution
                mses[r] = mse
                energies[r] = energy
                if float(energy.detach().cpu()) < float(best_energy.detach().cpu()):
                    best_mask = proposal.clone()
                    best_solution = solution.clone()
                    best_mse = mse.clone()
                    best_energy = energy.clone()

        for start_pair in (0, 1):
            for r in range(start_pair, n_tempering_replicas - 1, 2):
                delta = (1.0 / temperatures[r] - 1.0 / temperatures[r + 1]) * (energies[r] - energies[r + 1])
                swap = bool(torch.log(torch.rand((), device=x.device, dtype=x.dtype, generator=generator).clamp_min(1e-12)) < delta.clamp_max(80.0))
                swap_attempts += 1
                if swap:
                    swap_accepts += 1
                    masks[[r, r + 1]] = masks[[r + 1, r]]
                    energies[[r, r + 1]] = energies[[r + 1, r]]
                    mses[[r, r + 1]] = mses[[r + 1, r]]
                    solutions[r], solutions[r + 1] = solutions[r + 1], solutions[r]

    selected = best_mask.nonzero(as_tuple=False).squeeze(-1)
    if selected.ndim == 0:
        selected = selected.unsqueeze(0)
    weight = torch.zeros(layer.embed_dim, feature_dim, device=x.device, dtype=x.dtype)
    weight[:, selected] = best_solution[:-1].T.contiguous()
    bias = best_solution[-1].contiguous()
    layer.out_proj.weight.copy_(weight)
    layer.out_proj.bias.copy_(bias)

    pred = base + layer.residual_scale * layer.out_proj(features)
    programmed_mse = torch.mean((pred - targets).square())
    selected_count = int(best_mask.sum().detach().cpu())
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    return ParallelTemperedMaskFitResult(
        method="parallel_tempered_mask_alignment",
        ridge=float(ridge),
        feature_repeats=int(feature_repeats),
        fit_wall_ms=float(fit_wall_ms),
        train_mse=float(programmed_mse.detach().cpu()),
        best_energy=float(best_energy.detach().cpu()),
        physical_time=layer.physical_time * feature_repeats,
        n_examples=int(x.shape[0]),
        feature_dim=int(feature_dim),
        target_dim=int(layer.embed_dim),
        selected_features=selected_count,
        selected_fraction=float(selected_count / feature_dim),
        n_tempering_replicas=int(n_tempering_replicas),
        n_tempering_steps=int(n_tempering_steps),
        swap_attempts=int(swap_attempts),
        swap_acceptance=float(swap_accepts / swap_attempts) if swap_attempts else 0.0,
    )
