from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ThermodynamicNeuronConfig
from .integration import ThermodynamicFFN


@dataclass(frozen=True)
class FlowMatchFitResult:
    """Training summary for conditional flow matching."""

    initial_loss: float
    final_loss: float
    n_steps: int
    fit_wall_ms: float


@dataclass(frozen=True)
class FlowReadoutFitResult:
    """Closed-form readout fitting summary for thermodynamic flow matching."""

    method: str
    ridge: float
    initial_loss: float
    final_loss: float
    n_pairs: int
    fit_wall_ms: float
    physical_time: float
    feature_dim: int
    target_dim: int


@dataclass(frozen=True)
class FlowSampleResult:
    """Sampling summary for a flow-matching model."""

    samples: Tensor
    n_flow_steps: int
    function_evaluations: int
    wall_ms: float
    modeled_physical_time: float


class FlowVelocityMLP(nn.Module):
    """Small time-conditioned MLP velocity field for flow matching."""

    def __init__(
        self,
        data_dim: int,
        *,
        hidden_dim: int = 64,
        time_features: int = 6,
    ) -> None:
        super().__init__()
        if data_dim <= 0 or hidden_dim <= 0:
            raise ValueError("data_dim and hidden_dim must be positive")
        self.data_dim = int(data_dim)
        self.time_features = int(time_features)
        in_dim = self.data_dim + self.time_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.data_dim),
        )

    @property
    def physical_time(self) -> float:
        return 0.0

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return self.net(torch.cat([x, time_embedding(t, self.time_features)], dim=-1))


class ThermodynamicFlowVelocity(nn.Module):
    """Time-conditioned velocity field using a thermodynamic FFN core."""

    def __init__(
        self,
        data_dim: int,
        *,
        embed_dim: int = 16,
        thermo_hidden_dim: int = 64,
        time_features: int = 6,
        neuron_config: ThermodynamicNeuronConfig | None = None,
        memory_efficient_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if data_dim <= 0 or embed_dim <= 0 or thermo_hidden_dim <= 0:
            raise ValueError("data_dim, embed_dim, and thermo_hidden_dim must be positive")
        self.data_dim = int(data_dim)
        self.embed_dim = int(embed_dim)
        self.time_features = int(time_features)
        self.input_proj = nn.Linear(self.data_dim + self.time_features, self.embed_dim)
        self.thermo = ThermodynamicFFN(
            self.embed_dim,
            thermo_hidden_dim,
            neuron_config=neuron_config or ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0),
            memory_efficient_chunk_size=memory_efficient_chunk_size,
        )
        self.output_proj = nn.Linear(self.embed_dim, self.data_dim)

    @property
    def physical_time(self) -> float:
        return self.thermo.physical_time

    def forward(self, x: Tensor, t: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        return self.output_proj(self.readout_features(x, t, generator=generator))

    def readout_features(self, x: Tensor, t: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
        """Return frozen thermodynamic features used by the velocity readout."""

        features = torch.cat([x, time_embedding(t, self.time_features)], dim=-1)
        hidden = self.input_proj(features).unsqueeze(1)
        flowed = self.thermo(hidden, generator=generator).squeeze(1)
        return torch.tanh(flowed)


def time_embedding(t: Tensor, n_features: int) -> Tensor:
    """Return compact sinusoidal time features with shape `[batch, n_features]`."""

    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        t = t.unsqueeze(-1)
    elif t.ndim != 2 or t.shape[-1] != 1:
        raise ValueError("t must have shape [batch] or [batch, 1]")
    parts = [t]
    frequency = 1.0
    while sum(part.shape[-1] for part in parts) < n_features:
        parts.append(torch.sin(math.pi * frequency * t))
        if sum(part.shape[-1] for part in parts) >= n_features:
            break
        parts.append(torch.cos(math.pi * frequency * t))
        frequency *= 2.0
    return torch.cat(parts, dim=-1)[..., :n_features]


def flow_matching_loss(
    model: nn.Module,
    x1: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Conditional flow-matching objective for a straight Gaussian-to-data path."""

    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype, generator=generator)
    return flow_matching_pair_loss(model, x0, x1, t, generator=generator)


def flow_matching_pair_loss(
    model: nn.Module,
    x0: Tensor,
    x1: Tensor,
    t: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Flow-matching loss for a fixed base/data/time triplet."""

    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must have matching shapes")
    if t.ndim == 1:
        t = t.unsqueeze(-1)
    if t.shape != (x1.shape[0], 1):
        raise ValueError("t must have shape [batch] or [batch, 1]")
    xt = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    predicted = _call_velocity(model, xt, t, generator=generator)
    return F.mse_loss(predicted, target_velocity)


def fit_flow_matching(
    model: nn.Module,
    data: Tensor,
    *,
    n_steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    generator: torch.Generator | None = None,
) -> FlowMatchFitResult:
    """Train a small flow-matching velocity model."""

    if n_steps <= 0 or batch_size <= 0:
        raise ValueError("n_steps and batch_size must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    eval_x1 = data[: min(batch_size, data.shape[0])]
    eval_x0 = torch.randn(eval_x1.shape, device=eval_x1.device, dtype=eval_x1.dtype, generator=generator)
    eval_t = torch.linspace(0.05, 0.95, eval_x1.shape[0], device=eval_x1.device, dtype=eval_x1.dtype).unsqueeze(-1)
    with torch.no_grad():
        initial_loss = float(flow_matching_pair_loss(model, eval_x0, eval_x1, eval_t, generator=generator).cpu())
    start = time.perf_counter()
    for _ in range(n_steps):
        batch = _sample_batch(data, batch_size, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = flow_matching_loss(model, batch, generator=generator)
        loss.backward()
        optimizer.step()
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    model.eval()
    with torch.no_grad():
        final_loss = float(flow_matching_pair_loss(model, eval_x0, eval_x1, eval_t, generator=generator).cpu())
    return FlowMatchFitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        n_steps=int(n_steps),
        fit_wall_ms=float(fit_wall_ms),
    )


def fit_flow_matching_end_to_end(
    model: nn.Module,
    data: Tensor,
    *,
    n_steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    generator: torch.Generator | None = None,
) -> FlowMatchFitResult:
    """Alias for no-ridge inductive flow-matching training."""

    return fit_flow_matching(
        model,
        data,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        generator=generator,
    )


@torch.no_grad()
def fit_flow_matching_readout_ridge(
    model: ThermodynamicFlowVelocity,
    data: Tensor,
    *,
    ridge: float = 1e-3,
    n_pairs: int = 512,
    eval_pairs: int | None = None,
    generator: torch.Generator | None = None,
) -> FlowReadoutFitResult:
    """Fit only the thermodynamic flow velocity readout with one ridge solve.

    The input projection and thermodynamic core are treated as a fixed
    stochastic feature generator. The final velocity readout is solved in
    closed form against flow-matching targets `x1 - x0`.
    """

    if not isinstance(model, ThermodynamicFlowVelocity):
        raise TypeError("fit_flow_matching_readout_ridge requires ThermodynamicFlowVelocity")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    eval_pairs = min(n_pairs, data.shape[0]) if eval_pairs is None else int(eval_pairs)
    if eval_pairs <= 0:
        raise ValueError("eval_pairs must be positive")

    was_training = model.training
    model.eval()
    eval_x1 = data[: min(eval_pairs, data.shape[0])]
    eval_x0 = torch.randn(eval_x1.shape, device=eval_x1.device, dtype=eval_x1.dtype, generator=generator)
    eval_t = torch.linspace(0.05, 0.95, eval_x1.shape[0], device=eval_x1.device, dtype=eval_x1.dtype).unsqueeze(-1)
    initial_loss = float(flow_matching_pair_loss(model, eval_x0, eval_x1, eval_t, generator=generator).detach().cpu())

    start = time.perf_counter()
    x0, x1, t = _make_flow_pairs(data, n_pairs, generator=generator)
    xt = (1.0 - t) * x0 + t * x1
    target_velocity = x1 - x0
    features = model.readout_features(xt, t, generator=generator)
    solution = _ridge_solve(_with_bias(features), target_velocity, ridge)
    model.output_proj.weight.copy_(solution[:-1].T.contiguous())
    model.output_proj.bias.copy_(solution[-1].contiguous())

    final_loss = float(flow_matching_pair_loss(model, eval_x0, eval_x1, eval_t, generator=generator).detach().cpu())
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    if was_training:
        model.train()
    return FlowReadoutFitResult(
        method="flow_readout_ridge",
        ridge=float(ridge),
        initial_loss=initial_loss,
        final_loss=final_loss,
        n_pairs=int(n_pairs),
        fit_wall_ms=float(fit_wall_ms),
        physical_time=float(model.physical_time),
        feature_dim=int(features.shape[-1]),
        target_dim=int(model.data_dim),
    )


@torch.no_grad()
def sample_flow(
    model: nn.Module,
    n_samples: int,
    *,
    data_dim: int = 2,
    n_flow_steps: int = 8,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> FlowSampleResult:
    """Sample by Euler-integrating the learned probability flow ODE."""

    if n_samples <= 0 or data_dim <= 0 or n_flow_steps <= 0:
        raise ValueError("n_samples, data_dim, and n_flow_steps must be positive")
    inferred_device = device or next(model.parameters()).device
    x = torch.randn(n_samples, data_dim, device=inferred_device, dtype=dtype, generator=generator)
    dt = 1.0 / float(n_flow_steps)
    start = time.perf_counter()
    for step in range(n_flow_steps):
        t_value = torch.full((n_samples, 1), (step + 0.5) * dt, device=x.device, dtype=x.dtype)
        velocity = _call_velocity(model, x, t_value, generator=generator)
        x = x + dt * velocity
    wall_ms = (time.perf_counter() - start) * 1000.0
    modeled_physical_time = float(getattr(model, "physical_time", 0.0)) * n_flow_steps
    return FlowSampleResult(
        samples=x,
        n_flow_steps=int(n_flow_steps),
        function_evaluations=int(n_flow_steps),
        wall_ms=float(wall_ms),
        modeled_physical_time=modeled_physical_time,
    )


def make_mog2d(
    n_samples: int,
    *,
    n_modes: int = 8,
    radius: float = 2.0,
    noise: float = 0.08,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample a lightweight 2D mixture of Gaussians arranged on a circle."""

    if n_samples <= 0 or n_modes <= 0:
        raise ValueError("n_samples and n_modes must be positive")
    device = device or torch.device("cpu")
    modes = torch.randint(n_modes, (n_samples,), device=device, generator=generator)
    angles = 2.0 * math.pi * modes.to(torch.float32) / float(n_modes)
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    return centers + noise * torch.randn(n_samples, 2, device=device, generator=generator)


def rbf_mmd2(x: Tensor, y: Tensor, *, bandwidth: float = 1.0) -> Tensor:
    """Biased RBF MMD^2 estimate for tiny distribution comparisons."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have shape [batch, dim] with matching dim")
    gamma = 1.0 / (2.0 * bandwidth * bandwidth)
    xx = torch.exp(-gamma * torch.cdist(x, x).square()).mean()
    yy = torch.exp(-gamma * torch.cdist(y, y).square()).mean()
    xy = torch.exp(-gamma * torch.cdist(x, y).square()).mean()
    return xx + yy - 2.0 * xy


def flow_speedup_vs_diffusion(n_diffusion_steps: int, n_flow_steps: int) -> float:
    """Simple neural-evaluation speedup proxy versus iterative diffusion."""

    if n_diffusion_steps <= 0 or n_flow_steps <= 0:
        raise ValueError("step counts must be positive")
    return float(n_diffusion_steps) / float(n_flow_steps)


def _sample_batch(data: Tensor, batch_size: int, *, generator: torch.Generator | None) -> Tensor:
    indices = torch.randint(data.shape[0], (batch_size,), device=data.device, generator=generator)
    return data[indices]


def _make_flow_pairs(
    data: Tensor,
    n_pairs: int,
    *,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor, Tensor]:
    x1 = _sample_batch(data, n_pairs, generator=generator)
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], 1, device=x1.device, dtype=x1.dtype, generator=generator)
    return x0, x1, t


def _with_bias(features: Tensor) -> Tensor:
    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1)


def _ridge_solve(design: Tensor, targets: Tensor, ridge: float) -> Tensor:
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = design.T @ targets
    return torch.linalg.solve(gram + reg, rhs)


def _call_velocity(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    try:
        return model(x, t, generator=generator)
    except TypeError:
        return model(x, t)
