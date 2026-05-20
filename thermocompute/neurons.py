from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn


def quartic_potential(x: Tensor, current: Tensor, j2: Tensor, j3: Tensor, j4: Tensor) -> Tensor:
    return 0.5 * j2 * x.square() + (j3 / 3.0) * x.pow(3) + 0.25 * j4 * x.pow(4) - current * x


@dataclass(frozen=True)
class ThermodynamicRunInfo:
    physical_time: float
    n_steps: int
    n_replicas: int
    used_tempering: bool
    swap_attempts: int
    swap_acceptance: float


class ThermodynamicNeuronLayer(nn.Module):
    """Vectorized quartic thermodynamic neuron layer.

    Input currents are computed digitally as `x @ W.T + b`; the stochastic
    nonlinear activation is emulated by fixed-time Langevin dynamics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        j2: float = 1.0,
        j3: float = 0.0,
        j4: float = 1.5,
        temperature: float = 1.0,
        t_f: float = 2.0,
        dt: float = 0.05,
        n_replicas: int = 1,
        tempering: bool = False,
        swap_interval: int = 4,
        output: Literal["cold", "mean"] = "cold",
        state_clip: float = 8.0,
        force_clip: float = 80.0,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if j4 <= 0.0:
            raise ValueError("j4 must be positive to keep the quartic potential confining")
        if t_f <= 0.0 or dt <= 0.0:
            raise ValueError("t_f and dt must be positive")
        if n_replicas <= 0:
            raise ValueError("n_replicas must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.t_f = float(t_f)
        self.dt = float(dt)
        self.n_steps = max(1, int(round(self.t_f / self.dt)))
        self.n_replicas = int(n_replicas)
        self.tempering = bool(tempering)
        self.swap_interval = max(1, int(swap_interval))
        self.output = output
        self.state_clip = float(state_clip)
        self.force_clip = float(force_clip)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

        self.j2 = nn.Parameter(torch.full((out_features,), float(j2)))
        self.j3 = nn.Parameter(torch.full((out_features,), float(j3)))
        self.j4 = nn.Parameter(torch.full((out_features,), float(j4)))
        temps = torch.linspace(1.0, 1.0 + 0.4 * (self.n_replicas - 1), self.n_replicas) * float(temperature)
        self.register_buffer("temperatures", temps)

    @property
    def physical_time(self) -> float:
        return self.n_steps * self.dt

    def currents(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def _attempt_tempering_swaps(self, state: Tensor, current: Tensor) -> tuple[Tensor, int, int]:
        if self.n_replicas < 2:
            return state, 0, 0
        attempts = 0
        accepts = 0
        beta = 1.0 / self.temperatures.to(device=state.device, dtype=state.dtype)
        j2 = self.j2.view(1, 1, -1)
        j3 = self.j3.view(1, 1, -1)
        j4 = self.j4.view(1, 1, -1)
        current_b = current.unsqueeze(0)
        for start in (0, 1):
            for r in range(start, self.n_replicas - 1, 2):
                x_a = state[r]
                x_b = state[r + 1]
                e_a = quartic_potential(x_a, current_b.squeeze(0), j2.squeeze(0), j3.squeeze(0), j4.squeeze(0))
                e_b = quartic_potential(x_b, current_b.squeeze(0), j2.squeeze(0), j3.squeeze(0), j4.squeeze(0))
                log_alpha = (beta[r] - beta[r + 1]) * (e_a - e_b)
                accept = torch.log(torch.rand_like(log_alpha).clamp_min(1e-12)) < log_alpha.clamp_max(80.0)
                old_a = state[r].clone()
                state[r] = torch.where(accept, state[r + 1], state[r])
                state[r + 1] = torch.where(accept, old_a, state[r + 1])
                attempts += accept.numel()
                accepts += int(accept.sum().detach().cpu())
        return state, attempts, accepts

    def forward(
        self,
        x: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        current = self.currents(x)
        state = torch.zeros(
            self.n_replicas,
            current.shape[0],
            current.shape[1],
            device=current.device,
            dtype=current.dtype,
        )
        temperatures = self.temperatures.to(device=current.device, dtype=current.dtype).view(self.n_replicas, 1, 1)
        j2 = self.j2.view(1, 1, -1)
        j3 = self.j3.view(1, 1, -1)
        j4 = self.j4.view(1, 1, -1)
        current_b = current.unsqueeze(0)
        swap_attempts = 0
        swap_accepts = 0

        for step in range(self.n_steps):
            force = -(j2 * state + j3 * state.square() + j4 * state.pow(3) - current_b)
            force = force.clamp(-self.force_clip, self.force_clip)
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            state = state + self.dt * force + torch.sqrt(2.0 * temperatures * self.dt) * noise
            state = torch.nan_to_num(state, nan=0.0, posinf=self.state_clip, neginf=-self.state_clip)
            state = state.clamp(-self.state_clip, self.state_clip)
            if self.tempering and self.n_replicas > 1 and (step + 1) % self.swap_interval == 0:
                state, attempts, accepts = self._attempt_tempering_swaps(state, current)
                swap_attempts += attempts
                swap_accepts += accepts

        if self.output == "mean":
            y = state.mean(dim=0)
        else:
            y = state[0]
        info = ThermodynamicRunInfo(
            physical_time=self.physical_time,
            n_steps=self.n_steps,
            n_replicas=self.n_replicas,
            used_tempering=self.tempering,
            swap_attempts=swap_attempts,
            swap_acceptance=float(swap_accepts / swap_attempts) if swap_attempts else 0.0,
        )
        if return_info:
            return y, info
        return y


class ThermodynamicMLP(nn.Module):
    """Feed-forward thermodynamic MLP."""

    def __init__(
        self,
        layer_sizes: list[int],
        *,
        t_f: float = 2.0,
        dt: float = 0.05,
        n_replicas: int = 1,
        tempering: bool = False,
        hidden_activation: str = "tanh",
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output size")
        self.layers = nn.ModuleList(
            [
                ThermodynamicNeuronLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    t_f=t_f,
                    dt=dt,
                    n_replicas=n_replicas,
                    tempering=tempering,
                    output="cold",
                )
                for i in range(len(layer_sizes) - 1)
            ]
        )
        self.hidden_activation = hidden_activation

    @property
    def physical_time(self) -> float:
        return sum(layer.physical_time for layer in self.layers)

    def _activate(self, x: Tensor) -> Tensor:
        if self.hidden_activation == "identity":
            return x
        if self.hidden_activation == "gelu":
            return torch.nn.functional.gelu(x)
        return torch.tanh(x)

    def forward(
        self,
        x: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        total_attempts = 0
        weighted_accept = 0.0
        for i, layer in enumerate(self.layers):
            result = layer(x, generator=generator, return_info=True)
            x, info = result
            total_attempts += info.swap_attempts
            weighted_accept += info.swap_acceptance * info.swap_attempts
            if i < len(self.layers) - 1:
                x = self._activate(x)
        aggregate = ThermodynamicRunInfo(
            physical_time=self.physical_time,
            n_steps=sum(layer.n_steps for layer in self.layers),
            n_replicas=max(layer.n_replicas for layer in self.layers),
            used_tempering=any(layer.tempering for layer in self.layers),
            swap_attempts=total_attempts,
            swap_acceptance=float(weighted_accept / total_attempts) if total_attempts else 0.0,
        )
        if return_info:
            return x, aggregate
        return x
