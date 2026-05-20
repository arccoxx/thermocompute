from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn


def _as_tensor(x: Tensor | float, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


@dataclass
class IsingEnergy:
    """Vectorized Ising energy for p-bit experiments.

    States are represented as 0/1 tensors, then internally mapped to -1/+1.
    Energy is `-0.5 * s_pm @ J @ s_pm - h @ s_pm`.
    """

    J: Tensor
    h: Tensor

    def __post_init__(self) -> None:
        if self.J.ndim != 2 or self.J.shape[0] != self.J.shape[1]:
            raise ValueError("J must be square")
        if self.h.shape[-1] != self.J.shape[0]:
            raise ValueError("h must match J")
        self.J = 0.5 * (self.J + self.J.T)
        self.J = self.J.clone()
        self.J.fill_diagonal_(0.0)

    @property
    def n(self) -> int:
        return int(self.h.shape[-1])

    def energy(self, states: Tensor) -> Tensor:
        s = states.to(dtype=self.J.dtype, device=self.J.device) * 2.0 - 1.0
        pair = torch.einsum("...i,ij,...j->...", s, self.J, s)
        field = torch.einsum("...i,i->...", s, self.h)
        return -0.5 * pair - field

    def local_field(self, states: Tensor) -> Tensor:
        s = states.to(dtype=self.J.dtype, device=self.J.device) * 2.0 - 1.0
        return torch.matmul(s, self.J.T) + self.h


class BinaryPBit(nn.Module):
    """Parallel p-bit sampler with logistic control voltage."""

    def __init__(self, beta: float = 1.0, tau0: float = 100e-9) -> None:
        super().__init__()
        self.beta = float(beta)
        self.tau0 = float(tau0)

    def probabilities(self, control_voltage: Tensor) -> Tensor:
        return torch.sigmoid(self.beta * control_voltage)

    def sample(self, control_voltage: Tensor, *, generator: Optional[torch.Generator] = None) -> Tensor:
        p = self.probabilities(control_voltage)
        return torch.bernoulli(p, generator=generator)

    def gibbs_step(self, states: Tensor, energy: IsingEnergy, *, generator: Optional[torch.Generator] = None) -> Tensor:
        field = energy.local_field(states)
        p_one = torch.sigmoid(2.0 * self.beta * field)
        return torch.bernoulli(p_one, generator=generator)


class CategoricalPDIT(nn.Module):
    """Categorical thermodynamic sampling unit."""

    def __init__(self, beta: float = 1.0, tau0: float = 100e-9) -> None:
        super().__init__()
        self.beta = float(beta)
        self.tau0 = float(tau0)

    def probabilities(self, logits: Tensor) -> Tensor:
        return torch.softmax(self.beta * logits, dim=-1)

    def sample(self, logits: Tensor, *, generator: Optional[torch.Generator] = None) -> Tensor:
        probs = self.probabilities(logits)
        flat = probs.reshape(-1, probs.shape[-1])
        idx = torch.multinomial(flat, 1, replacement=True, generator=generator)
        return idx.reshape(probs.shape[:-1])


class PMODE(nn.Module):
    """Programmable Gaussian mode using an exact OU transition."""

    def __init__(self, tau0: float = 100e-9) -> None:
        super().__init__()
        self.tau0 = float(tau0)

    def ou_step(self, x: Tensor, mu: Tensor, sigma: Tensor, dt: float, *, generator: Optional[torch.Generator] = None) -> Tensor:
        decay = math.exp(-float(dt) / self.tau0)
        noise_scale = math.sqrt(max(1.0 - decay * decay, 0.0))
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return mu + (x - mu) * decay + sigma * noise_scale * noise

    def sample(
        self,
        mu: Tensor,
        sigma: Tensor | float,
        *,
        n_samples: int = 1024,
        t_total: float = 1e-6,
        dt: float | None = None,
        burnin: float = 0.0,
        x0: Tensor | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        dt = float(dt if dt is not None else self.tau0)
        sigma_t = _as_tensor(sigma, device=mu.device, dtype=mu.dtype)
        sample_shape = (int(n_samples),) + tuple(mu.shape)
        mu_e = mu.expand(sample_shape)
        sigma_e = sigma_t.expand(sample_shape)
        if x0 is None:
            x = torch.zeros(sample_shape, device=mu.device, dtype=mu.dtype)
        else:
            x = x0.to(device=mu.device, dtype=mu.dtype).expand(sample_shape).clone()
        steps = max(1, int(math.ceil((t_total + burnin) / dt)))
        for _ in range(steps):
            x = self.ou_step(x, mu_e, sigma_e, dt, generator=generator)
        return x

    def trajectory(
        self,
        mu: Tensor,
        sigma: Tensor | float,
        *,
        n_steps: int,
        dt: float | None = None,
        x0: Tensor | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Return a single-device OU trajectory for diagnostics."""

        dt = float(dt if dt is not None else self.tau0)
        sigma_t = _as_tensor(sigma, device=mu.device, dtype=mu.dtype)
        x = torch.zeros_like(mu) if x0 is None else x0.to(device=mu.device, dtype=mu.dtype)
        samples: list[Tensor] = []
        for _ in range(max(1, n_steps)):
            x = self.ou_step(x, mu, sigma_t, dt, generator=generator)
            samples.append(x.clone())
        return torch.stack(samples, dim=0)


class PMOG(nn.Module):
    """Mixture-of-Gaussians thermodynamic sampler.

    This emulates PMOG as a categorical mode selector coupled to a PMODE OU
    relaxation. With `switch_rate > 0`, the mode can resample during the fixed
    physical window.
    """

    def __init__(self, n_components: int, tau0: float = 100e-9, beta: float = 1.0) -> None:
        super().__init__()
        if n_components < 2:
            raise ValueError("n_components must be at least 2")
        self.n_components = int(n_components)
        self.tau0 = float(tau0)
        self.beta = float(beta)
        self._pmode = PMODE(tau0=tau0)

    def sample(
        self,
        logits: Tensor,
        means: Tensor,
        scales: Tensor,
        *,
        n_samples: int = 1024,
        t_total: float = 1e-6,
        dt: float | None = None,
        burnin: float = 0.0,
        switch_rate: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[Tensor, Tensor]:
        if logits.shape[-1] != self.n_components:
            raise ValueError("logits last dimension must equal n_components")
        dt = float(dt if dt is not None else self.tau0)
        probs = torch.softmax(self.beta * logits, dim=-1)
        event_shape = probs.shape[:-1]
        sample_event_shape = (int(n_samples),) + tuple(event_shape)
        probs_e = probs.expand(sample_event_shape + (self.n_components,))
        flat_probs = probs_e.reshape(-1, self.n_components)
        mode = torch.multinomial(flat_probs, 1, replacement=True, generator=generator).reshape(sample_event_shape)

        means = means.to(device=logits.device, dtype=logits.dtype)
        scales = scales.to(device=logits.device, dtype=logits.dtype)
        mode_expanded = mode.unsqueeze(-1)
        means_e = means.expand(sample_event_shape + (self.n_components,))
        scales_e = scales.expand(sample_event_shape + (self.n_components,))
        mu = torch.gather(means_e, -1, mode_expanded).squeeze(-1)
        sigma = torch.gather(scales_e, -1, mode_expanded).squeeze(-1)
        x = mu + sigma * torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)

        steps = max(1, int(math.ceil((t_total + burnin) / dt)))
        p_switch = 1.0 - math.exp(-max(float(switch_rate), 0.0) * dt)

        for _ in range(steps):
            if p_switch > 0.0:
                mask = torch.rand(sample_event_shape, device=logits.device, dtype=logits.dtype, generator=generator) < p_switch
                if bool(mask.any()):
                    new_mode = torch.multinomial(flat_probs, 1, replacement=True, generator=generator).reshape(sample_event_shape)
                    mode = torch.where(mask, new_mode, mode)
            mode_expanded = mode.unsqueeze(-1)
            mu = torch.gather(means_e, -1, mode_expanded).squeeze(-1)
            sigma = torch.gather(scales_e, -1, mode_expanded).squeeze(-1).clamp_min(1e-8)
            x = self._pmode.ou_step(x, mu, sigma, dt, generator=generator)
        return x, mode
