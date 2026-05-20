from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DeviceConfig:
    """Runtime device choice for experiments and tests."""

    device: torch.device
    dtype: torch.dtype = torch.float32

    @classmethod
    def auto(cls, prefer_cuda: bool = True, dtype: torch.dtype = torch.float32) -> "DeviceConfig":
        if prefer_cuda and torch.cuda.is_available():
            return cls(torch.device("cuda"), dtype=dtype)
        return cls(torch.device("cpu"), dtype=dtype)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class ThermodynamicNeuronConfig:
    """Shared parameters for quartic thermodynamic neuron layers."""

    j2: float = 1.0
    j3: float = 0.0
    j4: float = 1.5
    temperature: float = 1.0
    t_f: float = 0.2
    dt: float = 0.04
    n_replicas: int = 1
    tempering: bool = False
    swap_interval: int = 4
    output: Literal["cold", "mean"] = "cold"
    state_clip: float = 8.0
    force_clip: float = 80.0

    def build_layer(self, in_features: int, out_features: int):
        from .neurons import ThermodynamicNeuronLayer

        return ThermodynamicNeuronLayer(
            in_features,
            out_features,
            j2=self.j2,
            j3=self.j3,
            j4=self.j4,
            temperature=self.temperature,
            t_f=self.t_f,
            dt=self.dt,
            n_replicas=self.n_replicas,
            tempering=self.tempering,
            swap_interval=self.swap_interval,
            output=self.output,
            state_clip=self.state_clip,
            force_clip=self.force_clip,
        )


@dataclass(frozen=True)
class ThermodynamicTransformerConfig:
    """Convenience configuration for thermodynamic transformer-style blocks."""

    embed_dim: int
    num_heads: int
    thermo_hidden_dim: int
    attention_mode: Literal["softmax", "pdit"] = "softmax"
    n_attention_samples: int = 1
    attention_beta: float = 1.0
    attention_t_f: float = 0.0
    neuron: ThermodynamicNeuronConfig = ThermodynamicNeuronConfig()
    residual_scale: float = 1.0
    memory_efficient_chunk_size: int | None = None

    def build_layer(self):
        from .transformer import ThermodynamicTransformerLayer

        return ThermodynamicTransformerLayer(
            self.embed_dim,
            self.num_heads,
            thermo_hidden_dim=self.thermo_hidden_dim,
            attention_mode=self.attention_mode,
            n_attention_samples=self.n_attention_samples,
            attention_beta=self.attention_beta,
            attention_t_f=self.attention_t_f,
            j2=self.neuron.j2,
            j3=self.neuron.j3,
            j4=self.neuron.j4,
            temperature=self.neuron.temperature,
            t_f=self.neuron.t_f,
            dt=self.neuron.dt,
            n_replicas=self.neuron.n_replicas,
            tempering=self.neuron.tempering,
            swap_interval=self.neuron.swap_interval,
            thermo_output=self.neuron.output,
            state_clip=self.neuron.state_clip,
            force_clip=self.neuron.force_clip,
            residual_scale=self.residual_scale,
            memory_efficient_chunk_size=self.memory_efficient_chunk_size,
        )

    def build_ffn(self):
        from .integration import ThermodynamicFFN

        return ThermodynamicFFN(
            self.embed_dim,
            self.thermo_hidden_dim,
            neuron_config=self.neuron,
            residual_scale=self.residual_scale,
            memory_efficient_chunk_size=self.memory_efficient_chunk_size,
        )


@dataclass(frozen=True)
class PhysicalTimeReport:
    """Compact physical-time and size report for thermodynamic modules."""

    physical_time: float
    n_steps: int
    n_replicas: int
    parameter_count: int
    module_type: str

    @classmethod
    def from_module(cls, module: nn.Module) -> "PhysicalTimeReport":
        physical_time = float(getattr(module, "physical_time", 0.0))
        n_steps = int(getattr(module, "n_steps", 0))
        n_replicas = int(getattr(module, "n_replicas", 1))
        if hasattr(module, "thermo_ff"):
            thermo_ff = getattr(module, "thermo_ff")
            n_steps = int(getattr(thermo_ff, "n_steps", n_steps))
            n_replicas = int(getattr(thermo_ff, "n_replicas", n_replicas))
        if hasattr(module, "thermo"):
            thermo = getattr(module, "thermo")
            n_steps = int(getattr(thermo, "n_steps", n_steps))
            n_replicas = int(getattr(thermo, "n_replicas", n_replicas))
        if hasattr(module, "ffn") and hasattr(getattr(module, "ffn"), "thermo"):
            thermo = getattr(module, "ffn").thermo
            n_steps = int(getattr(thermo, "n_steps", n_steps))
            n_replicas = int(getattr(thermo, "n_replicas", n_replicas))
        params = sum(p.numel() for p in module.parameters())
        return cls(
            physical_time=physical_time,
            n_steps=n_steps,
            n_replicas=n_replicas,
            parameter_count=int(params),
            module_type=type(module).__name__,
        )
