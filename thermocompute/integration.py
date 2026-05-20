from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ThermodynamicNeuronConfig, ThermodynamicTransformerConfig
from .neurons import ThermodynamicRunInfo
from .transformer import ThermodynamicSelfAttention, ThermodynamicTransformerInfo, _aggregate_chunk_infos


class ThermodynamicFFN(nn.Module):
    """Drop-in transformer feed-forward block with thermodynamic hidden width.

    The input and output shape is `[batch, seq, embed_dim]`.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        neuron_config: ThermodynamicNeuronConfig | None = None,
        residual_scale: float = 1.0,
        memory_efficient_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        if memory_efficient_chunk_size is not None and memory_efficient_chunk_size <= 0:
            raise ValueError("memory_efficient_chunk_size must be positive when provided")
        self.memory_efficient_chunk_size = (
            int(memory_efficient_chunk_size) if memory_efficient_chunk_size is not None else None
        )
        self.neuron_config = neuron_config or ThermodynamicNeuronConfig()
        self.thermo = self.neuron_config.build_layer(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    @property
    def physical_time(self) -> float:
        return self.thermo.physical_time

    def forward(
        self,
        x: Tensor,
        *,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq, embed_dim]")
        batch, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError("last dimension must match embed_dim")
        flat = x.reshape(batch * seq_len, embed_dim)
        effective_chunk = chunk_size if chunk_size is not None else self.memory_efficient_chunk_size
        if effective_chunk is not None and effective_chunk < self.hidden_dim:
            y, info = self._forward_chunked(flat, batch, seq_len, int(effective_chunk), generator=generator)
        else:
            hidden, info = self.thermo(
                flat,
                generator=generator,
                return_info=True,
            )
            y = self.out_proj(torch.tanh(hidden)).view(batch, seq_len, embed_dim)
        if return_info:
            return y, info
        return y

    def _forward_chunked(
        self,
        flat: Tensor,
        batch: int,
        seq_len: int,
        chunk_size: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, ThermodynamicRunInfo]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        flat_out = flat.new_zeros(flat.shape[0], self.embed_dim)
        infos: list[ThermodynamicRunInfo] = []
        for start in range(0, self.hidden_dim, chunk_size):
            end = min(start + chunk_size, self.hidden_dim)
            hidden, info = self.thermo.forward_chunk(flat, start, end, generator=generator, return_info=True)
            flat_out = flat_out + F.linear(torch.tanh(hidden), self.out_proj.weight[:, start:end], None)
            infos.append(info)
        flat_out = flat_out + self.out_proj.bias
        return flat_out.view(batch, seq_len, self.embed_dim), _aggregate_chunk_infos(infos)


class ThermodynamicTransformerBlock(nn.Module):
    """Production-shaped pre-norm transformer block using a thermodynamic FFN."""

    def __init__(self, config: ThermodynamicTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attention = ThermodynamicSelfAttention(
            config.embed_dim,
            config.num_heads,
            mode=config.attention_mode,
            n_attention_samples=config.n_attention_samples,
            beta=config.attention_beta,
            attention_t_f=config.attention_t_f,
        )
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = config.build_ffn()

    @property
    def physical_time(self) -> float:
        return self.attention.physical_time + self.ffn.physical_time

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Tensor | None = None,
        causal: bool = False,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicTransformerInfo]:
        x = x + self.config.residual_scale * self.attention(
            self.norm1(x),
            attn_mask=attn_mask,
            causal=causal,
            generator=generator,
        )
        ff_result = self.ffn(self.norm2(x), generator=generator, chunk_size=chunk_size, return_info=True)
        ff, ff_info = ff_result
        y = x + self.config.residual_scale * ff
        info = ThermodynamicTransformerInfo(
            physical_time=self.physical_time,
            attention_physical_time=self.attention.physical_time,
            feedforward_physical_time=self.ffn.physical_time,
            n_steps=ff_info.n_steps,
            n_replicas=max(ff_info.n_replicas, self.attention.n_attention_samples),
            attention_mode=self.attention.mode,
            attention_samples=self.attention.n_attention_samples,
            used_tempering=ff_info.used_tempering,
            swap_attempts=ff_info.swap_attempts,
            swap_acceptance=ff_info.swap_acceptance,
        )
        if return_info:
            return y, info
        return y


def _infer_sequential_embed_dim(module: nn.Sequential) -> int | None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if len(linears) < 2:
        return None
    first, last = linears[0], linears[-1]
    if first.in_features != last.out_features:
        return None
    return int(first.in_features)


def replace_ffn(
    module: nn.Module,
    selector: Callable[[str, nn.Module], bool],
    config: ThermodynamicTransformerConfig,
) -> int:
    """Replace selected plain `nn.Sequential` FFNs with `ThermodynamicFFN`.

    Returns the number of replaced child modules.
    """

    replaced = 0

    def visit(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            qualified = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Sequential) and selector(qualified, child):
                embed_dim = _infer_sequential_embed_dim(child)
                if embed_dim is not None and embed_dim == config.embed_dim:
                    replacement = config.build_ffn()
                    first_param = next(child.parameters(), None)
                    if first_param is not None:
                        if first_param.dtype.is_floating_point:
                            replacement = replacement.to(device=first_param.device, dtype=first_param.dtype)
                        else:
                            replacement = replacement.to(device=first_param.device)
                    replacement.train(child.training)
                    parent._modules[name] = replacement
                    replaced += 1
                    continue
            visit(child, qualified)

    visit(module)
    return replaced
