from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Optional

import torch
from torch import Tensor, nn

from .neurons import ThermodynamicNeuronLayer, ThermodynamicRunInfo


AttentionMode = Literal["softmax", "pdit"]


@dataclass(frozen=True)
class ThermodynamicTransformerInfo:
    """Runtime metadata for a thermodynamic transformer layer."""

    physical_time: float
    attention_physical_time: float
    feedforward_physical_time: float
    n_steps: int
    n_replicas: int
    attention_mode: AttentionMode
    attention_samples: int
    used_tempering: bool
    swap_attempts: int
    swap_acceptance: float


class ThermodynamicSelfAttention(nn.Module):
    """Self-attention with optional thermodynamic PDIT-style key sampling.

    `mode="softmax"` is differentiable conventional attention. `mode="pdit"`
    treats each query as a categorical thermodynamic sampling unit over keys and
    averages `n_attention_samples` sampled values. The PDIT mode is a stochastic
    emulator of sampled attention, not a differentiable training primitive.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mode: AttentionMode = "softmax",
        n_attention_samples: int = 1,
        beta: float = 1.0,
        attention_t_f: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide embed_dim")
        if n_attention_samples <= 0:
            raise ValueError("n_attention_samples must be positive")
        if mode not in ("softmax", "pdit"):
            raise ValueError("mode must be 'softmax' or 'pdit'")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.mode: AttentionMode = mode
        self.n_attention_samples = int(n_attention_samples)
        self.beta = float(beta)
        self.attention_t_f = float(attention_t_f)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    @property
    def physical_time(self) -> float:
        return self.attention_t_f if self.mode == "pdit" else 0.0

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _apply_masks(self, scores: Tensor, attn_mask: Optional[Tensor], causal: bool) -> Tensor:
        if causal:
            seq_len = scores.shape[-1]
            causal_mask = torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(causal_mask.view(1, 1, seq_len, seq_len), torch.finfo(scores.dtype).min)
        if attn_mask is None:
            return scores
        mask = attn_mask.to(device=scores.device)
        if mask.dtype == torch.bool:
            while mask.ndim < scores.ndim:
                mask = mask.unsqueeze(0)
            return scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        while mask.ndim < scores.ndim:
            mask = mask.unsqueeze(0)
        return scores + mask.to(dtype=scores.dtype)

    def _sample_attention(self, scores: Tensor, values: Tensor, generator: Optional[torch.Generator]) -> Tensor:
        batch, heads, query_len, key_len = scores.shape
        probs = torch.softmax((self.beta * scores).float(), dim=-1).to(dtype=values.dtype)
        flat_probs = probs.reshape(-1, key_len)
        sampled = torch.multinomial(
            flat_probs,
            self.n_attention_samples,
            replacement=True,
            generator=generator,
        )
        values_bh = values.reshape(batch * heads, key_len, self.head_dim)
        bh_index = torch.arange(batch * heads, device=values.device).repeat_interleave(query_len)
        gathered = values_bh[bh_index.unsqueeze(-1), sampled]
        return gathered.mean(dim=1).view(batch, heads, query_len, self.head_dim)

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, embed_dim]")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self._apply_masks(scores, attn_mask, causal)

        if self.mode == "pdit":
            y = self._sample_attention(scores, v, generator)
        else:
            attn = torch.softmax(scores, dim=-1)
            y = torch.matmul(attn, v)

        y = y.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], self.embed_dim)
        return self.out_proj(y)


class ThermodynamicTransformerLayer(nn.Module):
    """Transformer block with a fixed-time thermodynamic feed-forward core.

    Formulation:

    1. Normalize tokens and mix them with self-attention.
    2. Normalize the residual stream again.
    3. Convert each token to input currents for a variable-width array of
       quartic thermodynamic neurons.
    4. Evolve that whole array for the same fixed physical window `t_f`.
    5. Project the thermodynamic states back to the residual width.

    Increasing `thermo_hidden_dim` adds parallel thermodynamic units. It does
    not increase `physical_time` for a fixed-depth layer in the hardware model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        thermo_hidden_dim: int,
        attention_mode: AttentionMode = "softmax",
        n_attention_samples: int = 1,
        attention_beta: float = 1.0,
        attention_t_f: float = 0.0,
        j2: float = 1.0,
        j3: float = 0.0,
        j4: float = 1.5,
        temperature: float = 1.0,
        t_f: float = 0.5,
        dt: float = 0.025,
        n_replicas: int = 1,
        tempering: bool = False,
        swap_interval: int = 4,
        thermo_output: Literal["cold", "mean"] = "cold",
        state_clip: float = 8.0,
        force_clip: float = 80.0,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if thermo_hidden_dim <= 0:
            raise ValueError("thermo_hidden_dim must be positive")
        self.embed_dim = int(embed_dim)
        self.thermo_hidden_dim = int(thermo_hidden_dim)
        self.residual_scale = float(residual_scale)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = ThermodynamicSelfAttention(
            embed_dim,
            num_heads,
            mode=attention_mode,
            n_attention_samples=n_attention_samples,
            beta=attention_beta,
            attention_t_f=attention_t_f,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.thermo_ff = ThermodynamicNeuronLayer(
            embed_dim,
            thermo_hidden_dim,
            j2=j2,
            j3=j3,
            j4=j4,
            temperature=temperature,
            t_f=t_f,
            dt=dt,
            n_replicas=n_replicas,
            tempering=tempering,
            swap_interval=swap_interval,
            output=thermo_output,
            state_clip=state_clip,
            force_clip=force_clip,
        )
        self.out_proj = nn.Linear(thermo_hidden_dim, embed_dim)

    @property
    def physical_time(self) -> float:
        return self.attention.physical_time + self.thermo_ff.physical_time

    def forward(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        generator: Optional[torch.Generator] = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicTransformerInfo]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, embed_dim]")
        x = x + self.residual_scale * self.attention(
            self.norm1(x),
            attn_mask=attn_mask,
            causal=causal,
            generator=generator,
        )
        ff_in = self.norm2(x)
        batch, seq_len, embed_dim = ff_in.shape
        ff_flat = ff_in.reshape(batch * seq_len, embed_dim)
        thermo, ff_info = self.thermo_ff(ff_flat, generator=generator, return_info=True)
        thermo = torch.tanh(thermo)
        ff = self.out_proj(thermo).view(batch, seq_len, embed_dim)
        y = x + self.residual_scale * ff

        info = ThermodynamicTransformerInfo(
            physical_time=self.physical_time,
            attention_physical_time=self.attention.physical_time,
            feedforward_physical_time=ff_info.physical_time,
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

    def thermodynamic_features(
        self,
        x: Tensor,
        *,
        attn_mask: Optional[Tensor] = None,
        causal: bool = False,
        generator: Optional[torch.Generator] = None,
        return_base: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, ThermodynamicRunInfo]:
        """Return fixed-time thermodynamic feed-forward features.

        This is useful for reservoir/readout-style training. The returned
        features have shape `[batch, seq_len, thermo_hidden_dim]`.
        """

        if x.ndim != 3:
            raise ValueError("x must have shape [batch, seq_len, embed_dim]")
        base = x + self.residual_scale * self.attention(
            self.norm1(x),
            attn_mask=attn_mask,
            causal=causal,
            generator=generator,
        )
        ff_in = self.norm2(base)
        batch, seq_len, embed_dim = ff_in.shape
        ff_flat = ff_in.reshape(batch * seq_len, embed_dim)
        thermo, info = self.thermo_ff(ff_flat, generator=generator, return_info=True)
        features = torch.tanh(thermo).view(batch, seq_len, self.thermo_hidden_dim)
        if return_base:
            return features, base, info
        return features
