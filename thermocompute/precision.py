from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ThermodynamicNeuronConfig
from .neurons import ThermodynamicRunInfo
from .transformer import _aggregate_chunk_infos


FormatKind = Literal["float", "float8", "integer", "binary"]


@dataclass(frozen=True)
class NumericFormatSpec:
    """Description of a supported numeric representation."""

    name: str
    bits: int
    kind: FormatKind
    torch_dtype: torch.dtype | None = None
    description: str = ""


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization-aware thermodynamic forward configuration.

    Low-bit integer formats use symmetric fake quantization with a
    straight-through estimator. Floating formats are cast to the storage dtype
    and dequantized back to the compute dtype for portable emulation.
    """

    format: str = "int8"
    compute_dtype: torch.dtype | None = torch.float32
    per_channel: bool = False
    channel_axis: int = 0
    use_ste: bool = True
    quantize_inputs: bool = True
    quantize_weights: bool = True
    quantize_bias: bool = False
    quantize_coefficients: bool = False
    quantize_currents: bool = True
    quantize_states: bool = True
    eps: float = 1e-8


@dataclass(frozen=True)
class QuantizedFitResult:
    """Compact result for quantization-aware FFN training."""

    format: str
    initial_loss: float
    final_loss: float
    n_steps: int
    fit_wall_ms: float


@dataclass(frozen=True)
class QuantizedFFNMemoryEstimate:
    """Bit-level estimate for quantized thermodynamic FFN inference memory."""

    parameter_bits: int
    state_bits: int
    output_bits: int
    peak_bytes: int
    parameter_format: str
    state_format: str
    output_format: str
    hidden_width: int
    batch_tokens: int
    replicas: int = 1
    chunk_size: int | None = None


def _format_table() -> dict[str, NumericFormatSpec]:
    table: dict[str, NumericFormatSpec] = {
        "fp64": NumericFormatSpec("fp64", 64, "float", torch.float64, "native float64"),
        "float64": NumericFormatSpec("float64", 64, "float", torch.float64, "native float64"),
        "fp32": NumericFormatSpec("fp32", 32, "float", torch.float32, "native float32"),
        "float32": NumericFormatSpec("float32", 32, "float", torch.float32, "native float32"),
        "tf32": NumericFormatSpec("tf32", 19, "float", torch.float32, "TensorFloat-32 style matmul precision"),
        "fp16": NumericFormatSpec("fp16", 16, "float", torch.float16, "IEEE float16"),
        "float16": NumericFormatSpec("float16", 16, "float", torch.float16, "IEEE float16"),
        "bf16": NumericFormatSpec("bf16", 16, "float", torch.bfloat16, "brain floating point"),
        "bfloat16": NumericFormatSpec("bfloat16", 16, "float", torch.bfloat16, "brain floating point"),
        "int8": NumericFormatSpec("int8", 8, "integer", None, "symmetric signed int8 fake quantization"),
        "int4": NumericFormatSpec("int4", 4, "integer", None, "symmetric signed int4 fake quantization"),
        "int2": NumericFormatSpec("int2", 2, "integer", None, "symmetric signed int2 fake quantization"),
        "binary": NumericFormatSpec("binary", 1, "binary", None, "scaled sign quantization"),
        "bin1": NumericFormatSpec("bin1", 1, "binary", None, "scaled sign quantization"),
    }
    fp8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    if fp8_e4m3fn is not None:
        table["fp8_e4m3fn"] = NumericFormatSpec("fp8_e4m3fn", 8, "float8", fp8_e4m3fn, "float8 e4m3fn")
    fp8_e5m2 = getattr(torch, "float8_e5m2", None)
    if fp8_e5m2 is not None:
        table["fp8_e5m2"] = NumericFormatSpec("fp8_e5m2", 8, "float8", fp8_e5m2, "float8 e5m2")
    return table


def available_numeric_formats() -> tuple[str, ...]:
    """Return low-precision formats supported by this PyTorch build."""

    return tuple(sorted(_format_table()))


def get_numeric_format(name: str) -> NumericFormatSpec:
    key = str(name).lower().replace("-", "_")
    table = _format_table()
    if key not in table:
        available = ", ".join(available_numeric_formats())
        raise ValueError(f"unknown numeric format {name!r}; available formats: {available}")
    return table[key]


def numeric_format_bits(name: str) -> int:
    return get_numeric_format(name).bits


def quantized_storage_nbytes(num_scalars: int, format: str) -> int:
    if num_scalars < 0:
        raise ValueError("num_scalars must be non-negative")
    return int(math.ceil(num_scalars * numeric_format_bits(format) / 8.0))


def quantize_tensor(
    x: Tensor,
    config: QuantizationConfig | str,
    *,
    channel_axis: int | None = None,
    use_ste: bool | None = None,
    compute_dtype: torch.dtype | None = None,
) -> Tensor:
    """Quantize and dequantize a tensor for portable low-precision emulation."""

    qconfig = config if isinstance(config, QuantizationConfig) else QuantizationConfig(format=str(config))
    fmt = get_numeric_format(qconfig.format)
    target_dtype = compute_dtype or qconfig.compute_dtype or x.dtype
    ste = qconfig.use_ste if use_ste is None else bool(use_ste)

    if fmt.kind in ("float", "float8"):
        if fmt.name == "tf32":
            return _uniform_quantize(x, 10, qconfig, channel_axis=channel_axis, use_ste=ste, compute_dtype=target_dtype)
        return _float_cast_quantize(x, fmt, target_dtype, use_ste=ste)
    if fmt.kind == "binary":
        return _binary_quantize(x, qconfig, channel_axis=channel_axis, use_ste=ste, compute_dtype=target_dtype)
    return _uniform_quantize(
        x,
        fmt.bits,
        qconfig,
        channel_axis=channel_axis,
        use_ste=ste,
        compute_dtype=target_dtype,
    )


def estimate_quantized_thermo_ffn_memory(
    input_dim: int,
    hidden_dim: int,
    *,
    output_dim: int | None = None,
    batch_tokens: int = 1,
    parameter_format: str = "int4",
    state_format: str = "fp16",
    output_format: str = "fp16",
    replicas: int = 1,
    chunk_size: int | None = None,
    state_overhead: float = 3.0,
) -> QuantizedFFNMemoryEstimate:
    """Estimate quantized thermodynamic FFN inference memory at bit granularity."""

    output = input_dim if output_dim is None else int(output_dim)
    if input_dim <= 0 or hidden_dim <= 0 or output <= 0:
        raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    if state_overhead < 0.0:
        raise ValueError("state_overhead must be non-negative")

    active_width = min(hidden_dim, int(chunk_size)) if chunk_size is not None else hidden_dim
    parameter_scalars = hidden_dim * (input_dim + output + 4)
    state_scalars = int(math.ceil(batch_tokens * active_width * (replicas + state_overhead)))
    output_scalars = batch_tokens * output
    parameter_bits = parameter_scalars * numeric_format_bits(parameter_format)
    state_bits = state_scalars * numeric_format_bits(state_format)
    output_bits = output_scalars * numeric_format_bits(output_format)
    peak_bytes = math.ceil((parameter_bits + state_bits + output_bits) / 8.0)
    return QuantizedFFNMemoryEstimate(
        parameter_bits=int(parameter_bits),
        state_bits=int(state_bits),
        output_bits=int(output_bits),
        peak_bytes=int(peak_bytes),
        parameter_format=str(parameter_format),
        state_format=str(state_format),
        output_format=str(output_format),
        hidden_width=int(hidden_dim),
        batch_tokens=int(batch_tokens),
        replicas=int(replicas),
        chunk_size=int(chunk_size) if chunk_size is not None else None,
    )


class QuantizedThermodynamicFFN(nn.Module):
    """Thermodynamic FFN with quantization-aware low-precision forward passes."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        *,
        quantization: QuantizationConfig | str = QuantizationConfig(),
        neuron_config: ThermodynamicNeuronConfig | None = None,
        residual_scale: float = 1.0,
        memory_efficient_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embed_dim and hidden_dim must be positive")
        if memory_efficient_chunk_size is not None and memory_efficient_chunk_size <= 0:
            raise ValueError("memory_efficient_chunk_size must be positive when provided")
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.quantization = (
            quantization if isinstance(quantization, QuantizationConfig) else QuantizationConfig(format=str(quantization))
        )
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
            current = self._currents(flat, 0, self.hidden_dim)
            hidden, info = self._simulate_quantized(current, 0, self.hidden_dim, generator=generator)
            y = self._readout(hidden, 0, self.hidden_dim).view(batch, seq_len, embed_dim)
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
            current = self._currents(flat, start, end)
            hidden, info = self._simulate_quantized(current, start, end, generator=generator)
            flat_out = flat_out + self._readout(hidden, start, end, include_bias=False)
            infos.append(info)
        flat_out = flat_out + self.out_proj.bias
        return flat_out.view(batch, seq_len, self.embed_dim), _aggregate_chunk_infos(infos)

    def _currents(self, flat: Tensor, start: int, end: int) -> Tensor:
        x = self._maybe_quantize(flat, enabled=self.quantization.quantize_inputs, channel_axis=-1)
        weight = self._maybe_quantize(
            self.thermo.weight[start:end],
            enabled=self.quantization.quantize_weights,
            channel_axis=0,
        )
        bias = self._maybe_quantize(
            self.thermo.bias[start:end],
            enabled=self.quantization.quantize_bias,
            channel_axis=0,
        )
        current = F.linear(x, weight, bias)
        return self._maybe_quantize(current, enabled=self.quantization.quantize_currents, channel_axis=-1)

    def _simulate_quantized(
        self,
        current: Tensor,
        start: int,
        end: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, ThermodynamicRunInfo]:
        j2 = self._maybe_quantize(self.thermo.j2[start:end], enabled=self.quantization.quantize_coefficients)
        j3 = self._maybe_quantize(self.thermo.j3[start:end], enabled=self.quantization.quantize_coefficients)
        j4 = self._maybe_quantize(self.thermo.j4[start:end], enabled=self.quantization.quantize_coefficients)
        hidden, info = self.thermo._simulate_current(current, j2, j3, j4, generator=generator)
        hidden = self._maybe_quantize(hidden, enabled=self.quantization.quantize_states, channel_axis=-1)
        return hidden, info

    def _readout(self, hidden: Tensor, start: int, end: int, *, include_bias: bool = True) -> Tensor:
        activated = torch.tanh(hidden)
        activated = self._maybe_quantize(activated, enabled=self.quantization.quantize_states, channel_axis=-1)
        weight = self._maybe_quantize(
            self.out_proj.weight[:, start:end],
            enabled=self.quantization.quantize_weights,
            channel_axis=0,
        )
        bias = self.out_proj.bias if include_bias else None
        if bias is not None:
            bias = self._maybe_quantize(bias, enabled=self.quantization.quantize_bias, channel_axis=0)
        return F.linear(activated, weight, bias)

    def _maybe_quantize(self, x: Tensor, *, enabled: bool, channel_axis: int = 0) -> Tensor:
        if not enabled:
            target_dtype = self.quantization.compute_dtype
            return x.to(dtype=target_dtype) if target_dtype is not None and x.dtype != target_dtype else x
        return quantize_tensor(x, self.quantization, channel_axis=channel_axis)


def fit_quantized_ffn_mse(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    n_steps: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
) -> QuantizedFitResult:
    """Train a quantization-aware FFN on a tiny supervised MSE objective."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    quantization = getattr(model, "quantization", QuantizationConfig(format="fp32"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    with torch.no_grad():
        initial_loss = float(F.mse_loss(_as_tensor_output(model(inputs)), targets).detach().cpu())
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        output = _as_tensor_output(model(inputs))
        loss = F.mse_loss(output, targets)
        loss.backward()
        optimizer.step()
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    model.eval()
    with torch.no_grad():
        final_loss = float(F.mse_loss(_as_tensor_output(model(inputs)), targets).detach().cpu())
    return QuantizedFitResult(
        format=quantization.format,
        initial_loss=initial_loss,
        final_loss=final_loss,
        n_steps=int(n_steps),
        fit_wall_ms=float(fit_wall_ms),
    )


def _float_cast_quantize(x: Tensor, fmt: NumericFormatSpec, compute_dtype: torch.dtype, *, use_ste: bool) -> Tensor:
    if fmt.torch_dtype is None:
        return x.to(dtype=compute_dtype)
    try:
        dequantized = x.to(dtype=fmt.torch_dtype).to(dtype=compute_dtype)
    except RuntimeError:
        dequantized = _uniform_quantize(
            x,
            min(fmt.bits, 8),
            QuantizationConfig(format="int8", compute_dtype=compute_dtype),
            channel_axis=None,
            use_ste=False,
            compute_dtype=compute_dtype,
        )
    return _apply_ste(x.to(dtype=compute_dtype), dequantized, use_ste=use_ste)


def _uniform_quantize(
    x: Tensor,
    bits: int,
    config: QuantizationConfig,
    *,
    channel_axis: int | None,
    use_ste: bool,
    compute_dtype: torch.dtype,
) -> Tensor:
    x_compute = x.to(dtype=compute_dtype)
    qmax = float(2 ** (bits - 1) - 1)
    if qmax <= 0:
        return _binary_quantize(x, config, channel_axis=channel_axis, use_ste=use_ste, compute_dtype=compute_dtype)
    scale = _symmetric_scale(x_compute, qmax, config, channel_axis=channel_axis)
    q = torch.round((x_compute / scale).clamp(-qmax, qmax))
    dequantized = q * scale
    return _apply_ste(x_compute, dequantized, use_ste=use_ste)


def _binary_quantize(
    x: Tensor,
    config: QuantizationConfig,
    *,
    channel_axis: int | None,
    use_ste: bool,
    compute_dtype: torch.dtype,
) -> Tensor:
    x_compute = x.to(dtype=compute_dtype)
    if config.per_channel and x_compute.ndim > 0:
        dims = _reduction_dims(x_compute, channel_axis if channel_axis is not None else config.channel_axis)
        if dims:
            scale = x_compute.abs().mean(dim=dims, keepdim=True).clamp_min(config.eps)
        else:
            scale = x_compute.abs().clamp_min(config.eps)
    else:
        scale = x_compute.abs().mean().clamp_min(config.eps)
    signs = torch.where(x_compute >= 0, torch.ones_like(x_compute), -torch.ones_like(x_compute))
    dequantized = signs * scale
    return _apply_ste(x_compute, dequantized, use_ste=use_ste)


def _symmetric_scale(x: Tensor, qmax: float, config: QuantizationConfig, *, channel_axis: int | None) -> Tensor:
    if config.per_channel and x.ndim > 0:
        dims = _reduction_dims(x, channel_axis if channel_axis is not None else config.channel_axis)
        if dims:
            return (x.abs().amax(dim=dims, keepdim=True) / qmax).clamp_min(config.eps)
    return (x.abs().amax() / qmax).clamp_min(config.eps)


def _reduction_dims(x: Tensor, axis: int) -> tuple[int, ...]:
    normalized = axis if axis >= 0 else x.ndim + axis
    normalized = min(max(normalized, 0), x.ndim - 1)
    return tuple(i for i in range(x.ndim) if i != normalized)


def _apply_ste(x: Tensor, dequantized: Tensor, *, use_ste: bool) -> Tensor:
    if use_ste and x.requires_grad:
        return x + (dequantized - x).detach()
    return dequantized


def _as_tensor_output(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output
