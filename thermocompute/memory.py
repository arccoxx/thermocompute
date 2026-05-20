from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FFNMemoryEstimate:
    """First-order memory estimate for FFN-style layers.

    The estimate is intentionally simple and architecture-facing. It counts
    dense weights and the dominant inference activation/state terms; framework
    allocator overhead, attention tensors, KV cache, and autograd saved tensors
    are outside this estimate.
    """

    parameter_bytes: int
    state_bytes: int
    output_bytes: int
    peak_bytes: int
    hidden_width: int
    batch_tokens: int
    dtype_bytes: int
    replicas: int = 1
    chunk_size: int | None = None


def estimate_classical_ffn_memory(
    input_dim: int,
    hidden_dim: int,
    *,
    output_dim: int | None = None,
    batch_tokens: int = 1,
    dtype_bytes: int = 2,
) -> FFNMemoryEstimate:
    """Estimate dense FFN inference memory.

    Counts `input_dim -> hidden_dim` and `hidden_dim -> output_dim` weights,
    one hidden activation, and one output activation.
    """

    output = input_dim if output_dim is None else int(output_dim)
    _validate_memory_args(input_dim, hidden_dim, output, batch_tokens, dtype_bytes)
    parameter_scalars = hidden_dim * (input_dim + output)
    state_scalars = batch_tokens * hidden_dim
    output_scalars = batch_tokens * output
    return FFNMemoryEstimate(
        parameter_bytes=int(dtype_bytes * parameter_scalars),
        state_bytes=int(dtype_bytes * state_scalars),
        output_bytes=int(dtype_bytes * output_scalars),
        peak_bytes=int(dtype_bytes * (parameter_scalars + state_scalars + output_scalars)),
        hidden_width=int(hidden_dim),
        batch_tokens=int(batch_tokens),
        dtype_bytes=int(dtype_bytes),
    )


def estimate_thermo_ffn_memory(
    input_dim: int,
    hidden_dim: int,
    *,
    output_dim: int | None = None,
    batch_tokens: int = 1,
    dtype_bytes: int = 2,
    replicas: int = 1,
    chunk_size: int | None = None,
    state_overhead: float = 3.0,
) -> FFNMemoryEstimate:
    """Estimate thermodynamic FFN inference memory.

    Counts current weights, readout weights, per-neuron thermodynamic
    coefficients, output activations, and the dominant stochastic state term.
    If `chunk_size` is set, peak state memory is estimated using that chunk
    rather than the full hidden width.
    """

    output = input_dim if output_dim is None else int(output_dim)
    _validate_memory_args(input_dim, hidden_dim, output, batch_tokens, dtype_bytes)
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    if state_overhead < 0.0:
        raise ValueError("state_overhead must be non-negative")

    active_width = min(hidden_dim, int(chunk_size)) if chunk_size is not None else hidden_dim
    parameter_scalars = hidden_dim * (input_dim + output + 4)
    state_scalars = batch_tokens * active_width * (replicas + state_overhead)
    output_scalars = batch_tokens * output
    peak = parameter_scalars + state_scalars + output_scalars
    return FFNMemoryEstimate(
        parameter_bytes=int(dtype_bytes * parameter_scalars),
        state_bytes=int(dtype_bytes * state_scalars),
        output_bytes=int(dtype_bytes * output_scalars),
        peak_bytes=int(dtype_bytes * peak),
        hidden_width=int(hidden_dim),
        batch_tokens=int(batch_tokens),
        dtype_bytes=int(dtype_bytes),
        replicas=int(replicas),
        chunk_size=int(chunk_size) if chunk_size is not None else None,
    )


def _validate_memory_args(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    batch_tokens: int,
    dtype_bytes: int,
) -> None:
    if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim, hidden_dim, and output_dim must be positive")
    if batch_tokens <= 0:
        raise ValueError("batch_tokens must be positive")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
