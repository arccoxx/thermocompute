# What This Proves / What It Does Not Prove

## What This Proves

- `thermocompute` can emulate p-bits/PDITs, PMODE, PMOG, quartic thermodynamic neurons, and thermodynamic transformer-style blocks in PyTorch.
- Fixed-depth thermodynamic neuron and transformer feed-forward layers report constant modeled physical time as width increases.
- The benchmark suite separates modeled physical time from PyTorch wall time.
- Wider thermodynamic blocks can be evaluated under a constant physical-time model while classical FFN FLOP proxies grow with width.
- The superiority demo makes the same comparison against a dense digital FFN work proxy and explicitly records state-of-the-art context: optimized attention kernels do not remove the width-dependent FFN term.
- Training APIs exist for conventional cold training, parallel-tempered full-model training, readout alignment, and sparse readout mask search.

## What The CUDA Benchmarks Suggest

- On CUDA, the PyTorch emulator can show a useful wall-clock plateau across moderate thermodynamic width ranges.
- This makes the package useful as a GPU-only stochastic layer family, not only as a future hardware emulator.
- The plateau is an empirical software observation caused by available GPU parallelism, vectorized width, and shape-specific kernel behavior.
- The benchmark artifacts should be read as two separate measurements: `wall_ms_median` for practical GPU performance and `physical_time` for the modeled thermodynamic substrate.

## What This Does Not Prove

- It does not prove PyTorch wall time is universally constant with width.
- It does not prove PyTorch emulator wall time is faster than optimized production kernels such as FlashAttention-class attention kernels.
- It does not prove training is constant time.
- It does not prove training is faster than state-of-the-art transformer training.
- It does not prove real chip speedups or energy gains; there is no hardware backend yet.
- It does not provide full production model support for every Hugging Face architecture.
- It does not make the optional CUDA extension required or production-ready.

## Current Best Claims

The current hardware-facing research claim is:

**Increasing neural width can increase representational capacity while modeled thermodynamic inference time remains fixed under the parallel thermodynamic substrate model.**

The current GPU-facing engineering claim is:

**The PyTorch/CUDA emulator can exhibit a practical wall-clock plateau over useful thermodynamic width ranges, making the software valuable for GPU-only stochastic networks before dedicated thermodynamic hardware exists.**

The first claim is a modeled physical-time result. The second claim is empirical and must be remeasured for each GPU, tensor shape, and dtype.

## Benchmark JSON Schema

Every benchmark artifact is a JSON object:

```json
{
  "name": "benchmark_name",
  "metrics": {
    "device": "cuda",
    "rows": [],
    "physical_time_range": 0.0
  }
}
```

Width-scaling rows should include:

- width identifier: `width`, `out_features`, or `thermo_hidden_dim`
- `physical_time` or `thermo_physical_time`
- PyTorch wall-clock timing such as `wall_ms_median`
- model size fields such as `parameter_count`, `parameter_bytes`, or `*_param_count`
- loss or stability fields where applicable

Acceptance benchmarks should keep `physical_time_range == 0.0` for fixed-depth thermodynamic width sweeps.
