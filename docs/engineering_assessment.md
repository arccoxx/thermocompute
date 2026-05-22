# Engineering Assessment

This package is a credible research-proof release, not a finished production
runtime. The following checks capture what a strict maintainer would look for
before presenting it publicly.

## Added In This Pass

- CI workflow for Linux/Python 3.10 and 3.12.
- Physical PolyForm Noncommercial license file matching package metadata.
- Commercial licensing and contribution policy files.
- Generic probability distribution support through `torch.distributions`.
- Custom distribution adapter for user-provided distribution objects.
- First-order FFN memory estimators for classical and thermodynamic layers.
- Quantization-aware thermodynamic FFN path for fp16/bf16/fp8/int8/int4/int2/binary experiments.
- CPU-light flow matching module with classical and thermodynamic velocity fields.
- CPU-light thermodynamic CNN module, classifier wrapper, and toy local-feature
  experiment.
- Bounded stress script for chunked no-replica inference, cold training,
  distribution families, low-precision formats, CNN coverage, flow matching,
  and memory-law estimates.

## Strong Current Components

- Public PyTorch modules have ordinary tensor contracts and support `.to(...)`,
  `state_dict`, autograd, and `return_info`.
- No-replica chunked inference is the recommended path for large GPU-only
  experiments.
- Low-precision training uses master floating-point parameters plus
  straight-through quantized forward passes, which keeps small experiments
  portable across CPU and GPU.
- Flow matching has a reproducible toy experiment that reports sampling steps,
  MMD, wall time, and diffusion-step speedup proxies.
- Modeled physical time is separated from PyTorch wall time in benchmark
  artifacts.
- Claim boundaries are explicit: GPU wall-clock plateau is empirical; modeled
  physical time is the hardware-facing result.

## Known Research-Grade Areas

- Optional CUDA extension is experimental and not required.
- Training is not constant-time and not claimed to beat classical baselines.
- Parallel tempering has not shown enough value in the current experiments to
  be the default path.
- Chunked inference reduces peak state memory, not parameter memory.
- Chunked training works, but autograd can retain chunk graphs and integration
  intermediates; custom backward/checkpointing would be the next serious memory
  improvement.
- Low-precision support is quantization-aware emulation. It is not a native
  packed int4 CUDA kernel path yet.
- Flow matching is currently a toy 2D feasibility path, not a production
  diffusion replacement.
- CNN support is currently a toy 8x8 coverage path, not a production
  computer-vision benchmark.
- Distribution support covers any distribution available through
  `torch.distributions` plus custom adapters. It does not implement every named
  distribution ever described in probability theory from scratch.
- Earlier public commits were released under MIT. The current noncommercial
  license protects future versions, but it cannot retroactively remove rights
  already granted for older versions.
- If outside contributions become significant, a formal CLA is stronger than
  the lightweight `CONTRIBUTING.md` certification.

## Next Senior-Engineer Improvements

- Add `ruff`/formatting and static type checks.
- Add GPU CI when a hosted runner with CUDA is available.
- Add allocator-level memory profiling benchmarks.
- Add checkpointed/custom backward for thermodynamic integration.
- Add optional custom kernels for chunked projected inference.
- Add versioned benchmark baselines so regressions are visible over time.
- Have counsel review the noncommercial/commercial licensing posture before
  relying on it for a large transaction.
