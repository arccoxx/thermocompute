# Engineering Assessment

This package is a credible research-proof release, not a finished production
runtime. The following checks capture what a strict maintainer would look for
before presenting it publicly.

## Added In This Pass

- CI workflow for Linux/Python 3.10 and 3.12.
- Physical MIT license file matching package metadata.
- Generic probability distribution support through `torch.distributions`.
- Custom distribution adapter for user-provided distribution objects.
- First-order FFN memory estimators for classical and thermodynamic layers.
- Bounded stress script for chunked no-replica inference, cold training,
  distribution families, and memory-law estimates.

## Strong Current Components

- Public PyTorch modules have ordinary tensor contracts and support `.to(...)`,
  `state_dict`, autograd, and `return_info`.
- No-replica chunked inference is the recommended path for large GPU-only
  experiments.
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
- Distribution support covers any distribution available through
  `torch.distributions` plus custom adapters. It does not implement every named
  distribution ever described in probability theory from scratch.

## Next Senior-Engineer Improvements

- Add `ruff`/formatting and static type checks.
- Add GPU CI when a hosted runner with CUDA is available.
- Add allocator-level memory profiling benchmarks.
- Add checkpointed/custom backward for thermodynamic integration.
- Add optional custom kernels for chunked projected inference.
- Add versioned benchmark baselines so regressions are visible over time.
