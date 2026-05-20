# Thermocompute Superiority Demo

Modeled thermodynamic physical inference time stays fixed as width grows; dense digital FFN work grows with width.

This demo compares modeled thermodynamic physical latency against a dense digital FFN work proxy.
It reports PyTorch wall time separately and does not claim emulator wall time beats optimized kernels.

## State-of-the-Art Context

- Optimized attention kernels can make exact attention much faster, but transformer FFNs still carry width-dependent dense work.
- Dense digital FFNs must touch width-dependent weights and activations unless the model class changes through sparsity, low-rank structure, quantization, or approximation.
- The thermodynamic model maps extra width to extra parallel physical units and reads them after the same fixed observation window.
- This is an inference-scaling demo, not a training-speed benchmark and not a silicon measurement.

## Measured Widths

| width | thermo physical time | digital work growth | modeled advantage | thermo MSE | classical MSE |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.200 | 1.0x | 1.0x | 0.3373 | 0.2125 |
| 128 | 0.200 | 1.9x | 1.9x | 0.3244 | 0.2039 |
| 256 | 0.200 | 3.7x | 3.7x | 0.3163 | 0.2176 |
| 512 | 0.200 | 7.2x | 7.2x | 0.3573 | 0.2652 |
| 1024 | 0.200 | 14.3x | 14.3x | 0.6451 | 0.4891 |
| 2048 | 0.200 | 28.6x | 28.6x | 0.7969 | 0.7058 |
| 4096 | 0.200 | 57.0x | 57.0x | 0.3099 | 0.4003 |

## Projected Widths

| width | thermo physical time | digital work growth | modeled advantage |
|---:|---:|---:|---:|
| 8192 | 0.200 | 113.9x | 113.9x |
| 16384 | 0.200 | 227.7x | 227.7x |
| 32768 | 0.200 | 455.2x | 455.2x |
| 65536 | 0.200 | 910.3x | 910.3x |

## Claim Boundary

- Strongly supported: fixed modeled physical time as width grows.
- Strongly supported: dense digital FFN work grows with width.
- Not claimed: PyTorch emulator wall time is faster than all production kernels.
- Not claimed: training is constant time.