# thermocompute

`thermocompute` is a PyTorch-first emulator for thermodynamic probabilistic
computing. It provides software models of p-bits/PDITs, PMODE Gaussian
samplers, PMOG mixture samplers, generic `torch.distributions` probability
families, and fixed-observation-time thermodynamic neuron layers.

**The punchline:** `thermocompute` lets us study neural layers where width can
behave more like parallel fabric than sequential latency. Today, that matters
in two ways:

1. **GPU-only path:** use thermodynamic layers directly as PyTorch modules. In
   current CUDA experiments, wall-clock inference often enters a useful plateau
   across moderate width ranges because the stochastic width dimension is
   vectorized and the GPU still has parallel headroom.
2. **Sim-to-real path:** the same code models future thermodynamic hardware,
   where increasing width adds parallel physical units and does not lengthen
   the fixed observation window.

Under the massively parallel thermodynamic hardware model, we can run
variable-width fixed-depth neural inference in constant modeled physical time.
Increasing width adds parallel thermodynamic units; it does not lengthen the
fixed physical evolution window.

That is the central research result: a wider thermodynamic layer can expose
more stochastic nonlinear features while reporting the same physical inference
time. Classical dense digital layers must touch width-dependent weights and
activations. The thermodynamic substrate model turns that width term into
parallel physical fabric.

The flagship demonstration is `scripts/run_superiority_demo.py`: it sweeps
thermodynamic transformer FFN width up to 4096 units, projects the same model
to much larger widths, and shows a flat modeled physical-time line while the
dense digital FFN work proxy climbs with width. This is the cleanest current
package result: variable-width neural inference in constant modeled physical
time under the parallel thermodynamic substrate model.

The practical near-term bet is GPU-only thermodynamic inference. The emulator
is not only a future-hardware simulator; it is also a usable stochastic neural
layer family for CUDA today. If your GPU has enough unused parallel capacity,
widening the thermodynamic hidden array can be much cheaper in wall-clock time
than widening a classical dense digital FFN. That plateau is empirical, not a
guarantee, and the benchmark artifacts report wall-clock timing separately so
you can find the saturation point on your own hardware.

The second bet is sim-to-real. The same APIs let you prototype algorithms as
physical stochastic processes before dedicated thermodynamic hardware is
available, then compare modeled physical time against conventional sequential
sampling or digital neural computation.

This is not a claim that PyTorch magically has constant wall-clock time for all
sizes. The emulator runs on CPU or GPU and pays normal software costs. The
important empirical claim is narrower and more useful: on CUDA, there can be a
substantial width range where thermodynamic emulator wall time is roughly flat
because the GPU is executing the stochastic width in parallel. Eventually
memory bandwidth, occupancy, registers, launch overhead, or tensor size should
win and wall time should rise.

The most important distinction in the package is:

```text
PyTorch wall time     = how long this emulator takes on today's CPU/GPU
CUDA plateau          = observed wall-clock regime before GPU saturation
modeled physical time = how long the corresponding thermodynamic substrate runs
```

The first one is a practical software result. The second one is the GPU-only
opportunity. The third one is the hardware-theory anchor.

## License And Commercial Use

Current versions of `thermocompute` are source-available under the
PolyForm Noncommercial License 1.0.0. The code is public for research,
inspection, experimentation, education, and other noncommercial use.

Commercial use requires a separate written commercial license from the project
owner. See [COMMERCIAL.md](COMMERCIAL.md).

Contributions are welcome only under terms that preserve this dual-license
option. See [CONTRIBUTING.md](CONTRIBUTING.md).

This repository was previously published under MIT. Those older versions keep
the terms they were already released under, but future versions are distributed
under the current noncommercial license unless the project owner grants
different terms. The project owner may relicense future versions or grant
commercial licenses at their discretion.

## Use Cases

### 1. GPU-Only Thermodynamic Layers Today

This is the path to prioritize now. `thermocompute` gives you PyTorch modules
that can be dropped into real experiments:

- `ThermodynamicFFN`: `[batch, seq, embed] -> [batch, seq, embed]`
- `ThermodynamicTransformerBlock`: pre-norm attention plus a thermodynamic FFN
- `ThermodynamicTransformerLayer`: research layer with sampled PDIT attention
  and thermodynamic feed-forward width
- `ThermodynamicMLP`: fixed-time stochastic MLPs for non-sequence experiments

The goal is not to beat every optimized dense kernel on every shape. The goal
is to exploit GPU parallelism to emulate very wide stochastic thermodynamic
feature arrays where increasing width may be much cheaper than increasing
classical dense FFN width. That makes the package useful for:

- wide stochastic feature maps
- reservoir/readout experiments
- uncertainty-aware transformer blocks
- synthetic sequence modeling
- energy-based or diffusion-like latent transformations
- fast ablations before thermodynamic hardware exists

The default engineering path is **no replica**:

```text
n_replicas = 1
tempering = False
```

This is the fastest and most memory-efficient mode. Parallel tempering remains
available, but it is an optional exploration method for rugged objectives. In
our current small end-to-end experiment it added cost and provided little
benefit, so the package should be evaluated first in the no-replica setting.

Use the benchmark scripts to measure your own plateau:

```powershell
python scripts/run_benchmarks.py --outdir artifacts
python scripts/run_superiority_demo.py --outdir artifacts
```

Read the `wall_ms_median` fields as GPU software measurements and
`physical_time` fields as modeled substrate measurements.

### 2. Sim-To-Real Thermodynamic Hardware

The same layer definitions are also a clean hardware target. A thermodynamic
substrate would replace the numerical Langevin loop with physical evolution and
readout. The API already exposes the hardware-relevant knobs:

- `t_f`: fixed observation window
- `dt`: emulator integration step
- `temperature`: noise strength
- `j2`, `j3`, `j4`: quartic potential shape
- `n_replicas`, `tempering`, `swap_interval`: replica and tempering schedule

The no-replica path maps to one thermodynamic array and one readout. Replica
ladders map to additional arrays and should be treated as an expensive
capability, not the default substrate assumption.

That makes `thermocompute` a bridge: run GPU-only experiments now, then map
successful thermodynamic blocks to future physical arrays.

### 3. Research Proofs And Claim Boundaries

The package ships benchmark artifacts and JSON schemas so results are
inspectable. The core proof is modeled physical-time scaling. The promising
engineering observation is the CUDA wall-clock plateau. They are related, but
they are not the same claim.

## Motivation

Modern probabilistic inference usually relies on digital iteration: Gibbs
sampling, Langevin updates, diffusion steps, MCMC sweeps, or repeated neural
network evaluations. Those methods are powerful, but their runtime grows with
the number of variables, the number of sampling steps, and often the number of
model layers.

Thermodynamic computing starts from a different premise. Instead of simulating
randomness digitally, the hardware uses physical thermal noise as the sampling
resource. Local stochastic units relax under programmable biases, couplings,
temperatures, and nonlinear potentials. A full array of units evolves
simultaneously for a chosen physical window, then the final voltages or states
are read out.

`thermocompute` is built around that idea:

- Use PyTorch tensors to emulate many thermodynamic units in parallel.
- Treat CUDA as a practical first deployment target, not just a simulator.
- Make physical time explicit through `tau0`, `dt`, and `t_f`.
- Support discrete, continuous, and multimodal probabilistic primitives.
- Build thermodynamic neural layers whose nonlinear activation is produced by
  fixed-time stochastic dynamics instead of a hand-coded activation function.
- Measure whether GPU wall time enters a useful plateau as thermodynamic width
  grows.
- Demonstrate the key scaling idea: variable-width neural inference can keep
  constant modeled physical time when the extra width is mapped to parallel
  thermodynamic units.
- Run experiments that separate emulator wall time from modeled physical time.

The library is intentionally minimal. It does not try to be a full model zoo or
a hardware driver. It focuses on the smallest set of abstractions needed to ask
the research question cleanly:

```text
Can we increase neural width and representational capacity without increasing
modeled physical inference latency?
```

For the thermodynamic neuron and transformer FFN layers in this release, the
answer is yes under the parallel-substrate model. The benchmarks make that
claim measurable and keep it separate from ordinary PyTorch runtime.

## Installation

From this repository:

```powershell
python -m pip install -e .
```

For local tests without installing `pytest`:

```powershell
python scripts/run_tests.py
```

If `pytest` is installed:

```powershell
python -m pytest
```

## Quick Start

```python
import torch
from thermocompute import (
    PMODE,
    PMOG,
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerBlock,
    ThermodynamicTransformerConfig,
    ThermodynamicMLP,
    ThermodynamicTransformerLayer,
    fit_transformer_end_to_end_cold,
    fit_transformer_end_to_end_parallel_tempering,
    fit_transformer_readout_parallel_tempering,
    fit_transformer_readout_ridge,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

pmode = PMODE(tau0=100e-9).to(device)
mu = torch.zeros(4096, device=device)
sigma = torch.ones(4096, device=device) * 0.75
samples = pmode.sample(mu, sigma, n_samples=64, t_total=1e-6)

pmog = PMOG(n_components=3, tau0=100e-9).to(device)
mix_samples, modes = pmog.sample(
    logits=torch.tensor([0.1, 1.2, -0.3], device=device),
    means=torch.tensor([-2.0, 0.25, 2.25], device=device),
    scales=torch.tensor([0.35, 0.25, 0.45], device=device),
    n_samples=2048,
    t_total=1e-6,
)

model = ThermodynamicMLP([1, 32, 1], t_f=1.0, dt=0.05).to(device)
x = torch.linspace(-2, 2, 128, device=device).unsqueeze(-1)
y, info = model(x, return_info=True)
print(y.shape)
print(info.physical_time)

layer = ThermodynamicTransformerLayer(
    embed_dim=64,
    num_heads=4,
    thermo_hidden_dim=512,
    attention_mode="pdit",
    n_attention_samples=4,
    attention_t_f=0.05,
    t_f=0.4,
    dt=0.04,
    memory_efficient_chunk_size=128,
).to(device)

tokens = torch.randn(8, 32, 64, device=device)
tokens_out, layer_info = layer(tokens, causal=True, return_info=True)
tokens_out_low_mem, _ = layer(tokens, causal=True, chunk_size=64, return_info=True)
print(tokens_out.shape)
print(layer_info.physical_time)

target_tokens = torch.sin(tokens)
fit = fit_transformer_readout_ridge(layer, tokens, target_tokens, ridge=1e-1)
print(fit.train_mse)

pt_fit = fit_transformer_readout_parallel_tempering(
    layer,
    tokens,
    target_tokens,
    ridge=1e-1,
    keep_fraction=0.35,
    n_tempering_replicas=6,
    n_tempering_steps=24,
)
print(pt_fit.selected_features, pt_fit.swap_acceptance)

e2e_fit = fit_transformer_end_to_end_parallel_tempering(
    layer,
    tokens,
    target_tokens,
    n_tempering_replicas=5,
    n_tempering_steps=40,
    learning_rate=4e-3,
)
print(e2e_fit.final_train_loss)

cold_fit = fit_transformer_end_to_end_cold(
    layer,
    tokens,
    target_tokens,
    n_steps=40,
    learning_rate=4e-3,
)
print(cold_fit.final_train_loss)

block_config = ThermodynamicTransformerConfig(
    embed_dim=64,
    num_heads=4,
    thermo_hidden_dim=512,
    neuron=ThermodynamicNeuronConfig(t_f=0.2, dt=0.04),
    memory_efficient_chunk_size=128,
)
block = ThermodynamicTransformerBlock(block_config).to(device)
block_out, block_info = block(tokens, causal=True, return_info=True)
print(block_info.physical_time)
```

## Package Layout

```text
thermocompute/
  primitives.py     p-bits, PDIT, PMODE, PMOG, Ising energy
  distributions.py  generic torch.distributions wrappers and adapters
  neurons.py        fixed-time thermodynamic neuron layers and MLPs
  transformer.py    thermodynamic transformer attention and FFN blocks
  integration.py    production-shaped FFN/block wrappers and replacement helper
  training.py       cold/PT end-to-end and readout training helpers
  experiments.py    smoke checks, proof-of-concept checks, scaling studies
  benchmarks.py     research-proof width and baseline benchmarks
  memory.py         FFN memory scaling estimators
  metrics.py        result serialization and small metric helpers
  config.py         device and random seed utilities
  cuda_ext.py       optional CUDA extension loader
  cuda/             experimental custom CUDA kernel source
scripts/
  run_benchmarks.py
  run_stress.py
  run_tests.py
  run_smoke.py
  run_poc.py
  run_experiments.py
tests/
  test_smoke.py
  test_poc.py
```

## Core Concepts

### Physical Time

The library tracks modeled physical time separately from software runtime.

- `tau0` is the relaxation timescale for primitive samplers such as PMODE.
- `dt` is the numerical integration step used by the emulator.
- `t_f` is the fixed observation time for thermodynamic neurons.

For a fixed-depth thermodynamic MLP, modeled physical time is the sum of the
fixed windows for each sequential layer. It does not grow with layer width,
batch size, or number of parallel replicas in the hardware model. In PyTorch,
wall time still depends on tensor sizes, memory bandwidth, and kernel overhead.

This is the central reason the package exists: it gives a concrete software
interface for variable-width neural inference in constant modeled physical time.
The width dimension is treated as parallel physical fabric, not sequential
digital work.

For a single thermodynamic layer, the modeled latency is:

```text
T_layer = n_steps * dt ~= t_f
```

and for a fixed-depth unpipelined stack:

```text
T_model = sum_l T_layer_l
```

There is no width term in that physical-time expression. Width still matters:
it increases parameter count, memory in the emulator, hardware area in the
substrate model, and total power. The research advantage is specifically that
width does not increase the modeled forward-pass latency when all units evolve
in parallel for the same fixed observation window.

### Memory-Efficient Inference

Very wide thermodynamic layers are memory-bound before they are conceptually
latency-bound. A dense classical FFN and a thermodynamic FFN both carry
width-linear parameter memory:

```text
M_params ~= q * H * (D + E)
```

where `q` is bytes per scalar, `H` is hidden thermodynamic width, `D` is input
dimension, and `E` is output dimension. The thermodynamic emulator also needs
state memory:

```text
M_state ~= q * (batch * seq) * H * (replicas + overhead)
```

For inference, `thermocompute` now supports chunked projected evaluation. The
full width still exists in the weights, but the emulator only materializes a
slice of hidden thermodynamic state at a time:

```text
M_state_chunked ~= q * (batch * seq) * C * (replicas + overhead)
```

where `C = memory_efficient_chunk_size`. The readout is accumulated chunk by
chunk:

```text
y = bias + sum_i W_out[:, i:i+C] * tanh(thermo_chunk_i(x))
```

This keeps peak activation/state memory proportional to `C` instead of `H`.
It does not reduce parameter memory, and it can increase software wall time
because chunks run sequentially in the emulator. The modeled hardware physical
time remains `t_f`, because the chunks represent parallel physical units on
the target substrate.

Enable it at construction:

```python
layer = ThermodynamicTransformerLayer(
    embed_dim=4096,
    num_heads=32,
    thermo_hidden_dim=500_000,
    t_f=0.2,
    dt=0.04,
    memory_efficient_chunk_size=8192,
).to(device)
```

or override per call:

```python
y, info = layer(tokens, chunk_size=4096, return_info=True)
```

The same option is available on `ThermodynamicFFN` and
`ThermodynamicTransformerConfig`.

For quick planning, the package includes first-order memory estimators:

```python
from thermocompute import estimate_classical_ffn_memory, estimate_thermo_ffn_memory

classical = estimate_classical_ffn_memory(
    input_dim=4096,
    hidden_dim=500_000,
    batch_tokens=2048,
    dtype_bytes=2,
)

thermo = estimate_thermo_ffn_memory(
    input_dim=4096,
    hidden_dim=500_000,
    batch_tokens=2048,
    dtype_bytes=2,
    replicas=1,
    chunk_size=8192,
)

print(classical.peak_bytes / 1e9)
print(thermo.peak_bytes / 1e9)
```

### Best-Case Memory Scaling: No Replicas

For production GPU use, the best case is the no-replica thermodynamic FFN:

```text
R = 1
tempering = False
memory_efficient_chunk_size = C
```

Compare a classical dense FFN and a no-replica thermodynamic FFN with:

```text
D = input/embed dimension
E = output/embed dimension
H = hidden thermodynamic width
N = batch * sequence length
q = bytes per scalar
C = chunk size
```

Classical dense FFN:

```text
params_classical ~= H(D + E)
activation_classical_peak ~= N H
M_classical_peak ~= q [ H(D + E) + N H + N E ]
```

No-replica thermodynamic FFN, unchunked:

```text
params_thermo ~= H(D + E + 4)
state_thermo_peak ~= N H (1 + overhead)
M_thermo_peak ~= q [ H(D + E + 4) + N H(1 + overhead) + N E ]
```

No-replica thermodynamic FFN, chunked:

```text
params_thermo ~= H(D + E + 4)
state_thermo_chunked_peak ~= N C (1 + overhead)
M_thermo_chunked_peak ~= q [ H(D + E + 4) + N C(1 + overhead) + N E ]
```

So the best-case no-replica memory result is:

```text
parameter memory:  O(H), essentially comparable to classical dense FFN
state memory:      O(NC), independent of full width H when chunked
replica overhead:  none
```

This does not make thermodynamic layers parameter-light. The current dense
readout still stores `D -> H` current weights and `H -> E` readout weights.
The win is that peak stochastic state/activation memory can be capped by `C`
instead of `H`, and the modeled physical-time claim remains tied to the full
parallel array.

For classical dense FFNs, the arithmetic cost still scales with `H`:

```text
F_classical ~= 2N H(D + E)
```

For the thermodynamic hardware model, width maps to physical sites:

```text
T_thermo_physical ~= t_f
```

For the PyTorch emulator, chunking trades memory for software time:

```text
num_chunks = ceil(H / C)
```

Use chunking when VRAM is the bottleneck. Use larger chunks when wall-clock
latency is the bottleneck and VRAM is available.

Cold no-replica training also works with `memory_efficient_chunk_size`, but
training memory is harder than inference memory because autograd may retain
chunk graphs and integration intermediates for backward. Treat chunked cold
training as a compatibility path today; the strongest memory guarantee is for
inference. Future checkpointing or custom backward kernels should improve the
training side.

### BinaryPBit

`BinaryPBit` is a Bernoulli sampler controlled by a voltage-like input:

```python
from thermocompute import BinaryPBit

pbit = BinaryPBit(beta=2.0)
probabilities = pbit.probabilities(control_voltage)
states = pbit.sample(control_voltage)
```

It also supports a vectorized Ising-style Gibbs step when paired with
`IsingEnergy`. States are represented as `0/1`, while the energy internally maps
them to `-1/+1`.

### CategoricalPDIT

`CategoricalPDIT` generalizes the binary p-bit to a categorical sampler:

```python
from thermocompute import CategoricalPDIT

pdit = CategoricalPDIT(beta=1.0)
index = pdit.sample(logits)
```

This emulates a programmable k-state thermodynamic sampling unit where logits
or biases define the categorical distribution.

### PMODE

`PMODE` is a programmable Gaussian sampler. It uses an exact
Ornstein-Uhlenbeck transition rather than a fragile tiny-step approximation:

```python
from thermocompute import PMODE

pmode = PMODE(tau0=100e-9).to(device)
samples = pmode.sample(mu, sigma, n_samples=4096, t_total=1e-6)
```

Each returned sample represents an independent device evolving through the same
physical window. For diagnostics, `PMODE.trajectory(...)` returns the time trace
of a single emulated device.

### PMOG

`PMOG` is a probabilistic mixture-of-Gaussians sampler. It couples a categorical
mode selector to PMODE-style Gaussian relaxation:

```python
from thermocompute import PMOG

pmog = PMOG(n_components=4).to(device)
samples, modes = pmog.sample(logits, means, scales, n_samples=20000)
```

The categorical logits set the mode probabilities, while `means` and `scales`
set the local Gaussian parameters. `switch_rate` can be used to emulate mode
resampling during the physical window.

### Generic Probability Distributions

The package does not try to reimplement every named probability law from
scratch. Instead, it exposes a generic distribution layer over
`torch.distributions`, plus an adapter for custom user-defined distributions.
That gives broad practical coverage: every distribution available in the
installed PyTorch build can be constructed, sampled, and scored through the
same `thermocompute` API.

```python
import torch

from thermocompute import (
    DistributionAdapter,
    DistributionSampler,
    DistributionSpec,
    available_distributions,
    make_distribution,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(available_distributions())

normal = DistributionSampler(
    "normal",
    loc=torch.zeros(4096, device=device),
    scale=torch.ones(4096, device=device),
).to(device)

samples = normal.sample(128)
logp = normal.log_prob(samples)

beta = make_distribution(
    "beta",
    concentration1=torch.ones(32, device=device) * 2,
    concentration0=torch.ones(32, device=device) * 3,
)

spec = DistributionSpec("poisson", {"rate": torch.ones(16, device=device) * 4})
poisson = spec.build()

custom = DistributionAdapter(
    torch.distributions.TransformedDistribution(
        torch.distributions.Normal(torch.zeros(8, device=device), torch.ones(8, device=device)),
        [torch.distributions.transforms.ExpTransform()],
    )
)
custom_samples = custom.sample(64)
```

This is the maintainable version of "support every distribution": the package
supports the full PyTorch distribution ecosystem and any custom distribution
object that follows the `torch.distributions.Distribution` interface. Generic
distribution sampling uses PyTorch's RNG behavior; use `set_seed(...)` for
reproducibility.

### ThermodynamicNeuronLayer

`ThermodynamicNeuronLayer` computes a digital input current and then evolves a
quartic stochastic dynamical system for a fixed observation time:

```python
from thermocompute import ThermodynamicNeuronLayer

layer = ThermodynamicNeuronLayer(
    in_features=16,
    out_features=64,
    j2=1.0,
    j3=0.0,
    j4=1.5,
    temperature=1.0,
    t_f=0.5,
    dt=0.025,
).to(device)

y, info = layer(x, return_info=True)
```

The potential is:

```text
V(x) = 0.5 * J2 * x^2 + (J3 / 3) * x^3 + 0.25 * J4 * x^4 - I * x
```

The Euler-Maruyama update emulates overdamped Langevin dynamics under that
potential. The layer supports multiple replicas and parallel tempering. The
implementation also applies state and force rails, which are a practical
emulator equivalent of bounded circuit voltages and prevent quartic blow-ups in
explicit integration.

### ThermodynamicMLP

`ThermodynamicMLP` stacks thermodynamic neuron layers:

```python
from thermocompute import ThermodynamicMLP

model = ThermodynamicMLP(
    [16, 128, 128, 8],
    t_f=0.4,
    dt=0.04,
).to(device)

y, info = model(x, return_info=True)
print(info.physical_time)
```

The current implementation uses digital matrix multiplication to compute input
currents, then uses stochastic thermodynamic dynamics as the nonlinear sampling
core. This is the right split for a software emulator: it gives the same
algorithmic surface while keeping the analog part explicit.

### ThermodynamicTransformerLayer

`ThermodynamicTransformerLayer` formulates a transformer block whose
feed-forward width is a parallel thermodynamic array. The block is:

```text
x1 = x + attention(norm1(x))
x2 = x1 + out_proj(tanh(thermo_ff(norm2(x1))))
```

The thermodynamic feed-forward core maps each token to input currents for a
variable-width array of quartic neurons. Every neuron in that array evolves for
the same fixed window `t_f`. Therefore, increasing `thermo_hidden_dim` changes
emulator tensor size and hardware area, but not modeled physical time.

```python
from thermocompute import ThermodynamicTransformerLayer

layer = ThermodynamicTransformerLayer(
    embed_dim=128,
    num_heads=8,
    thermo_hidden_dim=2048,
    attention_mode="softmax",
    t_f=0.4,
    dt=0.04,
    memory_efficient_chunk_size=512,
).to(device)

y, info = layer(tokens, return_info=True)
print(info.feedforward_physical_time)
```

Attention has two modes:

- `attention_mode="softmax"` uses ordinary differentiable scaled dot-product
  attention. It is digital token mixing and contributes no thermodynamic
  physical-time window.
- `attention_mode="pdit"` treats each query as a categorical PDIT over keys,
  samples values, and averages them. This is a stochastic sampled-attention
  emulator. Set `attention_t_f` to count a fixed attention sampling window.

The transformer layer is the most direct expression of the package thesis:
variable-width neural networks can be inferenced in constant modeled physical
time when width is implemented as parallel thermodynamic fabric.

### Production-Shaped PyTorch Blocks

For integration with ordinary PyTorch models, use `ThermodynamicFFN`,
`ThermodynamicTransformerBlock`, and `replace_ffn`.

```python
from torch import nn
from thermocompute import (
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerBlock,
    ThermodynamicTransformerConfig,
    replace_ffn,
)

config = ThermodynamicTransformerConfig(
    embed_dim=128,
    num_heads=8,
    thermo_hidden_dim=2048,
    neuron=ThermodynamicNeuronConfig(t_f=0.2, dt=0.04, n_replicas=2, output="mean"),
    memory_efficient_chunk_size=512,
)

block = ThermodynamicTransformerBlock(config).to(device)
out, info = block(tokens, causal=True, return_info=True)
out_low_mem, _ = block(tokens, causal=True, chunk_size=256, return_info=True)

model = nn.Module()
model.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
replaced = replace_ffn(model, lambda name, module: name == "ffn", config)
```

`ThermodynamicFFN` has the same tensor contract as a transformer FFN:
`[batch, seq, embed] -> [batch, seq, embed]`. It is intentionally residual-free
so it can be dropped into blocks that already own their residual connection.
`ThermodynamicTransformerBlock` is a complete pre-norm block with attention,
residuals, and a thermodynamic FFN. Both modules support `.to(device, dtype)`,
`state_dict` save/load, autograd, train/eval mode, and `return_info=True`.

For very wide layers, set `memory_efficient_chunk_size` in the config or pass
`chunk_size` during `forward`. This keeps peak inference state memory bounded
by the chunk size while preserving the public shape contract.

`replace_ffn` is deliberately conservative. It only replaces plain
`nn.Sequential` MLPs whose first linear input size and last linear output size
match `config.embed_dim`. When it replaces a module, it preserves the target
module's device, floating dtype, and train/eval mode. That makes it practical
for real PyTorch experiments without pretending to cover every architecture.

This is production-shaped PyTorch support, not a full Hugging Face integration.
The core package has no hard Hugging Face dependency and no hardware driver.

### Thermodynamic Readout Alignment

`fit_transformer_readout_ridge` implements Thermodynamic Readout Alignment
(TRA), a fast training analogue for the fixed-time inference model.

TRA treats the thermodynamic transformer feed-forward block as a physical
reservoir:

1. Run the token stream through attention and the thermodynamic neuron array.
2. Collect fixed-time stochastic features from the thermodynamic array.
3. Fit only the final projection with one closed-form ridge regression solve.
4. Keep inference exactly the same: one fixed thermodynamic evolution window,
   followed by the learned readout.

```python
from thermocompute import ThermodynamicTransformerLayer, fit_transformer_readout_ridge

layer = ThermodynamicTransformerLayer(
    embed_dim=64,
    num_heads=4,
    thermo_hidden_dim=1024,
    t_f=0.2,
    dt=0.04,
    memory_efficient_chunk_size=256,
).to(device)

tokens = torch.randn(128, 16, 64, device=device)
targets = torch.sin(tokens + 0.25 * torch.roll(tokens, shifts=1, dims=1))

fit = fit_transformer_readout_ridge(
    layer,
    tokens,
    targets,
    ridge=1e-1,
    feature_repeats=2,
)

print(fit.train_mse)
print(fit.fit_wall_ms)
print(fit.physical_time)
```

This is not full end-to-end transformer training. It is closer to reservoir
computing or extreme learning machines, but with the reservoir generated by
fixed-time thermodynamic dynamics. It is valuable because it gives a training
path whose expensive nonlinear feature generation has the same parallel
physical-time structure as inference. The digital work is the ridge solve and
readout programming.

### Cold End-To-End Training

`fit_transformer_end_to_end_cold` exposes ordinary single-replica end-to-end
training. This is the default no-ridge training path: no replica ladder, no
tempering swaps, and the lowest memory footprint among the end-to-end methods.
It is the right first choice for smooth objectives, large widths, and GPU-only
experiments.

```python
from thermocompute import ThermodynamicTransformerLayer, fit_transformer_end_to_end_cold

layer = ThermodynamicTransformerLayer(
    embed_dim=32,
    num_heads=4,
    thermo_hidden_dim=128,
    t_f=0.2,
    dt=0.04,
    memory_efficient_chunk_size=64,
).to(device)

fit = fit_transformer_end_to_end_cold(
    layer,
    train_tokens,
    train_targets,
    eval_inputs=eval_tokens,
    eval_targets=eval_targets,
    n_steps=40,
    learning_rate=4e-3,
)

print(fit.final_train_loss)
print(fit.final_eval_loss)
print(fit.fit_wall_ms)
```

Use cold training when:

- the objective is smooth or easy
- you need the cheapest no-ridge baseline
- memory is tight
- training time matters more than exploration
- you have not yet shown that replica ladders help your scale/task

In the current toy inductive transformer experiment, cold training performs
surprisingly well: it reaches nearly the same train and eval loss as the
parallel-tempered version while using one model replica instead of five. The PT
version gives a small eval improvement, but costs more wall time and memory.

### Parallel Tempered End-To-End Training

`fit_transformer_end_to_end_parallel_tempering` is an optional exploration
method: no ridge solve, no closed-form readout fitting, and no frozen reservoir
assumption, but it keeps several full model copies. Use it only after a cold
single-replica baseline gets stuck or when there is evidence the task has a
rugged loss landscape.

The method keeps several full copies of the transformer layer:

1. Each replica trains all differentiable parameters by gradient descent.
2. Hot replicas receive more Langevin parameter noise.
3. Cold replicas exploit low-loss regions.
4. Adjacent replicas periodically swap whole parameter states with a
   Metropolis parallel-tempering rule.
5. The best final replica is loaded back into the original layer.

```python
from thermocompute import (
    ThermodynamicTransformerLayer,
    fit_transformer_end_to_end_parallel_tempering,
)

layer = ThermodynamicTransformerLayer(
    embed_dim=32,
    num_heads=4,
    thermo_hidden_dim=128,
    t_f=0.2,
    dt=0.04,
    memory_efficient_chunk_size=64,
).to(device)

fit = fit_transformer_end_to_end_parallel_tempering(
    layer,
    train_tokens,
    train_targets,
    eval_inputs=eval_tokens,
    eval_targets=eval_targets,
    n_tempering_replicas=5,
    n_tempering_steps=40,
    learning_rate=4e-3,
    noise_scale=2e-3,
)

print(fit.initial_train_loss)
print(fit.final_train_loss)
print(fit.final_eval_loss)
print(fit.swap_acceptance)
```

Use parallel-tempered end-to-end training when:

- losses are rugged or multimodal
- runs get stuck in poor basins
- extra memory for full replicas is acceptable
- a small accuracy gain is worth more training compute

This is not currently the package's main scaling result. We have not yet shown
that parallel tempering becomes more valuable at large thermodynamic width. It
is included because it is a plausible exploration tool, not because it beat the
no-replica path in the first experiment.

### Parallel Tempered Mask Alignment

`fit_transformer_readout_parallel_tempering` is the sparse structural training
method. It uses parallel tempering directly during training.

The method searches over sparse binary masks on the thermodynamic reservoir
features:

1. Each replica holds a candidate feature mask.
2. Low-temperature replicas exploit good sparse masks.
3. High-temperature replicas explore by accepting worse masks more often.
4. Adjacent replicas swap masks using a Metropolis tempering rule.
5. For each mask, the readout is solved by ridge regression on the selected
   features.
6. The best mask is programmed into the output projection as a sparse readout.

```python
from thermocompute import (
    ThermodynamicTransformerLayer,
    fit_transformer_readout_parallel_tempering,
)

layer = ThermodynamicTransformerLayer(
    embed_dim=64,
    num_heads=4,
    thermo_hidden_dim=1024,
    t_f=0.2,
    dt=0.04,
    memory_efficient_chunk_size=256,
).to(device)

fit = fit_transformer_readout_parallel_tempering(
    layer,
    tokens,
    targets,
    ridge=1e-1,
    keep_fraction=0.35,
    sparsity_penalty=1e-3,
    n_tempering_replicas=6,
    n_tempering_steps=32,
)

print(fit.train_mse)
print(fit.selected_fraction)
print(fit.swap_acceptance)
```

This is useful when the hardware or deployment target benefits from sparse
readout wiring. The search problem is discrete and nonconvex, so parallel
tempering is a natural fit: hot replicas discover alternate feature subsets,
while cold replicas preserve strong candidates. Inference remains the same
fixed-time thermodynamic pass.

## Running Experiments

Smoke checks:

```powershell
python scripts/run_smoke.py
```

Proof-of-concept checks:

```powershell
python scripts/run_poc.py
```

Full experiment suite:

```powershell
python scripts/run_experiments.py --outdir artifacts
```

Research-proof benchmark suite:

```powershell
python scripts/run_benchmarks.py --outdir artifacts
```

Flagship superiority demo:

```powershell
python scripts/run_superiority_demo.py --outdir artifacts
```

Current experiments include:

- Fixed-physical-time width scaling for thermodynamic MLPs.
- Parallel tempering escape on a double-well landscape.
- PMOG multimodal fidelity against programmed target mixture moments.
- Fixed-physical-time width scaling for thermodynamic transformer FFN width.
- PDIT sampled-attention convergence against softmax attention.
- Thermodynamic Readout Alignment on a nonlinear token mapping.
- Parallel Tempered Mask Alignment for sparse thermodynamic readout training.
- End-to-end parallel-tempered inductive transformer training with no ridge
  solve.
- Cold single-replica end-to-end inductive training as the default no-ridge
  training path.
- Superiority demo against dense digital FFN scaling: measured widths up to
  4096 and projected widths up to 65536, with modeled physical time flat and
  digital work increasing.

The benchmark suite writes JSON artifacts using this shape:

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

The flagship benchmark artifact is `fixed_time_advantage.png`: thermodynamic
modeled physical time stays flat with width while the classical FFN FLOP proxy
grows.

The flagship demo artifacts are:

- `superiority_demo.json`: source-of-truth metrics and claim boundary.
- `superiority_demo.md`: compact academic-style summary table.
- `superiority_latency_advantage.png`: fixed thermodynamic physical time vs
  rising dense digital work.
- `superiority_advantage_factor.png`: modeled latency advantage factor as width
  grows.
- `superiority_loss_vs_width.png`: sanity task loss after a matched readout fit.

## What This Proves / What It Does Not Prove

See [docs/claims.md](docs/claims.md) for the explicit claim boundary.
See [docs/engineering_assessment.md](docs/engineering_assessment.md) for the
maintainer-style assessment of current strengths, gaps, and next hardening
steps.

In short:

- Proves: fixed-depth thermodynamic neuron/transformer blocks report constant
  modeled physical time as width increases.
- Proves: the benchmark suite separates modeled physical time from PyTorch wall
  time and classical FLOP proxies.
- Suggests: on CUDA, emulator wall time can remain roughly flat across useful
  thermodynamic width ranges before hardware saturation.
- Does not prove: PyTorch wall time is universally constant with width.
- Does not prove: training is constant-time or faster than state-of-the-art
  classical training.
- Does not prove: real chip speedups; this is an emulator with no hardware
  backend.

Outputs are written as JSON and PNG files under `artifacts/`.

## Optional CUDA Kernels

The default implementation is pure PyTorch and works on CPU or CUDA. The
package also includes an experimental CUDA extension source in
`thermocompute/cuda/thermo_kernels.cu` and a loader in `thermocompute/cuda_ext.py`.

The extension is optional. If compilation fails or a CUDA compiler is not
available, the package falls back to PyTorch kernels.

```python
from thermocompute.cuda_ext import has_cuda_extension

print(has_cuda_extension())
```

## Scaling Advantage

There are now two scaling stories in the package.

The first is practical and immediate: **GPU-only thermodynamic inference can
show a wall-clock plateau.** The PyTorch implementation vectorizes the
thermodynamic width dimension. On a CUDA device with unused parallel capacity,
increasing `thermo_hidden_dim` may not produce a proportional wall-clock
increase. This makes the software useful on its own, even before a physical
thermodynamic chip exists.

The second is theoretical and hardware-facing: **modeled physical inference
time is constant with width** under the parallel thermodynamic substrate model.
That claim is stronger and cleaner, but it depends on the future hardware
assumption that width maps to parallel physical units.

The important scaling distinction is therefore between GPU emulator runtime and
modeled physical runtime.

In a conventional digital sampler, sampling cost usually grows with at least one
of these quantities:

- number of variables
- number of MCMC sweeps or diffusion steps
- number of replicas or chains
- number of hidden units
- batch size

In the thermodynamic hardware model, many of those dimensions become parallel
area and power costs rather than sequential time costs. If a million units are
available on the substrate, they all evolve during the same physical window.
The modeled forward-pass time is controlled by the relaxation window, not by the
number of units participating in that window.

This package exposes that separation directly. In the width-scaling
experiments, the thermodynamic MLP and transformer FFN modeled physical time
stay fixed as width increases. PyTorch wall time is still a normal GPU
measurement, and it is valuable in its own right: if it stays roughly flat over
your target width range, then the GPU-only path is already useful as a wide
stochastic layer family. The physical-time metric shows what the same
computation would ask from a massively parallel thermodynamic substrate.

The same principle now applies to the thermodynamic transformer layer. The
`thermo_hidden_dim` can be varied from small to very wide while the modeled
feed-forward physical time remains `t_f`. That is the central result: variable
width neural inference in constant modeled physical time.

For the GPU-only path, the key engineering question is not "is wall time
mathematically constant?" It is "where is the plateau on this GPU for this
batch, sequence length, dtype, and width range?" The shipped benchmarks answer
that question empirically by reporting `wall_ms_median` beside
`physical_time`.

The readout-alignment experiment adds a first training counterpart: a wider
thermodynamic transformer reservoir improves one-shot fitted error while the
inference window remains fixed. Training still includes a digital solve, but it
does not require backpropagating through many stochastic inference passes.

## What This Could Mean For Huge Models

For huge models, the first serious use case is GPU-only experimentation. Treat
the thermodynamic layer as a practical stochastic PyTorch module that can
approximate very wide nonlinear feature arrays while measuring whether the CUDA
wall-clock cost remains flat enough to matter. This can be useful before any
thermodynamic chip exists.

The most interesting target is not replacing every digital operation. The
strongest near-term opportunity is replacing the stochastic and probabilistic
core: sampling, uncertainty propagation, energy-based inference,
diffusion-like refinement, latent-variable search, and ensemble-style
exploration.

The highest-leverage claim is simple and aggressive: if the model's width is
mapped onto parallel thermodynamic units, then widening the neural computation
does not have to lengthen the modeled physical inference time. To our knowledge,
this is not how mainstream transformer scaling is usually framed. Conventional
GPU inference treats width as more arithmetic. A thermodynamic substrate treats
width as more parallel physical dynamics.

If those computations can be mapped to dense thermodynamic arrays, the scaling
profile changes:

- On GPUs today, moderate width increases may fit inside an existing parallel
  execution plateau rather than producing proportional latency.
- Wider latent spaces increase hardware area, not necessarily physical latency.
- Wider transformer feed-forward blocks increase thermodynamic fabric, not
  necessarily the fixed-time inference window.
- More parallel samples increase device count, not necessarily sequential
  sampling time.
- Multimodal exploration can use replica temperature ladders in the same fixed
  physical window.
- Energy-based and diffusion-like models could trade long digital sampling
  loops for short analog relaxation windows.
- Training can remain digital while inference and sampling move to a lower
  energy physical substrate.

The practical implication is that frontier-scale probabilistic models may not
need to pay digital iteration costs for every sample forever. If the analog
substrate is large enough and well matched to the model, the bottleneck can move
from "how many sequential sampling steps do we need?" to "how much parallel
thermodynamic fabric can we afford?"

For huge transformers, the thermodynamic transformer layer points at a concrete
hybrid path: keep token routing, memory, and parts of attention digital where
that is practical, but move the wide nonlinear feed-forward and stochastic
sampling work into thermodynamic layers. On GPUs, that means exploiting the
wall-clock plateau where it exists. On hardware, that means fixed-time physical
dynamics. That is how this package frames the frontier opportunity: inference
of variable-width neural blocks with unusually weak latency scaling today, and
constant modeled physical time on the target substrate.

Thermodynamic Readout Alignment is the first practical training answer in this
direction. Instead of training every stochastic element by backpropagation, the
thermodynamic block can act as a huge physical feature generator and the readout
can be solved or locally adapted. For very large models, that suggests a hybrid
training stack: digital pretraining and routing where needed, local or
closed-form readout alignment where possible, and thermodynamic inference for
the wide stochastic nonlinear core.

Parallel Tempered End-To-End Training is an optional answer when a frozen
reservoir is not enough and the cold path appears stuck. It trains the whole
thermodynamic transformer layer by running multiple temperature replicas of the
model itself. That gives a path toward ordinary inductive training while
retaining thermodynamic exploration as an optimization mechanism, but it should
not be assumed to win without scale-specific evidence.

Cold End-To-End Training matters for pragmatism. If a single replica already
learns well, it is the right default because it is cheaper in memory and wall
time. Parallel tempering should be treated as an exploration upgrade, not a
free lunch.

Parallel Tempered Mask Alignment adds the next ingredient: sparse structural
search over the physical feature fabric. That matters for hardware because not
every thermodynamic feature needs to be wired into every downstream channel. A
parallel-tempered training phase can search the wiring pattern while preserving
the constant-time inference story after the sparse readout is programmed.

That does not remove engineering constraints. Coupling precision, calibration,
readout bandwidth, memory movement, temperature control, and train-to-hardware
mapping all matter. But the upside is large enough to justify building software
emulators now: they let us discover which algorithms actually benefit from the
physics before the hardware path is fixed.

## Development Notes

Run all local checks:

```powershell
python scripts/run_tests.py
python scripts/run_stress.py
python scripts/run_smoke.py
python scripts/run_poc.py
python scripts/run_experiments.py --outdir artifacts
python scripts/run_benchmarks.py --outdir artifacts
```

Keep claims grounded when interpreting results:

- `physical_time` is an emulator metric, not a measured chip benchmark.
- PyTorch/CUDA wall time is a practical software result. A flat wall-clock
  plateau is useful for GPU-only deployment, but it is not the same as a
  hardware-theory guarantee.
- The GPU wall-clock plateau is shape- and device-dependent. Expect saturation
  once width, batch, sequence length, dtype, memory bandwidth, or kernel
  occupancy becomes limiting.
- Constant physical time applies to parallel units inside a fixed sequential
  layer schedule. Deeper unpipelined networks still add sequential fixed-time
  windows.
- The variable-width constant-time claim applies to modeled physical time under
  a parallel thermodynamic substrate. It does not mean CPU/GPU software runtime
  is independent of tensor width for all shapes.
- Thermodynamic Readout Alignment is a fast readout-training method, not a
  complete replacement for all gradient-based training.
- Cold End-To-End Training is the default no-ridge inductive path. It is the
  first method to try when memory or training time matters.
- Parallel Tempered End-To-End Training is an optional exploration upgrade. It
  is more expensive because it trains full model replicas, and our current
  small experiment did not show enough value to make it the default.
- Parallel Tempered Mask Alignment uses tempering to search sparse readout
  structure. It is currently a training-time algorithm; inference still uses a
  single fixed thermodynamic pass.
