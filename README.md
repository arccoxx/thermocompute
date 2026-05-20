# thermocompute

`thermocompute` is a PyTorch-first emulator for thermodynamic probabilistic
computing. It provides software models of p-bits/PDITs, PMODE Gaussian
samplers, PMOG mixture samplers, and fixed-observation-time thermodynamic
neuron layers.

**The punchline:** `thermocompute` lets us study neural layers where width is
area, not latency. Under the massively parallel thermodynamic hardware model,
we can run variable-width fixed-depth neural inference in constant modeled
physical time. Increasing width adds parallel thermodynamic units; it does not
lengthen the fixed physical evolution window. The PyTorch emulator still pays
normal software runtime, but the modeled computation is the important target.

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

The package is meant for algorithm research before dedicated thermodynamic
hardware is available. It lets you prototype the computation as a physical
stochastic process, measure the emulated physical time of that process, and
compare it against conventional sequential sampling or digital neural
computation.

This is not a claim that PyTorch itself magically has constant wall-clock time.
The emulator runs on CPU or GPU and pays normal software costs. The point is to
model the process that an analog thermodynamic substrate would execute in
parallel.

The most important distinction in the package is:

```text
PyTorch wall time     = how long this emulator takes on today's CPU/GPU
modeled physical time = how long the corresponding thermodynamic substrate runs
```

The first one is an implementation cost. The second one is the research object.

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
- Make physical time explicit through `tau0`, `dt`, and `t_f`.
- Support discrete, continuous, and multimodal probabilistic primitives.
- Build thermodynamic neural layers whose nonlinear activation is produced by
  fixed-time stochastic dynamics instead of a hand-coded activation function.
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
).to(device)

tokens = torch.randn(8, 32, 64, device=device)
tokens_out, layer_info = layer(tokens, causal=True, return_info=True)
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
    neuron=ThermodynamicNeuronConfig(t_f=0.2, dt=0.04, n_replicas=2, output="mean"),
)
block = ThermodynamicTransformerBlock(block_config).to(device)
block_out, block_info = block(tokens, causal=True, return_info=True)
print(block_info.physical_time)
```

## Package Layout

```text
thermocompute/
  primitives.py     p-bits, PDIT, PMODE, PMOG, Ising energy
  neurons.py        fixed-time thermodynamic neuron layers and MLPs
  transformer.py    thermodynamic transformer attention and FFN blocks
  integration.py    production-shaped FFN/block wrappers and replacement helper
  training.py       cold/PT end-to-end and readout training helpers
  experiments.py    smoke checks, proof-of-concept checks, scaling studies
  benchmarks.py     research-proof width and baseline benchmarks
  metrics.py        result serialization and small metric helpers
  config.py         device and random seed utilities
  cuda_ext.py       optional CUDA extension loader
  cuda/             experimental custom CUDA kernel source
scripts/
  run_benchmarks.py
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
    n_replicas=4,
    tempering=True,
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
    n_replicas=4,
    tempering=True,
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
    n_replicas=4,
    tempering=True,
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
)

block = ThermodynamicTransformerBlock(config).to(device)
out, info = block(tokens, causal=True, return_info=True)

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
    n_replicas=4,
    thermo_output="mean",
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

### Parallel Tempered End-To-End Training

`fit_transformer_end_to_end_parallel_tempering` is the direct inductive
training method: no ridge solve, no closed-form readout fitting, and no frozen
reservoir assumption.

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
    n_replicas=2,
    thermo_output="mean",
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

This is the cleanest training counterpart to the inference story because it is
end-to-end and inductive. It still runs in software here, but structurally it
matches the thermodynamic idea: many parameter states explore in parallel,
temperature helps escape poor basins, and the selected model still performs
fixed-time thermodynamic inference.

### Cold End-To-End Training

`fit_transformer_end_to_end_cold` exposes the ordinary single-replica version of
end-to-end training. It is useful enough to keep as a public method because on
simple or smooth objectives it can match parallel-tempered training closely
while using much less memory and wall time.

```python
from thermocompute import ThermodynamicTransformerLayer, fit_transformer_end_to_end_cold

layer = ThermodynamicTransformerLayer(
    embed_dim=32,
    num_heads=4,
    thermo_hidden_dim=128,
    t_f=0.2,
    dt=0.04,
    n_replicas=2,
    thermo_output="mean",
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
- you need a cheap baseline
- memory is tight
- training time matters more than exploration

Use parallel-tempered end-to-end training when:

- losses are rugged or multimodal
- runs get stuck in poor basins
- extra memory for full replicas is acceptable
- a small accuracy gain is worth more training compute

In the current toy inductive transformer experiment, cold training performs
surprisingly well: it reaches nearly the same train and eval loss as the
parallel-tempered version while using one model replica instead of five. The PT
version gives a small eval improvement, but costs more wall time and memory.

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
    n_replicas=4,
    thermo_output="mean",
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
- Cold single-replica end-to-end inductive training as a cheaper baseline.
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

In short:

- Proves: fixed-depth thermodynamic neuron/transformer blocks report constant
  modeled physical time as width increases.
- Proves: the benchmark suite separates modeled physical time from PyTorch wall
  time and classical FLOP proxies.
- Does not prove: PyTorch wall time is constant with width.
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

The important scaling distinction is between digital emulator runtime and
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

This package exposes that separation directly. In the width-scaling experiment,
the thermodynamic MLP's modeled physical time stays fixed as width increases.
PyTorch wall time is still a normal GPU measurement, but the physical-time
metric shows what the same computation would ask from a massively parallel
thermodynamic substrate.

The same principle now applies to the thermodynamic transformer layer. The
`thermo_hidden_dim` can be varied from small to very wide while the modeled
feed-forward physical time remains `t_f`. That is the central result: variable
width neural inference in constant modeled physical time.

The readout-alignment experiment adds a first training counterpart: a wider
thermodynamic transformer reservoir improves one-shot fitted error while the
inference window remains fixed. Training still includes a digital solve, but it
does not require backpropagating through many stochastic inference passes.

## What This Could Mean For Huge Models

For huge models, the most interesting target is not replacing every digital
operation. The strongest near-term opportunity is replacing the stochastic and
probabilistic core: sampling, uncertainty propagation, energy-based inference,
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
sampling work into fixed-time physical dynamics. That is how this package
frames the frontier opportunity: inference of variable-width neural blocks in
constant modeled physical time.

Thermodynamic Readout Alignment is the first practical training answer in this
direction. Instead of training every stochastic element by backpropagation, the
thermodynamic block can act as a huge physical feature generator and the readout
can be solved or locally adapted. For very large models, that suggests a hybrid
training stack: digital pretraining and routing where needed, local or
closed-form readout alignment where possible, and thermodynamic inference for
the wide stochastic nonlinear core.

Parallel Tempered End-To-End Training is the more direct answer when a frozen
reservoir is not enough. It trains the whole thermodynamic transformer layer by
running multiple temperature replicas of the model itself. That gives a path
toward ordinary inductive training while retaining thermodynamic exploration as
the optimization mechanism.

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
python scripts/run_smoke.py
python scripts/run_poc.py
python scripts/run_experiments.py --outdir artifacts
python scripts/run_benchmarks.py --outdir artifacts
```

Keep claims grounded when interpreting results:

- `physical_time` is an emulator metric, not a measured chip benchmark.
- PyTorch wall time is useful for software optimization, not for proving
  thermodynamic hardware speed.
- Constant physical time applies to parallel units inside a fixed sequential
  layer schedule. Deeper unpipelined networks still add sequential fixed-time
  windows.
- The variable-width constant-time claim applies to modeled physical time under
  a parallel thermodynamic substrate. It does not mean CPU/GPU software runtime
  is independent of tensor width.
- Thermodynamic Readout Alignment is a fast readout-training method, not a
  complete replacement for all gradient-based training.
- Parallel Tempered End-To-End Training is the no-ridge inductive baseline. It
  is more expensive than readout alignment in the emulator because it trains
  full model replicas.
- Cold End-To-End Training is the cheapest no-ridge baseline. It can be close
  in loss or accuracy on easy tasks, but it has less ability to escape poor
  basins than the parallel-tempered version.
- Parallel Tempered Mask Alignment uses tempering to search sparse readout
  structure. It is currently a training-time algorithm; inference still uses a
  single fixed thermodynamic pass.
