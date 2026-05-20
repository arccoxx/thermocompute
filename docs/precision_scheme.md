# Low-Precision Thermodynamic Numerics

`thermocompute` supports low-precision thermodynamic inference through a
portable quantization-aware emulation scheme. The goal is not to claim that
PyTorch natively trains true int4 tensors. The goal is to expose the numerical
regimes that thermodynamic hardware and GPU quantization stacks care about,
while keeping training differentiable enough for research experiments.

## Supported Formats

The public precision API supports:

- `fp64`, `fp32`
- `tf32` style mantissa-limited emulation
- `fp16`, `bf16`
- `fp8_e4m3fn`, `fp8_e5m2` when the installed PyTorch build exposes them
- `int8`, `int4`, `int2`
- `binary`

Use:

```python
from thermocompute import available_numeric_formats

print(available_numeric_formats())
```

## Method

For floating formats, tensors are cast to the target storage dtype and then
dequantized to the configured compute dtype. This gives a portable approximation
of storage/rounding effects without requiring all downstream kernels to support
that dtype directly.

For low-bit integer formats, tensors use symmetric fake quantization:

```text
s = max(abs(x)) / qmax
q = round(clamp(x / s, -qmax, qmax))
x_q = q * s
```

For `int8`, `qmax = 127`; for `int4`, `qmax = 7`; for `int2`, `qmax = 1`.

For binary:

```text
s = mean(abs(x))
x_q = s * sign(x)
```

Training uses a straight-through estimator:

```text
forward:  x_q
backward: dL/dx_q is passed through as dL/dx
```

This keeps master parameters trainable in ordinary floating point while the
forward computation experiences the low-precision numerical bottleneck.

## Thermodynamic FFN Support

`QuantizedThermodynamicFFN` applies the precision scheme to:

- input currents
- thermodynamic input weights
- optional biases and thermodynamic coefficients
- evolved thermodynamic state/readout activations
- readout weights

Example:

```python
import torch
from thermocompute import (
    QuantizationConfig,
    QuantizedThermodynamicFFN,
    ThermodynamicNeuronConfig,
)

model = QuantizedThermodynamicFFN(
    256,
    4096,
    quantization=QuantizationConfig(format="int4", per_channel=True),
    neuron_config=ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0),
    memory_efficient_chunk_size=256,
)

x = torch.randn(2, 16, 256)
y = model(x)
```

Training support:

```python
from thermocompute import fit_quantized_ffn_mse

result = fit_quantized_ffn_mse(model, x, torch.zeros_like(x), n_steps=32)
print(result.initial_loss, result.final_loss)
```

## Memory Scaling

Low-precision storage changes the parameter-memory coefficient, not the model
shape. For thermodynamic FFN input dimension `d`, output dimension `o`, width
`W`, batch tokens `B`, and chunk size `C`:

```text
parameter bits ~= b_p W(d + o + 4)
state bits     ~= b_s B C(replicas + state_overhead)
```

The important chunked no-replica case remains:

```text
peak state memory: O(BW) -> O(BC)
```

Low precision multiplies this by the selected bit width. For example, int4
parameters use one half the parameter storage of int8 and one quarter the
storage of fp16.

## Claim Boundary

This is quantization-aware emulation. It is useful for research, memory
estimation, and training-through-low-precision experiments. It is not a claim
that every backend has native int4 training kernels, nor that low-bit training
will beat classical training on all tasks.

## Training Comparison

Use:

```powershell
python scripts/run_precision_training_comparison.py --device cpu --steps 20 --repeats 3 --outdir artifacts
```

This compares:

- standard fp32 training;
- autocast mixed-precision fp16/bf16 training with fp32 master parameters;
- quantization-aware fp16/bf16/fp8/int8/int4/int2/binary thermodynamic FFN
  training with straight-through gradients.

The current artifact is
`artifacts/precision_training_comparison.json`. In the checked-in tiny CPU
experiment, fp16, bf16, fp8, int8, and int4 stayed within roughly 1% of fp32
final eval loss. Int2 was materially worse, and binary still reduced loss but
had weaker training improvement. That is a useful feasibility signal, not a
general benchmark claim.
