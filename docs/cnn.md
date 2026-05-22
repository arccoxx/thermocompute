# Thermodynamic CNNs

`thermocompute` includes a minimal convolutional path so the fixed-time
thermodynamic width idea is not limited to MLPs and transformers.

## API

- `ThermodynamicConv2d`: local receptive-field thermodynamic convolution.
- `ThermodynamicCNNClassifier`: tiny classifier wrapper around one
  thermodynamic convolution, tanh pooling, and a linear classifier.
- `make_toy_cnn_data`: CPU-light vertical-vs-horizontal bar dataset.
- `fit_cnn_classifier_end_to_end`: full-batch no-ridge AdamW trainer for tiny
  coverage tests.
- `fit_cnn_readout_ridge`: fast frozen-feature readout solve from pooled
  thermodynamic hidden channels.
- `fit_cnn_classifier`: backward-compatible alias for the end-to-end trainer.

## How The Layer Works

For every image patch, the layer computes current inputs with a convolutional
weight bank, evolves a thermodynamic hidden channel fabric for a fixed time
window, and projects those hidden channels to normal output channels.

```text
[B, C, H, W]
  -> local patch currents
  -> fixed-time thermodynamic hidden channels
  -> readout projection
  -> [B, C_out, H_out, W_out]
```

The modeled physical time is inherited from the thermodynamic neuron layer:

```text
T_conv = n_steps * dt ~= t_f
```

Increasing `thermo_channels` increases parameter memory and emulator work, but
under the parallel thermodynamic substrate model it does not increase the
modeled physical observation window.

## Memory-Efficient Path

`ThermodynamicConv2d` supports `memory_efficient_chunk_size`. In software, the
hidden thermodynamic channel fabric is evaluated in chunks and the readout
contributions are accumulated. This reduces peak thermodynamic state memory
from:

```text
O(batch * out_height * out_width * thermo_channels * replicas)
```

to:

```text
O(batch * out_height * out_width * chunk_size * replicas)
```

Parameter memory remains linear in `thermo_channels` because the current bank
and readout weights must still be stored.

## Experiment

Run the CPU-light coverage experiment:

```powershell
python scripts/run_cnn_experiment.py --device cpu --train-steps 80 --outdir artifacts
```

Current checked-in result on an 8x8 vertical-vs-horizontal bar task:

| Model | Final Loss | Final Accuracy | Fit Wall ms | Modeled Physical Time |
|---|---:|---:|---:|---:|
| classical tiny CNN | 0.000015 | 1.000 | 92.0 | 0.0 |
| thermodynamic CNN | 0.043337 | 1.000 | 459.4 | 0.08 |

This is a coverage experiment, not a production computer-vision result. It
shows that the package can express and train convolutional thermodynamic
modules with PyTorch autograd, state dicts, and chunked inference.

## Fast Readout Vs End-To-End Training

Run:

```powershell
python scripts/run_readout_training_comparison.py --device cpu --flow-steps 96 --cnn-steps 80 --outdir artifacts
```

Current CNN result:

| Method | Final Loss | Final Accuracy | Fit Wall ms | Physical Time |
|---|---:|---:|---:|---:|
| readout ridge | 0.6612 | 1.000 | 5.65 | 0.08 |
| end-to-end no ridge | 0.0433 | 1.000 | 501.16 | 0.08 |

The ridge trainer solves directly from pooled thermodynamic hidden channels,
not from the random convolutional output projection. That is the CNN analogue
of thermodynamic readout alignment: keep the physical feature fabric fixed and
solve a small digital readout. End-to-end training is far slower on CPU here,
but it learns a much more confident classifier.

Implementation note: ridge fitting switches the classifier to
`readout_mode="thermo"`. To load a ridge-trained CNN state dict, instantiate
`ThermodynamicCNNClassifier(..., readout_mode="thermo")` before loading.

## Claim Boundary

What this supports:

- Thermodynamic convolution is a natural extension of the package's fixed-time
  width model.
- Local stochastic feature banks can be represented as PyTorch CNN modules.
- The no-replica chunked path works for CNN inference and tiny training.

What this does not yet prove:

- Production CNN superiority over cuDNN or specialized vision kernels.
- Constant PyTorch wall time for arbitrary image sizes or channel widths.
- State-of-the-art image accuracy.
