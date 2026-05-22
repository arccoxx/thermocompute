# Flow Matching

`thermocompute` includes a lightweight conditional flow-matching implementation
because flow models are a natural fit for probabilistic thermodynamic
computing.

The training setup uses a straight path from Gaussian base samples `x0` to data
samples `x1`:

```text
x_t = (1 - t)x0 + t x1
u_t = x1 - x0
```

A velocity network learns:

```text
v_theta(x_t, t) ~= u_t
```

Sampling starts from Gaussian noise and integrates:

```text
dx/dt = v_theta(x, t)
```

with a small number of Euler steps.

## Why This Matters

Long diffusion samplers often use tens or hundreds of denoising network
evaluations. Flow matching can use far fewer velocity evaluations. The simple
speed proxy is:

```text
speedup ~= diffusion_steps / flow_steps
```

That advantage is independent of the thermodynamic substrate. The
thermodynamic version adds a second possible speed path: each velocity
evaluation can use a fixed-time thermodynamic FFN whose modeled physical time
does not grow with width under the parallel substrate model.

## Public API

- `FlowVelocityMLP`: standard time-conditioned MLP velocity field.
- `ThermodynamicFlowVelocity`: time-conditioned velocity field with a
  thermodynamic FFN core.
- `fit_flow_matching_end_to_end`: CPU-safe no-ridge AdamW trainer.
- `fit_flow_matching_readout_ridge`: fast frozen-feature velocity readout
  solve for `ThermodynamicFlowVelocity`.
- `fit_flow_matching`: backward-compatible alias for the end-to-end trainer.
- `sample_flow`: Euler sampler for the learned probability-flow ODE.
- `make_mog2d`: tiny eight-mode 2D mixture generator for smoke experiments.
- `rbf_mmd2`: lightweight distribution-distance metric.

## Experiment

Run:

```powershell
python scripts/run_flow_matching_experiment.py --device cpu --train-steps 96 --sample-count 128 --outdir artifacts
```

Current checked-in result on the tiny 2D mixture:

| Model | Best Flow Steps | MMD² To Reference | Wall ms | Speedup vs 50-Step Diffusion |
|---|---:|---:|---:|---:|
| classical flow MLP | 8 | 0.051954 | 0.550 | 6.25x |
| thermodynamic flow | 1 | 0.062004 | 0.542 | 50.0x |

Interpretation: the one-step thermodynamic flow is slightly less accurate than
the best 8-step classical flow in this small CPU run, but it reaches a
comparable distribution score with one velocity evaluation. That is the
research reason to keep exploring this path.

## Fast Readout Vs End-To-End Training

Run:

```powershell
python scripts/run_readout_training_comparison.py --device cpu --flow-steps 96 --cnn-steps 80 --outdir artifacts
```

Current flow result:

| Method | Final Loss | One-Step MMD2 | Fit Wall ms | Physical Time |
|---|---:|---:|---:|---:|
| readout ridge | 2.6620 | 0.0522 | 2.51 | 0.08 |
| end-to-end no ridge | 2.6482 | 0.0550 | 325.81 | 0.08 |

The ridge path is much faster because it freezes the thermodynamic feature
fabric and solves only the velocity readout. In this tiny run, both methods
reach nearly the same fixed-pair supervised loss, while ridge is over 100x
faster than 96 CPU AdamW steps.

## Claim Boundary

This is a toy feasibility experiment. It does not prove production-scale image,
audio, or video diffusion superiority. It proves that the package has a
reproducible flow-matching path and that flow-step reduction can be measured
cleanly next to thermodynamic modeled physical time.
