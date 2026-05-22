# Gaussian Processes

`thermocompute` now includes Gaussian-process support as a probabilistic layer
family. The goal is twofold:

- provide an exact small-data GP reference for posterior inference
- provide a scalable random-feature GP layer that fits naturally beside the
  package's readout-alignment story

## API

- `RBFKernelConfig`: lengthscale/output-scale settings for RBF kernels.
- `rbf_kernel`: standalone RBF kernel matrix helper.
- `ExactGaussianProcessRegressor`: exact Cholesky GP regression for small data.
- `GaussianProcessPrediction`: posterior mean/covariance/variance container.
- `RandomFeatureGaussianProcessLayer`: random Fourier feature approximation to
  an RBF GP, usable as a PyTorch layer.
- `fit_gp_readout_ridge`: one-shot ridge solve for the random-feature readout.
- `make_gp_regression_data`: tiny 1D regression task for checks.
- `gp_regression_rmse`: regression metric helper.

## Exact GP

The exact GP uses:

```text
k(x, x') = sigma_f^2 exp(-||x - x'||^2 / (2 l^2))
K_y = K(X, X) + sigma_n^2 I
mu_* = K(X_*, X) K_y^{-1} y
Sigma_* = K(X_*, X_*) - K(X_*, X) K_y^{-1} K(X, X_*)
```

It is the reference implementation for small data where exact posterior
covariance matters.

```python
from thermocompute import ExactGaussianProcessRegressor, RBFKernelConfig

gp = ExactGaussianProcessRegressor(
    kernel=RBFKernelConfig(lengthscale=0.75, output_scale=1.0),
    noise=0.0016,
)
fit = gp.fit(train_x, train_y)
prediction = gp.predict(test_x)
samples = gp.sample_posterior(test_x, n_samples=4)
```

## Random-Feature GP Layer

The scalable layer approximates the RBF kernel with random Fourier features:

```text
phi(x) = sqrt(2 sigma_f^2 / M) cos(Omega x + b)
f(x) = phi(x) W + c
```

`Omega` and `b` are sampled once, then the readout can be trained with a single
ridge solve:

```python
from thermocompute import RandomFeatureGaussianProcessLayer, fit_gp_readout_ridge

layer = RandomFeatureGaussianProcessLayer(
    in_features=1,
    out_features=1,
    n_random_features=128,
)
fit = fit_gp_readout_ridge(layer, train_x, train_y, ridge=1e-3)
pred = layer(test_x)
```

This is the GP analogue of readout alignment: a broad stochastic feature bank
is fixed, and the small digital readout is solved directly.

## Experiment

Run:

```powershell
python scripts/run_gaussian_process_experiment.py --device cpu --train-samples 32 --test-samples 96 --random-features 128 --outdir artifacts
```

Current checked-in result:

| Model | Test RMSE | Fit Wall ms | Predict Wall ms | Notes |
|---|---:|---:|---:|---|
| exact GP | 0.0248 | 11.67 | 1.23 | exact posterior covariance and samples |
| random-feature GP layer | 0.0536 | 0.61 | 0.05 | fast ridge readout, scalable layer form |

## Claim Boundary

This is CPU-light GP emulation. Exact GP support is intentionally small-data
because Cholesky inference is cubic in training set size. The random-feature GP
layer is the scalable package path, but it is an approximation and should be
benchmarked against task-specific baselines before making production claims.
