from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RBFKernelConfig:
    """Configuration for an isotropic squared-exponential/RBF kernel."""

    lengthscale: float = 1.0
    output_scale: float = 1.0


@dataclass(frozen=True)
class GaussianProcessFitResult:
    """Fit summary for an exact Gaussian-process regressor."""

    method: str
    n_train: int
    input_dim: int
    target_dim: int
    negative_log_marginal_likelihood: float
    fit_wall_ms: float


@dataclass(frozen=True)
class GaussianProcessPrediction:
    """Posterior prediction from an exact Gaussian-process regressor."""

    mean: Tensor
    covariance: Tensor
    variance: Tensor


@dataclass(frozen=True)
class GPReadoutFitResult:
    """Closed-form readout fit summary for a random-feature GP layer."""

    method: str
    ridge: float
    initial_mse: float
    final_mse: float
    fit_wall_ms: float
    n_examples: int
    feature_dim: int
    target_dim: int


def rbf_kernel(x: Tensor, y: Tensor, *, lengthscale: float = 1.0, output_scale: float = 1.0) -> Tensor:
    """Compute an RBF kernel matrix between two batches of vectors."""

    if x.ndim != 2 or y.ndim != 2 or x.shape[-1] != y.shape[-1]:
        raise ValueError("x and y must have shape [n, dim] and [m, dim] with matching dim")
    if lengthscale <= 0.0 or output_scale <= 0.0:
        raise ValueError("lengthscale and output_scale must be positive")
    scaled_x = x / float(lengthscale)
    scaled_y = y / float(lengthscale)
    distances = torch.cdist(scaled_x, scaled_y).square()
    return float(output_scale) * torch.exp(-0.5 * distances)


class ExactGaussianProcessRegressor(nn.Module):
    """Small-data exact GP regression with an RBF kernel.

    This is the reference probabilistic layer: after `fit`, `forward` returns
    posterior means at query points and `predict` also exposes posterior
    covariance/variance. It is intentionally exact and CPU-friendly for small
    experiments; use `RandomFeatureGaussianProcessLayer` for wider layer-style
    approximation.
    """

    def __init__(
        self,
        *,
        kernel: RBFKernelConfig | None = None,
        noise: float = 1e-3,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        if noise < 0.0 or jitter <= 0.0:
            raise ValueError("noise must be non-negative and jitter must be positive")
        self.kernel = kernel or RBFKernelConfig()
        self.noise = float(noise)
        self.jitter = float(jitter)
        self.register_buffer("train_x", torch.empty(0, 0))
        self.register_buffer("train_y", torch.empty(0, 0))
        self.register_buffer("cholesky", torch.empty(0, 0))
        self.register_buffer("alpha", torch.empty(0, 0))

    @property
    def is_fit(self) -> bool:
        return self.train_x.numel() > 0 and self.cholesky.numel() > 0 and self.alpha.numel() > 0

    @property
    def physical_time(self) -> float:
        return 0.0

    def fit(self, train_x: Tensor, train_y: Tensor) -> GaussianProcessFitResult:
        """Fit exact GP posterior state from training inputs and targets."""

        if train_x.ndim != 2:
            raise ValueError("train_x must have shape [n_train, input_dim]")
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        if train_y.ndim != 2 or train_y.shape[0] != train_x.shape[0]:
            raise ValueError("train_y must have shape [n_train] or [n_train, target_dim]")
        start = time.perf_counter()
        kernel = self._kernel(train_x, train_x)
        eye = torch.eye(train_x.shape[0], device=train_x.device, dtype=train_x.dtype)
        cov = kernel + (self.noise + self.jitter) * eye
        chol = _stable_cholesky(cov, self.jitter)
        alpha = torch.cholesky_solve(train_y, chol)
        nll = _negative_log_marginal_likelihood(train_y, chol, alpha)
        self.train_x = train_x.detach().clone()
        self.train_y = train_y.detach().clone()
        self.cholesky = chol.detach().clone()
        self.alpha = alpha.detach().clone()
        return GaussianProcessFitResult(
            method="exact_gp_regression",
            n_train=int(train_x.shape[0]),
            input_dim=int(train_x.shape[1]),
            target_dim=int(train_y.shape[1]),
            negative_log_marginal_likelihood=float(nll.detach().cpu()),
            fit_wall_ms=float((time.perf_counter() - start) * 1000.0),
        )

    def predict(self, x: Tensor, *, return_cov: bool = True) -> GaussianProcessPrediction:
        """Return posterior mean and uncertainty at query points."""

        if not self.is_fit:
            raise RuntimeError("ExactGaussianProcessRegressor must be fit before prediction")
        if x.ndim != 2 or x.shape[-1] != self.train_x.shape[-1]:
            raise ValueError("x must have shape [n_query, input_dim]")
        k_x_train = self._kernel(x, self.train_x)
        mean = k_x_train @ self.alpha
        if return_cov:
            v = torch.linalg.solve_triangular(self.cholesky, k_x_train.T, upper=False)
            cov = self._kernel(x, x) - v.T @ v
            cov = 0.5 * (cov + cov.T)
            variance = cov.diagonal().clamp_min(0.0)
        else:
            train_v = torch.linalg.solve_triangular(self.cholesky, k_x_train.T, upper=False)
            prior_diag = torch.full((x.shape[0],), self.kernel.output_scale, device=x.device, dtype=x.dtype)
            variance = (prior_diag - train_v.square().sum(dim=0)).clamp_min(0.0)
            cov = torch.diag(variance)
        return GaussianProcessPrediction(mean=mean, covariance=cov, variance=variance)

    def sample_posterior(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw posterior function samples with shape `[n_samples, n_query, target_dim]`."""

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        pred = self.predict(x, return_cov=True)
        cov = pred.covariance + self.jitter * torch.eye(x.shape[0], device=x.device, dtype=x.dtype)
        chol = _stable_cholesky(cov, self.jitter)
        eps = torch.randn(
            n_samples,
            x.shape[0],
            pred.mean.shape[-1],
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        return pred.mean.unsqueeze(0) + torch.einsum("ij,sjo->sio", chol, eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.predict(x, return_cov=False).mean

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in ("train_x", "train_y", "cholesky", "alpha"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _kernel(self, x: Tensor, y: Tensor) -> Tensor:
        return rbf_kernel(
            x,
            y,
            lengthscale=self.kernel.lengthscale,
            output_scale=self.kernel.output_scale,
        )


class RandomFeatureGaussianProcessLayer(nn.Module):
    """Scalable random Fourier feature approximation to an RBF GP layer.

    The layer maps `[*, in_features] -> [*, out_features]`. Its frozen cosine
    feature bank approximates an RBF GP prior, and its readout can be trained
    with `fit_gp_readout_ridge` or ordinary gradient descent.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        *,
        n_random_features: int = 128,
        kernel: RBFKernelConfig | None = None,
        trainable_features: bool = False,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or n_random_features <= 0:
            raise ValueError("in_features, out_features, and n_random_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_random_features = int(n_random_features)
        self.kernel = kernel or RBFKernelConfig()
        omega = torch.randn(self.n_random_features, self.in_features) / float(self.kernel.lengthscale)
        phase = 2.0 * math.pi * torch.rand(self.n_random_features)
        if trainable_features:
            self.omega = nn.Parameter(omega)
            self.phase = nn.Parameter(phase)
        else:
            self.register_buffer("omega", omega)
            self.register_buffer("phase", phase)
        self.readout = nn.Linear(self.n_random_features, self.out_features)
        nn.init.normal_(self.readout.weight, mean=0.0, std=1.0 / math.sqrt(self.n_random_features))
        nn.init.zeros_(self.readout.bias)

    @property
    def physical_time(self) -> float:
        return 0.0

    def features(self, x: Tensor) -> Tensor:
        original_shape = x.shape[:-1]
        if x.shape[-1] != self.in_features:
            raise ValueError("input last dimension must match in_features")
        flat = x.reshape(-1, self.in_features)
        projection = flat @ self.omega.T + self.phase
        scale = math.sqrt(2.0 * float(self.kernel.output_scale) / float(self.n_random_features))
        features = scale * torch.cos(projection)
        return features.reshape(*original_shape, self.n_random_features)

    def forward(self, x: Tensor, *, return_features: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        features = self.features(x)
        out = self.readout(features)
        if return_features:
            return out, features
        return out

    def sample_prior(
        self,
        x: Tensor,
        *,
        n_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample approximate GP prior functions at x.

        Returns `[n_samples, *x.shape[:-1], out_features]`.
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        features = self.features(x).reshape(-1, self.n_random_features)
        weights = torch.randn(
            n_samples,
            self.n_random_features,
            self.out_features,
            device=features.device,
            dtype=features.dtype,
            generator=generator,
        )
        samples = torch.einsum("nf,sfo->sno", features, weights)
        return samples.reshape(n_samples, *x.shape[:-1], self.out_features)


@torch.no_grad()
def fit_gp_readout_ridge(
    layer: RandomFeatureGaussianProcessLayer,
    inputs: Tensor,
    targets: Tensor,
    *,
    ridge: float = 1e-3,
) -> GPReadoutFitResult:
    """Fit the readout of a random-feature GP layer with one ridge solve."""

    if not isinstance(layer, RandomFeatureGaussianProcessLayer):
        raise TypeError("fit_gp_readout_ridge requires RandomFeatureGaussianProcessLayer")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if targets.ndim == inputs.ndim - 1:
        targets = targets.unsqueeze(-1)
    if targets.shape[:-1] != inputs.shape[:-1] or targets.shape[-1] != layer.out_features:
        raise ValueError("targets must have shape [*inputs.shape[:-1], out_features]")
    was_training = layer.training
    layer.eval()
    initial_mse = float(F.mse_loss(layer(inputs), targets).detach().cpu())
    start = time.perf_counter()
    features = layer.features(inputs).reshape(-1, layer.n_random_features)
    y = targets.reshape(-1, layer.out_features)
    solution = _ridge_solve(_with_bias(features), y, ridge)
    layer.readout.weight.copy_(solution[:-1].T.contiguous())
    layer.readout.bias.copy_(solution[-1].contiguous())
    final_mse = float(F.mse_loss(layer(inputs), targets).detach().cpu())
    if was_training:
        layer.train()
    return GPReadoutFitResult(
        method="gp_random_feature_readout_ridge",
        ridge=float(ridge),
        initial_mse=initial_mse,
        final_mse=final_mse,
        fit_wall_ms=float((time.perf_counter() - start) * 1000.0),
        n_examples=int(features.shape[0]),
        feature_dim=int(layer.n_random_features),
        target_dim=int(layer.out_features),
    )


def make_gp_regression_data(
    n_samples: int,
    *,
    noise: float = 0.05,
    x_min: float = -3.0,
    x_max: float = 3.0,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Tiny one-dimensional nonlinear regression task for GP checks."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if x_max <= x_min:
        raise ValueError("x_max must exceed x_min")
    device = device or torch.device("cpu")
    x = torch.linspace(x_min, x_max, n_samples, device=device).unsqueeze(-1)
    clean = torch.sin(1.7 * x) + 0.25 * torch.cos(4.3 * x)
    y = clean + noise * torch.randn(clean.shape, device=device, generator=generator)
    return x, y


def gp_regression_rmse(pred: Tensor, target: Tensor) -> float:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have matching shapes")
    return float(torch.sqrt(torch.mean((pred - target).square())).detach().cpu())


def _negative_log_marginal_likelihood(y: Tensor, chol: Tensor, alpha: Tensor) -> Tensor:
    n = y.shape[0]
    target_dim = y.shape[1]
    data_fit = 0.5 * torch.sum(y * alpha)
    logdet = torch.sum(torch.log(torch.diagonal(chol))) * target_dim
    constant = 0.5 * n * target_dim * math.log(2.0 * math.pi)
    return data_fit + logdet + constant


def _stable_cholesky(matrix: Tensor, jitter: float) -> Tensor:
    eye = torch.eye(matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
    current_jitter = float(jitter)
    for _ in range(5):
        chol, info = torch.linalg.cholesky_ex(matrix + current_jitter * eye)
        if int(info.detach().cpu()) == 0:
            return chol
        current_jitter *= 10.0
    return torch.linalg.cholesky(matrix + current_jitter * eye)


def _with_bias(features: Tensor) -> Tensor:
    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1)


def _ridge_solve(design: Tensor, targets: Tensor, ridge: float) -> Tensor:
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = design.T @ targets
    return torch.linalg.solve(gram + reg, rhs)
