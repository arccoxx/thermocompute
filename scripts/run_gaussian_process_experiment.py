from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from thermocompute import (
    ExactGaussianProcessRegressor,
    RandomFeatureGaussianProcessLayer,
    RBFKernelConfig,
    fit_gp_readout_ridge,
    gp_regression_rmse,
    make_gp_regression_data,
    set_seed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-light Gaussian-process layer experiment.")
    parser.add_argument("--outdir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--train-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=96)
    parser.add_argument("--random-features", type=int, default=128)
    args = parser.parse_args()

    if args.device == "cuda":
        raise RuntimeError("This experiment is intentionally CPU-safe while external GPU training is active.")
    device = torch.device("cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    set_seed(7070)
    generator = torch.Generator(device=device).manual_seed(7070)
    train_x, train_y = make_gp_regression_data(args.train_samples, noise=0.04, device=device, generator=generator)
    test_x, test_y = make_gp_regression_data(args.test_samples, noise=0.0, device=device, generator=generator)
    kernel = RBFKernelConfig(lengthscale=0.75, output_scale=1.0)

    exact = ExactGaussianProcessRegressor(kernel=kernel, noise=0.04**2)
    exact_fit = exact.fit(train_x, train_y)
    start = time.perf_counter()
    exact_pred = exact.predict(test_x)
    exact_predict_wall_ms = (time.perf_counter() - start) * 1000.0
    exact_rmse = gp_regression_rmse(exact_pred.mean, test_y)
    exact_samples = exact.sample_posterior(test_x[:12], n_samples=3, generator=generator)

    set_seed(7070)
    rff = RandomFeatureGaussianProcessLayer(
        1,
        1,
        n_random_features=args.random_features,
        kernel=kernel,
    )
    before_rmse = gp_regression_rmse(rff(test_x), test_y)
    rff_fit = fit_gp_readout_ridge(rff, train_x, train_y, ridge=1e-3)
    start = time.perf_counter()
    rff_pred = rff(test_x)
    rff_predict_wall_ms = (time.perf_counter() - start) * 1000.0
    rff_rmse = gp_regression_rmse(rff_pred, test_y)

    payload: dict[str, Any] = {
        "name": "gaussian_process_experiment",
        "device": str(device),
        "task": "1D nonlinear regression",
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "models": {
            "exact_gp": {
                **exact_fit.__dict__,
                "test_rmse": exact_rmse,
                "predict_wall_ms": exact_predict_wall_ms,
                "posterior_sample_shape": list(exact_samples.shape),
                "mean_variance": float(exact_pred.variance.mean().detach().cpu()),
            },
            "random_feature_gp_layer": {
                **rff_fit.__dict__,
                "initial_test_rmse": before_rmse,
                "test_rmse": rff_rmse,
                "predict_wall_ms": rff_predict_wall_ms,
                "n_random_features": args.random_features,
            },
        },
        "claim_boundary": (
            "Exact GP support is small-data reference inference. Random-feature GP support is a scalable layer-style "
            "approximation with fast ridge readout fitting. This is CPU-light emulation, not a production GP benchmark."
        ),
    }
    out = args.outdir / "gaussian_process_experiment.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
