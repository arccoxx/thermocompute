from __future__ import annotations

from thermocompute.config import DeviceConfig
from thermocompute.experiments import proof_of_concept_checks


def test_poc_thresholds_cpu() -> None:
    result = proof_of_concept_checks(DeviceConfig.auto(prefer_cuda=False))
    metrics = result.metrics
    assert metrics["pbit_probability_mae"] < 0.04
    assert metrics["pmode_mean_abs_error"] < 0.08
    assert metrics["pmode_std_abs_error"] < 0.08
    assert metrics["pmog_weight_l1"] < 0.12
    assert metrics["thermo_activation_input_corr"] > 0.85
