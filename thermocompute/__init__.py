"""Thermodynamic probabilistic computing emulator."""

__version__ = "0.1.0"

from .config import (
    DeviceConfig,
    PhysicalTimeReport,
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerConfig,
    set_seed,
)
from .distributions import (
    DistributionAdapter,
    DistributionSampler,
    DistributionSpec,
    available_distributions,
    distribution_class,
    make_distribution,
    sample_distribution,
)
from .integration import ThermodynamicFFN, ThermodynamicTransformerBlock, replace_ffn
from .metrics import ExperimentResult, moment_summary, physical_advantage_ratio
from .memory import (
    FFNMemoryEstimate,
    estimate_classical_ffn_memory,
    estimate_thermo_ffn_memory,
)
from .neurons import (
    ThermodynamicMLP,
    ThermodynamicNeuronLayer,
    ThermodynamicRunInfo,
    quartic_potential,
)
from .transformer import (
    ThermodynamicSelfAttention,
    ThermodynamicTransformerInfo,
    ThermodynamicTransformerLayer,
)
from .training import (
    ColdEndToEndFitResult,
    ParallelTemperedEndToEndFitResult,
    ParallelTemperedMaskFitResult,
    ReadoutFitResult,
    fit_transformer_end_to_end_cold,
    fit_transformer_end_to_end_parallel_tempering,
    fit_transformer_readout_parallel_tempering,
    fit_transformer_readout_ridge,
)
from .superiority_demo import run_superiority_demo
from .primitives import (
    BinaryPBit,
    CategoricalPDIT,
    IsingEnergy,
    PMODE,
    PMOG,
)

__all__ = [
    "BinaryPBit",
    "CategoricalPDIT",
    "DeviceConfig",
    "DistributionAdapter",
    "DistributionSampler",
    "DistributionSpec",
    "ExperimentResult",
    "FFNMemoryEstimate",
    "IsingEnergy",
    "PhysicalTimeReport",
    "PMODE",
    "PMOG",
    "ThermodynamicMLP",
    "ThermodynamicNeuronLayer",
    "ThermodynamicNeuronConfig",
    "ThermodynamicRunInfo",
    "ThermodynamicSelfAttention",
    "ThermodynamicFFN",
    "ThermodynamicTransformerBlock",
    "ThermodynamicTransformerConfig",
    "ThermodynamicTransformerInfo",
    "ThermodynamicTransformerLayer",
    "ColdEndToEndFitResult",
    "ParallelTemperedEndToEndFitResult",
    "ParallelTemperedMaskFitResult",
    "ReadoutFitResult",
    "fit_transformer_end_to_end_cold",
    "fit_transformer_end_to_end_parallel_tempering",
    "fit_transformer_readout_parallel_tempering",
    "fit_transformer_readout_ridge",
    "available_distributions",
    "distribution_class",
    "estimate_classical_ffn_memory",
    "estimate_thermo_ffn_memory",
    "make_distribution",
    "moment_summary",
    "physical_advantage_ratio",
    "quartic_potential",
    "replace_ffn",
    "run_superiority_demo",
    "sample_distribution",
    "set_seed",
]
