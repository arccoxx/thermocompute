from __future__ import annotations

import torch
from torch import nn

from thermocompute import (
    BinaryPBit,
    CategoricalPDIT,
    DistributionAdapter,
    DistributionSampler,
    DistributionSpec,
    PhysicalTimeReport,
    PMODE,
    PMOG,
    ThermodynamicFFN,
    ThermodynamicMLP,
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerLayer,
    ThermodynamicTransformerBlock,
    ThermodynamicTransformerConfig,
    fit_transformer_end_to_end_cold,
    fit_transformer_end_to_end_parallel_tempering,
    fit_transformer_readout_parallel_tempering,
    fit_transformer_readout_ridge,
    estimate_classical_ffn_memory,
    estimate_thermo_ffn_memory,
    available_distributions,
    make_distribution,
    sample_distribution,
    replace_ffn,
    run_superiority_demo,
)
from thermocompute.experiments import smoke_checks


def test_primitives_shapes_cpu() -> None:
    pbit = BinaryPBit(beta=1.5)
    p = pbit.probabilities(torch.zeros(4))
    assert p.shape == (4,)
    assert torch.allclose(p, torch.full((4,), 0.5))

    pdit = CategoricalPDIT()
    cat = pdit.sample(torch.zeros(3, 5))
    assert cat.shape == (3,)

    pmode = PMODE()
    samples = pmode.sample(torch.zeros(2), 0.5, n_samples=4, t_total=4e-7)
    assert samples.shape == (4, 2)

    pmog = PMOG(2)
    mix, modes = pmog.sample(torch.zeros(2), torch.tensor([-1.0, 1.0]), torch.ones(2) * 0.1, n_samples=5)
    assert mix.shape == (5,)
    assert modes.shape == (5,)


def test_thermodynamic_mlp_info() -> None:
    model = ThermodynamicMLP([2, 4, 1], t_f=0.1, dt=0.05, n_replicas=2, tempering=True)
    y, info = model(torch.randn(3, 2), return_info=True)
    assert y.shape == (3, 1)
    assert info.physical_time == 0.2
    assert info.n_replicas == 2


def test_thermodynamic_transformer_layer_width_constant_physical_time() -> None:
    x = torch.randn(2, 5, 16)
    narrow = ThermodynamicTransformerLayer(16, 4, thermo_hidden_dim=32, t_f=0.1, dt=0.05)
    wide = ThermodynamicTransformerLayer(16, 4, thermo_hidden_dim=128, t_f=0.1, dt=0.05)
    y_narrow, info_narrow = narrow(x, return_info=True)
    y_wide, info_wide = wide(x, return_info=True)
    assert y_narrow.shape == x.shape
    assert y_wide.shape == x.shape
    assert info_narrow.physical_time == info_wide.physical_time == 0.1


def test_config_constructors_and_physical_report() -> None:
    neuron_config = ThermodynamicNeuronConfig(t_f=0.1, dt=0.05, n_replicas=2, output="mean", j4=2.0)
    layer = neuron_config.build_layer(3, 7)
    y, info = layer(torch.randn(4, 3), return_info=True)
    report = PhysicalTimeReport.from_module(layer)
    assert y.shape == (4, 7)
    assert info.physical_time == 0.1
    assert report.physical_time == 0.1
    assert layer.j4.detach()[0].item() == 2.0

    transformer_config = ThermodynamicTransformerConfig(8, 2, 16, neuron=neuron_config)
    transformer = transformer_config.build_layer()
    out, tinfo = transformer(torch.randn(2, 3, 8), return_info=True)
    assert out.shape == (2, 3, 8)
    assert tinfo.physical_time == 0.1
    assert transformer.thermo_ff.j4.detach()[0].item() == 2.0


def test_integration_ffn_block_and_replace() -> None:
    config = ThermodynamicTransformerConfig(
        embed_dim=8,
        num_heads=2,
        thermo_hidden_dim=16,
        neuron=ThermodynamicNeuronConfig(t_f=0.1, dt=0.05),
    )
    ffn = ThermodynamicFFN(8, 16, neuron_config=config.neuron)
    y, report = ffn(torch.randn(2, 3, 8), return_info=True)
    assert y.shape == (2, 3, 8)
    assert report.physical_time == 0.1

    block = ThermodynamicTransformerBlock(config)
    out, info = block(torch.randn(2, 3, 8), return_info=True)
    block_report = PhysicalTimeReport.from_module(block)
    assert out.shape == (2, 3, 8)
    assert info.feedforward_physical_time == 0.1
    assert block_report.n_steps == 2

    model = nn.Module()
    model.ffn = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8)).to(dtype=torch.float64)
    model.eval()
    count = replace_ffn(model, lambda name, module: name == "ffn", config)
    assert count == 1
    assert isinstance(model.ffn, ThermodynamicFFN)
    assert next(model.ffn.parameters()).dtype == torch.float64
    assert not model.ffn.training


def test_memory_efficient_ffn_matches_full_deterministic() -> None:
    config = ThermodynamicNeuronConfig(t_f=0.1, dt=0.05, temperature=0.0)
    full = ThermodynamicFFN(8, 20, neuron_config=config)
    chunked = ThermodynamicFFN(8, 20, neuron_config=config, memory_efficient_chunk_size=6)
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(2, 3, 8)
    y_full, info_full = full(x, return_info=True)
    y_chunked, info_chunked = chunked(x, return_info=True)
    assert torch.allclose(y_full, y_chunked, atol=1e-6)
    assert info_full.physical_time == info_chunked.physical_time


def test_memory_efficient_transformer_layer_matches_full_deterministic() -> None:
    full = ThermodynamicTransformerLayer(
        8,
        2,
        thermo_hidden_dim=20,
        t_f=0.1,
        dt=0.05,
        temperature=0.0,
    )
    chunked = ThermodynamicTransformerLayer(
        8,
        2,
        thermo_hidden_dim=20,
        t_f=0.1,
        dt=0.05,
        temperature=0.0,
        memory_efficient_chunk_size=6,
    )
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(2, 3, 8)
    y_full, info_full = full(x, return_info=True)
    y_chunked, info_chunked = chunked(x, return_info=True)
    assert torch.allclose(y_full, y_chunked, atol=1e-6)
    assert info_full.physical_time == info_chunked.physical_time


def test_no_replica_memory_efficient_cold_training() -> None:
    torch.manual_seed(17)
    x = torch.randn(6, 3, 6)
    target = torch.zeros_like(x)
    layer = ThermodynamicTransformerLayer(
        6,
        2,
        thermo_hidden_dim=14,
        t_f=0.1,
        dt=0.05,
        temperature=0.0,
        memory_efficient_chunk_size=5,
    )
    assert layer.thermo_ff.n_replicas == 1
    assert not layer.thermo_ff.tempering
    result = fit_transformer_end_to_end_cold(
        layer,
        x,
        target,
        n_steps=6,
        learning_rate=5e-3,
    )
    assert result.memory_replicas == 1
    assert result.final_train_loss < result.initial_train_loss


def test_memory_estimators_show_chunked_state_reduction() -> None:
    classical = estimate_classical_ffn_memory(32, 1024, batch_tokens=128, dtype_bytes=2)
    thermo_full = estimate_thermo_ffn_memory(32, 1024, batch_tokens=128, dtype_bytes=2, replicas=1)
    thermo_chunked = estimate_thermo_ffn_memory(
        32,
        1024,
        batch_tokens=128,
        dtype_bytes=2,
        replicas=1,
        chunk_size=128,
    )
    assert classical.parameter_bytes < thermo_full.parameter_bytes
    assert thermo_chunked.parameter_bytes == thermo_full.parameter_bytes
    assert thermo_chunked.state_bytes < thermo_full.state_bytes
    assert thermo_chunked.peak_bytes < thermo_full.peak_bytes


def test_generic_distribution_support() -> None:
    assert "Normal" in available_distributions()

    normal = DistributionSampler("normal", loc=torch.zeros(2), scale=torch.ones(2)).to(dtype=torch.float64)
    samples = normal.sample((5,))
    log_prob = normal.log_prob(samples)
    assert samples.shape == (5, 2)
    assert samples.dtype == torch.float64
    assert log_prob.shape == (5, 2)
    assert torch.isfinite(log_prob).all()

    categorical = DistributionSampler("categorical", logits=torch.zeros(3, 4))
    cat_samples = categorical.sample(7)
    assert cat_samples.shape == (7, 3)

    beta = make_distribution("beta", concentration1=torch.ones(3), concentration0=torch.ones(3) * 2)
    beta_samples = beta.sample((16,))
    assert beta_samples.shape == (16, 3)
    assert torch.all((0.0 <= beta_samples) & (beta_samples <= 1.0))

    direct = sample_distribution("normal", 4, loc=torch.zeros(2), scale=torch.ones(2))
    assert direct.shape == (4, 2)

    poisson_spec = DistributionSpec("poisson", {"rate": torch.ones(3) * 2.0})
    poisson = poisson_spec.build()
    assert poisson.sample((4,)).shape == (4, 3)

    adapter = DistributionAdapter(torch.distributions.Uniform(torch.zeros(2), torch.ones(2)))
    adapted = adapter.sample(6)
    assert adapted.shape == (6, 2)
    assert torch.isfinite(adapter.log_prob(adapted)).all()


def test_integration_block_reports_tempering_swaps() -> None:
    config = ThermodynamicTransformerConfig(
        embed_dim=8,
        num_heads=2,
        thermo_hidden_dim=16,
        neuron=ThermodynamicNeuronConfig(
            t_f=0.12,
            dt=0.04,
            n_replicas=2,
            tempering=True,
            swap_interval=1,
            output="mean",
        ),
    )
    block = ThermodynamicTransformerBlock(config)
    out, info = block(torch.randn(2, 3, 8), return_info=True)
    assert out.shape == (2, 3, 8)
    assert info.used_tempering
    assert info.swap_attempts > 0


def test_state_dict_roundtrip() -> None:
    neuron = ThermodynamicNeuronConfig(t_f=0.1, dt=0.05).build_layer(3, 5)
    neuron_copy = ThermodynamicNeuronConfig(t_f=0.1, dt=0.05).build_layer(3, 5)
    neuron_copy.load_state_dict(neuron.state_dict())
    for key, value in neuron.state_dict().items():
        assert torch.allclose(value, neuron_copy.state_dict()[key])

    transformer = ThermodynamicTransformerLayer(8, 2, thermo_hidden_dim=16, t_f=0.1, dt=0.05)
    transformer_copy = ThermodynamicTransformerLayer(8, 2, thermo_hidden_dim=16, t_f=0.1, dt=0.05)
    transformer_copy.load_state_dict(transformer.state_dict())
    for key, value in transformer.state_dict().items():
        assert torch.allclose(value, transformer_copy.state_dict()[key])


def test_thermodynamic_transformer_pdit_attention() -> None:
    x = torch.randn(2, 4, 12)
    layer = ThermodynamicTransformerLayer(
        12,
        3,
        thermo_hidden_dim=24,
        attention_mode="pdit",
        n_attention_samples=3,
        attention_t_f=0.05,
        t_f=0.1,
        dt=0.05,
    )
    y, info = layer(x, causal=True, return_info=True)
    assert y.shape == x.shape
    assert info.attention_mode == "pdit"
    assert info.attention_samples == 3
    assert info.physical_time == 0.15000000000000002


def test_transformer_readout_ridge_reduces_error() -> None:
    torch.manual_seed(3)
    x = torch.randn(8, 3, 8)
    target = torch.sin(x)
    layer = ThermodynamicTransformerLayer(8, 2, thermo_hidden_dim=32, t_f=0.1, dt=0.05)
    with torch.no_grad():
        before = torch.mean((layer(x) - target).square()).item()
    result = fit_transformer_readout_ridge(layer, x, target, ridge=1e-2, feature_repeats=2)
    assert result.train_mse < before


def test_transformer_parallel_tempering_mask_training() -> None:
    torch.manual_seed(5)
    x = torch.randn(10, 3, 8)
    target = torch.sin(x + 0.2 * torch.roll(x, shifts=1, dims=1))
    layer = ThermodynamicTransformerLayer(
        8,
        2,
        thermo_hidden_dim=24,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
    )
    with torch.no_grad():
        before = torch.mean((layer(x) - target).square()).item()
    result = fit_transformer_readout_parallel_tempering(
        layer,
        x,
        target,
        ridge=1e-2,
        keep_fraction=0.5,
        feature_repeats=1,
        n_tempering_replicas=4,
        n_tempering_steps=6,
        flips_per_step=2,
    )
    assert result.train_mse < before
    assert 0 < result.selected_features < result.feature_dim
    assert result.swap_attempts > 0


def test_transformer_end_to_end_parallel_tempering_training() -> None:
    torch.manual_seed(9)
    x = torch.randn(8, 3, 6)
    target = torch.zeros_like(x)
    layer = ThermodynamicTransformerLayer(
        6,
        2,
        thermo_hidden_dim=12,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
    )
    with torch.no_grad():
        layer.thermo_ff.temperatures.fill_(0.01)
    result = fit_transformer_end_to_end_parallel_tempering(
        layer,
        x,
        target,
        n_tempering_replicas=3,
        n_tempering_steps=8,
        learning_rate=5e-3,
        noise_scale=0.0,
        swap_interval=2,
    )
    assert result.final_train_loss < result.initial_train_loss
    assert result.swap_attempts > 0


def test_transformer_cold_end_to_end_training() -> None:
    torch.manual_seed(10)
    x = torch.randn(8, 3, 6)
    target = torch.zeros_like(x)
    layer = ThermodynamicTransformerLayer(
        6,
        2,
        thermo_hidden_dim=12,
        t_f=0.1,
        dt=0.05,
        n_replicas=2,
        thermo_output="mean",
    )
    with torch.no_grad():
        layer.thermo_ff.temperatures.fill_(0.01)
    result = fit_transformer_end_to_end_cold(
        layer,
        x,
        target,
        n_steps=8,
        learning_rate=5e-3,
    )
    assert result.final_train_loss < result.initial_train_loss
    assert result.memory_replicas == 1


def test_smoke_checks_runs() -> None:
    result = smoke_checks()
    assert result.name == "smoke_checks"
    assert "layer_output_shape" in result.metrics


def test_public_imports() -> None:
    import thermocompute as tc

    required = [
        "BinaryPBit",
        "PMODE",
        "PMOG",
        "ThermodynamicNeuronLayer",
        "ThermodynamicTransformerLayer",
        "ThermodynamicFFN",
        "ThermodynamicTransformerBlock",
        "fit_transformer_end_to_end_cold",
        "fit_transformer_end_to_end_parallel_tempering",
        "DistributionSampler",
        "available_distributions",
        "estimate_classical_ffn_memory",
        "estimate_thermo_ffn_memory",
        "make_distribution",
        "sample_distribution",
        "run_superiority_demo",
        "__version__",
    ]
    for name in required:
        assert hasattr(tc, name)
