from __future__ import annotations

import torch
from torch import nn

from thermocompute import (
    BinaryPBit,
    CategoricalPDIT,
    DistributionAdapter,
    DistributionSampler,
    DistributionSpec,
    FlowVelocityMLP,
    PhysicalTimeReport,
    PMODE,
    PMOG,
    QuantizationConfig,
    QuantizedThermodynamicFFN,
    ThermodynamicFlowVelocity,
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
    available_numeric_formats,
    estimate_quantized_thermo_ffn_memory,
    fit_flow_matching,
    fit_quantized_ffn_mse,
    flow_speedup_vs_diffusion,
    make_mog2d,
    numeric_format_bits,
    make_distribution,
    quantize_tensor,
    quantized_storage_nbytes,
    rbf_mmd2,
    sample_flow,
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


def test_quantized_numeric_formats_and_ste_gradients() -> None:
    formats = {"fp32", "fp16", "bf16", "int8", "int4", "int2", "binary"}
    formats.update(name for name in ("fp8_e4m3fn", "fp8_e5m2") if name in available_numeric_formats())
    assert formats.issubset(set(available_numeric_formats()))
    assert numeric_format_bits("int4") == 4
    assert quantized_storage_nbytes(3, "int4") == 2

    x = torch.linspace(-1.5, 1.5, 17, requires_grad=True)
    for name in sorted(formats):
        q = quantize_tensor(x, QuantizationConfig(format=name, compute_dtype=torch.float32, per_channel=False))
        assert q.shape == x.shape
        assert q.dtype == torch.float32
        assert torch.isfinite(q).all()
    q4 = quantize_tensor(x, QuantizationConfig(format="int4", compute_dtype=torch.float32))
    q4.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_quantized_thermodynamic_ffn_training_and_memory() -> None:
    torch.manual_seed(123)
    config = ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0)
    qconfig = QuantizationConfig(format="int4", compute_dtype=torch.float32, per_channel=True)
    model = QuantizedThermodynamicFFN(
        4,
        12,
        quantization=qconfig,
        neuron_config=config,
        memory_efficient_chunk_size=5,
    )
    x = torch.randn(4, 2, 4)
    target = torch.zeros_like(x)
    y, info = model(x, return_info=True)
    assert y.shape == x.shape
    assert info.physical_time == config.t_f
    assert torch.isfinite(y).all()

    result = fit_quantized_ffn_mse(model, x, target, n_steps=6, learning_rate=5e-3)
    assert result.format == "int4"
    assert result.final_loss < result.initial_loss

    estimate = estimate_quantized_thermo_ffn_memory(
        4,
        1024,
        batch_tokens=8,
        parameter_format="int4",
        state_format="fp16",
        chunk_size=64,
    )
    assert estimate.peak_bytes > 0
    assert estimate.parameter_bits == 1024 * (4 + 4 + 4) * 4


def test_flow_matching_tiny_cpu() -> None:
    torch.manual_seed(321)
    generator = torch.Generator().manual_seed(321)
    data = make_mog2d(64, generator=generator)
    model = FlowVelocityMLP(2, hidden_dim=16, time_features=4)
    result = fit_flow_matching(model, data, n_steps=20, batch_size=16, learning_rate=3e-3, generator=generator)
    assert result.final_loss < result.initial_loss

    samples = sample_flow(model, 16, n_flow_steps=2, generator=generator)
    assert samples.samples.shape == (16, 2)
    assert samples.function_evaluations == 2
    assert flow_speedup_vs_diffusion(50, 2) == 25.0
    assert torch.isfinite(rbf_mmd2(samples.samples, data[:16]))

    thermo = ThermodynamicFlowVelocity(
        2,
        embed_dim=8,
        thermo_hidden_dim=12,
        time_features=4,
        neuron_config=ThermodynamicNeuronConfig(t_f=0.04, dt=0.04, temperature=0.0),
        memory_efficient_chunk_size=6,
    )
    velocity = thermo(data[:4], torch.zeros(4, 1))
    assert velocity.shape == (4, 2)
    assert thermo.physical_time == 0.04


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
        "FlowVelocityMLP",
        "QuantizationConfig",
        "QuantizedThermodynamicFFN",
        "ThermodynamicFlowVelocity",
        "available_distributions",
        "available_numeric_formats",
        "estimate_classical_ffn_memory",
        "estimate_quantized_thermo_ffn_memory",
        "estimate_thermo_ffn_memory",
        "fit_flow_matching",
        "fit_quantized_ffn_mse",
        "flow_speedup_vs_diffusion",
        "make_mog2d",
        "make_distribution",
        "numeric_format_bits",
        "quantize_tensor",
        "quantized_storage_nbytes",
        "sample_flow",
        "sample_distribution",
        "run_superiority_demo",
        "__version__",
    ]
    for name in required:
        assert hasattr(tc, name)
