from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thermocompute import (
    ThermodynamicNeuronConfig,
    ThermodynamicTransformerBlock,
    ThermodynamicTransformerConfig,
)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ThermodynamicTransformerConfig(
        embed_dim=64,
        num_heads=4,
        thermo_hidden_dim=512,
        neuron=ThermodynamicNeuronConfig(t_f=0.2, dt=0.04, n_replicas=2, output="mean"),
    )
    block = ThermodynamicTransformerBlock(config).to(device)
    tokens = torch.randn(2, 16, 64, device=device)
    out, info = block(tokens, causal=True, return_info=True)
    print(out.shape)
    print(info.physical_time)


if __name__ == "__main__":
    main()
