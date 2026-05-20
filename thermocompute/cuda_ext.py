from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import torch


@lru_cache(maxsize=1)
def load_cuda_extension(verbose: bool = False) -> Any | None:
    """Try to compile and load the optional CUDA extension.

    The pure PyTorch kernels are the supported default. This function returns
    `None` when CUDA compilation is unavailable.
    """

    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
    except Exception:
        return None

    source = Path(__file__).parent / "cuda" / "thermo_kernels.cu"
    if not source.exists():
        return None
    try:
        return load(
            name="thermocompute_cuda",
            sources=[str(source)],
            verbose=verbose,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception:
        return None


def has_cuda_extension() -> bool:
    return load_cuda_extension(verbose=False) is not None
