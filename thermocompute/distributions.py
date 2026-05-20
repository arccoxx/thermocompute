from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.distributions as td


_EXCLUDED_BASE_CLASSES = {"Distribution", "ExponentialFamily"}


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _as_sample_shape(sample_shape: tuple[Any, ...]) -> torch.Size:
    if len(sample_shape) == 1 and isinstance(sample_shape[0], torch.Size):
        return sample_shape[0]
    if len(sample_shape) == 1 and isinstance(sample_shape[0], (tuple, list)):
        return torch.Size(sample_shape[0])
    return torch.Size(sample_shape)


def _distribution_classes() -> dict[str, type[td.Distribution]]:
    classes: dict[str, type[td.Distribution]] = {}
    for attr in dir(td):
        obj = getattr(td, attr)
        try:
            if (
                isinstance(obj, type)
                and issubclass(obj, td.Distribution)
                and attr not in _EXCLUDED_BASE_CLASSES
            ):
                classes[attr] = obj
        except TypeError:
            continue
    return classes


def available_distributions() -> tuple[str, ...]:
    """Return distribution class names exposed by `torch.distributions`."""

    return tuple(sorted(_distribution_classes()))


def distribution_class(name: str) -> type[td.Distribution]:
    """Resolve a distribution name against `torch.distributions`.

    Names are forgiving: `multivariate_normal`, `MultivariateNormal`, and
    `multivariate-normal` all resolve to the same class when available.
    """

    normalized = _normalize_name(name)
    for class_name, cls in _distribution_classes().items():
        aliases = {
            _normalize_name(class_name),
            _normalize_name(_camel_to_snake(class_name)),
        }
        if normalized in aliases:
            return cls
    available = ", ".join(available_distributions())
    raise ValueError(f"unknown distribution {name!r}; available distributions: {available}")


@dataclass(frozen=True)
class DistributionSpec:
    """Serializable-ish recipe for a torch distribution.

    Parameters may be tensors, scalars, other distributions, transforms, or any
    constructor argument accepted by the target `torch.distributions` class.
    """

    name: str
    params: Mapping[str, Any]
    validate_args: bool | None = None

    def build(self) -> td.Distribution:
        return make_distribution(self.name, validate_args=self.validate_args, **dict(self.params))


def make_distribution(name: str, *, validate_args: bool | None = None, **params: Any) -> td.Distribution:
    """Construct any distribution available in `torch.distributions`."""

    cls = distribution_class(name)
    kwargs = dict(params)
    if validate_args is not None:
        kwargs["validate_args"] = validate_args
    return cls(**kwargs)


def sample_distribution(
    name: str,
    *sample_shape: Any,
    validate_args: bool | None = None,
    reparameterized: bool = False,
    **params: Any,
) -> Tensor:
    """Construct and sample a torch distribution in one call."""

    dist = make_distribution(name, validate_args=validate_args, **params)
    shape = _as_sample_shape(sample_shape)
    if reparameterized and getattr(dist, "has_rsample", False):
        return dist.rsample(shape)
    return dist.sample(shape)


class DistributionSampler(nn.Module):
    """Module wrapper for named `torch.distributions` families.

    Tensor parameters are registered as buffers so `.to(device, dtype)` works
    naturally. Non-tensor parameters are kept as Python values. For arbitrary
    custom distribution objects, use `DistributionAdapter`.
    """

    def __init__(self, name: str, *, validate_args: bool | None = None, **params: Any) -> None:
        super().__init__()
        self.name = str(name)
        self.validate_args = validate_args
        self._tensor_param_keys: list[str] = []
        self._tensor_param_buffers: list[str] = []
        self._plain_params: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, Tensor):
                buffer_name = f"_param_{len(self._tensor_param_keys)}"
                self.register_buffer(buffer_name, value)
                self._tensor_param_keys.append(key)
                self._tensor_param_buffers.append(buffer_name)
            else:
                self._plain_params[key] = value

    def distribution(self) -> td.Distribution:
        params = dict(self._plain_params)
        for key, buffer_name in zip(self._tensor_param_keys, self._tensor_param_buffers):
            params[key] = getattr(self, buffer_name)
        return make_distribution(self.name, validate_args=self.validate_args, **params)

    @property
    def batch_shape(self) -> torch.Size:
        return self.distribution().batch_shape

    @property
    def event_shape(self) -> torch.Size:
        return self.distribution().event_shape

    @property
    def has_rsample(self) -> bool:
        return bool(getattr(self.distribution(), "has_rsample", False))

    @property
    def mean(self) -> Tensor:
        return self.distribution().mean

    @property
    def variance(self) -> Tensor:
        return self.distribution().variance

    def sample(
        self,
        *sample_shape: Any,
        reparameterized: bool = False,
    ) -> Tensor:
        dist = self.distribution()
        shape = _as_sample_shape(sample_shape)
        if reparameterized and getattr(dist, "has_rsample", False):
            return dist.rsample(shape)
        return dist.sample(shape)

    def rsample(self, *sample_shape: Any) -> Tensor:
        return self.distribution().rsample(_as_sample_shape(sample_shape))

    def log_prob(self, value: Tensor) -> Tensor:
        return self.distribution().log_prob(value)

    def entropy(self) -> Tensor:
        return self.distribution().entropy()

    def cdf(self, value: Tensor) -> Tensor:
        return self.distribution().cdf(value)

    def icdf(self, value: Tensor) -> Tensor:
        return self.distribution().icdf(value)


class DistributionAdapter:
    """Thin adapter for custom user-provided distribution objects."""

    def __init__(self, distribution: td.Distribution) -> None:
        if not isinstance(distribution, td.Distribution):
            raise TypeError("distribution must be a torch.distributions.Distribution")
        self.distribution = distribution

    def sample(self, *sample_shape: Any, reparameterized: bool = False) -> Tensor:
        shape = _as_sample_shape(sample_shape)
        if reparameterized and getattr(self.distribution, "has_rsample", False):
            return self.distribution.rsample(shape)
        return self.distribution.sample(shape)

    def log_prob(self, value: Tensor) -> Tensor:
        return self.distribution.log_prob(value)

    def entropy(self) -> Tensor:
        return self.distribution.entropy()
