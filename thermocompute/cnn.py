from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ThermodynamicNeuronConfig
from .neurons import ThermodynamicRunInfo
from .transformer import _aggregate_chunk_infos


@dataclass(frozen=True)
class CNNFitResult:
    """Small supervised CNN training summary."""

    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    n_steps: int
    fit_wall_ms: float


@dataclass(frozen=True)
class CNNReadoutFitResult:
    """Closed-form classifier readout fitting summary for thermodynamic CNNs."""

    method: str
    ridge: float
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    fit_wall_ms: float
    n_examples: int
    feature_dim: int
    n_classes: int
    physical_time: float


class ThermodynamicConv2d(nn.Module):
    """2D convolution with a thermodynamic hidden channel fabric.

    For every spatial receptive field, the layer computes a bank of input
    currents, evolves those currents through fixed-time thermodynamic neurons,
    then reads the hidden thermodynamic channels into output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        thermo_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        neuron_config: ThermodynamicNeuronConfig | None = None,
        memory_efficient_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or thermo_channels <= 0:
            raise ValueError("in_channels, out_channels, and thermo_channels must be positive")
        if memory_efficient_chunk_size is not None and memory_efficient_chunk_size <= 0:
            raise ValueError("memory_efficient_chunk_size must be positive when provided")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.thermo_channels = int(thermo_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.memory_efficient_chunk_size = (
            int(memory_efficient_chunk_size) if memory_efficient_chunk_size is not None else None
        )

        patch_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.neuron_config = neuron_config or ThermodynamicNeuronConfig()
        self.thermo = self.neuron_config.build_layer(patch_dim, self.thermo_channels)
        self.out_proj = nn.Linear(self.thermo_channels, self.out_channels)

    @property
    def physical_time(self) -> float:
        return self.thermo.physical_time

    def forward(
        self,
        x: Tensor,
        *,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, channels, height, width]")
        if x.shape[1] != self.in_channels:
            raise ValueError("input channel dimension must match in_channels")
        effective_chunk = chunk_size if chunk_size is not None else self.memory_efficient_chunk_size
        if effective_chunk is not None and effective_chunk < self.thermo_channels:
            y, info = self._forward_chunked(x, int(effective_chunk), generator=generator)
        else:
            current = self._conv_current(x, 0, self.thermo_channels)
            hidden, info = self._simulate_current_image(current, 0, self.thermo_channels, generator=generator)
            y = self._readout_image(hidden, x.shape[0], current.shape[2], current.shape[3])
        if return_info:
            return y, info
        return y

    def _forward_chunked(
        self,
        x: Tensor,
        chunk_size: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, ThermodynamicRunInfo]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        flat_out: Tensor | None = None
        height = width = 0
        infos: list[ThermodynamicRunInfo] = []
        for start in range(0, self.thermo_channels, chunk_size):
            end = min(start + chunk_size, self.thermo_channels)
            current = self._conv_current(x, start, end)
            height, width = int(current.shape[2]), int(current.shape[3])
            hidden, info = self._simulate_current_image(current, start, end, generator=generator)
            chunk_out = F.linear(torch.tanh(hidden), self.out_proj.weight[:, start:end], None)
            flat_out = chunk_out if flat_out is None else flat_out + chunk_out
            infos.append(info)
        if flat_out is None:
            raise RuntimeError("no CNN chunks were evaluated")
        flat_out = flat_out + self.out_proj.bias
        y = flat_out.view(x.shape[0], height, width, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return y, _aggregate_chunk_infos(infos)

    def _conv_current(self, x: Tensor, start: int, end: int) -> Tensor:
        weight = self.thermo.weight[start:end].view(
            end - start,
            self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        bias = self.thermo.bias[start:end]
        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def _simulate_current_image(
        self,
        current: Tensor,
        start: int,
        end: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, ThermodynamicRunInfo]:
        flat_current = current.permute(0, 2, 3, 1).reshape(-1, end - start)
        hidden, info = self.thermo._simulate_current(
            flat_current,
            self.thermo.j2[start:end],
            self.thermo.j3[start:end],
            self.thermo.j4[start:end],
            generator=generator,
        )
        return hidden, info

    def _readout_image(self, hidden: Tensor, batch: int, height: int, width: int) -> Tensor:
        flat = self.out_proj(torch.tanh(hidden))
        return flat.view(batch, height, width, self.out_channels).permute(0, 3, 1, 2).contiguous()


class ThermodynamicCNNClassifier(nn.Module):
    """Tiny image classifier built around one thermodynamic convolution."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        *,
        conv_channels: int = 8,
        thermo_channels: int = 24,
        kernel_size: int = 3,
        neuron_config: ThermodynamicNeuronConfig | None = None,
        memory_efficient_chunk_size: int | None = None,
        readout_mode: Literal["conv", "thermo"] = "conv",
    ) -> None:
        super().__init__()
        if n_classes <= 0:
            raise ValueError("n_classes must be positive")
        if readout_mode not in {"conv", "thermo"}:
            raise ValueError("readout_mode must be 'conv' or 'thermo'")
        self.conv = ThermodynamicConv2d(
            in_channels,
            conv_channels,
            thermo_channels,
            kernel_size,
            padding=kernel_size // 2,
            neuron_config=neuron_config or ThermodynamicNeuronConfig(t_f=0.08, dt=0.04, temperature=0.0),
            memory_efficient_chunk_size=memory_efficient_chunk_size,
        )
        self.readout_mode: Literal["conv", "thermo"] = readout_mode
        classifier_in = self.conv.thermo_channels if readout_mode == "thermo" else conv_channels
        self.classifier = nn.Linear(classifier_in, n_classes)

    @property
    def physical_time(self) -> float:
        return self.conv.physical_time

    def forward(self, x: Tensor, *, return_info: bool = False) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        if self.readout_mode == "thermo":
            pooled, info = self.thermodynamic_readout_features(x, return_info=True)
        else:
            pooled, info = self.readout_features(x, return_info=True)
        logits = self.classifier(pooled)
        if return_info:
            return logits, info
        return logits

    def readout_features(self, x: Tensor, *, return_info: bool = False) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        """Return pooled thermodynamic CNN features before the classifier."""

        features, info = self.conv(x, return_info=True)
        pooled = F.adaptive_avg_pool2d(torch.tanh(features), output_size=1).flatten(1)
        if return_info:
            return pooled, info
        return pooled

    def thermodynamic_readout_features(
        self,
        x: Tensor,
        *,
        generator: torch.Generator | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, ThermodynamicRunInfo]:
        """Return pooled hidden thermodynamic channels before convolutional readout."""

        effective_chunk = self.conv.memory_efficient_chunk_size
        if effective_chunk is not None and effective_chunk < self.conv.thermo_channels:
            chunks: list[Tensor] = []
            infos: list[ThermodynamicRunInfo] = []
            height = width = 0
            for start in range(0, self.conv.thermo_channels, effective_chunk):
                end = min(start + effective_chunk, self.conv.thermo_channels)
                current = self.conv._conv_current(x, start, end)
                height, width = int(current.shape[2]), int(current.shape[3])
                hidden, info = self.conv._simulate_current_image(current, start, end, generator=generator)
                pooled = torch.tanh(hidden).view(x.shape[0], height, width, end - start).mean(dim=(1, 2))
                chunks.append(pooled)
                infos.append(info)
            features = torch.cat(chunks, dim=-1)
            info = _aggregate_chunk_infos(infos)
        else:
            current = self.conv._conv_current(x, 0, self.conv.thermo_channels)
            height, width = int(current.shape[2]), int(current.shape[3])
            hidden, info = self.conv._simulate_current_image(current, 0, self.conv.thermo_channels, generator=generator)
            features = torch.tanh(hidden).view(x.shape[0], height, width, self.conv.thermo_channels).mean(dim=(1, 2))
        if return_info:
            return features, info
        return features


def make_toy_cnn_data(
    n_samples: int,
    *,
    image_size: int = 8,
    noise: float = 0.08,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Create a tiny vertical-vs-horizontal bar image classification dataset."""

    if n_samples <= 0 or image_size < 4:
        raise ValueError("n_samples must be positive and image_size must be at least 4")
    device = device or torch.device("cpu")
    labels = torch.randint(2, (n_samples,), device=device, generator=generator)
    images = torch.zeros(n_samples, 1, image_size, image_size, device=device)
    center = image_size // 2
    for idx in range(n_samples):
        if int(labels[idx]) == 0:
            images[idx, 0, :, center - 1 : center + 1] = 1.0
        else:
            images[idx, 0, center - 1 : center + 1, :] = 1.0
    images = images + noise * torch.randn(images.shape, device=device, generator=generator)
    return images.clamp(0.0, 1.0), labels


def fit_cnn_classifier(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    *,
    eval_x: Tensor | None = None,
    eval_y: Tensor | None = None,
    n_steps: int = 64,
    learning_rate: float = 3e-3,
) -> CNNFitResult:
    """Train a tiny CNN classifier with full-batch AdamW."""

    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    eval_x = train_x if eval_x is None else eval_x
    eval_y = train_y if eval_y is None else eval_y
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(_as_logits(model(eval_x)), eval_y).detach().cpu())
        initial_accuracy = accuracy(_as_logits(model(eval_x)), eval_y)
    start = time.perf_counter()
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = _as_logits(model(train_x))
        loss = F.cross_entropy(logits, train_y)
        loss.backward()
        optimizer.step()
    fit_wall_ms = (time.perf_counter() - start) * 1000.0
    model.eval()
    with torch.no_grad():
        logits = _as_logits(model(eval_x))
        final_loss = float(F.cross_entropy(logits, eval_y).detach().cpu())
        final_accuracy = accuracy(logits, eval_y)
    return CNNFitResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        n_steps=int(n_steps),
        fit_wall_ms=float(fit_wall_ms),
    )


def fit_cnn_classifier_end_to_end(
    model: nn.Module,
    train_x: Tensor,
    train_y: Tensor,
    *,
    eval_x: Tensor | None = None,
    eval_y: Tensor | None = None,
    n_steps: int = 64,
    learning_rate: float = 3e-3,
) -> CNNFitResult:
    """Alias for no-ridge inductive CNN training."""

    return fit_cnn_classifier(
        model,
        train_x,
        train_y,
        eval_x=eval_x,
        eval_y=eval_y,
        n_steps=n_steps,
        learning_rate=learning_rate,
    )


@torch.no_grad()
def fit_cnn_readout_ridge(
    model: ThermodynamicCNNClassifier,
    train_x: Tensor,
    train_y: Tensor,
    *,
    eval_x: Tensor | None = None,
    eval_y: Tensor | None = None,
    ridge: float = 1e-3,
) -> CNNReadoutFitResult:
    """Fit only the CNN classifier readout with one ridge solve.

    The thermodynamic convolution is treated as a frozen local feature fabric.
    Labels are solved as one-hot regression targets, then evaluated as logits
    with the usual cross-entropy and accuracy metrics.
    """

    if not isinstance(model, ThermodynamicCNNClassifier):
        raise TypeError("fit_cnn_readout_ridge requires ThermodynamicCNNClassifier")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    eval_x = train_x if eval_x is None else eval_x
    eval_y = train_y if eval_y is None else eval_y
    was_training = model.training
    model.eval()
    feature_dim = model.conv.thermo_channels
    if model.classifier.in_features != feature_dim:
        dtype = next(model.parameters()).dtype
        model.classifier = nn.Linear(feature_dim, model.classifier.out_features).to(device=train_x.device, dtype=dtype)
    model.readout_mode = "thermo"
    initial_logits = model(eval_x)
    initial_loss = float(F.cross_entropy(_as_logits(initial_logits), eval_y).detach().cpu())
    initial_accuracy = accuracy(_as_logits(initial_logits), eval_y)

    start = time.perf_counter()
    features = _as_features(model.thermodynamic_readout_features(train_x))
    targets = F.one_hot(train_y, num_classes=model.classifier.out_features).to(device=features.device, dtype=features.dtype)
    solution = _ridge_solve(_with_bias(features), targets, ridge)
    model.classifier.weight.copy_(solution[:-1].T.contiguous())
    model.classifier.bias.copy_(solution[-1].contiguous())
    fit_wall_ms = (time.perf_counter() - start) * 1000.0

    final_logits = model(eval_x)
    final_loss = float(F.cross_entropy(_as_logits(final_logits), eval_y).detach().cpu())
    final_accuracy = accuracy(_as_logits(final_logits), eval_y)
    if was_training:
        model.train()
    return CNNReadoutFitResult(
        method="cnn_readout_ridge",
        ridge=float(ridge),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_accuracy=initial_accuracy,
        final_accuracy=final_accuracy,
        fit_wall_ms=float(fit_wall_ms),
        n_examples=int(features.shape[0]),
        feature_dim=int(features.shape[-1]),
        n_classes=int(model.classifier.out_features),
        physical_time=float(model.physical_time),
    )


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().detach().cpu())


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("tuple values must have length 2")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _as_logits(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _as_features(output: Any) -> Tensor:
    return output[0] if isinstance(output, tuple) else output


def _with_bias(features: Tensor) -> Tensor:
    ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1)


def _ridge_solve(design: Tensor, targets: Tensor, ridge: float) -> Tensor:
    gram = design.T @ design
    reg = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype) * float(ridge)
    reg[-1, -1] = 0.0
    rhs = design.T @ targets
    return torch.linalg.solve(gram + reg, rhs)
