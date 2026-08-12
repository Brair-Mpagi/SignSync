"""Trainable temporal models (plan §8.2).

The staged progression from plan §8.2, in one file so the stages can be compared on
the same data with the same training loop:

1. ``lstm``        — recurrent baseline. Start here.
2. ``tcn``         — dilated temporal convolutions.
3. ``transformer`` — self-attention encoder, matching the architecture the Makerere
                     USL work validated for USL specifically (plan §3).
4. ``multimodal``  — separate encoders per landmark stream, fused late.

Importing this module requires ``torch`` (``pip install -e ".[models]"``). Nothing
else in the package imports it at module level, so the pipeline stays runnable on
the prototype recogniser without it (plan §17).

A note on why the transformer is not the default: on the V1/V2 corpus sizes plan
§9.2 anticipates — a few thousand clips — a transformer has more capacity than the
data supports and will memorise signers. The staging is a data-size decision, not a
fashion one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..capabilities import require
from ..errors import SignSyncError

torch = require("torch", feature="trainable temporal models")

if TYPE_CHECKING:
    # `nn = torch.nn` is a runtime alias and a type checker cannot see through it,
    # so every `nn.Module` annotation below would be an undefined name.
    from torch import nn
else:
    nn = torch.nn

__all__ = [
    "ModelConfig",
    "build_model",
    "MODEL_NAMES",
    "LSTMRecogniser",
    "TCNRecogniser",
    "TransformerRecogniser",
    "MultiStreamRecogniser",
]

MODEL_NAMES = ("lstm", "gru", "tcn", "transformer", "multimodal")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters shared across model families."""

    input_dim: int
    n_classes: int
    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.3
    n_heads: int = 4
    kernel_size: int = 3
    bidirectional: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.n_classes <= 1:
            raise SignSyncError(
                f"need input_dim > 0 and n_classes > 1, got {self.input_dim}, {self.n_classes}"
            )
        if self.hidden_dim % self.n_heads:
            raise SignSyncError(
                f"hidden_dim {self.hidden_dim} must divide evenly by n_heads {self.n_heads}"
            )


class _MaskedMeanPool(nn.Module):
    """Average over time, ignoring padded frames.

    Clips are padded to the longest in a batch, and signing durations vary by a
    factor of three between signers. Pooling over the padding would make a model's
    output depend on what else happened to be in the batch.
    """

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        mask = mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class LSTMRecogniser(nn.Module):
    """Plan §8.2 stage 1: recurrent baseline over landmark sequences."""

    def __init__(self, config: ModelConfig, cell: str = "lstm") -> None:
        super().__init__()
        self.config = config
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        out_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.pool = _MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, config.n_classes),
        )

    def forward(self, x, mask=None):
        encoded, _ = self.rnn(x)
        return self.head(self.pool(encoded, mask))


class _TemporalBlock(nn.Module):
    """Dilated causal-width residual block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class TCNRecogniser(nn.Module):
    """Plan §8.2 stage 2: dilated temporal convolutions.

    Dilation doubles per layer so the receptive field covers a whole sign within a
    few layers, while keeping the constant-time-per-frame behaviour that matters for
    the streaming path.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _TemporalBlock(config.hidden_dim, config.kernel_size, 2**i, config.dropout)
                for i in range(config.n_layers)
            ]
        )
        self.pool = _MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_classes),
        )

    def forward(self, x, mask=None):
        h = self.project(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(self.pool(h.transpose(1, 2), mask))


class _PositionalEncoding(nn.Module):
    """Sinusoidal positions.

    Fixed rather than learned: with a few thousand training clips, learned position
    embeddings mostly memorise the typical length of each sign in the training set.
    """

    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * div)
        encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class TransformerRecogniser(nn.Module):
    """Plan §8.2 stage 3: self-attention encoder over landmark sequences."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.project = nn.Linear(config.input_dim, config.hidden_dim)
        self.positions = _PositionalEncoding(config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.pool = _MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_classes),
        )

    def forward(self, x, mask=None):
        h = self.positions(self.project(x))
        padding_mask = None if mask is None else ~mask.bool()
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.head(self.pool(h, mask))


class MultiStreamRecogniser(nn.Module):
    """Plan §8.2 stage 4: per-stream encoders fused late.

    Hands, body and face are encoded separately before fusion because they carry
    different linguistic parameters at different rates — handshape changes fast,
    non-manual marking spans whole clauses — and an early-fused model tends to let
    the highest-variance stream (the hands) dominate the face entirely, which is how
    a system silently loses the ability to mark questions and negation.

    ``streams`` maps a name to the column slice of that block in the feature vector;
    :class:`~signsync.vision.features.FeatureLayout` provides exactly that.
    """

    def __init__(self, config: ModelConfig, streams: dict[str, slice]) -> None:
        super().__init__()
        if not streams:
            raise SignSyncError("multimodal model needs at least one stream")
        self.config = config
        self.stream_names = list(streams)
        self.slices = [streams[name] for name in self.stream_names]

        per_stream = max(config.hidden_dim // len(streams), 16)
        self.encoders = nn.ModuleList(
            [
                nn.GRU(
                    input_size=max(s.stop - s.start, 1),
                    hidden_size=per_stream,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
                for s in self.slices
            ]
        )
        fused = per_stream * 2 * len(streams)
        self.pool = _MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Dropout(config.dropout),
            nn.Linear(fused, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.n_classes),
        )

    def forward(self, x, mask=None):
        pooled = []
        for encoder, block in zip(self.encoders, self.slices, strict=True):
            encoded, _ = encoder(x[:, :, block])
            pooled.append(self.pool(encoded, mask))
        return self.head(torch.cat(pooled, dim=-1))


def build_model(name: str, config: ModelConfig, **kwargs):
    """Construct a model by name. See :data:`MODEL_NAMES`."""
    key = name.lower()
    if key in ("lstm", "gru"):
        return LSTMRecogniser(config, cell=key)
    if key == "tcn":
        return TCNRecogniser(config)
    if key == "transformer":
        return TransformerRecogniser(config)
    if key == "multimodal":
        streams = kwargs.get("streams")
        if not streams:
            raise SignSyncError(
                "the multimodal model needs streams={name: slice}; pass a FeatureLayout's blocks"
            )
        return MultiStreamRecogniser(config, streams)
    raise SignSyncError(f"unknown model {name!r}; known models: {', '.join(MODEL_NAMES)}")
