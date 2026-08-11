"""Sign recognition: isolated, continuous and streaming (plan §8.2, §8.3).

    (T, D) features ──▶ Recogniser ──▶ SignPrediction
                            ▲
      ┌─────────────────────┴─────────────────────┐
      │                                           │
  PrototypeRecogniser                     TorchRecogniser
  (NumPy, always available)          (lstm/tcn/transformer/multimodal)

``ContinuousRecogniser`` segments a continuous clip and recognises each span;
``StreamingRecogniser`` does the same for a live frame stream.

:mod:`signsync.recognition.torch_models` and :mod:`~signsync.recognition.torch_runtime`
are deliberately not imported here — they need the ``models`` extra, and everything
else in this package must work without it.
"""

from __future__ import annotations

from .base import (
    UNKNOWN_GLOSS,
    Recogniser,
    RecogniserConfig,
    SignPrediction,
    Vocabulary,
    top_k,
)
from .dataset import FeatureSet, build_feature_set, encode_clip, feature_sets_for_split
from .infer import StreamingConfig, StreamingRecogniser
from .prototype import PrototypeRecogniser
from .segmentation import (
    ContinuousRecogniser,
    Segment,
    SegmentationConfig,
    segment_motion,
)
from .train import (
    TrainingResult,
    TrainingRun,
    confusion_pairs,
    evaluate,
    train,
    train_from_corpus,
)

__all__ = [
    "ContinuousRecogniser",
    "FeatureSet",
    "PrototypeRecogniser",
    "Recogniser",
    "RecogniserConfig",
    "Segment",
    "SegmentationConfig",
    "SignPrediction",
    "StreamingConfig",
    "StreamingRecogniser",
    "TrainingResult",
    "TrainingRun",
    "UNKNOWN_GLOSS",
    "Vocabulary",
    "build_feature_set",
    "confusion_pairs",
    "encode_clip",
    "evaluate",
    "feature_sets_for_split",
    "segment_motion",
    "top_k",
    "train",
    "train_from_corpus",
]
