"""Corpus schema, consent enforcement, splitting and augmentation (plan §9, §16).

    Corpus (manifest) + ConsentRegistry
              |
        CorpusLoader          <- consent and quality gates live here
              |
      signer_independent_split
              |
           augment            <- only on the training side
"""

from __future__ import annotations

from .augment import AugmentationPolicy, augment, mirror_handedness, temporal_rescale
from .consent import (
    ConsentAudit,
    ConsentRecord,
    ConsentRegistry,
    ConsentScope,
    ConsentStatus,
)
from .corpus import ClipData, CorpusLoader, QualityGate
from .schema import (
    AgeBand,
    ClipRecord,
    Corpus,
    Handedness,
    MarkerType,
    NonManualMarker,
    RecordingConditions,
    SignerProfile,
    SigningBackground,
)
from .splits import Split, signer_independent_split, signer_kfold, validate_split
from .synthetic import (
    DEMO_SENTENCES,
    DEMO_VOCABULARY,
    SyntheticCorpusSpec,
    build_synthetic_corpus,
)

__all__ = [
    "AgeBand",
    "AugmentationPolicy",
    "ClipData",
    "ClipRecord",
    "ConsentAudit",
    "ConsentRecord",
    "ConsentRegistry",
    "ConsentScope",
    "ConsentStatus",
    "Corpus",
    "CorpusLoader",
    "DEMO_SENTENCES",
    "DEMO_VOCABULARY",
    "Handedness",
    "MarkerType",
    "NonManualMarker",
    "QualityGate",
    "RecordingConditions",
    "SignerProfile",
    "SigningBackground",
    "Split",
    "SyntheticCorpusSpec",
    "augment",
    "build_synthetic_corpus",
    "mirror_handedness",
    "signer_independent_split",
    "signer_kfold",
    "temporal_rescale",
    "validate_split",
]
