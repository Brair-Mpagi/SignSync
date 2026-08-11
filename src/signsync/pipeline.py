"""End-to-end orchestration — the three modes of plan §18.3.

    Mode A  camera ─▶ recognition ─▶ frame ─▶ English ─▶ speech
    Mode B  speech ─▶ English ─▶ frame ─▶ glosses ─▶ avatar motion
    Mode C  both, with turn-taking

Everything before this module is a component; this is where they become a system.
Three responsibilities that only exist at this level:

**Latency accounting.** Objective O11 budgets under two seconds per exchange, and
the only place that can be measured is across the whole round trip. Every stage
reports into a :class:`~signsync.perf.LatencyTracker`, so a deployment can be
checked on its own hardware rather than on a developer laptop.

**Honesty propagation.** Recognition confidence, an unvalidated lexicon,
procedurally generated motion and a silent TTS engine are all facts the person
using the system needs. Each result object carries them to the client rather than
letting them evaporate between layers (plan §16.3).

**Degrading rather than failing.** Plan §17 expects deployments with no speech
engine, no camera, or no trained model. A missing component disables its feature
and says so; it does not stop the pipeline from starting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .avatar.rig import Animation, Rig, default_rig
from .datasets.schema import MarkerType
from .errors import SignSyncError
from .motion.generator import GeneratedMotion, MotionGenerator
from .motion.library import ClipLibrary
from .perf import LatencyTracker
from .recognition.base import UNKNOWN_GLOSS, Recogniser, SignPrediction
from .recognition.segmentation import ContinuousRecogniser
from .speech.base import AudioClip, SpeechResult, SpeechToText, TextToSpeech, Transcript
from .speech.stt import NullSTT
from .speech.tts import TextOnlyTTS
from .translation.english_to_sign import EnglishToSign, GlossSequence
from .translation.lexicon import Lexicon, default_lexicon
from .translation.sign_to_english import SignToEnglish, TranslationResult
from .vision.features import FeatureConfig, encode_sequence
from .vision.normalise import normalise_sequence
from .vision.schema import LandmarkSequence

__all__ = [
    "PipelineWarning",
    "SignToSpeechResult",
    "SpeechToSignResult",
    "SignSyncPipeline",
]


@dataclass(frozen=True)
class PipelineWarning:
    """Something the user should be told about this particular output."""

    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


#: Warnings that are properties of the deployment rather than of one request.
UNVALIDATED_LEXICON = PipelineWarning(
    "unvalidated_lexicon",
    "The sign lexicon has not been reviewed by a USL linguist. Output is provisional.",
)
LOW_CONFIDENCE = PipelineWarning(
    "low_confidence", "Recognition confidence was low; the translation may be wrong."
)
UNRECOGNISED_SIGNS = PipelineWarning(
    "unrecognised_signs", "Some signs were not recognised and are missing from the translation."
)
NO_AUDIO = PipelineWarning(
    "no_audio", "No speech engine is available; show the text instead of waiting for audio."
)
GENERATED_MOTION = PipelineWarning(
    "generated_motion",
    "Some signs were rendered from generated motion, not recordings of a signer.",
)
MISSING_SIGNS = PipelineWarning(
    "missing_signs", "Some words have no sign in the motion library and were not animated."
)


@dataclass
class SignToSpeechResult:
    """Mode A output."""

    glosses: tuple[str, ...]
    predictions: tuple[SignPrediction, ...]
    translation: TranslationResult
    speech: SpeechResult
    warnings: tuple[PipelineWarning, ...] = ()

    @property
    def text(self) -> str:
        return self.translation.text

    @property
    def confidence(self) -> float:
        return self.translation.confidence

    @property
    def needs_repeat(self) -> bool:
        """Whether the signer should be asked to repeat.

        Better to ask than to speak a confident guess: the hearing participant
        cannot tell a mistranslation from a translation, but the signer can always
        sign again (plan §16.3).
        """
        return any(w.code in ("low_confidence", "unrecognised_signs") for w in self.warnings)


@dataclass
class SpeechToSignResult:
    """Mode B output."""

    transcript: Transcript
    sequence: GlossSequence
    motion: GeneratedMotion
    warnings: tuple[PipelineWarning, ...] = ()

    @property
    def animation(self) -> Animation:
        return self.motion.animation

    @property
    def glosses(self) -> tuple[str, ...]:
        return self.sequence.glosses


class SignSyncPipeline:
    """The whole system, wired together (plan §18.3)."""

    def __init__(
        self,
        *,
        recogniser: Recogniser | None = None,
        lexicon: Lexicon | None = None,
        library: ClipLibrary | None = None,
        rig: Rig | None = None,
        stt: SpeechToText | None = None,
        tts: TextToSpeech | None = None,
        feature_config: FeatureConfig | None = None,
        latency: LatencyTracker | None = None,
        low_confidence_threshold: float = 0.6,
    ) -> None:
        self.rig = rig or default_rig()
        self.lexicon = lexicon or default_lexicon()
        self.recogniser = recogniser
        self.sign_to_english = SignToEnglish(self.lexicon)
        self.english_to_sign = EnglishToSign(self.lexicon)
        self.motion = MotionGenerator(library, self.rig)
        self.stt = stt or NullSTT()
        self.tts = tts or TextOnlyTTS()
        self.feature_config = feature_config
        self.latency = latency or LatencyTracker()
        self.low_confidence_threshold = low_confidence_threshold

    # ---------------------------------------------------------------- capabilities

    def capabilities(self) -> dict[str, bool]:
        """What this deployment can actually do right now.

        The client asks on connect and hides what is unavailable, rather than
        offering a button that silently does nothing.
        """
        return {
            "recognition": self.recogniser is not None,
            "speech_input": not isinstance(self.stt, NullSTT),
            "speech_output": not isinstance(self.tts, TextOnlyTTS),
            "avatar": True,
            "validated_lexicon": self.lexicon.is_validated,
        }

    def deployment_warnings(self) -> tuple[PipelineWarning, ...]:
        """Warnings that hold for every request in this deployment."""
        warnings: list[PipelineWarning] = []
        if not self.lexicon.is_validated:
            warnings.append(UNVALIDATED_LEXICON)
        if isinstance(self.tts, TextOnlyTTS):
            warnings.append(NO_AUDIO)
        return tuple(warnings)

    # ---------------------------------------------------------------- mode A

    def recognise(self, sequence: LandmarkSequence, *, dominant: str = "right") -> list[SignPrediction]:
        """Landmarks to a gloss sequence, segmenting continuous signing (plan §8.3)."""
        if self.recogniser is None:
            raise SignSyncError(
                "no recogniser loaded; train one with `signsync train` or construct the "
                "pipeline with recogniser=..."
            )
        with self.latency.measure("vision"):
            normalised = normalise_sequence(sequence, dominant=dominant)
            features, _ = encode_sequence(normalised, self.feature_config)
        with self.latency.measure("recognition"):
            continuous = ContinuousRecogniser(self.recogniser)
            return continuous.recognise(normalised, features)

    def sign_to_speech(
        self,
        source: LandmarkSequence | list[SignPrediction] | list[str],
        *,
        markers: tuple[MarkerType, ...] = (),
        dominant: str = "right",
        speak: bool = True,
    ) -> SignToSpeechResult:
        """Mode A: signing in, spoken English out."""
        if isinstance(source, LandmarkSequence):
            predictions = self.recognise(source, dominant=dominant)
        else:
            predictions = [
                p if isinstance(p, SignPrediction) else SignPrediction(str(p).upper(), 1.0)
                for p in source
            ]

        with self.latency.measure("translation"):
            translation = self.sign_to_english.translate(list(predictions), markers=markers)

        if speak and translation.text:
            with self.latency.measure("tts"):
                speech = self.tts.synthesise(translation.text)
        else:
            speech = SpeechResult(text=translation.text, engine=self.tts.name, detail="not spoken")

        warnings = list(self.deployment_warnings())
        if translation.confidence < self.low_confidence_threshold:
            warnings.append(LOW_CONFIDENCE)
        if translation.unresolved or any(p.gloss == UNKNOWN_GLOSS for p in predictions):
            warnings.append(UNRECOGNISED_SIGNS)

        return SignToSpeechResult(
            glosses=tuple(p.gloss for p in predictions),
            predictions=tuple(predictions),
            translation=translation,
            speech=speech,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # ---------------------------------------------------------------- mode B

    def speech_to_sign(self, source: AudioClip | str) -> SpeechToSignResult:
        """Mode B: spoken English in, avatar motion out."""
        if isinstance(source, AudioClip):
            with self.latency.measure("stt"):
                transcript = self.stt.transcribe(source)
        else:
            transcript = Transcript(text=source, confidence=1.0, engine="text-input")

        with self.latency.measure("translation"):
            sequence = self.english_to_sign.translate(transcript.text)
        with self.latency.measure("motion"):
            motion = self.motion.generate(sequence)

        warnings = list(self.deployment_warnings())
        if motion.missing:
            warnings.append(MISSING_SIGNS)
        if motion.procedural:
            warnings.append(GENERATED_MOTION)
        if sequence.unresolved:
            warnings.append(
                PipelineWarning(
                    "untranslated_words",
                    f"No sign for: {', '.join(sequence.unresolved)}.",
                )
            )

        return SpeechToSignResult(
            transcript=transcript,
            sequence=sequence,
            motion=motion,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    # ---------------------------------------------------------------- mode C

    def conversation_turn(
        self,
        *,
        signing: LandmarkSequence | list[str] | None = None,
        speech: AudioClip | str | None = None,
    ) -> dict[str, object]:
        """Mode C: one exchange, in whichever direction the input arrived.

        Both inputs at once is rejected rather than arbitrated. Two people talking
        over each other is a real situation, but resolving it is an interface
        decision — who gets the floor — and making it silently here would hide that
        choice from the client that has to show it.
        """
        if signing is not None and speech is not None:
            raise SignSyncError(
                "conversation_turn takes one input at a time; the client decides who has "
                "the floor when both parties act at once"
            )
        if signing is not None:
            result = self.sign_to_speech(signing)
            return {"direction": "sign_to_speech", "result": result}
        if speech is not None:
            result_b = self.speech_to_sign(speech)
            return {"direction": "speech_to_sign", "result": result_b}
        raise SignSyncError("conversation_turn needs either signing or speech input")

    # ---------------------------------------------------------------- diagnostics

    def latency_report(self) -> dict[str, object]:
        """Per-stage latency and whether the round trip meets objective O11."""
        report = self.latency.report()
        slowest = report.slowest()
        return {
            "stages": [
                {
                    "stage": s.stage,
                    "count": s.count,
                    "mean_ms": round(s.mean_ms, 2),
                    "p50_ms": round(s.p50_ms, 2),
                    "p95_ms": round(s.p95_ms, 2),
                }
                for s in report.stages
            ],
            "total_p95_ms": round(report.total_p95_ms, 2),
            "meets_o11": report.meets(2000.0),
            "bottleneck": slowest.stage if slowest else None,
        }
