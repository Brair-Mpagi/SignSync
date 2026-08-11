"""``signsync`` command-line entry point.

Subcommands are declared here but their implementations are imported *inside* the
handler functions. That keeps ``signsync doctor`` fast and, more importantly, keeps
it working on a machine where the thing being diagnosed is a broken optional
dependency — an eagerly imported subcommand module would take the diagnostic tool
down with it.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from . import __version__
from .errors import SignSyncError

Handler = Callable[[argparse.Namespace], int]


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .capabilities import format_report

    colour = sys.stdout.isatty() and not args.no_colour
    print(format_report(colour=colour))
    return 0


def _cmd_corpus_build(args: argparse.Namespace) -> int:
    from .datasets.synthetic import SyntheticCorpusSpec, build_synthetic_corpus

    corpus, registry = build_synthetic_corpus(
        args.root, SyntheticCorpusSpec(n_signers=args.signers, repeats_per_gloss=args.repeats)
    )
    print(f"wrote {len(corpus)} clips from {len(corpus.signers)} signers to {corpus.root}")
    print(f"vocabulary: {len(corpus.vocabulary())} glosses")
    print(f"consent records: {len(registry)}")
    print("\nThis corpus is synthetic and contains no USL. See docs/limitations.md.")
    return 0


def _cmd_corpus_stats(args: argparse.Namespace) -> int:
    from datetime import date

    from .datasets.consent import ConsentRegistry, ConsentScope
    from .datasets.corpus import CorpusLoader
    from .datasets.schema import Corpus

    corpus = Corpus.load(args.root)
    print(f"corpus  : {corpus.name} ({len(corpus)} clips, {len(corpus.signers)} signers)")
    print(f"vocab   : {len(corpus.vocabulary())} glosses")
    print(f"isolated: {len(corpus.isolated())}   continuous: {len(corpus.continuous())}")

    consent_path = Path(args.root) / "consent.json"
    if consent_path.exists():
        registry = ConsentRegistry.load(consent_path)
        loader = CorpusLoader(corpus, registry, scope=ConsentScope.parse(args.scope))
        print(f"consent : {loader.audit().summary()}")
        expiring = registry.expiring_before(date.today().replace(year=date.today().year + 1))
        if expiring:
            print(f"          retention lapses within a year: {', '.join(expiring)}")
    else:
        print(f"consent : no consent.json at {consent_path} — no clip here is usable")

    warnings = corpus.diversity_warnings()
    if warnings:
        print("\ndiversity warnings:")
        for warning in warnings:
            print(f"  ! {warning}")
    else:
        print("\ndiversity: no dimension over its concentration limit")
    return 0


def _cmd_corpus_split(args: argparse.Namespace) -> int:
    import json

    from .datasets.consent import ConsentRegistry, ConsentScope
    from .datasets.corpus import CorpusLoader
    from .datasets.schema import Corpus
    from .datasets.splits import signer_independent_split

    corpus = Corpus.load(args.root)

    # Split over the clips consent actually permits. A split built from the full
    # manifest would place withdrawn or expired signers into train/test, and the
    # loader would then drop them mid-run — leaving a split whose reported sizes
    # never matched the data the model saw.
    records = None
    consent_path = Path(args.root) / "consent.json"
    if consent_path.exists():
        loader = CorpusLoader(corpus, ConsentRegistry.load(consent_path), scope=ConsentScope.TRAINING)
        records = loader.permitted_clips()
        excluded = len(corpus) - len(records)
        if excluded:
            print(f"excluding {excluded} clip(s) without training consent\n")
    split = signer_independent_split(
        corpus, ratios=(args.train, args.val, args.test), seed=args.seed, records=records
    )
    print(split.summary())
    for name, signers in split.signers.items():
        print(f"  {name:<6} signers: {', '.join(signers) or '-'}")

    if args.out:
        payload = {
            "seed": split.seed,
            "signers": {k: list(v) for k, v in split.signers.items()},
            **{name: list(split[name]) for name in ("train", "val", "test")},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .datasets.consent import ConsentRegistry
    from .datasets.corpus import CorpusLoader
    from .datasets.schema import Corpus
    from .recognition.base import RecogniserConfig
    from .recognition.train import TrainingRun, confusion_pairs, train_from_corpus

    corpus = Corpus.load(args.corpus)
    consent_path = Path(args.corpus) / "consent.json"
    if not consent_path.exists():
        print(
            f"error: no consent.json at {consent_path}. Training requires consent records; "
            "see docs/data-protection.md.",
            file=sys.stderr,
        )
        return 2

    loader = CorpusLoader(corpus, ConsentRegistry.load(consent_path))
    print(f"consent: {loader.audit().summary()}\n")

    result = train_from_corpus(
        loader,
        run=TrainingRun(
            backend=args.backend,
            recogniser=RecogniserConfig(
                n_frames=args.frames, min_confidence=args.min_confidence
            ),
            augmentations=args.augmentations,
            seed=args.seed,
        ),
        save_to=args.out,
    )
    print(result.summary())

    if args.out:
        print(f"\nsaved model to {args.out}")
    print(
        "\nThese numbers describe this corpus only. On synthetic data they say nothing "
        "about USL recognition — see docs/limitations.md."
    )
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    from .translation import EnglishToSign, SignToEnglish, default_lexicon

    lexicon = default_lexicon()
    if not lexicon.is_validated:
        print(f"warning: {lexicon.warning}\n", file=sys.stderr)

    if args.direction == "sign-to-english":
        result = SignToEnglish(lexicon).translate([g.upper() for g in args.input])
        print(result.text)
        if args.trace:
            print(f"\nframe     : {result.frame.describe()}")
            print(f"confidence: {result.confidence:.0%}")
        if result.unresolved:
            print(f"\nnot translated: {', '.join(result.unresolved)}", file=sys.stderr)
    else:
        sequence = EnglishToSign(lexicon).translate(" ".join(args.input))
        print(sequence.notation())
        if args.trace:
            print(f"\nframe   : {sequence.frame.describe()}")
            print(f"markers : {[m.marker.value for m in sequence.markers]}")
        if sequence.unresolved:
            print(f"\nno sign for: {', '.join(sequence.unresolved)}", file=sys.stderr)
    return 0


def _build_pipeline(model_path: str | None):  # type: ignore[no-untyped-def]
    """Assemble a pipeline, loading a recogniser if one was given."""
    from .pipeline import SignSyncPipeline
    from .speech.stt import best_available_stt
    from .speech.tts import best_available_tts

    recogniser = None
    if model_path:
        from .recognition.prototype import PrototypeRecogniser

        recogniser = PrototypeRecogniser.load(model_path)

    return SignSyncPipeline(
        recogniser=recogniser,
        stt=best_available_stt(),  # type: ignore[arg-type]
        tts=best_available_tts(),  # type: ignore[arg-type]
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    from .api import create_app

    uvicorn = __import__("uvicorn")
    pipeline = _build_pipeline(args.model)

    print(f"capabilities: {pipeline.capabilities()}")
    for warning in pipeline.deployment_warnings():
        print(f"  ! {warning}")
    print(f"\nserving on http://{args.host}:{args.port}")

    uvicorn.run(create_app(pipeline), host=args.host, port=args.port, log_level=args.log_level)
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """Run one exchange through the whole pipeline, on the terminal."""
    from .vision.synthetic import synthetic_sentence

    pipeline = _build_pipeline(args.model)

    if args.mode == "speech-to-sign":
        result = pipeline.speech_to_sign(args.text)
        print(f'heard   : "{result.transcript.text}"')
        print(f"glosses : {' '.join(result.glosses) or '(none)'}")
        print(f"\n{result.sequence.notation()}\n")
        print(
            f"motion  : {len(result.animation)} frames, "
            f"{result.animation.duration:.1f}s, {len(result.motion.missing)} sign(s) missing"
        )
    else:
        glosses = args.glosses or ["ME", "NEED", "HELP"]
        if args.synthetic and pipeline.recogniser is not None:
            sequence = synthetic_sentence(glosses, "demo-signer")
            result_a = pipeline.sign_to_speech(sequence)
            print(f"signed  : {' '.join(glosses)}  (synthetic signer)")
            print(f"detected: {' '.join(result_a.glosses) or '(nothing)'}")
        else:
            result_a = pipeline.sign_to_speech(glosses)
            print(f"glosses : {' '.join(glosses)}")
        print(f'english : "{result_a.text}"')
        print(f"spoken  : {result_a.speech.engine} (audible={result_a.speech.is_audible})")
        print(f"confidence: {result_a.confidence:.0%}")

    warnings = getattr(result if args.mode == "speech-to-sign" else result_a, "warnings", ())
    if warnings:
        print()
        for warning in warnings:
            print(f"  ! {warning}")
    print(f"\nlatency: {pipeline.latency_report()['total_p95_ms']:.0f} ms (p95, this run)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signsync",
        description="Bidirectional Ugandan Sign Language <-> spoken English translation.",
    )
    parser.add_argument("--version", action="version", version=f"signsync {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    doctor = sub.add_parser(
        "doctor",
        help="report which optional capabilities are installed and what is disabled without them",
    )
    doctor.add_argument("--no-colour", action="store_true", help="disable ANSI colour output")
    doctor.set_defaults(handler=_cmd_doctor)

    corpus = sub.add_parser("corpus", help="inspect and prepare corpora")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", metavar="<subcommand>")

    build = corpus_sub.add_parser(
        "build-synthetic", help="generate a synthetic corpus for development and CI"
    )
    build.add_argument("root", help="directory to write the corpus into")
    build.add_argument("--signers", type=int, default=6)
    build.add_argument("--repeats", type=int, default=2, help="clips per gloss per signer")
    build.set_defaults(handler=_cmd_corpus_build)

    stats = corpus_sub.add_parser(
        "stats", help="corpus size, consent status and diversity warnings"
    )
    stats.add_argument("root", help="corpus directory or manifest.json")
    stats.add_argument("--scope", default="training", help="consent scope to audit against")
    stats.set_defaults(handler=_cmd_corpus_stats)

    split = corpus_sub.add_parser("split", help="produce a signer-independent split")
    split.add_argument("root", help="corpus directory or manifest.json")
    split.add_argument("--train", type=float, default=0.7)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)
    split.add_argument("--seed", type=int, default=0)
    split.add_argument("--out", help="write the split to this JSON file")
    split.set_defaults(handler=_cmd_corpus_split)

    train = sub.add_parser(
        "train", help="train a sign recogniser and report held-out-signer accuracy"
    )
    train.add_argument("corpus", help="corpus directory containing manifest.json + consent.json")
    train.add_argument(
        "--backend",
        default="prototype",
        help="prototype (NumPy, no extras) or a torch model: lstm, gru, tcn, transformer",
    )
    train.add_argument("--augmentations", type=int, default=2, help="augmented copies per clip")
    train.add_argument("--frames", type=int, default=32, help="frames each clip is resampled to")
    train.add_argument("--min-confidence", type=float, default=0.45, dest="min_confidence")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--out", help="write the trained model here")
    train.set_defaults(handler=_cmd_train)

    translate = sub.add_parser("translate", help="translate between USL glosses and English")
    translate.add_argument(
        "direction",
        choices=("sign-to-english", "english-to-sign"),
        help="which way to translate",
    )
    translate.add_argument("input", nargs="+", help="glosses, or an English sentence")
    translate.add_argument("--trace", action="store_true", help="show the semantic frame")
    translate.set_defaults(handler=_cmd_translate)

    serve = sub.add_parser("serve", help="run the API and browser client")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--model", help="path to a trained recogniser (.npz)")
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=_cmd_serve)

    demo = sub.add_parser("demo", help="run one exchange through the whole pipeline")
    demo.add_argument("mode", choices=("sign-to-speech", "speech-to-sign"))
    demo.add_argument("--text", default="Where is the hospital?", help="input for speech-to-sign")
    demo.add_argument("--glosses", nargs="*", help="input for sign-to-speech")
    demo.add_argument("--model", help="path to a trained recogniser (.npz)")
    demo.add_argument(
        "--synthetic",
        action="store_true",
        help="generate synthetic signing and recognise it, exercising the vision path",
    )
    demo.set_defaults(handler=_cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler: Handler | None = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except SignSyncError as exc:
        # Errors we raise on purpose carry an actionable message; a traceback would
        # bury it. Anything else is a bug and should keep its traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
