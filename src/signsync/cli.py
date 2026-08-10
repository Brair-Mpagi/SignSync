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
