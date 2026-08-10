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

from . import __version__
from .errors import SignSyncError

Handler = Callable[[argparse.Namespace], int]


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .capabilities import format_report

    colour = sys.stdout.isatty() and not args.no_colour
    print(format_report(colour=colour))
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
