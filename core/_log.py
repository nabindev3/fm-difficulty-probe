"""Shared logging setup for the experiment entrypoints.

The stages used to `print()` directly. Routing through the stdlib `logging` module
gives unattended `reproduce.sh` runs timestamps, and gives interactive users a
verbosity knob — without changing any logic. Library code (core/, modalities/)
stays silent: it never configures logging, only the entrypoints do, so importing
the package never hijacks a host application's logging config.

Usage in an entrypoint:

    import logging
    from core._log import setup_logging, add_logging_args

    log = logging.getLogger(__name__)
    ...
    add_logging_args(parser)          # adds -v/--verbose and -q/--quiet
    args = parser.parse_args()
    setup_logging(args.verbose, args.quiet)
    log.info("started")
"""
from __future__ import annotations

import argparse
import logging
import sys

_FMT = "%(asctime)s %(levelname)-7s %(message)s"
_DATEFMT = "%H:%M:%S"

# Third-party loggers that are noisy at INFO; pinned to WARNING unless -v is given.
_NOISY = ("transformers", "datasets", "urllib3", "filelock", "matplotlib", "huggingface_hub")


def add_logging_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add mutually compatible -v/--verbose (repeatable) and -q/--quiet flags."""
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for DEBUG (repeat for third-party DEBUG too).")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Only warnings and errors.")
    return parser


def setup_logging(verbose: int = 0, quiet: bool = False) -> logging.Logger:
    """Configure the root logger once, idempotently.

    Levels: quiet -> WARNING; default -> INFO; -v (or more) -> DEBUG. Logs go to
    stderr so stdout stays clean for any piped data. Calling this again replaces
    the handler (so the in-process causal dispatch doesn't double-print)."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root = logging.getLogger()
    root.handlers[:] = [handler]      # replace, so repeated setup never stacks handlers
    root.setLevel(level)

    # Keep third-party chatter down unless the user asked for deep verbosity (-vv).
    noisy_level = logging.DEBUG if verbose >= 2 else logging.WARNING
    for name in _NOISY:
        logging.getLogger(name).setLevel(noisy_level)
    return root
