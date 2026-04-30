"""Command-line entrypoint for ``lerobot-bench``.

Wired into ``[project.scripts]`` in ``pyproject.toml``. Subcommands will
be added incrementally as the eval lib lands; for now this only exposes
``--version`` so the daily smoke workflow has something concrete to call.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from lerobot_bench.__version__ import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-bench",
        description="Multi-policy benchmark for pretrained LeRobot policies.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"lerobot-bench {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
