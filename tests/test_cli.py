"""Tests for the ``lerobot-bench`` CLI entrypoint."""

from __future__ import annotations

import pytest

from lerobot_bench import __version__
from lerobot_bench.cli import build_parser, main


def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "lerobot-bench"


def test_version_flag_prints_version_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


def test_main_with_no_args_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 0
    # Help text mentions the program name.
    assert "lerobot-bench" in captured.out
