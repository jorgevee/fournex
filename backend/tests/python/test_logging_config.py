"""Tests for CLI logging configuration (-v/--verbose, --debug)."""
import argparse
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from fournex.cli import _build_parser, _configure_logging


@pytest.fixture(autouse=True)
def _reset_fournex_logger():
    """Snapshot and restore the ``fournex`` logger so tests don't leak state."""
    log = logging.getLogger("fournex")
    saved_level, saved_handlers = log.level, log.handlers[:]
    log.handlers = []
    yield
    log.handlers = saved_handlers
    log.setLevel(saved_level)


def _ns(**kw) -> argparse.Namespace:
    return argparse.Namespace(**kw)


def test_default_level_is_warning():
    _configure_logging(_ns(verbose=False, debug=False))
    assert logging.getLogger("fournex").level == logging.WARNING


def test_verbose_sets_info():
    _configure_logging(_ns(verbose=True, debug=False))
    assert logging.getLogger("fournex").level == logging.INFO


def test_debug_takes_precedence_over_verbose():
    _configure_logging(_ns(verbose=True, debug=True))
    assert logging.getLogger("fournex").level == logging.DEBUG


def test_handler_added_once_and_targets_stderr():
    _configure_logging(_ns(verbose=True, debug=False))
    _configure_logging(_ns(debug=True, verbose=False))
    handlers = logging.getLogger("fournex").handlers
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].stream is sys.stderr


def test_missing_attrs_default_to_warning():
    # _configure_logging uses getattr() defaults; an empty namespace must not raise.
    _configure_logging(_ns())
    assert logging.getLogger("fournex").level == logging.WARNING


def test_parser_exposes_verbose_and_debug_flags():
    args = _build_parser().parse_args(["-v", "smoke-test"])
    assert args.verbose is True and args.debug is False
    args = _build_parser().parse_args(["--debug", "smoke-test"])
    assert args.debug is True
