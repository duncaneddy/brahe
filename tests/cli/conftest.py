import re

import pytest


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return _ANSI_RE.sub("", text)


class _CleanResult:
    """Wrapper around Click/Typer Result that strips ANSI from output."""

    def __init__(self, result):
        self._result = result

    @property
    def stdout(self):
        return strip_ansi(self._result.stdout)

    @property
    def output(self):
        return strip_ansi(self._result.output)

    def __getattr__(self, name):
        return getattr(self._result, name)


@pytest.fixture(autouse=True)
def _patch_cli_runner(monkeypatch):
    """Patch CliRunner.invoke to strip ANSI codes from stdout/stderr.

    Rich/Typer can emit ANSI escape codes when FORCE_COLOR is set,
    which breaks substring assertions like ``'--group' in result.stdout``
    because the dashes get wrapped in escape sequences.
    """
    from typer.testing import CliRunner

    _orig_invoke = CliRunner.invoke

    def _clean_invoke(self, *args, **kwargs):
        return _CleanResult(_orig_invoke(self, *args, **kwargs))

    monkeypatch.setattr(CliRunner, "invoke", _clean_invoke)
