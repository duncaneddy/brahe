"""Execute the Python examples embedded in brahe docstrings.

Every ```python fenced block in a docstring reachable from the ``brahe``
package (the compiled ``brahe._brahe`` bindings and the pure-Python modules)
runs as a standalone script in its own interpreter and temporary directory.
Blocks fenced as ```python {.no_run} are compiled but not executed; use that
for examples that need credentials, network access, or user-supplied files.
"""

import importlib
import inspect
import os
import pkgutil
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import brahe

pytestmark = pytest.mark.docstring_examples

REPO_ROOT = Path(__file__).resolve().parents[1]
LINKED_ASSETS = ("test_assets", "data")
TIMEOUT_SECONDS = 120

FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*)```python(?:[ \t]*\{(?P<attrs>[^}\n]*)\})?[ \t]*\n"
    r"(?P<body>.*?)^(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)


def _brahe_modules():
    names = [
        "brahe",
        *(info.name for info in pkgutil.walk_packages(brahe.__path__, "brahe.")),
    ]
    for name in names:
        yield importlib.import_module(name)


def _documented_objects():
    """Yield ``(qualified_name, docstring)`` for every documented public object."""
    for module in _brahe_modules():
        yield module.__name__, module.__doc__
        for name, obj in sorted(vars(module).items()):
            if name.startswith("_"):
                continue
            owner = getattr(obj, "__module__", module.__name__) or module.__name__
            if not owner.startswith("brahe"):
                continue
            qualname = f"{module.__name__}.{name}"
            yield qualname, getattr(obj, "__doc__", None)
            if inspect.isclass(obj):
                for member_name in sorted(vars(obj)):
                    if member_name.startswith("__") and member_name != "__init__":
                        continue
                    member = getattr(obj, member_name, None)
                    yield f"{qualname}.{member_name}", getattr(member, "__doc__", None)


def _collect_examples():
    examples = []
    seen = set()
    for qualname, doc in _documented_objects():
        if not isinstance(doc, str) or "```python" not in doc:
            continue
        for index, match in enumerate(FENCE_RE.finditer(inspect.cleandoc(doc))):
            code = textwrap.dedent(match.group("body"))
            if code in seen:
                continue
            seen.add(code)
            attrs = (match.group("attrs") or "").split()
            examples.append(
                pytest.param(code, ".no_run" in attrs, id=f"{qualname}[{index}]")
            )
    return examples


@pytest.mark.parametrize(("code", "no_run"), _collect_examples())
def test_docstring_example(code, no_run, tmp_path):
    if no_run:
        compile(code, "<docstring example>", "exec")
        return

    for asset in LINKED_ASSETS:
        os.symlink(REPO_ROOT / asset, tmp_path / asset, target_is_directory=True)
    script = tmp_path / "example.py"
    script.write_text(code)
    env = dict(
        os.environ,
        BRAHE_NETWORK_MODE="offline",
        MPLBACKEND="Agg",
        PLOTLY_RENDERER="json",
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"example exited with {result.returncode}\n--- code ---\n{code}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
