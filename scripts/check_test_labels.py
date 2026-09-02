#!/usr/bin/env python3
"""Check that every Rust test carries an explicit `#[serial]` or `#[parallel]`.

`serial_test`'s `#[serial]` only excludes tests that are themselves annotated
with `#[serial]` or `#[parallel]`. An unannotated test runs concurrently with a
serial one, so a single missing label silently reopens every race the serial
annotations exist to prevent.
"""

import re
from pathlib import Path

import typer
from _build_utils import REPO_ROOT, console

# Crates whose test binaries are built by `just test` / CI.
SEARCH_ROOTS = ("src", "crates", "tests", "profiles")

TEST_ATTR = re.compile(r"^#\[(?:test|rstest(?:\(.*\))?|tokio::test(?:\(.*\))?)\]$")
FN_DECL = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z0-9_]+)")
LABEL_ATTR = re.compile(r"^#\[(?:serial_test::)?(?:serial|parallel)\b")
CASE_ATTR = re.compile(r"^#\[(?:values|case)\b")


def _strip_literals(line: str) -> str:
    """Blank out string and char literals so their brackets are not counted.

    `#[case("[")]` is balanced Rust but not balanced text, and miscounting it
    would make the scanner lose the attribute block and skip a real test.
    """
    return re.sub(r'r?"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)\'', "", line)


def _attribute_block(lines: list[str], start: int) -> tuple[list[str], int, str] | None:
    """Collect the attributes from `start` up to the `fn` they decorate.

    Multi-line attributes are joined into one entry so the caller sees an
    ordered list of complete attributes. Returns `None` if anything other than
    an attribute or doc comment intervenes, which means `start` does not
    decorate a function.
    """
    attrs: list[str] = []
    pending = ""
    depth = 0
    i = start

    while i < len(lines):
        stripped = lines[i].strip()

        if depth == 0:
            decl = FN_DECL.match(stripped)
            if decl:
                return attrs, i, decl.group(1)
            if stripped.startswith("//") or not stripped:
                i += 1
                continue
            if not stripped.startswith("#["):
                return None

        pending = f"{pending} {stripped}".strip() if pending else stripped
        code = _strip_literals(stripped)
        depth += code.count("[") - code.count("]")
        if depth <= 0:
            attrs.append(pending)
            pending, depth = "", 0
        i += 1

    return None


def unlabeled_tests(path: Path) -> list[tuple[int, str]]:
    """Return `(line number, test name)` for each unlabeled test in `path`."""
    lines = path.read_text(encoding="utf-8").split("\n")
    missing: list[tuple[int, str]] = []

    i = 0
    while i < len(lines):
        if not TEST_ATTR.match(lines[i].strip()):
            i += 1
            continue

        # A label may sit above the test attribute, so rewind to the start of
        # the contiguous attribute block before collecting it.
        start = i
        while start > 0:
            previous = lines[start - 1].strip()
            if previous.startswith(("#[", "//")):
                start -= 1
            else:
                break

        block = _attribute_block(lines, start)
        if block is None:
            i += 1
            continue

        attrs, fn_line, name = block

        # rstest expands #[case]/#[values] rows into separate test functions and
        # only carries attributes that follow the last such row, so a label
        # placed above them is silently dropped. Only labels after the final
        # case row count.
        last_case = max(
            (n for n, attr in enumerate(attrs) if CASE_ATTR.match(attr)), default=-1
        )
        if not any(LABEL_ATTR.match(attr) for attr in attrs[last_case + 1 :]):
            missing.append((fn_line + 1, name))

        i = fn_line + 1

    return missing


def main() -> None:
    """Fail if any Rust test is missing a parallelization label."""
    findings: list[tuple[Path, int, str]] = []
    checked = 0

    for root in SEARCH_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.rs")):
            if "target" in path.parts:
                continue
            checked += 1
            for line, name in unlabeled_tests(path):
                findings.append((path.relative_to(REPO_ROOT), line, name))

    if findings:
        # markup=False throughout: the attribute names carry square brackets,
        # which rich would otherwise parse as style tags and swallow.
        console.print(
            f"✗ {len(findings)} Rust test(s) missing #[serial] or #[parallel]:",
            style="red",
            markup=False,
        )
        for path, line, name in findings:
            console.print(f"  {path}:{line} {name}", markup=False)
        console.print(
            "\nAdd #[parallel] to tests that touch no process-global state, or "
            "#[serial] to tests that mutate it (global EOP, gravity, space "
            "weather, the SPICE kernel registry, the frame and object "
            "registries, the thread pool, or environment variables).\n"
            "On an rstest the attribute must follow the last #[case] or "
            "#[values] row; rstest silently drops one placed above them, and "
            "every generated case then runs unserialized.",
            markup=False,
        )
        raise typer.Exit(1)

    console.print(f"[green]✓ All Rust tests labeled ({checked} files checked)[/green]")


if __name__ == "__main__":
    typer.run(main)
