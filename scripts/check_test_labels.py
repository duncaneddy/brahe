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


def unlabeled_tests(path: Path) -> list[tuple[int, str]]:
    """Return `(line number, test name)` for each unlabeled test in `path`."""
    lines = path.read_text(encoding="utf-8").split("\n")
    missing: list[tuple[int, str]] = []

    i = 0
    while i < len(lines):
        if not TEST_ATTR.match(lines[i].strip()):
            i += 1
            continue

        # Scan forward across the attribute block (rstest `#[case]` rows can be
        # numerous) to the `fn` the attributes decorate.
        labeled = False
        j = i
        while j < len(lines):
            stripped = lines[j].strip()
            if LABEL_ATTR.match(stripped):
                labeled = True
            decl = FN_DECL.match(stripped)
            if decl:
                if not labeled:
                    missing.append((j + 1, decl.group(1)))
                break
            j += 1
        i = j + 1

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
            "The attribute must sit directly above fn — rstest drops it if "
            "placed among the #[case] rows.",
            markup=False,
        )
        raise typer.Exit(1)

    console.print(f"[green]✓ All Rust tests labeled ({checked} files checked)[/green]")


if __name__ == "__main__":
    typer.run(main)
