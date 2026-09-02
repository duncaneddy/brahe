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

# Whitespace is legal between an attribute's tokens -- `# [ test ]` is the same
# attribute as `#[test]` -- so the prefilter tolerates it, the scanner accepts
# it when it opens an attribute, and each attribute is compacted (all
# whitespace removed) before the patterns below are matched against it.
ATTR_PREFILTER = re.compile(r"#\s*!?\s*\[\s*(?:test|rstest|tokio\s*::\s*test)\b")
ATTR_START = re.compile(r"#\s*!?\s*\[")
TEST_ATTR = re.compile(r"^#\[(?:test[\](]|rstest\b|tokio::test\b)")
LABEL_ATTR = re.compile(r"^#\[(?:serial_test::)?(?:serial|parallel)\b")
CASE_ATTR = re.compile(r"^#\[(?:values|case)\b")

# A function declaration, including the modifiers Rust allows before `fn`.
# They must be part of the match: treating `async` or `pub` as an unrelated
# token would end the attribute run and lose the test.
# `(?:r#)?[^\W\d]\w*` is a Rust identifier: Python's \w is Unicode-aware, and
# Rust accepts non-ASCII identifiers, so an ASCII-only name pattern would miss
# `fn 测试()` entirely and truncate `fn café_test()`.
FN_DECL = re.compile(
    r"(?:(?:pub\s*(?:\([^)]*\)\s*)?|const\s+|async\s+|unsafe\s+"
    r'|extern\s+(?:"[^"]*"\s+)?)\s*)*'
    r"fn\s+((?:r#)?[^\W\d]\w*)"
)

# Openers for the regions that must be blanked. Matching them in one
# alternation is what keeps a `//` inside a string from reading as a comment,
# and a quote inside a comment from opening a string.
OPENER = re.compile(
    r'(?P<raw>r(?P<hashes>#*)")'
    r'|(?P<string>")'
    r"|(?P<char>'(?:[^'\\\n]|\\.)')"
    r"|(?P<line>//)"
    r"|(?P<block>/\*)"
)
STRING_BODY = re.compile(r'(?:[^"\\]|\\.)*"', re.DOTALL)
BLOCK_EDGE = re.compile(r"/\*|\*/")


def _blank(text: str) -> str:
    """Replace literals and comments with spaces, preserving every offset.

    Attribute matching then operates on text that cannot contain a stray `[`,
    `]`, `#` or `fn` inside a string or comment, while line numbers and
    positions still map back to the original file. Block comments nest in
    Rust, so they are matched by depth rather than to the first `*/`.
    """
    out = list(text)

    def blank(start: int, end: int) -> None:
        for k in range(start, end):
            if out[k] != "\n":
                out[k] = " "

    i = 0
    while True:
        m = OPENER.search(text, i)
        if m is None:
            break
        start = m.start()

        if m.lastgroup == "char":
            end = m.end()
        elif m.group("line") is not None:
            end = text.find("\n", start)
            end = len(text) if end < 0 else end
        elif m.group("block") is not None:
            depth = 0
            end = len(text)
            for edge in BLOCK_EDGE.finditer(text, start):
                depth += 1 if edge.group() == "/*" else -1
                if depth == 0:
                    end = edge.end()
                    break
        elif m.group("raw") is not None:
            close = text.find('"' + m.group("hashes"), m.end())
            end = len(text) if close < 0 else close + 1 + len(m.group("hashes"))
        else:  # ordinary string
            body = STRING_BODY.match(text, m.end())
            end = len(text) if body is None else body.end()

        blank(start, end)
        i = max(end, start + 1)

    return "".join(out)


def _attribute_end(text: str, bracket: int) -> int | None:
    """Return the offset just past the attribute whose `[` is at `bracket`."""
    depth = 0
    i = bracket
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def unlabeled_tests(path: Path) -> list[tuple[int, str]]:
    """Return `(line number, test name)` for each unlabeled test in `path`."""
    raw = path.read_text(encoding="utf-8")
    if not ATTR_PREFILTER.search(raw):
        return []

    text = _blank(raw)
    missing: list[tuple[int, str]] = []

    # Walk the file as an alternating stream of attributes and items, so a
    # test is found wherever it sits -- attributes and `fn` on one line, or
    # separated by comments and blank lines.
    attrs: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]

        if char.isspace():
            i += 1
            continue

        opener = ATTR_START.match(text, i) if char == "#" else None
        if opener:
            end = _attribute_end(text, opener.end() - 1)
            if end is None:
                # An unterminated attribute means the file does not parse the
                # way this scanner assumes. Report it: staying quiet here is
                # what would let an unlabeled test through.
                missing.append((text.count("\n", 0, i) + 1, "<unparsed attribute>"))
                break
            # Compacted: with literals already blanked, dropping every space
            # makes `# [ tokio :: test ]` and `#[tokio::test]` one spelling.
            attrs.append("".join(text[i:end].split()))
            i = end
            continue

        decl = FN_DECL.match(text, i)
        if decl:
            if any(TEST_ATTR.match(attr) for attr in attrs):
                # rstest expands #[case]/#[values] rows into separate test
                # functions and only carries attributes that follow the last
                # such row, so a label placed above them is silently dropped.
                last_case = max(
                    (n for n, attr in enumerate(attrs) if CASE_ATTR.match(attr)),
                    default=-1,
                )
                tail = attrs[last_case + 1 :]
                if not any(LABEL_ATTR.match(attr) for attr in tail):
                    line = text.count("\n", 0, decl.start(1)) + 1
                    missing.append((line, decl.group(1)))
            attrs.clear()
            i = decl.end()
            continue

        # Any other token ends the current attribute run.
        attrs.clear()
        i += 1

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
