#!/usr/bin/env python3
"""Generate release notes and update CHANGELOG.md from merged PRs since the prior tag.

Replaces the towncrier+news-fragments pipeline. PR bodies are the single source of truth:
each PR contributes one or more Keep-a-Changelog sections (`### Added`, `### Changed`, ...)
in its description; this script aggregates them across all PRs merged since the previous
release tag and writes:

  - <changelog>: a new `## [<version>] - <date>` block inserted after the
    `<!-- release notes start -->` marker.
  - <release-notes>: the same block, suitable for use as a GitHub Release body.

Bot PRs and PRs labeled `automated`, `data-update`, or `dependencies` are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SECTIONS: tuple[str, ...] = ("Added", "Changed", "Deprecated", "Removed", "Fixed")
SKIP_LABELS: frozenset[str] = frozenset({"automated", "data-update", "dependencies"})
MARKER = "<!-- release notes start -->"


@dataclass(frozen=True)
class PRInfo:
    number: int
    title: str
    author: str
    labels: tuple[str, ...]
    body: str


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(
            f"command failed (exit {res.returncode}): {' '.join(cmd)}\n"
            f"stderr: {res.stderr.strip() or '<empty>'}"
        )
    return res.stdout


PR_REF_RE = re.compile(r"\(#(\d+)\)\s*$")


def pr_numbers_in_range(prev_tag: str, head: str = "HEAD") -> list[int]:
    """Return PR numbers parsed from squash-merge subjects in <prev_tag>..<head>."""
    log = run(["git", "log", f"{prev_tag}..{head}", "--no-merges", "--format=%s"])
    seen: dict[int, None] = {}
    for line in log.splitlines():
        m = PR_REF_RE.search(line.strip())
        if m:
            seen[int(m.group(1))] = None
    return list(seen)


def fetch_prs_since(prev_tag: str, repo: str) -> list[PRInfo]:
    """Return contributor PRs merged into main since the prior tag, oldest-first.

    Driven by `git log` rather than search dates so PRs that share a calendar day
    with the previous tag are correctly excluded.
    """
    numbers = pr_numbers_in_range(prev_tag)
    prs: list[PRInfo] = []
    for n in numbers:
        raw = json.loads(
            run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(n),
                    "--repo",
                    repo,
                    "--json",
                    "number,title,author,labels,body",
                ]
            )
        )
        author = raw.get("author") or {}
        if author.get("is_bot"):
            continue
        prs.append(
            PRInfo(
                number=raw["number"],
                title=raw["title"],
                author=author.get("login", "unknown"),
                labels=tuple(lbl["name"] for lbl in raw.get("labels") or []),
                body=raw.get("body") or "",
            )
        )
    prs.sort(key=lambda p: p.number)
    return prs


def is_skip(pr: PRInfo) -> bool:
    return any(lbl in SKIP_LABELS for lbl in pr.labels)


def parse_pr_sections(body: str) -> dict[str, list[str]]:
    """Extract `### <Section>` blocks of bullet entries from a PR body.

    Tolerates header suffixes (e.g. `### Deprecated / Future Warning`) and ignores
    HTML-comment placeholders. Returns {section_name: [entry, ...]} for sections
    that have at least one non-comment bullet.
    """
    out: dict[str, list[str]] = {}
    for section in SECTIONS:
        pattern = rf"###\s+{section}[^\n]*\n((?:- (?!<!--).*\n?)*)"
        m = re.search(pattern, body, re.IGNORECASE)
        if not m:
            continue
        items: list[str] = []
        for line in m.group(1).strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("<!--"):
                continue
            items.append(re.sub(r"^-\s+", "", line))
        if items:
            out[section] = items
    return out


def format_entry(text: str, pr: PRInfo, repo: str) -> str:
    """Format a single bullet: `- <text>. [@author](url) ([#PR](url))`."""
    text = text.rstrip()
    if not text.endswith("."):
        text = text + "."
    author_url = f"https://github.com/{pr.author}"
    pr_url = f"https://github.com/{repo}/pull/{pr.number}"
    return f"- {text} [@{pr.author}]({author_url}) ([#{pr.number}]({pr_url}))"


EMPTY_RELEASE_NOTE = "_No notable changes for this release._"


def build_release_section(
    version: str, prs: list[PRInfo], repo: str, today: str
) -> str:
    aggregated: dict[str, list[str]] = {s: [] for s in SECTIONS}
    for pr in prs:
        for section, entries in parse_pr_sections(pr.body).items():
            for entry in entries:
                aggregated[section].append(format_entry(entry, pr, repo))

    lines: list[str] = [f"## [{version}] - {today}"]
    if not any(aggregated.values()):
        # Maintenance releases (only bot/data/dependency PRs, or no merges at all)
        # still need a non-empty body so the GitHub Release draft and CHANGELOG
        # entry render legibly.
        lines.append("")
        lines.append(EMPTY_RELEASE_NOTE)
        return "\n".join(lines) + "\n"

    for section in SECTIONS:
        if not aggregated[section]:
            continue
        lines.append("")
        lines.append(f"### {section}")
        lines.append("")
        lines.extend(aggregated[section])
    return "\n".join(lines) + "\n"


def update_changelog(changelog_path: Path, section: str) -> bool:
    """Insert <section> after the towncrier marker. Idempotent: returns False if a
    block for the same version heading already exists."""
    text = changelog_path.read_text()
    version_heading = section.split("\n", 1)[0]
    if version_heading in text:
        return False
    if MARKER not in text:
        raise SystemExit(f"Could not find marker {MARKER!r} in {changelog_path}")
    head, tail = text.split(MARKER, 1)
    new = head + MARKER + "\n\n" + section + "\n" + tail.lstrip("\n")
    changelog_path.write_text(new)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version, e.g. 1.5.0")
    parser.add_argument(
        "--prev-tag", required=True, help="Previous release tag, e.g. v1.4.2"
    )
    parser.add_argument("--repo", default="duncaneddy/brahe")
    parser.add_argument("--changelog", default="CHANGELOG.md")
    parser.add_argument(
        "--release-notes",
        default=None,
        help="Optional path to also write the release section to "
        "(e.g. release_notes.md). Omit to only update the CHANGELOG.",
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Release date in ISO format (default: today, UTC)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated section to stdout; don't write files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prs = [p for p in fetch_prs_since(args.prev_tag, args.repo) if not is_skip(p)]
    if not prs:
        # Not a failure: maintenance releases that contain only skipped PRs
        # (dependabot, bundled-data updates, etc.) still need a release block.
        print(
            f"No contributor PRs since {args.prev_tag}; emitting empty release note.",
            file=sys.stderr,
        )
    section = build_release_section(args.version, prs, args.repo, args.date)

    if args.dry_run:
        print(section, end="")
        return 0

    if args.release_notes:
        Path(args.release_notes).write_text(section)
        print(f"Wrote {args.release_notes} ({len(section.splitlines())} lines)")
    inserted = update_changelog(Path(args.changelog), section)
    if inserted:
        print(f"Inserted v{args.version} block into {args.changelog}")
    else:
        print(f"v{args.version} block already in {args.changelog}; left unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
