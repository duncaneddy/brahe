#!/usr/bin/env python3
"""Extract a single version's release-notes block from CHANGELOG.md.

Used by CI at release time to produce `release_notes.md` for the GitHub Release
body, sourced from the CHANGELOG that the maintainer already committed and
tagged. This keeps the published release notes consistent with the file users
read in the repo, and avoids re-querying the GitHub API in CI.

Typical usage in CI:

    python3 scripts/extract_release_notes.py \
        --version 1.5.2 \
        --changelog CHANGELOG.md \
        --output release_notes.md

The extracted block excludes the `## [version] - date` heading itself, so it
renders as the body of a GitHub Release whose title is "Release v1.5.2".
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def extract_version_block(changelog_text: str, version: str) -> str:
    """Return the body of the `## [<version>] - <date>` block from <changelog_text>.

    The returned text excludes the version heading itself and any leading or
    trailing whitespace, so it can be used directly as a GitHub Release body.

    Raises ValueError if no `## [<version>] - YYYY-MM-DD` heading is present, or
    if the matched block has an empty body. CI uses these failures to catch
    "forgot to run `just generate-changelog`" or a malformed entry.
    """
    pattern = rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$"
    lines = changelog_text.splitlines()
    start = next((i for i, line in enumerate(lines) if re.match(pattern, line)), None)
    if start is None:
        raise ValueError(f"no '## [{version}] - <date>' heading in changelog")
    end = next((i for i in range(start + 1, len(lines))
                if lines[i].startswith("## [")), len(lines))
    body = "\n".join(lines[start + 1:end]).strip("\n")
    if not body.strip():
        raise ValueError(f"version {version} has empty body")
    return body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True,
                        help="Release version to extract, e.g. 1.5.2")
    parser.add_argument("--changelog", default="CHANGELOG.md",
                        help="Path to the CHANGELOG.md file (default: %(default)s)")
    parser.add_argument("--output", default=None,
                        help="Write the extracted block to this path. "
                             "If omitted, prints to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    changelog = Path(args.changelog)
    if not changelog.exists():
        print(f"error: CHANGELOG not found at {changelog}", file=sys.stderr)
        return 1

    try:
        block = extract_version_block(changelog.read_text(), args.version)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(block + "\n")
        print(f"Wrote {args.output} ({len(block.splitlines())} lines)")
    else:
        print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
