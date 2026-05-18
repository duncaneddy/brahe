#!/usr/bin/env python3
"""Bump the workspace version across all version-bearing files.

Updates root Cargo.toml [workspace.package] version, then delegates to
sync_wasm_version.py to propagate to crates/brahe-wasm/package.json, then
runs `cargo check --workspace` to refresh Cargo.lock.

Usage: python scripts/set_version.py <version>
Example: python scripts/set_version.py 1.5.2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_wasm_version.py"

SEMVER_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)

WORKSPACE_VERSION_RE = re.compile(
    r"(\[workspace\.package\][^\[]*?\nversion\s*=\s*\")[^\"]+(\")",
    re.DOTALL,
)


def update_cargo_toml(new_version: str) -> str:
    """Rewrite [workspace.package] version. Returns the previous version."""
    text = CARGO_TOML.read_text()
    match = WORKSPACE_VERSION_RE.search(text)
    if not match:
        raise SystemExit(
            f"Could not find [workspace.package] version in {CARGO_TOML}"
        )
    old_version = re.search(r'version\s*=\s*"([^"]+)"', match.group(0)).group(1)
    new_text = WORKSPACE_VERSION_RE.sub(
        rf"\g<1>{new_version}\g<2>", text, count=1
    )
    CARGO_TOML.write_text(new_text)
    return old_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "version",
        help="New workspace version (semver, e.g. 1.5.2 or 2.0.0-rc.1)",
    )
    args = parser.parse_args()

    if not SEMVER_RE.match(args.version):
        sys.exit(f"Error: '{args.version}' is not a valid semver string.")

    old = update_cargo_toml(args.version)
    print(f"✓ Cargo.toml workspace.package.version: {old} → {args.version}")

    subprocess.run([sys.executable, str(SYNC_SCRIPT)], check=True)

    print("Refreshing Cargo.lock via `cargo check --workspace --offline`...")
    subprocess.run(
        ["cargo", "check", "--workspace", "--offline"],
        cwd=REPO_ROOT,
        check=True,
    )
    print("✓ Cargo.lock refreshed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
