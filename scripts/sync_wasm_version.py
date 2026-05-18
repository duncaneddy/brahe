#!/usr/bin/env python3
"""Sync crates/brahe-wasm/package.json `version` field from root Cargo.toml.

Reads `[workspace.package] version` from Cargo.toml and writes it into
crates/brahe-wasm/package.json. Idempotent: if the versions already match,
no write occurs. The root Cargo.toml is the single source of truth.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CARGO_TOML = REPO_ROOT / "Cargo.toml"
PACKAGE_JSON = REPO_ROOT / "crates" / "brahe-wasm" / "package.json"


def read_cargo_version() -> str:
    with CARGO_TOML.open("rb") as f:
        data = tomllib.load(f)
    try:
        return data["workspace"]["package"]["version"]
    except KeyError as exc:
        raise SystemExit(
            f"Could not find [workspace.package] version in {CARGO_TOML}"
        ) from exc


def main() -> int:
    if not PACKAGE_JSON.exists():
        print(f"⚠ {PACKAGE_JSON} does not exist yet; skipping sync.")
        return 0

    target = read_cargo_version()
    text = PACKAGE_JSON.read_text()
    pkg = json.loads(text)
    current = pkg.get("version")

    if current == target:
        print(f"✓ package.json version already {target}")
        return 0

    pkg["version"] = target
    trailing_newline = "\n" if text.endswith("\n") else ""
    PACKAGE_JSON.write_text(json.dumps(pkg, indent=2) + trailing_newline)
    print(f"✓ package.json version: {current} → {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
