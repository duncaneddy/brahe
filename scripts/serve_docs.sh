#!/bin/bash
#
# Serve documentation with full rebuild
#
# This script:
# 1. Reinstalls the brahe package (rebuilding Rust extensions)
# 2. Regenerates Python type stubs with complete docstrings
# 3. Builds and serves the documentation
#
# Usage:
#   $ ./scripts/serve_docs.sh

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "==> Step 1/3: Reinstalling brahe package..."
.venv/bin/pip install -e . --quiet

echo "==> Step 2/3: Generating stub files..."
./scripts/generate_stubs.sh

echo "==> Step 3/3: Building and serving documentation..."
.venv/bin/mkdocs serve

echo "==> Documentation server stopped."
