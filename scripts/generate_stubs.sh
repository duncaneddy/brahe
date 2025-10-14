#!/bin/bash
#
# Generate Python type stub files for the brahe._brahe extension module
#
# This script:
# 1. Generates initial stubs using pyo3-stubgen
# 2. Adds proper type annotations using our Python script
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "==> Generating stub files with pyo3-stubgen..."
.venv/bin/pyo3-stubgen brahe._brahe .

echo "==> Adding type annotations..."
.venv/bin/python "$SCRIPT_DIR/add_stub_annotations.py"

echo "==> Stub generation complete!"
echo "    Stub file: $REPO_ROOT/brahe/_brahe.pyi"
