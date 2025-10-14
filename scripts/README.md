# Brahe Development Scripts

This directory contains utility scripts for brahe development.

## Type Stub Generation

### `generate_stubs.sh`

Generates Python type stub files (`.pyi`) for the `brahe._brahe` extension module.

**Usage:**
```bash
./scripts/generate_stubs.sh
```

**When to run:**
- After modifying PyO3 bindings in `src/pymodule/`
- Before building documentation locally
- The CI/CD pipeline runs this automatically

**How it works:**
1. Uses `pyo3-stubgen` to generate initial stubs with docstrings
2. Runs `add_stub_annotations.py` to add proper type annotations
3. Outputs to `brahe/_brahe.pyi`

### `add_stub_annotations.py`

Internal script that parses docstrings and adds type annotations to stub files.

**Direct usage:**
```bash
.venv/bin/python scripts/add_stub_annotations.py
```

This script is automatically called by `generate_stubs.sh`.

## Requirements

- Python virtual environment must be active (`.venv`)
- `pyo3-stubgen` must be installed (`uv sync --all-extras`)
- The brahe extension module must be built (`uv pip install -e .`)

## Integration

### CI/CD

Stub generation is integrated into:
- `.github/workflows/update_docs.yml` - Runs before building documentation
- Release workflow inherits from `update_docs.yml`

### Package Distribution

Stub files are included in the Python package distribution via `pyproject.toml`:
```toml
[tool.maturin]
include = ["brahe/_brahe.pyi"]
```

This ensures that type hints are available to users of the package.
