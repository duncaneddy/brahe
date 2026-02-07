# Brahe Build System
# Usage: just <recipe> [args...]

# Use .venv python by default for script commands
scripts_dir := "scripts"
python := ".venv/bin/python"

# ───── Testing ─────

# Run all tests (Rust + Python)
test *flags: test-rust test-python

# Run Rust tests
test-rust *flags:
    cargo test {{flags}}

# Run Python tests
test-python *flags:
    uv pip install -e . --quiet
    {{python}} -m pytest tests/ -v {{flags}}

# Test all documentation examples (delegates to scripts/test_examples.py)
test-examples *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_examples.py {{args}}

# Test a specific example (delegates to scripts/test_example.py)
test-example *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_example.py {{args}}

# ───── Code Quality ─────

# Format all code (Rust + Python)
format:
    cargo fmt
    uv run ruff format

# Check formatting without changes
format-check:
    cargo fmt -- --check
    uv run ruff format --check

# Run linters (clippy + ruff)
lint:
    cargo clippy --all-targets --all-features -- -D warnings
    uv run ruff check

# Run linters with auto-fix
lint-fix:
    cargo clippy --all-targets --all-features --fix --allow-dirty -- -D warnings
    uv run ruff check --fix

# ───── Documentation ─────

# Build documentation
docs:
    ./scripts/generate_stubs.sh
    uv run mkdocs build

# Serve documentation locally
docs-serve:
    ./scripts/generate_stubs.sh
    uv run mkdocs serve

# ───── Plots & Figures ─────

# Generate all documentation plots (delegates to scripts/make_plots.py)
make-plots *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plots.py {{args}}

# Generate a specific plot (delegates to scripts/make_plot.py)
make-plot *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plot.py {{args}}

# ───── Build & Package ─────

# Build packages for distribution (delegates to scripts/build_packages.py)
build *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/build_packages.py {{args}}

# Install Python extension in development mode
install:
    uv pip install -e .

# Generate type stubs
stubs:
    ./scripts/generate_stubs.sh

# ───── Utilities ─────

# Clean build artifacts
clean:
    cargo clean
    rm -rf dist build .ruff_cache .mypy_cache .pytest_cache
    find . -path ./.venv -prune -o -name '__pycache__' -type d -print -exec rm -rf {} +
    rm -f brahe/*.so

# List all examples (delegates to scripts/list_examples.py)
list-examples *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_examples.py {{args}}

# List all plots (delegates to scripts/list_plots.py)
list-plots *args:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_plots.py {{args}}

# Show statistics about examples and plots (delegates to scripts/stats.py)
stats:
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/stats.py

# ───── Full Quality Check ─────

# Run full quality check (test + lint + format + stubs + docs)
check: test lint format-check stubs docs
    @echo "✓ All quality checks passed!"
