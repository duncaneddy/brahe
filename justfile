# Brahe Build System
# Usage: just <recipe> [args...]

# Use .venv python by default for script commands
scripts_dir := "scripts"
python := ".venv/bin/python"

# Ensure uv is installed and .venv exists
_setup:
    @which uv > /dev/null 2>&1 || { echo "Error: uv is not installed. Install from https://docs.astral.sh/uv/"; exit 1; }
    @test -d .venv || uv sync

# ───── Testing ─────

# Run all tests (Rust + Python)
test *flags: test-rust test-python

# Run Rust tests
test-rust *flags:
    cargo test {{flags}}

# Run Python tests
test-python *flags: _setup
    uv pip install -e . --quiet
    {{python}} -m pytest tests/ -v {{flags}}

# Test all documentation examples (delegates to scripts/test_examples.py)
test-examples *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_examples.py {{args}}

# Test a specific example (delegates to scripts/test_example.py)
test-example *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_example.py {{args}}

# ───── Benchmarks ─────

# Run all benchmarks
bench *flags:
    cargo bench {{flags}}

# Run provider benchmarks only
bench-providers *flags:
    cargo bench --bench provider_benchmarks {{flags}}

# Run propagator benchmarks only
bench-propagators *flags:
    cargo bench --bench propagator_benchmarks {{flags}}

# ───── Code Quality ─────

# Format all code (Rust + Python)
format: _setup
    cargo fmt
    uv run ruff format

# Check formatting without changes
format-check: _setup
    cargo fmt -- --check
    uv run ruff format --check

# Run linters (clippy + ruff)
lint: _setup
    cargo clippy --all-targets --all-features -- -D warnings
    uv run ruff check

# Run linters with auto-fix
lint-fix: _setup
    cargo clippy --all-targets --all-features --fix --allow-dirty -- -D warnings
    uv run ruff check --fix

# ───── Documentation ─────

# Build documentation
docs: _setup
    ./scripts/generate_stubs.sh
    uv run mkdocs build

# Serve documentation locally
docs-serve: _setup
    ./scripts/generate_stubs.sh
    uv run mkdocs serve

# ───── Plots & Figures ─────

# Generate all documentation plots (delegates to scripts/make_plots.py)
make-plots *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plots.py {{args}}

# Generate a specific plot (delegates to scripts/make_plot.py)
make-plot *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plot.py {{args}}

# ───── Build & Package ─────

# Build packages for distribution (delegates to scripts/build_packages.py)
build *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/build_packages.py {{args}}

# Install Python extension in development mode
install: _setup
    uv pip install -e .

# Generate type stubs
stubs: _setup
    ./scripts/generate_stubs.sh

# ───── Utilities ─────

# Clean build artifacts
clean:
    cargo clean
    rm -rf dist build .ruff_cache .mypy_cache .pytest_cache
    find . -path ./.venv -prune -o -name '__pycache__' -type d -print -exec rm -rf {} +
    rm -f brahe/*.so

# List all examples (delegates to scripts/list_examples.py)
list-examples *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_examples.py {{args}}

# List all plots (delegates to scripts/list_plots.py)
list-plots *args: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_plots.py {{args}}

# Show statistics about examples and plots (delegates to scripts/stats.py)
stats: _setup
    PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/stats.py

# ───── Full Quality Check ─────

# Run full quality check (test + lint + format + stubs + docs)
check: test lint format-check stubs docs
    @echo "✓ All quality checks passed!"
