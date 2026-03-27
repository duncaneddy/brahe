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

# ───── Coverage ─────

# Run Rust tests with coverage
coverage-rust:
    cargo llvm-cov --workspace --lcov --output-path lcov.info
    @echo "Rust coverage → lcov.info"

# Run Python tests with coverage
coverage-python: _setup
    uv pip install -e . --quiet
    {{python}} -m pytest tests/ --cov=brahe --cov-report=html --cov-report=xml:python-coverage.xml --cov-report=term

# Run Python tests with instrumented Rust extension (combined coverage)
coverage-combined: _setup
    #!/usr/bin/env bash
    set -euo pipefail
    cargo llvm-cov clean --workspace
    eval "$(cargo llvm-cov show-env --export-prefix)"
    uv pip install maturin --quiet
    uv run maturin develop --uv --features pyo3/extension-module,python
    {{python}} -m pytest tests/ \
        --cov=brahe \
        --cov-report=html \
        --cov-report=xml:python-coverage.xml \
        --cov-report=term
    cargo llvm-cov report --lcov --output-path python-rust-coverage.lcov
    echo ""
    echo "Coverage reports:"
    echo "  Python HTML:  htmlcov/index.html"
    echo "  Python XML:   python-coverage.xml"
    echo "  Rust LCOV:    python-rust-coverage.lcov"

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

# Paths for comparative benchmark implementations
_bench_java_dir := "benchmarks/comparative/implementations/java"
_bench_rust_manifest := "benchmarks/comparative/implementations/rust/Cargo.toml"
_orekit_data_url := "https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip"
_orekit_data_dir := env("OREKIT_DATA", "~/.orekit/orekit-data")

# Install all comparative benchmark dependencies
bench-compare-setup: _setup _bench-compare-build-rust _bench-compare-build-java _bench-compare-orekit-data
    uv pip install -e . --quiet
    @echo "✓ Comparative benchmark setup complete. Run: just bench-compare"

# Build Rust benchmark binary
_bench-compare-build-rust:
    @echo "Building Rust benchmark binary..."
    cargo build --release --manifest-path {{_bench_rust_manifest}}

# Build Java/OreKit benchmark project (generates Gradle wrapper if needed)
_bench-compare-build-java:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Building Java benchmark project..."
    cd {{_bench_java_dir}}
    if [ ! -f gradlew ]; then
        if command -v gradle &> /dev/null; then
            echo "  Generating Gradle wrapper..."
            gradle wrapper
        else
            echo "ERROR: No Gradle wrapper and gradle not installed."
            echo "  macOS:  brew install gradle"
            echo "  Linux:  sdk install gradle  (via https://sdkman.io)"
            exit 1
        fi
    fi
    ./gradlew build

# Download OreKit data if not present
_bench-compare-orekit-data:
    #!/usr/bin/env bash
    set -euo pipefail
    OREKIT_DIR="$(eval echo {{_orekit_data_dir}})"
    if [ -d "$OREKIT_DIR" ] && [ "$(ls -A "$OREKIT_DIR" 2>/dev/null)" ]; then
        echo "OreKit data found: $OREKIT_DIR"
        exit 0
    fi
    echo "Downloading OreKit data to $OREKIT_DIR..."
    mkdir -p "$(dirname "$OREKIT_DIR")"
    TMP_ZIP="$(dirname "$OREKIT_DIR")/orekit-data.zip"
    curl -fSL "{{_orekit_data_url}}" -o "$TMP_ZIP"
    unzip -q "$TMP_ZIP" -d "$(dirname "$OREKIT_DIR")"
    # The zip extracts to orekit-data-main/ — rename to expected directory
    if [ -d "$(dirname "$OREKIT_DIR")/orekit-data-main" ]; then
        rm -rf "$OREKIT_DIR"
        mv "$(dirname "$OREKIT_DIR")/orekit-data-main" "$OREKIT_DIR"
    fi
    rm -f "$TMP_ZIP"
    echo "✓ OreKit data installed: $OREKIT_DIR"

# Run comparative benchmarks across languages
bench-compare *args: _setup
    uv pip install -e . --quiet
    {{python}} -m benchmarks.comparative.runner run {{args}}

# Generate comparison plots from latest benchmark results
bench-compare-plot *args: _setup
    {{python}} -m benchmarks.comparative.runner plot {{args}}

# List available comparative benchmark tasks
bench-compare-list: _setup
    {{python}} -m benchmarks.comparative.runner list

# Run comparative benchmarks, generate figures + CSV tables, and stage for commit
bench-compare-publish *args: _setup
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running comparative benchmarks..."
    uv pip install -e . --quiet
    {{python}} -m benchmarks.comparative.runner run {{args}}
    echo ""
    echo "Generating benchmark figures and CSV tables..."
    BRAHE_FIGURE_OUTPUT_DIR=./docs/figures/ {{python}} plots/fig_comparative_benchmarks.py
    echo ""
    echo "Staging benchmark artifacts..."
    git add -f docs/figures/bench_*.csv
    git add -f docs/figures/fig_bench_*_light.html docs/figures/fig_bench_*_dark.html
    echo ""
    echo "✓ Benchmark artifacts staged. Review with 'git status' and commit when ready."

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
    uv run properdocs build

# Serve documentation locally
docs-serve: _setup
    ./scripts/generate_stubs.sh
    uv run properdocs serve

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

# ───── LSP Support ─────

# Set up LSP support (install + generate stubs)
setup-lsp: install stubs
    @echo "LSP support ready. Restart your editor/LSP server to pick up changes."

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
