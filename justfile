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
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_examples.py {{args}}

# Test a specific example (delegates to scripts/test_example.py)
test-example *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_example.py {{args}}

# ───── Coverage ─────

# Run Rust tests with coverage (core crate only — `brahe-py` has no unit tests;
# its Rust coverage is captured by `just coverage-combined` via pytest).
coverage-rust:
    cargo llvm-cov -p brahe --lcov --output-path lcov.info
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
    uv run maturin develop --uv --features pyo3/extension-module
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
bench-compare-setup: _setup _bench-compare-build-rust _bench-compare-build-java _bench-compare-build-basilisk _bench-compare-build-gmat _bench-compare-orekit-data
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

# Install Basilisk Python wheel, pre-fetch its large data files, and
# download the high-precision Earth orientation binary PCK so that
# pyswice-based frame transforms can use ITRF93 (matches Orekit's ITRF much
# more closely than IAU_EARTH, which is a cartographic frame off by ~tens
# of km from ITRF).
_bench-compare-build-basilisk:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing Basilisk (bsk) Python wheel..."
    uv pip install --quiet bsk
    echo "Pre-fetching Basilisk large data (gravity coefficients, SPICE ephemerides)..."
    {{python}} - <<'PY' || echo "  (large-data pre-fetch skipped; pooch will fetch lazily on first run)"
    try:
        from Basilisk.utilities.supportDataTools.dataFetcher import fetchAll
        fetchAll()
    except Exception as exc:
        raise SystemExit(f"large-data pre-fetch failed: {exc}")
    PY
    BSK_DATA_DIR="$HOME/.cache/bsk-data"
    BPC="$BSK_DATA_DIR/earth_latest_high_prec.bpc"
    if [ ! -f "$BPC" ]; then
        mkdir -p "$BSK_DATA_DIR"
        echo "Downloading high-precision Earth orientation binary PCK (~10 MB)..."
        curl -fSL "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc" -o "$BPC"
    else
        echo "Earth high-precision PCK already present: $BPC"
    fi
    echo "✓ Basilisk installed"

# Generate GMAT's absolute-path API startup file if GMAT_ROOT_PATH is set.
# Idempotent: re-running overwrites api_startup_file.txt with identical content.
# Silent skip if GMAT_ROOT_PATH is unset — GMAT is dev-machine-only, not a CI dependency.
_bench-compare-build-gmat:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -z "${GMAT_ROOT_PATH:-}" ]; then
        echo "  GMAT_ROOT_PATH not set; GMAT baseline will be skipped at runtime"
        exit 0
    fi
    if [ ! -f "$GMAT_ROOT_PATH/bin/gmat_startup_file.txt" ]; then
        echo "  GMAT_ROOT_PATH=$GMAT_ROOT_PATH but bin/gmat_startup_file.txt missing — check install"
        exit 1
    fi
    echo "Generating GMAT api_startup_file.txt..."
    PYTHON_ABS="$(realpath "{{python}}")"
    cd "$GMAT_ROOT_PATH/api"
    "$PYTHON_ABS" BuildApiStartupFile.py
    echo "✓ GMAT api_startup_file.txt generated at $GMAT_ROOT_PATH/bin/"

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

# Run comparative performance benchmarks (timing only) across languages
bench-compare *args: _setup
    @uv pip install -e . --quiet
    @{{python}} -m benchmarks.comparative.runner perf {{args}}

# Run comparative accuracy benchmarks (sweep over initial conditions) vs OreKit
bench-compare-accuracy *args: _setup
    @uv pip install -e . --quiet
    @{{python}} -m benchmarks.comparative.runner accuracy {{args}}

# Generate comparison plots from latest benchmark results
bench-compare-plot *args: _setup
    @{{python}} -m benchmarks.comparative.runner plot {{args}}

# List available comparative benchmark tasks
bench-compare-list: _setup
    @{{python}} -m benchmarks.comparative.runner list

# Run comparative benchmarks (perf + accuracy), generate figures + CSV tables, and stage for commit
bench-compare-publish *args: _setup
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Running performance benchmarks..."
    uv pip install -e . --quiet
    {{python}} -m benchmarks.comparative.runner perf {{args}}
    echo ""
    echo "Running accuracy benchmarks..."
    {{python}} -m benchmarks.comparative.runner accuracy {{args}}
    echo ""
    echo "Generating benchmark figures and CSV tables..."
    BRAHE_FIGURE_OUTPUT_DIR=./docs/figures/ {{python}} plots/fig_comparative_benchmarks.py
    echo ""
    echo "Staging benchmark artifacts..."
    git add -f docs/figures/bench_*.csv
    git add -f docs/figures/fig_bench_*_light.html docs/figures/fig_bench_*_dark.html
    echo ""
    echo "✓ Benchmark artifacts staged. Review with 'git status' and commit when ready."

# ───── Profiling ─────

# Set up the full dev environment (Python dev deps + Rust dev tools).
# Idempotent: only installs samply if not already on PATH.
dev-setup: _setup
    uv sync --group dev
    @command -v samply > /dev/null 2>&1 || cargo install samply
    @echo "✓ Dev environment ready (samply $(samply --version 2>/dev/null || echo 'missing'), py-spy $(py-spy --version 2>/dev/null || echo 'missing'))"

# Internal: verify the profiling tools are installed and bail with a helpful
# message if not. Split per tool so `profile-rust` doesn't fail when py-spy
# is missing (and vice versa).
_check-samply:
    @command -v samply > /dev/null 2>&1 || { echo "samply not found. Run: just dev-setup"; exit 1; }

_check-py-spy:
    @test -x .venv/bin/py-spy || command -v py-spy > /dev/null 2>&1 || { echo "py-spy not found. Run: just dev-setup"; exit 1; }

# Profile a Rust workload (CPU sampling via samply by default; pass --heap
# for dhat-heap allocation profiling).
# Flags: --duration N (default 10), --no-open, --heap
profile-rust name *flags: _check-samply
    #!/usr/bin/env bash
    set -euo pipefail
    DURATION=10; OPEN=1; HEAP=0
    set -- {{flags}}
    while [ $# -gt 0 ]; do
      case "$1" in
        --duration=*)   DURATION="${1#*=}" ;;
        --duration)     shift; DURATION="$1" ;;
        --no-open)      OPEN=0 ;;
        --heap)         HEAP=1 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
      esac
      shift
    done
    TS="$(date +%Y-%m-%dT%H-%M-%S)"
    mkdir -p profiles/results
    if [ "$HEAP" -eq 1 ]; then
        echo "  Building rk4 task (profile=profiling, features=dhat-heap)..."
        cargo build --manifest-path profiles/rust/Cargo.toml \
            --profile profiling --features dhat-heap --bin {{name}}
        OUT="profiles/results/${TS}_{{name}}.dhat.json"
        BIN_ABS="$(pwd)/profiles/rust/target/profiling/{{name}}"
        echo "  Running {{name}} (duration=${DURATION}s, heap mode)..."
        (cd profiles/results && PROFILE_DURATION_S=$DURATION "$BIN_ABS")
        mv profiles/results/dhat-heap.json "$OUT"
        echo "  Wrote $OUT"
        if [ "$OPEN" -eq 1 ]; then
            (open "https://nnethercote.github.io/dh_view/dh_view.html" 2>/dev/null \
                || xdg-open "https://nnethercote.github.io/dh_view/dh_view.html" 2>/dev/null \
                || echo "  (open manually: https://nnethercote.github.io/dh_view/dh_view.html)")
            echo "  → Drag $OUT into the dh_view page to load it."
        fi
    else
        echo "  Building {{name}} (profile=profiling)..."
        cargo build --manifest-path profiles/rust/Cargo.toml --profile profiling --bin {{name}}
        OUT="profiles/results/${TS}_{{name}}.json.gz"
        echo "  Sampling {{name}} (duration=${DURATION}s)..."
        PROFILE_DURATION_S=$DURATION samply record \
            --save-only --output "$OUT" \
            profiles/rust/target/profiling/{{name}}
        echo "  Wrote $OUT"
        if [ "$OPEN" -eq 1 ]; then
            samply load "$OUT"
        else
            echo "  (--no-open: load later with: samply load $OUT)"
        fi
    fi

# Profile a Python workload using py-spy with --native to unwind into the
# Rust extension.
# Flags: --duration N (default 10), --no-open
profile-python name *flags: _check-py-spy
    #!/usr/bin/env bash
    set -euo pipefail
    DURATION=10; OPEN=1
    set -- {{flags}}
    while [ $# -gt 0 ]; do
      case "$1" in
        --duration=*)   DURATION="${1#*=}" ;;
        --duration)     shift; DURATION="$1" ;;
        --no-open)      OPEN=0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
      esac
      shift
    done
    SCRIPT="profiles/python/{{name}}.py"
    if [ ! -f "$SCRIPT" ]; then
        echo "no such profile: $SCRIPT" >&2
        exit 1
    fi
    TS="$(date +%Y-%m-%dT%H-%M-%S)"
    mkdir -p profiles/results
    OUT="profiles/results/${TS}_{{name}}.svg"
    echo "  Sampling {{name}} (duration=${DURATION}s, py-spy --native)..."
    if [ -x .venv/bin/py-spy ]; then PYSPY=".venv/bin/py-spy"; else PYSPY="py-spy"; fi
    PROFILE_DURATION_S=$DURATION PYTHONPATH=profiles/python \
        "$PYSPY" record --native --output "$OUT" --format flamegraph \
            --duration "$DURATION" \
            -- "$(pwd)/.venv/bin/python" "$SCRIPT"
    echo "  Wrote $OUT"
    if [ "$OPEN" -eq 1 ]; then
        (open "$OUT" 2>/dev/null \
            || xdg-open "$OUT" 2>/dev/null \
            || echo "  (open manually: $OUT)")
    else
        echo "  (--no-open: open later with your browser)"
    fi

# List available profile tasks (auto-discovered from profiles/rust/src/bin/
# and profiles/python/).
profile-list:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Rust profiles (run with 'just profile-rust <name>'):"
    for f in profiles/rust/src/bin/*.rs; do
      name="$(basename "$f" .rs)"
      [[ "$name" == _* ]] && continue
      echo "  - $name"
    done
    echo ""
    echo "Python profiles (run with 'just profile-python <name>'):"
    for f in profiles/python/*.py; do
      name="$(basename "$f" .py)"
      [[ "$name" == _* ]] && continue
      echo "  - $name"
    done

# Run both Rust and Python flavors of the same task for side-by-side comparison.
# Honors --duration and --no-open; --heap is not supported (CPU only).
profile-compare name *flags:
    @just profile-rust {{name}} {{flags}}
    @just profile-python {{name}} {{flags}}

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
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plots.py {{args}}

# Generate a specific plot (delegates to scripts/make_plot.py)
make-plot *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/make_plot.py {{args}}

# ───── Build & Package ─────

# Build packages for distribution (delegates to scripts/build_packages.py)
build *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/build_packages.py {{args}}

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
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_examples.py {{args}}

# List all plots (delegates to scripts/list_plots.py)
list-plots *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/list_plots.py {{args}}

# Show statistics about examples and plots (delegates to scripts/stats.py)
stats: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/stats.py

# ───── Release ─────

# Set the workspace version in Cargo.toml (single source of truth for all crates)
set-version version:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [[ "{{version}}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "error: version must be MAJOR.MINOR.PATCH (got '{{version}}')" >&2
        exit 1
    fi
    # Update the [workspace.package] version. The brahe and brahe-py crates
    # inherit it via `version.workspace = true`.
    python3 -c "
    import re, pathlib
    p = pathlib.Path('Cargo.toml')
    text = p.read_text()
    new = re.sub(
        r'(\[workspace\.package\][^\[]*?\nversion = )\"[^\"]+\"',
        r'\\g<1>\"{{version}}\"',
        text,
        count=1,
        flags=re.DOTALL,
    )
    if new == text:
        raise SystemExit('error: could not find [workspace.package] version in Cargo.toml')
    p.write_text(new)
    "
    # Refresh Cargo.lock so the workspace version bump is reflected there too.
    cargo update --workspace --quiet
    echo "✓ Set workspace version to {{version}}"

# Regenerate the CHANGELOG.md entry for an upcoming release.
# Defaults: version from Cargo.toml, prev-tag from `git describe --tags`.
generate-changelog version="" prev_tag="":
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION="{{version}}"
    PREV_TAG="{{prev_tag}}"
    if [ -z "$VERSION" ]; then
        VERSION=$(grep -E '^version = ' Cargo.toml | head -1 | sed -E 's/version = "(.*)"/\1/')
        echo "Using version from Cargo.toml: $VERSION"
    fi
    if [ -z "$PREV_TAG" ]; then
        PREV_TAG=$(git describe --tags --abbrev=0)
        echo "Using previous tag from git: $PREV_TAG"
    fi
    if ! command -v gh > /dev/null 2>&1; then
        echo "error: gh CLI is required (https://cli.github.com/)" >&2
        exit 1
    fi
    python3 scripts/generate_release_notes.py \
        --version "$VERSION" \
        --prev-tag "$PREV_TAG" \
        --changelog CHANGELOG.md
    echo "✓ CHANGELOG.md updated for v$VERSION. Review the diff, then commit and tag."

# ───── Full Quality Check ─────

# Run full quality check (test + lint + format + stubs + docs)
check: test lint format-check stubs docs
    @echo "✓ All quality checks passed!"
