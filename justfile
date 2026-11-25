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

# Run all tests (Rust + Python) with coverage
test *flags: test-rust test-python

# Run all tests (Rust + Python) without coverage — fast iteration
test-fast *flags: test-rust-fast test-python-fast

# Update Rust crates locally
_update-rust:
    cargo update

# Mirrors test_rust.yml: instrument the core `brahe` crate → lcov.info, then
# print a terminal summary from the same profdata.

# Run Rust tests with coverage (mirrors CI test_rust.yml)
test-rust *flags: _update-rust
    cargo llvm-cov -p brahe {{flags}} --lcov --output-path lcov.info
    @cargo llvm-cov report -p brahe
    @cargo llvm-cov report -p brahe --html
    @echo "Rust coverage → lcov.info + target/llvm-cov/html/index.html"

# Run Rust tests without coverage — fast iteration
test-rust-fast *flags: _update-rust
    cargo test {{flags}}

# Mirrors test_python.yml's coverage job: instrument Rust via cargo-llvm-cov,
# rebuild with maturin, run pytest --cov → python-coverage.xml +
# python-rust-coverage.lcov.

# Run Python tests with combined Python+Rust coverage (mirrors CI test_python.yml)
test-python *flags: _setup
    #!/usr/bin/env bash
    set -euo pipefail
    cargo llvm-cov clean --workspace
    eval "$(cargo llvm-cov show-env --export-prefix)"
    uv pip install maturin --quiet
    uv run maturin develop --uv --features pyo3/extension-module
    {{python}} -m pytest tests/ -n auto --dist loadscope \
        --cov=brahe \
        --cov-report=html \
        --cov-report=xml:python-coverage.xml \
        --cov-report=term {{flags}}
    cargo llvm-cov report --lcov --output-path python-rust-coverage.lcov
    cargo llvm-cov report --html
    echo ""
    echo "Coverage reports:"
    echo "  Python HTML:  htmlcov/index.html"
    echo "  Python XML:   python-coverage.xml"
    echo "  Rust HTML:    target/llvm-cov/html/index.html"
    echo "  Rust LCOV:    python-rust-coverage.lcov"

# Run Python tests without coverage — fast iteration
test-python-fast *flags: _setup
    uv pip install -e . --quiet
    {{python}} -m pytest tests/ -n auto --dist loadscope {{flags}}

# Run the Python examples embedded in docstrings (mirrors `cargo test --doc`)
test-python-docs *flags: _setup
    uv pip install -e . --quiet
    {{python}} -m pytest tests/test_docstring_examples.py -m docstring_examples -n auto {{flags}}

# Run integration (network) tests — Rust + Python (needs TEST_SPACETRACK_USER/PASS)
test-integration: test-integration-rust test-integration-python

# Run Rust integration tests. TEST_SPACETRACK_BASE_URL defaults to the
# for-testing endpoint (the SpaceTrack client tests `.expect()` it); override by
# exporting it yourself. TEST_SPACETRACK_USER/PASS must be set in your environment.
test-integration-rust *flags:
    TEST_SPACETRACK_BASE_URL="${TEST_SPACETRACK_BASE_URL:-https://for-testing-only.space-track.org}" cargo test -p brahe --features integration {{flags}}

# Run Python integration tests. Installs the `all` extra so the plot integration
# tests (matplotlib/plotly/pillow/httpx) can be collected, matching CI's --all-extras.
test-integration-python *flags: _setup
    uv pip install -e ".[all]" --quiet
    {{python}} -m pytest tests/ -v -m integration {{flags}}

# Warm every network resource the examples and plots read, once, serially.
# `setup` runs it for a fresh checkout and the CI example, plot, and docs jobs
# run it before their offline steps; run it again after editing
# .github/brahe-data-manifest.txt. Idempotent: each downloader fast-paths if
# its resource is cached, so the parallel example/plot pools never race a
# cold download.
# Pre-download network resources (Natural Earth texture, land basemap, cartopy)
download-resources: _setup
    @{{python}} -c "from brahe.plots.texture_utils import download_natural_earth_texture; download_natural_earth_texture('50m')"
    @{{python}} -c "from brahe.plots.texture_utils import download_planet_texture; [download_planet_texture(b) for b in ('moon', 'mars', 'ceres')]"
    @{{python}} -c "from brahe.plots.basemap import get_natural_earth_land_shapefile; get_natural_earth_land_shapefile()"
    @{{python}} {{scripts_dir}}/warm_cartopy.py
    @{{python}} -c "import brahe as bh; bh.load_common_spice_kernels()"
    @{{python}} -c "import brahe as bh; bh.load_spice_kernel('mar099s')"
    @{{python}} -c "import brahe as bh; obj=bh.datasets.sbdb.SBDBClient().lookup('Ceres'); bh.load_spice_kernel('de440s'); t0=bh.Epoch.from_datetime(2015,12,1,0,0,0.0,0.0,bh.TimeSystem.TDB); t1=bh.Epoch.from_datetime(2016,3,1,0,0,0.0,0.0,bh.TimeSystem.TDB); bh.datasets.horizons.HorizonsClient().get_spk(bh.datasets.horizons.HorizonsSPKRequest.for_spkid(obj.naif_id(), t0, t1))"
    @{{python}} -c "import brahe as bh; bh.datasets.star_catalogs.get_fk5(); bh.datasets.star_catalogs.get_hipparcos()"
    @{{python}} -c "import brahe as bh; bh.datasets.icgem.download_model('moon', 'GRGM660PRIM'); bh.datasets.icgem.download_model('mars', 'ggm2bc80')"
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/seed_celestrak_cache.py

# Kept out of download-resources because it would replace a live Starlink
# manifest with the five-entry fixture; the script refuses that without --force.
# Install the committed Starlink fixtures into the brahe cache for offline examples and docs
seed-starlink-cache *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/seed_starlink_cache.py {{args}}

# Refresh the committed Celestrak GP snapshots in test_assets/celestrak from
# the live API. Run when the snapshots should track current orbital data; the
# groups fetched come from .github/brahe-data-manifest.txt.
refresh-celestrak-snapshots *args: _setup
    @{{python}} {{scripts_dir}}/refresh_celestrak_snapshots.py {{args}}

# Test all documentation examples (delegates to scripts/test_examples.py)
test-examples *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_examples.py {{args}}

# Test a specific example (delegates to scripts/test_example.py)
test-example *args: _setup
    @PYTHONPATH={{scripts_dir}} {{python}} {{scripts_dir}}/test_example.py {{args}}

# ───── Coverage ─────

# No Rust instrumentation — faster than `test-python` when you only need `.py`
# line coverage.

# Run Python tests with Python-only coverage (no Rust instrumentation)
coverage-python: _setup
    uv pip install -e . --quiet
    {{python}} -m pytest tests/ --cov=brahe --cov-report=html --cov-report=xml:python-coverage.xml --cov-report=term

# ───── Profiling ─────

# One-time local development setup: uv venv (Python 3.14) + dev deps,
# pre-commit hooks, and the major SPICE kernels tests expect to be cached
# (de440s, moon_pa_de440, mar099s). Idempotent: kernel downloads fast-path
# when already cached.
setup:
    @which uv > /dev/null 2>&1 || { echo "Error: uv is not installed. Install from https://docs.astral.sh/uv/"; exit 1; }
    uv venv --python 3.14 --allow-existing
    uv sync --group dev
    uv pip install -e . --quiet
    .venv/bin/pre-commit install
    @just download-resources
    @echo "✓ Setup complete (.venv ready, pre-commit hooks installed, data cached)"

# ───── Code Quality ─────

# Format all code (Rust + Python)
format: _setup
    cargo fmt
    uv run ruff format

# Check formatting without changes
format-check: _setup
    cargo fmt -- --check
    uv run ruff format --check

# Run linters (clippy + ruff + test labels)
lint: _setup
    cargo clippy --workspace --all-targets --all-features -- -D warnings
    uv run ruff check
    uv run scripts/check_test_labels.py

# Run linters with auto-fix
lint-fix: _setup
    cargo clippy --workspace --all-targets --all-features --fix --allow-dirty -- -D warnings
    uv run ruff check --fix
    uv run scripts/check_test_labels.py

# Check every Rust test carries #[serial] or #[parallel]
check-test-labels: _setup
    uv run scripts/check_test_labels.py

# ───── Documentation ─────

# Build documentation
docs: _setup
    ./scripts/generate_stubs.sh
    uv run zensical build

# Serve documentation locally
docs-serve: _setup
    ./scripts/generate_stubs.sh
    uv run zensical serve

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
# and mirror it into CITATION.cff
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
    new, count = re.subn(
        r'(\[workspace\.package\][^\[]*?\nversion = )\"[^\"]+\"',
        r'\\g<1>\"{{version}}\"',
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count == 0:
        raise SystemExit('error: could not find [workspace.package] version in Cargo.toml')
    p.write_text(new)
    "
    # Refresh Cargo.lock so the workspace version bump is reflected there too.
    cargo update --workspace --quiet
    # Mirror the version into CITATION.cff so citation metadata tracks the release.
    python3 -c "
    import re, pathlib
    p = pathlib.Path('CITATION.cff')
    text = p.read_text()
    new, count = re.subn(
        r'(?m)^version: \".*\"$',
        'version: \"{{version}}\"',
        text,
        count=1,
    )
    if count == 0:
        raise SystemExit('error: could not find version field in CITATION.cff')
    p.write_text(new)
    "
    echo "✓ Set workspace version to {{version}} (Cargo.toml, Cargo.lock, CITATION.cff)"

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
check: test test-python-docs lint format-check stubs docs
    @echo "✓ All quality checks passed!"
