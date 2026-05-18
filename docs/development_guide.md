# Development Guide

## Development Workflow

For all development we recommend using [uv](https://uv.sh/) to manage your environment.
The guidelines for contributing, developing, and extending brahe assume you are using uv.

### Setting up your environment

If you need to setup the development environment, including installing the necessary development dependencies.

First, you need to install Rust from [rustup.rs](https://rustup.rs/).

After this you can now setup your python environment with:

```bash
uv sync --dev
```

Finally, you can install the pre-commit hooks with:

```bash
uv run pre-commit install
```

### Testing

The package includes Rust tests, Python tests, and documentation example tests.

```bash
# Run Rust tests
cargo test
# Run Python tests
uv pip install -e ".[all]" && uv run pytest
# Run documentation examples
just test-examples
# Test specific example
just test-example <example_name> # Can just be the file name without extension, e.g. impulsive_maneuver or impulsive_maneuver.py
```


### Development Workflow: Implementing a New Feature

When adding new functionality to Brahe, follow this sequence:

**1. Rust Implementation**
- Implement functionality in the appropriate module under `src/`
- Use SI base units (meters, seconds) in all public APIs
- Follow existing patterns and naming conventions

**2. Rust Tests**
- Write comprehensive unit tests in the same file (in a `#[cfg(test)] mod tests {}` module)
- Test edge cases and typical use cases
- Run: `cargo test`
- Ensure all tests pass before proceeding

**3. Python Bindings**
- Create 1:1 Python bindings in `src/pymodule/`
- Use identical function names and parameter names as Rust
- Add complete Google-style docstrings with Args, Returns, Examples
- Export new classes in `src/pymodule/mod.rs`
- Export in Python package (`brahe/*.py` files)
- Reinstall: `uv pip install -e .`

**4. Python Tests**
- Write Python tests that mirror Rust tests in `tests/`
- Follow the same test structure and assertions
- Run: `uv run pytest tests/ -v`

**5. Documentation Examples**
- Create standalone example files in `examples/<module>/`
- Create both Python and Rust versions (see templates below)
- Test: `just test-examples`

**6. Documentation**
- Update or create documentation in `docs/`
- Reference examples using snippet includes (see template below)
- Build Locally: `uv run properdocs serve`

**7. Final Checks**
```bash
# Formatting
cargo fmt
ruff check --fix
ruff format
# Tests
cargo test
uv pip install -e ".[all]" && ./scripts/generate_stubs.sh && uv run pytest
# Documentation
just test-examples
just make-plots
uv run properdocs build --strict
uv run properdocs serve
```

## Rust Standards and Guidelines


### Rust Testing Conventions

New functions implemented in rust are expected to have unit tests and documentation tests. Unit tests should cover
all edge cases and typical use cases for the function. Documentation tests should provide examples of how to use the function.

Unit tests should be placed in the same file as the function they are testing, in a module named `tests`. The names of tests should follow the general convention of `test_<struct>_<trait>_<method>_<case>` or `test_<function>_<case>`.

### Rust Docstring Template

New functions implemented in rust are expected to use the following docstring to standardize information on functions to
enable users to more easily navigate and learn the library.

```markdown
{{ Function Description }}

## Arguments

* `argument_name`: {{ Arugment description}}. Units: {{ Optional, Units as (value). e.g. (rad) or (deg)}}

## Returns

* `value_name`: {{ Value description}}. Units: {{ Optional, Units as (value). e.g. (rad) or (deg)}}

## Examples
\`\`\`
{{ Implement shor function in language }}
\`\`\`

## References:
1. {{ author, *title/journal*, pp. page_number, eq. equation_number, year}}
2. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 24, eq. 2.43 & 2.44, 2012.
```

### Python Standards and Guidelines

#### Python Testing Conventions

Python tests should be placed in the `tests` directory. The test structure and names should mirror the structure of the `brahe` package. For example, tests for `brahe.orbits.keplerian` should be placed in `tests/orbits/test_keplerian.py`.

All Python tests should be exact mirrors of the Rust tests, ensuring that both implementations are equivalent and consistent. There are a few exceptions to this rule, such as tests that check for Python-specific functionality or behavior, or capabilities that are not possible to reproduce in Python due to language limitations.

## Documentation Examples

Documentation examples are standalone executable files that demonstrate library functionality. Every example must exist in both Python and Rust versions to ensure API parity.

### Example File Structure

Examples are organized by module in `examples/`:
```
examples/
├── time/           # Time system examples
├── orbits/         # Orbital mechanics examples
├── coordinates/    # Coordinate transformation examples
├── frames/         # Reference frame examples
├── attitude/       # Attitude representation examples
├── eop/            # Earth orientation parameter examples
├── trajectories/   # Trajectory examples
└── workflows/      # Complete workflow examples
```

### Naming Convention

Example files should follow this pattern:
```
<module>_<functionality>_<description>.{py,rs}
```

Examples:
- `time_epoch_creation.py` / `time_epoch_creation.rs`
- `orbits_keplerian_conversion.py` / `orbits_keplerian_conversion.rs`
- `coordinates_geodetic_transform.py` / `coordinates_geodetic_transform.rs`

### Python Example Template

See `examples/TEMPLATE.py`:

```python
# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Brief description of what this example demonstrates.
"""
import brahe as bh
import pytest

if __name__ == '__main__':
    # Setup: Define any input parameters
    value = 1.0

    # Action: Demonstrate the functionality
    result = value * 2.0  # Replace with actual brahe function call

    # Validation: Assert the result is correct
    expected = 2.0
    assert result == pytest.approx(expected, abs=1e-10)

    print("✓ Example validated successfully!")
```

**Note**: The `# /// script` header makes this a [uv script](https://docs.astral.sh/uv/guides/scripts/), allowing it to be run standalone with `uv run example.py`.

### Rust Example Template

See `examples/TEMPLATE.rs`:

```rust
//! Brief description of what this example demonstrates.

use approx::assert_abs_diff_eq;
use brahe::time::{Epoch, TimeSystem};

fn main() {
    // Setup: Define any input parameters
    let value = 1.0;

    // Action: Demonstrate the functionality
    let result = value * 2.0; // Replace with actual brahe function call

    // Validation: Assert the result is correct
    let expected = 2.0;
    assert_abs_diff_eq!(result, expected, epsilon = 1e-10);

    println!("✓ Example validated successfully!");
}
```

### Testing Examples

Test examples locally:
```bash
just test-examples
```

The build system will:
1. Execute all `.rs` files via `rust-script`
2. Execute all `.py` files via `uv run python`
3. Verify every `.rs` has a matching `.py` (and vice versa)
4. Report pass/fail for each example

### Including Examples in Documentation

Use the `pymdownx.snippets` directive to include examples in markdown files. See the [snippets plugin documentation](https://facelessuser.github.io/pymdown-extensions/extensions/snippets/) for additional details on usage.

```markdown
## Example: Creating Epochs

=== "Python"

    ``` python
    --8<-- "../examples/time/epoch_creation.py"
    ```

=== "Rust"

    ``` rust
    --8<-- "../examples/time/epoch_creation.rs"
    ```
```

This will:
- Create tabbed interface with Python shown first
- Include the actual file contents (always in sync)
- Automatically update when examples change

## Documentation Plots

Interactive plots are generated from Python scripts in `plots/` and embedded in documentation.

### Plot Naming Convention

Plot files should follow this pattern:
```
fig_<description>.py
```

Examples:
- `fig_time_system_offsets.py`
- `fig_orbital_period.py`
- `fig_anomaly_conversions.py`

### Plot Template

See `plots/TEMPLATE_plot.py`:

```python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Brief description of what this plot visualizes.
"""
import os
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import brahe as bh
import numpy as np

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/")
OUTFILE = f"{OUTDIR}/{SCRIPT_NAME}.html"

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)  # Replace with actual data

# Create figure
fig = go.Figure()
fig.update_layout(
    title="Plot Title",
    xaxis_title="X Axis Label",
    yaxis_title="Y Axis Label",
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent for dark mode
    plot_bgcolor='rgba(0,0,0,0)'
)

# Add traces
fig.add_trace(go.Scatter(x=x, y=y, name="Data", mode='lines'))

# Write HTML (partial, not full page)
pio.write_html(
    fig,
    file=OUTFILE,
    include_plotlyjs='cdn',
    full_html=False,
    auto_play=False
)

print(f"✓ Generated {OUTFILE}")
```

**Note**: The `# /// script` header allows standalone execution with `uv run fig_plot.py`.

### Generating Plots

Generate all plots:
```bash
just make-plots
```

Plots are written to `docs/figures/` as partial HTML files for embedding.

### Including Plots in Documentation

```markdown
## Time System Offsets

The following plot shows time system offsets from UTC:

--8<-- "./docs/figures/fig_time_system_offsets.html"

??? "Plot Source"

    ``` python title="fig_time_system_offsets.py"
    --8<-- "../plots/fig_time_system_offsets.py"
    ```
```

This will:
- Embed the interactive Plotly plot
- Add a collapsible section showing the source code

## Pull Request Changelog

When you open a pull request, fill in the `## Changelog` section of the PR description with entries under the appropriate [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) headings:

- **Added** - new features
- **Changed** - changes to existing functionality
- **Deprecated** - APIs still present but scheduled for removal
- **Removed** - APIs that have been removed
- **Fixed** - bug fixes

A single PR may contribute to multiple sections. The PR description is the single source of truth — there is no separate fragment file to maintain.

### Example

```markdown
## Changelog

### Added
- Support for new SGP4 propagation mode

### Fixed
- Memory leak in trajectory interpolation
- Edge case in geodetic coordinate conversion
```

### How It Works

1. **Validation on open**: a GitHub Action checks that the PR description has at least one non-empty section. It posts a comment with instructions if validation fails. Dependabot PRs are exempt.
2. **At release time**: `scripts/generate_release_notes.py` walks every PR merged into `main` since the previous release tag, parses each PR's `### Section` blocks, aggregates them under the version heading in `CHANGELOG.md`, and writes the same content to `release_notes.md` for use as the GitHub Release body. Each entry is attributed as `[@author](url) ([#PR](url))`.
3. **Skipped PRs**: PRs labeled `automated`, `data-update`, or `dependencies`, and any PR opened by a bot account, are excluded from the generated changelog.

### Previewing the Changelog Locally

To see what the next release's changelog would look like without writing any files:

```bash
python3 scripts/generate_release_notes.py \
    --version 1.5.1 \
    --prev-tag v1.5.0 \
    --dry-run
```

This requires the `gh` CLI to be authenticated (`gh auth status`).

## Release Process

CHANGELOG generation and version bumps happen **locally before tagging**, so the tagged commit contains everything the published artifacts ship. CI never mutates the repository during a release.

### Initiating a Release

1. **Bump the workspace version**:
   ```bash
   just set-version 1.2.3
   ```
   This updates `[workspace.package].version` in `Cargo.toml` (inherited by `brahe` and `brahe-py`) and refreshes `Cargo.lock`.

2. **Regenerate the CHANGELOG entry** for this release:
   ```bash
   just generate-changelog
   ```
   By default, the version is read from `Cargo.toml` and the previous tag from `git describe --tags --abbrev=0`. Override either with `just generate-changelog 1.2.3 v1.2.2`. The script aggregates `### Section` blocks from PR bodies merged since the previous tag — `gh` must be authenticated (`gh auth status`).

   Review the diff to `CHANGELOG.md` and edit if needed; it is the canonical source of release notes.

3. **Run quality checks**:
   ```bash
   just check
   ```

4. **Commit and tag**:
   ```bash
   git add Cargo.toml Cargo.lock CHANGELOG.md
   git commit -m "Prepare release v1.2.3"
   git push origin main
   git tag v1.2.3
   git push origin v1.2.3
   ```

### Automated Workflow

Once the tag is pushed, GitHub Actions automatically:

1. Validates the tag version matches `Cargo.toml` **and** that `CHANGELOG.md` contains a `## [1.2.3]` entry (fails fast if `just generate-changelog` was skipped).
2. Runs all tests (Rust, Python, examples).
3. Extracts `release_notes.md` from the committed `CHANGELOG.md` (via `scripts/extract_release_notes.py`) for use as the GitHub Release body — no commits or pushes from CI.
4. Builds documentation and deploys to GitHub Pages.
5. Builds Python wheels and source distribution.
6. Publishes to PyPI and crates.io.
7. Publishes the GitHub Release (non-draft) with artifacts and release notes.
8. Updates the "latest" tag and release.

### Verification

After publishing, verify:

- PyPI: [https://pypi.org/project/brahe/](https://pypi.org/project/brahe/)
- Crates.io: [https://crates.io/crates/brahe](https://crates.io/crates/brahe)
- Docs: [https://duncaneddy.github.io/brahe/latest/](https://duncaneddy.github.io/brahe/latest/)
- GitHub: [https://github.com/duncaneddy/brahe/releases](https://github.com/duncaneddy/brahe/releases)

## Benchmarks

Brahe has two benchmark layers: **Criterion micro-benchmarks** for internal Rust performance regression testing, and a **comparative benchmark framework** that measures both runtime performance and numerical accuracy across Python (Brahe), Rust (Brahe), and Java (OreKit).

### Criterion Micro-Benchmarks

These are standard Rust benchmarks using the [Criterion](https://bheisler.github.io/criterion.rs/book/) harness, located in `benchmarks/`:

```bash
# Run all Criterion benchmarks
just bench

# Run specific benchmark suites
just bench-providers
just bench-propagators
```

Criterion generates HTML reports in `target/criterion/` with statistical analysis, regression detection, and timing distributions.

### Comparative Benchmark Framework

The comparative framework lives in `benchmarks/comparative/` and compares equivalent implementations across languages using a standardized JSON stdin/stdout protocol. Each language implementation is a standalone process that receives task parameters as JSON and returns timing data and numerical results.

#### Setup

Before running comparative benchmarks, install all dependencies with a single command:

```bash
just bench-compare-setup
```

This builds the Rust benchmark binary, builds the Java/Gradle project (generating a Gradle wrapper if needed), and downloads OreKit data to `~/.orekit/orekit-data`.

**Prerequisites:**

- **Rust**: Install from [rustup.rs](https://rustup.rs/) (used for the Rust benchmark binary)
- **JDK 17+**: Install via `brew install openjdk` (macOS) or your system package manager (used for Java/OreKit benchmarks)
- **Gradle**: Install via `brew install gradle` (macOS) or `sdk install gradle` via [SDKMAN](https://sdkman.io/) (Linux). Only needed if the Gradle wrapper doesn't exist yet — after first setup, `gradlew` is committed and Gradle is no longer required.

You can override the OreKit data location with the `OREKIT_DATA` environment variable.

#### Running Benchmarks

```bash
# List available benchmark tasks
just bench-compare-list

# Run all tasks across all available languages
just bench-compare

# Run with specific options
just bench-compare --iterations 100 --seed 42
just bench-compare --module coordinates --language python
just bench-compare --task orbits.keplerian_to_cartesian

# Generate plots from the latest run
just bench-compare-plot
```

#### Output

Each run prints two Rich tables to the console:

- **Performance Comparison** — mean, median, std, min, max per task per language, with speedup ratios relative to the OreKit (Java) baseline.
- **Numerical Accuracy (vs OreKit baseline)** — max absolute error, max relative error, and RMS error for each implementation compared against OreKit.

Results are saved as JSON to `benchmarks/comparative/results/` (gitignored). Plots are generated as themed Plotly HTML to `docs/figures/`.

#### Architecture

```
benchmarks/comparative/
    runner.py              # Typer CLI orchestrator
    config.py              # Defaults, paths, system info collection
    registry.py            # Task discovery and filtering
    results.py             # Result dataclasses and JSON serialization
    reporting.py           # Rich console table formatting
    plotting.py            # Plotly charts with brahe_theme
    tasks/
        base.py            # BenchmarkTask ABC
        coordinates_tasks.py
        orbits_tasks.py
    implementations/
        python/            # Brahe Python — called in-process
        rust/              # Brahe Rust — standalone binary, JSON protocol
        java/              # OreKit Java — Gradle project, JSON protocol
```

The orchestrator dispatches each task to each language. Python implementations run in-process. Rust and Java implementations are invoked as subprocesses with JSON piped to stdin and results read from stdout.

#### Adding a New Benchmark Task

To add a new benchmark task (e.g., a frame transformation benchmark):

**1. Define the task specification** in `benchmarks/comparative/tasks/`:

```python
# benchmarks/comparative/tasks/frames_tasks.py
from benchmarks.comparative.tasks.base import BenchmarkTask

class EciToEcefTask(BenchmarkTask):
    @property
    def name(self) -> str:
        return "frames.eci_to_ecef"

    @property
    def module(self) -> str:
        return "frames"

    @property
    def description(self) -> str:
        return "Transform ECI position to ECEF"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust"]  # list languages you'll implement

    def generate_params(self, seed: int) -> dict:
        """Generate deterministic test parameters."""
        import random
        rng = random.Random(seed)
        # Generate test data...
        return {"states": [...], "epoch_mjd": 60000.0}
```

**2. Register the task** in `benchmarks/comparative/tasks/__init__.py`:

```python
from benchmarks.comparative.tasks.frames_tasks import EciToEcefTask

ALL_TASKS = [
    # ... existing tasks ...
    EciToEcefTask(),
]
```

**3. Add the Python implementation** in `benchmarks/comparative/implementations/python/`:

```python
# benchmarks/comparative/implementations/python/frames.py
import numpy as np
import brahe
from benchmarks.comparative.implementations.python.base import ensure_eop, time_iterations
from benchmarks.comparative.results import TaskResult

def eci_to_ecef(params: dict, iterations: int) -> TaskResult:
    ensure_eop()
    # ... implementation that calls brahe functions and times them ...
    times, results = time_iterations(run, iterations)
    return TaskResult(task_name="frames.eci_to_ecef", ...)
```

Register the function in `implementations/python/__init__.py` by adding it to `_DISPATCH_TABLE`.

**4. Add the Rust implementation** in `benchmarks/comparative/implementations/rust/src/`:

Create the module file (e.g., `frames.rs`) with functions that deserialize JSON params, run the benchmark loop with `std::time::Instant`, and return `(Vec<f64>, serde_json::Value)`. Add the module and dispatch arm in `main.rs`.

**5. (Optional) Add the Java/OreKit implementation** following the same pattern in the Gradle project.

#### Key Design Decisions

- **OreKit as baseline**: Java/OreKit is the reference implementation for both performance speedup ratios and numerical accuracy comparisons. OreKit runs first for each task, and all other implementations are compared against it.
- **Deterministic parameters**: `generate_params(seed)` ensures reproducible benchmarks across runs. Always use the seed to initialize your RNG.
- **JSON protocol**: Language implementations are decoupled from the orchestrator. Any language that can read JSON from stdin and write JSON to stdout can participate.
- **First-iteration results**: Only the first iteration's numerical results are stored for accuracy comparison. All iterations contribute timing data.
- **Angle normalization**: Orbital element comparisons normalize angular differences modulo 360 degrees to handle different library conventions for angle ranges.
- **EOP initialization**: Benchmarks use `StaticEOPProvider.from_zero()` (zero EOP values) to avoid file I/O overhead and ensure reproducibility. This is sufficient for coordinate and orbital element conversions that don't depend on Earth orientation.
