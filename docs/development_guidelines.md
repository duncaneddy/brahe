# Development Guidelines

## Development Workflow

For all development we recommend using [uv](https://uv.sh/) to manage your environment.
The guidelines for contributing, developing, and extending brahe assume you are using uv.

### Setting up your environment

If you need to setup the development environment, including installing the necessary
development dependencies.

First, you need to install Rust from [rustup.rs](https://rustup.rs/).

Then you can install the nightly toolchain with:

```bash
rustup toolchain install nightly
rustup default nightly
```

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

**Run all tests:**
```bash
make test
```

**Individual test suites:**
```bash
make test-rust          # Rust tests only
make test-python        # Python tests only
make test-examples      # Documentation examples (warn on parity issues)
```

**Pre-ship validation** (runs all tests, formatters, linters, and doc builds):
```bash
make ship-tests
```

### Development Workflow: Implementing a New Feature

When adding new functionality to Brahe, follow this sequence:

**1. Rust Implementation**
- Implement functionality in the appropriate module under `src/`
- Use SI base units (meters, radians, seconds) in all public APIs
- Follow existing patterns and naming conventions

**2. Rust Tests**
- Write comprehensive unit tests in the same file (in `#[cfg(test)] mod tests`)
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
- Test: `make test-examples`

**6. Documentation**
- Update or create documentation in `docs/`
- Reference examples using snippet includes (see template below)
- Build: `make build-docs`
- Preview: `make serve-docs`

**7. Quality Checks**
```bash
make format      # Auto-format code
make lint        # Check for issues
make ship-tests  # Full validation
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
uv run make.py test-examples
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
uv run make.py make-plots
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

### Automatic Changelog Generation

When you create a pull request, you must fill in the changelog section in the PR description. The changelog uses [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format with four categories:

- **Added** - New features
- **Changed** - Changes to existing functionality
- **Fixed** - Bug fixes
- **Removed** - Removed features or functionality

### How It Works

1. **Fill in PR description**: When opening a PR, add entries under the appropriate changelog section(s)
   ```markdown
   ### Fixed
   - Fixed memory leak in trajectory interpolation
   - Corrected EOP data loading for edge cases
   ```

2. **Validation**: A GitHub Action checks that at least one changelog section has entries
   - PR will fail validation if all sections are empty
   - You'll receive a comment with instructions if validation fails

3. **Automatic fragment creation**: When the PR is merged:
   - A GitHub Action parses your changelog entries
   - Creates fragment files in `news/` directory (e.g., `123.added.md`, `123.fixed.md`)
   - Commits the fragments to the main branch

4. **Release compilation**: During release:
   - Towncrier collects all fragments from `news/`
   - Generates formatted release notes
   - Updates `CHANGELOG.md` with the new version section
   - Deletes fragment files

### Example PR Changelog

```markdown
## Changelog

### Added
- Support for new SGP4 propagation mode
- EOP data caching to improve performance

### Fixed
- Memory leak in trajectory interpolation
- Edge case in geodetic coordinate conversion
```

This will automatically create:
- `news/123.added.md` with both Added items
- `news/123.fixed.md` with both Fixed items

### Manual Fragment Creation (Rare)

In rare cases where you need to create fragments manually, see `news/README.md` for instructions. Fragment files use the format `<PR#>.<type>.md` where type is one of: `added`, `changed`, `fixed`, `removed`.

### Previewing the Changelog

To see what changelog fragments are currently queued:

```bash
# List all fragment files
ls -la news/*.md

# Or see just the fragment names
find news/ -name '*.md' ! -name '.template.md' ! -name 'README.md'
```

To see what the next release changelog would look like without making changes:

```bash
# Preview the changelog for the next release
uv run towncrier build --version 1.2.3 --draft
```

This shows the formatted output without modifying `CHANGELOG.md` or deleting fragments.

### Releases Without Changelog Fragments

If you create a release when there are no changelog fragments in `news/`:
- The release workflow will succeed
- A minimal release will be created with "No significant changes"
- This is useful for releases that only contain dependency updates or internal changes

## Release Process

### Initiating a Release

Before creating a release:

1. **Update version** in `Cargo.toml`:
   ```bash
   # Edit version in Cargo.toml
   vim Cargo.toml  # Update version = "1.2.3"
   ```

2. **Run quality checks**:
   ```bash
   ruff check && cargo fmt -- --check && cargo test && uv pip install -e ".[all]" && uv run pytest && uv run make.py test-examples && uv run make.py make-plots && uv run mkdocs build --strict
   ```

3. **Push version tag**:
   ```bash
   git add Cargo.toml
   git commit -m "Prepare release v1.2.3"
   git push origin main
   git tag v1.2.3
   git push origin v1.2.3
   ```

### Automated Workflow

**Note**: Changelog fragments are automatically created from PR descriptions. You don't need to manually create fragment files.

Once the tag is pushed, GitHub Actions automatically:

1. Validates version matches between tag and `Cargo.toml`
2. Runs all tests (Rust, Python, examples)
3. Generates release notes with towncrier (commits CHANGELOG.md)
4. Builds documentation and deploys to GitHub Pages
5. Builds Python wheels and source distribution
6. Publishes to PyPI and crates.io
7. Creates **draft** GitHub Release with artifacts and release notes
8. Updates "latest" tag and release

### Completing the Release

After automation completes:

1. **Review draft release** at `https://github.com/duncaneddy/brahe/releases`
2. **Edit release notes** (optional):
   - Add highlights or breaking changes
   - Include migration notes if needed
3. **Publish release** by clicking "Publish release"

### Verification

After publishing, verify:

- PyPI: [https://pypi.org/project/brahe/](https://pypi.org/project/brahe/)
- Crates.io: [https://crates.io/crates/brahe](https://crates.io/crates/brahe)
- Docs: [https://duncaneddy.github.io/brahe/latest/](https://duncaneddy.github.io/brahe/latest/)
- GitHub: [https://github.com/duncaneddy/brahe/releases](https://github.com/duncaneddy/brahe/releases)
