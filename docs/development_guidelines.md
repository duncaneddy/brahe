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

Use the `pymdownx.snippets` directive to include examples in markdown files:

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
