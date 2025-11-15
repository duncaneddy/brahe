# Brahe Development Guidelines for Claude

## 🚨 CRITICAL DEVELOPMENT RULES

### Testing Requirements (MANDATORY)
- **ALWAYS run `cargo test` after ANY Rust changes** - All tests must pass
- **ALWAYS run `uv pip install -e . && .venv/bin/python -m pytest tests/ -v` after changes**
- **NEVER consider a task complete until ALL quality checks pass**:
  - ✅ Rust tests pass (`cargo test`)
  - ✅ Python tests pass (`.venv/bin/python -m pytest tests/ -v`)
  - ✅ No Rust warnings (`cargo clippy --all-targets --all-features -- -D warnings`)
  - ✅ No Python warnings (`ruff check`)
  - ✅ Documentation builds (`.venv/bin/mkdocs build`)
  - ✅ Type stubs current (`./scripts/generate_stubs.sh`)

### Test Naming Convention
- **Standard format**: `test_<functionality>` (e.g., `test_epoch_creation`, `test_tle_parsing`)
- **Trait implementations**: `test_<StructName>_<TraitName>_<MethodName>` (e.g., `test_strajectory_trajectory_new`)
- **Test parity**: Every Rust test should have a corresponding Python test
- **Python test structure** must mirror Rust module file structure
- **Test tools**:
  - Floating-point: Use `assert_abs_diff_eq!` (Rust) or `pytest.approx()` (Python)
  - Parameterized: Use `rstest` with `#[rstest]` and `#[case]`
  - EOP required: Call `setup_global_test_eop()` at test start

### Refactoring Requirements
- **NEVER remove/change tests without asking** - Discuss with user first
- **REMOVE legacy code** - Delete old code, tests, and Python bindings when refactoring
- **UPDATE Python bindings to match Rust changes**:
  1. Remove old Python bindings that no longer match
  2. Create new bindings that exactly mirror Rust API
  3. Never maintain legacy interfaces for compatibility
  4. Python API = 1:1 translation of Rust API

## Documentation Guidelines

### Diataxis Framework
Brahe uses [Diataxis](https://diataxis.fr/) with four documentation types:

**1. Learn (`docs/learn/`)** - EXPLANATION (Understanding-oriented)
- **Purpose**: Explain concepts, design decisions, the "why"
- **Content**: Background, context, trade-offs, comparisons
- **Example**: "What are Earth Orientation Parameters and why do they matter?"

**2. Examples (`docs/examples/`)** - HOW-TO GUIDES (Goal-oriented)
- **Purpose**: Show how to accomplish specific tasks
- **Content**: Complete workflows, step-by-step instructions
- **Example**: "How to compute ground station contacts"

**3. Library API Reference (`docs/library_api/`)** - REFERENCE (Information-oriented)
- **Purpose**: Technical API descriptions
- **Content**: Auto-generated from docstrings using `::: brahe.ClassName`
- **Style**: Austere, factual, objective - "describe and only describe"

**4. Tutorials** (Future) - TUTORIALS (Learning-oriented)
- Currently: Basic content in main index.md

### Docstring Standards (CRITICAL)

**Every new/modified function MUST have complete documentation:**

**Rust**: Provide rustdoc comments with description, parameters, returns, examples
```rust
/// Function description here.
///
/// # Arguments
///
/// * `param1` - Description. Units: (m)
///
/// # Returns
///
/// * `result` - Description. Units: (s)
///
/// # Examples
/// ```
/// use brahe::module::function_name;
/// let result = function_name(param1);
/// ```
```

**Python bindings** (`src/pymodule/*.rs`): Google-style docstrings
```rust
/// Function description here.
///
/// Args:
///     param1 (type): Description
///
/// Returns:
///     return_type: Description
///
/// Example:
///     ```python
///     import brahe as bh
///     result = bh.function_name(param1)
///     ```
```

**CRITICAL Return type format**: Must be `Type: Description` (e.g., `Any: Property value if found`)
- **WRONG**: `Property value if found` (parser sees "Property" as type!)

### Example Documentation Rules

**Good Documentation**: 
- `docs/learn/time/epoch.md` is a documentation file that is a good example of `Learn` content.
- `docs/examples/ground_contacts.md` and `docs/examples/doppler_compensation.md` are good reference `Examples` content.

**Functions/Classes/Methods**: Examples MUST be in Python binding docstrings (`src/pymodule/*.rs`)
- Examples in docstrings auto-render as collapsible boxes in docs
- Markdown reference pages should ONLY contain `::: brahe.ClassName` directive

**Module Constants**: Examples MUST be in markdown files (constants can't have docstring examples)

**Complete Workflows**: Multi-function examples belong in `docs/examples/` as How-To Guides

**Standalone Example Files (CRITICAL)**:
Code examples in Learn documentation (`docs/learn/`) MUST be implemented as standalone executable files:

- **Location**: `examples/<topic>/` (e.g., `examples/time/`, `examples/access/properties/`)
- **File format**:
  - Python: `# /// script` header with dependencies
  - Rust: `//! ```cargo` header with dependencies (no shebang)
- **Content**: Complete, runnable code with expected output as comments
- **Documentation inclusion**: Use `--8<--` directive to include in markdown
  ```markdown
  === "Python"
      ``` python
      --8<-- "./examples/time/epoch_datetime.py:8"
      ```

  === "Rust"
      ``` rust
      --8<-- "./examples/time/epoch_datetime.rs:4"
      ```
  ```
- **Line numbers**:
  - Python: Start at line 8 (skips header + docstring)
  - Rust: Start at line 4 (skips header)
- **One example per file**: Do NOT reuse snippets from same file across multiple documentation sections
- **Exception**: Small configuration snippets or pure illustration (e.g., showing enum variants) can be inline

**Example file structure (Python)**:
```python
# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Brief description of what this example demonstrates
"""

import brahe as bh

bh.initialize_eop()

# Example code here
result = bh.some_function()
print(f"Result: {result}")
# Result: expected output here
```

**Example file structure (Rust)**:
```rust
//! ```cargo
//! [dependencies]
//! brahe = { path = "../../.." }
//! ```

use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Example code here
    let result = bh::some_function();
    println!("Result: {:?}", result);
    // Result: expected output here
}
```

**Example Writing Standards**:
- **Use constants**: `bh.R_EARTH + 500e3` not `6878000.0`
- **SI base units**: meters, seconds (not km, minutes)
- **Standard orbital parameters**:
  - LEO: `bh.R_EARTH + 500e3`
  - GEO: `bh.R_EARTH + 35786e3`
  - Eccentricity: `0.01` for near-circular
  - Inclination: Realistic degrees
- **Get states from conversions**: Define Keplerian elements, convert to Cartesian
  ```python
  oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
  state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
  ```
- **Use LaTeX over Unicode**: For markdown documentation in docs/* use LaTeX syntax $\mu$ not Unicode 'μ'

### Documentation Build Process
```bash
# Regenerate stubs after Python binding changes
./scripts/generate_stubs.sh

# Build documentation
.venv/bin/mkdocs build

# Expected: ~407 griffe warnings about parameter names (normal - PyO3 limitation)
# Fix: Missing return types in .pyi, missing docstrings in Rust code, broken links
```

## Code Quality Standards

### Core Conventions
- **Orbital elements order**: Always `[a, e, i, raan, argp, anomaly]` (mean anomaly unless specified)
- **Units**: Semi-major axis in meters, angles in radians/degrees
- **SI base units ALWAYS**: meters, meters/second, seconds in ALL public functions
  - **NEVER use km, km/s** in public APIs
  - Exception: `AngleFormat` enum allows explicit degree input
  - Examples and access functions should use degrees for usability
- **Minimize unnecessary conversions**: Keep internal calculations in SI units
- **Geodetic lon/lat order**: ALWAYS `(longitude, latitude, altitude)` for `PointLocation`, inputs are always in degrees
- **Use existing library functions**: Check `time::conversions`, `coordinates`, `orbits::keplerian`, `frames` before implementing
- **Clean professional code**: No AI assistance comments or correction indicators

### Python Bindings Philosophy
**1:1 mirror of Rust functionality. No legacy compatibility layers.**

**Implementation Requirements**:
- **CRITICAL: ALWAYS run `uv pip install -e .` after ANY Rust changes** before testing Python
- **Module structure**: ALL imports defined in `pymodule/mod.rs` (PyO3 constraint)
- **Complete docstrings**: Args, Returns, Examples with proper type annotations
- **Export process**: Add to `mod.rs` (`module.add_class::<PyTypeName>()?`) AND `brahe/*.py` (`__all__` list)

**Refactoring workflow**:
1. Update Rust implementation + tests
2. **DELETE old Python bindings**
3. Create new Python bindings (1:1 with Rust)
4. Export in `mod.rs` and `brahe/*.py`
5. Update Python tests
6. Regenerate stubs
7. Update documentation (Reference, Learn, Examples as needed)
8. **DO NOT STOP EARLY** - Complete the full task

## Development Workflow

**Standard Process**: Rust Implementation → Rust Tests → Python Bindings → Python Tests → Quality Checks

### Testing Examples

**CRITICAL**: All examples in `examples/` must be tested before committing.

**Python examples** (`.py` files with `# /// script` header):
```bash
# Test a single Python example
uv run python make.py test-example <category>/<example_name> --lang python

# Examples:
uv run python make.py test-example coordinates/geodetic_conversion --lang python
uv run python make.py test-example access/basic_workflow --lang python
```

**Rust examples** (`.rs` files):
```bash
# Test a single Rust example
uv run python make.py test-example <category>/<example_name> --lang rust

# Examples:
uv run python make.py test-example coordinates/geodetic_conversion --lang rust
uv run python make.py test-example access/basic_workflow --lang rust
```

**Rust example requirements**:
- Use `nalgebra as na` for creating vectors (e.g., `na::SVector::<f64, 6>::new(...)`)
- Use `AngleFormat::Degrees` or `AngleFormat::Radians` (PascalCase, not UPPERCASE)
- Call `bh::initialize_eop().unwrap()` at the start of `main()`
- Include expected output as comments at the end
- Follow same structure as existing examples in `examples/coordinates/`

**Example structure**:
```rust
//! Brief description of what the example demonstrates

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define inputs using nalgebra vectors for 6D state vectors
    let state = na::SVector::<f64, 6>::new(x, y, z, vx, vy, vz);

    // Perform operations
    let result = bh::some_function(state, bh::AngleFormat::Degrees);

    // Print results
    println!("Result: {:.3}", result[0]);
    // Expected output:
    // Result: 1234.567
}
```

### Quick Quality Check
```bash
cargo test && \
cargo fmt -- --check && \
cargo clippy --all-targets --all-features -- -D warnings && \
uv pip install -e . && \
.venv/bin/python -m pytest tests/ -v && \
ruff format --check && \
ruff check && \
./scripts/generate_stubs.sh && \
.venv/bin/mkdocs build
```

## Release Process

### Overview

Brahe uses an automated release workflow that publishes to PyPI, crates.io, and GitHub Releases. The workflow is triggered by pushing a version tag (e.g., `v1.2.3`) and includes automated testing, documentation deployment, and artifact publishing.

### Pre-Release Checklist

Before creating a release, ensure:

1. **Version is synchronized**:
   - Update `Cargo.toml` version field to match the planned release version
   - Python version in `pyproject.toml` is dynamic and syncs to `Cargo.toml`
   - Verify: `grep '^version = ' Cargo.toml`

2. **Changelog fragments exist**:
   - Create towncrier news fragments in `news/` directory for all changes
   - Format: `<issue_number>.<type>.md` (e.g., `123.feature.md`, `45.bugfix.md`)
   - Types: `feature`, `bugfix`, `doc`, `removal`, `misc`

3. **All tests pass locally**:
   ```bash
   cargo test && \
   .venv/bin/python -m pytest tests/ -v && \
   uv run python make.py test-examples
   ```

4. **Quality checks pass**:
   ```bash
   cargo clippy --all-targets --all-features -- -D warnings && \
   ruff check && \
   ./scripts/generate_stubs.sh && \
   .venv/bin/mkdocs build
   ```

5. **Documentation builds successfully**:
   - Run `.venv/bin/mkdocs build` locally
   - Verify all examples render correctly
   - Check for broken links or missing references

## Repository Architecture

### Key Design Principles
- Rust core provides all functionality
- Python bindings are 1:1 mirrors of Rust API
- SI base units (meters, seconds, radians) throughout
- Type stubs auto-generated for IDE support
- All Geocentric/Geodetic locations are in (lon, lat, alt) order
- PointLocation and PolygonLocation use degrees for usability and are in lon, lat order

### Main Entry Points
- **`lib.rs`**: Main library entry, re-exports all public modules
- **`pymodule/mod.rs`**: Python bindings aggregator (includes all binding files - PyO3 constraint)

### Core Module Summary

**Time (`time/`)** - Time representation and conversions
- **`Epoch` struct is foundational** - primary time representation for entire library
  - Methods: `from_datetime()`, `from_jd()`, `from_mjd()`, `jd()`, `mjd()`, `to_datetime()`
  - Supports: UTC, TAI, GPS, TT, UT1
  - Nanosecond precision, all arithmetic in seconds
- `conversions.rs`: Time system conversions (`datetime_to_jd()`, `jd_to_datetime()`)
- `TimeRange`: Iterator for time sequences

**Orbital Mechanics (`orbits/`)** - Elements, propagation, TLE
- **`keplerian.rs`**: Orbital element utilities
  - **Element order**: `[a, e, i, raan, argp, anomaly]` (meters, radians)
  - Functions: `state_osculating_to_cartesian()`, `state_cartesian_to_osculating()`
  - Utilities: `mean_motion()`, `orbital_period()`, `semimajor_axis()`, anomaly conversions
- `KeplerianPropagator`: Analytical two-body propagation
- `SGPPropagator`: SGP4/SDP4 TLE propagation
- `TLE` struct: Two-Line Element format
- **`traits.rs`**: Common propagator traits (`Propagator`, `PropagatorMulti`)

**Coordinate Systems (`coordinates/`)** - Position/velocity transformations
- `cartesian.rs`: State vectors `[x, y, z, vx, vy, vz]` (meters, m/s)
- `geocentric.rs`: Spherical `[radius, lat, lon]` (meters, radians)
  - Functions: `position_geocentric_to_ecef()`, `position_ecef_to_geocentric()`
- `geodetic.rs`: WGS84 ellipsoidal `[lat, lon, alt]` (radians, meters)
  - Functions: `position_geodetic_to_ecef()`, `position_ecef_to_geodetic()`
  - Constants: `WGS84_A`, `WGS84_F`
- `topocentric.rs`: Local horizontal (ENZ/SEZ), azimuth-elevation
  - Functions: `relative_position_ecef_to_enz()`, `position_enz_to_azel()`

**Reference Frames (`frames.rs`)** - ECI/ECEF transformations
- Functions: `position_ecef_to_eci()`, `position_eci_to_ecef()`, `state_ecef_to_eci()`, `state_eci_to_ecef()`
- Implementation: IAU 2006/2000A precession-nutation model
- **Requires EOP** for accurate transformations

**Earth Orientation (`eop/`)** - EOP data providers
- `FileEOPProvider`: Load from files (production)
- `StaticEOPProvider`: Built-in historical (testing)
- `download.rs`: Download latest from IERS
- `global.rs`: Global provider management
- **Testing**: Use `setup_global_test_eop()` to initialize

**Attitude (`attitude/`)** - 3D rotation representations
- `Quaternion`: `[w, x, y, z]` (singularity-free)
- `RotationMatrix`: 3x3 DCM
- `EulerAngle`: 12 sequences (XYZ, ZYX, etc.)
- `EulerAxis`: Unit vector + angle
- All conversions between representations via **traits**

**Trajectories (`trajectories/`)** - Time-series state storage
- **`traits.rs`**: Common trajectory interface
  - `InterpolationMethod` enum
  - `TrajectoryEvictionPolicy` enum
  - `OrbitFrame`, `OrbitRepresentation` enums
- `DTrajectory`: Dynamic-dimension (runtime sized)
- `STrajectory6`: Static 6D (optimized for orbital states)
- `OrbitTrajectory`: Specialized for orbital mechanics

**Constants (`constants/`)** - Physical and mathematical constants
- `math.rs`: `DEG2RAD`, `RAD2DEG`, `AS2RAD`, `RAD2AS`
- `time.rs`: `MJD_ZERO`, `MJD2000`, `GPS_ZERO`, `TAI_GPS`, `TT_TAI`
- `physical.rs`: **Use for all orbital calculations**
  - Gravitational parameters: `GM_EARTH`, `GM_SUN`, `GM_MOON` [m³/s²]
  - Radii: `R_EARTH`, `R_SUN`, `R_MOON` [m]

### Development Conventions

**Conversion Function Priority** - Always prefer existing library functions:
- `time::conversions` for time system conversions
- `coordinates` module for coordinate transformations
- `orbits::keplerian` for orbital element conversions
- `frames` module for reference frame transformations
- **Never reimplement existing functionality**

**Naming Conventions**:
- Functions/Methods: `snake_case`
- Structs/Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Use descriptive names: `position_ecef_to_eci` not `pos_conv`

**Error Handling**:
- Rust: `Result` types with descriptive errors
- Python: Convert to appropriate Python exceptions
- Include context: what failed, why, with what values

### Test Structure (`tests/`)
Python tests mirror Rust module structure:
- `tests/time/`: Time system and Epoch tests
- `tests/orbits/`: Orbital mechanics tests
- `tests/coordinates/`: Coordinate transformation tests
- `tests/attitude/`: Attitude representation tests
- `tests/trajectories/`: Trajectory tests

**Test Fixtures** (`conftest.py`):
- **ALWAYS check conftest.py first** before creating fixtures
- Auto-use fixtures: `eop` (file-based EOP provider, module scope)
- Optional fixtures: `eop_original_brahe`, `iau2000_standard_filepath`, `iau2000_c04_20_filepath`
- Create local fixtures only for test-specific data or specialized configurations

### Maintenance
- **Update CLAUDE.md** when making significant architectural changes
- **Place working notes in `.claude/`** to avoid cluttering repo
- **Investigate conflicts** when findings disagree with this document
