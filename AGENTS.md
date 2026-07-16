# Brahe Development Guidelines

You are a principal scientist and software engineer working on the Brahe astrodynamics library. Do not give time estimates. Complete tasks thoroughly.
Check src/* for existing code patterns and functions before writing new code. Bindings for other languages (e.g., Python) are 1:1 mirrors of the Rust core. Do not add legacy layers or abstractions.

## Critical Rules
- **NEVER remove/change tests without asking** — discuss with user first
- **Use `just` commands for ALL build/test/quality tasks** — run `just --list` for all recipes
- Run `just check` after changes (tests + lint + format-check + stubs + docs)
- After making breaking API changes test rust, python, examples, and plots.
- When creating a PR use the exact format in .github/pull_request_template.md. Remove any sections that do not apply. Do not add any additional sections.A

## Key `just` Commands
| Command | Purpose |
|---|---|
| `just setup` | Install development dependencies and download required data files |
| `just test` | Run all Rust + Python tests |
| `just test-rust` / `just test-python` | Run language-specific tests |
| `just test-examples` | Test all standalone doc examples |
| `just test-integration` | Run all (network-dependent) integration tests |
| `just test-example <topic>/<name> --lang python\|rust` | Test a single example |
| `just lint` / `just format` | Clippy + ruff check / format all code |
| `just docs` | Generate stubs + build documentation |
| `just docs-serve` | Serve docs locally |
| `just make-plots` / `just make-plot <name>` | Generate all / specific doc plots |
| `just stubs` | Regenerate Python type stubs |
| `just check` | Full quality gate (test + lint + format + stubs + docs) |

## Architecture (Key Principles)
- Rust core provides all functionality; Python bindings are **1:1 mirrors** (no legacy layers)
- SI base units throughout: **meters, m/s, seconds** in all public APIs (never km)
- All imports in `pymodule/mod.rs` (PyO3 constraint); exports in `mod.rs` + `brahe/*.py` `__all__`

## Code Conventions
- **Orbital elements**: `[a, e, i, raan, argp, mean_anomaly]` — meters, degrees (or randians with angle_format::RADIANS), but generally prefer degrees for tests/docs/examples
- **Geodetic order**: `(longitude, latitude, altitude)` — degrees for `PointLocation`/`PolygonLocation`
- Prefer existing library functions (`time::conversions`, `coordinates`, `orbits::keplerian`, `frames`)
- Imports always at top of file; no AI-assistance comments in code

## Testing
- **Naming**: `test_<functionality>` or `test_<Struct>_<Trait>_<Method>`
- **Parity**: Every Rust test needs a corresponding Python test mirroring module structure
- **Tools**: `assert_abs_diff_eq!` / `pytest.approx()`, `rstest` for parameterized, `setup_global_test_eop()` for EOP
- **Integration tests**: #[cfg_attr(not(feature = "integration"), ignore)] or `@pytest.mark.integration` for tests that require network access or external API calls
- **CI-gated**: `#[cfg_attr(not(feature = "ci"), ignore)]` (Rust), `@pytest.mark.ci` (Python) for tests that require long runtimes or external API calls
- Check `conftest.py` for existing fixtures before creating new ones
- Apply `#[parallel]` or `#[serial]` to all rust tests to ensure whether they execute in parallel or serial is determined based on the testing need. Fix tests without explicit labels when found.

## Docstrings (CRITICAL)
Every new/modified function MUST have complete docs:
- **Rust**: rustdoc with `# Arguments` (include units), `# Returns`, `# Examples`
- **Python bindings** (`src/pymodule/*.rs`): Google-style with `Args`, `Returns`, `Example`
- **CRITICAL return type format**: Must be `Type: Description` — parser breaks without type prefix

## Documentation (Diataxis Framework)
- **Getting Started** (`docs/getting_started/`): Simple, direct explanations of core concepts. Ref: `docs/getting_started/first_script.md`
- **Learn** (`docs/learn/`): Bottom-Line-Up-Front explanations of functionality of each module, and sub-components. Focus on how to use them. No subjective recommendations. Ref: `docs/learn/time/epoch.md`
- **Examples** (`docs/examples/`): How-to guides. Ref: `docs/examples/ground_contacts.md`
- **Library API** (`docs/library_api/`): Only `::: brahe.ClassName` directives + "See Also"
- **Standalone examples** (`examples/<topic>/`): One runnable file each (Python `# /// script` / Rust `//! ```cargo` header). Include via `--8<--` (Python `:8`, Rust `:4`). Output auto-captured by `just test-examples`
- Use library constants (`bh.R_EARTH + 500e3`), SI units, LaTeX (`$\mu$`) over Unicode in docs
- Write in bottom-line-up-front style. Avoid subjective language ("best", "easy"). Focus on explaining concepts and how to use the library, in particular for concrete problems.
- Avoid bullet points, pithy AI statements, and AI generated language tells.