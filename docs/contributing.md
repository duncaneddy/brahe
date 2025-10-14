# Contributing

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

The package includes both Rust and Python tests.

To execute the Rust test suite run the following command:

```bash
cargo test
```

To execute the python test suite first install the package in editable mode with
development dependencies:

```bash
uv pip install -e .
```

Then run the test suite with:

```bash
uv run pytest
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
