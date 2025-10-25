<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - Practical Astrodynamics</em>
</p>
<p align="center">
<a href="https://github.com/duncaneddy/brahe/actions/workflows/unit_tests.yml" target="_blank">
    <img src="https://github.com/duncaneddy/brahe/actions/workflows/unit_tests.yml/badge.svg" alt="Tests">
</a>
<a href="https://codecov.io/gh/duncaneddy/brahe">  
  <img src="https://codecov.io/gh/duncaneddy/brahe/graph/badge.svg?token=1JDXP549Q4"></a>
<a href="https://crates.io/crates/brahe" target="_blank">
    <img src="https://img.shields.io/crates/v/brahe.svg" alt="Crate">
</a>
<a href="https://pypi.org/project/brahe" target="_blank">
    <img src="https://img.shields.io/pypi/v/brahe?color=blue" alt="PyPi">
</a>
<a href="https://duncaneddy.github.io/brahe/latest" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs">
</a>
<a href="https://github.com/duncaneddy/brahe/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg", alt="License">
</a>
</p>

----

Documentation: https://duncaneddy.github.io/brahe/latest

Rust Library Reference: https://docs.rs/crate/brahe/latest

Source Code: https://github.com/duncaneddy/brahe

----

# Brahe

Brahe is a modern satellite dynamics library for research and engineering applications. It is designed to be easy-to-learn, quick-to-deploy, and easy to build on. The north-star of the development is enabling users to solve meaningful problems quickly and correctly.

Brahe is permissively licensed under an [MIT License](https://github.com/duncaneddy/brahe/blob/main/LICENSE) to encourage enable people to use and build on the work without worrying about licensing restrictions. We want people to be able to stop reinventing the astrodynamics wheel because commercial licenses are expensive, or open-source options are hard to use or incomplete.

Finally, we also try to make the software library easy to understand and extend. Many astrodynamics libraries are written in a way that makes them hard to read, understand, or modify. Brahe is written in a modern style with an emphasis on code clarity and modularity to make it easier to understand how algorithms are implemented and to make it easier to extend the library to support new use-cases. This also has the added benefit of making it easier to verify and validate the correctness of the implementation.

If you do find this useful, please consider starring the repository on GitHub to help increase its visibility. If you're using Brahe for school, research, a commercial endeavour, or flying a mission. I'd love to know about it!

If you find a bug, have a feature request, want to contribute, please open an issue or a pull request on the GitHub repository. Contributions are welcome and encouraged! If you see something missing, but don't know how to start contributing, please open an issue and we can discuss it. We are building software to help everyone on this planet explore the universe. We encourage you to bring your unique perspective to help make us stronger. We appreciate contributions from everyone, no prior space experience is needed to participate.

We hope you find Brahe useful for your work!

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- **[Learn](https://duncaneddy.github.io/brahe/learn/)**: Provides short-form documentation of major concepts of the package.
- **[Examples](https://duncaneddy.github.io/brahe/examples/)**: Provides longer-form examples of how-to examples of accomplish common tasks.
- **[Python API Reference](https://duncaneddy.github.io/brahe/library_api/)**: Provides detailed reference documentation of the Python API.
- **[Rust API Reference](https://docs.rs/brahe)**: Provides detailed reference documentation of the Rust API.

## License

The project is licensed under the MIT License - see the [LICENSE](license.md) for details.

We want to make it easy for people to use and build on the work without worrying about licensing restrictions.

<!-- ## Citation / Acknowledgement -->

## Quick Start

### Python

To install the latest release of brahe for Python, simply run:

```bash
pip install brahe
```

You can then import the package in your Python code with:

```python
import brahe as bh
```

And do something fun like calculate the orbital-period of a satellite in low Earth orbit:

``` python
--8<-- "./examples/common/orbital_period.py:9"
```

Here are some common operations to get you started:

**Working with Time:**
``` python
--8<-- "./examples/common/working_with_time.py:10"
```

**Coordinate Transformations:**
``` python
--8<-- "./examples/common/coordinate_transformations.py:10"
```
**Propagating an Orbit:**
``` python
--8<-- "./examples/common/propagating_an_orbit.py:8"
```

**Computing ISS Access Windows:**
``` python
--8<-- "./examples/common/iss_access_windows.py:9"
```

----

### Rust

To use brahe in your Rust project, add it to your `Cargo.toml`:

```toml
[dependencies]
brahe = "0.5"
```

You can then use the crate in your rust code with:

```rust
use brahe as bh;
```

And still calculate the orbital-period of a satellite in low Earth orbit:

``` rust
--8<-- "./examples/common/orbital_period.rs:5"
```

You can do everything that you can do in Python in Rust as well:

**Working with Time:**
``` rust
--8<-- "./examples/common/working_with_time.rs:5"
```

**Coordinate Transformations:**
``` rust
--8<-- "./examples/common/coordinate_transformations.rs:6"
```

**Propagating an Orbit:**
``` rust
--8<-- "./examples/common/propagating_an_orbit.rs:4"
```

**Computing ISS Access Windows:**
``` rust
--8<-- "./examples/common/iss_access_windows.rs:5"
```