<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/pages/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - easy-to-learn, high-performance, and quick-to-deploy</em>
</p>
<p align="center">
<a href="https://github.com/duncaneddy/brahe/actions/workflows/commit.yml" target="_blank">
    <img src="https://github.com/duncaneddy/brahe/actions/workflows/commit.yml/badge.svg" alt="Test">
</a>
<a href="https://codecov.io/gh/duncaneddy/brahe">  
  <img src="https://codecov.io/gh/duncaneddy/brahe/graph/badge.svg?token=1JDXP549Q4"></a>
<a href="https://crates.io/crates/brahe" target="_blank">
    <img src="https://img.shields.io/crates/v/brahe.svg" alt="Crate">
</a>
<a href="https://pypi.org/project/brahe" target="_blank">
    <img src="https://img.shields.io/pypi/v/brahe?color=blue" alt="PyPi">
</a>
<a href="https://duncaneddy.github.io/brahe" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs">
</a>
<a href="https://github.com/duncaneddy/brahe/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg", alt="License">
</a>
</p>

----

Documentation: https://duncaneddy.github.io/brahe

Rust Library Reference: https://docs.rs/crate/brahe/latest

Source Code: https://github.com/duncaneddy/brahe

----

# Brahe

> [!WARNING]
>
> The older pure-Python version of brahe is currently being deprecated in favor of a mixed
> Rust-Python implementation, along with improved documentation. That means that the development
> on the `master` branch is being frozen and will no longer be developed against. Moving forward
> the `main` branch will be the primary branch for the project.
>
> There will be point commits (less than `1.0.0`) during this period as part
> of improving the CI/CD workflow for the project. Furthermore, initially the features of the
> new implementation will not be at partity with the old python implementation, so users should
> pin their requirements file to use the latest commit of the master branch:
>
> ```
> brahe @ git+https://github.com/duncaneddy/brahe@master
> ```
>
> To install and use the latest master branch via pip
>
> ```
> pip install git+https://github.com/duncaneddy/brahe.git@master
> ```
>
> The old master branch can be found [here](https://github.com/duncaneddy/brahe/tree/master).


Brahe is a modern satellite dynamics library for research and engineering
applications. It is designed to be easy-to-learn, high-performance, and quick-to-deploy.
The north-star of the development is enabling users to solve meaningful problems
and answer questions quickly, easily, and correctly.

The key features of the library are:

- **Intuitive API**: API designed to be easily composable, making it easy to
  solve complex problems correctly by building on core functionality.
- **Easy-to-Learn**: Designed to be easy to use and learn. The objective is
  to provide clear documentation and visibility into what the software is doing
  so that users don't need to spend time reverse engineering internal routines
  and more time solving their own problems.
- **High-Performance**: Brahe provides a Python 3.6+ wrapper that is
  auto-generated from a core Rust library. This provides fast core implementation,
  while allowing users to take advantage of Python's rich scientific ecosystem
  if they so choose.
- **Answer Questions Quickly**: Brahe is designed to make it easy to code up
  solutions to meaningful problems. High-fieldity, high-performance APIs are not
  the end-objective, but helping users solve their problems.

Brahe gets its name from the combination of Rust and astrodynamics (Rust +
astrodynamics = Brahe). The library specifically focuses on satellite astrodynamics
and space mission analysis. While the underlying concepts have been studied and known since
Kepler wrote down his three laws, there are few modern software
libraries that make these concepts easily accessible. While extremely well tested,
other astrodynamics and mission analysis software can have an extremely steep
learning curve, making it difficult to quickly run simple analysis that is known
to be correct.

Because of this, students, researchers, and engineers frequently end up
reimplementing common astrodynamics and mission analysis tools with unfortunately
frequent regularity. While reimplementation of common code can be a good learning
mechanisms, in most cases it is both error-prone and costs time better spent
on other endeavours. This project seeks to providing an easy-to-use,
well-tested library, to enable everyone to more easily, and quickly
perform astrodynamics and space mission analysis without sacrificing performance
or correctness. The software built in Rust for performance with bindings to
Python for ease of use.

The implementation approach is opinionated, the objective is to provide an
easy-to-use and accurate astrodynamics library to enable users to quickly
and correctly solve most common problem types. it is not practical to try to
implement _every_ aerodynamics model and function utilized in practice or historically.
Since Brahe is open source, if a specific function is not present, or a different
implementation is required, users can modify the code to address their specific
use case. This means that Brahe, while we want to continue expanding the
capabilities of the module over time, the immediate goal is to provide a well-tested,
flexible, composable API to quickly address modern problems in astrodynamics.

One example of this in practice is that the built-in Earth reference frame transformation
utilizes the IAU 2006/2000A precession-nutation model, CIO-based transformation.
Even through there are multiple ways to construct this transformation, Brahe
only implements one. Another example, is that the geodetic and geocentric
transformations use the latest NIMA technical report definitions for Earth's radius and flatness.
If a desired model isn't implemented users are free to extend the software to
address and functionality or modeling gaps that exist to address their specific application.

## Documentation

You can find the package documentation [here](https://duncaneddy.github.io/brahe).
This documentation is meant to provide a human-friendly walk through of the
software and package. Brahe is currently in the early stages of development so
the documentation will likely not be complete. Sections marked **[WIP]**
will have some software functionality implemented but not be considered
documented.

The most complete API reference guide will always be the Rust crate API
reference, found on [crates.io](https://docs.rs/brahe/). This is always up-to-date with the latest release
since it is autogenerated at build time during the release process.

## Software Usage and License

The Brahe package is licensed and distributed under
an [MIT License](https://github.com/duncaneddy/brahe/blob/main/LICENSE) to
encourage adoption and to make it easy to integrate with other tools.

The only thing asked is that if you do use the package in your work, or
appreciate the project, either send a message or star the project. Knowing
that the project is being actively used is a large motivator for continued
development.

## Support and Acknowledgement

Brahe is currently being developed primarily for my own enjoyment and
because I find having these tools helpful in professional and hobby work. I plan to
continue developing it for the time being regardless of greater adoption as time permitting.

That being said, it's incredibly encouraging and useful to know if the
software is being adopted or found useful in wider practice. If you're
using Brahe for school, research, or a commercial endeavour, I'd
love to know about it! Tweet me [@duncaneddy](https://twitter.com/DuncanEddy) or
email me at duncan.eddy (at) gmail.com.

## Contribution

To get started with developing for Brahe you need to have both Rust and Python
installed on your system.

To install the Rust toolchain visit [rustup.rs](https://rustup.rs/). To
install Python, we recommend using [pyenv](https://github.com/pyenv/pyenv) and
[pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv).

Currently, the project needs to be built using the `nightly` toolchain. To install the nightly
toolchain, run the following commands:

```bash
rustup toolchain install nightly
rustup default nightly
```

### Testing

To execute the Rust test suite run the following command:

```bash
cargo test
```

To execute the python test suite first install the package in editable mode with
development dependencies:

```bash
pip install -e ".[dev]"
```

Then run the test suite with:

```bash
pytest
```

### Documentation

To build the documentation for the project, first install the development dependencies
with the following command:

```bash
./scripts/build-docs.sh build
```

Then you can build and serve the documentation with:

```bash
./scripts/build-docs.sh serve
```

### Contribution

Once you have made your changes, please open a pull request with a clear description of
the changes you have made. If you are adding a new feature, please include tests and
documentation for the new feature.
