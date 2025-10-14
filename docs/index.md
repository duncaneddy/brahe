<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
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

!!! warning "Pure-Python Brahe Deprecation Notice"
    
    The older pure-Python implementaiton of brahe is currently being deprecated in favor of an improved Rust-based implementation. There will be breaking changes during this period that include breaking changes. You can read more about this change in the [deprecation notice](about/python_deprecation.md).

# Brahe

!!! quote ""'s 

    All software is wrong, but some is useful.

Brahe is a modern satellite dynamics library for research and engineering
applications. It is designed to be easy-to-learn, quick-to-deploy, and easy to build on.
The north-star of the development is enabling users to solve meaningful problems
and answer questions quickly and correctly.

The Brahe permissively licensed and distributed under an [MIT License](https://github.com/duncaneddy/brahe/blob/main/LICENSE) to encourage adoption and enable the
broader community to build on the work.

If you do find it useful, please consider starring the repository on GitHub to help
increase its visibility. If you're using Brahe for school, research, a commercial endeavour, or flying a mission. I'd love to know about it!
You can find my contact information on my [personal website](https://duncaneddy.com), 
or open an issue on the GitHub repository.


## Quick Start

To install the latest release of brahe, simply run:

```bash
pip install brahe
```

You can then import the package in your Python code with:

```python
import brahe as bh
```

And do something fun like calculate the orbital-period of a satellite in low Earth orbit:

```python
import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.EARTH_RADIUS + 400e3  # 400 km altitude

# Calculate the orbital period using Kepler's third law
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
```

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- [Learn](learn/index.md): Provides short-form documentation of major concepts of the package.
- [Examples](examples/index.md): Provides longer-form examples of how-to examples of accomplish common tasks.
- [Library API](library_api/index.md): Provides detailed reference documentation of the Python API.
- [Rust API](https://docs.rs/brahe): Provides detailed reference documentation of the Rust API.


## Support and Acknowledgement

Brahe is currently being developed primarily for my own enjoyment and
because I find having these tools helpful in professional and hobby work. I plan to
continue developing it for the time being regardless of greater adoption as time permitting.

That being said, it's incredibly encouraging and useful to know if the
software is being adopted or found useful in wider practice.
