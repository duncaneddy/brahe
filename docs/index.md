<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - Practical Astrodynamics</em>
</p>
<p align="center">
<a href="https://docs.brahe.space/latest" target="_blank">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg" alt="Docs">
</a>
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
<a href="https://github.com/duncaneddy/brahe/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-green.svg", alt="License">
</a>
<a href="https://joss.theoj.org/papers/a7ec6268a42c9fada797a3cb213c1d17">
    <img src="https://joss.theoj.org/papers/a7ec6268a42c9fada797a3cb213c1d17/status.svg">
</a>
<a href="https://arxiv.org/abs/2601.06452">
    <img src="https://img.shields.io/badge/arXiv-2601.06452-b31b1b.svg" alt="arXiv">
</a>
</p>

----

Documentation: [https://docs.brahe.space/latest](https://docs.brahe.space/latest)

Rust Library Reference: [https://docs.rs/crate/brahe/latest](https://docs.rs/crate/brahe/latest)

Source Code: [https://github.com/duncaneddy/brahe](https://github.com/duncaneddy/brahe)

----

# Brahe

Brahe is a modern satellite dynamics library for research and engineering applications. It is designed to be quick-to-deploy, composable, extensible, and easy-to-learn. The north-star of the development is enabling users to solve meaningful problems quickly and correctly.

Brahe is permissively licensed under an [MIT License](https://github.com/duncaneddy/brahe/blob/main/LICENSE) to enable people to use and build on the work without worrying about licensing restrictions. We want people to be able to stop reinventing the astrodynamics "wheel" because commercial licenses are expensive and open-source options are hard to use.

We try to prioritize making the software library easy to learn, use, and verify. Many astrodynamics libraries are written with many layers of abstraction for flexibility that can make it challenging for new users to understand _where_ the actual logic and algorithms are being executed. Brahe is written in a modern style with an emphasis on code clarity and modularity to make it easier to understand what individual functions are actually doing. This approach has the added benefit of making it easier to verify and validate the correctness of the implementation.

If you do find this useful, please consider starring the repository on GitHub to help increase its visibility. If you're using Brahe for school, research, a commercial endeavour, or flying a mission. I'd love to know about it.

We hope you find Brahe useful for your work!

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- **[Getting Started](getting_started/index.md)**: The getting started guide provides a high-level overview of the main concepts and components of Brahe, along with a quick introduction to using it. It is designed to help new users get up to speed quickly and understand the core ideas behind Brahe before diving into the more detailed documentation in the other sections. If you are new to Brahe, this is a great place to start!
- **[User Guide](learn/index.md)**: The user guide provides more comprehensive module-by-module documentation covering the capabilities each module provides with examples on how to use it.
- **[Examples](examples/index.md)**: This section contains a collection of worked examples that demonstrate how to use Brahe to solve various problems or accomplish specific tasks. The examples are designed to be practical and cover a range of use cases, from basic to more advanced.
- **[Python API Reference](library_api/index.md)**: Provides detailed reference documentation of the Python API, including all public classes, functions, and methods organized by module.
- **[Rust API Reference](https://docs.rs/brahe)**: Provides detailed reference documentation of the Rust API, including all public structs, traits, functions, and methods organized by module.

## Quick Start

To install the latest release version of Brahe, you can use the following commands:

=== "Python"

    ``` bash
    # Using uv (preferred)
    uv add brahe

    # Or using pip
    pip install brahe

    # Or with optional plot dependencies
    uv add "brahe[plots]"
    pip install "brahe[plots]"
    ```

=== "Rust"

    ``` bash
    crago add brahe
    ```

You can then import the package in your code with:

=== "Python"

    ```python
    import brahe as bh
    ```

=== "Rust"

    ```rust
    use brahe as bh;
    ```

And do something fun like calculate the orbital-period of a satellite in low Earth orbit:

=== "Python"

    ``` python
    --8<-- "./examples/common/orbital_period.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/common/orbital_period.rs:5"
    ```

or find the when the ISS will next pass overhead:

``` python
--8<-- "./examples/common/simple_access.py:9"
```

## Citing Brahe

If you use Brahe in your work, please cite the following paper:

```bibtex
@article{eddy2026brahe,
      title={{Brahe: A Modern Astrodynamics Library for Research and Engineering Applications}}, 
      author={Duncan Eddy and Mykel J. Kochenderfer},
      year={2026},
      eprint={2601.06452},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2601.06452}, 
}
```

## License

The project is licensed under the MIT License - see the [LICENSE](./about/license.md) for details.

We want to make it easy for people to use and build on the work without worrying about licensing restrictions.

Additionally, brahe uses [cargo-deny](https://github.com/embarkstudios/cargo-deny) to confirm that all dependencies are permissively licensed and compatible with commercial use. The permitted licenses can be found in the [cargo-deny configuration file](https://github.com/duncaneddy/brahe/blob/main/deny.toml).

## Contributing

If you find a bug, have a feature request, want to contribute, please open an issue or a pull request on the GitHub repository. Contributions are welcome and encouraged! If you see something missing, but don't know how to start contributing, please open an issue and we can discuss it. We are building software to help everyone on this planet explore the universe. We encourage you to bring your unique perspective to help make us stronger. We appreciate contributions from everyone, no prior space experience is needed to participate.

## Sponsors

We are pleased to acknowledge the following sponsors for their support:

<p align="center">
    <a href="https://www.northwoodspace.io/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/5/5b/Northwood_Space_logo.svg" alt="Northwood Space" width="200">
    </a>
</p>

## Versioning

!!! warning "Versioning"

    Brahe follows a versioning scheme modeled on [NumPy's policy](https://numpy.org/doc/stable/dev/depending_on_numpy.html) rather than strict [SemVer](https://semver.org/). Versions are PEP 440 compliant and take the form `major.minor.bugfix`:

    - **Major** releases (`X.0.0`) are rare and signal significant API or ABI breaks.
    - **Minor** releases (`1.Y.0`) contain new features, deprecations, and removals of previously deprecated code.
    - **Bugfix** releases (`1.2.Z`) contain only fixes — no new features, deprecations, or removals.

    **Deprecation policy (transitional):** The long-term target — matching NumPy — is that backwards-incompatible API changes emit a `DeprecationWarning` for at least two minor releases before removal. **While Brahe is in its early adoption phase, a deprecation may occur and be removed within a single minor release.** This window will expand to multiple minor releases with deprecation warnings as adoption grows.

    **Pinning:** For most projects `brahe>=1.2` is sufficient. If you need guaranteed stability during the transitional deprecation period, pin to a specific `major.minor.patch` version (e.g., `1.2.3`) rather than using a floating specifier (e.g., `^1.2.0` or `>=1.2.0`). See [the versioning page](./about/versioning.md) for the full policy, including guidance on treating `DeprecationWarning` as an error in CI.

## AI Usage Policy

The development of Brahe has roots in 2014 when I first started writing astrodynamics software for my PhD. The main algorithms and code structure evolved over the years based on my own experience applying the software to both research problems and operational space missions. The core functionality of the library (time handling, reference frames, reference frame transformations, coordinate transformations) were all developed before the usage of AI tools. AI tools have since been intentionally adopted to help with improving and expanding capabilities that were on the nice-to-have feature list. They have also been used to help with writing documentation and improve code coverage. All results and outputs are manually reviewed, run, tested, and verified manually before being merged into the main branch, we expect the same from all contributions to the codebase. We are committed to maintaining the same standards of code clarity, modularity, and correctness for all contributions regardless of whether they were AI-assisted or not.

The use of AI-assisted coding in brahe is itself a bit of an expertiment. We are interesting in seeing how it can be used to help with the development of the library, however we will not compromise on the quality of the codebase overall. While we may get it wrong at times, times, producing correct, accurate, maintainable code is more important than producing code quickly.

For new contributions, we allow the use of AI-assited coding, however we expect that PRs will be manually reviewed and tested before being submitted and that all PRs follow the same standards of code clarity, modularity, and correctness as the rest of the codebase.

---

### Additional Examples

If you want to see more examples of how to use brahe, you can find a few quick examples below. You will also find more examples throughout the documentation and in the [Examples](./examples/index.md) section of the documentation.

**Working with Time:**
=== "Python"

    ``` python
    --8<-- "./examples/common/working_with_time.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/common/working_with_time.rs:5"
    ```


**Coordinate Transformations:**
=== "Python"

    ``` python
    --8<-- "./examples/common/coordinate_transformations.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/common/coordinate_transformations.rs:6"
    ```

**Propagating an Orbit:**
=== "Python"
    ``` python
    --8<-- "./examples/common/propagating_an_orbit.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/common/propagating_an_orbit.rs:4"
    ```


**Computing ISS Access Windows:**
=== "Python"
    ``` python
    --8<-- "./examples/common/iss_access_windows.py:9"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/common/iss_access_windows.rs:5"
    ```