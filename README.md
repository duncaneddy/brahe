<p align="center">
  <a href="https://github.com/duncaneddy/brahe/"><img src="https://raw.githubusercontent.com/duncaneddy/brahe/main/docs/assets/logo-gold.png" alt="Brahe"></a>
</p>
<p align="center">
    <em>Brahe - Practical Astrodynamics</em>
</p>
<p align="center">
<a href="https://duncaneddy.github.io/brahe/latest" target="_blank">
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

<!-- ## Citation / Acknowledgement -->

## Going Further

If you want to learn more about how to use the package the documentation is structured in the following way:

- **[Learn](https://duncaneddy.github.io/brahe/learn/)**: Provides short-form documentation of major concepts of the package.
- **[Examples](https://duncaneddy.github.io/brahe/examples/)**: Provides longer-form examples of how-to examples of accomplish common tasks.
- **[Python API Reference](https://duncaneddy.github.io/brahe/library_api/)**: Provides detailed reference documentation of the Python API.
- **[Rust API Reference](https://docs.rs/brahe)**: Provides detailed reference documentation of the Rust API.

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
import brahe as bh

# Define the semi-major axis of a low Earth orbit (in meters)
a = bh.constants.R_EARTH + 400e3  # 400 km altitude

# Calculate the orbital period
T = bh.orbital_period(a)

print(f"Orbital Period: {T / 60:.2f} minutes")
# Outputs:
# Orbital Period: 92.56 minutes
```

or find the when the ISS will next pass overhead:

``` python
import brahe as bh

bh.initialize_eop()

# Compute upcoming passes of the ISS over San Francisco
passes = bh.location_accesses(
    [bh.PointLocation(-122.4194, 37.7749, 0.0)],  # San Francisco
    [bh.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "active")],  # ISS
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0),
)
print(f"Number of passes in next 24 hours: {len(passes)}")
# Example Output: Number of passes in next 24 hours: 5
```

If you want to see more examples of how to use brahe, you can find even more with full source code in the [Examples section](https://duncaneddy.github.io/brahe/latest/examples/index.html) of the documentation.

## License

The project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

We want to make it easy for people to use and build on the work without worrying about licensing restrictions!