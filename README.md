| Testing | Coverage | PyPi | Documentation |
| :-----: | :------: | :--: | :-----------: |
| [![Build Status](https://travis-ci.org/duncaneddy/brahe.svg?branch=master)](https://travis-ci.org/duncaneddy/brahe) | [![Coverage Status](https://coveralls.io/repos/github/duncaneddy/brahe/badge.svg?branch=master)](https://coveralls.io/github/duncaneddy/brahe?branch=master) | [![PyPI version](https://badge.fury.io/py/brahe.svg)](https://badge.fury.io/py/brahe) | [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://duncaneddy.github.io/brahe/) |

# brahe
A Python satellite astrodynamics library meant for use in spacecraft guidance,
navigation, and control work.

Current high-fidelity satellite GNC modeling software generally runs into the following pitfalls:
1. It is commercially licensed and closed-source code, making it difficult, if not impossible, to be used by hobbyists and academic researchers.
2. Large open-source projects with a steep learning curve making correct use for small projects difficult
3. Out-dated API leading making it hard to incorporate into modern projects.
4. Hard to get up and running with.

These challenges make it an unfortunately common occurance that spacecraft
engineers and researchers will frequently reimplement common astrodynamics libraries 
for each new project or analysis.

With these deficienties in mind, brahe aims to provide a open-source, MIT-licensed,
high fidelity astrodynamics toolbox to help make it easy to perform high-quality 
simulation and analysis of satellite attitude and orbit dynamics.

This is the sister repository of [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl),
a Julia language implementation of the same functionality. In general, the 
functionality supported by each repository is interchangeable. The main differences
between the repositories come down to the underlying repositories. The Julia 
implementation tends to be more performant, while the Python implementation can
take advantage of the larger package ecosystem.

## Documentation

The documentation for the package can be found here: <https://duncaneddy.github.io/brahe/>

## Installation

This package is distributed from PyPi, and can be installed simply with:

```bash
pip3 install brahe
```

## License

The brahe package is licensed and distributed under the MIT License to encourage
usage and to make it easy to integrate with other tools.

The only thing asked is that if you do use the package in your work, or appreciate
the project, either send a message or star the project. Knowing that the project
is being actively used is a large motivator for continued development.

## Using brahe

If you use brahe for your research or work, I'd love to know about it. Please
message me, or let me know. If there are 