| Testing | Coverage | Documentation |
| :-----: | :------: | :-----------: |
| [![Build Status](https://travis-ci.org/duncaneddy/brahe.svg?branch=master)](https://travis-ci.org/duncaneddy/brahe) | [![Coverage Status](https://coveralls.io/repos/github/duncaneddy/brahe/badge.svg?branch=master)](https://coveralls.io/github/duncaneddy/brahe?branch=master) | [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://duncaneddy.github.io/brahe/latest) |

# brahe
A Python satellite guidance, navigation, and control library.

Current high-fidelity satellite GNC modeling software generally runs into the following pitfalls:
1. It is commercially licensed and closed-source code, making it difficult, if not impossible, to be used by hobbyists and academic researchers.
2. Large open-source projects with a steep learning curve making correct use for small projects difficult
3. Out-dated API leading making it hard to incorporate into modern projects.

These challenges make it an unfortunately common occurance that guidance, 
navigation, and control engineers will frequently reimplement common 
astrodynamics libraries for each new project or analysis.

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

The documentation for the package can be found here: <https://duncaneddy.github.io/brahe/latest>