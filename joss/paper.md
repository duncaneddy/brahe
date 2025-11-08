---
title: 'Brahe: A Modern Astrodynamics Dynamics Library for Research and Engineering Applications'
tags:
  - Python
  - Rust
  - astrodynamics
  - satellites
  - astronomy
  - space
  - brahe
  - orbital mechanics
  - space situational awareness
  - satellite scheduling
  - space operations
authors:
  - name: Duncan Eddy
    orcid: 0009-0000-2832-9711
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 07 November 2025
bibliography: paper.bib

---

# Summary

<!-- \href{https://github.com/duncaneddy/brahe}{brahe} -->

[`brahe`](https://github.com/duncaneddy/brahe) is a modern astrodynamics dynamics library for research and engineering applications. The representation and prediction of satellite motion is the fundamental problem of astrodynamics. The motion of celestial bodies has been studied for centuries with initial equations of motion dating back to Kepler [@kepler1953epitome] and Newton [@newton1833philosophiae]. Current research and applications in space situational awareness, satellite task planning, and space mission operations require accurate and efficient numerical tools to perform coordinate transformations, model perturbations, and propagate orbits. `brahe` incorporates the latest conventions and models for time systems and reference frame transformations from the International Astronomical Union (IAU) [@hohenkerk2017iau] and International Earth Rotation and Reference Systems Service (IERS) [@petit2010iers]. It implements force models for Earth-orbiting satellites including atmospheric drag, solar radiation pressure, and third-body perturbations from the Sun and Moon [@vallado2001fundamentals, @montenbruckgill2000]. It also provides standard orbit propagation algorithms, including the Simplified General Perturbations (SGP) Model [@vallado2006revisiting]. Finally, it implements recent algorithms for fast, parallelized computation of ground station and imaging-target visibility [@eddy2021maximum], a foundational problem in satellite scheduling and mission planning.

With `brahe`, predicing upcoming satellite passes over ground stations or imaging targets can be accomplished in seconds and three lines of code

```python
import brahe as bh

bh.initialize_eop()

passes = bh.location_accesses(
    bh.PointLocation(-122.4194, 37.7749, 0.0),  # San Francisco
    bh.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "active"),  # ISS
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0),
)
```

`brahe` allows users to quickly access Two-Line Element (TLE) data from Celestrak [@celestrak] and propagate orbits using the SGP4 dynamics model. This can be used to perform space situational awareness tasks such as predicting the orbits of all Starlink satellites over the next 24 hours

\newpage

```python
import brahe as bh

bh.initialize_eop()

starlink = bh.datasets.celestrak.get_tles_as_propagators("starlink", 60.0)

for sat in starlink:
    sat.propagate_to(sat.epoch + 86400.0)  # Propagate one orbit (24 hours)
```

The above routine can propagate orbits for all ~9000 Starlink satellites in approximately 5 minutes on an M1 Max MacBook Pro with 10 cores and 64 GB RAM. Finally, the package provides direct, easy-to-use functions for low-level astrodynamics routines such as Keplerian to Cartesian state conversions and reference frame transformations

```python
import brahe as bh
import numpy as np

# Initialize Earth Orientation Parameter data
bh.initialize_eop()

# Define orbital elements
a = bh.constants.R_EARTH + 700e3  # Semi-major axis in meters (700 km altitude)
e = 0.001                         # Eccentricity
i = 98.7                          # Inclination in radians
raan = 15.0                       # Right Ascension of Ascending Node in radians
arg_periapsis = 30.0              # Argument of Periapsis in radians
mean_anomaly = 45.0               # Mean Anomaly

# Create a state vector from orbital elements
state_kep = np.array([a, e, i, raan, arg_periapsis, mean_anomaly])

# Convert Keplerian state to ECI coordinates
state_eci = bh.state_osculating_to_cartesian(state_kep, bh.AngleFormat.DEGREES)

# Define a time epoch
epoch = bh.Epoch(2024, 6, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)

# Convert ECI coordinates to ECEF coordinates at the given epoch
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

# Convert back from ECEF to ECI coordinates
state_eci_2 = bh.state_ecef_to_eci(epoch, state_ecef)

# Convert back from ECI to Keplerian elements
state_kep_2 = bh.state_cartesian_to_osculating(state_eci_2, bh.AngleFormat.DEGREES)
```

# Statement of Need

While the core algorithms for predicting and modeling satellite motion have been known for decades, there is a lack of modern, open-source software that implements these algorithms in a way that is accessible to researchers and engineers. Generally, existing astrodynamics software packages have one or more barriers to entry for individuals and organizations looking to develop astrodynamics applications, and often leads to duplicated and redundant effort as researchers and engineers are forced to re-implement foundational algorithms.

Flagship commercial astrodynamics software like Systems Tool Kit (STK) [@stk] and FreeFlyer [@freeflyer] are individually licensed and closed-source, creating two major drawbacks. First, the licensing costs can be prohibitive for researchers, individuals, small organizations, and start-ups. Even for larger organizations, per-node licensing cost can make large-scale deployment prohibitive. Second, the closed-source nature of these packages makes it difficult to understand and verify the exact algorithms and model implementations, which is critical for high-stakes applications like space mission operations [@mcoMishap1999]. Major open-source projects like Orekit [@maisonobe2010orekit] and GMAT [@hughes2014gmat] provide extensive functionality, but are large codebases with steep learning curves, making quick-adoption and integration into projects difficult. Furthermore, Orekit is implemented in Java, which can be a barrier to adoption in the current scientific ecosystem with users who are more familiar with Python. GMAT uses a domain-specific scripting language and has limited documentation and examples, making it difficult for new users to get started. Finally, there are academic libraries such as poliastro [@rodriguezPoliastro2022] which are not actively maintained. Other tools like Basilisk [@basilisk2020], provide high-fidelity modeling capabilities for full spacecraft guidance, navigation, and control (GNC) simulations, but are not directly distributed through standard package managers like PyPI and must be compiled from source to be used. Finally, academic work often has limited documentation and usage examples, making it difficult for new users to get started.

`brahe` seeks to address these challenges by providing a modern, open-source astrodynamics library following design principles of the Zen of Python [@peters2004zen]. The core functionality is implemented in Rust for performance and safety, with Python bindings for ease-of-use and integration with the scientific Python ecosystem. `brahe` is permissively licensed under an MIT License to encourage adoption and enable individuals and organizations to integrate and extend the software without worrying about licensing or cost restrictions. To further promote adoption, the library is extensively documented following the Diátaxis framework [@procida_diataxis] to aid learning\textemdash every Rust and Python function documented with arguments, returns, and usage examples, there is a user guide that explains the major concepts of the library, and there is a set of longer-form examples that demonstrate how to accomplish common tasks. To maintain high code quality, the library has a comprehensive test suite for both Rust and Python code. Additionally, all code samples in the documentation are automatically tested to ensure they remain up-to-date and functional, and that the documentation accurately reflects the library's capabilities.

`brahe` has already been used in a number of scientific publications [@eddyOptimal2024, @kim2025scalable]. It has also been used by aerospace companies such as Northwood Space, Xona Space [@reid2020satellite], and Kongsberg Satellite Services for mission analysis and planning. The Earth Observation satellite imaging prediction and task planning algorithms have been used by Capella Space and demonstrated on-orbit with their synthetic aperture radar (SAR) constellation [@stringham2019capella].

# Acknowledgements

We would like to acknowledge Mykel J. Kochenderfer. Without his mentorship, continued support and guidance, and funding, this project would never have happened. We also want to acknowledge Shaurya Luthra, Adrien Perkins, and Arthur Kvalheim Merlin for supporting the adoption of the project in their organizations and providing valuable feedback. Finally, we would like to thank the Stanford Institute for Human-Centered AI for their funding and support of the author during this work.

# References