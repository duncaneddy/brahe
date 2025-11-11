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
  - name: Mykel J. Kochenderfer
    orcid: 0000-0002-7238-9663
    affiliation: 1
affiliations:
 - name: Stanford University
   index: 1
date: 07 November 2025
bibliography: paper.bib
header-includes: |
    \usepackage{listings}
---
<!-- \lstdefinelanguage{python}{
    keywords=[3]{initialize!, transition!, evaluate!, distance, isevent, isterminal, environment},
    keywords=[2]{Nothing, Tuple, Real, Bool, Simulation, BlackBox, GrayBox, Sampleable, Environment},
    keywords=[1]{function, abstract, type, end},
    sensitive=true,
    morecomment=[l]{\#},
    morecomment=[n]{\#=}{=\#},
    morestring=[s]{"}{"},
    morestring=[m]{'}{'},
    alsoletter=!?,
    literate={,}{{\color[HTML]{0F6FA3},}}1
             {\{}{{\color[HTML]{0F6FA3}\{}}1
             {\}}{{\color[HTML]{0F6FA3}\}}}1
} -->

\lstset{
    language         = Python,
    backgroundcolor  = \color[HTML]{F2F2F2},
    basicstyle       = \small\ttfamily\color[HTML]{19177C},
    numberstyle      = \ttfamily\scriptsize\color[HTML]{7F7F7F},
    keywordstyle     = [1]{\bfseries\color[HTML]{1BA1EA}},
    keywordstyle     = [2]{\color[HTML]{0F6FA3}},
    keywordstyle     = [3]{\color[HTML]{0000FF}},
    stringstyle      = \color[HTML]{F5615C},
    commentstyle     = \color[HTML]{AAAAAA},
    rulecolor        = \color[HTML]{000000},
    frame=lines,
    xleftmargin=10pt,
    framexleftmargin=10pt,
    framextopmargin=4pt,
    framexbottommargin=4pt,
    tabsize=4,
    captionpos=b,
    breaklines=true,
    breakatwhitespace=false,
    showstringspaces=false,
    showspaces=false,
    showtabs=false,
    columns=fullflexible,
    keepspaces=true,
    numbers=none,
}

# Summary

<!-- \href{https://github.com/duncaneddy/brahe}{brahe} -->

[`brahe`](https://github.com/duncaneddy/brahe) is a modern astrodynamics dynamics library for research and engineering applications. The representation and prediction of satellite motion is the fundamental problem of astrodynamics. The motion of celestial bodies has been studied for centuries with initial equations of motion dating back to Kepler [@kepler1953epitome] and Newton [@newton1833philosophiae]. Current research and applications in space situational awareness, satellite task planning, and space mission operations require accurate and efficient numerical tools to perform coordinate transformations, model perturbations, and propagate orbits. `brahe` incorporates the latest conventions and models for time systems and reference frame transformations from the International Astronomical Union (IAU) [@hohenkerk2017iau] and International Earth Rotation and Reference Systems Service (IERS) [@petit2010iers]. It implements force models for Earth-orbiting satellites including atmospheric drag, solar radiation pressure, and third-body perturbations from the Sun and Moon [@vallado2001fundamentals; @montenbruckgill2000]. It also provides standard orbit propagation algorithms, including the Simplified General Perturbations (SGP) Model [@vallado2006revisiting]. Finally, it implements recent algorithms for fast, parallelized computation of ground station and imaging-target visibility [@eddy2021maximum], a foundational problem in satellite scheduling and mission planning.

With `brahe`, predicing upcoming satellite passes over ground stations or imaging targets can be accomplished in seconds and three lines of code.

\begin{lstlisting}[language=Python]
import brahe as bh
bh.initialize_eop()
passes = bh.location_accesses(
    bh.PointLocation(-122.4194, 37.7749, 0.0),  # San Francisco
    bh.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "active"),  # ISS
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0)
)
\end{lstlisting}

`brahe` allows users to quickly access Two-Line Element (TLE) data from Celestrak [@celestrak] and propagate orbits using the SGP4 dynamics model. This can be used to perform space situational awareness tasks such as predicting the orbits of all Starlink satellites over the next 24 hours.

\begin{lstlisting}[language=Python]
import brahe as bh
bh.initialize_eop()
starlink = bh.datasets.celestrak.get_tles_as_propagators("starlink", 60.0)
bh.par_propagate_to(starlink, bh.Epoch.now() + 86400.0) # Predict next 24 hours
\end{lstlisting}

The above routine can propagate orbits for all ~9000 Starlink satellites in approximately 1 minute 30 seconds on an M1 Max MacBook Pro with 10 cores and 64 GB RAM. Finally, the package provides direct, easy-to-use functions for low-level astrodynamics routines such as Keplerian to Cartesian state conversions and reference frame transformations.

\begin{lstlisting}[language=Python]
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
\end{lstlisting}

Another example application of `brahe` is visualizing the positions of GPS satellites in Earth orbit. The package provides built-in functions for generating 3D visualizations of satellite constellations using Plotly [@plotly].

![Visualization of all GPS Satellite Orbits](./gps_satellites_cropped.pdf)

# Statement of Need

While the core algorithms for predicting and modeling satellite motion have been known for decades, there is a lack of modern, open-source software that implements these algorithms in a way that is accessible to researchers and engineers. Generally, existing astrodynamics software packages have one or more barriers to entry for individuals and organizations looking to develop astrodynamics applications, and often leads to duplicated and redundant effort as researchers and engineers are forced to re-implement foundational algorithms.

Flagship commercial astrodynamics software like Systems Tool Kit (STK) [@stk] and FreeFlyer [@freeflyer] are individually licensed and closed-source. The licensing costs can be prohibitive for researchers, individuals, small organizations, and start-ups. Even for larger organizations, the per-node licensing cost can make large-scale deployment prohibitive. The closed-source nature of these packages makes it difficult to understand and verify the exact algorithms and model implementations, which is critical for high-stakes applications like space mission operations [@mcoMishap1999]. Major open-source projects like Orekit [@maisonobe2010orekit] and GMAT [@hughes2014gmat] provide extensive functionality, but are large codebases with steep learning curves, making quick-adoption and integration into projects difficult. Furthermore, Orekit is implemented in Java, which can be a barrier to adoption in the current scientific ecosystem with users who are more familiar with Python. GMAT uses a domain-specific scripting language and has limited documentation and examples, making it difficult for new users to get started. Libraries such as poliastro [@rodriguezPoliastro2022] and Open Space Toolkit (OSTk) [@ostk] provides Python interfaces, but their object-oriented architecture adds layers of abstraction that can make it difficult to adapt them to problems that outside their predefined modeling frameworks. Additionally, poliastro is no longer actively maintained, while OSTk only supports Linux environments and requires a specialized Docker environment to run. Other academic tools like Basilisk [@basilisk2020], provide high-fidelity modeling capabilities for full spacecraft guidance, navigation, and control (GNC) simulations, but are not directly distributed through standard package managers like PyPI and must be compiled from source to be used. Finally, these works often have limited documentation and usage examples, making it difficult for new users to get started.

`brahe` seeks to address these challenges by providing a modern, open-source astrodynamics library following design principles of the _Zen of Python_ [@peters2004zen]. The core functionality is implemented in Rust for performance and safety, with Python bindings for ease-of-use and integration with the scientific Python ecosystem. `brahe` is provided under an MIT License to encourage adoption and facilitate integration and extensibility. To further promote adoption and aid user learning, the library is extensively documented following the Diátaxis framework [@procida_diataxis]\textemdash every Rust and Python function documented with types and usage examples, there is a user guide that explains the major concepts of the library, and set of longer-form examples demonstrating how to accomplish common tasks. To maintain high code quality, the library has a comprehensive test suite for both Rust and Python. Additionally, all code samples in the documentation are automatically tested to ensure they remain functional, and that the documentation accurately reflects the library's capabilities.

`brahe` has already been used in a number of scientific publications [@eddyOptimal2024; @kim2025scalable]. It has also been used by aerospace companies such as Northwood Space, Xona Space [@reid2020satellite], and Kongsberg Satellite Services for mission analysis and planning. The Earth Observation satellite imaging prediction and task planning algorithms have been used by Capella Space and demonstrated on-orbit with their synthetic aperture radar (SAR) constellation [@stringham2019capella].

# Acknowledgments

We also want to acknowledge Shaurya Luthra, Adrien Perkins, and Arthur Kvalheim Merlin for supporting the adoption of the project in their organizations and providing valuable feedback. Finally, we would like to thank the Stanford Institute for Human-Centered AI for funding in part this work.

# References