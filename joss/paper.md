---
title: 'Brahe: A Modern Astrodynamics Library for Research and Engineering Applications'
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

[`brahe`](https://github.com/duncaneddy/brahe) is a modern astrodynamics library for research and engineering applications. The representation and prediction of satellite motion is the fundamental problem of astrodynamics, with initial formulations of equations of motion dating back to @kepler1953epitome and @newton1833philosophiae. Current research and applications in space situational awareness, satellite task planning, and space mission operations require accurate and efficient numerical tools to perform coordinate transformations, model perturbations, and propagate orbits. `brahe` incorporates the latest conventions and models for time systems and reference frame transformations from the International Astronomical Union (IAU) [@hohenkerk2017iau] and International Earth Rotation and Reference Systems Service (IERS) [@petit2010iers]. It implements force models for Earth-orbiting satellites including atmospheric drag, solar radiation pressure, and third-body perturbations from the Sun and Moon [@vallado2001fundamentals; @montenbruckgill2000], standard orbit propagation algorithms including the Simplified General Perturbations (SGP) Model [@vallado2006revisiting], and recent algorithms for fast, parallelized computation of ground station and imaging-target visibility [@eddy2021maximum].

With `brahe`, predicting upcoming satellite passes over ground stations or imaging targets can be quickly accomplished in three lines of code:

\begin{lstlisting}[language=Python]
import brahe as bh
bh.initialize_eop()
passes = bh.location_accesses(
    bh.PointLocation(-122.4194, 37.7749, 0.0),  # San Francisco
    bh.celestrak.CelestrakClient().get_sgp_propagator(catnr=25544, step_size=60.0), # ISS
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0)
)
\end{lstlisting}

`brahe` allows users to quickly retrieve satellite ephemeris data from Space-Track [@spacetrack] or Celestrak [@celestrak] and propagate orbits using different dynamics models. This can be used for space situational awareness tasks such as predicting the orbits of all Starlink satellites over the next 24 hours:

\begin{lstlisting}[language=Python]
import brahe as bh
bh.initialize_eop()
gp_records = bh.celestrak.CelestrakClient().get_gp(group="starlink")
starlink = [rec.to_sgp_propagator(step_size=60.0) for rec in gp_records]
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
state_eci = bh.state_koe_to_eci(state_kep, bh.AngleFormat.DEGREES)

# Define a time epoch
epoch = bh.Epoch(2024, 6, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)

# Convert ECI coordinates to ECEF coordinates at the given epoch
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

# Convert back from ECEF to ECI coordinates
state_eci_2 = bh.state_ecef_to_eci(epoch, state_ecef)

# Convert back from ECI to Keplerian elements
state_kep_2 = bh.state_eci_to_koe(state_eci_2, bh.AngleFormat.DEGREES)
\end{lstlisting}

Another example application of `brahe` is predicting and visualizing GPS satellite orbits. The package provides built-in functions for generating 2D and 3D visualizations of satellite constellations using Plotly [@plotly] and matplotlib [@Hunter2007].

![Visualization of all GPS Satellite Orbits](./gps_satellites_cropped.pdf)

# Statement of Need

While the core algorithms for predicting and modeling satellite motion have been known for decades, there is a lack of modern, open-source software that implements these algorithms in a way that is accessible to researchers and engineers. Generally, existing astrodynamics software packages have one or more barriers to entry, which leads to duplicated effort as researchers frequently choose to re-implement and validate common algorithms to avoid issue with other software. Flagship commercial astrodynamics software like Systems Tool Kit (STK) [@stk] and FreeFlyer [@freeflyer] are individually licensed and closed-source. The licensing costs can be prohibitive for researchers, individuals, small organizations, and start-ups. Even for larger organizations, the per-node licensing cost can make large-scale deployment prohibitive. Their closed-source nature makes it difficult to understand and verify the exact algorithms and model implementations, which is critical for high-stakes applications like space mission operations [@mcoMishap1999]. `brahe` targets students, researchers, engineers, and organizations who need a well-documented, easily-installed astrodynamics library that integrates with the Python scientific ecosystem and provides transparent, verifiable implementations of standard algorithms.

# State of the Field

Major open-source projects like Orekit [@maisonobe2010orekit] and GMAT [@hughes2014gmat] provide extensive functionality, but are large codebases with steep learning curves, making quick-adoption and integration into projects difficult. Furthermore, Orekit is implemented in Java, which adds friction when integrating with the current, Python-focused, scientific software ecosystem. Libraries such as poliastro [@rodriguezPoliastro2022] and Open Space Toolkit (OSTk) [@ostk] provides Python interfaces, but their object-oriented architecture adds layers of abstraction that can make it difficult to adapt them to problems that outside their supported problem-domains. Additionally, poliastro is no longer actively maintained and OSTk requires a Linux environment or specialized Docker container to run, limiting integration with other applications. Academic tools like Basilisk [@basilisk2020], provide high-fidelity modeling capabilities for full spacecraft guidance, navigation, and control (GNC) simulations, but are primarily focused on spacecraft-system modeling.

We evaluated contributing to these existing libraries but concluded that their architectural choices---deep class hierarchies, framework-specific abstractions, or platform constraints---introduced too many layers of complexity for users to easily understand, validate, extend, or adopt when seeking solutions to simple, fundamental astrodynamics problems. `brahe` was built from the ground up to provide a composable, function-oriented alternative distributed through standard channels (crates.io and PyPI) with full cross-platform support.

# Software Design

`brahe` addresses these challenges by providing a modern, open-source astrodynamics library following design principles of the _Zen of Python_ [@peters2004zen]. We designed `brahe` to be modular, yet highly-composable\textemdash emphasizing simple functions that can be chained together to build more complex functionality. To address decision-fatigue from astrodynamics modeling complexity without burdening new users, we adopt a design philosophy of ``Do the Rightest Thing'' philosophy: provide colloquially-named functions with reasonable default for modeling decisions (e.g., time systems, reference frames, perturbation models) that are current and accurate, but allow users to swap-out default implementations for specific models when needed. This lets beginners get reasonably accurate results quickly while preserving full control for advanced users.

The core is implemented in Rust for performance and memory safety, with Python bindings via PyO3 for integration with the scientific Python ecosystem. Documentation follows the Diátaxis framework [@procida_diataxis], with every function documented with API reference entries and usage examples. In addition, there is a conceptual user guide and long-form tutorials. All documentation code blocks are automatically tested on every commit to ensure they remain accurate and up-to-date. The library has a comprehensive test suite for both Rust and Python, and automated CI/CD for testing and distribution.

# Research Impact Statement

`brahe` has been used by current satellite missions after its adoption by aerospace companies such as Capella Space, Northwood Space, Xona Space [@reid2020satellite], and Kongsberg Satellite Services. Most notably, the satellite imaging prediction and task planning algorithms in `brahe` were used operationally by Capella Space [@stringham2019capella] in the first private US synthetic aperture radar (SAR) satellite constellation to predict communications and imaging opportunities.

The library has been used in scientific publications on satellite scheduling optimization [@eddyOptimal2024] and scalable satellite constellation management [@kim2025scalable]. It is provided under an MIT License to facilitate adoption and integration.

# AI Usage Disclosure

`brahe` development dates back to 2014, predating the availability of generative AI tools. Core architecture, modules, and implementations were developed without their use. Recently, AI tools have been used to unblock development of a new submodules and features (e.g., trajectory data structures, advanced numerical propagators, ephemeris client moduels) and to improve test coverage and documentation. In all cases, generated code was carefully reviewed, tested, and modified by the authors prior to merging to ensure correctness and maintainability. It was also verified against reference tests or values whenever possible. AI tools were not used in the writing of original submission, but were used to assist in reformatting to fit the new JOSS template.

# Acknowledgments

We want to acknowledge Shaurya Luthra, Adrien Perkins, and Arthur Kvalheim Merlin for supporting the adoption of the project in their organizations and providing valuable feedback. We thank the Stanford Institute for Human-Centered AI for funding in part this work.

# References
