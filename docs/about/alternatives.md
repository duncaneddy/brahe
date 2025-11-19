# History, Inspiration, and Alternatives, Inspiration and Comparisons

## Intro

There have been many astrodynamics libraries developed over the years, each with its own strengths and weaknesses. Brahe draws inspiration from several of these projects while aiming to provide a unique combination of features, performance, and usability.


## History

The origins of Brahe started in 2014 when I first started graduate school at Stanford. At the time, I was working on a project that required high-fidelity orbit propagation and I found that high-fidelity tools were either inaccessible (proprietary, expensive, or both) or difficult to use (poor documentation, complex APIs). I wanted to create a tool that was both powerful and easy to use to make it easy for researchers and engineers to ask and answer questions they really cared about without needing to become experts in astrodynamics or software engineering themselves.

As I changed organizations and projects over the years, I kept running into the same problem of a lack of open, permissively license, astrodynamics software. So when I joined a new organization I'd write a new astrodynamics library using the core algorithms and data structures in new languages to make astrodynamics accessible to those organizations. Through this process, I kept refinding my approach to the software design --- learning and improving with each iteration. 

When I returned to finish my PhD in 2019, I decided to take the best parts of these previous implementations and create a new library from scratch that would be open and accessible to all so that others wouldn't have to go through the same process of reinventing the wheel. In 2023 I began rewriting the library in Rust to take advantage of its performance, safety, and modern language features. The result is Brahe as it exists today.

## Notable Alternatives

### [STK (Systems Tool Kit)](https://www.agi.com/products/stk)

Systems Tool Kit (STK) is likely the most well-known commercial astrodynamics software package. It is extremely well validated and provides built-in visualization capabilities. It also supports a wide variety of use-cases and has workflows for built-in analysis. However, it is expensive and closed-source which limits its accessibility and extensibility.

!!! tip "Inspiration"
    The ability to quickly perform access computations in Brahe draws inspiration from STK's built-in access analysis capabilities.

### [FreeFlyer](https://ai-solutions.com/)

FreeFlyer is another commercial astrodynamics software package. It has been used for trajectory design and mission analysis for many high-profile space missions. It is also known for its scripting capabilities. a.i. Solutions generously provides free licenses for academic users. However it is closed-source which can limit its accessibility and extensibility.

### [Orekit](https://www.orekit.org/)

Orekit is a popular open-source astrodynamics library written in Java. It provides a wide range of features, is well-validated, and well-documented. There are also Python bindings available through different wrappers, though all require a Java runtime.

While Orekit is powerful, its Java foundation makes it difficult to integrate into modern Python scientific computing ecosystem.

!!! tip "Inspiration"
    Brahe's open-source permissive licensing was inspired in part by Orekit's approach to open-source software.

### [GMAT (General Mission Analysis Tool)](https://sourceforge.net/projects/gmat/)

GMAT is an open-source astrodynamics software package developed by NASA. It provides a wide range of features and is well-validated. It also provides a custom scripting language for mission design and analysis.

GMAT documentation can be hard to find and navigate making it difficult to learn. It is also distributed as it's own standalone application, as opposed to a common library, which makes it difficult to integrate into new software.

!!! tip "Inspiration"
    GMAT is a wonderful example of open-source astrodynamics software that provides transparency into its algorithms and implementations.

### [poliastro](https://poliastro.github.io/)

poliastro is an open-source astrodynamics library written in Python. It is designed to be easy-to-use and integrates well with the scientific Python ecosystem. It provides a wide range of features for orbit propagation, maneuver planning, and mission design. It is well-documented with many examples and tutorials, though is no longer actively maintained.

!!! tip "Inspiration"
    Brahe's Python plotting routines draw inspiration from poliastro's built-in plotting capabilities.

### [Skyfield](https://rhodesmill.org/skyfield/)

Skyfield is an excellent open-source library for high-precision astronomy and satellite tracking written in Python. It is designed to be easy-to-use and provides accurate calculations for positions of planets, stars, and satellites. It is well documented, with lots of examples and well-maintained.

It primarily focuses on ephemeris calculations and star tracking for astronomy, but also supports satellite orbit propagation using two-line element (TLE) data.

!!! tip "Inspiration"
    Skyfield adopts a zen-of-python approach to its API design which influenced Brahe's Python API design. It prioritizes code readability and simplicity to make it easy for users to understand and use the library effectively.

### [Nyx Space](https://nyx.space/)

Nyx Space is a modern astrodynamics library written in Rust with python bindings. It focuses on validation and verification of its algorithms and provides high-performance implementations of common astrodynamics tasks. It's general focus in on trajectory design and orbit determination for interplanetary missions, though it also supports Earth-orbiting missions as well.

Nyx Space is licensed under an AGPLv3 license which requires derivative works to also be open-sourced under the same license, which can limit its use in commercial applications. Commercial licenses are available from Nyx Space for those who want to use it in closed-source applications.

!!! tip "Inspiration"
    Brahe's Rust implementation was inspired in part by Nyx Space's approach to using Rust for astrodynamics.

    Additionally Brahe uses the [anise](https://github.com/nyx-space/anise) crate from the nyx ecosystem for working with NAIF SPICE kernels and ephemerides.

### [Basilisk](https://avslab.github.io/basilisk/)

Basilisk is an open-source astrodynamics and spacecraft simulation framework developed by the [Autonomous Vehicle Systems (AVS) Lab](https://hanspeterschaub.info/AVSlab.html) at the University of Colorado Boulder. It provides a modular architecture for simulating spacecraft dynamics, control systems, and mission scenarios though a component-based approach.

Basilisk is primarily focused on spacecraft simulation and control system design, making it well-suited for simulating complex spacecraft missions with multiple interacting subsystems. However it is not distributed through common package managers which can make it difficult to integrate into new software projects.