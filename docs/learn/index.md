# User Guide

Here you will find the introductory and conceptual documentation for Brahe. This is the main user guide for the package, and is intended to help you get started with using Brahe for your work.

!!! tip "Contributions Welcome"
    If you find something missing or unclear in this documentation, particularly feedback on the conceptual flow, how things are introduced or otherwise, please consider contributing! We welcome contributions of all kinds, including:

    - Reporting issues or suggesting improvements
    - Writing new documentation pages or improving existing ones
    - Adding examples or tutorials
    - Enhancing the codebase with new features or bug fixes

    Check out our [contributing guide](https://duncaneddy.github.io/brahe/contributing/) to get started.

## Module Structure

The Brahe package is organized into several key modules, each providing providing some core functionality. We can generally think of these modules as falling into four main categories: Foundational Modules, Orbit & Attitude Representations, State Propagation & Dynamics, and Applications. Below is an overview of the main modules and their purposes:

### Foundational Modules

These modules provide the basic building blocks for the package, including utilities for time handling, constants, and Earth Orientation Parameters (EOP).

- **Constants**: This module contains physical and mathematical constants used throughout the package.
- **Time**: This module provides tools for representing and dealing with time. It provides the ubiquitous [Epoch](../library_api/time/epoch.md) class which is the basis for all time handling in Brahe.
- **EOP**: This module handles Earth Orientation Parameters, which are essential for accurate coordinate transformations and orbit propagation. An Earth Orientation Provider is required for many operations in Brahe. There are multiple kinds provided, but `initialize_eop()` is the easiest way to get started.

### Orbit & Attitude Representations

These modules help transform between different state and coordinate representations for both spacecraft orbits (position and velocity) and attitudes (orientation).

- **Coordinates**: This module provides functions to convert between different coordinate systems, such as Cartesian, Geocentric, Geodetic, and Topocentric coordinates.
- **Frames**: This module deals with reference frames, including Earth-Centered Inertial (ECI) and Earth-Centered Earth-Fixed (ECEF) frames, and provides rotation matrices and state transformations between them.
- **Orbits**: This module provides functions for working with orbital elements, including conversions between Keplerian elements and Cartesian states, as well as handling special orbit types like Sun-synchronous orbits and Two-Line Elements (TLEs).
- **Attitude**: This module provides tools for representing and manipulating spacecraft attitudes using rotation matrices, quaternions, Euler angles, and Euler axes.

### State Propagation & Dynamics

These modules focus on propagating spacecraft states over time using various dynamics models. It also provides methods for representing these state trajectories.

- **Trajectories**: This module defines traits and structures for representing dynamics trajectories, including orbit trajectories.
- **Keplerian Propagator**: This module implements a simple Keplerian propagator for orbit propagation based on Kepler's laws.
- **SGP Propagator**: This module implements the SGP (Simplified General Perturbations) propagator, a widely used method for propagating Earth-orbiting satellites, in particular those defined by TLEs.
- **Orbit Dynamics**: This module implements various force models used in orbit propagation, such as gravity, drag, solar radiation pressure, and more.

### Applications**

These modules provide higher-level functionalities for specific applications, such as working with datasets, computing access windows, and plotting.

- **Datasets**: This module provides access to common datasets used in space applications, such as ground stations and satellite ephemeris catalogs (e.g., Celestrak).
- **Access Computation**: This module provides tools for computing access windows between satellites and terrestrial locations, including defining access constraints and computing access properties.
- **Plotting**: This module offers functions for visualizing satellite data, including ground tracks, state vectors, orbital elements, and access geometry.

Each of these modules are defined to have composable, interoperable interfaces so that you can easily combine functionality from different modules to accomplish your tasks. As you explore the documentation further, you'll find detailed explanations and examples for each module to help you understand how to use them effectively.

For detailed information on all functions, classes, and methods available in each module, please refer to the [Python API Reference](../library_api/index.md) and [Rust API Reference](https://docs.rs/brahe).