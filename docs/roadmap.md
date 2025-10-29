# Roadmap

Here you can find an overview of planned features and improvements for Brahe aligned to releases. This roadmap is subject to change based on user feedback, development priorities, and other factors.

----

## v0.1.0 - Overhaul & Rust Migration

This release completes the migration of the Brahe core library from Python to Rust, and matches capabilities with the previous pure-Python version.
It also contains many new features, improvements to core functionality, and better documentation.

- [x] Complete conversion to Rust core library
- [x] Datasets Module
    - [x] Celestrak
    - [x] Ground Stations
- [x] Trajectories Module
    - [x] Core Traits
    - [x] Static Trajectory
    - [x] Dynamic Trajectory
- [x] Two Line Element Validation & Parsing
- [x] Propagators Module
    - [x] StateProvider
    - [x] OrbitPropagator
    - [x] Keplerian Propagator
    - [x] SGP4 Propagator
- [x] Implement Access Computation
    - [x] Access Constraints
    - [x] Access Properties
    - [x] Parallel Access Computation
- [x] Plotting Module
    - [x] Ground Track Plots
    - [x] State Trajectory Plots
    - [x] 3D Orbit Plots
    - [x] Gabbard Plots
- [ ] Documentation
    - [x] Automatically compiled and checked code samples
    - [ ] User Guide
    - [x] API Reference
    - [ ] Example
        - [ ] Downloading TLE Data
        - [ ] Computing Ground Station Accesses
        - [ ] Computing Imaging Opportunities
        - [ ] Visualizing Orbital Trajectories
- [x] CI/CD Pipeline
    - [x] Documentation-only fast releases
    - [x] MacOS testing and builds
    - [x] Windows testing and builds
    - [x] Automated latest release deployment
    - [x] Automated wheel builds
- [x] Package Quality
    - [x] Add Code of Conduct
    - [x] Add Contributing Guide
- [x] Implement `Epoch.now()` initialization
- [x] Caching Earth Orientation Parameters (EOP) data providers
- [x] Add consistent package caching strategy for data files

----

## v0.2.0 - Advanced Ground Station Operations

- [ ] Spacetrack Datasets Module
- [ ] Initialize SGPPropagator from GP Elements
- [ ] OEM File Support
    - [ ] Initialize Trajectory from OEM File
    - [ ] Export Trajectory to OEM File
- [ ] SP3 File Support
    - [ ] Initialize Trajectory from SP3 File
    - [ ] Export Trajectory to SP3 File
- [ ] Plotting Improvements
    - [ ] Access timeline plots
- [ ] Improved Interpolation Methods for Trajectories
    - [ ] Lagrange Interpolation
    - [ ] Hermite Interpolation
- [ ] Package Maintenance
    - [ ] Add `towncrier` for automated PR-request changelogs

----

## v0.3.0 - Numerical Integration

This release adds support for numerical orbit propagation using common perturbation models. It also introduces additional datasets and improves existing functionality.

- [ ] Add support for numerical integraiton
    - [ ] Fixed-step Runge-Kutta methods (RK4, RKF45)
    - [ ] Adaptive-step integrators
- [ ] Orbital Perturbations
    - [ ] Spherical Harmonic Gravity
    - [ ] Third Body Gravity
    - [ ] Atmospheric Drag
    - [ ] Solar Radiation Pressure
    - [ ] Relativity
    - [ ] Eclipse Models
- [ ] Numerical Orbit Propagation Module
    - [ ] Control Input Support
        - [ ] Impulsive Maneuvers
        - [ ] Continuous Thrust
- [ ] Space Weather Data Management
    - [ ] Data provider classes
        - [ ] Static provider
        - [ ] File provider
        - [ ] Caching provider
- [ ] NRLMSISE 2.0 Atmospheric Model Integration
- [ ] Documentation Improvements

----

## v0.4.0 - Estimation & Relative Orbits

- [ ] Mean Orbital Elements Support
    - [ ] Conversion between Mean and Osculating elements
- [ ] Relative Orbits
    - [ ] Relative Orbit Representations
        - [ ] RTN Cartesian States
        - [ ] Relative Orbital Elements
    - [ ] Propagation of relative motion
        - [ ] Hill-Clohessy-Wiltshire (HCW) Equations
        - [ ] Tschauner-Hempel Equations
    - [ ] Plotting Relative Orbits
        - [ ] RTN 3-panel plot
        - [ ] RTN 3D plot
- [ ] Estimation
    - [ ] Batch Least Squares Estimator
    - [ ] Extended Kalman Filter
    - [ ] Unscented Kalman Filter
    - [ ] Particle Filter

----

## v1.0.0 - Stable Release

- [ ] Address open issues and feature requests

----

## Considered, Unplanned Features

The following features have been considered for future releases but are not currently planned. They may be revisited based on user demand and development priorities.

- [ ] Tidal Force Models
    - [ ] Solid Earth Tides
    - [ ] Ocean Tides
- [ ] Albedeo Force Models
- [ ] Probability of Collision Estimation
- [ ] Monte Carlo Simulation Framework
- [ ] Move all rust functions over to generic float types to support f32 and f64
- [ ] Event Detection during Propagation
- [ ] Additional Examples and Tutorials
    - [ ] Mission Analysis Examples
    - [ ] Advanced Access Computation Examples
    - [ ] Custom Perturbation Models
- [ ] Improved Documentation
- [ ] Attitude dynamics module
- [ ] WASM bindings for web-based applications