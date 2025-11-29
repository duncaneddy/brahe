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
- [x] Documentation
    - [x] Automatically compiled and checked code samples
    - [x] User Guide
    - [x] API Reference
    - [x] Examples
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

## v0.2.0 - Numerical Integration 

- [x] Additional Integrators
    - [x] Fixed-step Runge-Kutta 4 (RK4)
    - [x] Adaptive-step Runge-Kutta-Fehlberg 4(5) (RKF45)
    - [x] Dormand-Prince 5(4) (DP54)
    - [x] Runge-Kutta-Nyström 12(10) (RKN1210)
- [x] State Transition Matrix (STM) propagation support
- [x] Package Maintenance
    - [x] Add `towncrier` for automated PR-request changelogs
- [x] Documentation for Numerical Integration Module

## v0.3.0 - Orbital Perturbations

- [x] Orbital Perturbations
    - [x] Spherical Harmonic Gravity
    - [x] Third Body Gravity
    - [x] Atmospheric Drag
    - [x] Solar Radiation Pressure
    - [x] Relativity
    - [x] Eclipse Models
    - [x] Low-fidelity planetary ephemerides
    - [x] High-fidelity planetary ephemerides (JPL DE430)

## v0.4.0 - Numerical Orbit Propagation

- [x] Numerical Orbit Propagation Module
    - [x] Control Input Support
        - [x] Impulsive Maneuvers
        - [x] Continuous Thrust
    - [x] Event Detection during Propagation
    - [x] Premade event detectors
- [x] General Numerical Propagation
- [x] Documentation for Orbital Perturbations and Numerical Propagation
- [x] Space Weather Data Management
    - [x] Data provider classes
        - [x] Static provider
        - [x] File provider
        - [x] Caching provider
- [x] NRLMSISE-00 Atmospheric Model Integration


----

## v1.0.0 - Stable Release with Foundational Features

- [x] Numerical Propagators
    - [x] User-defined Force Models
    - [x] Configurable Numerical propagator with default force models
- [x] Align with JOSS paper required release
- [x] Improved Interpolation Methods for Trajectories
    - [x] Lagrange Interpolation
    - [x] Hermite Interpolation
- [x] Mean Orbital Elements Support
    - [x] Conversion between Mean and Osculating elements
- [x] Relative Orbits
    - [x] Relative Orbit Representations
        - [x] RTN Cartesian States
        - [x] Relative Orbital Elements
    - [x] Cartesian to ROE Conversions
- [x] Walker Constellation Generator


----

## Planned Features

The following features are planned for future releases beyond v1.0.0. These features are prioritized based on user feedback and development resources. We welcome contributions and suggestions from the community to help shape the roadmap, as well as help with implementation.

- [ ] Spacetrack Datasets Module
- [ ] Initialize SGPPropagator from GP Elements
- [ ] Propagation of relative motion
    - [ ] Hill-Clohessy-Wiltshire (HCW) Equations
    - [ ] Numerical Relative Orbit Propagation
- [ ] Plotting Relative Orbits
    - [ ] RTN 3-panel plot
    - [ ] RTN 3D plot
- [ ] Estimation
    - [ ] Batch Least Squares Estimator
    - [ ] Extended Kalman Filter
    - [ ] Unscented Kalman Filter
    - [ ] Particle Filter
- [ ] TLE Estimation 
    - [ ] From GPS Observations
    - [ ] From Initial OPM State

----

## Considered, Unplanned Features

The following features have been considered for future releases but are not currently planned. They may be revisited based on user demand and development priorities.

- [ ] OEM File Support
    - [ ] Initialize Trajectory from OEM File
    - [ ] Export Trajectory to OEM File
- [ ] OMM File Support
    - [ ] Initialize Trajectory from OMM File
    - [ ] Export Trajectory to OMM File
- [ ] OPM File Support
    - [ ] Initialize Trajectory from OPM File
    - [ ] Export Trajectory to OPM File
- [ ] Additional Force Models
    - [ ] Tidal Force Models
        - [ ] Solid Earth Tides
        - [ ] Ocean Tides
    - [ ] Albedeo Force Models
- [ ] Probability of Collision Estimation
- [ ] Monte Carlo Simulation Framework
- [ ] Plotting Improvements
    - [ ] Access timeline plots
- [ ] Attitude dynamics module
    - [ ] Rigid Body Dynamics
    - [ ] Attitude Perturbations
    - [ ] Attitude Actuator Modeling
- [ ] Additional Examples and Tutorials
- [ ] Improved Atmospheric Models (EDU-only license)
    - [ ] NRLMSISE 2.0 Atmospheric Model Integration
    - [ ] NRLMSISE 2.1 Atmospheric Model Integration
    - [ ] DTM 2020 Atmospheric Model Integration
- [ ] Quality of Life Improvements
    - [ ] Improved error messages and handling
    - [ ] Enhanced logging capabilities
    - [ ] Automatically append script outputs to documentation examples & remove fixed outputs
