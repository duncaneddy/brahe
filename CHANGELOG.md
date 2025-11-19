# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- towncrier release notes start -->

## [0.3.0] - 2025-11-18
### Added

- - Added `Integrators` submodule with complete rust implementation, python bindings, documentation, and examples
  - Support for `RK4`, `RKF45`, `DP54`, and `RKN1210` numerical integration methods
  - License validation and compliance checks. Package is now automatically checked to ensure all dependencies have permissive, commercially-adoptable licenses [#63](https://github.com/duncaneddy/brahe/pull/63)
- - NAIF Development Ephemeride dataset download and caching
  - High-accuracy DE440s-based ephemeris prediction
  - High-accuracy DE440s-based third-body acceleration prediction
  - Python bindings for orbit dynamics functions [#68](https://github.com/duncaneddy/brahe/pull/68)
- - Added `geo_sma` function to directly return the semi-major axis needed for a geostationary orbit. [#69](https://github.com/duncaneddy/brahe/pull/69)

### Changed

- - Moved mathematics capabilities from `utils` submodule to dedicated `math` submodule.
  - Bumped version to `0.2.0` for release
  - Removed Rust test sections from coverage reporting [#63](https://github.com/duncaneddy/brahe/pull/63)
- - Orbit dynamics functions which previously required a position-only vector can now accept either a position-only or a state vector. Conversion will be handled by the `IntoPosition` trait. [#68](https://github.com/duncaneddy/brahe/pull/68)

### Fixed

- - Fix inconsistencies in Python API reference header levels
  - Missing Ground station
  - Miscellaneous documentation improvements and fixes [#63](https://github.com/duncaneddy/brahe/pull/63)

## [0.2.0] - 2025-11-17
### Added

- - Added `relative_motion` submodule to contain functionality related to relative motion
  - Implemented RTN rotation and state transformations: `rotation_eci_to_rtn`, `rotation_rtn_to_eci`, `state_eci_to_rtn`, and `state_rtn_to_eci`
  - Implemented Relative Orbital Element (ROE) state transformations `state_oe_to_roe` and `state_roe_to_oe`
  - Added `util::math` functions `sqrtm` and `spd_sqrtm` to calculate matrix square root
  - Added `util::math` functions `oe_to_radians` and `oe_to_degrees` to reduce duplication in converting angle values in orbital element calcualtions [#57](https://github.com/duncaneddy/brahe/pull/57)
- - Implement `CovarianceProvider` trait for `OrbitTrajectory`. Provides `covariance`, `covariance_eci`, `covariance_gcrf`, and `covariance_rtn`
  - Add `interpolation` submodule to store consistent covariance interpolation methods
  - Implement covariance rotation from ECI to RTN frame
  - Extends `OrbitTrajectory` to optionally store covariance information. [#59](https://github.com/duncaneddy/brahe/pull/59)

### Changed

- - Don't run PR tests on release CHANGELOG update. [#49](https://github.com/duncaneddy/brahe/pull/49)
- - Unit test and PR test workflows now cancel in-progress runs if a new commit lands
  - Added concurrency guards to auto-merge workflows. [#53](https://github.com/duncaneddy/brahe/pull/53)
- - Moved where internal vector/matrix (e.g. `SMatrix3`, `SVector6`) type aliases are defined from `coordinates` submodule to `utils` [#57](https://github.com/duncaneddy/brahe/pull/57)

### Fixed

- - Don't trigger multiple changelog merges on changelog a PR. [#49](https://github.com/duncaneddy/brahe/pull/49)
- - Only trigger auto-merge on `opened` due to changelog PRs not issuing `labeled` events [#51](https://github.com/duncaneddy/brahe/pull/51)
- - Added additional triggers to auto-merge workflows to ensure they are properly triggered. [#53](https://github.com/duncaneddy/brahe/pull/53)
- - Fixed assorted typos and errors in docstrings related to covariance interpolation features. [#61](https://github.com/duncaneddy/brahe/pull/61)
- - Fixes package-data update CI workflow by moving to auto-PR, auto-merge approach [#64](https://github.com/duncaneddy/brahe/pull/64)

## [0.1.3] - 2025-11-12
### Added

- - Added python missing bindings for `states_icrf` and `states_gcrf` for `KeplerianPropagator` and `SGPPropagator`
  - Added additional tests across various module to improve test coverage. [#36](https://github.com/duncaneddy/brahe/pull/36)

### Changed

- - Refactor `frames.rs` file into submodule with subfiles for long-term maintainability. [#14](https://github.com/duncaneddy/brahe/pull/14)
- - Automatically create and merge PRs for changelog updates [#16](https://github.com/duncaneddy/brahe/pull/16)
- - Auto-merge changelog PRs
  - Auto-merge dependabot PRs
  - Expand dependabot to cover python and rust packages [#26](https://github.com/duncaneddy/brahe/pull/26)
- - Bump package version to `v0.1.3` [#34](https://github.com/duncaneddy/brahe/pull/34)
- - Skip unit test suite on auto-generated changelog PRs. [#44](https://github.com/duncaneddy/brahe/pull/44)

### Fixed

- - PR changelogs were not being incorporated into the package changelog due to main-branch protection [#16](https://github.com/duncaneddy/brahe/pull/16)
- - Stop generation changelog PRs for auto-generated changelog PRs [#26](https://github.com/duncaneddy/brahe/pull/26)
- - Fixed issue with release pipeline release note generation [#38](https://github.com/duncaneddy/brahe/pull/38)
- - Fix auto-merge for changelog PRs by using PAT [#40](https://github.com/duncaneddy/brahe/pull/40)
- - Fix auto-merge workflow to accept PAT owner as actor. [#42](https://github.com/duncaneddy/brahe/pull/42)
- - Fix workflow release step to use workflow PAT and declare base branch [#46](https://github.com/duncaneddy/brahe/pull/46)
