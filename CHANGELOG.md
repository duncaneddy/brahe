# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Brahe's versioning and deprecation policy is modeled on [NumPy's policy](https://numpy.org/doc/stable/dev/depending_on_numpy.html) rather than strict SemVer — see the [versioning documentation](https://duncaneddy.github.io/brahe/latest/about/versioning/) for the full policy, including the transitional deprecation window currently in effect.

Each release groups entries under the Keep a Changelog section headings in the order **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**. Entries in **Deprecated** correspond to APIs that still work in this release but emit a `DeprecationWarning` or `FutureWarning`; entries in **Removed** list APIs that previous releases had deprecated.

<!-- towncrier release notes start -->

## [1.4.1] - 2026-04-21
### Changed

- Improve documentation around versioning practices. [#282](https://github.com/duncaneddy/brahe/pull/282)
- Add deprecation section to changelog [#284](https://github.com/duncaneddy/brahe/pull/284)
- Bump version for v1.4.1 release [#295](https://github.com/duncaneddy/brahe/pull/295)

### Fixed

- Fix a permissions issue that pervented CHANGELOG validation from running on fork PRs [#293](https://github.com/duncaneddy/brahe/pull/293)

## [1.4.0] - 2026-04-17
### Added

- Added [brahe-mcp](https://github.com/duncaneddy/brahe-mcp) information to docs [#278](https://github.com/duncaneddy/brahe/pull/278)

### Changed

- Refactored repository structure to use a Rust workspace to breakup the python module from the core rust module. This pure-rust package consumers to avoid having to build the `cdylib`, which is only needed for python module builds. [#278](https://github.com/duncaneddy/brahe/pull/278)
- `pyo3` and `numpy` are now optional dependencies, activated via the `python` feature flag. Pure-Rust consumers that do not need the Python bindings no longer compile these crates.
- Split the repository into a Cargo workspace so the PyO3 bindings live in a dedicated `brahe-py` member crate at `crates/brahe-py/`. The published `brahe` crate now builds only an `rlib`, eliminating the unnecessary `cdylib` artifact previously produced for every pure-Rust consumer. Rust downstream (`use brahe::*`) and Python downstream (`import brahe`) are unchanged.


### Fixed

- Fixed issue that would cause tests and auto-merge to not run on dependabot PRs [#278](https://github.com/duncaneddy/brahe/pull/278)

## [1.3.4] - 2026-04-15
### Fixed

- Fix failing celestrak queries by migrating from celestrak.com to celestrak.org [#272](https://github.com/duncaneddy/brahe/pull/272)

## [1.3.3] - 2026-04-10
### Added

- Add JD_J2000 constant to python bindings [#264](https://github.com/duncaneddy/brahe/pull/264)

## [1.3.2] - 2026-04-06
### Changed

- Updated JOSS manuscript to comply with latest guidelines [#256](https://github.com/duncaneddy/brahe/pull/256)

### Fixed

- Fixed issue with SGP4 propagators erroring during parallel propagation on satellite reentry. Addressed by adding a `termination_error` field to SGPPropagators which will be documented and stop propagation rather than erroring all propagation.
  Issued with documentation builds in CI failing due to dependency changes. Fixed by updating `uv.lock` [#256](https://github.com/duncaneddy/brahe/pull/256)

## [1.3.1] - 2026-03-29
### Added

- Added retry logic and exponential back-off to Celestrak client
  Added process-based rate-limiting to Celestrak client [#244](https://github.com/duncaneddy/brahe/pull/244)

### Changed

- Set default celestrak client cache age to 2 hours. [#244](https://github.com/duncaneddy/brahe/pull/244)
- Bump version for `v1.3.1` [#248](https://github.com/duncaneddy/brahe/pull/248)

### Fixed

- Pointed `mike` documentation deployment to use `properdocs.yml` [#244](https://github.com/duncaneddy/brahe/pull/244)
- Fix `mike set-default` invocation to use `properdocs.yml` config [#246](https://github.com/duncaneddy/brahe/pull/246)

## [1.3.0] - 2026-03-29
### Added

- Added python constructors for CCSDS OPM and OMM file types [#225](https://github.com/duncaneddy/brahe/pull/225)
- Added dedicated `celestrak` and `spacetrack` submodules to the brahe CLI for querying and interfacing with space track.
  Added `to_dict()` and `to_json()` methods for GPRecord python module to enable easy interoperability [#227](https://github.com/duncaneddy/brahe/pull/227)
- Added additional python test coverage [#228](https://github.com/duncaneddy/brahe/pull/228)
- Added `estimation` module with support for `EKF`, `UKF`, and `BLS` (batched least-squares) filters
  Added estimation plotting routines
  Added implementations of magnetic field models. In particular the IGRF and WMMHR models.
  Added `from_unix_timestamp` and `unix_timestamp` methods to support interoperability with native unix time systems
  Added sponsor logo [#241](https://github.com/duncaneddy/brahe/pull/241)

### Changed

- Added additional test coverage [#225](https://github.com/duncaneddy/brahe/pull/225)
- Renamed `MJD2000` constant to `MJD_J2000` to align with `JD_J2000` naming [#231](https://github.com/duncaneddy/brahe/pull/231)
- Improved documentation code samples by auto-generating, capturing, and rendering script outputs so values are guaranteed to be kept up to date. [#234](https://github.com/duncaneddy/brahe/pull/234)
- Migrated to `properdocs` from `mkdocs` for documentation due to instability in project. [#241](https://github.com/duncaneddy/brahe/pull/241)

### Fixed

- Fixed issue with anomaly conversion CLI not properly using `use_degrees` [#225](https://github.com/duncaneddy/brahe/pull/225)

## [1.2.0] - 2026-03-21
### Added

- Added polygon tessellation for large area access computation [#199](https://github.com/duncaneddy/brahe/pull/199)
- Added additional benchmarks and comparisons to OreKit in `benchmarks/comparisons` [#209](https://github.com/duncaneddy/brahe/pull/209)
- Add support for CCSDS OEM, OMM, OPM (blue-book) and CDM (pinkbook) types. Methods exist to load, access, modify, and create these data objects. [#214](https://github.com/duncaneddy/brahe/pull/214)
- Add full support for XML and JSON output across OEM, OPM, and OMM files. [#222](https://github.com/duncaneddy/brahe/pull/222)

### Changed

- Have Propagators set a UUID by default
  Have Locations set a UUID by default
  Migrated default-generated UUIDs to UUIDv7 [#202](https://github.com/duncaneddy/brahe/pull/202)
- Removed `release` environment flag from github release workflows
  Only retain last 3 minor documentation verions to prevent unbounded growth of github pages deployments. [#209](https://github.com/duncaneddy/brahe/pull/209)
- Cleaned up space-track and celestrak documentation to improve readability and understandability [#214](https://github.com/duncaneddy/brahe/pull/214)
- Added coverage reporting for python bindings and module [#218](https://github.com/duncaneddy/brahe/pull/218)
- Improved CCSDS module round trip test coverage
  Ensures that default OEM to Trajectory conversion returns a `DOrbitTrajectory` so that it can immediately be used with access computation. [#222](https://github.com/duncaneddy/brahe/pull/222)

### Fixed

- Reverted compilation regression due to dependabot update. Ignored packages to prevent future regressions [#194](https://github.com/duncaneddy/brahe/pull/194)
- Skip running CI-related tests on fork-PRs so that tests pass. [#204](https://github.com/duncaneddy/brahe/pull/204)
- Changes access module type annotations from `List` to `Sequence` so that LSPs don't erroneously report a warning that `location_accesses` and other functions might mutate the list.
  Fix issue where python bindings for access properties would return a copy of the underlying data so mutations would not persist. Introduce `AccessPropertyView` so that it is now properly possible to access, set, and modify access properties in python. [#206](https://github.com/duncaneddy/brahe/pull/206)
- Fixed issue with Gabbard plots not generating properly [#218](https://github.com/duncaneddy/brahe/pull/218)
- Fix coverage reporting [#220](https://github.com/duncaneddy/brahe/pull/220)

## [1.1.4] - 2026-03-05
### Fixed

- Fix regression from dependabot update
  Included type annotations in package generation [#188](https://github.com/duncaneddy/brahe/pull/188)

## [1.1.3] - 2026-03-04
### Added

- Added `SubdivisionConfig` a subfield of `AccessSearchConfig` to control how subdivisions are created [#181](https://github.com/duncaneddy/brahe/pull/181)

### Changed

- Migrated to PyO3 `v0.28`
  Update Cargo dependencies [#179](https://github.com/duncaneddy/brahe/pull/179)
- Removed `time_step` parameter from internal `find_access_windows` method. Configuration of stepping and subdivision is now unified as a property of `AccessSearchConfig` [#181](https://github.com/duncaneddy/brahe/pull/181)

### Fixed

- Fixed issue in dataframe creation from dependabot update. [#179](https://github.com/duncaneddy/brahe/pull/179)

## [1.1.2] - 2026-02-14
### Fixed

- Fix spacetrack client erroring out when query returns a lot of results
- Fix spacetrack `cdm_public` route having wrong parent stem [#169](https://github.com/duncaneddy/brahe/pull/169)

## [1.1.1] - 2026-02-14
### Added

- Added ability to configure `TrajectoryMode` on `SGP4Propagator` for applications where memory-use management is key. [#167](https://github.com/duncaneddy/brahe/pull/167)

### Changed

- The JOSS paper submission used code examples with the older celestrak API. This PR updates the code examples to use the updated, stabilized API.
- Slightly adjust language in paper to avoid silly page overflow [#165](https://github.com/duncaneddy/brahe/pull/165)
- Changed default interpolator used by `NumericalOrbitPropgator` for event detection from linear to 3rd degree Hermite. [#167](https://github.com/duncaneddy/brahe/pull/167)

### Fixed

- Make `.current_epoch()` property a method, not property in python bindings.
- Fixed issue with event detection system in `NumericalOrbitPropagator` where it not correctly process events co-located with the current time, and would sometimes miss events in between time steps. [#167](https://github.com/duncaneddy/brahe/pull/167)

## [1.1.0] - 2026-02-10
### Added

- Added `par_propagate_to_d` method. [#109](https://github.com/duncaneddy/brahe/pull/109)
- Add Github issue templates for bugs and feature requests [#113](https://github.com/duncaneddy/brahe/pull/113)
- Add arXiv citation to README and documentation. [#124](https://github.com/duncaneddy/brahe/pull/124)
- Modified python bindings to support access computation with numerical orbit propagators. [#126](https://github.com/duncaneddy/brahe/pull/126)
- Added `spacetrack` client module, enabling direct interaction with space-track.org APIs [#150](https://github.com/duncaneddy/brahe/pull/150)
- Added BSL-1.0 to allowed dependency license list
- Added [GCAT](https://planet4589.org/space/gcat/web/cat/index.html) dataset interface [#152](https://github.com/duncaneddy/brahe/pull/152)
- Added simplified `CelestrakClient` interface to bridge gap between new interface and old one. [#157](https://github.com/duncaneddy/brahe/pull/157)

### Changed

- Bump version for 1.0.1 release [#106](https://github.com/duncaneddy/brahe/pull/106)
- Renamed `par_propagate_to` to `par_propagate_to_s`
- Move SGPPropagator to D-Type event detection. [#109](https://github.com/duncaneddy/brahe/pull/109)
- Update documentation to add [JOSS](https://joss.theoj.org/) paper submission badge. [#113](https://github.com/duncaneddy/brahe/pull/113)
- Update JOSS workflow to export tex artifacts to support arXiv upload. [#122](https://github.com/duncaneddy/brahe/pull/122)
- Moved all Brahe errors in python bindings from `OSError` to `BraheError` to make them more distinguishable and easy to handle. [#126](https://github.com/duncaneddy/brahe/pull/126)
- Updated JOSS paper with [new required sections](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future). [#137](https://github.com/duncaneddy/brahe/pull/137)
- Enable manual triggering of latest documentation build in CI for handling EOP-update edge cases for releases. [#139](https://github.com/duncaneddy/brahe/pull/139)
- Overhauled `celestrak` client. The `celestrak` client now uses a common type and query structure shared with the `spacetrack` module. In particular, the `GPRecord` serves as the common representation of OMM elements returned by both sites. This enables greater interoperability between other library modules that can consume or act on `GPRecord` types. The `celestrak` module is now it's own, stand-alone, module independent of `datasets`.
- Migrate developer workflow from `make.py` to [just](https://github.com/casey/just)
- Remove dependency on [BTreeCursor](https://github.com/rust-lang/rust/issues/107540) nightly feature, allowing crate to compile with stable rust. [#150](https://github.com/duncaneddy/brahe/pull/150)
- Adjusted `CelestrakQuery` python bindings to properly use properties instead of methods for specific dataset queries. **Example:** New: `CelestrakQuery.gp` vs Old: `CelestrakQuery.gp()` [#157](https://github.com/duncaneddy/brahe/pull/157)

### Fixed

- Enabled event detection in `par_propagate_to` method in Python [#109](https://github.com/duncaneddy/brahe/pull/109)
- Fixed broken documentation links in readme. (Thank you Stuart Bartlett for pointing it out) [#129](https://github.com/duncaneddy/brahe/pull/129)
- Fix broken links in main documentation landing page. [#131](https://github.com/duncaneddy/brahe/pull/131)
- Fix provider names for SSC stations in NASA Near Earth Network [#143](https://github.com/duncaneddy/brahe/pull/143)
- Fix https://duncaneddy.github.io/brahe/ to redirect to the current default root URL. [#147](https://github.com/duncaneddy/brahe/pull/147)
- Fixed tables in ephemeris documentation not being justified. [#152](https://github.com/duncaneddy/brahe/pull/152)
- Reverted `pyo3` version update due to incompatibility with `numpy` depdency
- Prevented dependabot from auto-updating `pyo3` [#157](https://github.com/duncaneddy/brahe/pull/157)
- Properly pass environment secrets from the `release` workflow to child-workflows [#159](https://github.com/duncaneddy/brahe/pull/159)

## [1.0.1] - 2025-12-04
### Added

- Added `from_omm_elements` to `SGPPropagator`. Supports direct initialization of propagator from OMM elements without needing to go through TLE lines. This is both more accurate as well as more robust to TLE ID depletion. [#102](https://github.com/duncaneddy/brahe/pull/102)
- Add support for event detection to `SGPPropagator`
- Add `AOIEntryEvent` and `AOIExitEvent` as premade event detectors. [#104](https://github.com/duncaneddy/brahe/pull/104)

### Fixed

- Fix typo in JOSS paper title [#99](https://github.com/duncaneddy/brahe/pull/99)

## [1.0.0] - 2025-11-29
### Added

- `{D}NumericalOrbitPropagator` and python bindings
- `{D}NumericalPropagator` and python bindings
- Event detection system for numerical propagators to find events during propagation [#89](https://github.com/duncaneddy/brahe/pull/89)
- Mean to osculating, osculating to mean orbital element transformations
- Update OrbitStateProviders to provide `state(s)_koe_mean` and `state(s)_koe_osc`
- Implement ECI<>ROE direct transformations
- Add `ephemeris_age` to `SGPPropagator`
- Implement Largrange, cubic Hermite, and quintic Hermite interpolation methods.
- Add `WalkerConstellationGenerator` which enables rapidly generating propagators with walker-geometry configurations. [#96](https://github.com/duncaneddy/brahe/pull/96)

### Changed

- Make access computation use D-type propagators.
- Standardize on
- Add support for D-type returns to `KeplerianPropagator` and `SGPPropagator`
- Standardized trait traits for `{S|D}StateProvider`, `{S|D}CovarianceProvider`, `{S|D}StatePropagator`
- `DTrajectory` and `DOrbitTrajectory` can now store trajectories of arbitrary length
- `DTrajectory` and `DOrbitTrajectory` can now store stm and sensitivity matricies alongside state and covariance.
- Improved test coverage across modules
- Changed how third-body ephemeris sources are defined and loaded. DE source is now a parameter instead of set by function name. Dynamically load BSP files when called.
- Changed how Gravity models are defined in terms of enumeration types.
- Renamed `state_cartesian_to_osculating` and `state_osculating_to_cartesian` as `state_eci_to_koe` and `state_koe_to_eci` [#89](https://github.com/duncaneddy/brahe/pull/89)
- Improve trajectory module test coverage
- Improve events module test coverage
- Changed CI release workflow to publish documentation updates only after packages have been successfully published [#96](https://github.com/duncaneddy/brahe/pull/96)

### Fixed

- Numerical integration `step` and `step_with_varmat` now take explicit parameter values. [#89](https://github.com/duncaneddy/brahe/pull/89)

### Removed

- Removed `STrajectory6` from python bindings. [#89](https://github.com/duncaneddy/brahe/pull/89)

## [0.4.0] - 2025-11-28
### Added

- Add space weather data management and access modeled on Earth orientation data provider design. [#73](https://github.com/duncaneddy/brahe/pull/73)
- Added support for control inputs to numerical integration
- Added support for integration of sensitivity matrices to numerical integration functions
- Added support for analytical and numerical computation of sensitivity matrix
- Added documentation for integration with control inputs and sensitivity matrix integration
- Added documentation for computation and handling of sensitivity matrices [#78](https://github.com/duncaneddy/brahe/pull/78)
- Add JOSS draft paper [#80](https://github.com/duncaneddy/brahe/pull/80)
- Added NRLMSISE-00 atmospheric density model implementation [#82](https://github.com/duncaneddy/brahe/pull/82)

### Changed

- Consolidates internal implementation of numerical integration
- Added additional numerical integration tests to improve coverage
- Consolidates duplicate type definitions across integrator files [#78](https://github.com/duncaneddy/brahe/pull/78)
- Refactor `celestrak.rs` and `naif.rs` modules to use `HttpClient` structure to enable mocking and mock testing through [mockall](https://github.com/asomers/mockall) to make struct calls
- Refactored main package documentation [#84](https://github.com/duncaneddy/brahe/pull/84)

### Fixed

- Auto-merge of bundled data update wasn't auto-merging [#75](https://github.com/duncaneddy/brahe/pull/75)
- Removed erroring cache-check in python test workflow [#78](https://github.com/duncaneddy/brahe/pull/78)

## [0.3.0] - 2025-11-18
### Added

- Added `Integrators` submodule with complete rust implementation, python bindings, documentation, and examples
- Support for `RK4`, `RKF45`, `DP54`, and `RKN1210` numerical integration methods
- License validation and compliance checks. Package is now automatically checked to ensure all dependencies have permissive, commercially-adoptable licenses [#63](https://github.com/duncaneddy/brahe/pull/63)
- NAIF Development Ephemeride dataset download and caching
- High-accuracy DE440s-based ephemeris prediction
- High-accuracy DE440s-based third-body acceleration prediction
- Python bindings for orbit dynamics functions [#68](https://github.com/duncaneddy/brahe/pull/68)
- Added `geo_sma` function to directly return the semi-major axis needed for a geostationary orbit. [#69](https://github.com/duncaneddy/brahe/pull/69)

### Changed

- Moved mathematics capabilities from `utils` submodule to dedicated `math` submodule.
- Bumped version to `0.2.0` for release
- Removed Rust test sections from coverage reporting [#63](https://github.com/duncaneddy/brahe/pull/63)
- Orbit dynamics functions which previously required a position-only vector can now accept either a position-only or a state vector. Conversion will be handled by the `IntoPosition` trait. [#68](https://github.com/duncaneddy/brahe/pull/68)

### Fixed

- Fix inconsistencies in Python API reference header levels
- Missing Ground station
- Miscellaneous documentation improvements and fixes [#63](https://github.com/duncaneddy/brahe/pull/63)

## [0.2.0] - 2025-11-17
### Added

- Added `relative_motion` submodule to contain functionality related to relative motion
- Implemented RTN rotation and state transformations: `rotation_eci_to_rtn`, `rotation_rtn_to_eci`, `state_eci_to_rtn`, and `state_rtn_to_eci`
- Implemented Relative Orbital Element (ROE) state transformations `state_oe_to_roe` and `state_roe_to_oe`
- Added `util::math` functions `sqrtm` and `spd_sqrtm` to calculate matrix square root
- Added `util::math` functions `oe_to_radians` and `oe_to_degrees` to reduce duplication in converting angle values in orbital element calcualtions [#57](https://github.com/duncaneddy/brahe/pull/57)
- Implement `CovarianceProvider` trait for `OrbitTrajectory`. Provides `covariance`, `covariance_eci`, `covariance_gcrf`, and `covariance_rtn`
- Add `interpolation` submodule to store consistent covariance interpolation methods
- Implement covariance rotation from ECI to RTN frame
- Extends `OrbitTrajectory` to optionally store covariance information. [#59](https://github.com/duncaneddy/brahe/pull/59)

### Changed

- Don't run PR tests on release CHANGELOG update. [#49](https://github.com/duncaneddy/brahe/pull/49)
- Unit test and PR test workflows now cancel in-progress runs if a new commit lands
- Added concurrency guards to auto-merge workflows. [#53](https://github.com/duncaneddy/brahe/pull/53)
- Moved where internal vector/matrix (e.g. `SMatrix3`, `SVector6`) type aliases are defined from `coordinates` submodule to `utils` [#57](https://github.com/duncaneddy/brahe/pull/57)

### Fixed

- Don't trigger multiple changelog merges on changelog a PR. [#49](https://github.com/duncaneddy/brahe/pull/49)
- Only trigger auto-merge on `opened` due to changelog PRs not issuing `labeled` events [#51](https://github.com/duncaneddy/brahe/pull/51)
- Added additional triggers to auto-merge workflows to ensure they are properly triggered. [#53](https://github.com/duncaneddy/brahe/pull/53)
- Fixed assorted typos and errors in docstrings related to covariance interpolation features. [#61](https://github.com/duncaneddy/brahe/pull/61)
- Fixes package-data update CI workflow by moving to auto-PR, auto-merge approach [#64](https://github.com/duncaneddy/brahe/pull/64)

## [0.1.3] - 2025-11-12
### Added

- Added python missing bindings for `states_icrf` and `states_gcrf` for `KeplerianPropagator` and `SGPPropagator`
- Added additional tests across various module to improve test coverage. [#36](https://github.com/duncaneddy/brahe/pull/36)

### Changed

- Refactor `frames.rs` file into submodule with subfiles for long-term maintainability. [#14](https://github.com/duncaneddy/brahe/pull/14)
- Automatically create and merge PRs for changelog updates [#16](https://github.com/duncaneddy/brahe/pull/16)
- Auto-merge changelog PRs
- Auto-merge dependabot PRs
- Expand dependabot to cover python and rust packages [#26](https://github.com/duncaneddy/brahe/pull/26)
- Bump package version to `v0.1.3` [#34](https://github.com/duncaneddy/brahe/pull/34)
- Skip unit test suite on auto-generated changelog PRs. [#44](https://github.com/duncaneddy/brahe/pull/44)

### Fixed

- PR changelogs were not being incorporated into the package changelog due to main-branch protection [#16](https://github.com/duncaneddy/brahe/pull/16)
- Stop generation changelog PRs for auto-generated changelog PRs [#26](https://github.com/duncaneddy/brahe/pull/26)
- Fixed issue with release pipeline release note generation [#38](https://github.com/duncaneddy/brahe/pull/38)
- Fix auto-merge for changelog PRs by using PAT [#40](https://github.com/duncaneddy/brahe/pull/40)
- Fix auto-merge workflow to accept PAT owner as actor. [#42](https://github.com/duncaneddy/brahe/pull/42)
- Fix workflow release step to use workflow PAT and declare base branch [#46](https://github.com/duncaneddy/brahe/pull/46)
