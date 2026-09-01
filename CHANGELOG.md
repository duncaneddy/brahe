# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Brahe's versioning and deprecation policy is modeled on [NumPy's policy](https://numpy.org/doc/stable/dev/depending_on_numpy.html) rather than strict SemVer — see the [versioning documentation](https://duncaneddy.github.io/brahe/latest/about/versioning/) for the full policy, including the transitional deprecation window currently in effect.

Each release groups entries under the Keep a Changelog section headings in the order **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**. Entries in **Deprecated** correspond to APIs that still work in this release but emit a `DeprecationWarning` or `FutureWarning`; entries in **Removed** list APIs that previous releases had deprecated.

<!-- release notes start -->

## [1.7.0] - 2026-07-21

### Added

- Native SPICE kernel support: `SPK` and `BPCK` readers (DAF container, SPK Type 2/3, binary PCK Type 2), usable standalone or via the global registry. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Global multi-kernel registry: `load_kernel`, `unload_kernel`, `clear_kernels`, `loaded_kernels`; kernels stay resident simultaneously with most-recently-loaded precedence and epoch-aware chain fallback. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `NAIFId` enum (planets, planetary-system barycenters, major moons, `Id(i32)` catch-all) and `FrameId` enum (`MoonPaDe440` = 31008 + `Id(i32)`); all query functions take `impl Into<NAIFId>` / `impl Into<FrameId>`, so raw integer IDs keep working. Python mirrors as `NAIFId`/`FrameId` IntEnums. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `SPICEKernel` enum covering every downloadable kernel — DE (de430–de442s), JPL satellite ephemeris kernels (`mar099`, `mar099s`, `jup365`, `sat441`, `ura184`, `nep097`, `plu060`), and binary PCK (`moon_pa_de440`) — plus `KernelSource` for bring-your-own kernel paths. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `load_common_kernels()` (de440s + moon_pa_de440, ~46 MB) and `load_all_kernels()` (~2.5 GB) pre-initialization helpers. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Generic NAIF-ID ephemeris queries: `spk_position`, `spk_velocity`, `spk_state` (pooled, cross-kernel chaining) and kernel-scoped `spk_position_from_kernel` / `spk_velocity_from_kernel` / `spk_state_from_kernel`. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Per-body kernel-backed accessors `{body}_{position,velocity,state}_spice` for Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, and the SSB. The five outer planets return true body centers via a two-leg DE + satellite-ephemeris-kernel sum (kernel auto-downloaded on first use: mar099s ~68 MB, jup365 ~1.1 GB, sat441 ~662 MB, ura184 ~387 MB, nep097 ~105 MB); `{planet}_barycenter_{position,velocity,state}_spice` variants provide the planetary-system barycenter from the DE kernel alone. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- PCK orientation queries with typed attitude returns: `pck_euler_angle` (`EulerAngle`, ZXZ), `pck_euler_angle_and_rates`, `pck_euler_rates`, `pck_quaternion` (`Quaternion`), `pck_rotation_matrix` (`RotationMatrix`), plus the raw `pck_euler_angles`. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `Epoch::spice_et()` — SPICE ephemeris time (TDB seconds past J2000) convenience accessor; `Epoch::seconds_past_j2000_as_time_system` for other time systems. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `datasets::naif::download_spice_kernel` for any known NAIF kernel (DE, satellite ephemeris, binary PCK), exposed in Python as `datasets.naif.download_spice_kernel`. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- ANISE validation suite (mm-level matched-ET comparisons, network-gated lunar DCM validation) and native-vs-ANISE criterion benchmarks. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Learn/API documentation for the spice module and four standalone examples, enabled in CI with cached kernels. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Lunar reference frames `LCI`/`LFPA`/`LFME` and Mars frames `MCI`/`MCMF` with `{rotation|position|state}_{from}_to_{to}` transformations, cross-center `state_eci_to_lci`/`state_eci_to_mci`, and automatic `moon_pa_de440` PCK loading. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Native IAU/WGCCRE 2015 body-rotation engine (`rotation_icrf_to_body_fixed_iau`, `iau_rotation_model_ids`) with an embedded 17-body coefficient table transcribed from `pck00011.tpc` and validated against ANISE. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Centralized `ReferenceFrame` enum and frame router (`rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`) with generic NAIF-ID variants (`BodyCenteredICRF`, `BodyFixedIAU`, `BodyFixedPCK`) and `"ECI"`/`"ECEF"` string aliases. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `CentralBody` abstraction (Earth, Moon, Mars, EMB, SSB, Custom) with `from_naif_id()`, and `ForceModelConfig::for_body()`, `lunar_default()` (GRGM660PRIM 50×50, Earth+Sun third bodies, Moon+Earth occultation), `mars_default()` (GMM-2B 50×50, exponential drag, Mars occultation), and `cislunar_default()` (EMB-centered) constructors. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Barycentric (EMB/SSB) propagation with correct direct/differential third-body handling; `ThirdBody` gains `Earth`, `Phobos`, `Deimos`, and `Custom { name, naif_id, gm }` variants. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Multi-body eclipse modeling via `occulting_bodies` on the SRP configuration and new `OccultingBody` type. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Body-generic dynamics functions: `accel_third_body_for_body`, `eclipse_conical_for_body`, `eclipse_cylindrical_for_body`, `accel_relativity_for_body`, `accel_drag_for_body`, `state_eci_to_koe_for_body`. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `state_bci`, `state_bcbf`, and `state_in_frame(frame, epoch)` on the orbit state-provider traits (`DOrbitStateProvider`/`SOrbitStateProvider`), implemented on the numerical, Keplerian, and SGP propagators and both trajectory types. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `OrbitFrame::BodyCenteredInertial(center)`: non-Earth propagator trajectories carry their center's NAIF ID, and trajectory Earth-frame conversions (point queries, batch `to_*`, covariance, CCSDS OEM export) re-center through the frame router; `state_koe_osc` returns elements about the trajectory's own center using that body's GM. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- User-defined body-fixed frames: `register_custom_frame(key, rotation, omega=None)` + `ReferenceFrame::BodyFixedCustom{center, key}` (epoch->DCM callback with optional angular-velocity callback; numeric transport-term fallback), with a runnable example. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `state_koe_to_eci_for_body` (inverse of `state_eci_to_koe_for_body`), `ReferenceFrame::ECI`/`ECEF` aliases, `kernel_is_loaded()`, and the `SECONDS_PER_JULIAN_CENTURY` constant. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `just setup` recipe for local development (uv venv, pre-commit hooks, and pre-downloading the de440s/moon_pa_de440/mar099s kernels tests expect cached). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- NAIF `mar099s` Mars satellite kernel support (`download_satellite_kernel`, `load_kernel("mar099s")`) for Phobos/Deimos ephemerides. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Physical constants `R_MARS`, `OMEGA_MARS`, `OMEGA_MOON`, `GM_PHOBOS`, `GM_DEIMOS`. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `ForceModelConfig::validate()` with clear errors for invalid central-body/force-model combinations, called automatically at propagator construction. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Python bindings for all of the above (`ReferenceFrame`, `CentralBody`, `OccultingBody`, extended `ThirdBody`, config constructors, frame functions, `state_in_frame`). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- CI caching of kernel and gravity-model downloads across jobs and workflow runs, with a weekly keep-warm workflow. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Learn/API documentation pages and runnable lunar/Mars propagation examples (Python + Rust). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Clenshaw-summation spherical-harmonic gravity kernel (`GravityModel::compute_spherical_harmonics_clenshaw`, `accel_gravity_spherical_harmonics_clenshaw`), with serial and parallel (order-parallel, bitwise-identical) execution — Rust + Python. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `GravityModelCoefficients` load configuration (`Clenshaw` default / `Cunningham` / `Both`) with `from_model_type_with_coefficients` / `from_file_with_coefficients` constructors (Rust + Python) and `load_uncached_with_coefficients` (Rust only, matching `load_uncached`), plus `precompute_*` / `drop_*` / `has_*` coefficient-set lifecycle methods — Rust + Python. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `GravityModel::get_c(n, m)` / `get_s(n, m)` single-coefficient accessors — Rust + Python. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `accel_gravity_spherical_harmonics_cunningham` convenience function — Rust + Python. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- High-precision reference tests: Clenshaw kernel pinned against independent 40-digit mpmath evaluations of EGM2008_360 to 1e-13 relative at degrees 120–200, with the generator committed as `scripts/generate_clenshaw_gravity_reference.py`. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- The Cunningham kernel returns a descriptive error (instead of silent NaN) on denormalized-recursion overflow. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `set_global_gravity_model` Python binding (mirrors the existing Rust function). [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- Clenshaw-vs-Cunningham benchmark suite in `benchmarks/gravity_benchmarks.rs`. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `is_empty()` method on `StaticEOPProvider`, `FileEOPProvider`, and `CachingEOPProvider` (Python bindings), satisfying clippy's `len_without_is_empty` for the newly workspace-covered `brahe-py` crate. [@duncaneddy](https://github.com/duncaneddy) ([#380](https://github.com/duncaneddy/brahe/pull/380))
- Solid Earth tide accelerations for numerical propagation per IERS Conventions (2010) TN36 §6.2.1: frequency-independent lunisolar corrections to degree-2/3 geopotential coefficients with degree-4 coupling (Eqs. 6.6–6.7, anelastic Love numbers from Table 6.3), and optional frequency-dependent corrections from Tables 6.5a/b/c via `SolidTideConfig(frequency_dependent=True)`. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- `TidesConfiguration` and `PermanentTideConfig` (`Auto`/`ConvertTo(system)`/`Off`) on `ForceModelConfig.tides` to enable tidal corrections and control permanent-tide normalization of the static field's C̄20 (IERS TN36 §6.2.2); tides remain disabled by default in all force-model presets. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- `GravityModel.convert_tide_system` (Rust and Python) to convert a gravity model's C̄20 between mean-tide, zero-tide, and conventional tide-free systems. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- Rust API `accel_solid_earth_tides` / `solid_earth_tide_coefficients` in `orbit_dynamics::tides` for direct evaluation of tide accelerations and coefficient corrections. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- Warnings for tide configurations that double-count the permanent tide (zero-tide/mean-tide conversion combined with solid tides, or shared global gravity models): Rust warning at propagator construction, suppressible `UserWarning` in Python. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- Learn documentation page on tidal corrections (tide systems, permanent tide handling, solid Earth tides) and runnable Python/Rust examples: `force_model_tides`, `tides_permanent_only`, `tides_static`, `tides_static_time_varying`. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- `ReferenceFrame.EMR`, `ReferenceFrame.SER`, and `ReferenceFrame.GSE` synodic frames (NASA TP-20220014814) supported across the frame router (`rotation/position/state_frame_to_frame`, `state_in_frame`), with exact rotation-matrix time derivatives (GTDS/STK convention) for the velocity transform. [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- Pairwise synodic transforms `rotation/position/state_gcrf_to_{emr,ser,gse}` and inverses in Rust and Python. [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- Native SPK acceleration evaluation via analytic Chebyshev differentiation: `spk_acceleration` and `spk_acceleration_from_kernel` (Rust and Python). [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- `SUN_EARTH_BARYCENTER_ID` synthetic frame-center ID with GM-weighted Sun-Earth barycenter resolution in the router's translation seam. [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- Batch state-provider methods `states_bci`, `states_bcbf`, and `states_in_frame` on `SOrbitStateProvider`/`DOrbitStateProvider` and on the `SGPPropagator`, `KeplerianPropagator`, and `NumericalOrbitPropagator` Python classes. [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- Documentation: Synodic Reference Frames Learn page and Library API page, frame/kernel-requirement table entries, and SPK acceleration API docs. [@duncaneddy](https://github.com/duncaneddy) ([#393](https://github.com/duncaneddy/brahe/pull/393))
- Per-body third-body configuration: `ThirdBodyConfiguration` pairs each perturber with its own `ephemeris_source` and `gravity` model; `SphericalHarmonic` and `EarthZonal` fields are evaluated at the object's position relative to the perturbing body in that body's fixed frame, with a point-mass indirect term. [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- Public `accel_third_body_field_for_body` (Rust + Python) evaluating a configured gravity model (point-mass, spherical-harmonic, or Earth-zonal) for a perturbing body about any central body. [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- `ThirdBody` planetary-system barycenter variants `MarsBarycenter`–`NeptuneBarycenter` (NAIF 4–8, system GMs), used by the default Earth force models; `accel_third_body` accepts both barycenter and planet-center variants (the latter auto-load their satellite-system kernels). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- `DragConfiguration.body` to attribute drag to a non-central body: density and relative wind are evaluated at the object's state relative to that body (enables Earth drag on EMB-centered propagations). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- `GravityConfiguration::Zero` — explicit no-central-gravity term for barycentric propagation centers (used by `cislunar_default`). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- ECI↔EMBI frame translation helpers `position_eci_to_emb`/`position_emb_to_eci` and `state_eci_to_emb`/`state_emb_to_eci` (Rust + Python). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- Constants `GM_MARS_SYSTEM`–`GM_NEPTUNE_SYSTEM` and `GM_PLUTO_SYSTEM` (planetary-system barycentric GMs, NAIF `gm_de440.tpc`). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- `ThirdBody::as_central_body()` and `ThirdBody::body_fixed_frame()` helpers (Rust + Python). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- Standalone example `numerical_propagation/cislunar_earth_forces` (EMB-centered propagation with Earth-attributed forces). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- `Epoch.to_time_system()` (Rust and Python) returns a new `Epoch` for the same instant expressed in a different time system. The instant is unchanged, so the returned epoch compares equal to the original; only the scale it reports in differs. The method had been referenced by the `Epoch::time_system` documentation but was never implemented. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- `TimeSystem` API reference page under `library_api/time/`, documenting all ten time systems (`GPS`, `TAI`, `TT`, `UTC`, `UT1`, `TDB`, `TCG`, `TCB`, `BDT`, `GST`). It previously lived under `constants/units.md` and listed only five, so `TDB`, `TCG`, `TCB`, `BDT`, and `GST` did not render at all. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- "Time System" section in the Learn Epoch guide, covering how an epoch's time system is set and read, and the distinction between converting an epoch and projecting one, with runnable Rust and Python examples. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Example fixtures covering `FLAGS` declared past the tenth line of a file. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- FES2004 ocean tide model (`OceanTideModel`, `OceanTideConfig`) with admittance-wave expansion per IERS TN36 §6.3, configurable degree/order up to 100, and one-time cached download of the IERS coefficient file. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Solid Earth pole tide (`SolidTideConfig.pole_tide`) and ocean pole tide (`OceanTideConfig.pole_tide`) per IERS TN36 §6.4/§6.5 with the IERS 2018 linear secular pole model (`secular_pole`, `wobble_parameters`, `solid_earth_pole_tide_deltas`, `ocean_pole_tide_deltas`). [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- `TidesConfiguration.ocean` field enabling ocean tides in the numerical propagator force model; `ForceModelConfig::high_fidelity()` now enables solid tides with frequency-dependent corrections, both pole tides, and 30×30 ocean tides. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Python bindings for `OceanTideConfig` and the new `pole_tide` / `ocean` configuration fields, with regenerated type stubs. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- `ReferenceFrame.Synodic(origin, primary, secondary)` and `SynodicOrigin` for arbitrary two-body rotating frames, with `synodic_origin`/`synodic_primary`/`synodic_secondary` accessors (also populated for EMR/SER/GSE). [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `synodic_barycenter_id` and generalized synthetic-barycenter resolution in the frame router (Sun-Earth barycenter is now a special case of the generic encoding). [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Solar System Scope planet texture registry with `download_planet_texture` and generalized `load_body_texture` (CC BY 4.0, credited in the license page). [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `central_body` and `additional_bodies` parameters on `plot_trajectory_3d` for Moon/Mars/custom-body scenes, backed by a body-visuals registry. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `plot_synodic_3d` and `plot_earth_moon_rotating_3d` for rotating-frame trajectory visualization. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Worked examples: LRO lunar orbit, Earth-Moon free return, MRO Mars orbit, Dawn at Ceres (user-defined central body), with learn/API documentation for the new frames and plotting. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Standalone `Rx`/`Ry`/`Rz` matrix functions. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Planetary radius constants (`R_MERCURY` .. `R_NEPTUNE`). [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Tested learn examples for generic `ReferenceFrame.Synodic` frames and `plot_earth_moon_rotating_3d`. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Keplerian ↔ equinoctial element conversions (`state_koe_to_equinoctial`, `state_equinoctial_to_koe`) following the Vallado (eq. 2-99) formulation with Montenbruck `a, h, k, p, q, l` naming and an explicit retrograde factor (Rust + Python). [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- `MeanElementMethod` selector (`BROUWER_LYDDANE` analytical, `numerical(config)` windowed averaging) plus `WindowAlignment`, `WindowEdgeHandling`, `MeanElementNumericalMethodConfig`, and `MeanElementInverseConfig` configuration types (Rust + Python). [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- Numerical osculating → mean conversion via windowed averaging in equinoctial space, with trapezoidal time-weighting of the slow elements and configurable window length (seconds), alignment (centered / trailing / leading), and edge handling (truncate / preserve-window). [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- Numerical mean → osculating conversion via an iterative differential-correction inverse driven by a supplied force model and numerical propagation configuration. [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- Batch conversion functions `batch_state_koe_osc_to_mean` and `batch_state_koe_mean_to_osc` returning `(epoch, state)` pairs, with Python bindings returning `(list[Epoch], ndarray)`. [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- FK5, Hipparcos, and Tycho-2 star catalog datasets with cached downloads, typed records, `StarRecord` trait, filtering, and Polars DataFrame conversion (Rust + Python). [@duncaneddy](https://github.com/duncaneddy) ([#404](https://github.com/duncaneddy/brahe/pull/404))
- RA/Dec coordinate transformations: `position_radec_to_inertial`/`position_inertial_to_radec`, `state_radec_to_inertial`/`state_inertial_to_radec`, `position_radec_to_azel`/`position_azel_to_radec`. [@duncaneddy](https://github.com/duncaneddy) ([#404](https://github.com/duncaneddy/brahe/pull/404))
- Rigorous proper-motion epoch propagation `apply_proper_motion` (delegating to IAU SOFA `iauPmsafe` via `rsofa`; theory per ESA SP-1200 §1.5.5) and `StarRecord::radec_at_epoch`. [@duncaneddy](https://github.com/duncaneddy) ([#404](https://github.com/duncaneddy/brahe/pull/404))
- Star-field sensor simulation example with animated 3D and sensor-frame Plotly visualizations. [@duncaneddy](https://github.com/duncaneddy) ([#404](https://github.com/duncaneddy/brahe/pull/404))
- `datasets::ssn_sensors::load_ssn_sensors` / `bh.datasets.ssn_sensors.load()`: embedded dataset of 21 representative SSN sensor sites from Vallado 4th Ed. Tables 4-2/4-3/4-4 with locations, az/el/range field-of-view limits, and bias/noise calibration values. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `AzElRangeMeasurementModel`: topocentric azimuth/elevation/range measurement model with constant-bias support, azimuth residual wrapping, and a wrap-aware finite-difference Jacobian. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `SimpleSSNSensor`: ground-sensor measurement simulation with wrap-aware field-of-view checks, seeded Gaussian noise, step-wise (`measure`) and batched (`simulate_observations`) generation, and a matching filter-side `measurement_model()`. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `MeasurementModel::residual()`: overridable, dimension-validated residual hook used by all EKF/UKF/BLS residual computations. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `ExtendedKalmanFilter::propagate_to` / `UnscentedKalmanFilter::propagate_to`: measurement-free prediction step that records covariance growth across tracking gaps. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- SSN tracking examples: standalone Rust/Python pair (`examples/estimation/ssn_tracking`) and a six-hour BLS/EKF/UKF walkthrough with sensor-network, measurement, and filter-comparison figures (`docs/examples/ssn_tracking.md`). [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `brahe transform position` and `brahe transform rotation` CLI subcommands for position-vector and rotation-matrix transforms between reference frames. [@duncaneddy](https://github.com/duncaneddy) ([#407](https://github.com/duncaneddy/brahe/pull/407))
- `brahe transform frame` now accepts all named reference frames (GCRF, ITRF, EME2000, lunar LCI/LFPA/LFME, Mars MCI/MCMF, EMBI, SSBI, synodic EMR/SER/GSE) in addition to the ECI/ECEF aliases. [@duncaneddy](https://github.com/duncaneddy) ([#407](https://github.com/duncaneddy/brahe/pull/407))
- Runnable `fk5_frame_correction` example (Python and Rust) demonstrating the star-catalog to EME2000 to GCRF to ITRF transformation chain. [@duncaneddy](https://github.com/duncaneddy) ([#413](https://github.com/duncaneddy/brahe/pull/413))
- `builder()` constructors for `DNumericalOrbitPropagator`, `DNumericalPropagator`, `SGPPropagator` (OMM elements), `WalkerConstellationGenerator`, `ExtendedKalmanFilter`, `UnscentedKalmanFilter`, `BatchLeastSquares`, `AccessProperties`, and `CDMObjectMetadata`, with Python mirrors for all but `CDMObjectMetadata`. [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- `WalkerConstellationGeneratorBuilder::build()` returns an error on invalid t/p/f configurations instead of panicking. [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- `CDMObjectMetadataBuilder` exposes the optional CCSDS CDM metadata fields that `CDMObjectMetadata::new()` could not set, and enforces the conditional-mandatory rules (`ALT_COV_TYPE` requires `ALT_COV_REF_FRAME`; `EPHEMERIS_NAME=ODM` requires `ODM_MSG_LINK`). [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- Builder-focused documentation: Learn-page sections, library API pages, and runnable Rust/Python examples for each builder. [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- JPL Small-Body Database (SBDB) Lookup client (`brahe.datasets.sbdb.SBDBClient` returning `SBDBObject`) that resolves a name or designation to its NAIF/SPK ID and SI physical parameters (GM in m³/s², radius in m), with on-disk caching. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))
- JPL Horizons SPK client (`brahe.datasets.horizons`: `HorizonsClient`, `HorizonsSPKRequest`, `HorizonsSPKResponse`) that generates, caches, and loads targeted small-body SPK kernels, with a response handle exposing the cached `.bsp` path, raw bytes, and a direct load into the SPICE registry. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))
- SPK segment type 21 (Extended Modified Difference Arrays) reader, enabling Horizons-generated small-body SPKs (e.g. Ceres) to load and answer position, velocity, and state queries; the interpolation is a port of CSPICE `spke21` validated bit-exact against CSPICE `spkgeo` on a real Ceres kernel. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))
- Learn and Library-API documentation pages for the SBDB and Horizons clients. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))

### Changed

- **Breaking:** third-body ephemeris accessors renamed from `*_de` to `*_spice` (e.g. `sun_position_de` → `sun_position_spice`, `accel_third_body_sun_de` → `accel_third_body_sun_spice`) — the ephemeris source is the SPICE kernel system (DE + satellite ephemeris kernels), not DE alone. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** the five outer-planet accessors return the **true planet body center** (previously the planetary-system barycenter); the barycenter is available via the new `*_barycenter_*_spice` functions, and third-body force models use those (numerically identical to before, no satellite-kernel downloads in propagation). [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** per-body position outputs no longer apply a J2000→ICRF frame-bias rotation. NAIF documents that SPICE "J2000" output from DE kernels is already ICRF-aligned, so the previous rotation introduced a ~23 mas systematic offset (~1.7e4 m at 1 AU). [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** `EphemerisSource::SPK` payload type is `SPICEKernel` (was `SPKKernel`); serde variant names for the DE kernels are unchanged, so serialized force-model configs remain compatible. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- Switching between DE kernels no longer reloads global state; both kernels stay resident in the registry. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `ForceModelConfig` gains a `central_body` field (serde-default `Earth`; existing configs and constructors unchanged) and its validity is now checked at propagator construction instead of failing mid-propagation. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `SolarRadiationPressureConfiguration` gains `occulting_bodies` (serde-default `[Earth]`, preserving existing behavior). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `state_koe_osc` on the numerical propagator now computes osculating elements about the propagator's central body (Earth behavior unchanged; barycenter central bodies return an error). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `CentralBody::Mars` and the `MCI`/`MCMF` frames are centered on the Mars body center (NAIF 499), with the body-center leg auto-loaded from the `mar099s` satellite ephemeris (the default DE kernel loads first so a satellite kernel cannot suppress the auto-initialization); `ThirdBody::Mars` keeps the system barycenter (NAIF 4) per the standard third-body formulation. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- The lunar PCK / Mars SPK auto-load guards latch on first use (`OnceLock`): the kernel registry is consulted once, and unloading kernels mid-run is no longer re-detected (failed loads retry). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Kernel-only tests and doctests are no longer integration-gated: SPICE kernels are expected cached locally (`just setup`) and in CI; the pytest `integration` marker is reserved for remote-host access and long runtimes. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Rust `ThirdBody` no longer derives `Eq`/`Hash` (the new `Custom` variant carries a float GM); the Python `ThirdBody`/`CentralBody`/`OccultingBody`/`ReferenceFrame` types are plain wrapper classes — equality is preserved but they are not hashable and class attributes are not singletons (use `==`, not `is`). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- `compute_spherical_harmonics` and `accel_gravity_spherical_harmonics` now evaluate via the Clenshaw kernel when its coefficients are present (the default), falling back to Cunningham coefficients when those are the only set loaded. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- Gravity model loading now precomputes only Clenshaw coefficients by default (use `GravityModelCoefficients::Both`/`Cunningham` for the V/W kernel); `set_max_degree_order` rebuilds whichever coefficient sets exist. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- The numerical orbit propagator evaluates spherical-harmonic gravity through the Clenshaw kernel (the per-propagator V/W workspace matrices were removed); its per-propagator rotation cache is now an `lru`-crate LRU keyed on stage time. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `CLENSHAW_PARALLEL_THRESHOLD_NMAX` tuned to 180 from benchmarks (with a TODO to derive it from the execution environment); Cunningham's documented numerical limits added to its rustdoc. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- `GravityModel::compute_spherical_harmonics_with_workspace` and `accel_gravity_spherical_harmonics_with_workspace` — renamed to `compute_spherical_harmonics_cunningham_with_workspace` / `accel_gravity_spherical_harmonics_cunningham_with_workspace` (the Clenshaw kernel needs no workspace). The old allocating Cunningham entry point is now `GravityModel::compute_spherical_harmonics_cunningham` (`compute_spherical_harmonics` keeps its name and signature but dispatches). No deprecation aliases retained. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- Marked additional space-weather tests and integration to issues running locally if celestrak is unavailable. [@duncaneddy](https://github.com/duncaneddy) ([#378](https://github.com/duncaneddy/brahe/pull/378))
- Numerical propagator construction now applies the configured permanent-tide conversion to propagator-owned gravity models when `ForceModelConfig.tides` is set (models load with C̄20 exactly as published when tides are not configured). [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- **Breaking:** `ForceModelConfig.third_body` becomes `Option<Vec<ThirdBodyConfiguration>>` with per-body entries. The field deserializes from bare bodies, a single entry, a list, or the legacy pre-flattening object shape, which migrates on load with its original semantics (pre-split planet names map to the `*Barycenter` variants they then denoted). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- **Breaking:** `ThirdBody::Mars`–`Neptune` now denote planet centers (NAIF 499–899, planet-only GMs, resolved through satellite-system kernels) instead of system barycenters; use the new `*Barycenter` variants for the previous behavior. [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- **Breaking:** `GM_MARS`–`GM_NEPTUNE` and `GM_PLUTO` now hold planet-only GMs from NAIF `gm_de440.tpc`; the previous DE430-era system values moved to `GM_*_SYSTEM` and were updated to DE440 (full-precision tpc digits). [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- Force-model validation keys the Earth-atmosphere-model and radius/spin drag requirements on the *attributed* body rather than the central body, and validates per-third-body gravity models (`EarthZonal` requires `ThirdBody::Earth`; `SphericalHarmonic` requires a body with a known body-fixed frame). An attributed drag body's ephemeris is verified at propagator construction. [@duncaneddy](https://github.com/duncaneddy) ([#394](https://github.com/duncaneddy/brahe/pull/394))
- Renamed the `IGNORE` example flag to `NETWORK`, completing the split begun in #366. `IGNORE` named no reason to skip, while every file carrying it depends on a live third-party service. `--ignore` becomes `--network` on `just test-examples` and `just make-plots`. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Reclassified `plots/fig_access_benchmark.py` and `plots/fig_comparative_benchmarks.py` from `IGNORE` to `MANUAL`. Neither is network-driven: both regenerate committed artifacts and require a custom environment, and neither is driven by `just make-plots`. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Aligned the User Guide and Python API Reference navigation to a single section ordering and vocabulary. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- The Learn time-scale table and the `TimeSystem` overview rustdoc now document all ten time systems rather than five. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- All tidal corrections (solid Earth Step 1/2, pole tides, ocean tides) and the static gravity field are now evaluated in a single fold-in Clenshaw pass per dynamics call, replacing the separate solid-tide evaluation path. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Bundled EGM2008 gravity model truncated from degree/order 360 to 120 and renamed `GravityModelType::EGM2008_360` → `EGM2008_120` to keep the crates.io package within size limits (bit-identical coefficients through degree 120; higher degrees available via ICGEM). [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Ocean tide force-model documentation (`docs/learn/orbital_dynamics/tides.md`) expanded to cover ocean tides, admittance waves, pole tides, and the download/cache behavior. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Documentation is now served from `https://docs.brahe.space/`, with `brahe.space` and `www.brahe.space` redirecting to it. Existing `duncaneddy.github.io/brahe/` links continue to resolve through GitHub's permanent redirect. [@duncaneddy](https://github.com/duncaneddy) ([#398](https://github.com/duncaneddy/brahe/pull/398))
- Nightly development wheels now install from `https://docs.brahe.space/simple/`. [@duncaneddy](https://github.com/duncaneddy) ([#398](https://github.com/duncaneddy/brahe/pull/398))
- The `git clone` commands in the installation guide now pass `-c core.symlinks=true` and clone only the `main` branch. [@duncaneddy](https://github.com/duncaneddy) ([#398](https://github.com/duncaneddy/brahe/pull/398))
- `plot_trajectory_3d`: `earth_texture` renamed to `texture`, `show_earth` renamed to `show_body`, and all parameters after `trajectories` are now keyword-only (breaking; all in-repo usages updated). [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `load_earth_texture` removed in favor of `load_body_texture`. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `just download-resources` now also warms the moon/mars/ceres textures. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- `state_koe_to_eci_for_body`/`state_eci_to_koe_for_body` renamed to `state_koe_to_inertial_for_body`/`state_inertial_to_koe_for_body`, taking a `CentralBody` instead of a bare `gm` and referencing elements to the body's mean equator at J2000 instead of the ICRF axes. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- Live-API examples now `NETWORK`-flagged. [@duncaneddy](https://github.com/duncaneddy) ([#399](https://github.com/duncaneddy/brahe/pull/399))
- **Breaking:** `state_koe_osc_to_mean` and `state_koe_mean_to_osc` now take a `MeanElementMethod` argument (before `angle_format`) and return a `Result` (Rust) / raise on error (Python). Existing analytical behavior is preserved by passing `MeanElementMethod::BrouwerLyddane` / `bh.MeanElementMethod.BROUWER_LYDDANE`. [@duncaneddy](https://github.com/duncaneddy) ([#403](https://github.com/duncaneddy/brahe/pull/403))
- Upgraded Rust dependency version requirements to latest compatible releases (`cargo upgrade`). [@duncaneddy](https://github.com/duncaneddy) ([#404](https://github.com/duncaneddy/brahe/pull/404))
- UKF measurement statistics use a reference-point mean and residual-based deviations, making the measurement update robust for angular measurements near a wrap (identical results for non-angular models). [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- EKF/UKF/BLS pre-fit and post-fit residuals now route through `MeasurementModel::residual()` (default behavior unchanged). [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `rand`/`rand_distr` promoted from dev-dependencies to runtime dependencies for measurement noise sampling. [@duncaneddy](https://github.com/duncaneddy) ([#406](https://github.com/duncaneddy/brahe/pull/406))
- `brahe transform frame` dispatches through the core frame router (`state_frame_to_frame`) instead of a hardcoded ECI↔ECEF path. [@duncaneddy](https://github.com/duncaneddy) ([#407](https://github.com/duncaneddy/brahe/pull/407))
- Rewrote `docs/examples` note admonitions as per-example API deep dives covering `GPRecord.to_sgp_propagator`, `location_accesses`, `datasets.groundstations.load`, `AccessPropertyComputer`, `TimeEvent` callbacks, `additional_dynamics`/`control_input`, `ConstraintAll`, and `AOIExitEvent`. [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))
- Docs deploys now restore NETWORK-example figures saved by the weekly `--network` integration run, so example pages no longer reference figures that were never generated. [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))
- Star-catalog documentation now describes each catalog's native reference frame (Hipparcos and Tycho-2 on ICRS axes, FK5 realizing EME2000) and when the ~23 mas frame bias must be applied before use. [@duncaneddy](https://github.com/duncaneddy) ([#413](https://github.com/duncaneddy/brahe/pull/413))
- `DNumericalOrbitPropagatorBuilder` is now a runtime builder with required fields as `builder()` arguments, replacing the typestate (`Set`/`Unset`) builder; the entry point and setter names are otherwise unchanged (breaking, Rust only). [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- `DNumericalOrbitPropagator::new` and `DNumericalPropagator::new` validate `initial_covariance` dimensions against the state dimension and return an error on mismatch. [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- Reworked the Dawn-at-Ceres example to resolve Ceres via SBDB, load a Ceres SPK from Horizons, and model Sun and Jupiter third-body plus solar radiation pressure perturbations around the custom-defined body. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))
- Refactored the SPK reader behind a segment abstraction so a resolved chain can mix segment types, combining a type-21 small-body segment with the existing Chebyshev (types 2 and 3) segments from the DE kernels. [@duncaneddy](https://github.com/duncaneddy) ([#416](https://github.com/duncaneddy/brahe/pull/416))
- Crates.io and PyPI package-validation CI jobs now run in parallel with the Rust and Python test suites instead of waiting for them to finish. [@duncaneddy](https://github.com/duncaneddy) ([#417](https://github.com/duncaneddy/brahe/pull/417))
- Getting-started documentation now links function and class references to their corresponding library API and Learn pages. [@duncaneddy](https://github.com/duncaneddy) ([#417](https://github.com/duncaneddy/brahe/pull/417))

### Removed

- **Breaking:** `SPKKernel` — use `SPICEKernel`. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** `datasets::download_de_kernel` — use `datasets::naif::download_spice_kernel`, which accepts any known kernel name (DE, satellite ephemeris, binary PCK). [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** `set_global_almanac`, `get_loaded_kernel_type`, and `brahe_epoch_to_anise` (ANISE types no longer appear in the public API). [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- **Breaking:** ANISE as a runtime dependency (now dev-only). Downstream users relying on brahe's transitive `anise` dependency must add it directly. [@duncaneddy](https://github.com/duncaneddy) ([#376](https://github.com/duncaneddy/brahe/pull/376))
- `GravityModelType::EGM2008_360` (renamed to `EGM2008_120`; not deprecated — the variant was removed outright since the underlying packaged file changed). [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Internal fixed-size (degree ≤ 4) tidal coefficient evaluator (`TideCoefficients`, `accel_low_degree_harmonics`), superseded by the dynamically sized `TideDeltas` accumulator evaluated through the shared Clenshaw kernel. [@duncaneddy](https://github.com/duncaneddy) ([#397](https://github.com/duncaneddy/brahe/pull/397))
- Removed the Starlink visualization example (`docs/examples/visualizing_starlink.md`, `examples/examples/visualizing_starlink.py`) and its committed figures; the GPS visualization example covers the same workflow. [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))

### Fixed

- ICGEM GFC parser now handles the `gravity_constant` header key and Fortran `D`-exponent notation, unblocking all non-Earth gravity model downloads (e.g. lunar GRGM660PRIM, Mars GMM-2B). [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Earth-centered propagation configured with the new third-body variants (Phobos, Deimos, Custom) routes through the SPK-backed path instead of panicking; `validate()` rejects `LowPrecision` ephemerides for bodies other than Sun/Moon and third bodies that coincide with the central body. [@duncaneddy](https://github.com/duncaneddy) ([#377](https://github.com/duncaneddy/brahe/pull/377))
- Repo-wide `cargo fmt` and `ruff` lint/format debt that had accumulated since pre-commit hooks were never installed locally. [@duncaneddy](https://github.com/duncaneddy) ([#380](https://github.com/duncaneddy/brahe/pull/380))
- `crates/brahe-py` clippy errors that were silently exempted from the pre-commit gate because the clippy hook lacked `--workspace`; the hook now covers the whole workspace. This included `missing_safety_doc` on 5 numpy-conversion methods in `attitude.rs` — on inspection none of them perform an actual unsafe operation (`into_pyarray`/`reshape` are safe in the pinned `numpy` version), so the vestigial `unsafe` keyword was removed instead of documenting a non-existent invariant — plus `new_without_default`/`len_without_is_empty` on the EOP provider bindings. [@duncaneddy](https://github.com/duncaneddy) ([#380](https://github.com/duncaneddy/brahe/pull/380))
- Fix issue with some tests that rely on celestrak integration not being properly marked `pytest.mark.integration`. [@duncaneddy](https://github.com/duncaneddy) ([#385](https://github.com/duncaneddy/brahe/pull/385))
- `FLAGS` declared past the tenth line of an example were silently ignored, because the runner parsed only the first ten lines. `lunar_orbit.rs` and `mars_orbit.rs` declare theirs on line 13, so both ran on every default `just test-examples` — downloading the GRGM660PRIM gravity model and the `moon_pa_de440` kernel — despite being flagged, while their `.py` twins were correctly skipped. The runner now parses the whole leading comment block. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Corrected the TCG secular drift from `~0.7 s/year` to `~22 ms/year` (IAU 2000 Resolution B1.9, L_G = 6.969290134e-10). [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Corrected the `Epoch::time_system` documentation, which listed five of the ten time systems and directed readers to a `to_time_system()` method that did not exist. [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Corrected typos in the `TimeSystem` rustdoc ("supposed" to "supported", "expeccted" to "expected"). [@duncaneddy](https://github.com/duncaneddy) ([#395](https://github.com/duncaneddy/brahe/pull/395))
- Repaired six broken documentation links. The contributing guide, code of conduct, development guide, and license links omitted the `/latest/` version segment and resolved to nothing. The development guide was additionally linked as `developer_guidelines.html`, a page that has never existed. Two versioning policy links used a directory-style URL where the site emits `.html`. [@duncaneddy](https://github.com/duncaneddy) ([#398](https://github.com/duncaneddy/brahe/pull/398))
- `brahe transform frame` no longer emits duplicate output when the source and target frames are identical (missing early return). [@duncaneddy](https://github.com/duncaneddy) ([#407](https://github.com/duncaneddy/brahe/pull/407))
- `brahe transform coordinates` now rejects non-ECI/ECEF values for `--from-frame`/`--to-frame` at validation instead of silently mis-routing them. [@duncaneddy](https://github.com/duncaneddy) ([#407](https://github.com/duncaneddy/brahe/pull/407))
- Fixed 404s for all NETWORK-flagged example figures on the live documentation site (Doppler compensation, maximum communications gap, LRO, MRO, Dawn at Ceres, ground contacts, imaging opportunities, imaging data latency, tessellation). [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))
- Fixed the NAIF kernel and 3D-texture example caches being stranded on PR merge refs, which forced every CI run into live downloads from naif.jpl.nasa.gov and Solar System Scope; caches now save from main only. [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))
- Corrected the NASA NEN station count claim in the ground contacts example note. [@duncaneddy](https://github.com/duncaneddy) ([#409](https://github.com/duncaneddy/brahe/pull/409))
- `apply_proper_motion` documentation now describes the IAU SOFA `iauPmsafe` space-motion transformation the function actually calls, rather than a hand-rolled tangent-plane method. [@duncaneddy](https://github.com/duncaneddy) ([#413](https://github.com/duncaneddy/brahe/pull/413))
- Documentation figures for the lunar, Mars, Ceres, Earth-Moon free-return, and star-field examples now regenerate on every docs deploy instead of only via the weekly network run. [@duncaneddy](https://github.com/duncaneddy) ([#413](https://github.com/duncaneddy/brahe/pull/413))
- Mismatched `initial_covariance` dimensions previously constructed successfully and panicked inside the first propagation step; construction now fails with a descriptive error, including from Python with extended (6+N) states. [@duncaneddy](https://github.com/duncaneddy) ([#414](https://github.com/duncaneddy/brahe/pull/414))
- Captured output of the NETWORK-flagged getting-started examples (first script, Celestrak client) now reaches the deployed docs through a network-example-outputs cache, and the offline external-data examples emit output, so the affected pages no longer render empty output blocks. [@duncaneddy](https://github.com/duncaneddy) ([#417](https://github.com/duncaneddy/brahe/pull/417))
- Star-field sensor view no longer flips direction at orbital plane crossings by using the orbit normal as a continuous roll reference, scrolls the star field along the elevation axis, and preserves per-star identity so markers fade in and out instead of jumping when the visible-star count changes. [@duncaneddy](https://github.com/duncaneddy) ([#417](https://github.com/duncaneddy/brahe/pull/417))
- ci-success aggregation now includes the check-skip and delay-python gating jobs so a failed gate cannot be reported as a passing run. [@duncaneddy](https://github.com/duncaneddy) ([#417](https://github.com/duncaneddy/brahe/pull/417))

## [1.6.2] - 2026-07-01

### Added

- Added native Linux `arm64` wheel builds to the latest and release workflows, with platform-specific wheel artifact names to avoid artifact collisions. [@hectcastro](https://github.com/hectcastro) ([#372](https://github.com/duncaneddy/brahe/pull/372))
- `CachingEOPProvider` / `CachingSpaceWeatherProvider` (and `initialize_eop()` / `initialize_sw()`) now seed a missing cache from the compiled-in bundled data, so EOP and space-weather initialization succeed offline without an immediate network download. [@duncaneddy](https://github.com/duncaneddy) ([#374](https://github.com/duncaneddy/brahe/pull/374))
- EOP and space-weather downloads now retry transient failures (connection errors, timeouts, HTTP 429/5xx) with exponential backoff and jitter (up to 4 attempts). [@duncaneddy](https://github.com/duncaneddy) ([#374](https://github.com/duncaneddy/brahe/pull/374))

### Changed

- Changed standard EOP product source to [USNO finals2000A.all](https://maia.usno.navy.mil/ser7/finals2000A.all). [@duncaneddy](https://github.com/duncaneddy) ([#373](https://github.com/duncaneddy/brahe/pull/373))
- Changed C04 EOP product source to [Paris Observatory C04 Product](https://www.simplespacedata.org/eop/obspm/c04_1962now/latest/eopc04.1962-now). [@duncaneddy](https://github.com/duncaneddy) ([#373](https://github.com/duncaneddy/brahe/pull/373))
- Download failure messages now include the attempted URL and number of attempts. [@duncaneddy](https://github.com/duncaneddy) ([#374](https://github.com/duncaneddy/brahe/pull/374))
- Consolidated EOP and space-weather download logic into a shared `utils::download` helper (internal refactor, no API change). [@duncaneddy](https://github.com/duncaneddy) ([#374](https://github.com/duncaneddy/brahe/pull/374))

### Fixed

- Fix compilation issue due to transitive dependency breaking change in `ureq`/`time` crates. Pin `time` until issue is resolved. [@duncaneddy](https://github.com/duncaneddy) ([#373](https://github.com/duncaneddy/brahe/pull/373))
- Fixed issue with failing tests due to inability to update outdated EOP products. [@duncaneddy](https://github.com/duncaneddy) ([#373](https://github.com/duncaneddy/brahe/pull/373))
- `initialize_eop()` / `initialize_sw()` no longer fail on a fresh/empty cache when the remote EOP or space-weather server is transiently unreachable (e.g. `Standard EOP download request failed: io: Connection refused` in CI); bundled data is used instead. [@duncaneddy](https://github.com/duncaneddy) ([#374](https://github.com/duncaneddy/brahe/pull/374))

## [1.6.1] - 2026-06-16

### Added

- Added "Getting Started" section to main documentation page. [@duncaneddy](https://github.com/duncaneddy) ([#356](https://github.com/duncaneddy/brahe/pull/356))
- `integration` flag to rust and python test suites, replacing `ci`. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- `par_propagate_to` (Python) now accepts a list that mixes `KeplerianPropagator`, `SGPPropagator`, and `NumericalOrbitPropagator` instances. Propagators are grouped by type and each group is propagated in parallel; results are written back to the original objects in place, preserving list order. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))
- `MANUAL` example flag for examples that must never run automatically (scaffolding templates and credential-gated examples), so the `--ignore` live-network run no longer sweeps them in. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))


### Changed

- Update C04 data source to point to currently updating data series. [@duncaneddy](https://github.com/duncaneddy) ([#356](https://github.com/duncaneddy/brahe/pull/356))
- Makes the `plots` submodule of the python package an optional dependency installed with `brahe[plots]` and not installed by default to avoid importing the heavy dependencies by default. [@duncaneddy](https://github.com/duncaneddy) ([#360](https://github.com/duncaneddy/brahe/pull/360))
- Optimized the serial spherical-harmonic gravity evaluation (`GravityModel::compute_spherical_harmonics` / `compute_spherical_harmonics_with_workspace`) by precomputing the V/W recurrence reciprocal coefficients and reading the recurrence and accumulation buffers through bounds-check-free column slices. The workspace path is ~1.7–1.8× faster at degree/order 20–80 (the range common to LEO propagation) and ~1.6× faster at 360×360, with results unchanged to within 1e-12 relative. Measured on a 10-core Apple M1 Max. [@duncaneddy](https://github.com/duncaneddy) ([#361](https://github.com/duncaneddy/brahe/pull/361))
- Raised the `ParallelMode::Auto` parallelization threshold for spherical-harmonic gravity from degree 150 to 210. The faster serial path moved the serial/parallel break-even point, so `Auto` now stays serial until parallel evaluation is at least as fast. [@duncaneddy](https://github.com/duncaneddy) ([#361](https://github.com/duncaneddy/brahe/pull/361))
- `compute_spherical_harmonics`, `compute_spherical_harmonics_with_workspace`, `accel_gravity_spherical_harmonics`, and `accel_gravity_spherical_harmonics_with_workspace` now take a `parallel: ParallelMode` argument; `GravityConfiguration::SphericalHarmonic` gained a `parallel` field (defaults to `Auto`). [@duncaneddy](https://github.com/duncaneddy) ([#361](https://github.com/duncaneddy/brahe/pull/361))
- Updated CI workflow to only run integration tests weekly and at release time to reduce frequent occurrence of CI tests failing due to external integration test failures. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Improved CI caching to large texture / basemap files to be updated with integration tests weekly and avoid redownload on most CI runs. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Updated `pyo3` dependency version. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Updated `uv.lock`. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Tagged `TEMPLATE.{py,rs}` and `getting_started/clients_spacetrack.{py,rs}` as `MANUAL` so they are excluded from all automated example runs. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))
- Raised the per-example `TIMEOUT` to 600s for the `starlink_propagation.py` examples, which propagate the full live Starlink constellation and exceeded the default 180s limit during the weekly/release live-network run. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))

### Removed

- `ci` flag from rust and python test suites. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))

### Fixed

- CSSI space weather data loader has been fixed to use the observed f107 rather than the adjusted f107. [@jackyarndley](https://github.com/jackyarndley) ([#362](https://github.com/duncaneddy/brahe/pull/362))
- The NRLMSISE-00 implementation has been updated to use the previous day observed f107 as is standard. [@jackyarndley](https://github.com/jackyarndley) ([#362](https://github.com/duncaneddy/brahe/pull/362))
- Fixed race condition in examples CI tests where two could download the same Natural Earth basemap and corrupt the file for both. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Fixed regression from lazy-impoting `brahe.plots` that caused static-analyzers (griffee) to fail path-traversal. This caused `build-docs` to fail and slipped through due to other test-failure noise. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- Fixed `test-examples` flakyness on external integration dependencies. Ignore testing integration-based examples by default, but exercise them weekly in the integration tests to ensure continued coverage. [@duncaneddy](https://github.com/duncaneddy) ([#365](https://github.com/duncaneddy/brahe/pull/365))
- `par_propagate_to` (Python) no longer raises `TypeError` when given a list containing more than one propagator type; mixed-type lists are now propagated correctly. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))
- Fixed the release `test-examples` integration job failing on `--ignore`: `IGNORE` was an overloaded bucket that mixed live-network examples with templates, credential-gated, slow, and broken examples, so opting into the live-network set pulled in examples that could never pass there. [@duncaneddy](https://github.com/duncaneddy) ([#366](https://github.com/duncaneddy/brahe/pull/366))
## [1.6.0] - 2026-06-05

### Added

- Add `RKF78` / `RKF78Integrator` support for high-order adaptive integration in Rust and Python. [@Mtrya](https://github.com/Mtrya) ([#339](https://github.com/duncaneddy/brahe/pull/339))
- Add RKF78 tests, examples, and API/reference documentation. [@Mtrya](https://github.com/Mtrya) ([#339](https://github.com/duncaneddy/brahe/pull/339))
- Added [GMAT](https://github.com/nasa/GMAT) baselines. [@duncaneddy](https://github.com/duncaneddy) ([#340](https://github.com/duncaneddy/brahe/pull/340))
- Added [Basilisk](https://github.com/AVSLab/basilisk) baselines. [@duncaneddy](https://github.com/duncaneddy) ([#340](https://github.com/duncaneddy/brahe/pull/340))
- `profiles/` subdirectory with short, standalone rust and python scripts for use with profiling tools. [@duncaneddy](https://github.com/duncaneddy) ([#341](https://github.com/duncaneddy/brahe/pull/341))
- Added `justfile` commands to run profiling scripts. [@duncaneddy](https://github.com/duncaneddy) ([#341](https://github.com/duncaneddy/brahe/pull/341))
- Add baselines against Nyx 2.4.0 / ANISE 0.10.1. [@duncaneddy](https://github.com/duncaneddy) ([#342](https://github.com/duncaneddy/brahe/pull/342))
- Added `datasets.icgem` submodule to integrate with ICGEM gravity field model distribution network. Listing and downloading models uses the local brahe cache to minimize network traffic. [@duncaneddy](https://github.com/duncaneddy) ([#343](https://github.com/duncaneddy/brahe/pull/343))
- Added brahe vs astrojax benchmarks. [@duncaneddy](https://github.com/duncaneddy) ([#344](https://github.com/duncaneddy/brahe/pull/344))
- Add `ci-success` stage to enable gating CI auto-merge on completion. [@duncaneddy](https://github.com/duncaneddy) ([#350](https://github.com/duncaneddy/brahe/pull/350))

### Changed

- Extend numerical propagation integrator selection and comparison examples to include RKF78. [@Mtrya](https://github.com/Mtrya) ([#339](https://github.com/duncaneddy/brahe/pull/339))
- Split baselines into speed and accuracy tests. [@duncaneddy](https://github.com/duncaneddy) ([#340](https://github.com/duncaneddy/brahe/pull/340))
- Added `ICGEMModel(body, model_name)` to `GravityModelType` enabling ICGEM models to be specified for numerical orbit propagators. [@duncaneddy](https://github.com/duncaneddy) ([#343](https://github.com/duncaneddy/brahe/pull/343))
- Scaled back python-version test matrix on PRs to minimum supported python version. [@duncaneddy](https://github.com/duncaneddy) ([#345](https://github.com/duncaneddy/brahe/pull/345))
- Removed `anise` default feature dependency to just pull in SPICE kernel features. [@duncaneddy](https://github.com/duncaneddy) ([#350](https://github.com/duncaneddy/brahe/pull/350))

### Fixed

- `EulerAngleOrder` semantics now match what brahe's documentation always described and what the comparative benchmark expects. Prior to this fix, `attitude.euler_angle_to_quaternion` disagreed with OreKit, GMAT, and Basilisk. The discrepancy was traced through to a convention mismatch between the per-order formulas (faithfully transcribed from Diebel 2006) and the aerospace rotation-sequence interpretation users (and the docs) assumed. [@duncaneddy](https://github.com/duncaneddy) ([#338](https://github.com/duncaneddy/brahe/pull/338))
- Fixed issue with CHANGELOG parsing in PRs where a blank line after a section header would fail to validate. [@duncaneddy](https://github.com/duncaneddy) ([#338](https://github.com/duncaneddy/brahe/pull/338))
- Fixed issue with Orekit access baseline not properly inserting leap seconds in accuracy computation leading to in accurate error. [@duncaneddy](https://github.com/duncaneddy) ([#340](https://github.com/duncaneddy/brahe/pull/340))
- Fixed build breaking from conflicting `nalgebra` required dependencies. [@duncaneddy](https://github.com/duncaneddy) ([#350](https://github.com/duncaneddy/brahe/pull/350))
- Regression in use of custom python-based access constraints where they would give an error at use. [@nkgotcode](https://github.com/nkgotcode) [@duncaneddy](https://github.com/duncaneddy) ([#354](https://github.com/duncaneddy/brahe/pull/354))

## [1.5.2] - 2026-05-18

### Fixed

- Fix issue with new release workflow `CHANGELOG` generation that would result in an error causing the release to fail. Release notes now must be manually generated and committed _prior_ to running the release action. [@duncaneddy](https://github.com/duncaneddy) ([#334](https://github.com/duncaneddy/brahe/pull/334))

## [1.5.1] - 2026-05-18

### Added

- `DNumericalOrbitPropagator::builder()` - typestate builder that names required fields and lets optional fields be set by name or omitted. `build()` is only callable once `epoch`, `state`, and `force_config` are all set, enforced via `Set`/`Unset` marker type parameters. [@markusz](https://github.com/markusz) ([#321](https://github.com/duncaneddy/brahe/pull/321))
- Added `states_ecef` and `states_eme2000` to `SGPPropagator`. [@duncaneddy](https://github.com/duncaneddy) ([#331](https://github.com/duncaneddy/brahe/pull/331))
- Added `states_eme2000` to `KeplerianPropagator`. [@duncaneddy](https://github.com/duncaneddy) ([#331](https://github.com/duncaneddy/brahe/pull/331))
- Added `states_eci`, `states_ecef`, `states_gcrf`, `states_itrf`, and `states_eme2000` to `NumericalOrbitPropagator`. [@duncaneddy](https://github.com/duncaneddy) ([#331](https://github.com/duncaneddy/brahe/pull/331))
- Added `states` to `NumericalPropagator`. [@duncaneddy](https://github.com/duncaneddy) ([#331](https://github.com/duncaneddy/brahe/pull/331))
- New high-fidelity propagator benchmarks under `benchmarks/comparative/tasks/propagation_tasks.py`: `numerical_rk4_grav5x5`, `numerical_rk4_grav20x20_sun_moon`, `numerical_rk4_grav80x80_full`. Each runs RK4 over one LEO revolution with matched force-model and frame settings on both sides (OreKit loads brahe's `EGM2008_360.gfc` via `ICGEMFormatReader`, GCRF inertial frame, ITRF / IAU 2006/2000A body-fixed rotation, DE-440 third-body ephemerides, identical spacecraft mass / area / Cd / Cr). [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- New function-level acceleration benchmarks under `benchmarks/comparative/tasks/force_model_tasks.py` for point-mass gravity, 20×20 and 80×80 spherical-harmonic gravity, and third-body Sun/Moon. These evaluate a single acceleration at a fixed state and epoch, isolating force-model code from integrator behaviour. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- "Force Model" section in `docs/about/benchmarks.md` documenting the new function-level comparisons. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `GravityModel::load_uncached(model)` — explicit cold-load primitive for users who need deterministic memory or want to profile the parse path. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `GravityModel::compute_spherical_harmonics_with_workspace(...)` and `accel_gravity_spherical_harmonics_with_workspace(...)` — variants that accept caller-supplied V/W work matrices, letting hot-path callers (the numerical propagator's dynamics closure, batch orbit-determination residual code) amortize the per-call `DMatrix::zeros((n_max + 2)²)` allocation. At 80×80 the saved allocation+memset is ~15 µs per call. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `clear_gravity_model_cache()` — manual cache invalidation, useful after replacing a `FromFile(path)` source on disk or in tests. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `mimalloc` as the global allocator for the `brahe-py` Python extension and the comparative-benchmark Rust binary. The brahe core library remains allocator-agnostic so downstream Rust consumers pick their own. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- Added `SECONDS_PER_DAY` constant. [@duncaneddy](https://github.com/duncaneddy) ([#333](https://github.com/duncaneddy/brahe/pull/333))

### Changed

- `NumericalOrbitPropagator` and `NumericalPropagator` now no longer store accelerations by default to save space and improve performance. [@duncaneddy](https://github.com/duncaneddy) ([#330](https://github.com/duncaneddy/brahe/pull/330))
- The FD fallback for HermiteQuintic is removed. Previously, calling HermiteQuintic on a trajectory without stored accelerations would silently degrade to finite-difference acceleration estimation if 3+ points were available; now it errors. Combined with the earlier store_accelerations: false default flip, any code path that today uses HermiteQuintic either needs NumericalPropagationConfig::with_store_accelerations(true) (propagator path) or trajectory.enable_acceleration_storage() (direct path). [@duncaneddy](https://github.com/duncaneddy) ([#330](https://github.com/duncaneddy/brahe/pull/330))
- **Performance, numerical propagator (no behaviour change)**: the `DNumericalOrbitPropagator` hot path now caches the ECI→body-fixed rotation across integrator stages, shares one `Arc<GravityModel>` across propagators with the same gravity config (no more 60 ms `.gfc` reload per construction), borrows the state vector instead of cloning it on every stage call, returns the orbital derivative as a stack-allocated `Vector6<f64>` instead of allocating a fresh `DVector`, and reuses the spherical-harmonic V/W work matrices across stages. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `GravityModel::from_model_type` is now process-wide cache-backed. Repeated calls for the same `GravityModelType` parse the `.gfc` file once (~60 ms cold), then return owned clones from the cached `Arc` (~1 ms each). The public signature is unchanged. Cache is documented as unbounded — see the `# Caution` block on the function for the growth caveat with many distinct `FromFile(path)` sources, and use `clear_gravity_model_cache()` as the escape hatch. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- `bias_precession_nutation` switched from `iauXys06a` (full IAU 2000A nutation, ~1300 terms) to `iauXys00b` (truncated IAU 2000B, ~77 terms). Per-call cost dropped from ~150 µs to ~2.3 µs; agreement with the full series is sub-milliarcsecond, no regressions in tests. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- Refreshed benchmark plots and docs. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- Changed how adding a flaot second time to an Epoch (`epc + 42.5`) processes the internal addition. Previously, it would loop of remaining nanoseconds and convert them into integer seconds. For large second values, this would result in many loops for a single step, meaning that time addition, a frequent code hot-path, would grow linearly in time. Now it is an O(1) operation. [@duncaneddy](https://github.com/duncaneddy) ([#333](https://github.com/duncaneddy/brahe/pull/333))

### Fixed

- Update urllib3 to mitigate security vulnerability. [@duncaneddy](https://github.com/duncaneddy) ([#326](https://github.com/duncaneddy/brahe/pull/326))
- Some estimation plots used unicode sigma instead of `$\sigma$`. When latex is installed and defaults to pdfLaTeX compiler, this causes an error. Switch to `$\sigma$` everywhere. [@duncaneddy](https://github.com/duncaneddy) ([#331](https://github.com/duncaneddy/brahe/pull/331))
- **`DNumericalOrbitPropagator::propagate_to` may hang with fixed-step RK4.** At `src/propagators/dnumerical_orbit_propagator.rs:2312` the restore-`dt_next` guard used `>` instead of `>=`. With `target_epoch = epc + N * step_size` constructed by repeated arithmetic, float drift could leave `target_rel` slightly larger than the integer multiple; Fixed by changing the comparison to `>=` so the restore fires whenever the integrator returned its preferred step. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))
- Workspace mismatch in `benchmarks/comparative/implementations/rust/Cargo.toml`: the crate predated the brahe repo-root workspace and would fail with *"current package believes it's in a workspace when it's not"*. Added an empty `[workspace]` table so cargo treats the bench as its own workspace root. [@duncaneddy](https://github.com/duncaneddy) ([#332](https://github.com/duncaneddy/brahe/pull/332))

## [1.5.0] - 2026-05-06

### Added

- Added `accel_earth_zonal_gravity` — a hand-rolled closed-form J2–J6 zonal acceleration that mirrors `accel_gravity_spherical_harmonics` with `m = 0` but evaluates ~1.5–2x faster, with agreement to <3 km over a 24 h LEO propagation when both are driven in the same Earth-fixed frame. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Added `GravityConfiguration::EarthZonal { degree }` variant and supporting `ZonalHarmonicsDegree` enum (`J2`..`J6`) so the numerical propagator can use the fast zonal path directly. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Added `FrameTransformationModel` (variants `FullEarthRotation`, `EarthRotationOnly`) and a `frame_transform` field on `ForceModelConfig`. `EarthRotationOnly` skips precession, nutation, and polar motion for ~1.5x faster ECI↔ECEF rotations at the cost of ~0.07° pole-tilt accuracy; `FullEarthRotation` (default) preserves prior behavior. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Added `J3_EARTH`, `J4_EARTH`, `J5_EARTH`, and `J6_EARTH` constants derived from EGM2008. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Added Python bindings for `accel_earth_zonal_gravity`, `ZonalHarmonicsDegree`, `FrameTransformationModel`, `GravityConfiguration.earth_zonal(...)`, and the new `J3_EARTH`–`J6_EARTH` constants. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Added coefficient-derivation tests verifying each `J_n_EARTH` constant matches `-C_n,0 * sqrt(2n + 1)` against the EGM2008 fully-normalized Stokes coefficients. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))

### Changed

- Refined `J2_EARTH` to be derived from the EGM2008 `C_2,0` Stokes coefficient (`1.0826261738522227e-3`) instead of GGM05s. Downstream values change in the 5th significant digit. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Reused a single ECI→body-fixed rotation per dynamics step across gravity, NRLMSISE-00 density, and drag inside `DNumericalOrbitPropagator` rather than recomputing the rotation per force term. [@markusz](https://github.com/markusz) ([#302](https://github.com/duncaneddy/brahe/pull/302))
- Update package lockfile. [@duncaneddy](https://github.com/duncaneddy) ([#306](https://github.com/duncaneddy/brahe/pull/306))

### Removed

- Removed rust artifact caching from CI since existing rust environment already provides (working) caching. [@duncaneddy](https://github.com/duncaneddy) ([#315](https://github.com/duncaneddy/brahe/pull/315))

### Fixed

- Add missing `CHANGELOG.md` release entries. [@duncaneddy](https://github.com/duncaneddy) ([#306](https://github.com/duncaneddy/brahe/pull/306))
- Fix error with release note generation step of release workflow. [@duncaneddy](https://github.com/duncaneddy) ([#306](https://github.com/duncaneddy/brahe/pull/306))
- Fix issue with awk-based release-note generation step of release workflow not populating release note contents. [@duncaneddy](https://github.com/duncaneddy) ([#306](https://github.com/duncaneddy/brahe/pull/306))
- Fixed Starlink examples and documentation prose that still referenced the removed `bh.datasets.celestrak` / `bh::datasets::celestrak` API. [@Mtrya](https://github.com/Mtrya) ([#310](https://github.com/duncaneddy/brahe/pull/310))
- Fix 3rd party changes PRs not generating changelog fragments. [@duncaneddy](https://github.com/duncaneddy) ([#317](https://github.com/duncaneddy/brahe/pull/317))

## [1.4.2] - 2026-04-23
### Fixed

- Fix `brahe-py` Crate not using root `Cargo.toml` as single-source-of-truth for package version. [#298](https://github.com/duncaneddy/brahe/pull/298)
- Corrected `format_exponential` exponent offset in TLE generation ([#300](https://github.com/duncaneddy/brahe/issues/300)). BSTAR and second time derivative values are no longer 10x too small.


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
