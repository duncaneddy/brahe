//! Shared helpers for profile workloads.
//!
//! Every bin under `src/bin/` calls into this module to (a) initialise the EOP
//! and space-weather providers brahe needs at runtime, (b) read a default
//! initial orbital state, and (c) drive the inner workload in a loop until a
//! caller-specified wall-clock duration has elapsed.

use std::time::{Duration, Instant};

use brahe::constants::AngleFormat;
use brahe::coordinates::state_koe_to_eci;
use brahe::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
use brahe::math::SVector6;
use brahe::space_weather::{FileSpaceWeatherProvider, set_global_space_weather_provider};
use brahe::time::{Epoch, TimeSystem};
use nalgebra::DVector;

/// A fixed ISS-like TLE used by the SGP4 profile workloads.
///
/// The exact epoch in the TLE is irrelevant for profiling — we never compare
/// against a reference orbit — but pinning it keeps results comparable across
/// runs of the same workload.
pub const DEFAULT_ISS_TLE_LINE1: &str =
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
pub const DEFAULT_ISS_TLE_LINE2: &str =
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// Install the global EOP + space-weather providers brahe needs for any of
/// the propagators that consume EOP/SW data.
///
/// Mirrors `benchmarks/propagator_benchmarks.rs::setup_providers`. Safe to
/// call more than once: `set_global_*_provider` overwrites the current
/// provider on each call, so the last installation wins.
pub fn setup_providers() {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold)
        .expect("default EOP provider must load");
    set_global_eop_provider(eop);

    let sw = FileSpaceWeatherProvider::from_default_file()
        .expect("default space weather provider must load");
    set_global_space_weather_provider(sw);
}

/// A fixed LEO test orbit initial condition used by every numerical-propagation
/// workload: 500 km altitude sun-synchronous orbit at 2024-01-01 UTC.
///
/// Returns (epoch, eci-state) so callers can hand them directly to
/// `DNumericalOrbitPropagator::new`.
pub fn default_leo_state() -> (Epoch, DVector<f64>) {
    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let oe = SVector6::new(
        brahe::constants::R_EARTH + 500e3,
        0.01,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let dstate = DVector::from_column_slice(
        state_koe_to_eci(oe, AngleFormat::Degrees).as_slice(),
    );
    (epoch, dstate)
}

/// Run `body` in a tight loop until `duration_s` seconds of wall time have
/// elapsed since the first invocation. Returns the number of iterations.
///
/// The function does NOT sleep — it runs `body` back-to-back, so the profiler
/// sees the hot path with no idle gaps. `body` is intentionally `FnMut` so
/// callers can mutate state (e.g. recycle a propagator) across iterations.
pub fn run_until_elapsed<F: FnMut()>(duration_s: f64, mut body: F) -> usize {
    let deadline = Duration::from_secs_f64(duration_s);
    let start = Instant::now();
    let mut iters = 0;
    while start.elapsed() < deadline {
        body();
        iters += 1;
    }
    iters
}

/// Convenience: read `PROFILE_DURATION_S` from the environment with a default
/// of 10 seconds. Bins call this to honour the `--duration N` flag plumbed
/// through the just recipe.
pub fn duration_from_env() -> f64 {
    std::env::var("PROFILE_DURATION_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `run_until_elapsed` must actually run for ≥ the requested duration.
    /// Use a small duration (50 ms) to keep the test fast.
    #[test]
    fn run_until_elapsed_respects_duration() {
        let target_s = 0.05;
        let mut counter = 0;
        let start = Instant::now();
        let iters = run_until_elapsed(target_s, || {
            counter += 1;
        });
        let elapsed = start.elapsed().as_secs_f64();
        assert!(elapsed >= target_s, "elapsed {elapsed} < target {target_s}");
        assert!(iters > 0, "body should run at least once");
        assert_eq!(iters, counter, "iter count must match body invocations");
    }

    #[test]
    fn duration_from_env_reads_env_var_or_returns_ten_by_default() {
        // SAFETY: env var manipulation is process-global. By keeping both
        // branches inside one test, we serialize the mutation against
        // any concurrent test in this binary.
        unsafe { std::env::remove_var("PROFILE_DURATION_S"); }
        assert_eq!(duration_from_env(), 10.0);

        unsafe { std::env::set_var("PROFILE_DURATION_S", "0.25"); }
        assert_eq!(duration_from_env(), 0.25);

        unsafe { std::env::remove_var("PROFILE_DURATION_S"); }
    }
}
