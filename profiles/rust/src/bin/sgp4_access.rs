//! Profile: SGP4 + access window computation. ISS-like TLE, San Francisco
//! ground location, 7-day window, 10° elevation constraint. Mirrors the
//! workflow in `examples/access/computation/basic_workflow.rs`.

#![allow(missing_docs)]

use brahe::utils::Identifiable;
use brahe::{
    ElevationConstraint, Epoch, PointLocation, SGPPropagator, TimeSystem, location_accesses,
};
use profiles::common::{
    DEFAULT_ISS_TLE_LINE1, DEFAULT_ISS_TLE_LINE2, duration_from_env, run_until_elapsed,
    setup_providers,
};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    setup_providers();
    let duration_s = duration_from_env();

    // Build once outside the hot loop: TLE parsing and constraint construction
    // are negligible relative to the 7-day access search.
    let propagator = SGPPropagator::from_tle(
        DEFAULT_ISS_TLE_LINE1,
        DEFAULT_ISS_TLE_LINE2,
        60.0,
    )
    .expect("TLE must parse")
    .with_name("ISS");

    let location = PointLocation::new(-122.4194, 37.7749, 0.0).with_name("San Francisco");
    let epoch_start = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let epoch_end = epoch_start + 7.0 * 86400.0;
    let constraint = ElevationConstraint::new(Some(10.0), None)
        .expect("elevation constraint construction must succeed");

    let iters = run_until_elapsed(duration_s, || {
        let windows = location_accesses(
            &location,
            &propagator,
            epoch_start,
            epoch_end,
            &constraint,
            None,
            None,
        )
        .expect("location_accesses must succeed");
        std::hint::black_box(windows);
    });

    eprintln!("sgp4_access: {iters} iterations in ~{duration_s:.1}s");
}
