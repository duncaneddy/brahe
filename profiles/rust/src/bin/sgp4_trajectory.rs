//! Profile: SGP4 propagation of an ISS-like TLE over a 24h horizon at 60s
//! steps. Workload mirrors `benchmarks/propagator_benchmarks.rs::bench_sgp4_24hour`
//! but loops to fill the requested profile duration.

#![allow(missing_docs)]

use brahe::propagators::SGPPropagator;
use brahe::traits::SStatePropagator;
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

    let iters = run_until_elapsed(duration_s, || {
        let mut prop =
            SGPPropagator::from_tle(DEFAULT_ISS_TLE_LINE1, DEFAULT_ISS_TLE_LINE2, 60.0)
                .expect("TLE must parse");
        let target = prop.current_epoch() + 86400.0;
        prop.propagate_to(target);
        std::hint::black_box(prop);
    });

    eprintln!("sgp4_trajectory: {iters} iterations in ~{duration_s:.1}s");
}
