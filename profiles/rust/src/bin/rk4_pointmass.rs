//! Profile: RK4 numerical propagation, point-mass gravity only.
//!
//! No perturbations (no third body, no drag, no SRP, no relativity). Useful
//! as a CPU baseline — the flamegraph should be dominated by integrator stage
//! evaluations and point-mass force evaluation.

#![allow(missing_docs)]

use brahe::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, GravityConfiguration,
    NumericalPropagationConfig,
};
use brahe::traits::DStatePropagator;
use profiles::common::{
    default_leo_state, duration_from_env, run_until_elapsed, setup_providers,
};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    setup_providers();
    let (epoch, dstate) = default_leo_state();

    let force = ForceModelConfig {
        gravity: GravityConfiguration::PointMass,
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
        frame_transform: Default::default(),
    };
    let prop_cfg = NumericalPropagationConfig::default();
    let duration_s = duration_from_env();

    let iters = run_until_elapsed(duration_s, || {
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            dstate.clone(),
            prop_cfg.clone(),
            force.clone(),
            None,
            None,
            None,
            None,
        )
        .expect("propagator construction must succeed");
        prop.propagate_to(epoch + 86400.0);
        std::hint::black_box(prop);
    });

    eprintln!("rk4_pointmass: {iters} iterations in ~{duration_s:.1}s");
}
