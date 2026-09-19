//! Build the NSW frame on an inclined orbit with a moving Sun state, check the Y axis against the Sun direction, compare the fixed-Sun and moving-Sun rates, do a relative-state round trip, and evaluate it through the frame graph.

#[allow(unused_imports)]
use brahe as bh;
use brahe::frames::{CelestialFrame, FrameEphemerisSource, ReferenceFrame, set_frame_ephemeris_source};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::BraheError;
use brahe::utils::state_providers::SStateProvider;
use nalgebra::Vector6;

/// A state provider returning a fixed state at every epoch, for registering
/// the spacecraft as an object.
struct ConstantProvider(Vector6<f64>);
impl SStateProvider for ConstantProvider {
    fn state(&self, _epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.0)
    }
}

fn main() {
    bh::initialize_eop().unwrap();

    let oe = Vector6::new(bh::R_EARTH + 700e3, 0.05, 97.8, 15.0, 30.0, 45.0);
    let x_sc = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let r_sun = bh::sun_position(epc);
    let v_sun = (bh::sun_position(epc + 0.5) - bh::sun_position(epc - 0.5)) / 1.0;
    let x_sun = Vector6::new(
        r_sun[0], r_sun[1], r_sun[2], v_sun[0], v_sun[1], v_sun[2],
    );

    let r_nsw = bh::rotation_nsw_to_eci(x_sc, x_sun);
    let x_axis = r_nsw.column(0).into_owned();
    let y_axis = r_nsw.column(1).into_owned();
    let r_sc = x_sc.fixed_rows::<3>(0).into_owned();
    let sun_dir = (x_sun.fixed_rows::<3>(0).into_owned() - r_sc).normalize();
    println!("Y . sun_direction > 0: {}", y_axis.dot(&sun_dir) > 0.0);
    println!("|Y . X| < 1e-12: {}", y_axis.dot(&x_axis).abs() < 1e-12);

    // Fixed-Sun approximation: a zero-velocity Sun state omits the Sun's own motion
    let x_sun_fixed = Vector6::new(r_sun[0], r_sun[1], r_sun[2], 0.0, 0.0, 0.0);
    let omega_moving = bh::omega_nsw(x_sc, x_sun);
    let omega_fixed = bh::omega_nsw(x_sc, x_sun_fixed);
    let rate_diff = (omega_moving - omega_fixed).norm();
    println!(
        "Fixed-Sun approximation error within (1e-8, 1e-6) rad/s: {}",
        (1e-8..1e-6).contains(&rate_diff)
    );

    // Relative state round trip: 1 km along Y, 200 m along Z
    let x_rel = Vector6::new(0.0, 1000.0, 200.0, 0.0, 0.0, 0.0);
    let x_deputy = bh::state_nsw_to_eci(x_sc, x_rel, x_sun);
    println!(
        "Round trip error: {:.2e}",
        (bh::state_eci_to_nsw(x_sc, x_deputy, x_sun) - x_rel).norm()
    );

    // Frame graph, forcing the analytic Sun model
    set_frame_ephemeris_source(FrameEphemerisSource::Analytic);
    bh::register_object("SC", ConstantProvider(x_sc), CelestialFrame::GCRF).unwrap();
    let r_graph =
        bh::rotation_frame_to_frame(CelestialFrame::GCRF, ReferenceFrame::NSW("SC"), epc)
            .unwrap();
    println!(
        "Frame graph matches rotation_eci_to_nsw: {}",
        (r_graph - bh::rotation_eci_to_nsw(x_sc, x_sun)).norm() < 1e-9
    );
    bh::clear_object_registry();
    set_frame_ephemeris_source(FrameEphemerisSource::Auto);
}
