//! Convert a Cartesian inertial state to right ascension/declination/range
//! with rates and back, then apply the same site-relative
//! subtract-then-convert pattern to a topocentric line-of-sight state.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // --8<-- [start:state_transforms]
    // RA/Dec/range and their rates: [ra, dec, range, ra_dot, dec_dot, range_dot]
    let x_radec = na::SVector::<f64, 6>::new(45.0, 30.0, 7000e3, 0.01, -0.005, 50.0);
    let x_inertial = bh::state_radec_to_inertial(x_radec, bh::AngleFormat::Degrees);
    println!(
        "Inertial state: pos=[{:.3}, {:.3}, {:.3}] m",
        x_inertial[0], x_inertial[1], x_inertial[2]
    );
    println!(
        "                vel=[{:.6}, {:.6}, {:.6}] m/s",
        x_inertial[3], x_inertial[4], x_inertial[5]
    );

    let x_radec_back = bh::state_inertial_to_radec(x_inertial, bh::AngleFormat::Degrees);
    println!(
        "RA/Dec round-trip: ra={:.6} deg, dec={:.6} deg, range={:.3} m",
        x_radec_back[0], x_radec_back[1], x_radec_back[2]
    );
    println!(
        "                   ra_dot={:.6} deg/s, dec_dot={:.6} deg/s, range_dot={:.3} m/s",
        x_radec_back[3], x_radec_back[4], x_radec_back[5]
    );

    assert!((x_radec_back[0] - x_radec[0]).abs() < 1e-9);
    assert!((x_radec_back[3] - x_radec[3]).abs() < 1e-9);
    // --8<-- [end:state_transforms]

    // --8<-- [start:topocentric]
    // A satellite and an observing site, both as Cartesian inertial states (m, m/s)
    let x_sat = na::SVector::<f64, 6>::new(8000e3, 1000e3, 500e3, -1000.0, 7000.0, 2000.0);
    let x_site = na::SVector::<f64, 6>::new(6378e3, 0.0, 0.0, 0.0, 0.0, 0.0);

    let x_topocentric = x_sat - x_site;
    let x_radec_topo = bh::state_inertial_to_radec(x_topocentric, bh::AngleFormat::Degrees);
    println!(
        "\nTopocentric line of sight: ra={:.6} deg, dec={:.6} deg, range={:.3} m",
        x_radec_topo[0], x_radec_topo[1], x_radec_topo[2]
    );
    // --8<-- [end:topocentric]
}
