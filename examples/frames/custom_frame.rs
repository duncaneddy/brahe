//! Register a user-defined body-fixed frame from a rotation callback and use
//! it with the frame router.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // An uncatalogued body (e.g. a newly observed asteroid): self-assign a
    // unique negative NAIF ID for its center, mirroring NAIF's convention.
    const CENTER: i32 = -20001;

    // The frame key is an arbitrary integer handle. It is unrelated to NAIF
    // IDs; it only names the callback registered below so BodyFixedCustom can
    // look it up at evaluation time.
    const KEY: u32 = 1042;

    let t0 = bh::Epoch::from_date(2024, 3, 1, bh::TimeSystem::TDB);
    let rate = 2.0e-4; // spin rate (rad/s), ~8.7 h rotation period

    // The rotation callback is a proper function: it receives an Epoch and
    // returns the 3x3 ICRF -> body-fixed rotation matrix at that instant.
    // Here the body spins uniformly about its z-axis.
    let rotation = move |epc: bh::Epoch| {
        let theta = rate * (epc - t0);
        let (s, c) = theta.sin_cos();
        Ok(na::Matrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0))
    };

    // The optional omega callback is also a function of the epoch, returning
    // the frame's angular velocity vector (rad/s) for the exact velocity
    // transport term. Passing None falls back to numeric differentiation of
    // the rotation callback.
    let omega = move |_epc: bh::Epoch| Ok(na::Vector3::new(0.0, 0.0, rate));

    bh::register_custom_frame(KEY, rotation, Some(Box::new(omega)));

    let inertial = bh::ReferenceFrame::BodyCenteredICRF(CENTER);
    let fixed = bh::ReferenceFrame::BodyFixedCustom {
        center: CENTER,
        key: KEY,
    };

    // Convert an inertial state about the body into its body-fixed frame.
    // Both frames share the same center, so no ephemeris kernel is needed.
    let epc = t0 + 600.0;
    let x_inertial = na::SVector::<f64, 6>::new(7.0e5, -2.0e5, 3.0e5, 10.0, 25.0, -5.0);
    let x_fixed = bh::state_frame_to_frame(inertial, fixed, epc, x_inertial).unwrap();

    println!("Epoch: {}", epc);
    println!("Inertial state:   {:.3?}", x_inertial.as_slice());
    println!("Body-fixed state: {:.3?}", x_fixed.as_slice());

    // Round trip back to the inertial frame recovers the input.
    let x_back = bh::state_frame_to_frame(fixed, inertial, epc, x_fixed).unwrap();
    assert!((x_back - x_inertial).norm() < 1e-6);

    // A point co-rotating with the body is stationary in the body-fixed frame.
    let r_surf = rotation(epc).unwrap().transpose() * na::Vector3::new(1.0e3, 0.0, 0.0);
    let v_surf = omega(epc).unwrap().cross(&r_surf);
    let x_surf = na::SVector::<f64, 6>::new(
        r_surf[0], r_surf[1], r_surf[2], v_surf[0], v_surf[1], v_surf[2],
    );
    let x_surf_fixed = bh::state_frame_to_frame(inertial, fixed, epc, x_surf).unwrap();
    assert!(x_surf_fixed.fixed_rows::<3>(3).norm() < 1e-9);

    bh::unregister_custom_frame(KEY);
    println!("\nExample validated successfully!");
}
