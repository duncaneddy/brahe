//! Route an orbital state between frames with different centers via the frame router.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Initialize EOP and the DE440s ephemeris used to re-center GCRF (Earth)
    // to LCI (Moon) inside the router.
    bh::initialize_eop().unwrap();
    bh::load_common_spice_kernels().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // A GCRF (Earth-centered inertial) state, built from Keplerian elements.
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0);
    let x_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // state_frame_to_frame routes through ICRF and re-centers Earth -> Moon, so
    // the same physical state is now expressed relative to the Moon.
    let x_lci =
        bh::state_frame_to_frame(bh::ReferenceFrame::GCRF, bh::ReferenceFrame::LCI, epc, x_gcrf)
            .unwrap();

    // rotation_frame_to_frame returns only the 3x3 axis rotation (no
    // re-centering); GCRF -> MCMF uses the compiled-in WGCCRE Mars model and
    // needs no kernel.
    let r_gcrf_to_mcmf =
        bh::rotation_frame_to_frame(bh::ReferenceFrame::GCRF, bh::ReferenceFrame::MCMF, epc)
            .unwrap();

    println!("Epoch: {}", epc);
    println!("\nGCRF state (Earth-centered inertial):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        x_gcrf[0] / 1e3,
        x_gcrf[1] / 1e3,
        x_gcrf[2] / 1e3
    );
    println!("\nSame state expressed in LCI (Moon-centered inertial):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        x_lci[0] / 1e3,
        x_lci[1] / 1e3,
        x_lci[2] / 1e3
    );

    // The GCRF->LCI position shift equals the Moon's distance from Earth
    // (~384,400 km), since both frames share ICRF axes and differ only in origin.
    let offset = (x_lci.fixed_rows::<3>(0) - x_gcrf.fixed_rows::<3>(0)).norm();
    println!("\nGCRF->LCI position offset (km): {:.1}", offset / 1e3);
    assert!(offset > 350_000e3 && offset < 410_000e3);

    // The returned rotation is a proper orthonormal direction cosine matrix.
    let m = r_gcrf_to_mcmf;
    assert!((m * m.transpose() - na::Matrix3::identity()).norm() < 1e-9);
    println!("\nExample validated successfully!");
}
