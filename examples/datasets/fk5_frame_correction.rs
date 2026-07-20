//! Transform an FK5 catalog direction into GCRF for the IAU 2006/2000A frames.
//!
//! FK5 positions are referred to the mean equator and equinox of J2000.0 - the
//! same axes Brahe realizes as EME2000 - not the ICRS/GCRF axes used by the
//! IAU 2006/2000A (CIO-based) frame transformations. An FK5 direction must
//! therefore be rotated from EME2000 to GCRF (a small ~23 mas frame-bias
//! rotation) before it is mixed with a GCRF state or fed into the GCRF -> ITRF
//! transform. Hipparcos and Tycho-2 are already on the ICRS/GCRF axes and need
//! no such rotation - only proper-motion propagation.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::star_catalogs;
use bh::datasets::star_catalogs::StarRecord;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Download FK5 and look up a star by its running number
    let fk5 = star_catalogs::get_fk5_catalog(None).unwrap();
    let rec = fk5.get_by_id(699).unwrap();
    println!("FK5 699: {}", rec.name().unwrap_or_else(|| rec.id()));
    println!(
        "  Catalog (EME2000) position, J2000.0: RA={:.6} deg, Dec={:.6} deg",
        rec.ra, rec.dec
    );

    // Propagate the catalog position to the observation epoch with proper
    // motion. The FK5 axes are fixed and do not depend on the position epoch,
    // so the propagated (ra, dec) is still expressed on the EME2000 axes.
    let epc = bh::Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let (ra, dec) = rec.radec_at_epoch(epc, bh::AngleFormat::Degrees);
    println!("  Propagated to epoch:                 RA={ra:.6} deg, Dec={dec:.6} deg");

    // Direction as a unit vector on the EME2000 axes ...
    let u_eme2000 =
        bh::position_radec_to_inertial(na::Vector3::new(ra, dec, 1.0), bh::AngleFormat::Degrees);

    // ... rotated onto the GCRF/ICRS axes the IAU 2006/2000A transforms expect.
    let u_gcrf = bh::position_eme2000_to_gcrf(u_eme2000);

    // The frame bias is a small (~23 mas) but non-zero rotation.
    let sep_rad = u_eme2000.cross(&u_gcrf).norm().atan2(u_eme2000.dot(&u_gcrf));
    println!(
        "\nEME2000 -> GCRF frame-bias shift: {:.2} mas",
        sep_rad.to_degrees() * 3.6e6
    );

    // The GCRF direction can now be rotated into the Earth-fixed (ITRF) frame
    // with the IAU 2006/2000A CIO-based transform (e.g. for a topocentric
    // pointing).
    let u_itrf = bh::rotation_eci_to_ecef(epc) * u_gcrf;
    println!(
        "GCRF unit vector: [{:.6}, {:.6}, {:.6}]",
        u_gcrf[0], u_gcrf[1], u_gcrf[2]
    );
    println!(
        "ITRF unit vector: [{:.6}, {:.6}, {:.6}]",
        u_itrf[0], u_itrf[1], u_itrf[2]
    );
}
