//! Filter the Hipparcos catalog by magnitude and cone search to locate a
//! star, then propagate its position to a different epoch with
//! `radec_at_epoch`.
//!
//! FLAGS = ["NETWORK"]

#[allow(unused_imports)]
use brahe as bh;
use bh::datasets::star_catalogs;
use bh::datasets::star_catalogs::StarRecord;

fn main() {
    let hipparcos = star_catalogs::get_hipparcos_catalog(None).unwrap();

    // Magnitude filter: keeps vmag <= max_mag (smaller/more negative is brighter)
    let bright = hipparcos.filter_by_magnitude(2.0);
    println!("Stars brighter than Vmag 2.0: {}", bright.len());

    // Cone search around Sirius' catalog position, in degrees
    let nearby = bright.filter_by_cone(101.28, -16.72, 5.0, bh::AngleFormat::Degrees);
    println!("Bright stars within 5 deg of Sirius: {}", nearby.len());
    for r in nearby.records() {
        println!("  {}: Vmag={:.2}", r.name().unwrap_or_else(|| r.id()), r.vmag.unwrap());
    }

    // Propagate Sirius (HIP 32349) to a future epoch using radec_at_epoch
    // (same proper-motion transformation as apply_proper_motion, applied
    // directly to a catalog record)
    let sirius = hipparcos.get_by_id(32349).unwrap();
    let epc = bh::Epoch::from_datetime(2030, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let (ra, dec) = sirius.radec_at_epoch(epc, bh::AngleFormat::Degrees);
    println!("\nSirius at J2030.0:   RA={:.6} deg, Dec={:.6} deg", ra, dec);
    println!(
        "Sirius at J1991.25:  RA={:.6} deg, Dec={:.6} deg",
        sirius.ra, sirius.dec
    );
}
