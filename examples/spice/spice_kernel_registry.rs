//! Load SPICE kernels and query ephemeris data from the global registry.
//!
//! This example demonstrates the generic NAIF-ID queries (spk_position/velocity/state),
//! the kernel-scoped variants, and the per-body *_de convenience functions. Downloads
//! the de440s kernel (~33 MB) on first run.

#[allow(unused_imports)]
use brahe as bh;
use brahe::spice::{NAIFId, NAIFKernel};

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_date(2025, 1, 1, bh::TimeSystem::UTC);

    // Loading is idempotent and explicit; spk_* queries also auto-load de440s
    // if no kernel has been loaded yet.
    bh::spice::load_kernel("de440s").unwrap();
    println!("Loaded kernels: {:?}", bh::spice::loaded_kernels());

    // Generic queries take NAIF IDs and resolve across all loaded SPK kernels.
    let r_moon = bh::spice::spk_position(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
    let v_moon = bh::spice::spk_velocity(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
    let x_sun = bh::spice::spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap();

    println!(
        "\nMoon position rel. Earth (km): [{:.3}, {:.3}, {:.3}]",
        r_moon[0] / 1e3,
        r_moon[1] / 1e3,
        r_moon[2] / 1e3
    );
    println!(
        "Moon velocity rel. Earth (m/s): [{:.6}, {:.6}, {:.6}]",
        v_moon[0], v_moon[1], v_moon[2]
    );
    println!(
        "Sun distance rel. Earth (AU): {:.6}",
        x_sun.fixed_rows::<3>(0).norm() / bh::AU
    );

    // Kernel-scoped variants query a single named kernel directly, bypassing
    // cross-kernel chaining and precedence.
    let r_moon_de440s =
        bh::spice::spk_position_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
    println!(
        "\nMoon position from de440s directly (km): [{:.3}, {:.3}, {:.3}]",
        r_moon_de440s[0] / 1e3,
        r_moon_de440s[1] / 1e3,
        r_moon_de440s[2] / 1e3
    );

    // Per-body convenience functions wrap the same queries for the ten most
    // commonly used bodies, selecting the kernel via NAIFKernel.
    let r_mars = bh::spice::mars_position_de(epc, NAIFKernel::DE440s).unwrap();
    let v_mars = bh::spice::mars_velocity_de(epc, NAIFKernel::DE440s).unwrap();
    let x_mars = bh::spice::mars_state_de(epc, NAIFKernel::DE440s).unwrap();

    println!(
        "\nMars position rel. Earth (km): [{:.3}, {:.3}, {:.3}]",
        r_mars[0] / 1e3,
        r_mars[1] / 1e3,
        r_mars[2] / 1e3
    );
    println!(
        "Mars velocity rel. Earth (m/s): [{:.6}, {:.6}, {:.6}]",
        v_mars[0], v_mars[1], v_mars[2]
    );
    println!(
        "Position/state consistency check: {}",
        (r_mars - x_mars.fixed_rows::<3>(0).into_owned()).norm() < 1e-9
    );
}
