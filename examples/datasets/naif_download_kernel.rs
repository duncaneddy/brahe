//! Download a NAIF DE kernel for planetary ephemeris data.
//!
//! This example demonstrates how to download and cache DE (Development Ephemeris)
//! kernels from NASA JPL's NAIF archive.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Download de440s kernel (smaller variant, ~33MB)
    // This will download once and cache for future use
    let kernel_path = bh::datasets::naif::download_de_kernel("de440s", None).unwrap();

    println!("Kernel cached at: {}", kernel_path.display());

    // Subsequent calls use the cached file - no re-download
    let kernel_path_again = bh::datasets::naif::download_de_kernel("de440s", None).unwrap();
    println!("Retrieved from cache: {}", kernel_path_again.display());

    // Optionally copy to a specific location
    let output_path = std::path::PathBuf::from("/tmp/my_kernel.bsp");
    let copied_path = bh::datasets::naif::download_de_kernel("de440s", Some(output_path)).unwrap();
    println!("Copied to: {}", copied_path.display());

    // Expected output:
    // Kernel cached at: /Users/username/.cache/brahe/naif/de440s.bsp
    // Retrieved from cache: /Users/username/.cache/brahe/naif/de440s.bsp
    // Copied to: /tmp/my_kernel.bsp
}
