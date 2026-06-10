#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Static Providers

    // The simplest data providers are static providers. These always return the same value regarless of date.
    let static_provider = bh::StaticEOPProvider::from_zero();  // All values are zero

    // Now we set the global provider as the static provider, which will be used for all EOP data access across the library.
    bh::set_global_eop_provider(static_provider);

    // File Providers

    // Similarly we can use a file provider, which loads data from a local file
    // The from_default_standard loads the IERS Standard EOP product that is bundled with the package
    // The first argument (True) says to inerpolate values for times between data points
    // The second argument ("Hold") says to hold the last value for times outside the range of the data, instead of throwing an error
    let default_file_provider = bh::FileEOPProvider::from_default_standard(true, bh::EOPExtrapolation::Hold).unwrap();
    bh::set_global_eop_provider(default_file_provider);

    // There are also functions to explicitly download a specific file and and load it from a file
    // Uncomment these lines to try them out
    // let eop_path = "eop_c04.txt";
    // bh::download_c04_eop_file(eop_path).unwrap();
    // let c04_file_provider = bh::FileEOPProvider::from_c04_file(std::path::Path::new(eop_path), true, bh::EOPExtrapolation::Hold).unwrap();
    // bh::set_global_eop_provider(c04_file_provider);
    // println!("EOP data available through: {}", bh::Epoch::from_mjd(bh::get_global_eop_mjd_max(), bh::TimeSystem::UTC));

    // let eop_path = "eop_standard.txt";
    // bh::download_standard_eop_file(eop_path).unwrap();
    // let standard_file_provider = bh::FileEOPProvider::from_standard_file(std::path::Path::new(eop_path), true, bh::EOPExtrapolation::Hold).unwrap();
    // bh::set_global_eop_provider(standard_file_provider);
    // println!("EOP data available through: {}", bh::Epoch::from_mjd(bh::get_global_eop_mjd_max(), bh::TimeSystem::UTC));
}

