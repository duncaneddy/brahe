//! This example demonstrates ways to initialize EOP from files using Brahe.

use brahe as bh;

fn main() {
    // Method 1: Default Providers -> These are packaged data files within Brahe

    // File-based EOP Provider - Default IERS Standard with Hold Extrapolation
    let eop_file_default = bh::eop::FileEOPProvider::from_default_standard(
        true,                                   // Interpolation -> if True times between data points are interpolated
        bh::eop::EOPExtrapolation::Hold        // Extrapolation method -> How accesses outside data range are handled
    ).unwrap();
    bh::eop::set_global_eop_provider(eop_file_default);

    // File-based EOP Provider - Default C04 Standard with Zero Extrapolation
    let eop_file_c04 = bh::eop::FileEOPProvider::from_default_c04(
        false,
        bh::eop::EOPExtrapolation::Zero
    ).unwrap();
    bh::eop::set_global_eop_provider(eop_file_c04);

    // Method 2: Custom File Paths -> Replace 'path_to_file.txt' with actual file paths

    if false {  // Change to true to enable custom file examples
        // File-based EOP Provider - Custom Standard File
        let eop_file_custom = bh::eop::FileEOPProvider::from_standard_file(
            std::path::Path::new("path_to_standard_file.txt"),  // Replace with actual file path
            true,                                                 // Interpolation
            bh::eop::EOPExtrapolation::Hold                      // Extrapolation
        ).unwrap();
        bh::eop::set_global_eop_provider(eop_file_custom);

        // File-based EOP Provider - Custom C04 File
        let eop_file_custom_c04 = bh::eop::FileEOPProvider::from_c04_file(
            std::path::Path::new("path_to_c04_file.txt"),  // Replace with actual file path
            true,                                           // Interpolation
            bh::eop::EOPExtrapolation::Hold                // Extrapolation
        ).unwrap();
        bh::eop::set_global_eop_provider(eop_file_custom_c04);
    }
}
