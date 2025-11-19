//! Initialize Space Weather Providers from files

use brahe as bh;
use bh::space_weather::SpaceWeatherExtrapolation;
use std::path::Path;

fn main() {
    // Method 1: Default Provider -> Uses packaged data file within Brahe
    let sw_file_default =
        bh::space_weather::FileSpaceWeatherProvider::from_default_file().unwrap();
    bh::space_weather::set_global_space_weather_provider(sw_file_default);

    // Method 2: Custom File Path -> Replace with actual file path
    if false {
        // Change to true to enable custom file example
        let sw_file_custom = bh::space_weather::FileSpaceWeatherProvider::from_file(
            Path::new("/path/to/sw19571001.txt"), // Replace with actual file path
            SpaceWeatherExtrapolation::Hold,
        )
        .unwrap();
        bh::space_weather::set_global_space_weather_provider(sw_file_custom);
    }
}
