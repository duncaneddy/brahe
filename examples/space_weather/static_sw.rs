//! Initialize Static Space Weather Providers

use brahe as bh;

fn main() {
    // Method 1: Static Space Weather Provider - All Zeros
    let sw_static_zeros = bh::space_weather::StaticSpaceWeatherProvider::from_zero();
    bh::space_weather::set_global_space_weather_provider(sw_static_zeros);

    // Method 2: Static Space Weather Provider - Custom Constant Values
    // Parameters: kp, ap, f107_obs, f107_adj, sunspot_number
    let sw_static_values =
        bh::space_weather::StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 150.0, 100);
    bh::space_weather::set_global_space_weather_provider(sw_static_values);
}
