//! Demonstrates RateLimitConfig creation and client integration.

#[allow(unused_imports)]
use brahe as bh;
use bh::spacetrack::{RateLimitConfig, SpaceTrackClient};

fn main() {
    // Default conservative limits (25/min, 250/hour)
    let config = RateLimitConfig::default();
    println!("Default config: {}/min, {}/hour", config.max_per_minute, config.max_per_hour);

    // Custom limits
    let config = RateLimitConfig {
        max_per_minute: 10,
        max_per_hour: 100,
    };
    println!("Custom config:  {}/min, {}/hour", config.max_per_minute, config.max_per_hour);

    // Disable rate limiting entirely
    let config = RateLimitConfig::disabled();
    println!("Disabled config: {}/min, {}/hour", config.max_per_minute, config.max_per_hour);

    // Create a client with default rate limiting (no config needed)
    let _client = SpaceTrackClient::new("user@example.com", "password");
    println!("\nClient with default rate limiting created");

    // Create a client with custom rate limiting
    let config = RateLimitConfig {
        max_per_minute: 10,
        max_per_hour: 100,
    };
    let _client = SpaceTrackClient::with_rate_limit("user@example.com", "password", config);
    println!("Client with custom rate limiting created");

    // Create a client with rate limiting disabled
    let config = RateLimitConfig::disabled();
    let _client = SpaceTrackClient::with_rate_limit("user@example.com", "password", config);
    println!("Client with disabled rate limiting created");
}

