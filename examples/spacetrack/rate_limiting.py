# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrates RateLimitConfig creation and client integration.
"""

import brahe as bh

# Default conservative limits (25/min, 250/hour)
config = bh.RateLimitConfig()
print(f"Default config: {config.max_per_minute}/min, {config.max_per_hour}/hour")

# Custom limits
config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
print(f"Custom config:  {config.max_per_minute}/min, {config.max_per_hour}/hour")

# Disable rate limiting entirely
config = bh.RateLimitConfig.disabled()
print(f"Disabled config: {config.max_per_minute}/min, {config.max_per_hour}/hour")

# Create a client with default rate limiting (no config needed)
client = bh.SpaceTrackClient("user@example.com", "password")
print("\nClient with default rate limiting created")

# Create a client with custom rate limiting
config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
print("Client with custom rate limiting created")

# Create a client with rate limiting disabled
config = bh.RateLimitConfig.disabled()
client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
print("Client with disabled rate limiting created")
