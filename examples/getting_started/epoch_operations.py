# /// script
# dependencies = ["brahe"]

import brahe as bh

# Initialize EOP
bh.initialize_eop()

# Start with an epoch
epoch = bh.Epoch(2024, 1, 1)

# Add time (in seconds)
epoch_plus_1_day = epoch + 86400    # Add one day
epoch_plus_1_hour = epoch + 3600    # Add one hour
epoch_plus_1_ns = epoch + 1e-9      # Add one nanosecond

# Subtract time (in seconds)
epoch_minus_1_day = epoch - 86400   # Subtract one day
epoch_minus_1_hour = epoch - 3600   # Subtract one hour
epoch_minus_1_ns = epoch - 1e-9     # Subtract one nan

# Get difference between two epochs (in seconds)
difference = epoch_plus_1_day - epoch  # Should be 86400 seconds
print(f"Difference in seconds: {difference:.2f}")

# Comparison operations
print(f"epoch < epoch_plus_1_day: {epoch < epoch_plus_1_day}")
print(f"epoch == epoch_minus_1_day: {epoch == epoch_minus_1_day}")
print(f"epoch > epoch_minus_1_day: {epoch > epoch_minus_1_day}")