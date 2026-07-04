# /// script
# dependencies = ["brahe"]

import brahe as bh

# Create from calendar date
epoch_1 = bh.Epoch(2024, 1, 1, time_system=bh.TimeSystem.UTC)
epoch_2 = bh.Epoch(2024, 1, 1, 0, 0, 0, time_system=bh.TimeSystem.GPS)
print(f"Epoch 1: {epoch_1}")
print(f"Epoch 2: {epoch_2}")
print(f"(Epoch 2) - (Epoch 1): {epoch_2 - epoch_1} seconds")

# Compare two Epochs
epoch_3 = bh.Epoch(2024, 1, 1, time_system=bh.TimeSystem.TAI)
epoch_4 = bh.Epoch(2024, 1, 1, time_system=bh.TimeSystem.UTC)
print(f"Epoch 3: {epoch_3}")
print(f"Epoch 4: {epoch_4}")
print(f"Epoch 3 > Epoch 4: {epoch_3 > epoch_4}")

# Output as MJD in time system
print(f"Epoch 1 MJD (TT): {epoch_1.mjd_as_time_system(bh.TimeSystem.TT)}")
print(f"Epoch 2 MJD (TT): {epoch_2.mjd_as_time_system(bh.TimeSystem.TT)}")
