# /// script
# dependencies = ["brahe"]

import brahe as bh

# Initialize EOP
bh.initialize_eop()

# Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
client = bh.celestrak.CelestrakClient()
propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)

# Configure propagation window
epoch_start = propagator.current_epoch()
epoch_end = epoch_start + 7.0 * 86400.0

# Step propagator forward by 1 hour
propagator.propagate_to(epoch_end)

# Get final epoch and state
final_epoch = propagator.current_epoch()
final_state = propagator.current_state()
print(f"Initial epoch: {epoch_start}")
print(f"Final epoch:   {final_epoch}")
print(
    f"Position (km): [{final_state[0] / 1e3:.3f}, {final_state[1] / 1e3:.3f}, {final_state[2] / 1e3:.3f}]"
)
print(
    f"Velocity (m/s): [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}]"
)
