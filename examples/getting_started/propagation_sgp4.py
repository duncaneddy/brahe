# /// script
# dependencies = ["brahe"]

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()

# Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
client = bh.celestrak.CelestrakClient()
propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)

# Step propagator forward by 1 hour
epoch = propagator.current_epoch()
propagator.propagate_to(epoch + 3600.0)

# Get final epoch and state
final_epoch = propagator.current_epoch()
final_state = propagator.current_state()
print(f"Initial epoch: {epoch}")
print(f"Final epoch:   {final_epoch}")
print(
    f"Position (km): [{final_state[0] / 1e3:.3f}, {final_state[1] / 1e3:.3f}, {final_state[2] / 1e3:.3f}]"
)
print(
    f"Velocity (m/s): [{final_state[3]:.3f}, {final_state[4]:.3f}, {final_state[5]:.3f}]"
)