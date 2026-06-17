# /// script
# dependencies = ["brahe"]
# FLAGS = ["MANUAL"]

import brahe as bh
import os

# Initialize EOP
bh.initialize_eop()

# Authenticate with Space-Track using account credentials
client = bh.spacetrack.SpaceTrackClient(
    os.environ["SPACETRACK_USERNAME"], os.environ["SPACETRACK_PASSWORD"]
)

# Query the latest GP record for the ISS (NORAD ID 25544)
query = (
    bh.SpaceTrackQuery(bh.RequestClass.GP)
    .filter("NORAD_CAT_ID", "25544")
    .order_by("EPOCH", bh.SortOrder.DESC)
    .limit(1)
)
records = client.query_gp(query)

# Create an SGP4 propagator from the GP record
propagator = records[0].to_sgp_propagator(step_size=60.0)

# Configure propagation window
epoch_start = propagator.current_epoch()
epoch_end = epoch_start + 7.0 * 86400.0

# Propagate forward 7 days
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
