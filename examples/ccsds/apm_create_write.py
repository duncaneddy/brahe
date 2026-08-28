# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build an APM message from scratch and write it to KVN format.
"""

import numpy as np

import brahe as bh
from brahe.ccsds import APM, APMAngularVelocity, APMQuaternionState

bh.initialize_eop()

# Create a new APM with header info
epoch = bh.Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
apm = APM("BRAHE_EXAMPLE", "LEO SAT", "2024-100A", "UTC", epoch, center_name="EARTH")
apm.message_id = "APM-2024-001"

# Attitude quaternion: spacecraft body frame aligned with ICRF (identity rotation)
apm.add_quaternion_state(
    APMQuaternionState("ICRF", "SC_BODY_1", bh.Quaternion(1.0, 0.0, 0.0, 0.0))
)

# Angular velocity: body spinning about its Z axis at Earth's rotation rate
apm.add_angular_velocity(
    APMAngularVelocity(
        "ICRF", "SC_BODY_1", "SC_BODY_1", np.array([0.0, 0.0, bh.OMEGA_EARTH])
    )
)

print(
    f"Created APM with {len(apm.quaternion_states)} quaternion block, "
    f"{len(apm.angular_velocities)} angular velocity block"
)

# Write to KVN string
kvn = apm.to_string("KVN")
print(f"\nKVN output ({len(kvn)} chars):")
print(kvn)

# Write to file
apm.to_file("/tmp/brahe_example_apm.txt", "KVN")
print("\nWritten to /tmp/brahe_example_apm.txt")

# Verify round-trip
apm2 = APM.from_file("/tmp/brahe_example_apm.txt")
print(
    f"Round-trip: {len(apm2.quaternion_states)} quaternion block, "
    f"{len(apm2.angular_velocities)} angular velocity block"
)
