# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Sample two numerical propagators on a regular grid and write them as an SP3-d file.
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

epoch = bh.Epoch.from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)

# Two LEO satellites in different planes
propagators = []
for raan in (0.0, 90.0):
    oe = np.array([bh.R_EARTH + 600e3, 0.001, 97.8, raan, 0.0, 0.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    propagators.append(
        bh.NumericalOrbitPropagator(
            epoch,
            state,
            bh.NumericalPropagationConfig.default(),
            bh.ForceModelConfig.earth_gravity(),
        )
    )

# The file is built from the trajectories the propagators store, so propagate first
end = epoch + 3600.0
for prop in propagators:
    prop.propagate_to(end)

# Resample both trajectories every 60 s over the hour they cover
sp3 = bh.SP3.from_trajectories(
    [("L01", propagators[0].trajectory), ("L02", propagators[1].trajectory)],
    step=60.0,
    header=bh.SP3Header(agency="BRHE", comments=["Sampled from numerical propagators"]),
)

print(f"Satellites: {sp3.num_satellites}, epochs: {sp3.num_epochs}")

text = sp3.to_string()
print("\n".join(text.splitlines()[:10]))
