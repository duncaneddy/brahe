# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build a CDM message from scratch and write it to KVN format.
"""

import numpy as np

import brahe as bh
from brahe.ccsds import CDM, CDMObject, CDMRTNCovariance, CDMStateVector

# Define state vectors at TCA for both objects (meters, m/s)
sv1 = CDMStateVector(
    position=[bh.R_EARTH + 500e3, 0.0, 0.0],
    velocity=[0.0, 7612.0, 0.0],
)
sv2 = CDMStateVector(
    position=[bh.R_EARTH + 500.5e3, 10.0, -5.0],
    velocity=[0.0, -7612.0, 0.0],
)

# Define 6x6 RTN covariance matrices (m², m²/s, m²/s²)
cov1 = CDMRTNCovariance(matrix=(np.eye(6) * 1e4).tolist())
cov2 = CDMRTNCovariance(matrix=(np.eye(6) * 2e4).tolist())

# Build object metadata + data
obj1 = CDMObject(
    designator="12345",
    catalog_name="SATCAT",
    name="SATELLITE A",
    international_designator="2020-001A",
    ephemeris_name="NONE",
    covariance_method="CALCULATED",
    maneuverable="YES",
    ref_frame="EME2000",
    state_vector=sv1,
    rtn_covariance=cov1,
)
obj2 = CDMObject(
    designator="67890",
    catalog_name="SATCAT",
    name="DEBRIS FRAGMENT",
    international_designator="2019-050ZZ",
    ephemeris_name="NONE",
    covariance_method="CALCULATED",
    maneuverable="NO",
    ref_frame="EME2000",
    state_vector=sv2,
    rtn_covariance=cov2,
)

# Create CDM message
tca = bh.Epoch.from_datetime(2024, 6, 15, 14, 30, 0.0, 0.0, bh.TimeSystem.UTC)
cdm = CDM(
    originator="BRAHE_EXAMPLE",
    message_id="CDM-2024-001",
    tca=tca,
    miss_distance=502.3,
    object1=obj1,
    object2=obj2,
)

# Set optional collision probability
cdm.collision_probability = 1.5e-04
cdm.collision_probability_method = "FOSTER-1992"

print(f"CDM: {cdm.object1_name} vs {cdm.object2_name}")
print(f"Miss distance: {cdm.miss_distance} m")
print(f"Collision probability: {cdm.collision_probability}")

# Write to KVN
kvn = cdm.to_string("KVN")
print(f"\nKVN output ({len(kvn)} chars)")

# Verify round-trip
cdm2 = CDM.from_str(kvn)
print(f"Round-trip: {cdm2.object1_name} vs {cdm2.object2_name}")
