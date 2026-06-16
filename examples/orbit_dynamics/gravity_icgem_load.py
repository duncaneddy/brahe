# /// script
# dependencies = ["brahe"]
# FLAGS = ["IGNORE"]
# ///
"""
Load a GravityModel sourced from the ICGEM catalog.

GravityModelType.icgem(body, name) is interchangeable with the packaged
GravityModelType constants. The .gfc file is downloaded on first use and
cached under $BRAHE_CACHE/icgem/.

To pin a specific degree variant (when ICGEM publishes multiple), append
"-<DEGREE>" to the name argument, e.g. GravityModelType.icgem("earth",
"XGM2019e_2159-760").
"""

import brahe as bh

# Earth — JGM3 (~70x70), small and stable
earth_type = bh.GravityModelType.icgem("earth", "JGM3")
earth_model = bh.GravityModel.from_model_type(earth_type)
print(
    f"Loaded {earth_model.model_name} "
    f"({earth_model.n_max}x{earth_model.m_max}, GM={earth_model.gm:.6e} m^3/s^2)"
)

# Inspect a coefficient — normalized C_{2,0} (J2 term) from the Earth model
c20, s20 = earth_model.get(2, 0)
print(f"\nEarth normalized C(2,0) = {c20:.6e}")
