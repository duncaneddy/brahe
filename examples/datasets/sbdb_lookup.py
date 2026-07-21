# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Resolve a small body in the JPL Small-Body Database (SBDB).

Looks up a search string (name or designation) and returns its NAIF/SPK ID
and SI physical parameters. Responses are cached under $BRAHE_CACHE/sbdb and
reused for 30 days, so this hits the network only once per machine.
"""

import brahe as bh

client = bh.datasets.sbdb.SBDBClient()
ceres = client.lookup("Ceres")

print(f"NAIF ID: {ceres.naif_id()}")
print(f"Full name: {ceres.full_name}")
print(f"GM: {ceres.gm:.6e} m^3/s^2")
print(f"Radius: {ceres.radius:.1f} m")
