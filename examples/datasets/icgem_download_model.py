# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Download a spherical harmonic gravity model from ICGEM.

Downloads are cached under $BRAHE_CACHE/icgem/models/<body>/ keyed on
(name, degree, hash-prefix). Subsequent calls for the same model are
served from disk.

To pin a specific degree variant — when ICGEM publishes more than one
truncation of the same model — append "-<DEGREE>" to the name argument,
e.g. download_model("earth", "XGM2019e_2159-760").
"""

import brahe.datasets as datasets

# JGM3 is small (~70x70) and stable — a good demonstration target.
# Passing just the name selects the largest published degree variant.
path = datasets.icgem.download_model("earth", "JGM3")
print(f"Cached at: {path}")

# Optionally also copy the file to a chosen location (cache still populated)
copied = datasets.icgem.download_model(
    "earth", "JGM3", output_path="/tmp/icgem_jgm3.gfc"
)
print(f"Copied to: {copied}")

# Lunar model — body name routes to the celestial catalog
moon_path = datasets.icgem.download_model("moon", "GLGM-1")
print(f"Lunar model cached at: {moon_path}")
