#!/usr/bin/env python3
"""Pre-download the cartopy Natural Earth data used by the documentation plots.

Tessellation and groundtrack plots call ``ax.coastlines(resolution="10m")`` and
``cfeature.LAND/OCEAN/BORDERS``. Each of those lazily downloads Natural Earth
data into cartopy's own cache (``~/.local/share/cartopy``) from naciscdn.org. In
the parallel plot/example pools that download can race or hit upstream
flakiness, so we warm it once, serially, here.

We deliberately drive the *same* cartopy code paths the plots use rather than
hard-coding category/name/scale tuples, so the warmed set can never drift from
what the plots actually request. (No plot uses ``with_scale(...)``, so the
default feature scales are correct; only coastline is pinned to 10m.)
"""

import cartopy.feature as cfeature
from cartopy.io import shapereader


def main() -> None:
    print("Warming cartopy Natural Earth cache...")

    # Explicit 10m coastline (tessellation + groundtrack plots).
    shapereader.natural_earth(resolution="10m", category="physical", name="coastline")

    # Default-scale features touched by groundtrack/contact plots. Iterating the
    # geometries forces the download at each feature's default scale.
    for name, feature in (
        ("LAND", cfeature.LAND),
        ("OCEAN", cfeature.OCEAN),
        ("BORDERS", cfeature.BORDERS),
    ):
        list(feature.geometries())
        print(f"  cartopy {name} ready")

    print("Cartopy Natural Earth cache ready.")


if __name__ == "__main__":
    main()
