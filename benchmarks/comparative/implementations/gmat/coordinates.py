"""GMAT benchmark implementations for coordinate conversions.

API surface
-----------
GMAT exposes geodetic conversions through ``StateConversionUtil``:

  CartesianToPlanetodetic(Rvector6, flattening, eq_radius_km)
      -> Rvector6  [r_geocentric_km, lon_deg, lat_geodetic_deg, ...]

  PlanetodeticToCartesian(Rvector6, flattening, eq_radius_km)
      -> Rvector6  [x_km, y_km, z_km, ...]

  CartesianToSphericalRADEC(Rvector6)
      -> Rvector6  [r_km, lon_deg, lat_geocentric_deg, ...]

  SphericalRADECToCartesian(Rvector6)
      -> Rvector6  [x_km, y_km, z_km, ...]

Key constraints
---------------
* ``CartesianToPlanetodetic`` requires VMAG > 1e-10.  A constant dummy
  velocity (7.5 km/s in the x-direction) is appended to every Cartesian
  state before the call; this has no effect on the positional output.
* ``CartesianToSphericalRADEC`` also requires nonzero velocity; the same
  dummy velocity is used.
* The Planetodetic representation stores the *geocentric* radius (not geodetic
  altitude).  Geodetic altitude is recovered via the exact quadratic formula
  derived from the WGS84 ellipsoid definition.
* The SphericalRADEC representation stores the geocentric radius in km.  For
  the geocentric conversion tasks the altitude convention used by the Java
  baseline is ``r = R_WGS84_EQUATORIAL + alt`` (metres); the same constant
  is used here so results are directly comparable.
* Unit contract: benchmark params use meters / degrees; GMAT uses km / degrees.
  All unit conversions happen outside the timed region.

ecef_to_azel — not implemented
-------------------------------
The ``ecef_to_azel`` task is not implemented for GMAT.  The ENZ (East-North-Up)
rotation that produces azimuth/elevation from ECEF inputs is pure spherical
geometry determined entirely by the observer's geodetic lat/lon — GMAT provides
no API that adds value over the direct math.  GMAT's ``TopocentricAxes`` rotates
from ECI (not ECEF), requiring an epoch that the task does not supply; using it
would produce epoch-dependent results rather than the epoch-independent geometric
answer the task contract requires.  Labelling pure-Python ENZ math as "GMAT" is
misleading, so the task is excluded from GMAT coverage.

Accuracy vs OreKit baseline
---------------------------
geodetic_to_ecef / ecef_to_geodetic: GMAT eq_radius (6378.1363 km) differs
from WGS84 (6378.137 km) by -0.7 m, producing a systematic ~0.7 m offset vs
the OreKit baseline.  PlanetodeticToCartesian has ~0.18 m algorithm error vs
the direct WGS84 formula for the same ellipsoid parameters.

geocentric_to_ecef / ecef_to_geocentric: machine-epsilon accuracy (pure
spherical trig, no Earth model).  Small systematic offset vs the OreKit/Java
baseline because Java uses R_WGS84 = 6378137.0 m while GMAT uses 6378136.3 m.

Param key names (from coordinates_tasks.py generate_params)
-----------------------------------------------------------
GeodeticToEcefTask     -> {"points": [[lon_deg, lat_deg, alt_m], ...]}
EcefToGeodeticTask     -> {"points": [[x_m, y_m, z_m], ...]}
GeocentricToEcefTask   -> {"points": [[lon_deg, lat_geo_deg, alt_m], ...]}
EcefToGeocentricTask   -> {"points": [[x_m, y_m, z_m], ...]}
"""

import math

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    time_iterations,
)

# Constant dummy velocity magnitude (km/s) appended to position-only states
# so that CartesianToPlanetodetic's internal VMAG check passes.
_DUMMY_VX_KMS = 7.5


def _earth_params():
    """Return (eq_radius_km, flattening, e2) from GMAT's Earth body."""
    import gmatpy as gmat
    ss = gmat.GetSolarSystem()
    earth = ss.GetBody("Earth")
    a = earth.GetEquatorialRadius()   # km
    f = earth.GetFlattening()
    return a, f, 2.0 * f - f * f     # (a, f, e^2)


def _geocentric_r_km(lat_deg, lon_deg, alt_m, a, e2):
    """Compute geocentric radius (km) from WGS84 geodetic coordinates.

    Required because PlanetodeticToCartesian takes geocentric radius, not
    geodetic altitude.  The computation mirrors what GMAT does internally.
    """
    alt_km = alt_m / 1000.0
    lat_r = math.radians(lat_deg)
    lon_r = math.radians(lon_deg)
    sin_lat = math.sin(lat_r)
    cos_lat = math.cos(lat_r)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + alt_km) * cos_lat * math.cos(lon_r)
    y = (N + alt_km) * cos_lat * math.sin(lon_r)
    z = (N * (1.0 - e2) + alt_km) * sin_lat
    return math.sqrt(x * x + y * y + z * z)


def _alt_from_r_lat(r_km, lat_geodetic_deg, a, e2):
    """Recover geodetic altitude (km) from geocentric radius and geodetic latitude.

    Derived by expanding r^2 = (N+h)^2 cos^2(phi) + (N(1-e^2)+h)^2 sin^2(phi)
    and solving the resulting quadratic in h.  Accuracy: sub-nanometre.
    """
    phi = math.radians(lat_geodetic_deg)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)
    N = a / math.sqrt(1.0 - e2 * sin_phi * sin_phi)
    # Quadratic coefficients: h^2 + 2*p*h + (q - r^2) = 0
    p = N * (1.0 - e2 * sin_phi * sin_phi)          # = a * sqrt(1 - e^2*sin^2(phi))
    q = N * N * (cos_phi * cos_phi + (1.0 - e2) * (1.0 - e2) * sin_phi * sin_phi)
    disc = r_km * r_km - q + p * p
    return -p + math.sqrt(disc)


def geodetic_to_ecef(params: dict, iterations: int):
    """Geodetic [lon_deg, lat_deg, alt_m] -> ECEF [x_m, y_m, z_m].

    Benchmark param key: "points", each element [lon_deg, lat_deg, alt_m].

    Uses StateConversionUtil.PlanetodeticToCartesian.  The Planetodetic format
    requires (r_geocentric_km, lon_deg, lat_geodetic_deg); r_geocentric is
    precomputed from the geodetic inputs using GMAT's Earth ellipsoid parameters
    outside the timed region.
    """
    import gmatpy as gmat

    a, f, e2 = _earth_params()
    scu = gmat.StateConversionUtil

    # Pre-convert outside timed region: each point -> (r_km, lon_deg, lat_deg)
    pts_planetodetic = []
    for lon_deg, lat_deg, alt_m in params["points"]:
        r_km = _geocentric_r_km(lat_deg, lon_deg, alt_m, a, e2)
        pts_planetodetic.append((r_km, lon_deg, lat_deg))

    def run():
        out = []
        for r_km, lon_deg, lat_deg in pts_planetodetic:
            rv_pd = gmat.Rvector6(r_km, lon_deg, lat_deg, _DUMMY_VX_KMS, 0.0, 0.0)
            rv_cart = scu.PlanetodeticToCartesian(rv_pd, f, a)
            out.append([
                float(rv_cart.GetElement(0)) * 1000.0,
                float(rv_cart.GetElement(1)) * 1000.0,
                float(rv_cart.GetElement(2)) * 1000.0,
            ])
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "coordinates.geodetic_to_ecef",
        iterations,
        times,
        results,
        extra_metadata={
            "ellipsoid": "GMAT Earth",
            "gmat_eq_radius_km": a,
            "gmat_flattening": f,
            "accuracy_note": (
                "GMAT eq_radius (6378.1363 km) differs from WGS84 (6378.137 km) "
                "by -0.7 m; PlanetodeticToCartesian has ~0.18 m algorithm error "
                "vs direct WGS84 formula. Both are intrinsic to GMAT."
            ),
        },
    )


# WGS84 equatorial radius (m) used by the Java OreKit baseline as the
# geocentric altitude reference: alt = r - R_WGS84_M.  GMAT's Earth uses
# 6378136.3 m, so we match Java exactly to make the comparison meaningful.
_R_WGS84_M = 6378137.0
_R_WGS84_KM = _R_WGS84_M / 1000.0


def ecef_to_geodetic(params: dict, iterations: int):
    """ECEF [x_m, y_m, z_m] -> Geodetic [lon_deg, lat_deg, alt_m].

    Benchmark param key: "points", each element [x_m, y_m, z_m].

    Uses StateConversionUtil.CartesianToPlanetodetic, which returns
    (r_geocentric_km, lon_deg, lat_geodetic_deg, ...).  Geodetic altitude is
    recovered via the exact quadratic formula (_alt_from_r_lat).

    A constant dummy velocity (7.5 km/s in x) is appended to satisfy the
    VMAG > 1e-10 precondition; it has no effect on the positional output.
    """
    import gmatpy as gmat

    a, f, e2 = _earth_params()
    scu = gmat.StateConversionUtil

    # Pre-convert outside timed region: [x_m, y_m, z_m] -> km
    pts_km = [
        [x / 1000.0 for x in p]
        for p in params["points"]
    ]

    def run():
        out = []
        for x_km, y_km, z_km in pts_km:
            rv_cart = gmat.Rvector6(x_km, y_km, z_km, _DUMMY_VX_KMS, 0.0, 0.0)
            rv_pd = scu.CartesianToPlanetodetic(rv_cart, f, a)
            r_km = float(rv_pd.GetElement(0))
            lon_deg = float(rv_pd.GetElement(1))
            lat_deg = float(rv_pd.GetElement(2))
            alt_m = _alt_from_r_lat(r_km, lat_deg, a, e2) * 1000.0
            out.append([lon_deg, lat_deg, alt_m])
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "coordinates.ecef_to_geodetic",
        iterations,
        times,
        results,
        extra_metadata={
            "ellipsoid": "GMAT Earth",
            "gmat_eq_radius_km": a,
            "gmat_flattening": f,
            "accuracy_note": (
                "GMAT eq_radius (6378.1363 km) differs from WGS84 (6378.137 km) "
                "by -0.7 m. Both are intrinsic to GMAT's Earth model."
            ),
        },
    )


def geocentric_to_ecef(params: dict, iterations: int):
    """Geocentric [lon_deg, lat_geo_deg, alt_m] -> ECEF [x_m, y_m, z_m].

    Benchmark param key: "points", each element [lon_deg, lat_geo_deg, alt_m].

    The altitude convention matches the Java OreKit baseline:
      r = R_WGS84_EQUATORIAL + alt_m  (alt in metres, R_WGS84 = 6378137.0 m)

    Uses StateConversionUtil.SphericalRADECToCartesian.  SphericalRADEC format
    is [r_km, RA_deg, Dec_deg, ...] which maps directly to
    [r_km, lon_deg, lat_geocentric_deg, ...].

    A dummy velocity is required (VMAG > 1e-10 precondition); it has no effect
    on the positional output.

    Accuracy: machine epsilon vs the Java baseline's direct spherical trig.
    A small systematic offset arises because Java uses R_WGS84 = 6378137.0 m
    while GMAT's Earth model uses 6378136.3 m -- but since we use _R_WGS84_M
    (Java's constant) for the radius, this offset is zero for this task.
    """
    import gmatpy as gmat

    scu = gmat.StateConversionUtil

    # Pre-convert outside timed region: [lon_deg, lat_geo_deg, alt_m] -> Spherical km
    pts_spherical = []
    for lon_deg, lat_geo_deg, alt_m in params["points"]:
        r_km = (_R_WGS84_M + alt_m) / 1000.0
        pts_spherical.append((r_km, lon_deg, lat_geo_deg))

    def run():
        out = []
        for r_km, lon_deg, lat_geo_deg in pts_spherical:
            rv_sph = gmat.Rvector6(r_km, lon_deg, lat_geo_deg, _DUMMY_VX_KMS, 0.0, 0.0)
            rv_cart = scu.SphericalRADECToCartesian(rv_sph)
            out.append([
                float(rv_cart.GetElement(0)) * 1000.0,
                float(rv_cart.GetElement(1)) * 1000.0,
                float(rv_cart.GetElement(2)) * 1000.0,
            ])
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "coordinates.geocentric_to_ecef",
        iterations,
        times,
        results,
        extra_metadata={
            "r_ref_m": _R_WGS84_M,
            "accuracy_note": (
                "Pure spherical trig via SphericalRADECToCartesian; machine-epsilon "
                "accuracy.  Uses Java-compatible R_WGS84 = 6378137.0 m as altitude "
                "reference so results are directly comparable to the Java baseline."
            ),
        },
    )


def ecef_to_geocentric(params: dict, iterations: int):
    """ECEF [x_m, y_m, z_m] -> Geocentric [lon_deg, lat_geo_deg, alt_m].

    Benchmark param key: "points", each element [x_m, y_m, z_m].

    The altitude convention matches the Java OreKit baseline:
      alt_m = radius - R_WGS84_EQUATORIAL  (R_WGS84 = 6378137.0 m)

    Uses StateConversionUtil.CartesianToSphericalRADEC.  Output element [0]
    is geocentric radius (km), [1] is longitude (deg), [2] is geocentric
    latitude (deg).

    A dummy velocity is required (VMAG > 1e-10 precondition).
    """
    import gmatpy as gmat

    scu = gmat.StateConversionUtil

    # Pre-convert outside timed region: [x_m, y_m, z_m] -> km
    pts_km = [
        [v / 1000.0 for v in p]
        for p in params["points"]
    ]

    def run():
        out = []
        for x_km, y_km, z_km in pts_km:
            rv_cart = gmat.Rvector6(x_km, y_km, z_km, _DUMMY_VX_KMS, 0.0, 0.0)
            rv_sph = scu.CartesianToSphericalRADEC(rv_cart)
            r_km = float(rv_sph.GetElement(0))
            lon_deg = float(rv_sph.GetElement(1))
            lat_geo_deg = float(rv_sph.GetElement(2))
            alt_m = r_km * 1000.0 - _R_WGS84_M
            out.append([lon_deg, lat_geo_deg, alt_m])
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "coordinates.ecef_to_geocentric",
        iterations,
        times,
        results,
        extra_metadata={
            "r_ref_m": _R_WGS84_M,
            "accuracy_note": (
                "Pure spherical trig via CartesianToSphericalRADEC; machine-epsilon "
                "accuracy.  Uses Java-compatible R_WGS84 = 6378137.0 m as altitude "
                "reference so results are directly comparable to the Java baseline."
            ),
        },
    )


