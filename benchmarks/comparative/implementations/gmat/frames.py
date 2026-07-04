"""GMAT benchmark implementations for state frame conversions.

Uses CoordinateConverter with per-iteration CoordinateSystem construction
(EarthMJ2000Eq <-> EarthFixed). The gmat.Clear() call inside each iteration
wipes the named-object registry, so CoordinateSystem objects must be rebuilt
each time (matches the Basilisk per-iteration SimBaseClass pattern).

Epoch conversion: the task provides Julian Dates (UTC). GMAT's TimeSystemConverter
uses a modified Julian Date referenced to JD 2430000.0 (not the standard MJD
reference of JD 2400000.5). The correct mapping is:
    GMAT UTCMJD = JD - 2430000.0
    A1MJD = tcv.Convert(gmat_utcmjd, tcv.UTCMJD, tcv.A1MJD)

CoordinateConverter.Convert signature (5-arg, output is filled in-place):
    cc.Convert(a1mjd: float, in_state: Rvector6, in_cs: CoordinateSystem,
               out_state: Rvector6, out_cs: CoordinateSystem)

Unit conventions: task states are in meters / m/s; GMAT uses km / km/s.

Accuracy note: GMAT's EarthFixed uses FK5 + IAU 1980 nutation whereas
brahe/Orekit use IERS 2010 ITRF. Expect bounded errors ≤ ~10 m at LEO.
"""

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    gmat_clear,
    km_to_m_state,
    m_to_km_state,
    time_iterations,
)

# GMAT UTCMJD uses JD - 2430000.0 as its reference (not the standard JD - 2400000.5).
_GMAT_JD_REF = 2430000.0


def _jd_to_a1mjd(jd: float) -> float:
    """JD (UTC) -> GMAT A1MJD using TimeSystemConverter.

    GMAT's UTCMJD reference is JD 2430000.0, not the standard MJD reference
    of JD 2400000.5. Passing (jd - 2400000.5) to UTCMJD yields epoch errors
    of ~30,000 days, producing completely wrong rotation angles.
    """
    import gmatpy as gmat

    tcv = gmat.TimeSystemConverter.Instance()
    gmat_utcmjd = jd - _GMAT_JD_REF
    return float(tcv.Convert(gmat_utcmjd, tcv.UTCMJD, tcv.A1MJD))


def _build_systems():
    """Construct and initialize ECI and ECEF CoordinateSystems for one iteration.

    Must be called after gmat_clear() each iteration because Clear() wipes the
    named-object registry. SetSolarSystem + Initialize are required for the
    CoordinateConverter to resolve the body-fixed rotation.
    """
    import gmatpy as gmat

    ss = gmat.GetSolarSystem()
    cs_eci = gmat.Construct("CoordinateSystem", "EciCs", "Earth", "MJ2000Eq")
    cs_eci.SetSolarSystem(ss)
    cs_eci.Initialize()
    cs_ecef = gmat.Construct("CoordinateSystem", "EcefCs", "Earth", "BodyFixed")
    cs_ecef.SetSolarSystem(ss)
    cs_ecef.Initialize()
    return cs_eci, cs_ecef


def _rvec6_to_list(rv6) -> list[float]:
    """Extract a Python list[float] from a GMAT Rvector6."""
    return [float(rv6.GetElement(i)) for i in range(6)]


def state_eci_to_ecef(params: dict, iterations: int):
    """Convert ECI states (EarthMJ2000Eq) to ECEF (EarthFixed).

    Input and output states are in meters / m/s. Internal GMAT calls use km / km/s.
    Task params format: params["cases"] is a list of {"jd": float, "state": [6 floats]}.
    """
    import gmatpy as gmat

    cases = params["cases"]
    # Pre-compute A1MJD epochs and convert state units OUTSIDE the timed region.
    prepared = [(_jd_to_a1mjd(c["jd"]), m_to_km_state(c["state"])) for c in cases]

    def run():
        out = []
        for a1mjd, state_km in prepared:
            gmat_clear()
            cs_eci, cs_ecef = _build_systems()
            rv_in = gmat.Rvector6(*state_km)
            rv_out = gmat.Rvector6()
            gmat.CoordinateConverter().Convert(a1mjd, rv_in, cs_eci, rv_out, cs_ecef)
            out.append(km_to_m_state(_rvec6_to_list(rv_out)))
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "frames.state_eci_to_ecef",
        iterations,
        times,
        results,
        extra_metadata={"earth_fixed_frame": "EarthFixed (FK5/IAU 1980)"},
    )


def state_ecef_to_eci(params: dict, iterations: int):
    """Convert ECEF states (EarthFixed) to ECI (EarthMJ2000Eq).

    Input and output states are in meters / m/s. Internal GMAT calls use km / km/s.
    Task params format: params["cases"] is a list of {"jd": float, "state": [6 floats]}.
    """
    import gmatpy as gmat

    cases = params["cases"]
    prepared = [(_jd_to_a1mjd(c["jd"]), m_to_km_state(c["state"])) for c in cases]

    def run():
        out = []
        for a1mjd, state_km in prepared:
            gmat_clear()
            cs_eci, cs_ecef = _build_systems()
            rv_in = gmat.Rvector6(*state_km)
            rv_out = gmat.Rvector6()
            # Swap source/dest to convert from ECEF to ECI.
            gmat.CoordinateConverter().Convert(a1mjd, rv_in, cs_ecef, rv_out, cs_eci)
            out.append(km_to_m_state(_rvec6_to_list(rv_out)))
        return out

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "frames.state_ecef_to_eci",
        iterations,
        times,
        results,
        extra_metadata={"earth_fixed_frame": "EarthFixed (FK5/IAU 1980)"},
    )
