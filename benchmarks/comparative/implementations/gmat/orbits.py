"""GMAT benchmark implementations for orbital-element conversions.

GMAT exposes StateConversionUtil.Convert(state, from_rep, to_rep, mu,
flattening, eq_radius, anomaly_type) as a static call. The benchmark's
canonical orbital element vector uses [a (m), e, i (deg), RAAN (deg),
AOP (deg), M (deg)]; GMAT defaults to km and degrees. We pass
anomaly_type='MA' so no true-anomaly conversion is needed.

API signature (from R2026a):
  Convert(state, from_str, to_str, mu)                   -- 4-arg, TA default
  Convert(state, from_str, to_str, mu, flattening, R_eq) -- 6-arg, TA default
  Convert(state, from_str, to_str, mu, flattening, R_eq, anomaly_type) -- 7-arg

Units are reconciled outside the timed region: input meters -> km, output
km -> meters; mu in m^3/s^2 -> km^3/s^2. flattening=0.0, R_eq=6378.1363 km.
The return type is Rvector6; values are extracted via GetElement(i).
"""

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    km_to_m_state,
    m_to_km_state,
    mu_si_to_gmat,
    time_iterations,
)

MU_EARTH_SI = 3.986004418e14  # m^3/s^2 (matches Basilisk's MU_EARTH)
MU_EARTH_GMAT = mu_si_to_gmat(MU_EARTH_SI)  # km^3/s^2

# GMAT StateConversionUtil.Convert 7-arg form: (state, from, to, mu, flat, R_eq, anom_type)
# flattening=0.0 and R_eq=6378.1363 km are the Earth defaults in GMAT.
_GMAT_FLAT = 0.0
_GMAT_REQ = 6378.1363  # km


def _oe_si_to_gmat(oe_si):
    """[a (m), e, i, RAAN, AOP, M] (deg) -> [a (km), e, i, RAAN, AOP, M] (deg)."""
    a, e, i, raan, argp, M = oe_si
    return [a / 1000.0, e, i, raan, argp, M]


def _oe_gmat_to_si(oe_gmat):
    """[a (km), e, i, RAAN, AOP, M] (deg) -> [a (m), e, i, RAAN, AOP, M] (deg)."""
    a, e, i, raan, argp, M = oe_gmat
    return [a * 1000.0, e, i, raan, argp, M]


def _rvec6_to_list(rv6) -> list[float]:
    """Extract a Python list[float] from a GMAT Rvector6."""
    n = rv6.GetSize()
    return [rv6.GetElement(i) for i in range(n)]


def keplerian_to_cartesian(params: dict, iterations: int):
    """Convert Keplerian elements [a (m), e, i, RAAN, AOP, M] (deg) to Cartesian
    [x, y, z, vx, vy, vz] (m, m/s)."""
    import gmatpy as gmat

    # Pre-convert inputs to GMAT's (km, deg) convention OUTSIDE the timed region.
    elements_gmat = [_oe_si_to_gmat(oe) for oe in params["elements"]]

    def run():
        out = []
        for oe in elements_gmat:
            rv6 = gmat.StateConversionUtil.Convert(
                oe, "Keplerian", "Cartesian",
                MU_EARTH_GMAT, _GMAT_FLAT, _GMAT_REQ, "MA"
            )
            out.append(_rvec6_to_list(rv6))
        return out

    times, native_results = time_iterations(run, iterations)
    # Post-convert km -> m OUTSIDE the timed region.
    results = [km_to_m_state(s) for s in native_results]
    return build_task_result(
        "orbits.keplerian_to_cartesian", iterations, times, results
    )


def cartesian_to_keplerian(params: dict, iterations: int):
    """Convert Cartesian [x, y, z, vx, vy, vz] (m, m/s) to Keplerian
    [a (m), e, i, RAAN, AOP, M] (deg)."""
    import gmatpy as gmat

    # Pre-convert inputs to GMAT's (km, km/s) convention OUTSIDE the timed region.
    states_gmat = [m_to_km_state(s) for s in params["states"]]

    def run():
        out = []
        for s in states_gmat:
            rv6 = gmat.StateConversionUtil.Convert(
                s, "Cartesian", "Keplerian",
                MU_EARTH_GMAT, _GMAT_FLAT, _GMAT_REQ, "MA"
            )
            out.append(_rvec6_to_list(rv6))
        return out

    times, native_results = time_iterations(run, iterations)
    # Post-convert km -> m (semi-major axis only) OUTSIDE the timed region.
    results = [_oe_gmat_to_si(oe) for oe in native_results]
    return build_task_result(
        "orbits.cartesian_to_keplerian", iterations, times, results
    )
