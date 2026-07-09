"""GMAT benchmark implementations for force-model acceleration evaluations.

GMAT exposes `ForceModel.GetDerivativesForSpacecraft(sc)` which returns an
Rvector6 of [vx, vy, vz, ax, ay, az] in km/s and km/s².  We call it
`n_samples` times per iteration (matching the python/rust/java pattern) to
amortize construction overhead and produce comparable per-call timing.

Two input shapes are supported per task:

- Perf (single IC): one fixed state, inner loop of ``n_samples`` repetitions
  for amortized timing — returns one acceleration vector.
- Accuracy (multi-IC): a ``cases`` list of {jd, state_eci} ICs, evaluated once
  each — returns one acceleration per case so the harness can compare per-sample.

Unit conversions:
  - Input state: m / m/s (task params) -> km / km/s (GMAT SetField)
  - Output acceleration: km/s² (GMAT) -> m/s² (task result, elements 3–5)

Epoch: tasks supply a Julian Date; GMAT spacecraft epoch uses UTCMJD which
references JD 2430000.0.  We set DateFormat='UTCModJulian' then
Epoch=str(jd - 2430000.0).  The epoch only matters for third-body ephemeris
lookups (Sun, Luna); point-mass and spherical-harmonic evaluations are
epoch-independent.

ForceModel initialization pattern:
  1. gmat.Construct('ForceModel', 'FM')
  2. Add force sub-objects.
  3. fm.SetSolarSystem(gmat.GetSolarSystem()) — required for third-body forces.
  4. fm.Initialize()
  5. fm.GetDerivativesForSpacecraft(sc) — returns Rvector6.

Note: fm.SetSpacecraft() does NOT exist in gmatpy R2026a; the spacecraft is
passed directly to GetDerivativesForSpacecraft().
"""

from typing import Callable

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    gmat_clear,
    m_to_km_state,
    time_iterations,
)

# GMAT UTCMJD reference: subtract this from a Julian Date to get UTCMJD.
_JD_TO_UTCMJD_OFFSET = 2430000.0

# EGM96 gravity coefficient file shipped with GMAT R2026a.
# Supports up to degree/order 360, so it covers both the 20x20 and 80x80
# benchmark tasks.  JGM2.cof only covers degree ≤ 70 and would fail for 80x80.
# GetDerivativesForSpacecraft resolves the bare filename against GMAT's own
# data directory, so no absolute path is needed.
_EGM96_COF = "EGM96.cof"


def _set_spacecraft(sc, state_m: list[float], jd: float) -> None:
    """Configure spacecraft state and epoch from SI inputs."""
    state_km = m_to_km_state(state_m)
    sc.SetField("StateType", "Cartesian")
    sc.SetField("DateFormat", "UTCModJulian")
    sc.SetField("Epoch", str(jd - _JD_TO_UTCMJD_OFFSET))
    for name, val in zip(["X", "Y", "Z", "VX", "VY", "VZ"], state_km):
        sc.SetField(name, val)
    sc.SetField("DryMass", 100.0)
    sc.SetField("Cd", 2.2)
    sc.SetField("DragArea", 1.0)
    sc.SetField("Cr", 1.5)
    sc.SetField("SRPArea", 1.0)


def _extract_accel_m_s2(deriv) -> list[float]:
    """Extract acceleration from a GetDerivativesForSpacecraft Rvector6 as
    a flat [ax, ay, az] in m/s². GMAT stores [vx, vy, vz, ax, ay, az] in
    elements 0–5; acceleration occupies indices 3–5 in km/s².
    """
    return [deriv.GetElement(i) * 1000.0 for i in range(3, 6)]


# ---------------------------------------------------------------------------
# Force-model builders
# ---------------------------------------------------------------------------


def _build_point_mass_fm():
    """Build a ForceModel with Earth point-mass gravity only."""
    import gmatpy as gmat

    sc = gmat.Construct("Spacecraft", "Sat")
    fm = gmat.Construct("ForceModel", "FM")
    pmf = gmat.Construct("PointMassForce", "EarthPM")
    pmf.SetField("BodyName", "Earth")
    fm.AddForce(pmf)
    fm.SetSolarSystem(gmat.GetSolarSystem())
    fm.Initialize()
    return sc, fm


def _build_spherical_harmonics_fm(degree: int, order: int):
    """Build a ForceModel with EGM96 spherical-harmonic gravity."""
    import gmatpy as gmat

    sc = gmat.Construct("Spacecraft", "Sat")
    fm = gmat.Construct("ForceModel", "FM")
    grav = gmat.Construct("GravityField", "EarthGravity")
    grav.SetField("BodyName", "Earth")
    grav.SetField("PotentialFile", _EGM96_COF)
    grav.SetField("Degree", degree)
    grav.SetField("Order", order)
    fm.AddForce(grav)
    fm.SetSolarSystem(gmat.GetSolarSystem())
    fm.Initialize()
    return sc, fm


def _build_third_body_fm(body_name: str):
    """Build a ForceModel with a single third-body PointMassForce."""
    import gmatpy as gmat

    sc = gmat.Construct("Spacecraft", "Sat")
    fm = gmat.Construct("ForceModel", "FM")
    tb = gmat.Construct("PointMassForce", f"{body_name}TB")
    tb.SetField("BodyName", body_name)
    fm.AddForce(tb)
    fm.SetSolarSystem(gmat.GetSolarSystem())
    fm.Initialize()
    return sc, fm


# ---------------------------------------------------------------------------
# Generic timed evaluator: dispatches between perf single-IC and accuracy
# multi-IC sweep based on the params shape.
# ---------------------------------------------------------------------------


def _make_run(params: dict, build_fn: Callable):
    """Return a zero-argument callable that builds + evaluates the force model.

    The ForceModel and Spacecraft are reconstructed on every call so that
    object-construction cost is included in the timing (consistent with the
    per-iteration methodology used by the Python/Java baselines). On the
    accuracy path each case rebuilds the spacecraft state but reuses the
    force-model construction pattern.
    """
    cases = params.get("cases")

    if cases is None:
        state_m = params["state_eci"]
        jd = params["jd"]
        n_samples = params.get("n_samples", 1)

        def run_perf():
            gmat_clear()
            sc, fm = build_fn()
            _set_spacecraft(sc, state_m, jd)
            last_deriv = None
            for _ in range(n_samples):
                last_deriv = fm.GetDerivativesForSpacecraft(sc)
            return [_extract_accel_m_s2(last_deriv)]

        return run_perf

    def run_sweep():
        gmat_clear()
        sc, fm = build_fn()
        out = []
        for case in cases:
            _set_spacecraft(sc, case["state_eci"], case["jd"])
            deriv = fm.GetDerivativesForSpacecraft(sc)
            out.append(_extract_accel_m_s2(deriv))
        return out

    return run_sweep


# ---------------------------------------------------------------------------
# Public entry points (matched to force_model_tasks.py task names)
# ---------------------------------------------------------------------------


def accel_point_mass_gravity(params: dict, iterations: int):
    times, result = time_iterations(
        _make_run(params, _build_point_mass_fm),
        iterations,
    )
    return build_task_result(
        "force_model.accel_point_mass_gravity", iterations, times, result
    )


def accel_spherical_harmonics_20(params: dict, iterations: int):
    times, result = time_iterations(
        _make_run(params, lambda: _build_spherical_harmonics_fm(20, 20)),
        iterations,
    )
    return build_task_result(
        "force_model.accel_spherical_harmonics_20", iterations, times, result
    )


def accel_spherical_harmonics_80(params: dict, iterations: int):
    times, result = time_iterations(
        _make_run(params, lambda: _build_spherical_harmonics_fm(80, 80)),
        iterations,
    )
    return build_task_result(
        "force_model.accel_spherical_harmonics_80", iterations, times, result
    )


def accel_third_body_sun(params: dict, iterations: int):
    times, result = time_iterations(
        _make_run(params, lambda: _build_third_body_fm("Sun")),
        iterations,
    )
    return build_task_result(
        "force_model.accel_third_body_sun", iterations, times, result
    )


def accel_third_body_moon(params: dict, iterations: int):
    times, result = time_iterations(
        _make_run(params, lambda: _build_third_body_fm("Luna")),
        iterations,
    )
    return build_task_result(
        "force_model.accel_third_body_moon", iterations, times, result
    )
