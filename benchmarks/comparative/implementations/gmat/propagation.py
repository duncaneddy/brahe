"""GMAT propagation benchmarks: numerical (RK4), analytical (Keplerian), SGP4.

Numerical per-iteration pattern:
  gmat_clear()
  IC = StateConversionUtil.Convert(oe_gmat, 'Keplerian', 'Cartesian', mu, ...)
  sc = Construct('Spacecraft', 'Sat')  -> set Cartesian state
  ps = Construct('PropSetup', 'PS')
  ps.GetODEModel().SetSolarSystem(...)   # Earth point-mass already in ODE
  ps.AddPropObject(sc)
  ps.SetField('Type', 'RungeKutta4')
  ps.PrepareInternals()
  inner = ps.GetPropagator(); inner.SetStepSize(dt); inner.Initialize()
  for _ in range(n_steps): inner.Step(dt); state = inner.GetState()

Analytical Keplerian per-iteration pattern (keplerian_single / keplerian_trajectory):
  Uses PropSetup with Type='Keplerian'. GMAT R2026a supports this natively;
  all five name variants ('Keplerian', 'TwoBody', 'CKeplerian', 'Analytic',
  'AnalyticKeplerian') produce identical results. 'Keplerian' is used here.
  The ODE model still requires SetSolarSystem() before PrepareInternals().
  For keplerian_single, each case gets its own PropSetup stepped by dt in one
  shot (analytical propagation is exact regardless of step size).
  For keplerian_trajectory, a single PropSetup is stepped n_steps times.

SGP4 pattern (sgp4_single / sgp4_trajectory):
  GMAT's SGP4 propagator (Type=SPICESGP4) requires a TLE file and is only
  accessible via the script-based API (gmat.LoadScript / gmat.RunScript).
  The API-only path (Construct + PrepareInternals) always falls back to
  RungeKutta89; only LoadScript correctly wires up the TLEPropagator plugin.

  Implementation uses a dynamically generated GMAT script that:
  - Sets Spacecraft.EphemerisName to a temp TLE file written in TMPDIR
  - Uses Type=SPICESGP4, InitialStepSize large enough (2x max offset) to force
    exactly one SGP4 call per Propagate command (SGP4 is analytical)
  - Defines a TEME coordinate system (TEME_CS) so output matches the TEME frame
    used by brahe, Java, and Rust SGP4 implementations
  - Logs state via ReportFile; parses lines 1..N (skip line 0 = initial state)

  gmat_clear() is safe to call between SPICESGP4 iterations (no SetReference
  crash; that crash only affects DragForce + atmosphere model pairs).

GMAT units: km / km/s. IC conversion from Keplerian (MA) done via
StateConversionUtil.Convert(..., 'Keplerian', 'Cartesian', mu, 0.0, R_eq, 'MA').
Output is post-converted EarthMJ2000Eq -> GCRF and km -> m outside timed region.

GMAT UTCMJD reference is JD 2430000.0 (not 2400000.5).
"""

import os
import tempfile

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    gmat_clear,
    mj2000_to_gcrf,
    time_iterations,
)

# GMAT UTCMJD reference: jd_utc - this = UTCMJD
_JD_TO_UTCMJD = 2430000.0

# Earth mu in km^3/s^2 — matches brahe/Orekit/Basilisk value, converted from SI
_MU_EARTH_GMAT = 3.986004418e5  # km^3/s^2

# GMAT Earth equatorial radius and flattening used in StateConversionUtil
_GMAT_R_EQ = 6378.1363  # km
_GMAT_FLAT = 0.0


def _oe_deg_si_to_gmat(oe_deg_si):
    """[a (m), e, i_deg, RAAN_deg, AOP_deg, M_deg] -> [a (km), e, i, RAAN, AOP, M]."""
    a_m, e, i, raan, argp, M = oe_deg_si
    return [a_m / 1000.0, e, i, raan, argp, M]


def _keplerian_to_cartesian_km(oe_gmat: list[float]) -> list[float]:
    """Convert [a(km), e, i, RAAN, AOP, M] to Cartesian [x,y,z,vx,vy,vz] in km/km/s."""
    import gmatpy as gmat
    rv6 = gmat.StateConversionUtil.Convert(
        oe_gmat, "Keplerian", "Cartesian",
        _MU_EARTH_GMAT, _GMAT_FLAT, _GMAT_R_EQ, "MA",
    )
    return [rv6.GetElement(i) for i in range(6)]


def _build_propagator(sc, step_size_sec: float, configure_forces_fn=None):
    """Build a PropSetup with RK4, ready to step.

    The ODE model inside PropSetup already contains an Earth PointMassForce;
    we just need to attach the solar system before PrepareInternals.

    If ``configure_forces_fn`` is provided it is called with the PropSetup
    object *after* the solar system is attached but *before* AddPropObject.
    Callers that add non-default forces (e.g. GravityField) must call
    ``gmat.Initialize()`` inside ``configure_forces_fn`` — the API requires
    this to wire up coordinate-system references before PrepareInternals.
    """
    import gmatpy as gmat
    ps = gmat.Construct("PropSetup", "PS")
    fm = ps.GetODEModel()
    fm.SetSolarSystem(gmat.GetSolarSystem())
    if configure_forces_fn is not None:
        configure_forces_fn(ps)
    ps.AddPropObject(sc)
    ps.SetField("Type", "RungeKutta4")
    ps.PrepareInternals()
    inner = ps.GetPropagator()
    inner.SetStepSize(step_size_sec)
    inner.Initialize()
    return inner


def _propagate(r0_km: list[float], epoch_utcmjd: float,
               step_size_sec: float, n_steps: int,
               sc_props: tuple[float, float, float, float, float] | None = None,
               configure_forces_fn=None) -> list[tuple]:
    """One full simulation. Returns list of (r_km, v_kms) tuples.

    ``sc_props`` is an optional (mass_kg, drag_area_m2, Cd, srp_area_m2, Cr)
    tuple; when omitted the T13 defaults are used. GMAT expects mass in kg and
    areas in m^2 — the same units brahe/Orekit use.

    ``configure_forces_fn(propsetup)`` may add extra forces after the solar
    system is attached but before PrepareInternals.
    """
    import gmatpy as gmat
    if sc_props is None:
        mass, drag_area, cd, srp_area, cr = 100.0, 1.0, 2.2, 1.0, 1.5
    else:
        mass, drag_area, cd, srp_area, cr = sc_props

    sc = gmat.Construct("Spacecraft", "Sat")
    sc.SetField("DateFormat", "UTCModJulian")
    sc.SetField("Epoch", str(epoch_utcmjd))
    sc.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sc.SetField("StateType", "Cartesian")
    sc.SetField("X", r0_km[0]); sc.SetField("Y", r0_km[1]); sc.SetField("Z", r0_km[2])
    sc.SetField("VX", r0_km[3]); sc.SetField("VY", r0_km[4]); sc.SetField("VZ", r0_km[5])
    sc.SetField("DryMass", mass)
    sc.SetField("Cd", cd); sc.SetField("DragArea", drag_area)
    sc.SetField("Cr", cr); sc.SetField("SRPArea", srp_area)

    inner = _build_propagator(sc, step_size_sec, configure_forces_fn)

    samples = []
    for _ in range(n_steps):
        inner.Step(step_size_sec)
        state = inner.GetState()
        samples.append((state[:3], state[3:6]))
    return samples


def _samples_to_gcrf_meters(samples_km) -> list[list[float]]:
    """Convert each (r_km, v_kms) tuple to a 6-vector GCRF in m / m/s."""
    out = []
    for r_km, v_kms in samples_km:
        r_m = [x * 1000.0 for x in r_km]
        v_mps = [x * 1000.0 for x in v_kms]
        out.append(mj2000_to_gcrf(r_m, v_mps))
    return out


def _add_gravity_field(propsetup, degree: int, order: int,
                       potential_file: str = "JGM2.cof"):
    """Replace the default PointMassForce with a spherical-harmonic GravityField.

    Called as a ``configure_forces_fn`` argument to ``_build_propagator`` /
    ``_propagate``. GMAT's ODEModel rejects ``AddForce(GravityField)`` when a
    ``PointMassForce`` for the same body is already present; we must remove it
    first via ``DeleteForce(ptr)``.

    Also calls ``gmat.Initialize()`` after adding the GravityField; this is
    required by the GMAT API to wire coordinate-system references (EarthFixed,
    EarthMJ2000Eq) into the force model before PrepareInternals is called.
    Skipping it produces NaN derivatives on the first Step().

    The default ``potential_file`` is a bare filename; GMAT's FileManager
    resolves it against ``<GMAT_ROOT>/data/gravity/earth/``.
    """
    import gmatpy as gmat
    ode = propsetup.GetODEModel()

    # Remove any existing PointMassForce for Earth (the PropSetup default).
    for i in range(ode.GetNumForces()):
        f = ode.GetForce(i)
        if f.GetTypeName() == "PointMassForce" and f.GetBodyName() == "Earth":
            ode.DeleteForce(f)
            break

    grav = gmat.Construct("GravityField", "EarthGravity")
    grav.SetField("BodyName", "Earth")
    grav.SetField("PotentialFile", potential_file)
    grav.SetField("Degree", degree)
    grav.SetField("Order", order)
    ode.AddForce(grav)

    # Wire coordinate system references required by GravityField.
    gmat.Initialize()


def numerical_twobody(params: dict, iterations: int):
    """RK4 numerical propagation under two-body dynamics.

    Single-IC (perf) and multi-IC (accuracy) shapes — see keplerian_trajectory.
    """
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])
    cases = params.get("cases")

    def _prepared(case_jd: float, case_oe_deg_si: list) -> tuple:
        epoch_utcmjd = float(case_jd) - _JD_TO_UTCMJD
        r0_km = _keplerian_to_cartesian_km(_oe_deg_si_to_gmat(list(case_oe_deg_si)))
        return epoch_utcmjd, r0_km

    if cases is None:
        epoch_utcmjd, r0_km = _prepared(params["jd"], params["elements"])

        def run():
            gmat_clear()
            return _propagate(r0_km, epoch_utcmjd, step_size, n_steps)
    else:
        prepared_cases = [_prepared(c["jd"], c["elements"]) for c in cases]

        def run():
            finals = []
            for epoch_utcmjd, r0_km in prepared_cases:
                gmat_clear()
                traj = _propagate(r0_km, epoch_utcmjd, step_size, n_steps)
                finals.append(traj[-1])
            return finals

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.numerical_twobody",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RungeKutta4",
            "step_size": step_size,
            "gravity": "point_mass",
        },
    )


def _add_third_body_force(propsetup, body_name: str):
    """Add a PointMassForce for a third body to the PropSetup's ODE model.

    GMAT body names: 'Sun', 'Luna' (not 'Moon'), 'Mars', etc.
    The default ODE already contains an Earth PointMassForce; this adds
    additional ones for non-Earth bodies without removing the existing one.
    """
    import gmatpy as gmat
    ode = propsetup.GetODEModel()
    tb = gmat.Construct("PointMassForce", f"{body_name}TB")
    tb.SetField("BodyName", body_name)
    ode.AddForce(tb)
    return tb


def _build_rk4_run(
    params: dict,
    configure_forces_fn,
) -> "callable":
    """Build the per-call ``run()`` closure shared by the RK4 force-model tasks.

    Detects single-IC (perf) vs multi-IC (accuracy) ``params`` shape. In
    accuracy mode, each case is propagated separately and only the final
    state is returned per case, so the harness gets one IC sample per
    case rather than a correlated trajectory.
    """
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])
    param_vec = params.get("params", [1000.0, 10.0, 2.2, 10.0, 1.3])
    sc_props = (
        float(param_vec[0]),
        float(param_vec[1]),
        float(param_vec[2]),
        float(param_vec[3]),
        float(param_vec[4]),
    )
    cases = params.get("cases")

    def _prepared(case_jd: float, case_oe_deg_si: list) -> tuple:
        epoch_utcmjd = float(case_jd) - _JD_TO_UTCMJD
        r0_km = _keplerian_to_cartesian_km(_oe_deg_si_to_gmat(list(case_oe_deg_si)))
        return epoch_utcmjd, r0_km

    if cases is None:
        epoch_utcmjd, r0_km = _prepared(params["jd"], params["elements_deg"])

        def run():
            gmat_clear()
            return _propagate(
                r0_km, epoch_utcmjd, step_size, n_steps,
                sc_props=sc_props,
                configure_forces_fn=configure_forces_fn,
            )
    else:
        prepared_cases = [_prepared(c["jd"], c["elements"]) for c in cases]

        def run():
            finals = []
            for epoch_utcmjd, r0_km in prepared_cases:
                gmat_clear()
                traj = _propagate(
                    r0_km, epoch_utcmjd, step_size, n_steps,
                    sc_props=sc_props,
                    configure_forces_fn=configure_forces_fn,
                )
                finals.append(traj[-1])
            return finals

    return run, step_size


def numerical_rk4_grav5x5(params: dict, iterations: int):
    """RK4 + JGM-2 5x5 spherical-harmonic gravity. Single-IC (perf) or
    multi-IC (accuracy) — see ``_build_rk4_run``."""

    def configure_forces(ps):
        _add_gravity_field(ps, 5, 5, "JGM2.cof")

    run, step_size = _build_rk4_run(params, configure_forces)

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.numerical_rk4_grav5x5",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RungeKutta4",
            "step_size": step_size,
            "gravity": "5x5",
            "gravity_source": "JGM2",
        },
    )


def _add_drag_force(propsetup, atmosphere: str = "MSISE90"):
    """Add an MSISE-90 atmospheric drag force to PropSetup's ODE model.

    GMAT R2026a requires two steps to configure drag:
      1. Construct a DragForce and set its AtmosphereModel field.
      2. Construct the atmosphere model object and link it via SetReference.

    The atmosphere type name passed to gmat.Construct must match the GMAT
    factory name: 'MSISE90' (not 'Msise90Atmosphere').

    The spacecraft drag properties (Cd, DragArea) are set on the Spacecraft
    object by ``_propagate``; this helper only constructs the force objects.
    """
    import gmatpy as gmat
    ode = propsetup.GetODEModel()
    drag = gmat.Construct("DragForce", "Drag")
    drag.SetField("AtmosphereModel", atmosphere)
    atmos = gmat.Construct(atmosphere, "Atmos")
    drag.SetReference(atmos)
    ode.AddForce(drag)
    return drag


def _add_srp_force(propsetup):
    """Add a Solar Radiation Pressure force to PropSetup's ODE model.

    The spacecraft SRP properties (Cr, SRPArea) are set on the Spacecraft
    object by ``_propagate``; this helper only constructs the force object.
    """
    import gmatpy as gmat
    ode = propsetup.GetODEModel()
    srp = gmat.Construct("SolarRadiationPressure", "SRP")
    ode.AddForce(srp)
    return srp


def numerical_rk4_grav20x20_sun_moon(params: dict, iterations: int):
    """RK4 + 20x20 gravity + Sun/Moon third bodies. Single-IC (perf) or
    multi-IC (accuracy) — see ``_build_rk4_run``."""

    def configure_forces(ps):
        _add_gravity_field(ps, 20, 20, "JGM2.cof")
        _add_third_body_force(ps, "Sun")
        _add_third_body_force(ps, "Luna")
        import gmatpy as gmat
        gmat.Initialize()

    run, step_size = _build_rk4_run(params, configure_forces)

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.numerical_rk4_grav20x20_sun_moon",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RungeKutta4",
            "step_size": step_size,
            "gravity": "20x20",
            "gravity_source": "JGM2",
            "third_body": ["Sun", "Luna"],
        },
    )


def _propagate_full_forces(r0_km: list[float], epoch_utcmjd: float,
                           step_size_sec: float, n_steps: int,
                           sc_props: tuple, suffix: str) -> list[tuple]:
    """One full simulation with 80x80 gravity + Sun/Moon + drag + SRP.

    Constructs all GMAT objects with unique ``suffix``-appended names to avoid
    name collisions across iterations WITHOUT calling gmat.Clear() between them.
    gmat.Clear() crashes GMAT R2026a when a DragForce has been linked to an
    atmosphere model via SetReference (GMAT bug, similar to GMT-6950). The
    workaround is to avoid Clear() and instead use unique per-iteration names
    so each iteration's Construct() calls create new objects.

    Returns list of (r_km, v_kms) tuples.
    """
    import gmatpy as gmat

    mass, drag_area, cd, srp_area, cr = sc_props

    sc = gmat.Construct("Spacecraft", f"Sat{suffix}")
    sc.SetField("DateFormat", "UTCModJulian")
    sc.SetField("Epoch", str(epoch_utcmjd))
    sc.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sc.SetField("StateType", "Cartesian")
    sc.SetField("X", r0_km[0]); sc.SetField("Y", r0_km[1]); sc.SetField("Z", r0_km[2])
    sc.SetField("VX", r0_km[3]); sc.SetField("VY", r0_km[4]); sc.SetField("VZ", r0_km[5])
    sc.SetField("DryMass", mass)
    sc.SetField("Cd", cd); sc.SetField("DragArea", drag_area)
    sc.SetField("Cr", cr); sc.SetField("SRPArea", srp_area)

    ps = gmat.Construct("PropSetup", f"PS{suffix}")
    fm = ps.GetODEModel()
    fm.SetSolarSystem(gmat.GetSolarSystem())

    # Remove the default Earth PointMassForce before adding GravityField.
    for i in range(fm.GetNumForces()):
        f = fm.GetForce(i)
        if f.GetTypeName() == "PointMassForce" and f.GetBodyName() == "Earth":
            fm.DeleteForce(f)
            break

    # 80x80 EGM96 spherical-harmonic gravity.
    grav = gmat.Construct("GravityField", f"EarthGravity{suffix}")
    grav.SetField("BodyName", "Earth")
    grav.SetField("PotentialFile", "EGM96.cof")
    grav.SetField("Degree", 80)
    grav.SetField("Order", 80)
    fm.AddForce(grav)
    gmat.Initialize()  # required to wire coordinate references for GravityField

    # Sun and Moon third-body point masses.
    sun = gmat.Construct("PointMassForce", f"SunTB{suffix}")
    sun.SetField("BodyName", "Sun")
    fm.AddForce(sun)
    luna = gmat.Construct("PointMassForce", f"LunaTB{suffix}")
    luna.SetField("BodyName", "Luna")
    fm.AddForce(luna)

    # MSISE-90 atmospheric drag.  SetReference links the atmosphere model
    # object to the DragForce.  NOTE: gmat.Clear() after SetReference causes
    # a GMAT crash (pure virtual / segfault); this is why _propagate_full_forces
    # avoids Clear() and uses unique names instead.
    drag = gmat.Construct("DragForce", f"Drag{suffix}")
    drag.SetField("AtmosphereModel", "MSISE90")
    atmos = gmat.Construct("MSISE90", f"Atmos{suffix}")
    drag.SetReference(atmos)
    fm.AddForce(drag)

    # Solar Radiation Pressure.
    srp = gmat.Construct("SolarRadiationPressure", f"SRP{suffix}")
    fm.AddForce(srp)

    gmat.Initialize()

    ps.AddPropObject(sc)
    ps.SetField("Type", "RungeKutta4")
    ps.PrepareInternals()
    inner = ps.GetPropagator()
    inner.SetStepSize(step_size_sec)
    inner.Initialize()

    samples = []
    for _ in range(n_steps):
        inner.Step(step_size_sec)
        state = inner.GetState()
        samples.append((state[:3], state[3:6]))
    return samples


def numerical_rk4_grav80x80_full(params: dict, iterations: int):
    """RK4 numerical propagation: 80x80 EGM96 + Sun/Moon + MSISE90 drag + SRP.

    Parameters follow the force-model task layout generated by
    ``NumericalRk4Grav80x80FullTask.generate_params``:
      - jd            : UTC Julian date of epoch
      - elements_deg  : [a (m), e, i_deg, RAAN_deg, AOP_deg, M_deg]
      - step_size     : RK4 step size in seconds
      - n_steps       : number of integration steps
      - params        : [mass (kg), drag_area (m^2), Cd, srp_area (m^2), Cr]

    Forces active: 80×80 EGM96 gravity, Sun/Moon third-body, MSISE-90 drag,
    and solar radiation pressure. EGM96.cof is used because JGM2.cof only
    supports up to degree 70.

    NOTE: gmat.Clear() crashes GMAT R2026a after DragForce+Msise90Atmosphere
    SetReference setup (GMAT bug similar to GMT-6950). This function avoids
    Clear() by using unique per-iteration object names, so all objects from all
    iterations accumulate in GMAT's registry. This is safe for benchmark
    workloads (bounded iteration count) but will leak memory if run for very
    many iterations.
    """
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])

    param_vec = params.get("params", [1000.0, 10.0, 2.2, 10.0, 1.3])
    sc_props = (
        float(param_vec[0]),
        float(param_vec[1]),
        float(param_vec[2]),
        float(param_vec[3]),
        float(param_vec[4]),
    )

    cases = params.get("cases")

    def _prepared(case_jd: float, case_oe_deg_si: list) -> tuple:
        epoch_utcmjd = float(case_jd) - _JD_TO_UTCMJD
        r0_km = _keplerian_to_cartesian_km(_oe_deg_si_to_gmat(list(case_oe_deg_si)))
        return epoch_utcmjd, r0_km

    # Mutable counter so every _propagate_full_forces call uses unique
    # GMAT object names — required to work around the DragForce/Clear bug.
    counter = [0]

    if cases is None:
        epoch_utcmjd, r0_km = _prepared(params["jd"], params["elements_deg"])

        def run():
            counter[0] += 1
            return _propagate_full_forces(
                r0_km, epoch_utcmjd, step_size, n_steps, sc_props,
                suffix=str(counter[0]),
            )
    else:
        prepared_cases = [_prepared(c["jd"], c["elements"]) for c in cases]

        def run():
            finals = []
            for epoch_utcmjd, r0_km in prepared_cases:
                counter[0] += 1
                traj = _propagate_full_forces(
                    r0_km, epoch_utcmjd, step_size, n_steps, sc_props,
                    suffix=str(counter[0]),
                )
                finals.append(traj[-1])
            return finals

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.numerical_rk4_grav80x80_full",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RungeKutta4",
            "step_size": step_size,
            "gravity": "80x80",
            "gravity_source": "EGM96",
            "third_body": ["Sun", "Luna"],
            "drag": "MSISE90",
            "srp": True,
        },
    )


_KEPLERIAN_INTERNAL_STEP_SEC = 60.0
"""Internal step size used by GMAT's Keplerian propagator for ``keplerian_single``.

GMAT R2026a's Type='Keplerian' PropSetup has a step-size limitation: Step()
calls with dt larger than approximately one orbital period (~5800 s for LEO)
saturate to a capped state and produce errors up to tens of kilometres.
Step sizes <= the orbital period are insensitive to the exact value used (all
produce the same result to sub-nanometre precision). Using 60 s internal steps
matches the trajectory-task step size and gives sub-metre agreement with
brahe/OreKit analytical propagation.
"""


def _propagate_keplerian_case(r0_km: list[float], epoch_utcmjd: float,
                               dt_sec: float,
                               internal_step_sec: float = _KEPLERIAN_INTERNAL_STEP_SEC,
                               ) -> tuple:
    """Propagate one orbit analytically to dt_sec using GMAT's Keplerian propagator.

    Returns a single (r_km, v_kms) tuple.

    Uses small internal steps (``internal_step_sec``, default 60 s) because
    GMAT's Keplerian propagator saturates for single steps larger than roughly
    one orbital period. The propagator is otherwise insensitive to step size,
    so all dt >= internal_step_sec produce the same result to sub-nanometre
    precision. The ODE model still requires SetSolarSystem() even for the
    analytical propagator — GMAT enforces this in PrepareInternals().
    """
    import gmatpy as gmat
    sc = gmat.Construct("Spacecraft", "Sat")
    sc.SetField("DateFormat", "UTCModJulian")
    sc.SetField("Epoch", str(epoch_utcmjd))
    sc.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sc.SetField("StateType", "Cartesian")
    sc.SetField("X", r0_km[0]); sc.SetField("Y", r0_km[1]); sc.SetField("Z", r0_km[2])
    sc.SetField("VX", r0_km[3]); sc.SetField("VY", r0_km[4]); sc.SetField("VZ", r0_km[5])
    sc.SetField("DryMass", 100.0)

    ps = gmat.Construct("PropSetup", "PS")
    fm = ps.GetODEModel()
    fm.SetSolarSystem(gmat.GetSolarSystem())
    ps.SetField("Type", "Keplerian")
    ps.AddPropObject(sc)
    ps.PrepareInternals()
    inner = ps.GetPropagator()
    inner.SetStepSize(internal_step_sec)
    inner.Initialize()

    n_full = int(dt_sec / internal_step_sec)
    remainder = dt_sec - n_full * internal_step_sec
    for _ in range(n_full):
        inner.Step(internal_step_sec)
    if remainder > 1e-6:
        inner.Step(remainder)

    state = inner.GetState()
    return (state[:3], state[3:6])


def keplerian_single(params: dict, iterations: int):
    """Analytical Keplerian propagation — 20 orbits each to a single future epoch.

    Parameters (from KeplerianSingleTask.generate_params):
      - cases: list of 20 dicts, each containing:
          - jd        : UTC Julian date of epoch
          - elements  : [a (m), e, i_deg, RAAN_deg, AOP_deg, M_deg]
          - dt        : propagation duration in seconds

    GMAT's Type='Keplerian' PropSetup provides analytical two-body propagation.
    Each case is propagated using 60 s internal steps to dt (the GMAT Keplerian
    propagator saturates for single steps larger than one orbital period).
    """
    cases = params["cases"]

    # Pre-convert all Keplerian ICs to Cartesian outside timed region.
    case_data = []
    for case in cases:
        jd_utc = float(case["jd"])
        epoch_utcmjd = jd_utc - _JD_TO_UTCMJD
        oe_gmat = _oe_deg_si_to_gmat(list(case["elements"]))
        r0_km = _keplerian_to_cartesian_km(oe_gmat)
        dt = float(case["dt"])
        case_data.append((r0_km, epoch_utcmjd, dt))

    def run():
        results = []
        for r0_km, epoch_utcmjd, dt in case_data:
            gmat_clear()
            sample = _propagate_keplerian_case(r0_km, epoch_utcmjd, dt)
            results.append(sample)
        return results

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.keplerian_single",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "propagator": "Keplerian",
            "gravity": "two_body_analytical",
        },
    )


def _propagate_keplerian_trajectory(r0_km: list[float], epoch_utcmjd: float,
                                     step_size_sec: float, n_steps: int) -> list[tuple]:
    """Propagate one orbit analytically over n_steps using GMAT's Keplerian propagator.

    Returns list of (r_km, v_kms) tuples for each step.
    """
    import gmatpy as gmat
    sc = gmat.Construct("Spacecraft", "Sat")
    sc.SetField("DateFormat", "UTCModJulian")
    sc.SetField("Epoch", str(epoch_utcmjd))
    sc.SetField("CoordinateSystem", "EarthMJ2000Eq")
    sc.SetField("StateType", "Cartesian")
    sc.SetField("X", r0_km[0]); sc.SetField("Y", r0_km[1]); sc.SetField("Z", r0_km[2])
    sc.SetField("VX", r0_km[3]); sc.SetField("VY", r0_km[4]); sc.SetField("VZ", r0_km[5])
    sc.SetField("DryMass", 100.0)

    ps = gmat.Construct("PropSetup", "PS")
    fm = ps.GetODEModel()
    fm.SetSolarSystem(gmat.GetSolarSystem())
    ps.SetField("Type", "Keplerian")
    ps.AddPropObject(sc)
    ps.PrepareInternals()
    inner = ps.GetPropagator()
    inner.SetStepSize(step_size_sec)
    inner.Initialize()

    samples = []
    for _ in range(n_steps):
        inner.Step(step_size_sec)
        state = inner.GetState()
        samples.append((state[:3], state[3:6]))
    return samples


def keplerian_trajectory(params: dict, iterations: int):
    """Analytical Keplerian propagation — single trajectory (perf) or
    per-IC final state (accuracy).

    Single-IC perf shape: ``{jd, elements, step_size, n_steps}``.
    Multi-IC accuracy shape: ``{cases: [{jd, elements}], step_size, n_steps}``.
    """
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])
    cases = params.get("cases")

    def _prepared(case_jd: float, case_oe_deg_si: list) -> tuple:
        epoch_utcmjd = float(case_jd) - _JD_TO_UTCMJD
        oe_gmat = _oe_deg_si_to_gmat(list(case_oe_deg_si))
        r0_km = _keplerian_to_cartesian_km(oe_gmat)
        return epoch_utcmjd, r0_km

    if cases is None:
        epoch_utcmjd, r0_km = _prepared(params["jd"], params["elements"])

        def run():
            gmat_clear()
            return _propagate_keplerian_trajectory(
                r0_km, epoch_utcmjd, step_size, n_steps
            )
    else:
        # Pre-convert all cases outside the timed region.
        prepared_cases = [_prepared(c["jd"], c["elements"]) for c in cases]

        def run():
            finals = []
            for epoch_utcmjd, r0_km in prepared_cases:
                gmat_clear()
                traj = _propagate_keplerian_trajectory(
                    r0_km, epoch_utcmjd, step_size, n_steps
                )
                finals.append(traj[-1])
            return finals

    times, native_samples = time_iterations(run, iterations)
    results = _samples_to_gcrf_meters(native_samples)
    return build_task_result(
        "propagation.keplerian_trajectory",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "propagator": "Keplerian",
            "step_size": step_size,
            "gravity": "two_body_analytical",
        },
    )


# ---------------------------------------------------------------------------
# SGP4 propagation via GMAT's SPICESGP4 propagator
# ---------------------------------------------------------------------------

# Large initial step size that exceeds any single-Propagate duration for
# sgp4_single. With SPICESGP4 (an analytical SGP4 propagator), a step size
# larger than the requested ElapsedSecs forces exactly one SGP4 evaluation
# per Propagate command, so the ReportFile emits exactly one line per step
# (plus one line for the initial state at epoch which we discard).
_SGP4_SINGLE_STEP_SIZE_SEC = 200_000.0  # > 2 × 86400 s

_GMAT_SGP4_SCRIPT_TEMPLATE = """\
Create Spacecraft Sat

Sat.EphemerisName = '{tle_file}'
Sat.Id = 'ISS'

Create Propagator TLEProp

TLEProp.Type            = SPICESGP4
TLEProp.InitialStepSize = {initial_step_size}

Create CoordinateSystem TEME_CS
TEME_CS.Origin = Earth
TEME_CS.Axes   = TEME

Create ReportFile RF

RF.Filename     = '{report_file}'
RF.Precision    = 16
RF.Add          = {{ Sat.TEME_CS.X, Sat.TEME_CS.Y, Sat.TEME_CS.Z, Sat.TEME_CS.VX, Sat.TEME_CS.VY, Sat.TEME_CS.VZ}}
RF.WriteHeaders = False

BeginMissionSequence

{propagate_lines}
"""


def _write_tle_file(tle_file: str, line1: str, line2: str) -> None:
    """Write a 3LE file for GMAT's SPICESGP4 propagator.

    GMAT matches the spacecraft to a TLE by comparing Spacecraft.Id to the
    TLE name line (line 0). The NORAD catalog number from line 1 (characters
    3-7) is used as the fallback. Using 'ISS' as the name line works
    reliably when Spacecraft.Id = 'ISS'.
    """
    with open(tle_file, "w") as f:
        f.write("ISS\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n")


def _parse_report_teme_km(report_file: str, n_states: int) -> list[list[float]]:
    """Parse a GMAT ReportFile and return the first n_states states.

    GMAT writes the initial state (at epoch) as line 0 and then one state
    per Propagate command (or per step for multi-step Propagates). We skip
    line 0 (initial epoch state) and take lines 1..n_states.

    Returns a list of 6-vectors [x,y,z,vx,vy,vz] in km / km/s (TEME frame).
    """
    results = []
    with open(report_file) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i == 0:
                continue  # skip initial state at epoch
            vals = line.split()
            state = [float(v) for v in vals[:6]]
            results.append(state)
            if len(results) == n_states:
                break
    return results


def _run_sgp4_script(
    script_file: str,
    report_file: str,
    line1: str,
    line2: str,
    tle_file: str,
    initial_step_size: float,
    propagate_lines: str,
) -> None:
    """Write the GMAT script and execute it via LoadScript / RunScript."""
    import gmatpy as gmat

    script = _GMAT_SGP4_SCRIPT_TEMPLATE.format(
        tle_file=tle_file,
        initial_step_size=initial_step_size,
        report_file=report_file,
        propagate_lines=propagate_lines,
    )
    with open(script_file, "w") as f:
        f.write(script)

    if os.path.exists(report_file):
        os.remove(report_file)

    gmat_clear()
    gmat.LoadScript(script_file)
    gmat.RunScript()


def sgp4_single(params: dict, iterations: int):
    """SGP4 propagation to 50 non-uniform future epochs.

    Parameters (from Sgp4SingleTask.generate_params):
      - line1                 : TLE line 1
      - line2                 : TLE line 2
      - time_offsets_seconds  : sorted list of 50 time offsets from TLE epoch (seconds)

    Uses GMAT's SPICESGP4 propagator (TLEPropagator plugin). The TLE is written
    to a temp file in TMPDIR; each iteration regenerates the GMAT script, clears
    GMAT state, and runs it. InitialStepSize is set to 200,000 s (> max offset),
    forcing exactly one SGP4 call per Propagate command. Output is in TEME frame
    and units of km / km/s; results are converted to m / m/s before return.
    """
    line1 = params["line1"]
    line2 = params["line2"]
    offsets = sorted(params["time_offsets_seconds"])  # already sorted per task spec

    # Compute cumulative deltas so each Propagate advances from the prior stop.
    deltas = []
    prev = 0.0
    for t in offsets:
        deltas.append(t - prev)
        prev = t

    propagate_lines = "\n".join(
        "Propagate TLEProp(Sat) {Sat.ElapsedSecs = %.10f}" % dt
        for dt in deltas
    )
    n_states = len(offsets)

    tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
    tle_file = os.path.join(tmpdir, "gmat_sgp4_single.tle")
    script_file = os.path.join(tmpdir, "gmat_sgp4_single.script")
    report_file = os.path.join(tmpdir, "gmat_sgp4_single.txt")

    _write_tle_file(tle_file, line1, line2)

    def run():
        _run_sgp4_script(
            script_file, report_file, line1, line2, tle_file,
            _SGP4_SINGLE_STEP_SIZE_SEC, propagate_lines,
        )
        return _parse_report_teme_km(report_file, n_states)

    times, native_samples = time_iterations(run, iterations)

    # Convert km / km/s -> m / m/s
    results = [
        [x * 1000.0 for x in state]
        for state in native_samples
    ]

    return build_task_result(
        "propagation.sgp4_single",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "TEME",
            "propagator": "SPICESGP4",
        },
    )


def sgp4_trajectory(params: dict, iterations: int):
    """SGP4 propagation over 1 day at fixed step intervals.

    Parameters (from Sgp4TrajectoryTask.generate_params):
      - line1     : TLE line 1
      - line2     : TLE line 2
      - step_size : step size in seconds (60.0)
      - n_steps   : number of steps (1440)

    Uses GMAT's SPICESGP4 propagator. A single Propagate command runs for
    n_steps * step_size seconds with InitialStepSize = step_size, causing
    GMAT to log one state per 60 s step. The ReportFile produces n_steps + 2
    lines (initial state + n_steps states + one duplicate final); we skip
    line 0 and take lines 1..n_steps. Output is in TEME frame, converted to
    m / m/s.
    """
    line1 = params["line1"]
    line2 = params["line2"]
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])

    elapsed_days = n_steps * step_size / 86400.0
    propagate_lines = "Propagate TLEProp(Sat) {Sat.ElapsedDays = %.15f}" % elapsed_days

    tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
    tle_file = os.path.join(tmpdir, "gmat_sgp4_trajectory.tle")
    script_file = os.path.join(tmpdir, "gmat_sgp4_trajectory.script")
    report_file = os.path.join(tmpdir, "gmat_sgp4_trajectory.txt")

    _write_tle_file(tle_file, line1, line2)

    def run():
        _run_sgp4_script(
            script_file, report_file, line1, line2, tle_file,
            step_size, propagate_lines,
        )
        return _parse_report_teme_km(report_file, n_steps)

    times, native_samples = time_iterations(run, iterations)

    # Convert km / km/s -> m / m/s
    results = [
        [x * 1000.0 for x in state]
        for state in native_samples
    ]

    return build_task_result(
        "propagation.sgp4_trajectory",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "TEME",
            "propagator": "SPICESGP4",
            "step_size": step_size,
            "n_steps": n_steps,
        },
    )
