"""Basilisk numerical propagation benchmarks.

Pattern: each iteration builds a fresh SimBaseClass, configures gravity /
third-body / drag / SRP as needed, integrates with RK4 at the requested
step size, then transforms the recorded J2000 inertial trajectory to GCRF
for accuracy comparison with brahe/Orekit. Sim setup cost is included in
the timed region (matches Java/Rust methodology — they time the full run).

The output trajectory drops the IC sample so it matches the existing
brahe/Orekit trajectories which record post-step states for k=1..n_steps.
"""

import math

import numpy as np

from Basilisk.simulation import spacecraft
from Basilisk.utilities import (
    SimulationBaseClass,
    macros,
    orbitalMotion,
    simIncludeGravBody,
)
from Basilisk.utilities.supportDataTools.dataFetcher import DataFile, get_path

from benchmarks.comparative.implementations.basilisk.base import (
    build_task_result,
    j2000_to_gcrf,
    jd_to_spice_et_string,
    time_iterations,
)

# Match brahe and Orekit's Earth mu (3.986004418e14 m^3/s^2). Basilisk's
# default Earth.mu is 3.986004360e14 — a ~5.8e7 m^3/s^2 difference that
# produces ~24 m position drift over one LEO orbit in pure two-body. We
# override Basilisk's mu after createEarth() so the dynamics see the same
# central-body parameter all three baselines do.
_MU_EARTH_BENCHMARK = 3.986004418e14


def _classic_elements_deg_to_basilisk(oe_deg):
    """[a, e, i_deg, raan_deg, argp_deg, M_deg] -> orbitalMotion.ClassicElements."""
    a, e, i_deg, raan_deg, argp_deg, M_deg = oe_deg
    M = math.radians(M_deg)
    E = orbitalMotion.M2E(M, e)
    f = orbitalMotion.E2f(E, e)
    oe = orbitalMotion.ClassicElements()
    oe.a = a
    oe.e = e
    oe.i = math.radians(i_deg)
    oe.Omega = math.radians(raan_deg)
    oe.omega = math.radians(argp_deg)
    oe.f = f
    return oe


def _record_states_gcrf(dataRec, n_steps):
    """Pull recorded r_BN_N / v_BN_N samples (indices 1..n_steps inclusive)
    and return them in GCRF. Index 0 is the IC, dropped to match brahe/Orekit.
    """
    states = []
    for k in range(1, n_steps + 1):
        r = dataRec.r_BN_N[k]
        v = dataRec.v_BN_N[k]
        states.append(j2000_to_gcrf(r, v))
    return states


def _attach_spice_earth(scSim, gf, jd):
    """Wire SPICE so the spacecraft sees a time-varying Earth orientation.

    Without this, Basilisk's spherical-harmonic gravity is computed against
    a fixed Earth orientation, which produces secular drift over an orbit.
    With it, the IC and integration both see Earth's rotation derived from
    the PCK kernel — comparable to brahe/Orekit's body-fixed gravity.
    """
    spiceObject = gf.createSpiceInterface(
        time=jd_to_spice_et_string(jd),
        epochInMsg=True,
    )
    spiceObject.zeroBase = "Earth"
    scSim.AddModelToTask("benchTask", spiceObject)
    return spiceObject


def _gravity_file_for_degree(degree: int) -> str:
    """Pick the GGM gravity-coefficient file matching the requested degree.

    The Basilisk-bundled GGM03S provides coefficients up to degree 180, which
    covers all our benchmark cases (5x5, 20x20, 80x80).
    """
    return str(get_path(DataFile.LocalGravData.GGM03S))


def _run_numerical_rk4(
    task_name: str,
    params: dict,
    iterations: int,
    *,
    gravity_degree: int,
    gravity_order: int,
    third_body_sun: bool = False,
    third_body_moon: bool = False,
    drag: bool = False,
    srp: bool = False,
):
    """Shared driver for the three RK4 force-model propagation tasks.

    Reads params from the standard force-model task layout:
      - jd, elements_deg, step_size, n_steps
      - params: [mass, drag_area, Cd, srp_area, Cr]
    """
    jd = params["jd"]
    oe_deg = params["elements_deg"]
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])
    param_vec = params["params"]
    mass = float(param_vec[0])
    drag_area = float(param_vec[1])
    cd = float(param_vec[2])
    srp_area = float(param_vec[3])
    cr = float(param_vec[4])

    def run():
        scSim = SimulationBaseClass.SimBaseClass()
        dyn = scSim.CreateNewProcess("benchProc")
        dt_ns = macros.sec2nano(step_size)
        dyn.addTask(scSim.CreateNewTask("benchTask", dt_ns))

        sc = spacecraft.Spacecraft()
        sc.ModelTag = "bench"
        scSim.AddModelToTask("benchTask", sc)

        gf = simIncludeGravBody.gravBodyFactory()
        earth = gf.createEarth()
        earth.isCentralBody = True
        earth.mu = _MU_EARTH_BENCHMARK
        earth.useSphericalHarmonicsGravityModel(
            _gravity_file_for_degree(gravity_degree), gravity_degree
        )

        if third_body_sun:
            gf.createSun()
        if third_body_moon:
            gf.createMoon()

        sc.hub.mHub = mass

        if drag:
            from Basilisk.architecture import messaging
            from Basilisk.simulation import (
                dragDynamicEffector,
                msisAtmosphere,
            )
            atm = msisAtmosphere.MsisAtmosphere()
            atm.ModelTag = "msis"
            atm.addSpacecraftToModel(sc.scStateOutMsg)
            # MSIS requires 23 space-weather input messages (Ap history + F10.7).
            # Use representative quiet-Sun values matching the bsk example. This
            # introduces some divergence from Java's CssiSpaceWeatherData
            # interpolation but is what a Basilisk user would typically use
            # without bespoke SW preparation.
            sw_values = (
                [8] * 21        # ap (24h average + 3h history at -0..-57h)
                + [110, 110]    # f107 (1944h average, 24h average at -24h)
            )
            sw_msgs = []
            for idx, val in enumerate(sw_values):
                payload = messaging.SwDataMsgPayload()
                payload.dataValue = val
                msg = messaging.SwDataMsg().write(payload)
                sw_msgs.append(msg)
                atm.swDataInMsgs[idx].subscribeTo(msg)
            # Keep references on the spacecraft sim so messages aren't GC'd.
            scSim._bench_sw_msgs = sw_msgs
            scSim.AddModelToTask("benchTask", atm)

            drag_eff = dragDynamicEffector.DragDynamicEffector()
            drag_eff.ModelTag = "drag"
            drag_eff.coreParams.projectedArea = drag_area
            drag_eff.coreParams.dragCoeff = cd
            drag_eff.atmoDensInMsg.subscribeTo(atm.envOutMsgs[0])
            sc.addDynamicEffector(drag_eff)
            scSim.AddModelToTask("benchTask", drag_eff)

        # SRP needs the Sun ephemeris message which is published by the SPICE
        # interface (created in _attach_spice_earth below). Construct the
        # effector here but defer the message subscription until after SPICE
        # is wired.
        srp_eff = None
        if srp:
            from Basilisk.simulation import radiationPressure
            srp_eff = radiationPressure.RadiationPressure()
            srp_eff.ModelTag = "srp"
            srp_eff.area = srp_area
            srp_eff.coefficientReflection = cr
            sc.addDynamicEffector(srp_eff)
            scSim.AddModelToTask("benchTask", srp_eff)

        gf.addBodiesTo(sc)

        # SPICE Earth orientation (also provides Sun/Moon ephemerides for
        # third-body and SRP if requested).
        spiceObject = _attach_spice_earth(scSim, gf, jd)

        if srp_eff is not None:
            sun_index = list(gf.gravBodies.keys()).index("sun")
            srp_eff.sunEphmInMsg.subscribeTo(spiceObject.planetStateOutMsgs[sun_index])

        bsk_oe = _classic_elements_deg_to_basilisk(oe_deg)
        r0, v0 = orbitalMotion.elem2rv(_MU_EARTH_BENCHMARK, bsk_oe)
        sc.hub.r_CN_NInit = np.array(r0).reshape(3, 1)
        sc.hub.v_CN_NInit = np.array(v0).reshape(3, 1)

        rec = sc.scStateOutMsg.recorder(dt_ns)
        scSim.AddModelToTask("benchTask", rec)

        scSim.InitializeSimulation()
        scSim.ConfigureStopTime(macros.sec2nano(step_size * n_steps))
        scSim.ExecuteSimulation()
        return _record_states_gcrf(rec, n_steps)

    times, results = time_iterations(run, iterations)
    third = [name for name, on in (("sun", third_body_sun), ("moon", third_body_moon)) if on]
    return build_task_result(
        task_name,
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RK4",
            "step_size": step_size,
            "gravity": f"{gravity_degree}x{gravity_order}",
            "gravity_source": "GGM03S",
            "third_body": third,
            "drag": drag,
            "srp": srp,
        },
    )


def numerical_rk4_grav5x5(params: dict, iterations: int):
    return _run_numerical_rk4(
        "propagation.numerical_rk4_grav5x5",
        params, iterations,
        gravity_degree=5, gravity_order=5,
    )


def numerical_rk4_grav20x20_sun_moon(params: dict, iterations: int):
    return _run_numerical_rk4(
        "propagation.numerical_rk4_grav20x20_sun_moon",
        params, iterations,
        gravity_degree=20, gravity_order=20,
        third_body_sun=True, third_body_moon=True,
    )


def numerical_rk4_grav80x80_full(params: dict, iterations: int):
    return _run_numerical_rk4(
        "propagation.numerical_rk4_grav80x80_full",
        params, iterations,
        gravity_degree=80, gravity_order=80,
        third_body_sun=True, third_body_moon=True,
        drag=True, srp=True,
    )


def numerical_twobody(params: dict, iterations: int):
    """Point-mass Earth gravity over the requested step grid with Basilisk's
    default integrator (fixed-step RK4 at task period).

    Note: the Java/Orekit baseline uses an adaptive Dormand-Prince 8(5,3)
    integrator with 1 m position tolerance for this task; brahe uses DP54
    adaptive. Basilisk's default integrator is fixed-step RK4 — the only
    one in the framework that does not segfault during construction with
    the current bsk wheel. The resulting trajectory shows tens-of-meters
    accumulated truncation error vs. the adaptive baselines over one LEO
    revolution; this is intrinsic to RK4-at-60s, not a Basilisk bug.
    """
    jd = params["jd"]
    oe_deg = params["elements"]
    step_size = float(params["step_size"])
    n_steps = int(params["n_steps"])

    def run():
        scSim = SimulationBaseClass.SimBaseClass()
        dyn = scSim.CreateNewProcess("benchProc")
        dt_ns = macros.sec2nano(step_size)
        dyn.addTask(scSim.CreateNewTask("benchTask", dt_ns))

        sc = spacecraft.Spacecraft()
        sc.ModelTag = "bench"
        scSim.AddModelToTask("benchTask", sc)

        gf = simIncludeGravBody.gravBodyFactory()
        earth = gf.createEarth()
        earth.isCentralBody = True
        earth.mu = _MU_EARTH_BENCHMARK
        # Point-mass: leave spherical harmonics off (default).
        gf.addBodiesTo(sc)

        # IC from Keplerian (deg, mean anomaly) -> Cartesian via Basilisk's
        # own elem2rv so the IC matches what the harmonic-gravity tasks use.
        bsk_oe = _classic_elements_deg_to_basilisk(oe_deg)
        r0, v0 = orbitalMotion.elem2rv(earth.mu, bsk_oe)
        sc.hub.r_CN_NInit = np.array(r0).reshape(3, 1)
        sc.hub.v_CN_NInit = np.array(v0).reshape(3, 1)

        rec = sc.scStateOutMsg.recorder(dt_ns)
        scSim.AddModelToTask("benchTask", rec)

        scSim.InitializeSimulation()
        scSim.ConfigureStopTime(macros.sec2nano(step_size * n_steps))
        scSim.ExecuteSimulation()

        return _record_states_gcrf(rec, n_steps)

    times, results = time_iterations(run, iterations)
    return build_task_result(
        "propagation.numerical_twobody",
        iterations,
        times,
        results,
        extra_metadata={
            "frame": "GCRF",
            "integrator": "RK4",
            "step_size": step_size,
            "gravity": "point_mass",
        },
    )
