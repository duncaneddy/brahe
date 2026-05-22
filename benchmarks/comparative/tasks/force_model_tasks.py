"""
Force-model function-level benchmark tasks.

These tasks evaluate a single acceleration term at a fixed spacecraft state
and epoch. They isolate force-model implementations from integrator behavior,
so any difference between brahe and Orekit is attributable to the force-model
calculation itself (gravity coefficients, third-body ephemeris, frame
transformation) rather than to numerical integration error.

Two input shapes are supported per task, mirroring the propagation tasks:

- Perf (``generate_params``): single fixed state + ``n_samples`` inner-loop
  repetitions to amortize call overhead. Result: one acceleration vector.
- Accuracy (``generate_accuracy_samples``): ``cases: [{jd, state_eci}]`` IC
  sweep over the LEO range used by propagation. Result: one acceleration
  vector per case so the accuracy harness can build a distribution rather
  than a single residual.
"""

import math
import random

from benchmarks.comparative.tasks.base import BenchmarkTask

R_EARTH = 6378137.0
GM_EARTH = 3.986004418e14

# Fixed LEO test state in ECI (GCRF). Same epoch as propagation tasks so
# Earth-orientation, third-body ephemeris, and atmospheric density evaluations
# happen at the same instant on both sides.
_FIXED_EPOCH_JD = 2460310.5  # 2024-01-01T00:00:00 UTC
_FIXED_STATE_ECI = [
    6_525_919.0,
    1_710_416.0,
    2_508_886.0,
    -2_682.6,
    7_209.5,
    -1_953.7,
]
_N_SAMPLES = 100  # repeat acceleration call N times per iteration to amortize overhead


def _random_leo_state_eci(rng: random.Random) -> dict:
    """Generate one random LEO IC and return ``{jd, state_eci}``.

    The altitude/eccentricity range matches ``_random_leo_ic`` in
    propagation_tasks.py (400–1500 km, e ≤ 0.02). Keplerian → Cartesian
    is done in-task so every backend sees the identical Cartesian state
    per case; if each backend converted the same elements itself, sub-mm
    KOE→ECI implementation drift would leak into the force-model
    accuracy residual.
    """
    a = R_EARTH + rng.uniform(400e3, 1500e3)
    e = rng.uniform(0.001, 0.02)
    i = math.radians(rng.uniform(0.0, 180.0))
    raan = math.radians(rng.uniform(0.0, 360.0))
    argp = math.radians(rng.uniform(0.0, 360.0))
    nu = math.radians(rng.uniform(0.0, 360.0))

    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    r_pqw = [r * math.cos(nu), r * math.sin(nu), 0.0]
    v_mag = math.sqrt(GM_EARTH / p)
    v_pqw = [-v_mag * math.sin(nu), v_mag * (e + math.cos(nu)), 0.0]

    cos_raan, sin_raan = math.cos(raan), math.sin(raan)
    cos_argp, sin_argp = math.cos(argp), math.sin(argp)
    cos_i, sin_i = math.cos(i), math.sin(i)

    r11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_i
    r12 = -(cos_raan * sin_argp + sin_raan * cos_argp * cos_i)
    r21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_i
    r22 = -(sin_raan * sin_argp - cos_raan * cos_argp * cos_i)
    r31 = sin_argp * sin_i
    r32 = cos_argp * sin_i

    x = r11 * r_pqw[0] + r12 * r_pqw[1]
    y = r21 * r_pqw[0] + r22 * r_pqw[1]
    z = r31 * r_pqw[0] + r32 * r_pqw[1]
    vx = r11 * v_pqw[0] + r12 * v_pqw[1]
    vy = r21 * v_pqw[0] + r22 * v_pqw[1]
    vz = r31 * v_pqw[0] + r32 * v_pqw[1]

    return {
        "jd": _FIXED_EPOCH_JD + rng.uniform(0.0, 365.0),
        "state_eci": [x, y, z, vx, vy, vz],
    }


def _ic_swept_force_params(seed: int, n: int, base: dict) -> dict:
    """Build accuracy-sweep params: N random LEO ICs share the same
    force-model configuration drawn from ``base``.

    ``n_samples`` is forced to 1 on the accuracy path because each case
    already carries its own IC diversity — re-evaluating the same state
    100× per case would only inflate runtime without adding coverage.
    """
    rng = random.Random(seed)
    cases = [_random_leo_state_eci(rng) for _ in range(n)]
    out = dict(base)
    out.pop("jd", None)
    out.pop("state_eci", None)
    out["cases"] = cases
    out["n_samples"] = 1
    return out


def _altitude_km_from_state(state: list[float]) -> float:
    r = math.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    return (r - R_EARTH) / 1000.0


class _AccelTaskBase(BenchmarkTask):
    """Shared boilerplate for function-level acceleration tasks."""

    @property
    def module(self) -> str:
        return "force_model"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "gmat"]

    def _base_params(self) -> dict:
        return {
            "jd": _FIXED_EPOCH_JD,
            "state_eci": list(_FIXED_STATE_ECI),
            "n_samples": _N_SAMPLES,
        }

    def _accuracy_base(self) -> dict:
        """Per-task force-model configuration carried forward into every
        case dict. Subclasses override to attach gravity degree/order, etc.
        """
        return {}

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        return _ic_swept_force_params(seed, n, self._accuracy_base())

    def accuracy_sample_key(self, params: dict) -> dict:
        if "state_eci" in params and isinstance(params["state_eci"], list):
            return {"altitude_km": _altitude_km_from_state(params["state_eci"])}
        return {}


class AccelPointMassGravityTask(_AccelTaskBase):
    """Evaluate central-body point-mass gravity acceleration."""

    @property
    def name(self) -> str:
        return "force_model.accel_point_mass_gravity"

    @property
    def description(self) -> str:
        return "Evaluate point-mass gravity acceleration at a fixed LEO state"

    def generate_params(self, seed: int) -> dict:
        return self._base_params()


class AccelSphericalHarmonics20Task(_AccelTaskBase):
    """Evaluate 20x20 spherical-harmonic gravity acceleration."""

    @property
    def name(self) -> str:
        return "force_model.accel_spherical_harmonics_20"

    @property
    def description(self) -> str:
        return "Evaluate 20x20 spherical-harmonic gravity acceleration at a fixed LEO state"

    def generate_params(self, seed: int) -> dict:
        p = self._base_params()
        p["degree"] = 20
        p["order"] = 20
        return p

    def _accuracy_base(self) -> dict:
        return {"degree": 20, "order": 20}


class AccelSphericalHarmonics80Task(_AccelTaskBase):
    """Evaluate 80x80 spherical-harmonic gravity acceleration."""

    @property
    def name(self) -> str:
        return "force_model.accel_spherical_harmonics_80"

    @property
    def description(self) -> str:
        return "Evaluate 80x80 spherical-harmonic gravity acceleration at a fixed LEO state"

    def generate_params(self, seed: int) -> dict:
        p = self._base_params()
        p["degree"] = 80
        p["order"] = 80
        return p

    def _accuracy_base(self) -> dict:
        return {"degree": 80, "order": 80}


class AccelThirdBodySunTask(_AccelTaskBase):
    """Evaluate Sun third-body acceleration using DE440s ephemeris."""

    @property
    def name(self) -> str:
        return "force_model.accel_third_body_sun"

    @property
    def description(self) -> str:
        return "Evaluate Sun third-body acceleration (DE440s) at a fixed LEO state"

    def generate_params(self, seed: int) -> dict:
        return self._base_params()


class AccelThirdBodyMoonTask(_AccelTaskBase):
    """Evaluate Moon third-body acceleration using DE440s ephemeris."""

    @property
    def name(self) -> str:
        return "force_model.accel_third_body_moon"

    @property
    def description(self) -> str:
        return "Evaluate Moon third-body acceleration (DE440s) at a fixed LEO state"

    def generate_params(self, seed: int) -> dict:
        return self._base_params()
