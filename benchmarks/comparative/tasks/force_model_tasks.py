"""
Force-model function-level benchmark tasks.

These tasks evaluate a single acceleration term at a fixed spacecraft state
and epoch. They isolate force-model implementations from integrator behavior,
so any difference between brahe and Orekit is attributable to the force-model
calculation itself (gravity coefficients, third-body ephemeris, frame
transformation) rather than to numerical integration error.
"""

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
