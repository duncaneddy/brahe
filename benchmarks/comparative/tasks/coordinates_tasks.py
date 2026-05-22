"""
Coordinate conversion benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask

R_EARTH = 6378137.0  # meters


def _geodetic_position_error(
    ga: list[float], gb: list[float]
) -> tuple[float, float]:
    """Convert a pair of ``[lon_deg, lat_deg, alt_m]`` residuals into a
    pair of (component-wise meters, RMS meters) suitable for direct
    comparison against position-error magnitudes.

    Angle differences are converted to surface distance: the latitude
    delta multiplied by Earth's radius, and the longitude delta multiplied
    by ``R_EARTH * cos(lat_a)``. This collapses mixed-unit geodetic output
    into a single position-equivalent metric so the docs CSV doesn't have
    to mix degrees and meters in the same cell.
    """
    lon_a, lat_a, alt_a = ga[0], ga[1], ga[2]
    lon_b, lat_b, alt_b = gb[0], gb[1], gb[2]

    # Wrap longitude delta into [-180, 180] so a small numerical wrap
    # doesn't show up as a 360-deg jump.
    dlon = ((lon_a - lon_b + 540.0) % 360.0) - 180.0

    lat_rad = math.radians(lat_a)
    dlon_m = math.radians(dlon) * R_EARTH * math.cos(lat_rad)
    dlat_m = math.radians(lat_a - lat_b) * R_EARTH
    dalt_m = alt_a - alt_b

    max_abs = max(abs(dlon_m), abs(dlat_m), abs(dalt_m))
    rms = math.sqrt((dlon_m * dlon_m + dlat_m * dlat_m + dalt_m * dalt_m) / 3.0)
    return max_abs, rms


def _geocentric_position_error(
    ga: list[float], gb: list[float]
) -> tuple[float, float]:
    """Same as :func:`_geodetic_position_error` but the third component is
    geocentric radius (meters) rather than altitude above ellipsoid. The
    error metric is identical: the third-component delta is already a
    distance, and the angular components convert via surface arc.
    """
    return _geodetic_position_error(ga, gb)


def _build_position_comparison(
    task_name: str,
    language_a: str,
    language_b: str,
    results_a: list,
    results_b: list,
    pair_error: callable,
) -> AccuracyComparison:
    """Build an :class:`AccuracyComparison` from a per-sample function that
    returns ``(max_abs_meters, rms_meters)`` for a single input pair.

    Factored out so the geodetic and geocentric tasks share the
    aggregation logic without copying it.
    """
    n = min(len(results_a), len(results_b))
    if n == 0:
        return AccuracyComparison(
            task_name=task_name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=float("nan"),
            max_rel_error=float("nan"),
            rms_error=float("nan"),
        )
    abs_errors: list[float] = []
    rms_errors: list[float] = []
    for i in range(n):
        ga = results_a[i]
        gb = results_b[i]
        if not isinstance(ga, (list, tuple)) or len(ga) < 3:
            continue
        if not isinstance(gb, (list, tuple)) or len(gb) < 3:
            continue
        max_abs, rms = pair_error(list(ga), list(gb))
        abs_errors.append(max_abs)
        rms_errors.append(rms)

    if not abs_errors:
        return AccuracyComparison(
            task_name=task_name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=float("nan"),
            max_rel_error=float("nan"),
            rms_error=float("nan"),
        )

    overall_max = max(abs_errors)
    overall_rms = math.sqrt(sum(e * e for e in rms_errors) / len(rms_errors))
    # Relative error is meaningless for a position-equivalent metric
    # (denominator unit is meters but we have no canonical scale). Report
    # the absolute value normalized by 1 m so the column has *some* number.
    rel = overall_max / 1.0
    return AccuracyComparison(
        task_name=task_name,
        reference_language=language_a,
        comparison_language=language_b,
        max_abs_error=overall_max,
        max_rel_error=rel,
        rms_error=overall_rms,
    )


class GeodeticToEcefTask(BenchmarkTask):
    """Geodetic (lon, lat, alt) to ECEF conversion benchmark."""

    @property
    def name(self) -> str:
        return "coordinates.geodetic_to_ecef"

    @property
    def module(self) -> str:
        return "coordinates"

    @property
    def description(self) -> str:
        return "Convert geodetic coordinates (lon, lat, alt) to ECEF position vectors"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "basilisk", "gmat", "nyx"]

    def generate_params(self, seed: int) -> dict:
        return self._gen_points(seed, 50)

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        return self._gen_points(seed, n)

    @staticmethod
    def _gen_points(seed: int, n: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(n):
            lon = rng.uniform(-180.0, 180.0)
            lat = rng.uniform(-90.0, 90.0)
            alt = rng.uniform(0.0, 1000e3)  # 0 to 1000 km altitude
            points.append([lon, lat, alt])
        return {"points": points}


class EcefToGeodeticTask(BenchmarkTask):
    """ECEF to geodetic (lon, lat, alt) conversion benchmark."""

    @property
    def name(self) -> str:
        return "coordinates.ecef_to_geodetic"

    @property
    def module(self) -> str:
        return "coordinates"

    @property
    def description(self) -> str:
        return "Convert ECEF position vectors to geodetic coordinates (lon, lat, alt)"

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "basilisk", "gmat", "nyx"]

    def generate_params(self, seed: int) -> dict:
        return self._gen_points(seed, 50)

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        return self._gen_points(seed, n)

    @staticmethod
    def _gen_points(seed: int, n: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(n):
            lon = math.radians(rng.uniform(-180.0, 180.0))
            lat = math.radians(rng.uniform(-90.0, 90.0))
            alt = rng.uniform(0.0, 1000e3)
            r = R_EARTH + alt
            x = r * math.cos(lat) * math.cos(lon)
            y = r * math.cos(lat) * math.sin(lon)
            z = r * math.sin(lat)
            points.append([x, y, z])
        return {"points": points}

    def compare_results(
        self, results_a, results_b, language_a, language_b
    ) -> AccuracyComparison:
        """Compare ``[lon_deg, lat_deg, alt_m]`` triplets in meters by
        converting the angular components to surface distance.

        The default element-wise compare would mix degrees and meters in
        a single max-abs cell — the docs then either misleadingly
        label everything as one unit or hide the unit entirely. Reporting
        a single position-equivalent error in meters keeps the table
        honest and lets the CDF figure use a consistent axis.
        """
        return _build_position_comparison(
            self.name,
            language_a,
            language_b,
            results_a,
            results_b,
            _geodetic_position_error,
        )


class GeocentricToEcefTask(BenchmarkTask):
    """Geocentric (lon, lat, radius) to ECEF conversion benchmark."""

    @property
    def name(self) -> str:
        return "coordinates.geocentric_to_ecef"

    @property
    def module(self) -> str:
        return "coordinates"

    @property
    def description(self) -> str:
        return (
            "Convert geocentric coordinates (lon, lat, radius) to ECEF position vectors"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "gmat"]

    def generate_params(self, seed: int) -> dict:
        return self._gen_points(seed, 50)

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        return self._gen_points(seed, n)

    @staticmethod
    def _gen_points(seed: int, n: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(n):
            lon = rng.uniform(-180.0, 180.0)  # degrees
            lat = rng.uniform(-90.0, 90.0)  # degrees
            altitude = rng.uniform(0.0, 1000e3)  # meters above sphere
            points.append([lon, lat, altitude])
        return {"points": points}


class EcefToGeocentricTask(BenchmarkTask):
    """ECEF to geocentric (lon, lat, radius) conversion benchmark."""

    @property
    def name(self) -> str:
        return "coordinates.ecef_to_geocentric"

    @property
    def module(self) -> str:
        return "coordinates"

    @property
    def description(self) -> str:
        return (
            "Convert ECEF position vectors to geocentric coordinates (lon, lat, radius)"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "gmat"]

    def generate_params(self, seed: int) -> dict:
        return self._gen_points(seed, 50)

    def generate_accuracy_samples(self, seed: int, n: int) -> dict:
        return self._gen_points(seed, n)

    @staticmethod
    def _gen_points(seed: int, n: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(n):
            lon = math.radians(rng.uniform(-180.0, 180.0))
            lat = math.radians(rng.uniform(-90.0, 90.0))
            radius = R_EARTH + rng.uniform(0.0, 1000e3)
            x = radius * math.cos(lat) * math.cos(lon)
            y = radius * math.cos(lat) * math.sin(lon)
            z = radius * math.sin(lat)
            points.append([x, y, z])
        return {"points": points}

    def compare_results(
        self, results_a, results_b, language_a, language_b
    ) -> AccuracyComparison:
        """Compare geocentric ``[lon_deg, lat_deg, radius_m]`` triplets in
        meters: same approach as the geodetic compare, with the radius
        delta acting as the third position-distance component.
        """
        return _build_position_comparison(
            self.name,
            language_a,
            language_b,
            results_a,
            results_b,
            _geocentric_position_error,
        )


class EcefToAzelTask(BenchmarkTask):
    """ECEF station+satellite to azimuth/elevation/range conversion benchmark."""

    @property
    def name(self) -> str:
        return "coordinates.ecef_to_azel"

    @property
    def module(self) -> str:
        return "coordinates"

    @property
    def description(self) -> str:
        return (
            "Convert station and satellite ECEF positions to azimuth, elevation, range"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "nyx"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        pairs = []
        for _ in range(50):
            # Random station on Earth surface
            sta_lon = rng.uniform(-180.0, 180.0)  # degrees
            sta_lat = rng.uniform(-70.0, 70.0)  # degrees (avoid poles)
            sta_alt = rng.uniform(0.0, 3000.0)  # 0-3km altitude

            # Convert station geodetic to ECEF for comparison input
            sta_lon_rad = math.radians(sta_lon)
            sta_lat_rad = math.radians(sta_lat)
            f = 1.0 / 298.257223563  # WGS84 flattening
            e2 = 2 * f - f * f
            sin_lat = math.sin(sta_lat_rad)
            cos_lat = math.cos(sta_lat_rad)
            N = R_EARTH / math.sqrt(1 - e2 * sin_lat * sin_lat)
            sta_x = (N + sta_alt) * cos_lat * math.cos(sta_lon_rad)
            sta_y = (N + sta_alt) * cos_lat * math.sin(sta_lon_rad)
            sta_z = (N * (1 - e2) + sta_alt) * sin_lat

            # Satellite above station with some offset
            sat_lon = sta_lon + rng.uniform(-10.0, 10.0)
            sat_lat = sta_lat + rng.uniform(-10.0, 10.0)
            sat_alt = rng.uniform(200e3, 1000e3)

            sat_lon_rad = math.radians(sat_lon)
            sat_lat_rad = math.radians(sat_lat)
            sin_lat_s = math.sin(sat_lat_rad)
            cos_lat_s = math.cos(sat_lat_rad)
            N_s = R_EARTH / math.sqrt(1 - e2 * sin_lat_s * sin_lat_s)
            sat_x = (N_s + sat_alt) * cos_lat_s * math.cos(sat_lon_rad)
            sat_y = (N_s + sat_alt) * cos_lat_s * math.sin(sat_lon_rad)
            sat_z = (N_s * (1 - e2) + sat_alt) * sin_lat_s

            pairs.append(
                {
                    "station_ecef": [sta_x, sta_y, sta_z],
                    "satellite_ecef": [sat_x, sat_y, sat_z],
                    "station_geodetic": [sta_lon, sta_lat, sta_alt],
                }
            )
        return {"pairs": pairs}
