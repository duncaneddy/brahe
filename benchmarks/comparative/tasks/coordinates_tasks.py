"""
Coordinate conversion benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.tasks.base import BenchmarkTask

R_EARTH = 6378137.0  # meters


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
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(50):
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
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(50):
            lon = math.radians(rng.uniform(-180.0, 180.0))
            lat = math.radians(rng.uniform(-90.0, 90.0))
            alt = rng.uniform(0.0, 1000e3)
            r = R_EARTH + alt
            x = r * math.cos(lat) * math.cos(lon)
            y = r * math.cos(lat) * math.sin(lon)
            z = r * math.sin(lat)
            points.append([x, y, z])
        return {"points": points}


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
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(50):
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
        return ["python", "rust", "java"]

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)
        points = []
        for _ in range(50):
            lon = math.radians(rng.uniform(-180.0, 180.0))
            lat = math.radians(rng.uniform(-90.0, 90.0))
            radius = R_EARTH + rng.uniform(0.0, 1000e3)
            x = radius * math.cos(lat) * math.cos(lon)
            y = radius * math.cos(lat) * math.sin(lon)
            z = radius * math.sin(lat)
            points.append([x, y, z])
        return {"points": points}


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
        return ["python", "rust", "java"]

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
