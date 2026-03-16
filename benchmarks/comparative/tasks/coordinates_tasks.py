"""
Coordinate conversion benchmark task specifications.
"""

import random

from benchmarks.comparative.tasks.base import BenchmarkTask


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
        # Generate ECEF points on/near Earth surface
        import math

        R_EARTH = 6378137.0
        points = []
        for _ in range(50):
            # Random direction + altitude
            lon = math.radians(rng.uniform(-180.0, 180.0))
            lat = math.radians(rng.uniform(-90.0, 90.0))
            alt = rng.uniform(0.0, 1000e3)
            r = R_EARTH + alt
            x = r * math.cos(lat) * math.cos(lon)
            y = r * math.cos(lat) * math.sin(lon)
            z = r * math.sin(lat)
            points.append([x, y, z])
        return {"points": points}
