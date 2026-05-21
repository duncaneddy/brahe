"""
Access computation benchmark task specifications.
"""

import math
import random

from benchmarks.comparative.results import AccuracyComparison
from benchmarks.comparative.tasks.base import BenchmarkTask
from benchmarks.comparative.tasks.propagation_tasks import ISS_TLE_LINE1, ISS_TLE_LINE2


class Sgp4AccessTask(BenchmarkTask):
    """Benchmark SGP4 access computation: find satellite visibility windows
    for 100 ground locations over 1 day."""

    @property
    def name(self) -> str:
        return "access.sgp4_access"

    @property
    def module(self) -> str:
        return "access"

    @property
    def description(self) -> str:
        return (
            "Compute satellite access windows for 100 ground locations "
            "using SGP4 propagation with 10° minimum elevation over 1 day"
        )

    @property
    def languages(self) -> list[str]:
        return ["python", "rust", "java", "gmat"]

    @property
    def timeout(self) -> int:
        return 1800

    def generate_params(self, seed: int) -> dict:
        rng = random.Random(seed)

        # Generate 100 random ground locations (lon, lat, alt=0)
        locations = []
        for _ in range(100):
            lon = rng.uniform(-180.0, 180.0)
            lat = rng.uniform(-90.0, 90.0)
            locations.append({"lon": lon, "lat": lat, "alt": 0.0})

        return {
            "line1": ISS_TLE_LINE1,
            "line2": ISS_TLE_LINE2,
            "locations": locations,
            "min_elevation_deg": 10.0,
            "search_duration_seconds": 86400.0,
        }

    def detailed_sample_metrics(self, baseline_sample, comparison_sample) -> dict:
        """Per-location metrics surfaced into ``AccuracySample.sample_key``:

        - ``n_windows_baseline`` / ``n_windows_comparison``: contact counts
          each backend found for this ground location
        - ``window_count_diff``: ``|n_baseline - n_comparison|`` (integer
          difference; missed-detection / spurious-detection count)
        - ``start_err_s`` / ``end_err_s``: largest residual across matched
          windows at this location, in seconds
        - ``mean_start_err_s`` / ``mean_end_err_s``: mean residual across
          matched windows

        Unmatched windows (no counterpart within the 120 s greedy tolerance)
        do not contribute to start/end residuals but are reflected in
        ``window_count_diff``.
        """
        windows_a = baseline_sample if isinstance(baseline_sample, list) else []
        windows_b = comparison_sample if isinstance(comparison_sample, list) else []

        tolerance_jd = 120.0 / 86400.0
        unmatched_b = list(range(len(windows_b)))
        start_errs: list[float] = []
        end_errs: list[float] = []

        for wa in windows_a:
            best_idx = None
            best_dist = float("inf")
            for bi in unmatched_b:
                wb = windows_b[bi]
                dist = abs(wa["start_jd"] - wb["start_jd"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = bi
            if best_idx is not None and best_dist < tolerance_jd:
                wb = windows_b[best_idx]
                unmatched_b.remove(best_idx)
                start_errs.append(abs(wa["start_jd"] - wb["start_jd"]) * 86400.0)
                end_errs.append(abs(wa["end_jd"] - wb["end_jd"]) * 86400.0)

        n_a = len(windows_a)
        n_b = len(windows_b)
        n_matched = len(start_errs)

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "n_windows_baseline": n_a,
            "n_windows_comparison": n_b,
            "n_windows_matched": n_matched,
            "window_count_diff": abs(n_a - n_b),
            "start_err_s_max": max(start_errs) if start_errs else 0.0,
            "end_err_s_max": max(end_errs) if end_errs else 0.0,
            "start_err_s_mean": _mean(start_errs),
            "end_err_s_mean": _mean(end_errs),
        }

    def compare_results(
        self,
        results_a: list,
        results_b: list,
        language_a: str,
        language_b: str,
    ) -> AccuracyComparison:
        """Compare access windows using greedy nearest-start-time matching.

        Results format: list of 100 entries, each a list of
        {"start_jd": float, "end_jd": float} windows.
        """
        all_timing_errors = []
        total_windows_a = 0
        total_windows_b = 0

        tolerance_jd = 120.0 / 86400.0  # 120 seconds in days

        for loc_idx in range(min(len(results_a), len(results_b))):
            windows_a = results_a[loc_idx]
            windows_b = results_b[loc_idx]

            total_windows_a += len(windows_a)
            total_windows_b += len(windows_b)

            # Greedy nearest-start-time matching
            unmatched_b = list(range(len(windows_b)))

            for wa in windows_a:
                best_idx = None
                best_dist = float("inf")

                for bi in unmatched_b:
                    wb = windows_b[bi]
                    dist = abs(wa["start_jd"] - wb["start_jd"])
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = bi

                if best_idx is not None and best_dist < tolerance_jd:
                    wb = windows_b[best_idx]
                    unmatched_b.remove(best_idx)

                    # Compute timing errors in seconds
                    start_err = abs(wa["start_jd"] - wb["start_jd"]) * 86400.0
                    end_err = abs(wa["end_jd"] - wb["end_jd"]) * 86400.0
                    all_timing_errors.extend([start_err, end_err])

        max_total = max(total_windows_a, total_windows_b, 1)
        window_count_diff = abs(total_windows_a - total_windows_b) / max_total

        if all_timing_errors:
            max_abs = max(all_timing_errors)
            rms = math.sqrt(
                sum(e * e for e in all_timing_errors) / len(all_timing_errors)
            )
        else:
            max_abs = float("nan")
            rms = float("nan")

        return AccuracyComparison(
            task_name=self.name,
            reference_language=language_a,
            comparison_language=language_b,
            max_abs_error=max_abs,
            max_rel_error=window_count_diff,
            rms_error=rms,
        )
