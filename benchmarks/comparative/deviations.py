"""
Deviation-investigation stubs.

After an accuracy run, this module emits or updates one markdown file per
task whose Orekit-vs-other p99 max-abs error exceeds a per-module threshold.
The intent is twofold:

1. Force every notable residual to get a written cause-of-record. Known
   sources of disagreement (Euler-angle convention, EGM-96 vs EIGEN-6S
   gravity coefficients, ITRF93 vs ITRF2014 realizations, WGS84 vs
   GMAT-equatorial-radius offsets) get pre-populated explanations so
   they don't masquerade as bugs.
2. Surface new outliers loudly. Unfamiliar tasks crossing threshold land
   as ``TODO: investigate`` stubs the next time docs are regenerated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

# Per-module p99 max-abs thresholds (in module-native units). A task whose
# p99 exceeds the threshold for any comparison language is flagged.
THRESHOLDS = {
    "time": 1e-3,            # 1 ms
    "coordinates": 1e-3,     # 1 mm (1e-3 m)
    "attitude": 1e-6,        # tight on rotation-matrix Frobenius residuals
    "frames": 1.0,           # 1 m
    "orbits": 1e-3,          # 1 mm
    "propagation": 100.0,    # 100 m — propagation gaps are expected
    "force_model": 1e-12,    # 1 pm/s² — accelerations are tiny
    "access": 1.0,           # 1 s
}

# Pre-populated explanations for known deviations. Stubs for unlisted
# tasks are emitted with a TODO: investigate line.
KNOWN_CAUSES = {
    "propagation.numerical_twobody": (
        "Both brahe and Orekit are now configured to use fixed-step "
        "Classical Runge-Kutta 4 (RK4) on this task with the same step "
        "size, so the residual reflects purely floating-point ordering "
        "and accumulated round-off across one LEO orbit. Earlier "
        "iterations of this benchmark used the libraries' default "
        "adaptive integrators (DP54 for brahe, DP8(5,3) for Orekit), "
        "which coupled integrator-precision differences into the gap; "
        "the RK4 alignment removes that confound."
    ),
    "propagation.sgp4_single": (
        "Different SGP4 implementations. brahe ports the Vallado 2006 "
        "reference; Orekit uses its own implementation derived from the "
        "same source but with independent floating-point ordering. Residuals "
        "are well within the accepted SGP4 implementation-spread tolerance."
    ),
    "propagation.sgp4_trajectory": (
        "Different SGP4 implementations. See sgp4_single for context. "
        "Trajectory-level residuals accumulate proportionally to propagation "
        "duration."
    ),
    "propagation.numerical_rk4_grav5x5": (
        "brahe-vs-Orekit residual at 5x5 reflects RK4 step-size precision "
        "(sub-millimetre per step accumulated over the orbit). Gravity "
        "coefficient sources differ between backends but at degree 5 the "
        "contribution to position error is negligible (<<1 mm). The "
        "Basilisk-vs-Orekit gap (~75 m) is dominated by Basilisk's "
        "GGM03S vs Orekit's EIGEN-6S; the GMAT-vs-Orekit gap (~2 m) by "
        "GMAT's JGM-2 vs EIGEN-6S."
    ),
    "propagation.numerical_rk4_grav20x20_sun_moon": (
        "brahe-vs-Orekit residual at 20x20 is again the RK4 step-size "
        "floor (<500 µm max). Backend-vs-backend gaps reflect gravity "
        "coefficient sources (brahe: EGM2008, Orekit: EIGEN-6S, GMAT: "
        "JGM-2, Basilisk: GGM03S) and minor third-body ephemeris "
        "precision differences. No implementation defects expected."
    ),
    "propagation.numerical_rk4_grav80x80_full": (
        "After aligning SW and gravity coefficients between brahe and "
        "Orekit, the ~55 m max residual is unchanged — which means "
        "neither input is its dominant source. Specifically:\n\n"
        "**Space weather** — brahe loads Orekit's CSSI "
        "``SpaceWeather-All-v1.2.txt`` via "
        "``brahe.FileSpaceWeatherProvider.from_file`` (see "
        "``implementations/python/base.py::ensure_sw``). Ap and F10.7 "
        "are byte-identical between brahe and Orekit at every tested "
        "epoch.\n\n"
        "**Gravity coefficients** — Orekit's "
        "``GravityFieldFactory`` is reconfigured at the start of every "
        "Java propagation run to read brahe's bundled "
        "``data/gravity_models/EGM2008_360.gfc`` via an explicit "
        "``ICGEMFormatReader``. The benchmark orchestrator sets "
        "``BRAHE_GRAVITY_FILE`` (see "
        "``benchmarks/comparative/config.py``) which the Java adapter "
        "picks up; the adapter calls "
        "``clearPotentialCoefficientsReaders()`` first so Orekit's "
        "default EIGEN-6S file is no longer in the search chain. "
        "Rerunning with vs. without the env var produced numerically "
        "identical results — confirming that EGM2008 and EIGEN-6S "
        "agree closely enough at degree 80 to be invisible at this "
        "fidelity.\n\n"
        "**What's left** — the residual reflects implementation-level "
        "differences in the dynamics that aren't expressible as "
        "shared data: independent NRLMSISE-00 ports (brahe in Rust, "
        "Orekit in Java) accumulate slightly different intermediate "
        "values at LEO densities; SRP eclipse-model details and the "
        "third-body ephemeris source (brahe DE440s vs Orekit bundled "
        "JPL DE) contribute smaller amounts. None of these are "
        "configuration knobs the benchmark adapter can reach from the "
        "outside; closing the gap further would mean changing one "
        "library's algorithm to match the other, which is out of "
        "scope for a comparison benchmark."
    ),
    "coordinates.geodetic_to_ecef": (
        "GMAT uses an Earth equatorial radius of 6378.1363 km versus WGS84's "
        "6378.137 km, which accounts for ~0.7 m position error in geodetic "
        "↔ ECEF conversions. The non-GMAT residuals are sub-nanometer "
        "(WGS84 vs WGS84) and not flagged."
    ),
    "coordinates.ecef_to_geodetic": (
        "GMAT uses an Earth equatorial radius of 6378.1363 km versus WGS84's "
        "6378.137 km, which accounts for ~0.7 m position error in geodetic "
        "↔ ECEF conversions. The non-GMAT residuals are sub-nanometer "
        "(WGS84 vs WGS84) and not flagged."
    ),
    "orbits.cartesian_to_keplerian": (
        "Keplerian element vector ``[a, e, i, raan, argp, M]`` mixes a "
        "length (meters) with five angles (degrees). The current "
        "element-wise compare reports the worst-of-six in unit-free meters, "
        "which can be dominated by an angle disagreement near 360° wrap "
        "or by a small semi-major-axis residual. The future task-specific "
        "comparison (analogous to the geodetic surface-arc conversion) "
        "should split angle and length components before reporting."
    ),
    "frames.state_eci_to_ecef": (
        "Earth orientation parameters: brahe and Orekit use the same IERS "
        "2010 conventions but may load slightly different EOP files. "
        "Sub-meter residuals are expected; >10 m residuals would warrant "
        "investigation."
    ),
    "frames.state_ecef_to_eci": (
        "See state_eci_to_ecef — same EOP-source explanation. The inverse "
        "transformation reuses the same rotation matrix."
    ),
}


def write_deviation_stubs(
    summaries: list[dict],
    output_dir: Path | None = None,
) -> list[Path]:
    """Emit one stub per task whose p99 max-abs exceeds its module threshold.

    Returns the list of written file paths. Idempotent: stubs are
    overwritten on each call so magnitudes stay current with the latest
    accuracy run, but the explanatory prose for known causes is regenerated
    from :data:`KNOWN_CAUSES` rather than read from the existing file —
    callers who want to extend the prose should add to ``KNOWN_CAUSES``.
    """
    # TODO: Find a better way to do this. We don't want to pollute user-facing docs
    # # Resolve the default output directory relative to the repo root (this
    # # file is at <repo>/benchmarks/comparative/deviations.py — three parents
    # # up gets us to the repo root). The earlier ``Path("docs/...")`` form
    # # was cwd-sensitive and could write stubs into a nested directory when
    # # the harness ran from a subprocess that had cd'd elsewhere.
    # if output_dir is None:
    #     repo_root = Path(__file__).resolve().parents[2]
    #     output_dir = repo_root / "docs" / "about" / "benchmark-deviations"
    # output_dir.mkdir(parents=True, exist_ok=True)

    # Group summaries by task; keep the worst p99 across comparisons.
    by_task: dict[str, dict] = {}
    for s in summaries:
        task = s["task_name"]
        p99 = float(s.get("max_abs_p99", 0.0))
        if task not in by_task or p99 > by_task[task]["worst_p99"]:
            by_task[task] = {
                "worst_p99": p99,
                "worst_max": float(s.get("max_abs_max", 0.0)),
                "worst_lang": s["comparison_language"],
                "n_samples": int(s["n_samples"]),
            }

    written: list[Path] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for task, info in sorted(by_task.items()):
        module = task.split(".")[0]
        threshold = THRESHOLDS.get(module, 1.0)
        if info["worst_p99"] < threshold:
            continue

        stub_path = output_dir / f"{task.replace('.', '_')}.md"
        cause = KNOWN_CAUSES.get(task, "TODO: investigate.")
        stub_path.write_text(
            f"# {task}\n\n"
            f"_Last updated from accuracy run: {timestamp}_\n\n"
            f"## Observed residual\n\n"
            f"- p99 max-abs error: {info['worst_p99']:.3e} (vs {info['worst_lang']})\n"
            f"- Worst-sample max-abs error: {info['worst_max']:.3e}\n"
            f"- Samples in sweep: {info['n_samples']}\n"
            f"- Module threshold (p99): {threshold:.3e}\n\n"
            f"## Cause\n\n"
            f"{cause}\n",
            encoding="utf-8",
        )
        written.append(stub_path)
    return written
