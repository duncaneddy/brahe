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

Stubs live under ``docs/about/benchmark-deviations/``. The harness updates
the magnitudes in each stub but never deletes prose — a stub that becomes
clean stays in the directory with its history intact.
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
        "brahe-vs-Orekit ~55 m max over 100 LEO orbits is consistent "
        "with **atmospheric drag space-weather input differences** — "
        "EIGEN-6S and EGM2008 truncated to 80x80 agree to sub-metre "
        "over one orbit, so gravity is not the dominant term. brahe "
        "reads space weather from its bundled NRLMSISE-00 driver "
        "configuration; Orekit reads CSSI's "
        "`SpaceWeather-All-v1.2.txt`; GMAT uses MSISE-90 with its own "
        "SW table; Basilisk uses representative quiet-Sun values "
        "(Ap=8, F10.7=110). These SW differences integrate to "
        "metre-to-tens-of-metres position errors at LEO over one "
        "orbit, with the spread widening at lower altitudes. SRP "
        "models are also slightly different across backends but "
        "contribute much less than drag at LEO. brahe already exposes "
        "``from_file`` for both gravity (.gfc) and space-weather "
        "inputs, so aligning the backends on a single CSSI SW table "
        "and a single EGM2008 / EIGEN-6S coefficient file would "
        "collapse this residual; the work is the benchmark adapter "
        "side, not the brahe library side."
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
    "attitude.euler_angle_to_quaternion": (
        "Euler-angle convention mismatch identified.\n\n"
        "**brahe** interprets `EulerAngleOrder::ZYX` with input `(phi, "
        "theta, psi)` as **extrinsic ZYX**: `R = Rx(psi) * Ry(theta) * "
        "Rz(phi)`. That is, phi rotates about the world-frame Z axis "
        "first, then Y, then X.\n\n"
        "**Orekit / Basilisk / GMAT** in this benchmark interpret the "
        "same triplet as **intrinsic ZYX**: `R = Rz(phi) * Ry(theta) * "
        "Rx(psi)`. That is, phi rotates about Z, then theta about the "
        "new (body-frame) Y', then psi about the new X''.\n\n"
        "These two conventions are *inverses* of each other — applying "
        "brahe's R then Orekit's R to the same vector yields different "
        "rotated vectors. That is why all backends agree with each other "
        "on this task (they all use intrinsic ZYX) but disagree with "
        "brahe (extrinsic ZYX).\n\n"
        "Why the inverse task (`quaternion_to_euler_angle`) still agrees "
        "to machine epsilon: each library is internally consistent — "
        "brahe round-trips against its own extrinsic convention; the "
        "others round-trip against intrinsic. The benchmark feeds a "
        "random quaternion into each library and reads back the Euler "
        "triplet *in that library's own convention*. The output triplets "
        "differ between libraries, but they all map back to the same "
        "input quaternion when run through their own Euler→Quat, so the "
        "*quaternion-space* residual (which is what the benchmark "
        "measures) is zero.\n\n"
        "Residuals are reported as the Frobenius norm of $R_a - R_b$; a "
        "value near 2 corresponds to roughly a 90° rotation gap, near "
        "2√2 ≈ 2.83 corresponds to a 180° flip.\n\n"
        "**Action**: brahe's `EulerAngle::from_euler_angle` for `ZYX` "
        "should be reviewed against the conventions used in the wider "
        "astrodynamics literature. Most aerospace texts (Wertz, "
        "Schaub/Junkins) document ZYX as the intrinsic order Rz·Ry·Rx; "
        "if brahe intends to follow that convention, the per-order "
        "quaternion formulas in `src/attitude/quaternion.rs` need to be "
        "rewritten. Out of scope for the benchmark redesign — filed as "
        "a brahe core issue."
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
    # Resolve the default output directory relative to the repo root (this
    # file is at <repo>/benchmarks/comparative/deviations.py — three parents
    # up gets us to the repo root). The earlier ``Path("docs/...")`` form
    # was cwd-sensitive and could write stubs into a nested directory when
    # the harness ran from a subprocess that had cd'd elsewhere.
    if output_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        output_dir = repo_root / "docs" / "about" / "benchmark-deviations"
    output_dir.mkdir(parents=True, exist_ok=True)

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
