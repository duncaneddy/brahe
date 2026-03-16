# /// script
# dependencies = ["brahe", "plotly"]
# FLAGS = ["IGNORE"]
# ///
# ruff: noqa: E501
"""
Generate comparative benchmark charts for Brahe documentation.

Reads the latest JSON results from benchmarks/comparative/results/ and generates
themed Plotly figures comparing Java (OreKit), Python (Brahe), and Rust (Brahe)
performance across multiple task categories.

Run manually with:
    just make-plots --ignore
"""

import csv
import os
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass

import plotly.graph_objects as go

# Add plots directory to path for brahe_theme import
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

# Add project root for benchmark results import
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from benchmarks.comparative.results import BenchmarkRun

# Configuration
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
RESULTS_DIR = pathlib.Path("benchmarks/comparative/results")

# Specific run file with complete results (72 tasks, 48 accuracy comparisons)
RESULTS_FILE = RESULTS_DIR / "run_a8ef0371_2026-03-16.json"

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Module display order and human-readable names
MODULE_ORDER = [
    "time",
    "coordinates",
    "attitude",
    "frames",
    "orbits",
    "propagation",
    "access",
]
MODULE_LABELS = {
    "time": "Time",
    "coordinates": "Coordinates",
    "attitude": "Attitude",
    "frames": "Frames",
    "orbits": "Orbits",
    "propagation": "Propagation",
    "access": "Access",
}

# Language display order and colors
LANGUAGES = ["java", "python", "rust"]
LANGUAGE_LABELS = {
    "java": "Java (OreKit)",
    "python": "Python (Brahe)",
    "rust": "Rust (Brahe)",
}


def _get_language_colors(theme: str) -> dict[str, str]:
    """Get language-specific colors: Java=blue, Python=orange, Rust=green."""
    colors = get_theme_colors(theme)
    return {
        "java": colors["primary"],
        "python": colors["secondary"],
        "rust": colors["accent"],
    }


def _task_label(task_name: str) -> str:
    """Convert task_name like 'time.utc_to_tai' to 'UTC to TAI'."""
    # Strip module prefix
    _, _, task = task_name.partition(".")
    # Special case handling for common abbreviations
    label = task.replace("_", " ").title()
    # Fix known abbreviations
    for abbr in [
        "Ecef",
        "Eci",
        "Enz",
        "Azel",
        "Utc",
        "Tai",
        "Tt",
        "Gps",
        "Ut1",
        "Sgp4",
        "Twobody",
    ]:
        proper = {
            "Ecef": "ECEF",
            "Eci": "ECI",
            "Enz": "ENZ",
            "Azel": "AzEl",
            "Utc": "UTC",
            "Tai": "TAI",
            "Tt": "TT",
            "Gps": "GPS",
            "Ut1": "UT1",
            "Sgp4": "SGP4",
            "Twobody": "Two-Body",
        }.get(abbr, abbr.upper())
        label = label.replace(abbr, proper)
    return label


def _scale_times(seconds: list[float]) -> tuple[list[float], str]:
    """Scale times to appropriate display units."""
    mean = sum(seconds) / len(seconds) if seconds else 0
    if mean < 1e-6:
        return [t * 1e9 for t in seconds], "ns"
    elif mean < 1e-3:
        return [t * 1e6 for t in seconds], "\u00b5s"
    elif mean < 1.0:
        return [t * 1e3 for t in seconds], "ms"
    return seconds, "s"


def _format_time(seconds: float) -> str:
    """Format a single time value with appropriate units for hover text."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.1f} \u00b5s"
    elif seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def load_results() -> BenchmarkRun:
    """Load benchmark results from the specific complete run file."""
    if RESULTS_FILE.exists():
        return BenchmarkRun.load(RESULTS_FILE)
    # Fallback: find the run with the most task results
    run = BenchmarkRun.load_latest(RESULTS_DIR)
    if run is None:
        raise FileNotFoundError(f"No benchmark results found in {RESULTS_DIR}")
    return run


@dataclass
class TaskStats:
    """Mean and standard deviation for a single task/language pair."""

    mean: float
    std: float


def group_by_module(
    run: BenchmarkRun,
) -> dict[str, dict[str, dict[str, TaskStats]]]:
    """Group results by module -> task -> language -> TaskStats."""
    modules: dict[str, dict[str, dict[str, TaskStats]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for r in run.task_results:
        module = r.task_name.split(".")[0]
        modules[module][r.task_name][r.language] = TaskStats(mean=r.mean, std=r.std)
    return modules


# ── Figure: Speedup chart (all tasks, horizontal bars) ─────────────────────


def make_speedup_figure(run: BenchmarkRun):
    """Generate horizontal bar chart showing Python & Rust speedup vs Java."""
    modules = group_by_module(run)

    # Collect all tasks in module order
    tasks = []
    for module in MODULE_ORDER:
        if module in modules:
            for task in sorted(modules[module].keys()):
                tasks.append(task)

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        fig = go.Figure()

        labels = [_task_label(t) for t in tasks]
        py_speedups = []
        rs_speedups = []

        for task in tasks:
            module = task.split(".")[0]
            stats = modules[module][task]
            java_t = stats["java"].mean if "java" in stats else 0
            py_t = stats["python"].mean if "python" in stats else 0
            rs_t = stats["rust"].mean if "rust" in stats else 0
            py_speedups.append(java_t / py_t if py_t else 0)
            rs_speedups.append(java_t / rs_t if rs_t else 0)

        # Reverse for horizontal bar layout (top task at top)
        labels_rev = labels[::-1]
        py_rev = py_speedups[::-1]
        rs_rev = rs_speedups[::-1]

        fig.add_trace(
            go.Bar(
                name="Rust (Brahe)",
                y=labels_rev,
                x=rs_rev,
                orientation="h",
                marker_color=lang_colors["rust"],
                hovertemplate="%{y}<br>%{x:.1f}x vs Java<extra>Rust</extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                name="Python (Brahe)",
                y=labels_rev,
                x=py_rev,
                orientation="h",
                marker_color=lang_colors["python"],
                hovertemplate="%{y}<br>%{x:.1f}x vs Java<extra>Python</extra>",
            )
        )

        # Add vertical line at x=1 (Java baseline)
        fig.add_vline(
            x=1,
            line_dash="dash",
            line_color="gray",
            line_width=1,
        )

        fig.update_layout(
            title="Speedup vs Java (OreKit 12.2)",
            xaxis_title="Speedup Factor (higher is faster)",
            xaxis_type="log",
            barmode="group",
            height=1200,
            margin=dict(l=200, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        return fig

    save_themed_html(make_fig, OUTDIR / "fig_bench_speedup")


# ── Figure: Per-module grouped bar charts ──────────────────────────────────


def make_module_figure(run: BenchmarkRun, module: str):
    """Generate grouped bar chart for a single module."""
    modules = group_by_module(run)
    if module not in modules:
        print(f"  Skipping {module}: no results")
        return

    task_data = modules[module]
    task_names = sorted(task_data.keys())

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        fig = go.Figure()

        labels = [_task_label(t) for t in task_names]

        for lang in LANGUAGES:
            means_raw = [
                task_data[t][lang].mean if lang in task_data[t] else 0
                for t in task_names
            ]
            stds_raw = [
                task_data[t][lang].std if lang in task_data[t] else 0
                for t in task_names
            ]
            scaled, unit = _scale_times(means_raw)
            # Scale the 3-sigma error bars with the same factor as the means
            scale_factor = scaled[0] / means_raw[0] if means_raw[0] else 1
            error_upper = [s * scale_factor * 3 for s in stds_raw]
            # Clamp lower error bars so they don't exceed the bar value
            # (which would go negative/off-screen on a log scale)
            error_lower = [min(e, v * 0.9) for e, v in zip(error_upper, scaled)]
            hover_texts = [_format_time(m) for m in means_raw]

            fig.add_trace(
                go.Bar(
                    name=LANGUAGE_LABELS[lang],
                    x=labels,
                    y=scaled,
                    error_y=dict(
                        type="data",
                        array=error_upper,
                        arrayminus=error_lower,
                        visible=True,
                    ),
                    marker_color=lang_colors[lang],
                    hovertemplate="%{x}<br>%{customdata}<extra>"
                    + LANGUAGE_LABELS[lang]
                    + "</extra>",
                    customdata=hover_texts,
                )
            )

        fig.update_layout(
            title=f"{MODULE_LABELS[module]} Benchmark: Mean Execution Time",
            xaxis_title="Task",
            yaxis_title=f"Time ({unit})<br><sub>(lower is better)</sub>",
            yaxis_type="log",
            barmode="group",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            height=500,
        )
        return fig

    save_themed_html(make_fig, OUTDIR / f"fig_bench_{module}")


# ── CSV table generation ───────────────────────────────────────────────────


def _format_accuracy(value: float, module: str) -> str:
    """Format an accuracy error value with context-appropriate units.

    Raw errors from AccuracyComparison are in SI base units (meters for
    position, seconds for time). Units are chosen based on module context
    and value magnitude.
    """
    if module == "access":
        # Access errors are in seconds
        if value < 1.0:
            return f"{value * 1e3:.1f} ms"
        return f"{value:.1f} s"

    # Position-based errors (meters) — select unit by magnitude
    if module in ("coordinates", "orbits"):
        # These are typically nanometer-scale
        nm = value * 1e9
        if nm >= 100:
            return f"{nm:.0f} nm"
        elif nm >= 1:
            return f"{nm:.2f} nm"
        else:
            return f"{nm:.2f} nm"

    # Propagation and others: auto-select based on magnitude
    if value < 1e-6:
        nm = value * 1e9
        return f"{nm:.0f} nm" if nm >= 10 else f"{nm:.1f} nm"
    elif value < 1e-3:
        return f"{value * 1e6:.1f} µm"
    elif value < 1.0:
        return f"{value:.3f} m"
    else:
        return f"{value:.1f} m" if value >= 10 else f"{value:.2f} m"


def _format_speedup(speedup: float) -> str:
    """Format a speedup value like '14.4×'."""
    if speedup >= 10:
        return f"{speedup:.1f}×"
    else:
        return f"{speedup:.2f}×"


def _comparison_label(ref: str, comp: str) -> str:
    """Format a comparison label like 'Java vs Python'."""
    lang_labels = {"java": "Java", "python": "Python", "rust": "Rust"}
    return f"{lang_labels.get(ref, ref)} vs {lang_labels.get(comp, comp)}"


def _write_csv(filepath: pathlib.Path, headers: list[str], rows: list[list[str]]):
    """Write a CSV file with the given headers and rows."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_csv_tables(run: BenchmarkRun):
    """Generate CSV tables for all benchmark data alongside the Plotly figures."""
    modules = group_by_module(run)

    # ── Overview table ─────────────────────────────────────────────────────
    overview_rows = []
    for module in MODULE_ORDER:
        if module not in modules:
            continue
        task_data = modules[module]
        task_count = len(task_data)
        py_speedups = []
        rs_speedups = []
        for task_stats in task_data.values():
            java_t = task_stats["java"].mean if "java" in task_stats else 0
            py_t = task_stats["python"].mean if "python" in task_stats else 0
            rs_t = task_stats["rust"].mean if "rust" in task_stats else 0
            if py_t and java_t:
                py_speedups.append(java_t / py_t)
            if rs_t and java_t:
                rs_speedups.append(java_t / rs_t)
        avg_py = sum(py_speedups) / len(py_speedups) if py_speedups else 0
        avg_rs = sum(rs_speedups) / len(rs_speedups) if rs_speedups else 0
        overview_rows.append(
            [
                MODULE_LABELS[module],
                str(task_count),
                _format_speedup(avg_py),
                _format_speedup(avg_rs),
            ]
        )
    _write_csv(
        OUTDIR / "bench_overview.csv",
        ["Module", "Tasks", "Avg Python Speedup", "Avg Rust Speedup"],
        overview_rows,
    )

    # ── Per-module performance tables ──────────────────────────────────────
    for module in MODULE_ORDER:
        if module not in modules:
            continue
        task_data = modules[module]
        perf_rows = []
        for task_name in sorted(task_data.keys()):
            stats = task_data[task_name]
            java_t = stats["java"].mean if "java" in stats else 0
            py_t = stats["python"].mean if "python" in stats else 0
            rs_t = stats["rust"].mean if "rust" in stats else 0
            py_speedup = java_t / py_t if py_t and java_t else 0
            rs_speedup = java_t / rs_t if rs_t and java_t else 0
            perf_rows.append(
                [
                    _task_label(task_name),
                    _format_time(java_t),
                    _format_time(py_t),
                    _format_time(rs_t),
                    _format_speedup(py_speedup),
                    _format_speedup(rs_speedup),
                ]
            )
        _write_csv(
            OUTDIR / f"bench_perf_{module}.csv",
            ["Task", "Java", "Python", "Rust", "Python Speedup", "Rust Speedup"],
            perf_rows,
        )

    # ── Per-module accuracy tables ─────────────────────────────────────────
    # Group accuracy comparisons by module
    accuracy_by_module: dict[str, list] = defaultdict(list)
    for ac in run.accuracy_comparisons:
        mod = ac.task_name.split(".")[0]
        accuracy_by_module[mod].append(ac)

    for module in MODULE_ORDER:
        if module not in accuracy_by_module:
            continue
        comparisons = accuracy_by_module[module]

        # Determine table format based on module
        if module == "propagation":
            # Propagation accuracy table has a different format: no comparison
            # column, and includes notes. Group by task, take max across comparisons.
            task_accuracy: dict[str, dict] = {}
            for ac in comparisons:
                task = ac.task_name
                if task not in task_accuracy:
                    task_accuracy[task] = {
                        "max_abs": ac.max_abs_error,
                        "rms": ac.rms_error,
                    }
                else:
                    existing = task_accuracy[task]
                    existing["max_abs"] = max(existing["max_abs"], ac.max_abs_error)
                    existing["rms"] = max(existing["rms"], ac.rms_error)

            # Notes for propagation tasks
            prop_notes = {
                "propagation.keplerian_single": "Sub-millimeter agreement",
                "propagation.keplerian_trajectory": "Sub-millimeter agreement",
                "propagation.numerical_twobody": "Different integrators (RK4 vs RK78)",
                "propagation.sgp4_single": "Different SGP4 implementations",
                "propagation.sgp4_trajectory": "Different SGP4 implementations",
            }

            acc_rows = []
            for task_name in sorted(task_accuracy.keys()):
                vals = task_accuracy[task_name]
                acc_rows.append(
                    [
                        _task_label(task_name),
                        _format_accuracy(vals["max_abs"], module),
                        _format_accuracy(vals["rms"], module),
                        prop_notes.get(task_name, ""),
                    ]
                )
            _write_csv(
                OUTDIR / f"bench_accuracy_{module}.csv",
                ["Task", "Max Abs Error", "RMS Error", "Notes"],
                acc_rows,
            )
        else:
            # Standard accuracy table with comparison column
            acc_rows = []
            for ac in sorted(
                comparisons, key=lambda a: (a.task_name, a.comparison_language)
            ):
                acc_rows.append(
                    [
                        _task_label(ac.task_name),
                        _comparison_label(
                            ac.reference_language, ac.comparison_language
                        ),
                        _format_accuracy(ac.max_abs_error, module),
                        _format_accuracy(ac.rms_error, module),
                    ]
                )
            _write_csv(
                OUTDIR / f"bench_accuracy_{module}.csv",
                ["Task", "Comparison", "Max Abs Error", "RMS Error"],
                acc_rows,
            )


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading benchmark results...")
    run = load_results()
    print(
        f"  Run {run.run_id}: {len(run.task_results)} task results, "
        f"{len(run.accuracy_comparisons)} accuracy comparisons"
    )

    print("Generating speedup chart...")
    make_speedup_figure(run)

    for module in MODULE_ORDER:
        print(f"Generating {module} chart...")
        make_module_figure(run, module)

    print("Generating CSV tables...")
    generate_csv_tables(run)

    print(f"\nAll figures and tables written to {OUTDIR}/")
