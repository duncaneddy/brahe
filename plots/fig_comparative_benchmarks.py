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
from benchmarks.comparative.results import BenchmarkRun, read_jsonl

# Configuration
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
RESULTS_DIR = pathlib.Path("benchmarks/comparative/results")

# Default to the canonical run_latest.json written by every benchmark save();
# override with BRAHE_BENCH_RESULTS_FILE (absolute or relative to RESULTS_DIR).
# Falls back to BenchmarkRun.load_latest() if neither exists.
RESULTS_FILE_ENV = os.getenv("BRAHE_BENCH_RESULTS_FILE")
if RESULTS_FILE_ENV:
    _env_path = pathlib.Path(RESULTS_FILE_ENV)
    RESULTS_FILE = _env_path if _env_path.is_absolute() else RESULTS_DIR / _env_path
else:
    RESULTS_FILE = RESULTS_DIR / "run_latest.json"

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Per-module task display order. Tasks listed here render in the given
# sequence; tasks not listed fall back to alphabetical at the end. The
# ordering is meant to follow physical reasoning rather than alphabet —
# propagation walks from analytical to numerical with increasing
# force-model fidelity, which matches how a reader would step through the
# fidelity ladder.
TASK_ORDER: dict[str, list[str]] = {
    "propagation": [
        "propagation.keplerian_single",
        "propagation.keplerian_trajectory",
        "propagation.sgp4_single",
        "propagation.sgp4_trajectory",
        "propagation.numerical_twobody",
        "propagation.numerical_rk4_grav5x5",
        "propagation.numerical_rk4_grav20x20_sun_moon",
        "propagation.numerical_rk4_grav80x80_full",
    ],
}


def _order_tasks(module: str, tasks: list[str] | set[str]) -> list[str]:
    """Return ``tasks`` ordered by ``TASK_ORDER[module]`` first, then by
    alphabetic for anything not in the explicit list. Tasks present in
    the explicit order but missing from ``tasks`` are silently dropped."""
    explicit = TASK_ORDER.get(module, [])
    in_set = set(tasks)
    ordered = [t for t in explicit if t in in_set]
    remaining = sorted(in_set - set(ordered))
    return ordered + remaining


# Module display order and human-readable names
MODULE_ORDER = [
    "time",
    "coordinates",
    "attitude",
    "frames",
    "orbits",
    "propagation",
    "force_model",
    "access",
]
MODULE_LABELS = {
    "time": "Time",
    "coordinates": "Coordinates",
    "attitude": "Attitude",
    "frames": "Frames",
    "orbits": "Orbits",
    "propagation": "Propagation",
    "force_model": "Force Model",
    "access": "Access",
}

# Language display order and colors (Orekit -> GMAT -> Basilisk -> Nyx -> brahe-py -> brahe-rs)
LANGUAGES = ["java", "gmat", "basilisk", "nyx", "python", "rust"]
LANGUAGE_LABELS = {
    "java": "Orekit",
    "gmat": "GMAT",
    "basilisk": "Basilisk",
    "nyx": "Nyx",
    "python": "Brahe (Python)",
    "rust": "Brahe (Rust)",
}


def _get_language_colors(theme: str) -> dict[str, str]:
    """Per-language colors: Orekit=CNES blue, GMAT=NASA red, Brahe Python=Star-Trek orange, Brahe Rust=green, Basilisk=CU Boulder gold, Nyx=magenta."""
    colors = get_theme_colors(theme)
    return {
        "java": colors["primary"],
        "gmat": colors["quinary"],
        "python": colors["secondary"],
        "rust": colors["accent"],
        "basilisk": colors["quaternary"],
        "nyx": colors["tertiary"],
    }


def _task_label(task_name: str) -> str:
    """Convert task_name like 'time.utc_to_tai' to 'UTC to TAI'."""
    # Strip module prefix
    _, _, task = task_name.partition(".")
    # Custom labels for tasks where the automatic title-casing produces
    # awkward results (high-fidelity propagation and force-model tasks).
    custom_labels = {
        "numerical_rk4_grav5x5": "RK4 + 5x5 Gravity",
        "numerical_rk4_grav20x20_sun_moon": "RK4 + 20x20 + Sun/Moon",
        "numerical_rk4_grav80x80_full": "RK4 + 80x80 + Drag + SRP",
        "accel_point_mass_gravity": "Point Mass Gravity",
        "accel_spherical_harmonics_20": "Spherical Harmonics 20x20",
        "accel_spherical_harmonics_80": "Spherical Harmonics 80x80",
        "accel_third_body_sun": "Third Body (Sun)",
        "accel_third_body_moon": "Third Body (Moon)",
    }
    if task in custom_labels:
        return custom_labels[task]
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


_UNIT_FACTORS = {"ns": 1e9, "\u00b5s": 1e6, "ms": 1e3, "s": 1.0}


def _pick_unit(seconds: list[float]) -> str:
    """Pick a display unit from a set of times (seconds), based on the mean."""
    nonzero = [t for t in seconds if t > 0]
    mean = sum(nonzero) / len(nonzero) if nonzero else 0
    if mean < 1e-6:
        return "ns"
    if mean < 1e-3:
        return "\u00b5s"
    if mean < 1.0:
        return "ms"
    return "s"


def _scale_times(seconds: list[float]) -> tuple[list[float], str]:
    """Scale times to appropriate display units."""
    unit = _pick_unit(seconds)
    factor = _UNIT_FACTORS[unit]
    return [t * factor for t in seconds], unit


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


# ── Accuracy JSONL loader ─────────────────────────────────────────────────


def load_accuracy_records() -> tuple[list[dict], list[dict]]:
    """Load summary and sample records from the latest accuracy JSONL run.

    Returns (summaries, samples). Returns ([], []) if no accuracy run has
    been performed yet, so the plotting pipeline degrades gracefully on
    repos that haven't run ``bench-compare-accuracy``.
    """
    accuracy_file = RESULTS_DIR / "accuracy_latest.jsonl"
    if not accuracy_file.exists():
        return [], []
    records = read_jsonl(accuracy_file)
    summaries = [r for r in records if r.get("kind") == "summary"]
    samples = [r for r in records if r.get("kind") == "sample"]
    return summaries, samples


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


def _speedup_figure(
    run: BenchmarkRun,
    *,
    baseline: str,
    baseline_label: str,
    other_langs: list[str],
    output_name: str,
    require_baseline_only: bool = False,
    figure_height: int = 1200,
):
    """Build a horizontal-bar speedup figure relative to a given baseline.

    Bars are added in the natural order — Plotly stacks them top-to-bottom
    per task, so the on-screen visual order matches `other_langs`.

    If require_baseline_only is True (used for vs-Basilisk), tasks where the
    baseline does not have data are dropped entirely rather than displayed
    with empty bars.

    figure_height should match the iframe height assigned in the docs CSS
    (.plotly-embed.{medium=600,tall=800,x-tall=1200}) so the chart fills its
    frame without internal scrolling or clipping.
    """
    modules = group_by_module(run)

    tasks = []
    for module in MODULE_ORDER:
        if module in modules:
            for task in sorted(modules[module].keys()):
                stats = modules[module][task]
                if require_baseline_only and baseline not in stats:
                    continue
                tasks.append(task)

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        fig = go.Figure()

        labels = [_task_label(t) for t in tasks]
        per_lang_speedups = {lang: [] for lang in other_langs}

        for task in tasks:
            module = task.split(".")[0]
            stats = modules[module][task]
            base_t = stats[baseline].mean if baseline in stats else 0
            for lang in other_langs:
                lang_t = stats[lang].mean if lang in stats else 0
                per_lang_speedups[lang].append(
                    base_t / lang_t if (base_t and lang_t) else 0
                )

        # Plotly stacks bars top-to-bottom within a group in the same order
        # they are added. Reverse the per-task list so the FIRST task in
        # `tasks` ends up at the TOP of the chart, but add bar series in
        # the order callers expect to read them (top → bottom).
        labels_rev = labels[::-1]
        for lang in reversed(other_langs):
            fig.add_trace(
                go.Bar(
                    name=LANGUAGE_LABELS[lang],
                    y=labels_rev,
                    x=per_lang_speedups[lang][::-1],
                    orientation="h",
                    marker_color=lang_colors[lang],
                    hovertemplate=(
                        f"%{{y}}<br>%{{x:.1f}}x vs {baseline_label}"
                        f"<extra>{LANGUAGE_LABELS[lang]}</extra>"
                    ),
                )
            )

        fig.add_vline(x=1, line_dash="dash", line_color="gray", line_width=1)

        fig.update_layout(
            title=f"Speedup vs {baseline_label}",
            xaxis_title="Speedup Factor (higher is faster)",
            xaxis_type="log",
            barmode="group",
            height=figure_height,
            margin=dict(l=200, b=60),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        return fig

    save_themed_html(make_fig, OUTDIR / output_name)


def make_speedup_figure(run: BenchmarkRun):
    """Generate the canonical speedup chart relative to the Orekit baseline.

    Orekit is the single performance reference; per-baseline charts against
    Basilisk and GMAT were removed in the benchmark accuracy redesign because
    they are derivable from the Orekit-relative numbers and invited "winner"
    framing the docs explicitly disavow.
    """
    _speedup_figure(
        run,
        baseline="java",
        baseline_label="Java (OreKit 12.2)",
        other_langs=["gmat", "basilisk", "nyx", "python", "rust"],
        output_name="fig_bench_speedup",
        figure_height=1200,  # matches .plotly-embed.x-tall
    )


# ── Figure: Per-module grouped bar charts ──────────────────────────────────


def make_module_figure(run: BenchmarkRun, module: str):
    """Generate grouped bar chart for a single module."""
    modules = group_by_module(run)
    if module not in modules:
        print(f"  Skipping {module}: no results")
        return

    task_data = modules[module]
    task_names = _order_tasks(module, task_data.keys())

    # Pick a single display unit across all languages so the shared Y-axis
    # isn't lying about Java/Python magnitudes when Rust is fast enough to
    # land in a smaller unit bucket.
    all_means = [
        task_data[t][lang].mean
        for t in task_names
        for lang in LANGUAGES
        if lang in task_data[t]
    ]
    unit = _pick_unit(all_means)
    factor = _UNIT_FACTORS[unit]

    # Drop languages that have no data on any task in this module so the
    # legend doesn't carry a phantom entry. Basilisk participates in only
    # 14 of 32 tasks; without this filter, modules like force_model and
    # access still draw a "Basilisk" legend swatch even though no bar
    # ever appears for it.
    languages_present = [
        lang for lang in LANGUAGES if any(lang in task_data[t] for t in task_names)
    ]

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        fig = go.Figure()

        labels = [_task_label(t) for t in task_names]

        for lang in languages_present:
            means_raw = [
                task_data[t][lang].mean if lang in task_data[t] else 0
                for t in task_names
            ]
            stds_raw = [
                task_data[t][lang].std if lang in task_data[t] else 0
                for t in task_names
            ]
            scaled = [m * factor for m in means_raw]
            # Bars show ±1σ (one standard deviation of the per-iteration timing
            # distribution). Std is communicated up-front in the chart title so
            # the reader doesn't have to infer it from the bar widths.
            error_upper = [s * factor for s in stds_raw]
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
            title=f"{MODULE_LABELS[module]} Benchmark: Mean Execution Time (±1σ)",
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


# ── Accuracy figure generation ─────────────────────────────────────────────


def _accuracy_unit_label(module: str) -> tuple[float, str]:
    """Return (multiplier, axis label suffix) for accuracy errors in a module.

    Errors arrive in SI base units: meters for positions, m/s² for
    accelerations, radians for angles, seconds for time, dimensionless
    for attitude. Apply per-module unit normalization so the CDF axis is
    legible (e.g. "m" for positions, scientific for accelerations).
    """
    if module == "force_model":
        return 1.0, "m/s²"
    if module == "attitude":
        return 1.0, "dimensionless"
    if module == "time":
        return 1.0, "s"
    return 1.0, "m"


def make_accuracy_cdf_figure(module: str, samples: list[dict]) -> None:
    """Per-module error CDF rendered as a small-multiples grid.

    Each task gets its own subplot; within a subplot, one ECDF curve per
    comparison language (Brahe Python, Brahe Rust, GMAT, Basilisk) colored
    consistently with the rest of the docs. The previous "all curves in one
    panel" layout produced an unreadable 8-task × 4-language tangle for
    propagation; faceting per task keeps each panel comparable across
    languages without losing the ability to compare tasks side-by-side.

    Tasks whose samples are all identical (variance == 0) are skipped
    silently — these modules (force_model, access) compare a single fixed
    initial condition and can't meaningfully populate a distribution. The
    per-module CSV is the source of truth for those.
    """
    from plotly.subplots import make_subplots

    module_samples = [s for s in samples if s["task_name"].startswith(f"{module}.")]
    if not module_samples:
        return

    _, unit_label = _accuracy_unit_label(module)

    # Group samples by task → language → list of errors. Skip (task, lang)
    # pairs with zero variance; if all pairs in a task are zero-variance,
    # skip the whole task subplot.
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in module_samples:
        by_task[s["task_name"]][s["comparison_language"]].append(
            float(s["max_abs_error"])
        )

    drawable_tasks: list[str] = []
    for task_name, lang_to_vals in by_task.items():
        # A subplot is worth drawing if any language has at least 2 distinct
        # values to form a non-degenerate ECDF.
        has_variance = any(len(set(vals)) > 1 for vals in lang_to_vals.values())
        if has_variance:
            drawable_tasks.append(task_name)
    drawable_tasks = _order_tasks(module, drawable_tasks)

    if not drawable_tasks:
        # All samples are identical; the per-task table communicates this.
        # Do not emit a placeholder file — the docs page is responsible for
        # not embedding a CDF iframe when the module has no plottable data
        # (e.g. force_model). Returning here leaves the docs to fall back
        # to "no embed at all" rather than a placeholder card.
        return

    n_tasks = len(drawable_tasks)
    # Grid sizing rule: 2 columns for 1–4 tasks, 3 columns for 5+. The
    # height is then tuned to match the docs CSS iframe heights
    # (default 500, tall 800, x-tall 1200) so the figure fills its
    # iframe without leaving white space or being cropped.
    if n_tasks <= 4:
        n_cols = 2 if n_tasks > 1 else 1
    else:
        n_cols = 3
    n_rows = (n_tasks + n_cols - 1) // n_cols

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        subplot_titles = [_task_label(t) for t in drawable_tasks]
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.12,
            vertical_spacing=0.16,
        )

        # Track which languages have appeared so each one shows in the
        # legend exactly once across all subplots.
        seen_langs: set[str] = set()

        for idx, task_name in enumerate(drawable_tasks):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            lang_to_vals = by_task[task_name]
            for lang in sorted(lang_to_vals.keys()):
                vals = sorted(lang_to_vals[lang])
                n = len(vals)
                if n < 2 or len(set(vals)) == 1:
                    continue
                y_vals = [(i + 1) / n for i in range(n)]
                show_in_legend = lang not in seen_langs
                seen_langs.add(lang)
                fig.add_trace(
                    go.Scatter(
                        x=vals,
                        y=y_vals,
                        mode="lines",
                        name=LANGUAGE_LABELS.get(lang, lang),
                        legendgroup=lang,
                        showlegend=show_in_legend,
                        line=dict(color=lang_colors.get(lang, "#888"), width=2),
                        hovertemplate=(
                            f"{_task_label(task_name)}<br>"
                            f"{LANGUAGE_LABELS.get(lang, lang)}<br>"
                            "Error: %{x}<br>P(error ≤ x): %{y:.2%}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(
                type="log",
                title_text=f"Max abs error ({unit_label})",
                row=row,
                col=col,
            )
            fig.update_yaxes(
                range=[0, 1.05],
                tickformat=".0%",
                title_text="P(error ≤ x)" if col == 1 else "",
                row=row,
                col=col,
            )

        # Compute the figure height dynamically from the row count.
        # The docs page picks up the actual rendered height via the
        # postMessage handshake injected by ``brahe_theme.save_themed_html``
        # and resizes the iframe to fit, so this number doesn't have to
        # line up with a fixed iframe CSS class.
        #
        # Layout budget: ~110 px for the title + legend band at the
        # top, ~70 px for the X-axis title band at the bottom of each
        # row, and ~250 px of plot area per row. The plot area gets
        # bigger automatically when there are more rows because the
        # title/X-axis bands stay fixed.
        figure_height = 180 + 320 * n_rows

        # The title goes at the very top of the figure (container
        # coords, not paper coords) so it sits in the top margin
        # rather than competing with subplot content. The legend
        # goes between the title and the subplot grid — putting it
        # below the lowest row inevitably collides with that row's
        # X-axis tick labels and axis-title text (the previous
        # ``y=-0.02`` layout sandwiched the legend between them).
        # Top placement is also the convention for grids of subplots.
        fig.update_layout(
            title=dict(
                text=f"{MODULE_LABELS[module]} Accuracy: Error CDF vs Orekit",
                x=0.5,
                xanchor="center",
                y=0.985,
                yanchor="top",
                yref="container",
            ),
            height=figure_height,
            margin=dict(t=110, b=70, l=70, r=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.03,
                xanchor="center",
                x=0.5,
            ),
        )
        return fig

    save_themed_html(make_fig, OUTDIR / f"fig_bench_accuracy_{module}")


def make_accuracy_scatter_figure(task_name: str, samples: list[dict]) -> None:
    """Error-vs-sample-key scatter for a single task.

    Only emitted when at least one sample carries a non-empty
    ``sample_key`` with a single scalar; otherwise the CDF figure
    stands alone for this task.
    """
    task_samples = [s for s in samples if s["task_name"] == task_name]
    if not task_samples:
        return

    # Inspect first sample with a non-empty sample_key to pick an x-axis.
    x_field: str | None = None
    for s in task_samples:
        sk = s.get("sample_key") or {}
        if sk:
            # Pick the first numeric scalar key
            for k, v in sk.items():
                if isinstance(v, (int, float)):
                    x_field = k
                    break
        if x_field:
            break
    if x_field is None:
        return

    module = task_name.split(".")[0]
    _, unit_label = _accuracy_unit_label(module)

    # Group by comparison_language; each language is a scatter trace.
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for s in task_samples:
        sk = s.get("sample_key") or {}
        if x_field not in sk or not isinstance(sk[x_field], (int, float)):
            continue
        grouped[s["comparison_language"]].append(
            (float(sk[x_field]), float(s["max_abs_error"]))
        )

    if not grouped:
        return

    def make_fig(theme: str) -> go.Figure:
        lang_colors = _get_language_colors(theme)
        fig = go.Figure()
        for lang in sorted(grouped.keys()):
            pts = grouped[lang]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=LANGUAGE_LABELS.get(lang, lang),
                    marker=dict(color=lang_colors.get(lang, "#888"), size=6),
                    hovertemplate=(
                        f"{x_field}: %{{x}}<br>error: %{{y}}<extra>"
                        + LANGUAGE_LABELS.get(lang, lang)
                        + "</extra>"
                    ),
                )
            )
        fig.update_layout(
            title=f"{_task_label(task_name)}: Error vs {x_field.replace('_', ' ')}",
            xaxis_title=x_field.replace("_", " "),
            yaxis_title=f"Max abs error ({unit_label})",
            yaxis_type="log",
            height=420,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )
        return fig

    safe = task_name.replace(".", "_")
    save_themed_html(make_fig, OUTDIR / f"fig_bench_accuracy_{safe}_scatter")


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

    if module == "force_model":
        # Force-model errors are accelerations (m/s²), not positions.
        # Use scientific notation directly — typical magnitudes are 10⁻¹²
        # to 10⁻¹⁶ m/s², well below any nm/µm-style human-readable range.
        if value == 0.0:
            return "0 m/s²"
        return f"{value:.2e} m/s²"

    if module == "attitude":
        # Attitude comparisons are dimensionless: either quaternion-component
        # residuals (bounded by 2) or rotation-matrix Frobenius residuals
        # (bounded by ~2√2). Render in scientific notation when small,
        # decimal when human-scale.
        if value == 0.0:
            return "0"
        if value < 1e-6:
            return f"{value:.2e}"
        if value < 1.0:
            return f"{value:.3g}"
        return f"{value:.3f}"

    if module == "time":
        # Time-scale conversions yield residuals in seconds.
        if value == 0.0:
            return "0 s"
        if value < 1e-6:
            return f"{value * 1e9:.1f} ns"
        if value < 1e-3:
            return f"{value * 1e6:.2f} µs"
        if value < 1.0:
            return f"{value * 1e3:.3f} ms"
        return f"{value:.3f} s"

    # Position-based errors (meters) — select unit by magnitude.
    # Coordinates and orbits are typically nanometer-scale, but outliers
    # (e.g. GMAT's geodetic equatorial-radius offset → ~0.7 m) need to
    # escalate cleanly through µm/mm/m rather than printing "700 million nm".
    if module in ("coordinates", "orbits"):
        if value == 0.0:
            return "0 nm"
        if value < 1e-6:
            return f"{value * 1e9:.2f} nm"
        if value < 1e-3:
            return f"{value * 1e6:.2f} µm"
        if value < 1.0:
            return f"{value * 1e3:.2f} mm"
        return f"{value:.3f} m"

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
    """Format a comparison label.

    The benchmark vocabulary uses language tokens (``java``, ``python``,
    ``rust``) but the docs read more clearly when written in terms of
    *libraries*: Orekit is the Java library being measured against the
    brahe library (in its Python or Rust binding), GMAT, or Basilisk.
    """
    label = {
        "java": "Orekit",
        "gmat": "GMAT",
        "basilisk": "Basilisk",
        "nyx": "Nyx",
        "python": "Brahe (Python)",
        "rust": "Brahe (Rust)",
    }
    return f"{label.get(ref, ref)} vs {label.get(comp, comp)}"


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
    # Column order: Module, Tasks, then speedups in display order
    # (GMAT -> Basilisk -> Nyx -> Python -> Rust, relative to the Java baseline).
    overview_rows = []
    for module in MODULE_ORDER:
        if module not in modules:
            continue
        task_data = modules[module]
        task_count = len(task_data)
        py_speedups = []
        rs_speedups = []
        bsk_speedups = []
        gmat_speedups = []
        nyx_speedups = []
        for task_stats in task_data.values():
            java_t = task_stats["java"].mean if "java" in task_stats else 0
            py_t = task_stats["python"].mean if "python" in task_stats else 0
            rs_t = task_stats["rust"].mean if "rust" in task_stats else 0
            bsk_t = task_stats["basilisk"].mean if "basilisk" in task_stats else 0
            gmat_t = task_stats["gmat"].mean if "gmat" in task_stats else 0
            nyx_t = task_stats["nyx"].mean if "nyx" in task_stats else 0
            if py_t and java_t:
                py_speedups.append(java_t / py_t)
            if rs_t and java_t:
                rs_speedups.append(java_t / rs_t)
            if bsk_t and java_t:
                bsk_speedups.append(java_t / bsk_t)
            if gmat_t and java_t:
                gmat_speedups.append(java_t / gmat_t)
            if nyx_t and java_t:
                nyx_speedups.append(java_t / nyx_t)
        avg_py = sum(py_speedups) / len(py_speedups) if py_speedups else 0
        avg_rs = sum(rs_speedups) / len(rs_speedups) if rs_speedups else 0
        avg_bsk = sum(bsk_speedups) / len(bsk_speedups) if bsk_speedups else 0
        avg_gmat = sum(gmat_speedups) / len(gmat_speedups) if gmat_speedups else 0
        avg_nyx = sum(nyx_speedups) / len(nyx_speedups) if nyx_speedups else 0
        overview_rows.append(
            [
                MODULE_LABELS[module],
                str(task_count),
                _format_speedup(avg_gmat) if gmat_speedups else "—",
                _format_speedup(avg_bsk) if bsk_speedups else "—",
                _format_speedup(avg_nyx) if nyx_speedups else "—",
                _format_speedup(avg_py),
                _format_speedup(avg_rs),
            ]
        )
    _write_csv(
        OUTDIR / "bench_overview.csv",
        [
            "Module",
            "Tasks",
            "Avg GMAT Speedup",
            "Avg Basilisk Speedup",
            "Avg Nyx Speedup",
            "Avg Brahe (Python) Speedup",
            "Avg Brahe (Rust) Speedup",
        ],
        overview_rows,
    )

    # ── Per-module performance tables ──────────────────────────────────────
    # Column order: Task, then time columns and speedup columns in display
    # order (Java -> GMAT -> Basilisk -> Nyx -> Python -> Rust). Modules without
    # any GMAT-participating tasks omit GMAT columns; same for Basilisk and Nyx.
    for module in MODULE_ORDER:
        if module not in modules:
            continue
        task_data = modules[module]
        module_has_gmat = any("gmat" in task_data[t] for t in task_data)
        module_has_basilisk = any("basilisk" in task_data[t] for t in task_data)
        module_has_nyx = any("nyx" in task_data[t] for t in task_data)
        perf_rows = []
        for task_name in _order_tasks(module, task_data.keys()):
            stats = task_data[task_name]
            java_t = stats["java"].mean if "java" in stats else 0
            py_t = stats["python"].mean if "python" in stats else 0
            rs_t = stats["rust"].mean if "rust" in stats else 0
            bsk_t = stats["basilisk"].mean if "basilisk" in stats else 0
            gmat_t = stats["gmat"].mean if "gmat" in stats else 0
            nyx_t = stats["nyx"].mean if "nyx" in stats else 0
            py_speedup = java_t / py_t if py_t and java_t else 0
            rs_speedup = java_t / rs_t if rs_t and java_t else 0
            bsk_speedup = java_t / bsk_t if bsk_t and java_t else 0
            gmat_speedup = java_t / gmat_t if gmat_t and java_t else 0
            nyx_speedup = java_t / nyx_t if nyx_t and java_t else 0
            row = [_task_label(task_name), _format_time(java_t)]
            if module_has_gmat:
                row.append(_format_time(gmat_t) if gmat_t else "—")
            if module_has_basilisk:
                row.append(_format_time(bsk_t) if bsk_t else "—")
            if module_has_nyx:
                row.append(_format_time(nyx_t) if nyx_t else "—")
            row.extend([_format_time(py_t), _format_time(rs_t)])
            if module_has_gmat:
                row.append(_format_speedup(gmat_speedup) if gmat_speedup else "—")
            if module_has_basilisk:
                row.append(_format_speedup(bsk_speedup) if bsk_speedup else "—")
            if module_has_nyx:
                row.append(_format_speedup(nyx_speedup) if nyx_speedup else "—")
            row.extend([_format_speedup(py_speedup), _format_speedup(rs_speedup)])
            perf_rows.append(row)
        headers = ["Task", "Orekit"]
        if module_has_gmat:
            headers.append("GMAT")
        if module_has_basilisk:
            headers.append("Basilisk")
        if module_has_nyx:
            headers.append("Nyx")
        headers.extend(["Brahe (Python)", "Brahe (Rust)"])
        if module_has_gmat:
            headers.append("GMAT Speedup")
        if module_has_basilisk:
            headers.append("Basilisk Speedup")
        if module_has_nyx:
            headers.append("Nyx Speedup")
        headers.extend(["Brahe (Python) Speedup", "Brahe (Rust) Speedup"])
        _write_csv(
            OUTDIR / f"bench_perf_{module}.csv",
            headers,
            perf_rows,
        )

    # ── Per-module accuracy tables ─────────────────────────────────────────
    # Per-module fallback: if the accuracy JSONL has data for a module, use
    # the distribution-summary format; otherwise fall back to the legacy
    # single-sample AccuracyComparison format. This lets a partial accuracy
    # sweep (e.g. just one module rerun) coexist with full-suite legacy
    # data without one starving the other.
    accuracy_summaries, _ = load_accuracy_records()
    jsonl_by_module: dict[str, list[dict]] = defaultdict(list)
    for s in accuracy_summaries:
        jsonl_by_module[s["task_name"].split(".")[0]].append(s)

    legacy_by_module: dict[str, list] = defaultdict(list)
    for ac in run.accuracy_comparisons:
        legacy_by_module[ac.task_name.split(".")[0]].append(ac)

    # Access uses a richer per-sample metric set (contact counts +
    # per-window start/end residuals) populated by
    # ``Sgp4AccessTask.detailed_sample_metrics`` — route it to a
    # custom writer that reads the per-sample JSONL records rather
    # than the rolled-up summaries.
    _, all_samples = load_accuracy_records()

    for module in MODULE_ORDER:
        if module == "access" and any(
            s["task_name"].startswith("access.") for s in all_samples
        ):
            _write_accuracy_csv_access(all_samples)
        elif module in jsonl_by_module:
            _write_accuracy_csv_from_jsonl(module, jsonl_by_module[module])
        elif module in legacy_by_module:
            _write_accuracy_csv_legacy(module, legacy_by_module[module])


def _write_accuracy_csv_from_jsonl(module: str, summaries: list[dict]) -> None:
    """Per-module accuracy table built from the JSONL distribution summaries.

    Columns: Task, Comparison, Samples, p50/p95/p99/max Max Abs (all in
    module-appropriate units via :func:`_format_accuracy`).
    """
    task_order = _order_tasks(module, {s["task_name"] for s in summaries})
    task_rank = {name: i for i, name in enumerate(task_order)}
    rows = []
    for s in sorted(
        summaries,
        key=lambda x: (
            task_rank.get(x["task_name"], 1_000_000),
            x["comparison_language"],
        ),
    ):
        rows.append(
            [
                _task_label(s["task_name"]),
                _comparison_label(s["reference_language"], s["comparison_language"]),
                str(s["n_samples"]),
                _format_accuracy(float(s["max_abs_p50"]), module),
                _format_accuracy(float(s["max_abs_p95"]), module),
                _format_accuracy(float(s["max_abs_p99"]), module),
                _format_accuracy(float(s["max_abs_max"]), module),
            ]
        )
    _write_csv(
        OUTDIR / f"bench_accuracy_{module}.csv",
        [
            "Task",
            "Comparison",
            "Samples",
            "p50 Max Abs",
            "p95 Max Abs",
            "p99 Max Abs",
            "Max Abs",
        ],
        rows,
    )


def _write_accuracy_csv_access(samples: list[dict]) -> None:
    """Per-comparison breakdown for the access module.

    Reads the per-sample JSONL records (each represents one ground
    location's contact set vs OreKit) and groups by comparison language.
    For each comparison emits:

    - **Contacts found** by each backend (mean / max across locations).
    - **Contact count diff**: per-location ``|n_baseline - n_comparison|``
      (max across locations) — a per-location detection-mismatch count.
    - **Window start residual**: distribution across all matched windows
      at all locations (p50 / p95 / p99 / max).
    - **Window end residual**: same, for window end times.

    Time residuals come from ``Sgp4AccessTask.detailed_sample_metrics``
    via the sample_key dict written into each JSONL record.
    """
    by_lang: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        if s["task_name"] != "access.sgp4_access":
            continue
        by_lang[s["comparison_language"]].append(s)

    if not by_lang:
        return

    def _format_seconds(value: float) -> str:
        if value == 0.0:
            return "0 s"
        if value < 1e-6:
            return f"{value * 1e9:.1f} ns"
        if value < 1e-3:
            return f"{value * 1e6:.2f} µs"
        if value < 1.0:
            return f"{value * 1e3:.3f} ms"
        return f"{value:.3f} s"

    headers = [
        "Comparison",
        "Locations",
        "Contacts (Orekit)",
        "Contacts (Comparison)",
        "Contact Count Diff (max per loc)",
        "Start Err p50",
        "Start Err p95",
        "Start Err p99",
        "Start Err max",
        "End Err p50",
        "End Err p95",
        "End Err p99",
        "End Err max",
    ]
    rows = []
    for lang in sorted(by_lang.keys()):
        recs = by_lang[lang]
        n_loc = len(recs)
        n_base_total = sum(
            int(r.get("sample_key", {}).get("n_windows_baseline", 0)) for r in recs
        )
        n_comp_total = sum(
            int(r.get("sample_key", {}).get("n_windows_comparison", 0)) for r in recs
        )
        max_count_diff = max(
            int(r.get("sample_key", {}).get("window_count_diff", 0)) for r in recs
        )
        start_errs = [
            float(r.get("sample_key", {}).get("start_err_s_max", 0.0)) for r in recs
        ]
        end_errs = [
            float(r.get("sample_key", {}).get("end_err_s_max", 0.0)) for r in recs
        ]
        rows.append(
            [
                _comparison_label("java", lang),
                str(n_loc),
                str(n_base_total),
                str(n_comp_total),
                str(max_count_diff),
                _format_seconds(_percentile(start_errs, 50)),
                _format_seconds(_percentile(start_errs, 95)),
                _format_seconds(_percentile(start_errs, 99)),
                _format_seconds(max(start_errs) if start_errs else 0.0),
                _format_seconds(_percentile(end_errs, 50)),
                _format_seconds(_percentile(end_errs, 95)),
                _format_seconds(_percentile(end_errs, 99)),
                _format_seconds(max(end_errs) if end_errs else 0.0),
            ]
        )

    _write_csv(OUTDIR / "bench_accuracy_access.csv", headers, rows)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (matches accuracy.py's helper)."""
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (pct / 100.0) * (len(sorted_vals) - 1)
    import math as _math

    lower = int(_math.floor(pos))
    upper = int(_math.ceil(pos))
    if lower == upper:
        return sorted_vals[lower]
    fraction = pos - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * fraction


def _write_accuracy_csv_legacy(module: str, comparisons: list) -> None:
    """Fallback per-module accuracy CSV for a module that has no JSONL data
    yet. Uses the legacy single-sample format from the perf run's
    ``accuracy_comparisons`` list, but routed through the current
    :func:`_format_accuracy` so unit fixes (time → seconds, attitude →
    dimensionless, etc.) apply consistently.
    """
    acc_rows = []
    for ac in sorted(comparisons, key=lambda a: (a.task_name, a.comparison_language)):
        acc_rows.append(
            [
                _task_label(ac.task_name),
                _comparison_label(ac.reference_language, ac.comparison_language),
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
        print(f"Generating {module} performance chart...")
        make_module_figure(run, module)

    summaries, samples = load_accuracy_records()
    if samples:
        print(
            f"Generating accuracy figures from {len(samples)} sample records "
            f"across {len(summaries)} summaries..."
        )
        modules_with_samples = {s["task_name"].split(".")[0] for s in samples}
        for module in MODULE_ORDER:
            if module in modules_with_samples:
                make_accuracy_cdf_figure(module, samples)
        task_names = sorted({s["task_name"] for s in samples})
        for task_name in task_names:
            make_accuracy_scatter_figure(task_name, samples)
    else:
        print("No accuracy JSONL found; skipping accuracy figures.")

    print("Generating CSV tables...")
    generate_csv_tables(run)

    print(f"\nAll figures and tables written to {OUTDIR}/")
