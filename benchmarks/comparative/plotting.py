"""
Plotly chart generation for comparative benchmark results.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go

# Add plots/ to path for brahe_theme import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import get_color_sequence, save_themed_html

from benchmarks.comparative.config import FIGURES_DIR
from benchmarks.comparative.results import BenchmarkRun


def _format_time_label(seconds: float) -> str:
    """Format time for axis labels."""
    if seconds < 1e-6:
        return "ns"
    elif seconds < 1e-3:
        return "\u00b5s"
    elif seconds < 1.0:
        return "ms"
    return "s"


def _scale_times(seconds: list[float]) -> tuple[list[float], str]:
    """Scale times to appropriate units. Returns (scaled_values, unit_label)."""
    mean = sum(seconds) / len(seconds) if seconds else 0
    if mean < 1e-6:
        return [t * 1e9 for t in seconds], "ns"
    elif mean < 1e-3:
        return [t * 1e6 for t in seconds], "\u00b5s"
    elif mean < 1.0:
        return [t * 1e3 for t in seconds], "ms"
    return seconds, "s"


def generate_performance_bar_chart(
    run: BenchmarkRun, output_base: Path | None = None
) -> Path:
    """Generate grouped bar chart of mean times per task, grouped by language."""
    output_base = output_base or FIGURES_DIR / "fig_comparative_benchmark_performance"

    # Group results by task
    tasks: dict[str, dict[str, float]] = {}
    for r in run.task_results:
        tasks.setdefault(r.task_name, {})[r.language] = r.mean

    task_names = sorted(tasks.keys())
    languages = sorted({r.language for r in run.task_results})

    def make_fig(theme: str) -> go.Figure:
        colors = get_color_sequence(theme, len(languages))
        fig = go.Figure()

        for i, lang in enumerate(languages):
            means = [tasks.get(t, {}).get(lang, 0) for t in task_names]
            scaled, unit = _scale_times(means)
            fig.add_trace(
                go.Bar(
                    name=lang,
                    x=task_names,
                    y=scaled,
                    marker_color=colors[i],
                )
            )

        fig.update_layout(
            title="Comparative Benchmark: Mean Execution Time",
            xaxis_title="Task",
            yaxis_title=f"Time ({unit})",
            barmode="group",
            legend_title="Language",
        )
        return fig

    light_path, dark_path = save_themed_html(make_fig, output_base)
    return light_path


def generate_timing_box_plot(
    run: BenchmarkRun, output_base: Path | None = None
) -> Path:
    """Generate box plot of timing distributions per language per task."""
    output_base = output_base or FIGURES_DIR / "fig_comparative_benchmark_distribution"

    def make_fig(theme: str) -> go.Figure:
        color_seq = get_color_sequence(theme)
        languages = sorted({r.language for r in run.task_results})
        lang_colors = {
            lang: color_seq[i % len(color_seq)] for i, lang in enumerate(languages)
        }

        fig = go.Figure()
        for r in sorted(run.task_results, key=lambda x: (x.task_name, x.language)):
            scaled, unit = _scale_times(r.times_seconds)
            fig.add_trace(
                go.Box(
                    y=scaled,
                    name=f"{r.task_name}\n({r.language})",
                    marker_color=lang_colors[r.language],
                    boxmean=True,
                )
            )

        fig.update_layout(
            title="Comparative Benchmark: Timing Distributions",
            yaxis_title=f"Time ({unit})",
            showlegend=False,
        )
        return fig

    light_path, dark_path = save_themed_html(make_fig, output_base)
    return light_path


def generate_all_plots(run: BenchmarkRun) -> list[Path]:
    """Generate all comparison charts from a benchmark run."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(generate_performance_bar_chart(run))
    paths.append(generate_timing_box_plot(run))
    return paths
