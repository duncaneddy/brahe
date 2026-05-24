"""
GPU comparison figures + CSV tables from a `benchmarks/gpu_comparison` run.

Reads the latest results JSON in `benchmarks/gpu_comparison/results/` (or a
specific file via `--input`) and emits:

  - One CSV per task (throughput per (config, batch_size), wide format).
  - One CSV `bench_gpu_speedup.csv` summarising peak speedup vs the
    `brahe-rust-rayon` baseline.
  - One throughput-vs-batch HTML per task (light + dark).
  - One peak-speedup bar chart HTML (light + dark).

Usage:
    BRAHE_FIGURE_OUTPUT_DIR=./docs/figures/ uv run python plots/fig_gpu_comparison.py
    uv run python plots/fig_gpu_comparison.py --input benchmarks/gpu_comparison/results/run_<id>.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import plotly.graph_objects as go

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "gpu_comparison" / "results"
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
OUTDIR.mkdir(parents=True, exist_ok=True)

# Stable display order + per-config colour key for the four configs the suite
# ships with.
CONFIG_ORDER = [
    "brahe-rust-rayon",
    "astrojax-cpu",
    "astrojax-gpu",
    "astrojax-multigpu",
]
BASELINE = "brahe-rust-rayon"


def _human(value: float) -> str:
    """Format a throughput / batch number with K/M/B/T suffixes.

    Examples:
        1             -> "1"
        10            -> "10"
        107000        -> "107K"
        1109000       -> "1.11M"
        3.142e8       -> "314M"
        1.52e9        -> "1.52B"
    Integer-valued inputs below 1000 are rendered without decimals.
    """
    if value is None:
        return ""
    abs_v = abs(value)
    is_int = float(value).is_integer()
    if abs_v >= 1e12:
        scaled, suffix = value / 1e12, "T"
    elif abs_v >= 1e9:
        scaled, suffix = value / 1e9, "B"
    elif abs_v >= 1e6:
        scaled, suffix = value / 1e6, "M"
    elif abs_v >= 1e3:
        scaled, suffix = value / 1e3, "K"
    else:
        # < 1000: keep integers integer, give floats 3 sig figs.
        if is_int:
            return f"{int(value)}"
        if abs_v >= 100 or abs_v == 0:
            return f"{value:.0f}"
        if abs_v >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"
    # Trim trailing ".0" so 1000 / 10000 read as "1K" / "10K" rather than
    # "1.0K" / "10.0K".
    if scaled == int(scaled):
        return f"{int(scaled)}{suffix}"
    # 3 significant figures within the chosen suffix.
    if abs(scaled) >= 100:
        return f"{scaled:.0f}{suffix}"
    if abs(scaled) >= 10:
        return f"{scaled:.1f}{suffix}"
    return f"{scaled:.2f}{suffix}"


def _config_color(cfg: str, colors: dict) -> str:
    return {
        "brahe-rust-rayon": colors["accent"],     # Brahe green
        "astrojax-cpu":     colors["primary"],    # blue
        "astrojax-gpu":     colors["secondary"],  # orange
        "astrojax-multigpu":colors["tertiary"],   # magenta
    }.get(cfg, colors["error"])


def _latest_results() -> pathlib.Path:
    candidates = sorted(RESULTS_DIR.glob("run_*.json"))
    if not candidates:
        raise SystemExit(f"no results found in {RESULTS_DIR}")
    return candidates[-1]


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def _cells_by_task(data: dict) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for cell in data["cells"]:
        by_task[cell["task"]].append(cell)
    return by_task


def _write_task_csv(task_name: str, cells: list[dict]) -> pathlib.Path:
    """One row per batch_size, one column per config — throughput (ops/s) or
    'skipped: <reason>' string."""
    batches = sorted({c["batch_size"] for c in cells})
    configs = [c for c in CONFIG_ORDER if any(x["config"] == c for x in cells)]
    by_pair: dict[tuple[str, int], dict] = {
        (c["config"], c["batch_size"]): c for c in cells
    }
    safe = task_name.replace(".", "_")
    path = OUTDIR / f"bench_gpu_{safe}.csv"
    with path.open("w") as f:
        # Column header: include the unit so the K/M/B suffixes are unambiguous.
        f.write("batch_size," + ",".join(f"{c} (ops/s)" for c in configs) + "\n")
        for b in batches:
            row = [_human(b)]
            for cfg in configs:
                cell = by_pair.get((cfg, b))
                if cell is None:
                    row.append("")
                elif cell["status"] == "ok":
                    row.append(_human(cell["throughput_ops_per_sec"]))
                else:
                    row.append(f"skipped:{cell.get('skip_reason', 'unknown')}")
            f.write(",".join(row) + "\n")
    return path


def _write_speedup_csv(by_task: dict[str, list[dict]]) -> pathlib.Path:
    """Peak speedup-vs-baseline per task per non-baseline config."""
    path = OUTDIR / "bench_gpu_speedup.csv"
    configs = [c for c in CONFIG_ORDER if c != BASELINE]
    with path.open("w") as f:
        f.write("task," + ",".join(f"{c}_peak_speedup" for c in configs) + "\n")
        for task in sorted(by_task.keys()):
            row = [task]
            for cfg in configs:
                vals = [
                    c.get("speedup_vs_baseline")
                    for c in by_task[task]
                    if c["config"] == cfg and c["status"] == "ok"
                    and c.get("speedup_vs_baseline") is not None
                ]
                row.append(f"{max(vals):.2f}x" if vals else "n/a")
            f.write(",".join(row) + "\n")
    return path


def _throughput_fig(task_name: str, cells: list[dict]):
    def _make(theme: str):
        colors = get_theme_colors(theme)
        fig = go.Figure()
        configs = [c for c in CONFIG_ORDER if any(x["config"] == c for x in cells)]
        for cfg in configs:
            cfg_cells = sorted(
                (c for c in cells if c["config"] == cfg and c["status"] == "ok"),
                key=lambda c: c["batch_size"],
            )
            if not cfg_cells:
                continue
            xs = [c["batch_size"] for c in cfg_cells]
            ys = [c["throughput_ops_per_sec"] for c in cfg_cells]
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode="lines+markers",
                name=cfg,
                line=dict(color=_config_color(cfg, colors), width=2),
                marker=dict(size=7),
                hovertemplate="batch=%{x}<br>throughput=%{y:.3e} ops/s<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text=task_name, x=0.5, xanchor="center"),
            xaxis=dict(title="batch size", type="log"),
            yaxis=dict(title="throughput (ops / s)", type="log"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.18),
            margin=dict(l=70, r=30, t=50, b=80),
            # Matches the docs CSS .plotly-embed iframe height (500px)
            height=500,
        )
        return fig
    return _make


def _peak_speedup_fig(by_task: dict[str, list[dict]]):
    tasks = sorted(by_task.keys())
    configs = [c for c in CONFIG_ORDER if c != BASELINE]
    peak = {cfg: [] for cfg in configs}
    for task in tasks:
        for cfg in configs:
            vals = [
                c.get("speedup_vs_baseline")
                for c in by_task[task]
                if c["config"] == cfg and c["status"] == "ok"
                and c.get("speedup_vs_baseline") is not None
            ]
            peak[cfg].append(max(vals) if vals else None)

    def _make(theme: str):
        colors = get_theme_colors(theme)
        fig = go.Figure()
        for cfg in configs:
            fig.add_trace(go.Bar(
                x=tasks, y=peak[cfg], name=cfg,
                marker=dict(color=_config_color(cfg, colors)),
                hovertemplate="%{x}<br>peak speedup=%{y:.2f}x<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text="Peak speedup vs brahe-rust-rayon",
                       x=0.5, xanchor="center"),
            barmode="group",
            xaxis=dict(title="task", tickangle=-25),
            yaxis=dict(title="peak speedup (×)", type="log"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.30),
            margin=dict(l=70, r=30, t=50, b=160),
            # Matches the docs CSS .plotly-embed.medium iframe height (600px)
            height=600,
        )
        # Parity reference line
        fig.add_hline(y=1.0, line=dict(color=colors["line_color"], dash="dash"))
        return fig
    return _make


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=pathlib.Path,
                    help="results JSON (default: most recent in benchmarks/gpu_comparison/results/)")
    args = ap.parse_args()
    src = args.input if args.input else _latest_results()
    print(f"reading {src}")
    data = _load(src)
    by_task = _cells_by_task(data)

    # Per-task CSV + throughput figure
    for task_name, cells in sorted(by_task.items()):
        csv = _write_task_csv(task_name, cells)
        print(f"  wrote {csv}")
        safe = task_name.replace(".", "_")
        light, dark = save_themed_html(
            _throughput_fig(task_name, cells),
            OUTDIR / f"fig_gpu_{safe}",
        )
        print(f"  wrote {light}")

    speedup_csv = _write_speedup_csv(by_task)
    print(f"  wrote {speedup_csv}")
    light, dark = save_themed_html(
        _peak_speedup_fig(by_task),
        OUTDIR / "fig_gpu_peak_speedup",
    )
    print(f"  wrote {light}")


if __name__ == "__main__":
    main()
