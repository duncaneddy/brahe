"""
Covariance Evolution Plot

Generates a plot showing how position uncertainty (covariance) evolves over time
during orbital propagation.
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html, get_theme_colors

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagation config with STM enabled and history storage
prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()

# Create propagator (two-body for clean demonstration)
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.two_body(),
    None,
)

# Define initial covariance (diagonal)
# Position uncertainty: 10 m in each axis (100 m² variance)
# Velocity uncertainty: 0.01 m/s in each axis (0.0001 m²/s² variance)
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

# Propagate for 3 orbital periods
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
total_time = 3 * orbital_period
prop.propagate_to(epoch + total_time)

# Sample covariance evolution
times = []  # in orbital periods
pos_sigma_r = []  # Radial (x) std dev
pos_sigma_t = []  # Tangential (y) std dev
pos_sigma_n = []  # Normal (z) std dev
pos_total = []  # Total position std dev

dt = orbital_period / 50  # 50 samples per orbit
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    stm = prop.stm_at(current_epoch)

    if stm is not None:
        # Propagate covariance: P(t) = STM @ P0 @ STM^T
        P = stm @ P0 @ stm.T

        # Extract position standard deviations
        sigma_x = np.sqrt(P[0, 0])
        sigma_y = np.sqrt(P[1, 1])
        sigma_z = np.sqrt(P[2, 2])

        times.append(t / orbital_period)  # Convert to orbital periods
        pos_sigma_r.append(sigma_x)
        pos_sigma_t.append(sigma_y)
        pos_sigma_n.append(sigma_z)
        pos_total.append(np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2))

    t += dt


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Position uncertainty traces
    fig.add_trace(
        go.Scatter(
            x=times,
            y=pos_sigma_r,
            mode="lines",
            name="X (radial-like)",
            line=dict(color=colors["primary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=pos_sigma_t,
            mode="lines",
            name="Y (along-track-like)",
            line=dict(color=colors["secondary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=pos_sigma_n,
            mode="lines",
            name="Z (cross-track-like)",
            line=dict(color=colors["accent"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=pos_total,
            mode="lines",
            name="Total (RSS)",
            line=dict(color=colors["error"], width=2, dash="dash"),
        )
    )

    # Initial uncertainty reference
    initial_total = np.sqrt(3 * 100.0)  # sqrt(3 * 100 m²)
    fig.add_hline(
        y=initial_total,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Initial: {initial_total:.1f} m",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Position Uncertainty Evolution (Two-Body)",
        xaxis_title="Time (orbital periods)",
        yaxis_title="Position Std Dev (m)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
