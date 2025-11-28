"""
Sensitivity Evolution Plot

Generates a plot showing how parameter sensitivity magnitude evolves over time
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

# Create initial epoch and state (LEO orbit with significant drag)
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 400e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagation config with sensitivity enabled and history storage
prop_config = (
    bh.NumericalPropagationConfig.default()
    .with_sensitivity()
    .with_sensitivity_history()
)

# Define spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator with full force model
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.default(),
    params=params,
)

# Propagate for 3 orbital periods
orbital_period = bh.orbital_period(oe[0])
total_time = 3 * orbital_period
prop.propagate_to(epoch + total_time)

# Sample sensitivity evolution
param_names = ["mass", "drag_area", "Cd", "srp_area", "Cr"]
times = []  # in orbital periods
sens_mag = {name: [] for name in param_names}  # Position sensitivity magnitude

dt = orbital_period / 50  # 50 samples per orbit
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    sens = prop.sensitivity_at(current_epoch)

    if sens is not None:
        times.append(t / orbital_period)  # Convert to orbital periods

        # Compute position sensitivity magnitude for each parameter
        for i, name in enumerate(param_names):
            pos_sens = np.linalg.norm(sens[:3, i])
            sens_mag[name].append(pos_sens)

    t += dt


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    color_map = {
        "mass": colors["primary"],
        "drag_area": colors["secondary"],
        "Cd": colors["accent"],
        "srp_area": colors["error"],
        "Cr": "gray",
    }

    # Add traces for each parameter
    for name in param_names:
        if sens_mag[name]:
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=sens_mag[name],
                    mode="lines",
                    name=name,
                    line=dict(color=color_map[name], width=2),
                )
            )

    fig.update_layout(
        title="Position Sensitivity to Parameters (LEO, 400 km)",
        xaxis_title="Time (orbital periods)",
        yaxis_title="Position Sensitivity (m per unit param)",
        yaxis_type="log",
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
