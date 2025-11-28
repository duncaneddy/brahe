"""
Impulsive Maneuver Velocity Components Plot

Generates a plot showing spacecraft velocity components (vx, vy, vz) over time
during a Hohmann transfer, demonstrating the delta-v jumps from impulsive burns.
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

# Initial and target orbits
r1 = bh.R_EARTH + 400e3  # 400 km altitude
r2 = bh.R_EARTH + 800e3  # 800 km altitude

# Initial state (circular orbit)
oe_initial = np.array([r1, 0.0001, 0.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe_initial, bh.AngleFormat.DEGREES)

# Calculate Hohmann transfer parameters
v1_circular = np.sqrt(bh.GM_EARTH / r1)
a_transfer = (r1 + r2) / 2
v_perigee_transfer = np.sqrt(bh.GM_EARTH * (2 / r1 - 1 / a_transfer))
v_apogee_transfer = np.sqrt(bh.GM_EARTH * (2 / r2 - 1 / a_transfer))
v2_circular = np.sqrt(bh.GM_EARTH / r2)

dv1 = v_perigee_transfer - v1_circular
dv2 = v2_circular - v_apogee_transfer
transfer_time = np.pi * np.sqrt(a_transfer**3 / bh.GM_EARTH)

# Burn times
burn1_time_s = 1.0  # First burn at t=1s
burn2_time_s = burn1_time_s + transfer_time  # Second burn after half-transfer

# Calculate total propagation time
final_orbit_period = bh.orbital_period(r2)
total_time = burn2_time_s + final_orbit_period

# Use multi-stage propagation to avoid trajectory interpolation issues with events
# Stage 1: Initial orbit (t=0 to burn1)
prop1 = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)
prop1.propagate_to(epoch + burn1_time_s)

# Apply first burn
state_at_burn1 = prop1.current_state()
v = state_at_burn1[3:6]
v_hat = v / np.linalg.norm(v)
state_post_burn1 = state_at_burn1.copy()
state_post_burn1[3:6] += dv1 * v_hat

# Stage 2: Transfer orbit (burn1 to burn2)
epoch_burn1 = epoch + burn1_time_s
prop2 = bh.NumericalOrbitPropagator(
    epoch_burn1,
    state_post_burn1,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)
prop2.propagate_to(epoch_burn1 + transfer_time)

# Apply second burn
state_at_burn2 = prop2.current_state()
v = state_at_burn2[3:6]
v_hat = v / np.linalg.norm(v)
state_post_burn2 = state_at_burn2.copy()
state_post_burn2[3:6] += dv2 * v_hat

# Stage 3: Final circular orbit (burn2 onwards)
epoch_burn2 = epoch_burn1 + transfer_time
prop3 = bh.NumericalOrbitPropagator(
    epoch_burn2,
    state_post_burn2,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)
prop3.propagate_to(epoch_burn2 + final_orbit_period)

# Sample the trajectory at high resolution
times = []
vx_data = []
vy_data = []
vz_data = []
dt = 30.0  # 30 second intervals

t = 0.0
while t <= total_time:
    current_epoch = epoch + t

    # Determine which propagator to query
    if t < burn1_time_s:
        s = prop1.state_eci(current_epoch)
    elif t < burn2_time_s:
        s = prop2.state_eci(current_epoch)
    else:
        s = prop3.state_eci(current_epoch)

    times.append(t / 60.0)  # Convert to minutes
    vx_data.append(s[3] / 1e3)  # Convert to km/s
    vy_data.append(s[4] / 1e3)
    vz_data.append(s[5] / 1e3)
    t += dt

# Get burn times for vertical lines (in minutes)
burn1_time_min = burn1_time_s / 60.0
burn2_time_min = burn2_time_s / 60.0


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Velocity component traces
    fig.add_trace(
        go.Scatter(
            x=times,
            y=vx_data,
            mode="lines",
            name="vx",
            line=dict(color=colors["primary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=vy_data,
            mode="lines",
            name="vy",
            line=dict(color=colors["secondary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=vz_data,
            mode="lines",
            name="vz",
            line=dict(color=colors["accent"], width=2),
        )
    )

    # Burn 1 marker
    fig.add_vline(
        x=burn1_time_min,
        line_dash="dot",
        line_color=colors["error"],
        annotation_text="Burn 1",
        annotation_position="top left",
    )

    # Burn 2 marker
    fig.add_vline(
        x=burn2_time_min,
        line_dash="dot",
        line_color=colors["error"],
        annotation_text="Burn 2",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Hohmann Transfer: Velocity Components",
        xaxis_title="Time (minutes)",
        yaxis_title="Velocity (km/s)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
