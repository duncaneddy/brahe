"""
Mass Tracking - Orbital Elements Evolution Plot

Generates a plot showing how orbital elements evolve during a continuous thrust maneuver
with mass tracking using NumericalOrbitPropagator.
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
orbital_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Extended state: [x, y, z, vx, vy, vz, mass]
initial_mass = 1000.0  # kg
initial_state = np.concatenate([orbital_state, [initial_mass]])

# Thruster parameters
thrust_force = 10.0  # N
specific_impulse = 300.0  # s
g0 = 9.80665  # m/s^2
mass_flow_rate = thrust_force / (specific_impulse * g0)  # kg/s

# Timing parameters
pre_burn_coast = 300.0  # 5 minutes coast before burn
burn_duration = 600.0  # 10 minutes burn
post_burn_coast = 600.0  # 10 minutes coast after burn
burn_start = pre_burn_coast
burn_end = pre_burn_coast + burn_duration
total_time = pre_burn_coast + burn_duration + post_burn_coast

# Spacecraft parameters for force model
params = np.array([initial_mass, 2.0, 2.2, 2.0, 1.3])


# Define additional dynamics for mass tracking
def additional_dynamics(t, state, params):
    dx = np.zeros(len(state))
    if burn_start <= t < burn_end:
        dx[6] = -mass_flow_rate
    return dx


# Define control input for thrust acceleration
def control_input(t, state, params):
    dx = np.zeros(len(state))
    if burn_start <= t < burn_end:
        mass = state[6]
        vel = state[3:6]
        v_hat = vel / np.linalg.norm(vel)
        acc = (thrust_force / mass) * v_hat
        dx[3:6] = acc
    return dx


# Create propagator with two-body dynamics
force_config = bh.ForceModelConfig.two_body()
prop_config = bh.NumericalPropagationConfig.default()

prop = bh.NumericalOrbitPropagator(
    epoch,
    initial_state,
    prop_config,
    force_config,
    params=params,
    additional_dynamics=additional_dynamics,
    control_input=control_input,
)

# Propagate and collect orbital elements over time
prop.propagate_to(epoch + total_time)

# Sample trajectory using trajectory interpolation
traj = prop.trajectory
times = []
a_vals = []  # Semi-major axis (km)
e_vals = []  # Eccentricity
i_vals = []  # Inclination (deg)
raan_vals = []  # RAAN (deg)
argp_vals = []  # Argument of periapsis (deg)
ma_vals = []  # Mean anomaly (deg)

dt = 10.0  # 10 second samples
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    try:
        state = traj.interpolate(current_epoch)
        # Convert to orbital elements
        orbital_state_6d = state[:6]
        koe = bh.state_eci_to_koe(orbital_state_6d, bh.AngleFormat.DEGREES)

        times.append(t / 60.0)  # Convert to minutes
        a_vals.append((koe[0] - bh.R_EARTH) / 1e3)  # Altitude in km
        e_vals.append(koe[1])
        i_vals.append(koe[2])
        raan_vals.append(koe[3])
        argp_vals.append(koe[4])
        ma_vals.append(koe[5])  # Mean anomaly (koe[5] is mean anomaly)
    except RuntimeError:
        pass  # Skip if interpolation fails

    t += dt


def create_figure(theme):
    colors = get_theme_colors(theme)

    # Create 2x3 subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Semi-major Axis (Altitude)",
            "Eccentricity",
            "Inclination",
            "RAAN",
            "Arg. Periapsis",
            "Mean Anomaly",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )

    # Semi-major axis (altitude)
    fig.add_trace(
        go.Scatter(
            x=times, y=a_vals, mode="lines", line=dict(color=colors["primary"], width=2)
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)

    # Eccentricity
    fig.add_trace(
        go.Scatter(
            x=times,
            y=e_vals,
            mode="lines",
            line=dict(color=colors["secondary"], width=2),
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="e", range=[0, 0.1], row=1, col=2)

    # Inclination
    fig.add_trace(
        go.Scatter(
            x=times, y=i_vals, mode="lines", line=dict(color=colors["accent"], width=2)
        ),
        row=1,
        col=3,
    )
    fig.update_yaxes(title_text="i (deg)", range=[0, 90], row=1, col=3)

    # RAAN
    fig.add_trace(
        go.Scatter(
            x=times,
            y=raan_vals,
            mode="lines",
            line=dict(color=colors["primary"], width=2),
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="RAAN (deg)", range=[0, 360], row=2, col=1)

    # Argument of periapsis
    fig.add_trace(
        go.Scatter(
            x=times,
            y=argp_vals,
            mode="lines",
            line=dict(color=colors["secondary"], width=2),
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="\u03c9 (deg)", range=[0, 360], row=2, col=2)

    # Mean anomaly
    fig.add_trace(
        go.Scatter(
            x=times,
            y=ma_vals,
            mode="lines",
            line=dict(color=colors["accent"], width=2),
        ),
        row=2,
        col=3,
    )
    fig.update_yaxes(title_text="M (deg)", range=[0, 360], row=2, col=3)

    # Add burn start and end indicators to all subplots
    burn_start_min = burn_start / 60.0
    burn_end_min = burn_end / 60.0
    for row in [1, 2]:
        for col in [1, 2, 3]:
            # Burn start indicator
            fig.add_vline(
                x=burn_start_min,
                line_dash="dot",
                line_color=colors["accent"],
                line_width=1,
                row=row,
                col=col,
            )
            # Burn end indicator
            fig.add_vline(
                x=burn_end_min,
                line_dash="dot",
                line_color=colors["error"],
                line_width=1,
                row=row,
                col=col,
            )

    # Update x-axis labels for bottom row
    for col in [1, 2, 3]:
        fig.update_xaxes(title_text="Time (min)", row=2, col=col)

    fig.update_layout(
        title="Orbital Elements During Prograde Thrust (10 N, 10 min burn)",
        showlegend=False,
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
