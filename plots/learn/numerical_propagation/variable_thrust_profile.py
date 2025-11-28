"""
Variable Thrust Profile Plot

Generates a plot showing the thrust magnitude over time during a variable thrust
maneuver, demonstrating the trapezoidal thrust profile with ramp-up and ramp-down
phases.
"""

import os
import pathlib
import sys
import numpy as np
import plotly.graph_objects as go

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from brahe_theme import save_themed_html, get_theme_colors

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Maneuver parameters (matching the variable_thrust.py example)
max_thrust = 0.5  # N (500 mN thruster)
mass = 500.0  # kg
ramp_time = 300.0  # s (5 minute ramp)
burn_duration = 1800.0  # s (30 minute burn)
maneuver_start = 600.0  # s (10 minutes into propagation)


def thrust_profile(t):
    """Calculate thrust magnitude at time t (seconds from propagation start)."""
    t_maneuver = t - maneuver_start

    if t_maneuver < 0 or t_maneuver > burn_duration:
        return 0.0
    elif t_maneuver < ramp_time:
        # Ramp up phase
        return max_thrust * (t_maneuver / ramp_time)
    elif t_maneuver > burn_duration - ramp_time:
        # Ramp down phase
        return max_thrust * ((burn_duration - t_maneuver) / ramp_time)
    else:
        # Constant thrust phase
        return max_thrust


# Generate time series data
total_time = 3600.0  # 1 hour total propagation
dt = 5.0  # 5 second intervals for smooth curve

times = np.arange(0, total_time + dt, dt)
thrust_values = np.array([thrust_profile(t) for t in times])
accel_values = thrust_values / mass * 1e6  # Convert to micro-m/s^2

# Convert times to minutes for display
times_min = times / 60.0

# Key times for annotations (in minutes)
maneuver_start_min = maneuver_start / 60.0
ramp_end_min = (maneuver_start + ramp_time) / 60.0
constant_end_min = (maneuver_start + burn_duration - ramp_time) / 60.0
maneuver_end_min = (maneuver_start + burn_duration) / 60.0


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Thrust magnitude trace
    fig.add_trace(
        go.Scatter(
            x=times_min,
            y=thrust_values * 1000,  # Convert to mN
            mode="lines",
            name="Thrust",
            line=dict(color=colors["primary"], width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(int(colors['primary'].lstrip('#')[i : i + 2], 16) for i in (0, 2, 4)) + [0.2])}",
        )
    )

    # Phase annotations
    fig.add_annotation(
        x=(maneuver_start_min + ramp_end_min) / 2,
        y=max_thrust * 1000 * 0.5,
        text="Ramp Up",
        showarrow=False,
        font=dict(size=11, color=colors["font_color"]),
    )

    fig.add_annotation(
        x=(ramp_end_min + constant_end_min) / 2,
        y=max_thrust * 1000 * 1.1,
        text="Constant Thrust",
        showarrow=False,
        font=dict(size=11, color=colors["font_color"]),
    )

    fig.add_annotation(
        x=(constant_end_min + maneuver_end_min) / 2,
        y=max_thrust * 1000 * 0.5,
        text="Ramp Down",
        showarrow=False,
        font=dict(size=11, color=colors["font_color"]),
    )

    # Vertical lines marking phase boundaries
    for x_val, label in [
        (maneuver_start_min, "Burn Start"),
        (maneuver_end_min, "Burn End"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dot",
            line_color=colors["secondary"],
            annotation_text=label,
            annotation_position="top",
        )

    # Max thrust reference line
    fig.add_hline(
        y=max_thrust * 1000,
        line_dash="dash",
        line_color=colors["accent"],
        annotation_text=f"Max: {max_thrust * 1000:.0f} mN",
        annotation_position="right",
    )

    fig.update_layout(
        title="Variable Thrust Profile: Trapezoidal Maneuver",
        xaxis_title="Time (minutes)",
        yaxis_title="Thrust (mN)",
        showlegend=False,
        height=500,
        margin=dict(l=60, r=80, t=60, b=60),
        yaxis=dict(range=[-20, max_thrust * 1000 * 1.3]),
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
