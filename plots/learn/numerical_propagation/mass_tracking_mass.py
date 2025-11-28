"""
Mass Tracking - Mass Depletion Profile Plot

Generates a plot showing spacecraft mass depletion during a continuous thrust maneuver
using NumericalOrbitPropagator with additional_dynamics.
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

# Propagate and collect mass over time
prop.propagate_to(epoch + total_time)

# Sample trajectory using trajectory interpolation
traj = prop.trajectory
times = []
mass_vals = []
thrust_active = []

dt = 5.0  # 5 second samples
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    try:
        state = traj.interpolate(current_epoch)
        times.append(t / 60.0)  # Convert to minutes
        mass_vals.append(state[6])
        thrust_active.append(1 if burn_start <= t < burn_end else 0)
    except RuntimeError:
        pass  # Skip if interpolation fails

    t += dt

# Compute expected values
expected_fuel = mass_flow_rate * burn_duration
final_mass_expected = initial_mass - expected_fuel
delta_v_expected = specific_impulse * g0 * np.log(initial_mass / final_mass_expected)


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Mass profile
    fig.add_trace(
        go.Scatter(
            x=times,
            y=mass_vals,
            mode="lines",
            name="Spacecraft Mass",
            line=dict(color=colors["primary"], width=3),
        )
    )

    # Initial mass reference
    fig.add_hline(
        y=initial_mass,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Initial: {initial_mass:.0f} kg",
        annotation_position="top right",
    )

    # Final mass reference
    fig.add_hline(
        y=final_mass_expected,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Final: {final_mass_expected:.1f} kg",
        annotation_position="bottom right",
    )

    # Thrust phase shading
    burn_start_min = burn_start / 60.0
    burn_end_min = burn_end / 60.0
    fig.add_vrect(
        x0=burn_start_min,
        x1=burn_end_min,
        fillcolor=colors["secondary"],
        opacity=0.1,
        layer="below",
        line_width=0,
        annotation_text="Thrust On",
        annotation_position="top left",
    )

    # Burn start indicator
    fig.add_vline(
        x=burn_start_min,
        line_dash="dash",
        line_color=colors["accent"],
        line_width=2,
        annotation_text="Burn Start",
        annotation_position="top left",
    )

    # Burn end indicator
    fig.add_vline(
        x=burn_end_min,
        line_dash="dash",
        line_color=colors["error"],
        line_width=2,
        annotation_text="Burn End",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Mass Depletion Profile (F={thrust_force} N, Isp={specific_impulse} s)",
        xaxis_title="Time (min)",
        yaxis_title="Mass (kg)",
        showlegend=False,
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    # Add annotation with summary
    summary_text = (
        f"Fuel consumed: {expected_fuel:.2f} kg<br>"
        f"Mass flow rate: {mass_flow_rate * 1000:.2f} g/s<br>"
        f"Expected \u0394v: {delta_v_expected:.1f} m/s"
    )
    fig.add_annotation(
        x=0.98,
        y=0.5,
        xref="paper",
        yref="paper",
        text=summary_text,
        showarrow=False,
        font=dict(size=11),
        align="right",
        bordercolor="gray",
        borderwidth=1,
        borderpad=4,
        bgcolor="white" if theme == "light" else "#1e1e1e",
        opacity=0.9,
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
