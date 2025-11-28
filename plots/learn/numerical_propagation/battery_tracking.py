"""
Battery Tracking - Charge and Illumination Profile Plot

Generates a single combined plot showing battery state of charge and illumination
fraction during LEO orbit, demonstrating eclipse/sunlit cycles.
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

# Create initial epoch and state - LEO orbit
epoch = bh.Epoch.from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
orbital_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Extended state: [x, y, z, vx, vy, vz, battery_charge]
battery_capacity = 100.0  # Wh
initial_charge = 80.0  # Wh (80% SOC)
initial_state = np.concatenate([orbital_state, [initial_charge]])

# Power system parameters
solar_panel_power = 50.0  # W (when fully illuminated)
load_power = 30.0  # W (continuous consumption)

# Spacecraft parameters for force model
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])


# Define additional dynamics for battery tracking
def additional_dynamics(t, state, params):
    dx = np.zeros(len(state))
    r_eci = state[:3]

    # Get sun position at current epoch
    current_epoch = epoch + t
    r_sun = bh.sun_position(current_epoch)

    # Get illumination fraction (0 = umbra, 0-1 = penumbra, 1 = sunlit)
    illumination = bh.eclipse_conical(r_eci, r_sun)

    # Battery dynamics (Wh/s = W / 3600)
    power_in = illumination * solar_panel_power
    power_out = load_power
    charge_rate = (power_in - power_out) / 3600.0

    # Apply battery limits
    charge = state[6]
    if charge >= battery_capacity and charge_rate > 0:
        charge_rate = 0.0
    elif charge <= 0 and charge_rate < 0:
        charge_rate = 0.0

    dx[6] = charge_rate
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
)

# Propagate for 3 orbits
orbital_period = bh.orbital_period(oe[0])
num_orbits = 3
total_time = num_orbits * orbital_period

prop.propagate_to(epoch + total_time)

# Sample trajectory for plotting
traj = prop.trajectory
times = []
charge_vals = []
illumination_vals = []

dt = 10.0  # 10 second samples
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    try:
        state = traj.interpolate(current_epoch)
        r_eci = state[:3]
        r_sun = bh.sun_position(current_epoch)
        illumination = bh.eclipse_conical(r_eci, r_sun)

        times.append(t / 60.0)  # Convert to minutes
        charge_vals.append(state[6])
        illumination_vals.append(illumination)
    except RuntimeError:
        pass

    t += dt

# Find eclipse regions for shading
eclipse_starts = []
eclipse_ends = []
in_eclipse = False
eclipse_threshold = 0.01  # Consider <1% illumination as eclipse

for i, illum in enumerate(illumination_vals):
    if illum < eclipse_threshold and not in_eclipse:
        eclipse_starts.append(times[i])
        in_eclipse = True
    elif illum >= eclipse_threshold and in_eclipse:
        eclipse_ends.append(times[i])
        in_eclipse = False

# Close last eclipse if still in eclipse at end
if in_eclipse and len(eclipse_starts) > len(eclipse_ends):
    eclipse_ends.append(times[-1])

# Calculate statistics
final_charge = charge_vals[-1]
charge_change = final_charge - initial_charge
sunlit_time = sum(1 for i in illumination_vals if i > eclipse_threshold) * dt / 60.0
eclipse_time = len(times) * dt / 60.0 - sunlit_time


def create_figure(theme):
    colors = get_theme_colors(theme)

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Battery charge (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=charge_vals,
            mode="lines",
            name="Battery Charge",
            line=dict(color=colors["primary"], width=3),
        ),
        secondary_y=False,
    )

    # Illumination fraction (secondary y-axis) as filled area
    fig.add_trace(
        go.Scatter(
            x=times,
            y=illumination_vals,
            mode="lines",
            name="Illumination",
            line=dict(color=colors["secondary"], width=1),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.15)"
            if theme == "light"
            else "rgba(255, 170, 68, 0.15)",
        ),
        secondary_y=True,
    )

    # Add eclipse shading
    for start, end in zip(eclipse_starts, eclipse_ends):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(100, 100, 100, 0.2)"
            if theme == "light"
            else "rgba(50, 50, 50, 0.4)",
            layer="below",
            line_width=0,
        )

    # Add reference lines
    fig.add_hline(
        y=initial_charge,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Initial: {initial_charge:.0f} Wh",
        annotation_position="top right",
        secondary_y=False,
    )

    fig.add_hline(
        y=battery_capacity,
        line_dash="dot",
        line_color=colors["accent"],
        annotation_text=f"Capacity: {battery_capacity:.0f} Wh",
        annotation_position="bottom right",
        secondary_y=False,
    )

    # Update layout
    fig.update_layout(
        title=f"Battery Charge with Eclipse Cycles (LEO, {num_orbits} orbits)",
        xaxis_title="Time (min)",
        height=500,
        margin=dict(l=60, r=80, t=80, b=60),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
            if theme == "light"
            else "rgba(30,30,30,0.8)",
        ),
    )

    # Update y-axes
    fig.update_yaxes(
        title_text="Battery Charge (Wh)",
        range=[60, 105],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Illumination Fraction",
        range=[0, 1.1],
        secondary_y=True,
    )

    # Add summary annotation
    summary_text = (
        f"Charge change: {charge_change:+.2f} Wh<br>"
        f"Final SOC: {100 * final_charge / battery_capacity:.1f}%<br>"
        f"Eclipse: {eclipse_time:.1f} min ({100 * eclipse_time / (eclipse_time + sunlit_time):.0f}%)"
    )
    fig.add_annotation(
        x=0.98,
        y=0.02,
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

    # Add "Eclipse" label to first eclipse region
    if eclipse_starts:
        mid_eclipse = (eclipse_starts[0] + eclipse_ends[0]) / 2
        fig.add_annotation(
            x=mid_eclipse,
            y=0.95,
            xref="x",
            yref="paper",
            text="Eclipse",
            showarrow=False,
            font=dict(size=10, color=colors["font_color"]),
        )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
