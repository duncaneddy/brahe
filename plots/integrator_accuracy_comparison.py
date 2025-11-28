# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Compares position error of different numerical integrators against analytical
Keplerian propagation for a highly elliptical orbit (HEO).

This visualization demonstrates the accuracy characteristics of RK4, RKF45, DP54,
and RKN1210 integrators over one orbital period.
"""

import os
import pathlib
import sys
import plotly.graph_objects as go
import numpy as np
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import save_themed_html, get_color_sequence

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Initialize Brahe
bh.initialize_eop()

# Define Molniya-type HEO orbit
# Semi-major axis for 12-hour period
a = 26554e3  # meters
e = 0.74  # eccentricity
i = np.radians(63.4)  # inclination (critical inclination)
omega = 0.0  # argument of perigee
Omega = 0.0  # RAAN
M0 = 0.0  # mean anomaly

# Convert to Cartesian state
oe = np.array([a, e, i, Omega, omega, M0])
initial_state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)

# Orbital period
period = 2 * np.pi * np.sqrt(a**3 / bh.GM_EARTH)


# Define two-body dynamics
def two_body_dynamics(t, state):
    """Simple two-body dynamics for integration."""
    mu = bh.GM_EARTH
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate([v, a])


# Analytical solution using Keplerian propagation
def analytical_solution(t):
    """Compute analytical Keplerian state at time t."""
    # Mean motion
    n = np.sqrt(bh.GM_EARTH / a**3)
    # Mean anomaly at time t
    M = M0 + n * t
    # Convert back to Cartesian
    oe_t = np.array([a, e, i, Omega, omega, M])
    return bh.state_koe_to_eci(oe_t, bh.AngleFormat.RADIANS)


# Integration parameters
t_start = 0.0
t_end = period  # One orbital period
output_interval = 60.0  # Output every 60 seconds

# Common configuration for adaptive integrators
abs_tol = 1e-10
rel_tol = 1e-9

# Create integrators
config_rk4 = bh.IntegratorConfig.fixed_step(step_size=60.0)
config_adaptive = bh.IntegratorConfig.adaptive(abs_tol=abs_tol, rel_tol=rel_tol)

integrator_rk4 = bh.RK4Integrator(6, two_body_dynamics, config=config_rk4)
integrator_rkf45 = bh.RKF45Integrator(6, two_body_dynamics, config=config_adaptive)
integrator_dp54 = bh.DP54Integrator(6, two_body_dynamics, config=config_adaptive)
integrator_rkn1210 = bh.RKN1210Integrator(6, two_body_dynamics, config=config_adaptive)


# Propagate with each integrator
def propagate(integrator, is_adaptive=True):
    """Propagate orbit and record states at output intervals."""
    times = []
    states = []
    errors = []

    t = t_start
    state = initial_state.copy()
    dt = 60.0  # Initial step guess
    next_output = 0.0

    while t < t_end:
        # Check if we should save output
        if t >= next_output:
            times.append(t)
            states.append(state.copy())

            # Compute error vs analytical solution
            analytical = analytical_solution(t)
            pos_error = np.linalg.norm(state[0:3] - analytical[0:3])
            errors.append(pos_error)

            next_output += output_interval

        # Integrate one step
        if is_adaptive:
            result = integrator.step(t, state, min(dt, t_end - t))
            t += result.dt_used
            state = result.state
            dt = result.dt_next
        else:
            # Fixed step
            dt_actual = min(dt, t_end - t)
            new_state = integrator.step(t, state, dt_actual)
            t += dt_actual
            state = new_state

    # Final output
    if t >= t_end - 1e-6:  # Handle floating point comparison
        times.append(t_end)
        analytical = analytical_solution(t_end)
        pos_error = np.linalg.norm(state[0:3] - analytical[0:3])
        errors.append(pos_error)

    return np.array(times), np.array(errors)


print("Propagating with RK4...")
times_rk4, errors_rk4 = propagate(integrator_rk4, is_adaptive=False)

print("Propagating with RKF45...")
times_rkf45, errors_rkf45 = propagate(integrator_rkf45, is_adaptive=True)

print("Propagating with DP54...")
times_dp54, errors_dp54 = propagate(integrator_dp54, is_adaptive=True)

print("Propagating with RKN1210...")
times_rkn1210, errors_rkn1210 = propagate(integrator_rkn1210, is_adaptive=True)


# Create figure with theme support
def create_figure(theme):
    """Create figure with theme-specific colors."""
    colors = get_color_sequence(theme, num_colors=4)

    fig = go.Figure()

    # Add traces for each integrator with custom hover templates
    # First trace includes time at top, others don't to avoid duplication
    fig.add_trace(
        go.Scatter(
            x=times_rk4 / 3600,  # Convert to hours
            y=errors_rk4,
            name="RK4 (Fixed, dt=60s)",
            mode="lines",
            line=dict(color=colors[0], width=2),
            hovertemplate="t=%{x:.2f} hours<br><b>RK4</b><br>Error: %{y:.2e} m<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times_rkf45 / 3600,
            y=errors_rkf45,
            name="RKF45 (Adaptive)",
            mode="lines",
            line=dict(color=colors[1], width=2),
            hovertemplate="<b>RKF45</b><br>Error: %{y:.2e} m<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times_dp54 / 3600,
            y=errors_dp54,
            name="DP54 (Adaptive)",
            mode="lines",
            line=dict(color=colors[2], width=2),
            hovertemplate="<b>DP54</b><br>Error: %{y:.2e} m<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times_rkn1210 / 3600,
            y=errors_rkn1210,
            name="RKN1210 (Adaptive)",
            mode="lines",
            line=dict(color=colors[3], width=2),
            hovertemplate="<b>RKN1210</b><br>Error: %{y:.2e} m<extra></extra>",
        )
    )

    # Configure layout
    fig.update_layout(
        title="Integrator Accuracy Comparison: HEO Orbit",
        xaxis_title="Time (hours)",
        yaxis_title="Position Error (m)",
        yaxis_type="log",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Configure axes - hide default x-value in hover since we show it in first trace
    fig.update_xaxes(title_text="Time (hours)", unifiedhovertitle=dict(text=""))
    fig.update_yaxes(title_text="Position Error (m)", type="log")

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
