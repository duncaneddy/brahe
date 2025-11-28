"""
Simple Harmonic Oscillator Plot

Generates a plot showing position and velocity over time for a simple harmonic
oscillator propagated with NumericalPropagator.
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

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Simple Harmonic Oscillator parameters
omega = 2.0 * np.pi  # 1 Hz oscillation frequency
x0 = 1.0  # 1 meter initial displacement
v0 = 0.0  # Starting from rest
initial_state = np.array([x0, v0])


def sho_dynamics(t, state, params):
    """Simple harmonic oscillator dynamics."""
    x, v = state[0], state[1]
    omega_sq = params[0] if params is not None else omega**2
    return np.array([v, -omega_sq * x])


# Parameters (omega^2)
params = np.array([omega**2])

# Create propagator
prop = bh.NumericalPropagator(
    epoch,
    initial_state,
    sho_dynamics,
    bh.NumericalPropagationConfig.default(),
    params,
)

# Propagate for 3 periods
period = 2 * np.pi / omega  # Period = 1 second
total_time = 3 * period
prop.propagate_to(epoch + total_time)

# Sample trajectory at high resolution
dt = 0.01  # 10 ms intervals
times = []
positions = []
velocities = []
positions_analytical = []
velocities_analytical = []

t = 0.0
while t <= total_time:
    state = prop.state(epoch + t)
    times.append(t)
    positions.append(state[0])
    velocities.append(state[1])

    # Analytical solution: x(t) = x0*cos(omega*t), v(t) = -x0*omega*sin(omega*t)
    positions_analytical.append(x0 * np.cos(omega * t))
    velocities_analytical.append(-x0 * omega * np.sin(omega * t))

    t += dt


def create_figure(theme):
    colors = get_theme_colors(theme)

    # Create subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Position vs Time", "Velocity vs Time"),
        vertical_spacing=0.15,
    )

    # Position trace (numerical)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=positions,
            mode="lines",
            name="Numerical",
            line=dict(color=colors["primary"], width=2),
            legendgroup="numerical",
        ),
        row=1,
        col=1,
    )

    # Position trace (analytical) - dashed
    fig.add_trace(
        go.Scatter(
            x=times,
            y=positions_analytical,
            mode="lines",
            name="Analytical",
            line=dict(color=colors["secondary"], width=2, dash="dash"),
            legendgroup="analytical",
        ),
        row=1,
        col=1,
    )

    # Velocity trace (numerical)
    fig.add_trace(
        go.Scatter(
            x=times,
            y=velocities,
            mode="lines",
            name="Numerical",
            line=dict(color=colors["primary"], width=2),
            legendgroup="numerical",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Velocity trace (analytical) - dashed
    fig.add_trace(
        go.Scatter(
            x=times,
            y=velocities_analytical,
            mode="lines",
            name="Analytical",
            line=dict(color=colors["secondary"], width=2, dash="dash"),
            legendgroup="analytical",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title="Simple Harmonic Oscillator (ω = 2π rad/s)",
        height=600,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    # Update x-axes
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    # Update y-axes
    fig.update_yaxes(title_text="Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Velocity (m/s)", row=2, col=1)

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
