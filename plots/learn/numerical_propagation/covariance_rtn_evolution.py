"""
RTN Covariance Evolution Plot

Generates a plot showing how position uncertainty evolves in the RTN
(Radial-Tangential-Normal) frame during orbital propagation.
This frame provides physical insight into error behavior.
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
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Create propagation config with STM enabled and history storage
prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()

# Define initial covariance (diagonal)
# Position uncertainty: 10 m in each axis (100 m² variance)
# Velocity uncertainty: 0.01 m/s in each axis (0.0001 m²/s² variance)
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

# Create propagator with initial covariance
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    prop_config,
    bh.ForceModelConfig.two_body(),
    None,
    initial_covariance=P0,
)

# Propagate for 3 orbital periods
orbital_period = bh.orbital_period(oe[0])
total_time = 3 * orbital_period
prop.propagate_to(epoch + total_time)

# Sample RTN covariance evolution using STM-based propagation
# This avoids numerical issues with covariance interpolation
times = []  # in orbital periods
sigma_r = []  # Radial std dev
sigma_t = []  # Tangential std dev
sigma_n = []  # Normal std dev

dt = orbital_period / 50  # 50 samples per orbit
t = 0.0
while t <= total_time:
    current_epoch = epoch + t
    stm = prop.stm_at(current_epoch)

    if stm is not None:
        # Propagate covariance in ECI: P(t) = STM @ P0 @ STM^T
        P_eci = stm @ P0 @ stm.T

        # Get state at current epoch to compute RTN rotation
        state_t = prop.state(current_epoch)
        if state_t is not None:
            # Compute RTN rotation matrix from ECI state
            r = state_t[:3]
            v = state_t[3:]

            # RTN basis vectors
            r_hat = r / np.linalg.norm(r)  # Radial
            h = np.cross(r, v)  # Angular momentum
            n_hat = h / np.linalg.norm(h)  # Normal (cross-track)
            t_hat = np.cross(n_hat, r_hat)  # Tangential (along-track)

            # Rotation matrix from ECI to RTN (for position)
            R_eci_to_rtn = np.array([r_hat, t_hat, n_hat])

            # Transform position covariance to RTN
            P_pos_eci = P_eci[:3, :3]
            P_pos_rtn = R_eci_to_rtn @ P_pos_eci @ R_eci_to_rtn.T

            times.append(t / orbital_period)
            sigma_r.append(np.sqrt(P_pos_rtn[0, 0]))
            sigma_t.append(np.sqrt(P_pos_rtn[1, 1]))
            sigma_n.append(np.sqrt(P_pos_rtn[2, 2]))

    t += dt


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # RTN uncertainty traces
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sigma_r,
            mode="lines",
            name="Radial (R)",
            line=dict(color=colors["primary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sigma_t,
            mode="lines",
            name="Tangential (T)",
            line=dict(color=colors["secondary"], width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sigma_n,
            mode="lines",
            name="Normal (N)",
            line=dict(color=colors["accent"], width=2),
        )
    )

    # Add annotations for physical interpretation
    fig.add_annotation(
        x=2.5,
        y=sigma_t[-1] * 0.9,
        text="Along-track: unbounded growth",
        showarrow=False,
        font=dict(size=10),
    )

    fig.add_annotation(
        x=2.5,
        y=sigma_r[-1] * 1.5,
        text="Radial/Normal: bounded oscillation",
        showarrow=False,
        font=dict(size=10),
    )

    # Initial uncertainty reference
    initial_std = 10.0  # m
    fig.add_hline(
        y=initial_std,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Initial: {initial_std:.0f} m",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Position Uncertainty Evolution in RTN Frame",
        xaxis_title="Time (orbital periods)",
        yaxis_title="Position Std Dev (m)",
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
