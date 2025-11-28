"""
Hohmann Transfer Orbit Diagram

Generates a top-down 2D view showing the Earth, initial orbit, transfer ellipse,
and final orbit for a Hohmann transfer maneuver.
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

# Earth radius in km for display
R_EARTH_KM = 6378.137

# Orbit parameters (matching the impulsive maneuver example)
r1_km = R_EARTH_KM + 400  # Initial orbit radius (400 km altitude)
r2_km = R_EARTH_KM + 800  # Final orbit radius (800 km altitude)

# Transfer orbit parameters
a_transfer_km = (r1_km + r2_km) / 2  # Semi-major axis of transfer ellipse
e_transfer = (r2_km - r1_km) / (r2_km + r1_km)  # Eccentricity of transfer ellipse


def generate_circle(radius, n_points=100):
    """Generate x, y coordinates for a circle."""
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return x, y


def generate_ellipse_arc(a, e, theta_start, theta_end, n_points=100):
    """Generate x, y coordinates for an ellipse arc.

    The ellipse is centered at one focus (Earth), with perigee at theta=0.
    """
    theta = np.linspace(theta_start, theta_end, n_points)
    # Orbit equation: r = a(1-e^2) / (1 + e*cos(theta))
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def create_figure(theme):
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Earth (filled circle)
    earth_x, earth_y = generate_circle(R_EARTH_KM, n_points=50)
    fig.add_trace(
        go.Scatter(
            x=earth_x,
            y=earth_y,
            mode="lines",
            fill="toself",
            fillcolor="#4a90d9" if theme == "light" else "#3a7bc8",
            line=dict(color="#2d5986", width=1),
            name="Earth",
            hoverinfo="name",
        )
    )

    # Initial orbit (dashed circle)
    initial_x, initial_y = generate_circle(r1_km)
    fig.add_trace(
        go.Scatter(
            x=initial_x,
            y=initial_y,
            mode="lines",
            line=dict(color=colors["secondary"], width=2, dash="dash"),
            name=f"Initial Orbit ({r1_km - R_EARTH_KM:.0f} km)",
            hoverinfo="name",
        )
    )

    # Final orbit (dashed circle)
    final_x, final_y = generate_circle(r2_km)
    fig.add_trace(
        go.Scatter(
            x=final_x,
            y=final_y,
            mode="lines",
            line=dict(color=colors["accent"], width=2, dash="dash"),
            name=f"Final Orbit ({r2_km - R_EARTH_KM:.0f} km)",
            hoverinfo="name",
        )
    )

    # Transfer orbit arc (solid line, only the transfer portion from perigee to apogee)
    transfer_x, transfer_y = generate_ellipse_arc(a_transfer_km, e_transfer, 0, np.pi)
    fig.add_trace(
        go.Scatter(
            x=transfer_x,
            y=transfer_y,
            mode="lines",
            line=dict(color=colors["primary"], width=3),
            name="Transfer Orbit",
            hoverinfo="name",
        )
    )

    # Burn 1 point (at perigee, rightmost point)
    burn1_x = r1_km
    burn1_y = 0
    fig.add_trace(
        go.Scatter(
            x=[burn1_x],
            y=[burn1_y],
            mode="markers",
            marker=dict(color=colors["error"], size=12, symbol="star"),
            name="Burn 1",
            hoverinfo="name+text",
            text=["Prograde burn to enter transfer orbit"],
        )
    )

    # Burn 2 point (at apogee, leftmost point)
    burn2_x = -r2_km
    burn2_y = 0
    fig.add_trace(
        go.Scatter(
            x=[burn2_x],
            y=[burn2_y],
            mode="markers",
            marker=dict(color=colors["error"], size=12, symbol="star"),
            name="Burn 2",
            hoverinfo="name+text",
            text=["Circularization burn at apogee"],
        )
    )

    # Annotations for burns
    fig.add_annotation(
        x=burn1_x + 300,
        y=burn1_y + 400,
        text="Burn 1",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=colors["error"],
        ax=40,
        ay=-30,
        font=dict(size=11, color=colors["font_color"]),
    )

    fig.add_annotation(
        x=burn2_x - 300,
        y=burn2_y + 400,
        text="Burn 2",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=colors["error"],
        ax=-40,
        ay=-30,
        font=dict(size=11, color=colors["font_color"]),
    )

    # Layout
    max_r = r2_km * 1.15
    fig.update_layout(
        title="Hohmann Transfer: Orbit Geometry (Top-Down View)",
        xaxis=dict(
            title="X (km)",
            range=[-max_r, max_r],
            scaleanchor="y",
            scaleratio=1,
            showgrid=True,
            gridcolor=colors["grid_color"],
            zeroline=True,
            zerolinecolor=colors["line_color"],
        ),
        yaxis=dict(
            title="Y (km)",
            range=[-max_r, max_r],
            showgrid=True,
            gridcolor=colors["grid_color"],
            zeroline=True,
            zerolinecolor=colors["line_color"],
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


# Save themed HTML files
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
