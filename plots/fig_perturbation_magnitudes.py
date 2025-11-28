# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Compares the magnitude of various orbital perturbation forces across different altitude regimes.

This visualization shows how perturbation accelerations (gravity harmonics, third-body effects,
drag, solar radiation pressure, and relativistic corrections) vary with altitude from LEO to GEO.
"""

import os
import pathlib
import sys
import plotly.graph_objects as go
import numpy as np
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import save_themed_html, get_theme_colors

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Initialize Brahe
bh.initialize_eop()

# Initialize DE440s ephemeris for high-accuracy planetary positions
bh.initialize_ephemeris()

# Reference epoch for calculations (J2000)
epoch = bh.Epoch.from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Satellite parameters for drag and SRP
MASS = 500.0  # kg
AREA = 2.0  # m²
CD = 2.3  # Drag coefficient
CR = 1.8  # Radiation pressure coefficient

# Altitude range: 200 km to 40,000 km
altitudes_km = np.linspace(200, 40000, 500)
altitudes_m = altitudes_km * 1e3

# Storage for acceleration magnitudes
accel_point_mass = np.zeros_like(altitudes_m)
accel_j2 = np.zeros_like(altitudes_m)
accel_j22 = np.zeros_like(altitudes_m)
accel_sun = np.zeros_like(altitudes_m)
accel_moon = np.zeros_like(altitudes_m)
accel_venus = np.zeros_like(altitudes_m)
accel_jupiter = np.zeros_like(altitudes_m)
accel_saturn = np.zeros_like(altitudes_m)
accel_drag_mag = np.zeros_like(altitudes_m)
accel_srp_mag = np.zeros_like(altitudes_m)
accel_relativity_mag = np.zeros_like(altitudes_m)

print("Calculating perturbation accelerations...")

# Get Sun and Moon positions for reference epoch using DE ephemeris
r_sun = bh.sun_position_de(epoch, bh.EphemerisSource.DE440s)
r_moon = bh.moon_position_de(epoch, bh.EphemerisSource.DE440s)

# Identity rotation matrix for gravity calculations (assuming circular equatorial orbit for simplicity)
R_identity = np.eye(3)

# Load gravity model for harmonics calculations
gravity_model = bh.GravityModel.from_model_type(bh.GravityModelType.JGM3)

# Calculate accelerations at each altitude
for i, alt_m in enumerate(altitudes_m):
    r = bh.R_EARTH + alt_m

    # Position vector (assume circular equatorial orbit for reference)
    r_vec = np.array([r, 0.0, 0.0])

    # Orbital velocity for circular orbit
    v_circ = np.sqrt(bh.GM_EARTH / r)
    v_vec = np.array([0.0, v_circ, 0.0])
    state = np.concatenate([r_vec, v_vec])

    # Point mass gravity (for reference)
    a_pm = bh.accel_point_mass_gravity(r_vec, np.zeros(3), bh.GM_EARTH)
    accel_point_mass[i] = np.linalg.norm(a_pm)

    # Spherical harmonics (J2 only, n=2, m=0)
    a_sh_j2 = bh.accel_gravity_spherical_harmonics(
        r_vec, R_identity, gravity_model, 2, 0
    )
    accel_j2[i] = np.linalg.norm(a_sh_j2 - a_pm)  # Perturbation only

    # J22 component (n=2, m=2)
    a_sh_j22 = bh.accel_gravity_spherical_harmonics(
        r_vec, R_identity, gravity_model, 2, 2
    )
    accel_j22[i] = np.linalg.norm(a_sh_j22 - a_sh_j2)  # J22 perturbation only

    # Third-body: Sun (using DE ephemeris)
    a_sun = bh.accel_third_body_sun_de(epoch, r_vec, bh.EphemerisSource.DE440s)
    accel_sun[i] = np.linalg.norm(a_sun)

    # Third-body: Moon (using DE ephemeris)
    a_moon = bh.accel_third_body_moon_de(epoch, r_vec, bh.EphemerisSource.DE440s)
    accel_moon[i] = np.linalg.norm(a_moon)

    # Third-body: Venus (using DE ephemeris)
    a_venus = bh.accel_third_body_venus_de(epoch, r_vec, bh.EphemerisSource.DE440s)
    accel_venus[i] = np.linalg.norm(a_venus)

    # Third-body: Jupiter (using DE ephemeris)
    a_jupiter = bh.accel_third_body_jupiter_de(epoch, r_vec, bh.EphemerisSource.DE440s)
    accel_jupiter[i] = np.linalg.norm(a_jupiter)

    # Third-body: Saturn (using DE ephemeris)
    a_saturn = bh.accel_third_body_saturn_de(epoch, r_vec, bh.EphemerisSource.DE440s)
    accel_saturn[i] = np.linalg.norm(a_saturn)

    # Atmospheric drag (only significant below ~1000 km)
    if alt_m < 1000e3:
        # Use Harris-Priester atmospheric density model
        density = bh.density_harris_priester(r_vec, r_sun)

        # Only calculate drag if density is non-zero
        if density > 0.0:
            # Rotation matrix (identity for inertial frame)
            T_matrix = np.eye(3)

            a_drag = bh.accel_drag(state, density, MASS, AREA, CD, T_matrix)
            accel_drag_mag[i] = np.linalg.norm(a_drag)
        else:
            accel_drag_mag[i] = np.nan  # Below 100 km or above 1000 km
    else:
        accel_drag_mag[i] = np.nan  # Not plotted

    # Solar radiation pressure
    P0 = 4.56e-6  # N/m² at 1 AU
    a_srp = bh.accel_solar_radiation_pressure(r_vec, r_sun, MASS, CR, AREA, P0)
    accel_srp_mag[i] = np.linalg.norm(a_srp)

    # Relativistic effects
    a_rel = bh.accel_relativity(state)
    accel_relativity_mag[i] = np.linalg.norm(a_rel)

print("Calculations complete. Generating plots...")


# Create figure with theme support
def create_figure(theme):
    """Create figure with theme-specific colors."""
    theme_colors = get_theme_colors(theme)

    # Define category-specific colors
    if theme == "light":
        color_gravity = "#1f77b4"  # Blue - gravitational forces
        color_third_body = "#2ca02c"  # Green - third-body perturbations
        color_drag = "#d62728"  # Red - atmospheric drag
        color_srp = "#ffa500"  # Gold/Orange - solar radiation pressure
        color_relativity = "#9467bd"  # Purple - relativistic effects
    else:  # dark theme
        color_gravity = "#5599ff"  # Lighter blue
        color_third_body = "#55cc55"  # Lighter green
        color_drag = "#ff6b6b"  # Lighter red
        color_srp = "#ffcc44"  # Lighter gold
        color_relativity = "#bb88dd"  # Lighter purple

    fig = go.Figure()

    # Add shaded regions for orbital regimes
    # LEO: 200-2000 km (shaded red/purple - where drag dominates)
    fig.add_vrect(
        x0=200,
        x1=2000,
        fillcolor=color_drag,
        opacity=0.08,
        layer="below",
        line_width=0,
        annotation_text="LEO",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color=theme_colors["font_color"],
    )

    # MEO: 2000-35786 km (no shading, just annotation)
    fig.add_annotation(
        x=10000,  # Position in middle of MEO range
        y=1,
        yref="paper",
        text="MEO",
        showarrow=False,
        font=dict(size=11, color=theme_colors["font_color"]),
        yanchor="top",
        yshift=-10,
    )

    # GEO: 35786-40000 km (shaded gold - where SRP is significant)
    fig.add_vrect(
        x0=35786,
        x1=40000,
        fillcolor=color_srp,
        opacity=0.08,
        layer="below",
        line_width=0,
        annotation_text="GEO",
        annotation_position="top right",
        annotation_font_size=11,
        annotation_font_color=theme_colors["font_color"],
    )

    # ============================================================================
    # GRAVITATIONAL FORCES (Blue)
    # ============================================================================

    # Point mass gravity (reference)
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_point_mass,
            name="Point Mass Gravity",
            mode="lines",
            line=dict(color=color_gravity, width=2.5),
            hovertemplate="<b>Point Mass</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # J2 (oblateness)
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_j2,
            name="J₂ (Oblateness)",
            mode="lines",
            line=dict(color=color_gravity, width=2, dash="dash"),
            hovertemplate="<b>J₂</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # J22
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_j22,
            name="J₂₂",
            mode="lines",
            line=dict(color=color_gravity, width=1.5, dash="dot"),
            hovertemplate="<b>J₂₂</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # ============================================================================
    # THIRD-BODY PERTURBATIONS (Green)
    # ============================================================================

    # Third-body: Sun
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_sun,
            name="Third-Body (Sun)",
            mode="lines",
            line=dict(color=color_third_body, width=2.5),
            hovertemplate="<b>Sun</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # Third-body: Moon
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_moon,
            name="Third-Body (Moon)",
            mode="lines",
            line=dict(color=color_third_body, width=2.5, dash="dash"),
            hovertemplate="<b>Moon</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # Third-body: Venus
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_venus,
            name="Third-Body (Venus)",
            mode="lines",
            line=dict(color=color_third_body, width=1.5, dash="dash"),
            hovertemplate="<b>Venus</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # Third-body: Jupiter
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_jupiter,
            name="Third-Body (Jupiter)",
            mode="lines",
            line=dict(color=color_third_body, width=1.5, dash="dot"),
            hovertemplate="<b>Jupiter</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # Third-body: Saturn
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_saturn,
            name="Third-Body (Saturn)",
            mode="lines",
            line=dict(color=color_third_body, width=1.5, dash="dashdot"),
            hovertemplate="<b>Saturn</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # ============================================================================
    # ATMOSPHERIC DRAG (Red)
    # ============================================================================

    # Atmospheric drag
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_drag_mag,
            name="Atmospheric Drag",
            mode="lines",
            line=dict(color=color_drag, width=2.5),
            connectgaps=False,
            hovertemplate="<b>Drag</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # ============================================================================
    # SOLAR RADIATION PRESSURE (Gold)
    # ============================================================================

    # Solar radiation pressure
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_srp_mag,
            name="Solar Radiation Pressure",
            mode="lines",
            line=dict(color=color_srp, width=2),
            hovertemplate="<b>SRP</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # ============================================================================
    # RELATIVISTIC EFFECTS (Purple)
    # ============================================================================

    # Relativistic effects
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=accel_relativity_mag,
            name="Relativistic Effects",
            mode="lines",
            line=dict(color=color_relativity, width=1.5),
            hovertemplate="<b>Relativity</b><br>Altitude: %{x:.0f} km<br>Accel: %{y:.2e} m/s²<extra></extra>",
        )
    )

    # Configure layout
    fig.update_layout(
        title="Orbital Perturbation Force Magnitudes vs Altitude",
        xaxis_title="Altitude (km)",
        yaxis_title="Acceleration Magnitude (m/s²)",
        yaxis_type="log",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    fig.update_xaxes(title_text="Altitude (km)")
    fig.update_yaxes(title_text="Acceleration Magnitude (m/s²)", type="log")

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
