# /// script
# dependencies = ["brahe", "plotly", "kaleido"]
# ///
from pathlib import Path
import brahe as bh


bh.initialize_eop()

# Download TLE data for all GPS satellites from CelesTrak
propagators = bh.datasets.celestrak.get_tles_as_propagators("gps-ops", 60.0)

# Propagate each satellite one orbit
for prop in propagators:
    orbital_period = bh.orbital_period(prop.semi_major_axis)
    prop.propagate_to(prop.epoch + orbital_period)

# Create interactive 3D plot with Earth texture
fig = bh.plot_trajectory_3d(
    [
        {
            "trajectory": prop.trajectory,
            "mode": "markers",
            "size": 2,
            "label": prop.get_name(),
        }
        for prop in propagators
    ],
    units="km",
    show_earth=True,
    earth_texture="natural_earth_50m",
    backend="plotly",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=2.4,
)

# Enable grid and axis lines for better visualization
fig.update_layout(
    scene=dict(
        xaxis=dict(
            showbackground=False, gridcolor="lightgray", zerolinecolor="lightgray"
        ),
        yaxis=dict(
            showbackground=False, gridcolor="lightgray", zerolinecolor="lightgray"
        ),
        zaxis=dict(
            showbackground=False, gridcolor="lightgray", zerolinecolor="lightgray"
        ),
    )
)

# Disable the legend for clarity
fig.update_layout(showlegend=False)

# Remove title
fig.update_layout(title_text="")

# Save off plotly figure to svg and pdf files
fig.write_image(Path(__file__).parent / "gps_satellites.svg")
fig.write_image(Path(__file__).parent / "gps_satellites.pdf")
