# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Brief description of what this plot visualizes.

This template shows the minimal structure for a documentation plot.
Replace this with actual visualization.
"""

import os
import pathlib
import sys
import plotly.graph_objects as go
import numpy as np

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)  # Replace with actual data


# Create figure with theme support
def create_figure(theme):
    """Create figure with theme-specific colors."""
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Add traces with theme colors
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            name="Data",
            mode="lines",
            line=dict(color=colors["primary"], width=2),
        )
    )

    # Configure axes (theme-agnostic settings)
    fig.update_xaxes(title_text="X Axis Label")
    fig.update_yaxes(title_text="Y Axis Label")

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
