"""
Reusable Plotly theming utilities for Brahe documentation plots.

Provides consistent light and dark theme styling that matches the Material for MkDocs theme.
"""

import plotly.graph_objects as go
import plotly.io as pio


# Theme color definitions
LIGHT_THEME = {
    "primary": "#1f77b4",  # Blue
    "secondary": "#ff7f0e",  # Orange
    "accent": "#2ca02c",  # Green
    "error": "#d62728",  # Red
    "grid_color": "LightGrey",
    "line_color": "Grey",
    "font_color": "black",
    "bg_color": "white",
}

DARK_THEME = {
    "primary": "#5599ff",  # Lighter blue for dark mode
    "secondary": "#ffaa44",  # Lighter orange for dark mode
    "accent": "#55cc55",  # Lighter green for dark mode
    "error": "#ff6b6b",  # Lighter red for dark mode
    "grid_color": "#444444",
    "line_color": "#666666",
    "font_color": "#e0e0e0",
    "bg_color": "#1c1e24",  # Dark background to match Material slate theme
}


def get_theme_colors(theme="light"):
    """
    Get color palette for the specified theme.

    Args:
        theme (str): Either "light" or "dark"

    Returns:
        dict: Color palette with keys for primary, secondary, accent, error,
              grid_color, line_color, font_color, and bg_color
    """
    return DARK_THEME if theme == "dark" else LIGHT_THEME


def apply_brahe_theme(fig, theme="light", show_grid=True):
    """
    Apply Brahe theme styling to a Plotly figure.

    Args:
        fig (go.Figure): Plotly figure to style
        theme (str): Either "light" or "dark"
        show_grid (bool): Whether to show grid lines

    Returns:
        go.Figure: The figure with theme applied (modifies in-place and returns for chaining)
    """
    colors = get_theme_colors(theme)

    # Update layout with theme colors
    fig.update_layout(
        paper_bgcolor=colors["bg_color"],
        plot_bgcolor=colors["bg_color"],
        font=dict(color=colors["font_color"]),
        legend=dict(font=dict(color=colors["font_color"]), bgcolor="rgba(0,0,0,0)"),
    )

    # Update x-axis styling
    fig.update_xaxes(
        title_font=dict(color=colors["font_color"]),
        tickfont=dict(color=colors["font_color"]),
        showgrid=show_grid,
        gridwidth=1,
        gridcolor=colors["grid_color"],
        showline=True,
        linewidth=2,
        linecolor=colors["line_color"],
        zeroline=False,
    )

    # Update y-axis styling
    fig.update_yaxes(
        title_font=dict(color=colors["font_color"]),
        tickfont=dict(color=colors["font_color"]),
        showgrid=show_grid,
        gridwidth=1,
        gridcolor=colors["grid_color"],
        showline=True,
        linewidth=2,
        linecolor=colors["line_color"],
        zeroline=False,
    )

    return fig


def save_themed_html(fig_generator, outfile_base, custom_css=None):
    """
    Save figure as both light and dark themed HTML files.

    Args:
        fig_generator (callable or go.Figure): Either a function that takes theme ("light"/"dark")
                                                and returns a figure, or a figure object to theme
        outfile_base (Path or str): Base output path without _light/_dark suffix or .html extension
        custom_css (str, optional): Additional CSS to inject into the HTML

    Returns:
        tuple: (light_path, dark_path) - Paths to the generated files

    Example:
        # Option 1: Pass a generator function
        def create_fig(theme):
            colors = get_theme_colors(theme)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3], line=dict(color=colors["primary"])))
            return fig

        save_themed_html(create_fig, "output")

        # Option 2: Pass a figure (theme will be applied automatically)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3]))
        save_themed_html(fig, "output")
    """
    from pathlib import Path

    outfile_base = Path(outfile_base)
    outdir = outfile_base.parent
    filename = outfile_base.stem

    # Default CSS to remove body margins/padding
    if custom_css is None:
        custom_css = """
<style>
body {
    margin: 0;
    padding: 0;
    overflow: hidden;
}
</style>
"""

    # Determine if fig_generator is callable or a figure
    is_callable = callable(fig_generator)

    # Generate light theme version
    if is_callable:
        fig_light = fig_generator("light")
    else:
        fig_light = go.Figure(fig_generator)  # Copy figure
    apply_brahe_theme(fig_light, "light")
    light_path = outdir / f"{filename}_light.html"
    html_light = pio.to_html(
        fig_light, include_plotlyjs="cdn", full_html=False, auto_play=False
    )
    html_light = custom_css + html_light
    with open(light_path, "w") as f:
        f.write(html_light)

    # Generate dark theme version
    if is_callable:
        fig_dark = fig_generator("dark")
    else:
        fig_dark = go.Figure(fig_generator)  # Copy figure
    apply_brahe_theme(fig_dark, "dark")
    dark_path = outdir / f"{filename}_dark.html"
    html_dark = pio.to_html(
        fig_dark, include_plotlyjs="cdn", full_html=False, auto_play=False
    )
    html_dark = custom_css + html_dark
    with open(dark_path, "w") as f:
        f.write(html_dark)

    return light_path, dark_path


def get_color_sequence(theme="light", num_colors=None):
    """
    Get a sequence of colors for multi-line plots.

    Args:
        theme (str): Either "light" or "dark"
        num_colors (int, optional): Number of colors needed. If None, returns all available colors.

    Returns:
        list: List of color strings
    """
    colors = get_theme_colors(theme)
    sequence = [
        colors["primary"],
        colors["error"],
        colors["secondary"],
        colors["accent"],
    ]

    if num_colors is None:
        return sequence
    elif num_colors <= len(sequence):
        return sequence[:num_colors]
    else:
        # Repeat sequence if more colors needed
        return (sequence * ((num_colors // len(sequence)) + 1))[:num_colors]
