#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "matplotlib", "numpy", "cartopy"]
# TIMEOUT = 600
# ///
"""
Collection Planning with Tessellation — Ireland + NISAR

End-to-end workflow: tessellate Ireland for satellite imaging using the
NISAR SAR satellite (242 km swath width), then compute collection opportunities.
Generates both light and dark themed figures.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import contextlib
import os
import pathlib
import time

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import brahe as bh

bh.initialize_eop()

OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Dark theme colors matching Material for MkDocs slate theme
DARK_BG = "#1c1e24"
DARK_LAND = "#3a3a3a"
DARK_OCEAN = "#2a2a3e"
DARK_COAST = "#666666"
DARK_BORDER = "#555555"
DARK_GRID_LABEL = "#cccccc"
DARK_OUTLINE = "#e0e0e0"
DARK_MARKER = "#ff6b6b"
# --8<-- [end:preamble]

# --8<-- [start:download_nisar]
# Download NISAR TLE from CelesTrak
# NISAR is a NASA/ISRO L-band + S-band SAR satellite with a 242 km swath width
print("Downloading NISAR TLE from CelesTrak...")
start_time = time.time()
client = bh.celestrak.CelestrakClient()
records = client.get_gp(name="NISAR")

nisar_prop = records[0].to_sgp_propagator(60.0)
elapsed = time.time() - start_time
print(f"Found NISAR: {nisar_prop.get_name()} in {elapsed:.2f}s")
# --8<-- [end:download_nisar]

# --8<-- [start:define_ireland]
# Approximate boundary of Ireland (whole island)
ireland_verts = [
    [-6.265, 52.178, 0],
    [-5.940, 53.040, 0],
    [-6.038, 53.552, 0],
    [-6.200, 53.930, 0],
    [-5.399, 54.355, 0],
    [-5.475, 54.676, 0],
    [-6.060, 55.273, 0],
    [-6.807, 55.235, 0],
    [-7.381, 55.439, 0],
    [-8.389, 55.155, 0],
    [-8.866, 54.676, 0],
    [-8.714, 54.582, 0],
    [-8.313, 54.594, 0],
    [-8.725, 54.330, 0],
    [-10.079, 54.361, 0],
    [-10.328, 53.987, 0],
    [-10.252, 53.403, 0],
    [-9.884, 53.125, 0],
    [-9.472, 52.910, 0],
    [-9.971, 52.568, 0],
    [-10.643, 52.105, 0],
    [-10.285, 51.516, 0],
    [-9.830, 51.387, 0],
    [-9.429, 51.367, 0],
    [-8.616, 51.522, 0],
    [-7.631, 51.952, 0],
    [-6.905, 52.085, 0],
]
ireland = bh.PolygonLocation(np.array(ireland_verts)).with_name("Ireland")
print(f"\nIreland polygon: {ireland.num_vertices} vertices")
# --8<-- [end:define_ireland]


def draw_tiles_on_ax(ax, tiles, color_by_group=True, color_cycle=None, alpha=0.4):
    """Draw tessellation tiles on a cartopy axis."""
    if color_cycle is None:
        color_cycle = plt.cm.Set2.colors
    group_map = {}
    for tile in tiles:
        verts = tile.vertices
        lons = [v[0] for v in verts] + [verts[0][0]]
        lats = [v[1] for v in verts] + [verts[0][1]]
        if color_by_group:
            gid = tile.properties.get("tile_group_id", "default")
            if gid not in group_map:
                group_map[gid] = color_cycle[len(group_map) % len(color_cycle)]
            color = group_map[gid]
        else:
            color = color_cycle[0]
        ax.add_patch(
            mpatches.Polygon(
                list(zip(lons, lats)),
                closed=True,
                facecolor=(*color[:3], alpha),
                edgecolor=(*color[:3], 0.8),
                linewidth=0.5,
                transform=ccrs.PlateCarree(),
            )
        )


def draw_polygon(ax, verts, **kwargs):
    """Draw polygon outline on a cartopy axis."""
    lons = [v[0] for v in verts] + [verts[0][0]]
    lats = [v[1] for v in verts] + [verts[0][1]]
    defaults = {"color": "k", "linestyle": "--", "linewidth": 1.5}
    defaults.update(kwargs)
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), **defaults)


def style_map_axis(ax, theme="light"):
    """Add coastlines, borders, and land/ocean features with theme-aware colors."""
    if theme == "dark":
        ax.set_facecolor(DARK_OCEAN)
        ax.add_feature(cfeature.LAND, facecolor=DARK_LAND, edgecolor="none")
        ax.coastlines(resolution="10m", linewidth=0.6, color=DARK_COAST)
        ax.add_feature(
            cfeature.BORDERS, linewidth=0.3, linestyle=":", edgecolor=DARK_BORDER
        )
    else:
        ax.add_feature(cfeature.LAND, facecolor="#e8e8e8", edgecolor="none")
        ax.add_feature(cfeature.OCEAN, facecolor="#cce5ff", edgecolor="none")
        ax.coastlines(resolution="10m", linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")


def style_gridlines(ax, theme="light"):
    """Add gridlines with theme-aware label colors."""
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    if theme == "dark":
        gl.xlabel_style = {"color": DARK_GRID_LABEL, "fontsize": 8}
        gl.ylabel_style = {"color": DARK_GRID_LABEL, "fontsize": 8}


def set_dark_figure_bg(fig):
    """Set dark background on figure and all axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(DARK_BG)


def save_themed(fig, name, theme):
    """Save figure with theme suffix."""
    suffix = "_light" if theme == "light" else "_dark"
    fig.savefig(OUTDIR / f"{name}{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --8<-- [start:tessellate]
# Tessellate Ireland using NISAR's 242 km swath width
# NISAR images in strips along its ground track, so we set the cross-track
# width to match the swath and use a long along-track length
config = bh.OrbitGeometryTessellatorConfig(
    image_width=242_000,  # 242 km cross-track (NISAR swath width)
    image_length=500_000,  # 500 km along-track
    asc_dsc=bh.AscDsc.EITHER,  # Both ascending and descending passes
)

tess = bh.OrbitGeometryTessellator(
    nisar_prop, nisar_prop.epoch, config, spacecraft_id=nisar_prop.get_name()
)
tiles = tess.tessellate_polygon(ireland)
print(f"\nNISAR tessellation: {len(tiles)} tiles")
for i, tile in enumerate(tiles):
    props = tile.properties
    print(
        f"  Tile {i}: width={props['tile_width'] / 1000:.0f} km, "
        f"length={props['tile_length'] / 1000:.0f} km"
    )
# --8<-- [end:tessellate]

# --8<-- [start:compute_accesses]
# Compute collection opportunities using off-nadir constraints
# NISAR operates at near-nadir to moderate off-nadir angles
print("\nComputing collection opportunities...")
start_time = time.time()

# Define time window: 7 days from the satellite epoch
epoch_start = nisar_prop.epoch
epoch_end = epoch_start + 7 * 86400.0

# Off-nadir constraint for SAR imaging
constraint = bh.OffNadirConstraint(
    min_off_nadir_deg=10.0,
    max_off_nadir_deg=45.0,
)

# Reset propagator before access computation
nisar_prop.reset()

# Compute access windows between tiles and NISAR
windows = bh.location_accesses(tiles, nisar_prop, epoch_start, epoch_end, constraint)

elapsed = time.time() - start_time
print(f"Found {len(windows)} collection windows in {elapsed:.2f}s")
# --8<-- [end:compute_accesses]

# --8<-- [start:results]
# Summarize results
if windows:
    durations = [w.duration for w in windows]
    print(f"\n{'=' * 70}")
    print("Collection Opportunity Summary (7-day period)")
    print(f"{'=' * 70}")
    print(f"  Total windows:     {len(windows)}")
    print(
        f"  Min duration:      {min(durations):.1f} s ({min(durations) / 60:.1f} min)"
    )
    print(
        f"  Max duration:      {max(durations):.1f} s ({max(durations) / 60:.1f} min)"
    )
    print(
        f"  Average duration:  {np.mean(durations):.1f} s ({np.mean(durations) / 60:.1f} min)"
    )
    print(
        f"  Median duration:   {np.median(durations):.1f} s ({np.median(durations) / 60:.1f} min)"
    )
    print(f"{'=' * 70}")

    # Show first 5 windows
    print(f"\n{'Start':<28} {'End':<28} {'Duration (s)':>12}")
    print("-" * 70)
    for w in windows[:5]:
        print(
            f"  {str(w.window_open):<28} {str(w.window_close):<28} {w.duration:>10.1f}"
        )
else:
    print("No collection windows found.")
# --8<-- [end:results]

# ============================================================================
# Generate themed figures
# ============================================================================

for theme in ("light", "dark"):
    outline_color = DARK_OUTLINE if theme == "dark" else "k"
    aoi_color = DARK_MARKER if theme == "dark" else "tab:red"
    ctx = (
        plt.style.context("dark_background")
        if theme == "dark"
        else contextlib.nullcontext()
    )

    with ctx:
        # --8<-- [start:plot_aoi]
        fig, ax = plt.subplots(
            1, 1, figsize=(8, 7), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)
        ax.set_extent([-11.5, -5.0, 51.0, 55.8], crs=ccrs.PlateCarree())
        style_map_axis(ax, theme)
        draw_polygon(ax, ireland_verts, color=aoi_color, linewidth=2, linestyle="-")
        ax.set_title("Ireland — Area of Interest", fontsize=11)
        style_gridlines(ax, theme)
        plt.tight_layout()
        save_themed(fig, "tessellation_ireland_aoi", theme)
        # --8<-- [end:plot_aoi]

        # --8<-- [start:plot_tessellation]
        fig, ax = plt.subplots(
            1, 1, figsize=(8, 7), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)
        ax.set_extent([-13, -3, 50, 57], crs=ccrs.PlateCarree())
        style_map_axis(ax, theme)
        draw_tiles_on_ax(ax, tiles)
        draw_polygon(ax, ireland_verts, color=outline_color)
        ax.set_title(
            f"Ireland tessellation — NISAR 242 km swath ({len(tiles)} tiles)",
            fontsize=10,
        )
        style_gridlines(ax, theme)
        plt.tight_layout()
        save_themed(fig, "tessellation_ireland_tiles", theme)
        # --8<-- [end:plot_tessellation]

    print(f"Generated all {theme} figures")
# --8<-- [end:all]

print("\nDone. All figures saved to", OUTDIR)
