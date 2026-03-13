"""
Tessellation Learn Page Figures

Generates all figures for the tessellation Learn documentation page.
Each figure illustrates one concept from the tessellation system.
Outputs both light and dark themed variants for each figure.

Usage:
    python plots/learn/access_computation/tessellation_figures.py
"""

import os
import pathlib

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import brahe as bh

bh.initialize_eop()

# Configuration
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# ISS TLE used throughout
LINE1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
LINE2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"

# SC-2 TLE with offset inclination (~53°) for merging demonstration
LINE2_SC2 = "2 25544  53.0000 247.4627 0006703 130.5360 325.0288 15.72125391563532"

# Dark theme colors matching Material for MkDocs slate theme
DARK_BG = "#1c1e24"
DARK_LAND = "#3a3a3a"
DARK_OCEAN = "#2a2a3e"
DARK_COAST = "#666666"
DARK_BORDER = "#555555"
DARK_GRID_LABEL = "#cccccc"
DARK_OUTLINE = "#e0e0e0"
DARK_MARKER = "#ff6b6b"


def draw_tiles(ax, tiles, color_by_group=True, color_cycle=None, alpha=0.4):
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


def draw_polygon_outline(ax, verts, **kwargs):
    """Draw a polygon outline on a cartopy axis."""
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


def save_themed(fig, name, theme):
    """Save figure with theme suffix."""
    suffix = "_light" if theme == "light" else "_dark"
    fig.savefig(OUTDIR / f"{name}{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def themed_context(theme):
    """Return a matplotlib style context for the given theme."""
    if theme == "dark":
        return plt.style.context("dark_background")

    # Light mode: use a null context manager
    import contextlib

    return contextlib.nullcontext()


def set_dark_figure_bg(fig):
    """Set dark background on figure and all axes."""
    fig.patch.set_facecolor(DARK_BG)
    for ax in fig.get_axes():
        ax.set_facecolor(DARK_BG)


# ============================================================================
# Pre-compute all tessellation data (theme-independent)
# ============================================================================

# --- Point Tessellation data ---
sf_point = bh.PointLocation(-122.4, 37.8, 0.0)

prop_asc = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_asc = bh.OrbitGeometryTessellatorConfig(
    image_width=5000, image_length=5000, asc_dsc=bh.AscDsc.ASCENDING
)
tess_asc = bh.OrbitGeometryTessellator(
    prop_asc, prop_asc.epoch, config_asc, spacecraft_id="ISS"
)
tiles_asc = tess_asc.tessellate_point(sf_point)

prop_either = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_either = bh.OrbitGeometryTessellatorConfig(
    image_width=5000, image_length=5000, asc_dsc=bh.AscDsc.EITHER
)
tess_either = bh.OrbitGeometryTessellator(
    prop_either, prop_either.epoch, config_either, spacecraft_id="ISS"
)
tiles_either = tess_either.tessellate_point(sf_point)

# --- Polygon Tessellation data ---
england_verts = [
    [-5.7, 50.0, 0],
    [-5.0, 50.0, 0],
    [1.8, 51.4, 0],
    [1.8, 52.5, 0],
    [0.0, 53.0, 0],
    [-1.0, 54.5, 0],
    [-3.0, 55.0, 0],
    [-3.4, 54.0, 0],
    [-5.0, 51.5, 0],
]
england = bh.PolygonLocation(np.array(england_verts))

prop_eng = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_eng = bh.OrbitGeometryTessellatorConfig(
    image_width=50000, image_length=50000, asc_dsc=bh.AscDsc.EITHER
)
tess_eng = bh.OrbitGeometryTessellator(
    prop_eng, prop_eng.epoch, config_eng, spacecraft_id="ISS"
)
tiles_eng = tess_eng.tessellate_polygon(england)

# --- Config comparison data ---
sf_verts = [
    [-122.5, 37.7, 0],
    [-122.3, 37.7, 0],
    [-122.3, 37.9, 0],
    [-122.5, 37.9, 0],
]
sf_polygon = bh.PolygonLocation(np.array(sf_verts))
sf_extent = [-122.55, -122.25, 37.67, 37.93]

prop_5k = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_5k = bh.OrbitGeometryTessellatorConfig(
    image_width=5000, image_length=5000, asc_dsc=bh.AscDsc.ASCENDING
)
tess_5k = bh.OrbitGeometryTessellator(
    prop_5k, prop_5k.epoch, config_5k, spacecraft_id="ISS"
)
tiles_5k = tess_5k.tessellate_polygon(sf_polygon)

prop_15k = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_15k = bh.OrbitGeometryTessellatorConfig(
    image_width=15000, image_length=15000, asc_dsc=bh.AscDsc.ASCENDING
)
tess_15k = bh.OrbitGeometryTessellator(
    prop_15k, prop_15k.epoch, config_15k, spacecraft_id="ISS"
)
tiles_15k = tess_15k.tessellate_polygon(sf_polygon)

# --- Overlap comparison data ---
prop_no_ol = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_no_ol = bh.OrbitGeometryTessellatorConfig(
    image_width=5000,
    image_length=5000,
    crosstrack_overlap=0,
    alongtrack_overlap=0,
    asc_dsc=bh.AscDsc.ASCENDING,
)
tess_no_ol = bh.OrbitGeometryTessellator(
    prop_no_ol, prop_no_ol.epoch, config_no_ol, spacecraft_id="ISS"
)
tiles_no_ol = tess_no_ol.tessellate_polygon(sf_polygon)

prop_ol = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
config_ol = bh.OrbitGeometryTessellatorConfig(
    image_width=5000,
    image_length=5000,
    crosstrack_overlap=1000,
    alongtrack_overlap=1000,
    asc_dsc=bh.AscDsc.ASCENDING,
)
tess_ol = bh.OrbitGeometryTessellator(
    prop_ol, prop_ol.epoch, config_ol, spacecraft_id="ISS"
)
tiles_ol = tess_ol.tessellate_polygon(sf_polygon)

# --- Merging data (SC-2 has offset inclination) ---
config_merge = bh.OrbitGeometryTessellatorConfig(
    image_width=5000, image_length=5000, asc_dsc=bh.AscDsc.ASCENDING
)

prop_sc1 = bh.SGPPropagator.from_tle(LINE1, LINE2, step_size=60.0)
tess_sc1 = bh.OrbitGeometryTessellator(
    prop_sc1, prop_sc1.epoch, config_merge, spacecraft_id="SC-1"
)

prop_sc2 = bh.SGPPropagator.from_tle(LINE1, LINE2_SC2, step_size=60.0)
tess_sc2 = bh.OrbitGeometryTessellator(
    prop_sc2, prop_sc2.epoch, config_merge, spacecraft_id="SC-2"
)

tiles_sc1 = tess_sc1.tessellate_polygon(sf_polygon)
tiles_sc2 = tess_sc2.tessellate_polygon(sf_polygon)
tiles_before = tiles_sc1 + tiles_sc2
tiles_after = bh.tile_merge_orbit_geometry(tiles_before, 200.0, 200.0, 2.0)


# ============================================================================
# Generate themed figures
# ============================================================================

for theme in ("light", "dark"):
    outline_color = DARK_OUTLINE if theme == "dark" else "k"
    marker_color = DARK_MARKER if theme == "dark" else "r"

    with themed_context(theme):
        # ------------------------------------------------------------------
        # --8<-- [start:point_figure]
        # Figure 1: Point Tessellation — ascending-only vs ascending+descending
        # ------------------------------------------------------------------

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)

        extent = [-122.5, -122.3, 37.72, 37.88]
        for ax in (ax1, ax2):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            style_map_axis(ax, theme)
            ax.plot(
                -122.4,
                37.8,
                "*",
                color=marker_color,
                markersize=12,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        draw_tiles(ax1, tiles_asc)
        ax1.set_title(f"Ascending only ({len(tiles_asc)} tile)", fontsize=10)

        draw_tiles(ax2, tiles_either)
        ax2.set_title(
            f"Ascending + Descending ({len(tiles_either)} tiles)", fontsize=10
        )

        for ax in (ax1, ax2):
            style_gridlines(ax, theme)

        plt.tight_layout()
        save_themed(fig, "tessellation_point", theme)
        # --8<-- [end:point_figure]

        # ------------------------------------------------------------------
        # --8<-- [start:polygon_figure]
        # Figure 2: Polygon Tessellation — England with tiles colored by group
        # ------------------------------------------------------------------

        fig, ax = plt.subplots(
            1, 1, figsize=(8, 7), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)

        ax.set_extent([-7, 3, 49, 56], crs=ccrs.PlateCarree())
        style_map_axis(ax, theme)
        draw_tiles(ax, tiles_eng)
        draw_polygon_outline(ax, england_verts, color=outline_color)
        ax.set_title(
            f"England — {len(tiles_eng)} tiles, colored by tile_group_id", fontsize=10
        )
        style_gridlines(ax, theme)
        plt.tight_layout()
        save_themed(fig, "tessellation_polygon", theme)
        # --8<-- [end:polygon_figure]

        # ------------------------------------------------------------------
        # --8<-- [start:config_figures]
        # Figure 3: Tile Length — 5 km vs 15 km
        # Figure 4: Overlap — 0 m vs 1000 m
        # ------------------------------------------------------------------

        # Tile Length
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)

        for ax in (ax1, ax2):
            ax.set_extent(sf_extent, crs=ccrs.PlateCarree())
            style_map_axis(ax, theme)

        draw_tiles(ax1, tiles_5k)
        draw_polygon_outline(ax1, sf_verts, color=outline_color)
        ax1.set_title(f"5 km tiles ({len(tiles_5k)} tiles)", fontsize=10)

        draw_tiles(ax2, tiles_15k)
        draw_polygon_outline(ax2, sf_verts, color=outline_color)
        ax2.set_title(f"15 km tiles ({len(tiles_15k)} tiles)", fontsize=10)

        for ax in (ax1, ax2):
            style_gridlines(ax, theme)

        plt.tight_layout()
        save_themed(fig, "tessellation_tile_length", theme)

        # Overlap
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)

        for ax in (ax1, ax2):
            ax.set_extent(sf_extent, crs=ccrs.PlateCarree())
            style_map_axis(ax, theme)

        draw_tiles(ax1, tiles_no_ol)
        draw_polygon_outline(ax1, sf_verts, color=outline_color)
        ax1.set_title(f"0 m overlap ({len(tiles_no_ol)} tiles)", fontsize=10)

        draw_tiles(ax2, tiles_ol)
        draw_polygon_outline(ax2, sf_verts, color=outline_color)
        ax2.set_title(f"1000 m overlap ({len(tiles_ol)} tiles)", fontsize=10)

        for ax in (ax1, ax2):
            style_gridlines(ax, theme)

        plt.tight_layout()
        save_themed(fig, "tessellation_overlap", theme)
        # --8<-- [end:config_figures]

        # ------------------------------------------------------------------
        # --8<-- [start:merging_figure]
        # Figure 5: Merging — before/after merge from two spacecraft
        # SC-2 has ~1.4° inclination offset so tiles visibly differ
        # ------------------------------------------------------------------

        sc1_color = (0.12, 0.47, 0.71)  # blue
        sc2_color = (1.0, 0.50, 0.05)  # orange
        merged_color = (0.17, 0.63, 0.17)  # green

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if theme == "dark":
            set_dark_figure_bg(fig)

        for ax in (ax1, ax2):
            ax.set_extent(sf_extent, crs=ccrs.PlateCarree())
            style_map_axis(ax, theme)

        # Before: SC-1 in blue, SC-2 in orange
        draw_tiles(ax1, tiles_sc1, color_by_group=False, color_cycle=[sc1_color])
        draw_tiles(ax1, tiles_sc2, color_by_group=False, color_cycle=[sc2_color])
        draw_polygon_outline(ax1, sf_verts, color=outline_color)
        ax1.set_title(f"Before merge ({len(tiles_before)} tiles)", fontsize=10)
        ax1.plot([], [], "s", color=sc1_color, label="SC-1", markersize=8)
        ax1.plot([], [], "s", color=sc2_color, label="SC-2", markersize=8)
        ax1.legend(loc="lower right", fontsize=8)

        # After: merged tiles in green
        draw_tiles(ax2, tiles_after, color_by_group=False, color_cycle=[merged_color])
        draw_polygon_outline(ax2, sf_verts, color=outline_color)
        ax2.set_title(f"After merge ({len(tiles_after)} tiles)", fontsize=10)
        multi_count = sum(
            1 for t in tiles_after if len(t.properties.get("spacecraft_ids", [])) > 1
        )
        ax2.plot(
            [],
            [],
            "s",
            color=merged_color,
            label=f"Merged (SC-1 + SC-2): {multi_count}",
            markersize=8,
        )
        ax2.legend(loc="lower right", fontsize=8)

        for ax in (ax1, ax2):
            style_gridlines(ax, theme)

        plt.tight_layout()
        save_themed(fig, "tessellation_merging", theme)
        # --8<-- [end:merging_figure]

    print(f"Generated all {theme} figures")

print("\nAll tessellation Learn page figures generated.")
