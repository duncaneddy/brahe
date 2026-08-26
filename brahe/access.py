"""
Access Module

Access computation for satellite visibility and ground location analysis.

This module provides tools for computing when and how satellites can access ground
locations or imaging targets. Key features include:

**Constraints:**
- Elevation angle constraints (min/max elevation)
- Elevation mask constraints (azimuth-dependent elevation profiles)
- Off-nadir angle constraints (satellite pointing limitations)
- Local solar time constraints (daytime/nighttime access)
- Look direction constraints (left/right looking)
- Ascending/descending pass constraints
- Constraint composition (AND/OR/NOT logic)
- Custom constraint computers (user-defined constraint logic)

**Enums:**
- LookDirection: Left, Right, or Either relative to velocity vector
- AscDsc: Ascending, Descending, or Either pass type
"""

from brahe._brahe import (
    AccessConstraintComputer,
    AccessProperties,
    AccessPropertiesBuilder,
    AccessPropertiesView,
    # Property Computers
    AccessPropertyComputer,
    AccessSearchConfig,
    # Access Properties
    AccessWindow,
    AscDsc,
    AscDscConstraint,
    AzimuthConstraint,
    # Constraint Composition
    ConstraintAll,
    ConstraintAny,
    ConstraintNot,
    DopplerComputer,
    # Constraints
    ElevationConstraint,
    ElevationMaskConstraint,
    LocalTimeConstraint,
    # Enums
    LookDirection,
    LookDirectionConstraint,
    OffNadirConstraint,
    OrbitGeometryTessellator,
    # Tessellation
    OrbitGeometryTessellatorConfig,
    # Locations
    PointLocation,
    PolygonLocation,
    PropertiesDict,
    RangeComputer,
    RangeConstraint,
    RangeRateComputer,
    SamplingConfig,
    SubdivisionConfig,
    get_max_threads,
    # Access Computation
    location_accesses,
    # Threading
    set_max_threads,
    tile_merge_orbit_geometry,
)

__all__ = [
    "AccessConstraintComputer",
    "AccessProperties",
    "AccessPropertiesBuilder",
    "AccessPropertiesView",
    # Property Computers
    "AccessPropertyComputer",
    "AccessSearchConfig",
    # Access Properties
    "AccessWindow",
    "AscDsc",
    "AscDscConstraint",
    "AzimuthConstraint",
    # Constraint Composition
    "ConstraintAll",
    "ConstraintAny",
    "ConstraintNot",
    "DopplerComputer",
    # Constraints
    "ElevationConstraint",
    "ElevationMaskConstraint",
    "LocalTimeConstraint",
    # Enums
    "LookDirection",
    "LookDirectionConstraint",
    "OffNadirConstraint",
    "OrbitGeometryTessellator",
    # Tessellation
    "OrbitGeometryTessellatorConfig",
    # Locations
    "PointLocation",
    "PolygonLocation",
    "PropertiesDict",
    "RangeComputer",
    "RangeConstraint",
    "RangeRateComputer",
    "SamplingConfig",
    "SubdivisionConfig",
    "get_max_threads",
    # Access Computation
    "location_accesses",
    # Threading
    "set_max_threads",
    "tile_merge_orbit_geometry",
]
