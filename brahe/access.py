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

**Enums:**
- LookDirection: Left, Right, or Either relative to velocity vector
- AscDsc: Ascending, Descending, or Either pass type
"""

from brahe._brahe import (
    # Enums
    LookDirection,
    AscDsc,
    # Constraints
    ElevationConstraint,
    ElevationMaskConstraint,
    OffNadirConstraint,
    LocalTimeConstraint,
    LookDirectionConstraint,
    AscDscConstraint,
    # Constraint Composition
    ConstraintAll,
    ConstraintAny,
    ConstraintNot,
    # Locations
    PointLocation,
    PolygonLocation,
    PropertiesDict,
    # Access Properties
    AccessWindow,
    AccessProperties,
    AccessSearchConfig,
    # Property Computers
    AccessPropertyComputer,
    # Access Computation
    location_accesses,
)

__all__ = [
    # Enums
    "LookDirection",
    "AscDsc",
    # Constraints
    "ElevationConstraint",
    "ElevationMaskConstraint",
    "OffNadirConstraint",
    "LocalTimeConstraint",
    "LookDirectionConstraint",
    "AscDscConstraint",
    # Constraint Composition
    "ConstraintAll",
    "ConstraintAny",
    "ConstraintNot",
    # Locations
    "PointLocation",
    "PolygonLocation",
    "PropertiesDict",
    # Access Properties
    "AccessWindow",
    "AccessProperties",
    "AccessSearchConfig",
    # Property Computers
    "AccessPropertyComputer",
    # Access Computation
    "location_accesses",
]
