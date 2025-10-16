"""
Orbits Module

Orbital mechanics, propagators, and Two-Line Element (TLE) handling.

This module provides:

**Keplerian Elements:**
- Orbital element conversions (semi-major axis, mean motion, period)
- Anomaly conversions (mean, eccentric, true)
- Periapsis and apoapsis calculations
- Sun-synchronous orbit calculations

**Propagators:**
- SGPPropagator: SGP4/SDP4 propagator for TLE-based orbit prediction
- KeplerianPropagator: Analytical Keplerian propagation

**Two-Line Element (TLE) Support:**
- TLE parsing and validation
- TLE line creation and manipulation
- NORAD ID handling (numeric and alpha-5 formats)
- Conversion between TLE and Keplerian elements

Standard orbital element order: [a, e, i, raan, argp, anomaly]
where anomaly is mean anomaly unless otherwise specified.
"""

from brahe._brahe import (
    # Propagators
    SGPPropagator,
    KeplerianPropagator,
    # Orbital element calculations
    orbital_period,
    orbital_period_general,
    mean_motion,
    mean_motion_general,
    semimajor_axis,
    semimajor_axis_general,
    semimajor_axis_from_orbital_period,
    semimajor_axis_from_orbital_period_general,
    # Apsis calculations
    perigee_velocity,
    periapsis_velocity,
    periapsis_distance,
    apogee_velocity,
    apoapsis_velocity,
    apoapsis_distance,
    # Special orbits
    sun_synchronous_inclination,
    # Anomaly conversions
    anomaly_eccentric_to_mean,
    anomaly_mean_to_eccentric,
    anomaly_true_to_eccentric,
    anomaly_eccentric_to_true,
    anomaly_true_to_mean,
    anomaly_mean_to_true,
    # TLE validation and parsing
    validate_tle_lines,
    validate_tle_line,
    calculate_tle_line_checksum,
    parse_norad_id,
    # NORAD ID conversions
    norad_id_numeric_to_alpha5,
    norad_id_alpha5_to_numeric,
    # TLE conversions
    keplerian_elements_from_tle,
    keplerian_elements_to_tle,
    create_tle_lines,
    epoch_from_tle,
)

__all__ = [
    # Propagators
    "SGPPropagator",
    "KeplerianPropagator",
    # Orbital element calculations
    "orbital_period",
    "orbital_period_general",
    "mean_motion",
    "mean_motion_general",
    "semimajor_axis",
    "semimajor_axis_general",
    "semimajor_axis_from_orbital_period",
    "semimajor_axis_from_orbital_period_general",
    # Apsis calculations
    "perigee_velocity",
    "periapsis_velocity",
    "periapsis_distance",
    "apogee_velocity",
    "apoapsis_velocity",
    "apoapsis_distance",
    # Special orbits
    "sun_synchronous_inclination",
    # Anomaly conversions
    "anomaly_eccentric_to_mean",
    "anomaly_mean_to_eccentric",
    "anomaly_true_to_eccentric",
    "anomaly_eccentric_to_true",
    "anomaly_true_to_mean",
    "anomaly_mean_to_true",
    # TLE validation and parsing
    "validate_tle_lines",
    "validate_tle_line",
    "calculate_tle_line_checksum",
    "parse_norad_id",
    # NORAD ID conversions
    "norad_id_numeric_to_alpha5",
    "norad_id_alpha5_to_numeric",
    # TLE conversions
    "keplerian_elements_from_tle",
    "keplerian_elements_to_tle",
    "create_tle_lines",
    "epoch_from_tle",
]
