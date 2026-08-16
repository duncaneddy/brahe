"""
Orbits Module

Orbital mechanics and Two-Line Element (TLE) handling.

This module provides:

**Keplerian Elements:**
- Orbital element conversions (semi-major axis, mean motion, period)
- Anomaly conversions (mean, eccentric, true)
- Periapsis and apoapsis calculations
- Sun-synchronous orbit calculations

**Two-Line Element (TLE) Support:**
- TLE parsing and validation
- TLE line creation and manipulation
- NORAD ID handling (numeric and alpha-5 formats)
- Conversion between TLE and Keplerian elements

Standard orbital element order: [a, e, i, raan, argp, anomaly]
where anomaly is mean anomaly unless otherwise specified.

Note: Orbit propagators (SGPPropagator, KeplerianPropagator) have been moved
to the `brahe.propagators` module.
"""

from brahe._brahe import (
    MeanElementInverseConfig,
    MeanElementMethod,
    MeanElementNumericalMethodConfig,
    WalkerConstellationGenerator,
    WalkerConstellationGeneratorBuilder,
    # Constellation generators
    WalkerPattern,
    WindowAlignment,
    WindowEdgeHandling,
    # Anomaly conversions
    anomaly_eccentric_to_mean,
    anomaly_eccentric_to_true,
    anomaly_mean_to_eccentric,
    anomaly_mean_to_true,
    anomaly_true_to_eccentric,
    anomaly_true_to_mean,
    apoapsis_altitude,
    apoapsis_distance,
    apoapsis_velocity,
    apogee_altitude,
    apogee_velocity,
    calculate_tle_line_checksum,
    create_tle_lines,
    epoch_from_tle,
    geo_sma,
    # TLE conversions
    keplerian_elements_from_tle,
    keplerian_elements_to_tle,
    mean_motion,
    mean_motion_general,
    norad_id_alpha5_to_numeric,
    # NORAD ID conversions
    norad_id_numeric_to_alpha5,
    # Orbital element calculations
    orbital_period,
    orbital_period_from_state,
    orbital_period_general,
    parse_norad_id,
    periapsis_altitude,
    periapsis_distance,
    periapsis_velocity,
    perigee_altitude,
    # Apsis calculations
    perigee_velocity,
    semimajor_axis,
    semimajor_axis_from_orbital_period,
    semimajor_axis_from_orbital_period_general,
    semimajor_axis_general,
    state_equinoctial_to_koe,
    state_koe_mean_to_osc,
    # Mean-osculating Keplerian element conversions
    state_koe_osc_to_mean,
    # Equinoctial element conversions
    state_koe_to_equinoctial,
    states_koe_mean_to_osc,
    states_koe_osc_to_mean,
    # Special orbits
    sun_synchronous_inclination,
    validate_tle_line,
    # TLE validation and parsing
    validate_tle_lines,
)

__all__ = [
    "MeanElementInverseConfig",
    "MeanElementMethod",
    "MeanElementNumericalMethodConfig",
    "WalkerConstellationGenerator",
    "WalkerConstellationGeneratorBuilder",
    # Constellation generators
    "WalkerPattern",
    "WindowAlignment",
    "WindowEdgeHandling",
    # Anomaly conversions
    "anomaly_eccentric_to_mean",
    "anomaly_eccentric_to_true",
    "anomaly_mean_to_eccentric",
    "anomaly_mean_to_true",
    "anomaly_true_to_eccentric",
    "anomaly_true_to_mean",
    "apoapsis_altitude",
    "apoapsis_distance",
    "apoapsis_velocity",
    "apogee_altitude",
    "apogee_velocity",
    "calculate_tle_line_checksum",
    "create_tle_lines",
    "epoch_from_tle",
    "geo_sma",
    # TLE conversions
    "keplerian_elements_from_tle",
    "keplerian_elements_to_tle",
    "mean_motion",
    "mean_motion_general",
    "norad_id_alpha5_to_numeric",
    # NORAD ID conversions
    "norad_id_numeric_to_alpha5",
    # Orbital element calculations
    "orbital_period",
    "orbital_period_from_state",
    "orbital_period_general",
    "parse_norad_id",
    "periapsis_altitude",
    "periapsis_distance",
    "periapsis_velocity",
    "perigee_altitude",
    # Apsis calculations
    "perigee_velocity",
    "semimajor_axis",
    "semimajor_axis_from_orbital_period",
    "semimajor_axis_from_orbital_period_general",
    "semimajor_axis_general",
    "state_equinoctial_to_koe",
    "state_koe_mean_to_osc",
    # Mean-osculating Keplerian element conversions
    "state_koe_osc_to_mean",
    # Equinoctial element conversions
    "state_koe_to_equinoctial",
    "states_koe_mean_to_osc",
    "states_koe_osc_to_mean",
    # Special orbits
    "sun_synchronous_inclination",
    "validate_tle_line",
    # TLE validation and parsing
    "validate_tle_lines",
]
