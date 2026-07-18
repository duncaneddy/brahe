# Equinoctial Elements

Functions for converting between Keplerian and equinoctial orbital elements. Equinoctial
elements `[a, h, k, p, q, l]` are singularity-free at zero eccentricity and zero/180°
inclination, which makes them convenient for averaging and numerical differencing.

!!! note
    For conceptual background, see [Mean Elements](../../learn/orbits/mean_elements.md) in the
    User Guide, where equinoctial elements are used internally by numerical windowed averaging.

::: brahe.orbits.state_koe_to_equinoctial

::: brahe.orbits.state_equinoctial_to_koe

## See Also

- [Keplerian Elements](keplerian.md) - Classical orbital elements and anomaly conversions
- [Mean Elements](mean_elements.md) - Mean-osculating conversions built on this representation
