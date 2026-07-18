# Equinoctial Elements

Functions for converting between Keplerian and equinoctial orbital elements. Equinoctial
elements `[a, h, k, p, q, l]` are free of the zero-eccentricity singularity of the classical
elements (where $\Omega$ and $\omega$ become ill-defined), which makes them convenient for
averaging and numerical differencing.

The formulation is a two-chart one selected by the retrograde factor `fr`, which must be
either `+1` or `-1` (other values, including `0`, are invalid): `fr = +1` is regular for
direct orbits and remains well-defined up to (but not including) $i = 180°$, while `fr = -1`
is regular for retrograde orbits near $i = 180°$ and is singular at $i = 0°$. Use `fr = +1`
unless working with near-retrograde orbits.

!!! note
    For conceptual background, see [Mean Elements](../../learn/orbits/mean_elements.md) in the
    User Guide, where equinoctial elements are used internally by numerical windowed averaging.

::: brahe.orbits.state_koe_to_equinoctial

::: brahe.orbits.state_equinoctial_to_koe

## See Also

- [Keplerian Elements](keplerian.md) - Classical orbital elements and anomaly conversions
- [Mean Elements](mean_elements.md) - Mean-osculating conversions built on this representation
