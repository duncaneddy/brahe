# GCRF ↔ TEME ↔ ITRF Transformations

Transformations between the Geocentric Celestial Reference Frame (GCRF), the true equator and mean equinox of date (TEME) frame in which SGP4 expresses its output, and the International Terrestrial Reference Frame (ITRF).

!!! note
    For conceptual explanations and examples, see [GCRF ↔ TEME ↔ ITRF Transformations](../../learn/frames/teme.md) in the Learn section.

## Building Blocks

::: brahe.gmst82

::: brahe.greenwich_mean_sidereal_rotation

## GCRF ↔ TEME

::: brahe.rotation_gcrf_to_teme

::: brahe.rotation_teme_to_gcrf

::: brahe.position_gcrf_to_teme

::: brahe.position_teme_to_gcrf

::: brahe.state_gcrf_to_teme

::: brahe.state_teme_to_gcrf

## TEME ↔ ITRF

::: brahe.rotation_teme_to_itrf

::: brahe.rotation_itrf_to_teme

::: brahe.position_teme_to_itrf

::: brahe.position_itrf_to_teme

::: brahe.state_teme_to_itrf

::: brahe.state_itrf_to_teme

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md) - CIO-based transformations
- [GCRF ↔ MOD ↔ TOD Transformations](equinox.md) - Equinox-based transformations
- [Reference Frames Module](index.md) - Complete API reference for frames module
