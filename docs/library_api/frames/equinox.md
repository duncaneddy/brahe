# GCRF ↔ MOD ↔ TOD Transformations

Equinox-based transformations between the Geocentric Celestial Reference Frame (GCRF), the Earth mean equator and equinox of date (MOD), the true equator and equinox of date (TOD), and the International Terrestrial Reference Frame (ITRF).

!!! note
    For conceptual explanations and examples, see [GCRF ↔ MOD ↔ TOD Transformations](../../learn/frames/equinox_frames.md) in the Learn section.

## Building Blocks

::: brahe.bias_precession

::: brahe.nutation

::: brahe.greenwich_apparent_sidereal_rotation

## GCRF ↔ MOD

::: brahe.rotation_gcrf_to_mod

::: brahe.rotation_mod_to_gcrf

::: brahe.position_gcrf_to_mod

::: brahe.position_mod_to_gcrf

::: brahe.state_gcrf_to_mod

::: brahe.state_mod_to_gcrf

## MOD ↔ TOD

::: brahe.rotation_mod_to_tod

::: brahe.rotation_tod_to_mod

::: brahe.position_mod_to_tod

::: brahe.position_tod_to_mod

::: brahe.state_mod_to_tod

::: brahe.state_tod_to_mod

## GCRF ↔ TOD

::: brahe.rotation_gcrf_to_tod

::: brahe.rotation_tod_to_gcrf

::: brahe.position_gcrf_to_tod

::: brahe.position_tod_to_gcrf

::: brahe.state_gcrf_to_tod

::: brahe.state_tod_to_gcrf

## TOD ↔ ITRF

::: brahe.rotation_tod_to_itrf

::: brahe.rotation_itrf_to_tod

::: brahe.position_tod_to_itrf

::: brahe.position_itrf_to_tod

::: brahe.state_tod_to_itrf

::: brahe.state_itrf_to_tod

## See Also

- [GCRF ↔ ITRF Transformations](gcrf_itrf.md) - CIO-based transformations
- [EME2000 ↔ GCRF Transformations](eme2000_gcrf.md) - Constant frame bias
- [Reference Frames Module](index.md) - Complete API reference for frames module
