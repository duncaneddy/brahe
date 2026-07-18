# Mean Elements

Functions for converting between mean and osculating Keplerian orbital elements, using either
first-order Brouwer-Lyddane theory or numerical windowed averaging.

!!! note
    For conceptual explanations and usage examples, see [Mean Elements](../../learn/orbits/mean_elements.md) in the User Guide.

## Single-State Conversions

::: brahe.orbits.state_koe_osc_to_mean

::: brahe.orbits.state_koe_mean_to_osc

## Batch Conversions

::: brahe.orbits.batch_state_koe_osc_to_mean

::: brahe.orbits.batch_state_koe_mean_to_osc

## Methods and Configuration

::: brahe.MeanElementMethod

::: brahe.WindowAlignment

::: brahe.EdgeHandling

::: brahe.NumericalConfig

::: brahe.InverseConfig

## See Also

- [Mean Elements Guide](../../learn/orbits/mean_elements.md) - Conceptual overview and usage examples
- [Equinoctial Elements](equinoctial.md) - Equinoctial element conversions used internally by numerical averaging
