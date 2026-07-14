# Mars Frames

Transformations between Mars-Centered Inertial (MCI), Mars-Centered Mars-Fixed (MCMF), and Earth-Centered Inertial (ECI).

!!! note
    For conceptual explanations and examples, including MCI's body-center origin and the `mar099s` kernel it uses, see [Mars Reference Frames](../../learn/frames/mars_frames.md) in the Learn section.

## MCI ↔ MCMF

::: brahe.rotation_mci_to_mcmf

::: brahe.rotation_mcmf_to_mci

::: brahe.position_mci_to_mcmf

::: brahe.position_mcmf_to_mci

::: brahe.state_mci_to_mcmf

::: brahe.state_mcmf_to_mci

## ECI ↔ MCI

::: brahe.position_eci_to_mci

::: brahe.position_mci_to_eci

::: brahe.state_eci_to_mci

::: brahe.state_mci_to_eci

## See Also

- [Mars Reference Frames (Learn)](../../learn/frames/mars_frames.md) - Conceptual explanation, examples, and MCI's body-center origin
- [Lunar Frames](lunar.md) - Moon-centered frame transformations
- [Reference Frame Router](router.md) - Generic conversion between any two frames
- [Reference Frames Module](index.md) - Complete API reference for frames module
