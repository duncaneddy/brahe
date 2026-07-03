# Reference Frame Router

`ReferenceFrame` and the generic `rotation_frame_to_frame`/`position_frame_to_frame`/`state_frame_to_frame` functions convert between any two supported reference frames, including generic NAIF-ID variants for bodies without a dedicated named frame.

!!! note
    For conceptual explanations, the full frame/kernel-requirement tables, and central-body propagation defaults, see [Frame Router & Multibody Propagation](../../learn/frames/frame_transformations.md) in the Learn section.

## ReferenceFrame

::: brahe.ReferenceFrame

## Router Functions

::: brahe.rotation_frame_to_frame

::: brahe.position_frame_to_frame

::: brahe.state_frame_to_frame

## Generic IAU/WGCCRE Body-Fixed Rotations

::: brahe.rotation_icrf_to_body_fixed_iau

::: brahe.iau_rotation_model_ids

## See Also

- [Frame Router & Multibody Propagation (Learn)](../../learn/frames/frame_transformations.md) - Conceptual explanation, kernel requirements, and central-body propagation defaults
- [Lunar Frames](lunar.md) - Moon-centered frame transformations
- [Mars Frames](mars.md) - Mars-centered frame transformations
- [CentralBody / ForceModelConfig](../propagators/force_model_config.md) - Multibody propagation configuration
- [Reference Frames Module](index.md) - Complete API reference for frames module
