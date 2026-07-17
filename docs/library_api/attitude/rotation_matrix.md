# RotationMatrix Class

The `RotationMatrix` class represents attitude using Direction Cosine Matrices (DCM) for spacecraft orientation and coordinate transformations.

::: brahe._brahe.RotationMatrix

## Standalone Rotation Matrices

The `rotation_x`, `rotation_y`, and `rotation_z` free functions return bare 3x3 numpy arrays for the elementary axis rotations, using the same convention as the `RotationMatrix.Rx`/`Ry`/`Rz` constructors.

::: brahe._brahe.rotation_x

::: brahe._brahe.rotation_y

::: brahe._brahe.rotation_z
