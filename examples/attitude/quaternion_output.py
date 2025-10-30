# /// script
# dependencies = ["brahe"]
# ///

"""
Demonstrates how to access and output quaternion components.
"""

import brahe as bh

# Create a quaternion (45° rotation about Z-axis)
q = bh.Quaternion.from_rotation_matrix(bh.RotationMatrix.Rz(45, bh.AngleFormat.DEGREES))

# Access individual components
print("Individual components:")
print(f"  w (scalar): {q.w:.6f}")
print(f"  x: {q.x:.6f}")
print(f"  y: {q.y:.6f}")
print(f"  z: {q.z:.6f}")

# Directly access as a vector/array
vec = q.data
print("\nAs vector [w, x, y, z]:")
print(f"  {vec}: {type(vec)}")

# Or return copy as a NumPy array
vec_np = q.to_vector(scalar_first=True)
print("\nAs vector [w, x, y, z]:")
print(f"  {vec_np}: {type(vec_np)}")

# Return in different order (scalar last)
vec_np_last = q.to_vector(scalar_first=False)
print("\nAs scalar-last [x, y, z, w]:")
print(f"  {vec_np_last}: {type(vec_np_last)}")

# Display as string
print("\nString representation:")
print(f"  {q}")

print("\Repr representation:")
print(f"  {repr(q)}")

# Expected output:
# Individual components:
#   w (scalar): 0.923880
#   x: 0.000000
#   y: 0.000000
#   z: 0.382683

# As vector :
#   [0.92387953 0.         0.         0.38268343]: <class 'numpy.ndarray'>

# As vector :
#   [0.92387953 0.         0.         0.38268343]: <class 'numpy.ndarray'>

# As scalar-last :
#   [0.         0.         0.38268343 0.92387953]: <class 'numpy.ndarray'>

# String representation:
#   Quaternion: [s: 0.9238795325112867, v: [0, 0, 0.3826834323650897]]
# \Repr representation:
#   Quaternion<0.9238795325112867, 0, 0, 0.3826834323650897>
