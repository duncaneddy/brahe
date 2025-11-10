# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Get the EME2000 frame bias matrix and verify its relationship to rotation functions
"""

import brahe as bh

# Get the EME2000 frame bias matrix
B = bh.bias_eme2000()

print("EME2000 frame bias matrix:")
print(f"  [{B[0, 0]:13.10f}, {B[0, 1]:13.10f}, {B[0, 2]:13.10f}]")
print(f"  [{B[1, 0]:13.10f}, {B[1, 1]:13.10f}, {B[1, 2]:13.10f}]")
print(f"  [{B[2, 0]:13.10f}, {B[2, 1]:13.10f}, {B[2, 2]:13.10f}]\n")
# EME2000 frame bias matrix:
#   [ 1.0000000000, -0.0000000708,  0.0000000806]
#   [ 0.0000000708,  1.0000000000,  0.0000000331]
#   [-0.0000000806, -0.0000000331,  1.0000000000]
