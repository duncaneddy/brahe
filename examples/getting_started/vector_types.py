# /// script
# dependencies = ["brahe"]

import brahe as bh
import numpy as np

# Brahe inputs can be python lists
list_input = [-122.41, 37.77, 16.0]
ecef_list = bh.position_geodetic_to_ecef(list_input, bh.AngleFormat.DEGREES)

# It can also be numpy arrays
array_input = np.array([-122.41, 37.77, 16.0])
ecef_array = bh.position_geodetic_to_ecef(array_input, bh.AngleFormat.DEGREES)

# Any array-like (python set, tuple, etc.) will work as well
tuple_input = (-122.41, 37.77, 16.0)
ecef_tuple = bh.position_geodetic_to_ecef(tuple_input, bh.AngleFormat.DEGREES)

print(f"ECEF from list: {ecef_list}")
print(f"ECEF from array: {ecef_array}")
print(f"ECEF from tuple: {ecef_tuple}")

# The output of Brahe functions that return vectors is always a numpy array
print(f"Type of ECEF output: {type(ecef_list)}")
print(f"Type of ECEF output: {type(ecef_array)}")
