# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
TODO: Write a brief description of what this example demonstrates.
"""

import pytest

# Setup: Define any input parameters
value = 1.0

# Action: Demonstrate the functionality
result = value * 2.0  # Replace with actual brahe function call

# Validation: Assert the result is correct
expected = 2.0
assert result == pytest.approx(expected, abs=1e-10)

print("✓ Example validated successfully!")
