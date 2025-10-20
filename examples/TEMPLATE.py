# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
Brief description of what this example demonstrates.

This template shows the minimal structure for a Python documentation example.
Replace this with actual functionality demonstration.
"""

import pytest

if __name__ == "__main__":
    # Setup: Define any input parameters
    value = 1.0

    # Action: Demonstrate the functionality
    result = value * 2.0  # Replace with actual brahe function call

    # Validation: Assert the result is correct
    expected = 2.0
    assert result == pytest.approx(expected, abs=1e-10)

    print("✓ Example validated successfully!")
