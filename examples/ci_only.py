# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Test file with CI-ONLY flag - should be skipped by default.

This file tests that the FLAG system correctly skips CI-ONLY examples
unless --ci-only is passed to the test command.
"""

print("CI-ONLY flag test: This should only run with --ci-only flag")
print("If you see this during normal test runs, the flag system is broken!")
