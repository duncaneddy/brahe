# /// script
# dependencies = ["brahe"]
# ///
#
# This file exists to prove the runner reads FLAGS from the entire
# leading comment block, not just the first 10 lines. The parser
# used to stop scanning after 10 lines, which silently ignored FLAGS
# declared later in the header -- lunar_orbit.rs and mars_orbit.rs
# both had their FLAGS comment on line 13 and ran against the
# network despite declaring NETWORK. A regression back to the
# 10-line window would cause this fixture to run (and print) even
# though its FLAGS comment sits below line 10.
# FLAGS = ["NETWORK"]
"""
Test file with the NETWORK flag placed past line 10.
"""

print("NETWORK late-flag test: should only run with --network flag")
