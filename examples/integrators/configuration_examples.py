# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Examples of different integrator configurations for various scenarios.

This example shows how to configure integrators for different accuracy,
performance, and reliability requirements.
"""

import brahe as bh
import numpy as np

# Initialize EOP
bh.initialize_eop()


# Define orbital dynamics
def dynamics(t, state):
    mu = bh.GM_EARTH
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate([v, a])


# LEO orbit initial state
r0 = np.array([bh.R_EARTH + 600e3, 0.0, 0.0])
v0 = np.array([0.0, 7.5e3, 0.0])
state0 = np.concatenate([r0, v0])
period = 2 * np.pi * np.sqrt(np.linalg.norm(r0) ** 3 / bh.GM_EARTH)

print("Integrator Configuration Examples\n")
print("=" * 70)

# Example 1: Conservative (High Reliability)
print("\n1. CONSERVATIVE Configuration (Mission-Critical)")
print("-" * 70)

# --8<-- [start:conservative]
conservative_config = bh.IntegratorConfig(
    abs_tol=1e-12,
    rel_tol=1e-11,
    min_step=0.01,
    max_step=100.0,
    step_safety_factor=0.85,  # More conservative
    min_step_scale_factor=0.3,
    max_step_scale_factor=5.0,  # Limit step growth
    max_step_attempts=15,
)
# --8<-- [end:conservative]

print(f"  abs_tol: {conservative_config.abs_tol:.0e}")
print(f"  rel_tol: {conservative_config.rel_tol:.0e}")
print(f"  max_step: {conservative_config.max_step:.0f} s")
print(f"  safety_factor: {conservative_config.step_safety_factor}")
print("  Use case: Critical operations, high-precision ephemeris")

# Example 2: Balanced (Recommended Default)
print("\n2. BALANCED Configuration (Recommended)")
print("-" * 70)
# --8<-- [start:balanced]
balanced_config = bh.IntegratorConfig.adaptive(abs_tol=1e-10, rel_tol=1e-9)
# --8<-- [end:balanced]

print(f"  abs_tol: {balanced_config.abs_tol:.0e}")
print(f"  rel_tol: {balanced_config.rel_tol:.0e}")
print(f"  max_step: {balanced_config.max_step:.0e} s")
print(f"  safety_factor: {balanced_config.step_safety_factor}")
print("  Use case: Most applications, ~1-10m accuracy")

# Example 3: Aggressive (High Performance)
print("\n3. AGGRESSIVE Configuration (Fast Computation)")
print("-" * 70)
# --8<-- [start:aggressive]
aggressive_config = bh.IntegratorConfig(
    abs_tol=1e-8,
    rel_tol=1e-7,
    initial_step=60.0,
    min_step=1.0,
    max_step=1000.0,  # Large steps allowed
    step_safety_factor=0.95,  # Less conservative
    min_step_scale_factor=0.1,
    max_step_scale_factor=15.0,  # Allow rapid growth
    max_step_attempts=8,
)
# --8<-- [end:aggressive]

print(f"  abs_tol: {aggressive_config.abs_tol:.0e}")
print(f"  rel_tol: {aggressive_config.rel_tol:.0e}")
print(f"  max_step: {aggressive_config.max_step:.0f} s")
print(f"  safety_factor: {aggressive_config.step_safety_factor}")
print("  Use case: Fast trajectory analysis, ~10-100m accuracy")

# Example 4: High Precision (RKN1210)
print("\n4. HIGH PRECISION Configuration (Sub-meter)")
print("-" * 70)
# --8<-- [start:high_precision_config]
high_precision_config = bh.IntegratorConfig(
    abs_tol=1e-14,
    rel_tol=1e-13,
    min_step=0.001,
    max_step=200.0,
    step_safety_factor=0.9,
    min_step_scale_factor=0.2,
    max_step_scale_factor=10.0,
    max_step_attempts=12,
)
# --8<-- [end:high_precision_config]

print(f"  abs_tol: {high_precision_config.abs_tol:.0e}")
print(f"  rel_tol: {high_precision_config.rel_tol:.0e}")
print(f"  max_step: {high_precision_config.max_step:.0f} s")
print(f"  safety_factor: {high_precision_config.step_safety_factor}")
print("  Use case: High-precision orbit determination, <1m accuracy")
print("  Requires: RKN1210 integrator")

# Example 5: Fixed-Step Configuration
print("\n5. FIXED-STEP Configuration")
print("-" * 70)
fixed_config = bh.IntegratorConfig.fixed_step(step_size=60.0)

print(f"  step_size: {60.0} s")
print("  Note: Step size can be overridden with dt parameter in integrator.step()")
print("  Use case: Regular output intervals, predictable cost")

print("\n" + "=" * 70)
print("\nRecommendations:")
print("• Start with BALANCED for most applications")
print("• Use CONSERVATIVE for mission-critical operations")
print("• Use AGGRESSIVE only when accuracy can be sacrificed for speed")
print("• Use HIGH PRECISION with RKN1210 for sub-meter accuracy")
print("• Use FIXED-STEP when regular output intervals are required")
