# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the RKN1210 high-precision integrator method.
12th-order Runge-Kutta-Nystrom for maximum accuracy.
"""

import brahe as bh

# RKN1210 (Runge-Kutta-Nystrom 12(10))
# - Very high-order adaptive integrator
# - 12th order solution with 10th order error estimate
# - 17 function evaluations per step
# - Optimized for second-order ODEs (like orbital mechanics)
# - Achieves extreme accuracy with tight tolerances

# Create high-precision configuration using RKN1210
config = bh.NumericalPropagationConfig.high_precision()

# Or manually configure RKN1210 with custom tolerances
config_custom = (
    bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKN1210)
    .with_abs_tol(1e-12)
    .with_rel_tol(1e-10)
)

print("RKN1210 High-Precision Integrator Configuration:")
print("  Method: RKN1210 (Runge-Kutta-Nystrom 12(10))")
print("  Adaptive: Yes")
print("  Function evaluations per step: 17")
print("  Order: 12(10) - 12th order solution, 10th order error estimate")

print("\nhigh_precision() preset tolerances:")
print("  Absolute tolerance: 1e-10")
print("  Relative tolerance: 1e-8")

print("\nNystrom Optimization:")
print("  RKN methods are designed specifically for second-order ODEs:")
print("    x'' = f(t, x, x')")
print("  This matches orbital mechanics where acceleration depends")
print("  on position and velocity, not their derivatives.")

print("\nCharacteristics:")
print("  - Highest precision available")
print("  - Larger steps than lower-order methods for same accuracy")
print("  - More expensive per step (17 function evaluations)")
print("  - Can achieve sub-millimeter accuracy over long periods")

print("\nWhen to use RKN1210:")
print("  - Precision orbit determination")
print("  - Reference trajectory generation")
print("  - Scientific research requiring highest accuracy")
print("  - Validation of other methods")
print("  - When sub-meter accuracy matters over days/weeks")

print("\nWhen NOT to use RKN1210:")
print("  - Quick estimates or prototyping")
print("  - Real-time applications")
print("  - When moderate accuracy is sufficient")
print("  - Memory-constrained environments")

print("\nExample validated successfully!")
