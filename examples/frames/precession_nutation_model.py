# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Select the precession-nutation model and measure the difference between the two models
"""

import numpy as np

import brahe as bh

bh.initialize_eop()

print(f"Default precession-nutation model: {bh.get_precession_nutation_model()}")

epc = bh.Epoch(2024, 3, 1, 0, 0, 0.0, time_system=bh.UTC)
print(f"Epoch: {epc}")

R_2006a = bh.rotation_gcrf_to_itrf(epc)

bh.set_precession_nutation_model(bh.PrecessionNutationModel.IAU2000B)
print(f"Selected precession-nutation model: {bh.get_precession_nutation_model()}")

R_2000b = bh.rotation_gcrf_to_itrf(epc)

# Rotation angle between the two matrices, in the form that keeps its precision
# for the very small angles separating the two models
frobenius = np.linalg.norm(R_2006a - R_2000b, "fro")
theta_mas = 2.0 * np.arcsin(frobenius / (2.0 * np.sqrt(2.0))) * bh.RAD2AS * 1000.0

print("\nGCRF to ITRF rotation difference between the two models:")
print(f"  Angle: {theta_mas:.4f} mas")

rc2i_2006a = bh.bias_precession_nutation_model(epc, bh.PrecessionNutationModel.IAU2006A)
rc2i_2000b = bh.bias_precession_nutation_model(epc, bh.PrecessionNutationModel.IAU2000B)

print("\nBias-precession-nutation matrices evaluated per call:")
print(f"  Max absolute difference: {np.max(np.abs(rc2i_2006a - rc2i_2000b)):.2e}")

bh.set_precession_nutation_model(bh.PrecessionNutationModel.IAU2006A)
print(f"\nRestored precession-nutation model: {bh.get_precession_nutation_model()}")
