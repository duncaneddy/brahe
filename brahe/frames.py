"""
Reference Frames Module

Reference frame transformations between ECI and ECEF coordinate systems.

This module provides transformations between:
- ECI (Earth-Centered Inertial): J2000/GCRF frame
- ECEF (Earth-Centered Earth-Fixed): ITRF frame

The transformations implement the IAU 2006/2000A precession-nutation model
and use Earth Orientation Parameters (EOP) for high-precision conversions.

Functions are provided for:
- Rotation matrices (bias-precession-nutation, Earth rotation, polar motion)
- Position vector transformations
- State vector (position + velocity) transformations

Naming Conventions:
  Brahe provides two equivalent sets of function names for frame transformations:

  - ECI/ECEF naming: Traditional coordinate system names (e.g., rotation_eci_to_ecef)
  - GCRF/ITRF naming: Explicit reference frame names (e.g., rotation_gcrf_to_itrf)

  Both naming conventions provide identical results. Users can choose whichever
  convention they prefer. The ECI/ECEF names are more intuitive and widely used,
  while the GCRF/ITRF names explicitly identify the specific reference frame
  implementations used. The ECI/ECEF names are provided as the default to get
  the "best" reference frame transformations out-of-the-box, while the
  GCRF/ITRF names are for users who want to be explicit about the
  reference frames they are using.
"""

from brahe._brahe import (
    # Rotation matrix components
    bias_precession_nutation,
    earth_rotation,
    polar_motion,
    rotation_gcrf_to_itrf,
    rotation_itrf_to_gcrf,
    rotation_eci_to_ecef,
    rotation_ecef_to_eci,
    position_gcrf_to_itrf,
    position_itrf_to_gcrf,
    position_eci_to_ecef,
    position_ecef_to_eci,
    state_gcrf_to_itrf,
    state_itrf_to_gcrf,
    state_eci_to_ecef,
    state_ecef_to_eci,
    # EME2000 <> GCRF transformations
    bias_eme2000,
    rotation_gcrf_to_eme2000,
    rotation_eme2000_to_gcrf,
    position_gcrf_to_eme2000,
    position_eme2000_to_gcrf,
    state_gcrf_to_eme2000,
    state_eme2000_to_gcrf,
    # IAU/WGCCRE body rotation model
    rotation_icrf_to_body_fixed_iau,
    iau_rotation_model_ids,
    # Mars reference frames (MCI, MCMF)
    rotation_mci_to_mcmf,
    rotation_mcmf_to_mci,
    position_mci_to_mcmf,
    position_mcmf_to_mci,
    state_mci_to_mcmf,
    state_mcmf_to_mci,
    position_eci_to_mci,
    position_mci_to_eci,
    state_eci_to_mci,
    state_mci_to_eci,
    # Lunar reference frames (LCI, LFPA, LFME)
    rotation_lci_to_lfpa,
    rotation_lfpa_to_lci,
    rotation_lfme_to_lfpa,
    rotation_lfpa_to_lfme,
    rotation_lci_to_lfme,
    rotation_lfme_to_lci,
    position_lci_to_lfpa,
    position_lfpa_to_lci,
    position_lci_to_lfme,
    position_lfme_to_lci,
    state_lci_to_lfpa,
    state_lfpa_to_lci,
    state_lci_to_lfme,
    state_lfme_to_lci,
    position_eci_to_lci,
    position_lci_to_eci,
    state_eci_to_lci,
    state_lci_to_eci,
    # Synodic reference frames (EMR, SER, GSE)
    rotation_gcrf_to_emr,
    rotation_emr_to_gcrf,
    position_gcrf_to_emr,
    position_emr_to_gcrf,
    state_gcrf_to_emr,
    state_emr_to_gcrf,
    rotation_gcrf_to_ser,
    rotation_ser_to_gcrf,
    position_gcrf_to_ser,
    position_ser_to_gcrf,
    state_gcrf_to_ser,
    state_ser_to_gcrf,
    rotation_gcrf_to_gse,
    rotation_gse_to_gcrf,
    position_gcrf_to_gse,
    position_gse_to_gcrf,
    state_gcrf_to_gse,
    state_gse_to_gcrf,
    # Reference frame router
    ReferenceFrame,
    rotation_frame_to_frame,
    register_custom_frame,
    unregister_custom_frame,
    position_frame_to_frame,
    state_frame_to_frame,
)

__all__ = [
    # Rotation matrix components
    "bias_precession_nutation",
    "earth_rotation",
    "polar_motion",
    "rotation_gcrf_to_itrf",
    "rotation_itrf_to_gcrf",
    "rotation_eci_to_ecef",
    "rotation_ecef_to_eci",
    "position_gcrf_to_itrf",
    "position_itrf_to_gcrf",
    "position_eci_to_ecef",
    "position_ecef_to_eci",
    "state_gcrf_to_itrf",
    "state_itrf_to_gcrf",
    "state_eci_to_ecef",
    "state_ecef_to_eci",
    # EME2000 <> GCRF transformations
    "bias_eme2000",
    "rotation_gcrf_to_eme2000",
    "rotation_eme2000_to_gcrf",
    "position_gcrf_to_eme2000",
    "position_eme2000_to_gcrf",
    "state_gcrf_to_eme2000",
    "state_eme2000_to_gcrf",
    # IAU/WGCCRE body rotation model
    "rotation_icrf_to_body_fixed_iau",
    "iau_rotation_model_ids",
    # Mars reference frames (MCI, MCMF)
    "rotation_mci_to_mcmf",
    "rotation_mcmf_to_mci",
    "position_mci_to_mcmf",
    "position_mcmf_to_mci",
    "state_mci_to_mcmf",
    "state_mcmf_to_mci",
    "position_eci_to_mci",
    "position_mci_to_eci",
    "state_eci_to_mci",
    "state_mci_to_eci",
    # Lunar reference frames (LCI, LFPA, LFME)
    "rotation_lci_to_lfpa",
    "rotation_lfpa_to_lci",
    "rotation_lfme_to_lfpa",
    "rotation_lfpa_to_lfme",
    "rotation_lci_to_lfme",
    "rotation_lfme_to_lci",
    "position_lci_to_lfpa",
    "position_lfpa_to_lci",
    "position_lci_to_lfme",
    "position_lfme_to_lci",
    "state_lci_to_lfpa",
    "state_lfpa_to_lci",
    "state_lci_to_lfme",
    "state_lfme_to_lci",
    "position_eci_to_lci",
    "position_lci_to_eci",
    "state_eci_to_lci",
    "state_lci_to_eci",
    # Synodic reference frames (EMR, SER, GSE)
    "rotation_gcrf_to_emr",
    "rotation_emr_to_gcrf",
    "position_gcrf_to_emr",
    "position_emr_to_gcrf",
    "state_gcrf_to_emr",
    "state_emr_to_gcrf",
    "rotation_gcrf_to_ser",
    "rotation_ser_to_gcrf",
    "position_gcrf_to_ser",
    "position_ser_to_gcrf",
    "state_gcrf_to_ser",
    "state_ser_to_gcrf",
    "rotation_gcrf_to_gse",
    "rotation_gse_to_gcrf",
    "position_gcrf_to_gse",
    "position_gse_to_gcrf",
    "state_gcrf_to_gse",
    "state_gse_to_gcrf",
    # Reference frame router
    "ReferenceFrame",
    "rotation_frame_to_frame",
    "register_custom_frame",
    "unregister_custom_frame",
    "position_frame_to_frame",
    "state_frame_to_frame",
]
