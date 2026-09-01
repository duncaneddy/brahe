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
    BodyFrame,
    CelestialFrame,
    Frame,
    # Reference frame router
    SynodicOrigin,
    # EME2000 <> GCRF transformations
    bias_eme2000,
    # Rotation matrix components
    bias_precession_nutation,
    clear_frame_registry,
    clear_object_registry,
    earth_rotation,
    iau_rotation_model_ids,
    polar_motion,
    position_ecef_to_eci,
    position_eci_to_ecef,
    position_eci_to_emb,
    position_eci_to_lci,
    position_eci_to_mci,
    position_emb_to_eci,
    position_eme2000_to_gcrf,
    position_emr_to_gcrf,
    position_frame_to_frame,
    position_gcrf_to_eme2000,
    position_gcrf_to_emr,
    position_gcrf_to_gse,
    position_gcrf_to_itrf,
    position_gcrf_to_ser,
    position_gse_to_gcrf,
    position_itrf_to_gcrf,
    position_lci_to_eci,
    position_lci_to_lfme,
    position_lci_to_lfpa,
    position_lfme_to_lci,
    position_lfpa_to_lci,
    position_mci_to_eci,
    position_mci_to_mcmf,
    position_mcmf_to_mci,
    position_ser_to_gcrf,
    register_custom_frame,
    register_frame,
    register_object,
    register_object_from_naif,
    registered_objects,
    rotation_ecef_to_eci,
    rotation_eci_to_ecef,
    rotation_eme2000_to_gcrf,
    rotation_emr_to_gcrf,
    rotation_frame_to_frame,
    rotation_gcrf_to_eme2000,
    # Synodic reference frames (EMR, SER, GSE)
    rotation_gcrf_to_emr,
    rotation_gcrf_to_gse,
    rotation_gcrf_to_itrf,
    rotation_gcrf_to_ser,
    rotation_gse_to_gcrf,
    # IAU/WGCCRE body rotation model
    rotation_icrf_to_body_fixed_iau,
    rotation_itrf_to_gcrf,
    rotation_lci_to_lfme,
    # Lunar reference frames (LCI, LFPA, LFME)
    rotation_lci_to_lfpa,
    rotation_lfme_to_lci,
    rotation_lfme_to_lfpa,
    rotation_lfpa_to_lci,
    rotation_lfpa_to_lfme,
    # Mars reference frames (MCI, MCMF)
    rotation_mci_to_mcmf,
    rotation_mcmf_to_mci,
    rotation_ser_to_gcrf,
    state_ecef_to_eci,
    state_eci_to_ecef,
    state_eci_to_emb,
    state_eci_to_lci,
    state_eci_to_mci,
    state_emb_to_eci,
    state_eme2000_to_gcrf,
    state_emr_to_gcrf,
    state_frame_to_frame,
    state_gcrf_to_eme2000,
    state_gcrf_to_emr,
    state_gcrf_to_gse,
    state_gcrf_to_itrf,
    state_gcrf_to_ser,
    state_gse_to_gcrf,
    state_itrf_to_gcrf,
    state_lci_to_eci,
    state_lci_to_lfme,
    state_lci_to_lfpa,
    state_lfme_to_lci,
    state_lfpa_to_lci,
    state_mci_to_eci,
    state_mci_to_mcmf,
    state_mcmf_to_mci,
    state_ser_to_gcrf,
    unregister_custom_frame,
    unregister_frame,
    unregister_object,
)

__all__ = [
    "BodyFrame",
    "CelestialFrame",
    "Frame",
    # Reference frame router
    "SynodicOrigin",
    # EME2000 <> GCRF transformations
    "bias_eme2000",
    # Rotation matrix components
    "bias_precession_nutation",
    "clear_frame_registry",
    "clear_object_registry",
    "earth_rotation",
    "iau_rotation_model_ids",
    "polar_motion",
    "position_ecef_to_eci",
    "position_eci_to_ecef",
    "position_eci_to_emb",
    "position_eci_to_lci",
    "position_eci_to_mci",
    "position_emb_to_eci",
    "position_eme2000_to_gcrf",
    "position_emr_to_gcrf",
    "position_frame_to_frame",
    "position_gcrf_to_eme2000",
    "position_gcrf_to_emr",
    "position_gcrf_to_gse",
    "position_gcrf_to_itrf",
    "position_gcrf_to_ser",
    "position_gse_to_gcrf",
    "position_itrf_to_gcrf",
    "position_lci_to_eci",
    "position_lci_to_lfme",
    "position_lci_to_lfpa",
    "position_lfme_to_lci",
    "position_lfpa_to_lci",
    "position_mci_to_eci",
    "position_mci_to_mcmf",
    "position_mcmf_to_mci",
    "position_ser_to_gcrf",
    "register_custom_frame",
    "register_frame",
    "register_object",
    "register_object_from_naif",
    "registered_objects",
    "rotation_ecef_to_eci",
    "rotation_eci_to_ecef",
    "rotation_eme2000_to_gcrf",
    "rotation_emr_to_gcrf",
    "rotation_frame_to_frame",
    "rotation_gcrf_to_eme2000",
    # Synodic reference frames (EMR, SER, GSE)
    "rotation_gcrf_to_emr",
    "rotation_gcrf_to_gse",
    "rotation_gcrf_to_itrf",
    "rotation_gcrf_to_ser",
    "rotation_gse_to_gcrf",
    # IAU/WGCCRE body rotation model
    "rotation_icrf_to_body_fixed_iau",
    "rotation_itrf_to_gcrf",
    "rotation_lci_to_lfme",
    # Lunar reference frames (LCI, LFPA, LFME)
    "rotation_lci_to_lfpa",
    "rotation_lfme_to_lci",
    "rotation_lfme_to_lfpa",
    "rotation_lfpa_to_lci",
    "rotation_lfpa_to_lfme",
    # Mars reference frames (MCI, MCMF)
    "rotation_mci_to_mcmf",
    "rotation_mcmf_to_mci",
    "rotation_ser_to_gcrf",
    "state_ecef_to_eci",
    "state_eci_to_ecef",
    "state_eci_to_emb",
    "state_eci_to_lci",
    "state_eci_to_mci",
    "state_emb_to_eci",
    "state_eme2000_to_gcrf",
    "state_emr_to_gcrf",
    "state_frame_to_frame",
    "state_gcrf_to_eme2000",
    "state_gcrf_to_emr",
    "state_gcrf_to_gse",
    "state_gcrf_to_itrf",
    "state_gcrf_to_ser",
    "state_gse_to_gcrf",
    "state_itrf_to_gcrf",
    "state_lci_to_eci",
    "state_lci_to_lfme",
    "state_lci_to_lfpa",
    "state_lfme_to_lci",
    "state_lfpa_to_lci",
    "state_mci_to_eci",
    "state_mci_to_mcmf",
    "state_mcmf_to_mci",
    "state_ser_to_gcrf",
    "unregister_custom_frame",
    "unregister_frame",
    "unregister_object",
]
