/*//

MSIS_GFN Module: Contains subroutines to calculate global (horizontal and time-dependent) model basis functions

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2023-03-17: Translated the original Fortran code into Rust.
- 2023-03-17:
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

//==================================================================================================
// GLOBE: Calculate horizontal and time-dependent basis functions
//        (Same purpose as NRLMSISE-00 "GLOBE7" subroutine)
//==================================================================================================

//==================================================================================================
// SOLZEN: Calculate solar zenith angle (adapted from IRI subroutine)
//==================================================================================================

//==================================================================================================
// SFLUXMOD: Legacy nonlinear modulation of intra-annual, tide, and SPW terms
//==================================================================================================

//==================================================================================================
// GEOMAG: Legacy nonlinear ap dependence (daily ap mode and ap history mode), including mixed
//         ap/UT/Longitude terms.
// Master switch control is as follows:
//   swg(cmag) .nor. swg(cmag+1)   Do nothing: Return zero
//   swg(cmag) .and. swg(cmag+1)   Daily Ap mode
//   swg(cmag) .neqv. swg(cmag+1)  3-hour ap history mode
//==================================================================================================

//==================================================================================================
// UTDEP: Legacy nonlinear UT dependence
//==================================================================================================

// pub fn utdep(p0: &[f64; NUT], bf: &[f64; 9]) -> f64 {
//     let mut p = p0.clone();
//     let swg1 = crate::orbit_dynamics::nrlmsise21::msis_init::swg[CUT..CUT + NUT].to_vec();
//
//     for i in 3..NUT {
//         if !swg1[i] {
//             p[i] = 0.0;
//         }
//     }
//
//     let utdep = (bf[0] - p[0]).cos() *
//         (1.0 + p[3] * bf[4] * (bf[1] - p[1]).cos()) *
//         (1.0 + p[4] * bf[2]) * (1.0 + p[5] * bf[4]) *
//         (p[6] * bf[4] + p[7] * bf[5] + p[8] * bf[6]) +
//         (bf[0] - p[2] + 2.0 * bf[3]).cos() * (p[9] * bf[7] + p[10] * bf[8]) * (1.0 + p[11] * bf[2]);
//
//     utdep
// }