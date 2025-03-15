// /*!
// This module implements the simplified general perturbations 4 (SGP4) orbit propagator. SGP4 is a
// commonly used orbit propagator for Earth-orbiting satellites. It is a semi-analytical model that
// accounts for both conservative and non-conservative forces acting on the satellite. The implementation
// is based on the SGP4 algorithm described in the paper "Revisiting Spacetrack Report #3" by David A.
// Vallado, Paul Crawford, Richard Hujsak, and T.S. Kelso.
// */
//
// use crate::time::Epoch;
//
// /// The `SGP4Data` struct is used to store the data required to propagate an orbit using the SGP4
// /// algorithm.
// pub(crate) struct SGP4Data {
//     // Propagator settings
//     whichconst: SGP4GravModel,
//     afspc_mode: bool,
//     init: char,
//
//     // Generic values
//     satnum: u64,
//     classification: char,
//     intldesg: String,
//     epochyr: i64,
//     epoch: Epoch,
//     elnum: i64,
//     revnum: i64,
//     method: char,
//
//     // SGP4 Propagator variables
//     // Common terms
//     a: f64,
//     altp: f64,
//     alta: f64,
//     epochdays: f64,
//     jdsatepoch: f64,
//     nddot: f64,
//     ndot: f64,
//     bstar: f64,
//     rcse: f64,
//     inclo: f64,
//     nodeo: f64,
//     ecco: f64,
//     argpo: f64,
//     mo: f64,
//     no: f64,
//
//     // Near Earth terms
//     isimp: u64,
//     aycof: f64,
//     con41: f64,
//     cc1: f64,
//     cc4: f64,
//     cc5: f64,
//     d2: f64,
//     d3: f64,
//     d4: f64,
//     delmo: f64,
//     eta: f64,
//     argpdot: f64,
//     omgcof: f64,
//     sinmao: f64,
//     t: f64,
//     t2cof: f64,
//     t3cof: f64,
//     t4cof: f64,
//     t5cof: f64,
//     x1mth2: f64,
//     x7thm1: f64,
//     mdot: f64,
//     nodedot: f64,
//     xlcof: f64,
//     xmcof: f64,
//     nodecf: f64,
//
//     // Deep Space terms
//     irez: u64,
//     d2201: f64,
//     d2211: f64,
//     d3210: f64,
//     d3222: f64,
//     d4410: f64,
//     d4422: f64,
//     d5220: f64,
//     d5232: f64,
//     d5421: f64,
//     d5433: f64,
//     dedt: f64,
//     del1: f64,
//     del2: f64,
//     del3: f64,
//     didt: f64,
//     dmdt: f64,
//     dnodt: f64,
//     domdt: f64,
//     e3: f64,
//     ee2: f64,
//     peo: f64,
//     pgho: f64,
//     pho: f64,
//     pinco: f64,
//     plo: f64,
//     se2: f64,
//     se3: f64,
//     sgh2: f64,
//     sgh3: f64,
//     sgh4: f64,
//     sh2: f64,
//     sh3: f64,
//     si2: f64,
//     si3: f64,
//     sl2: f64,
//     sl3: f64,
//     sl4: f64,
//     gsto: f64,
//     xfact: f64,
//     xgh2: f64,
//     xgh3: f64,
//     xgh4: f64,
//     xh2: f64,
//     xh3: f64,
//     xi2: f64,
//     xi3: f64,
//     xl2: f64,
//     xl3: f64,
//     xl4: f64,
//     xlamo: f64,
//     zmol: f64,
//     zmos: f64,
//     atime: f64,
//     xli: f64,
//     xni: f64,
//
//     // Error Status Code
//     error: u8,
// }
//
// enum SGP4GravModel {
//     WGS72old,
//     WGS72,
//     WGS84,
// }
//
// ///  Internal SGP method to convert year and days to month and hours, minutes, seconds.
// ///
// /// # Arguments
// /// * `year` - The year of the date.
// /// * `days` - The day of the year.
// ///
// /// # Returns
// /// A tuple containing the month, day, hour, minute, and second of the date.
// /// * `mon` - The month of the date.
// /// * `day` - The day of the month.
// /// * `hr` - The hour of the day.
// /// * `minute` - The minute of the hour.
// /// * `sec` - The second of the minute.
// fn days2mdhms(year:i64, days:f64) -> (i64, i64, i64, i64, f64) {
//
//     let mut lmonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
//
//     let dayofyr = days.floor();
//
//     // ----------------- find month and day of month ----------------
//     if (year % 4) == 0 {
//         lmonth[1] = 29;
//     }
//
//     let mut i = 1;
//     let mut inttemp = 0;
//     while (dayofyr > inttemp + lmonth[i]) && (i < 12) {
//         inttemp = inttemp + lmonth[i];
//         i = i + 1;
//     }
//
//     let mon = i;
//     let day = dayofyr - inttemp;
//
//     // ----------------- find hours minutes and seconds -------------
//     let mut temp = (days - dayofyr) * 24.0;
//     let hr = temp.floor();
//     let temp = (temp - hr) * 60.0;
//     let minute = temp.floor();
//     let sec = (temp - minute) * 60.0;
//
//     return mon, day, hr, minute, sec
// }
