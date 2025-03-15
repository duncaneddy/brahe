// /*!
// */
//
// /// The `tle_checksum` function calculates the checksum for a given TLE line. The checksum is the
// /// modulo 10 sum of the digits in the line, excluding the last character. Any '-' characters are
// /// treated as 1.
// ///
// /// # Arguments
// /// * `line` - A string slice containing the TLE line to calculate the checksum for.
// ///
// /// # Returns
// /// A u8 value representing the checksum of the TLE line.
// ///
// /// # Example
// /// ```
// /// use brahe::tle_checksum;
// ///
// /// let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
// /// let checksum = tle_checksum(line);
// /// assert_eq!(checksum, 3);
// /// ```
// pub fn tle_checksum(line: &str) -> u8 {
//     let mut sum = 0;
//     for (i, c) in line.chars().enumerate() {
//         if c.is_digit(10) {
//             sum += c.to_digit(10).unwrap() * (i + 1) as u32;
//         } else if c == '-' {
//             sum += 1;
//         }
//     }
//     (sum % 10) as u8
// }
//
// /// The `valid_tle_checksum` function checks if the checksum of a given TLE line is valid. The
// /// checksum is the modulo 10 sum of the digits in the line, excluding the last character. Any '-'
// /// characters are treated as 1.
// ///
// /// # Arguments
// /// * `line` - A string slice containing the TLE line to check the checksum for.
// ///
// /// # Returns
// /// A boolean value indicating if the checksum is valid.
// ///
// /// # Example
// /// ```
// /// use brahe::valid_tle_checksum;
// ///
// /// let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
// /// assert!(valid_tle_checksum(line));
// /// ```
// pub fn valid_tle_checksum(line: &str) -> bool {
//     let checksum = tle_checksum(line);
//
//     // Check if the line length is at least 68 characters
//     if line.len() < 69 {
//         return false;
//     }
//
//     // Check if the last character is a digit matching the checksum
//     if let Some(last) = line.chars().last() {
//         if last.is_digit(10) {
//             checksum == last.to_digit(10).unwrap() as u8
//         } else {
//             false
//         }
//     } else {
//         false
//     }
// }
//
// // struct TLE {
// //     pub line1: String,
// //     pub line2: String,
// //     pub epoch: Epoch,
// // }
// //
// // impl TLE {
// //     pub fn new(line1: &str, line2: &str) -> Result<TLE, String> {
// //         // Check the checksums
// //         let checksum1 = tle_checksum(line1);
// //         let checksum2 = tle_checksum(line2);
// //
// //         if checksum1 != 0 || checksum2 != 0 {
// //             return Err("Checksum failed".to_string());
// //         }
// //
// //         // Parse the epoch
// //         // let epoch = Epoch::from_tle(line1)?;
// //
// //         Ok(TLE {
// //             line1: line1.to_string(),
// //             line2: line2.to_string(),
// //             epoch,
// //         })
// //     }
// // }
//
// // Tests
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn test_tle_checksum() {
//         let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
//         let checksum = tle_checksum(line);
//         assert_eq!(checksum, 3);
//     }
//
//     #[test]
//     fn test_valid_tle_checksum() {
//         let line = "1 25544U 98067A   24287.94238119  .00024791  00000-0  44322-3 0  9991";
//         assert!(valid_tle_checksum(line));
//     }
// }
