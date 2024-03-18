/*!
This module provides an implementaiton of the NRLMSIS-2.1 atmospheric model. It is licensed under the
Open Source Academic Research License Agreement found in the `licenses` directory. The source files
were retrived from https://map.nrl.navy.mil/map/pub/nrl/NRLMSIS/NRLMSIS2.1/ on 2024-03-15, and
translated into Rust.
 */
mod msis_utils;
mod msis_tfn;
mod msis_constants;
mod msis_calc;
mod msis_gfn;
mod msis_dfn;
mod msis_gtd8d;
mod msis_init;