// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Constants
//! Constant values, both universal and observatory specific.

mod neos;
mod spitzer;
mod universal;
mod wise;

pub use neos::{NEOS_HEIGHT, NEOS_WIDTH};
pub use spitzer::IRAC_WIDTH;
pub use universal::{
    AU_KM, C_AU_PER_DAY, C_AU_PER_DAY_INV, C_AU_PER_DAY_INV_SQUARED, C_M_PER_S, C_V, EARTH_J2,
    EARTH_J3, EARTH_J4, F0_OVER_C_AU_DAY2, GMS, GMS_SQRT, GOLDEN_RATIO, JUPITER_J2, SOLAR_FLUX,
    STEFAN_BOLTZMANN, SUN_DIAMETER, SUN_J2, SUN_TEMP, V_MAG_ZERO,
};
pub use wise::{
    WISE_WIDTH, w1_color_correction, w2_color_correction, w3_color_correction, w4_color_correction,
};
