//! # Constants
//! Constant values, both universal and observatory specific.
//!
mod gravity;
mod neos;
mod universal;
mod wise;

pub use gravity::{
    EARTH_J2, EARTH_J3, EARTH_J4, EM_BARY, GMS, GMS_SQRT, GravParams, JUPITER_J2, MASSES, PLANETS,
    SIMPLE_PLANETS, SUN_J2,
};
pub use neos::{NEOS_BANDS, NEOS_HEIGHT, NEOS_SUN_CORRECTION, NEOS_WIDTH, NEOS_ZERO_MAG};
pub use universal::{
    AU_KM, C_AU_PER_DAY, C_AU_PER_DAY_INV, C_AU_PER_DAY_INV_SQUARED, C_M_PER_S, C_V, GOLDEN_RATIO,
    SOLAR_FLUX, STEFAN_BOLTZMANN, SUN_DIAMETER, SUN_TEMP, V_MAG_ZERO,
};
pub use wise::{
    WISE_BANDS, WISE_BANDS_300K, WISE_CC, WISE_SUN_CORRECTION, WISE_WIDTH, WISE_ZERO_MAG,
    WISE_ZERO_MAG_300K, w1_color_correction, w2_color_correction, w3_color_correction,
    w4_color_correction,
};
