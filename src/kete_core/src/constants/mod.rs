//! # Constants
//! Constant values, both universal and observatory specific.
//!
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

mod gravity;
mod neos;
mod universal;
mod wise;

pub use gravity::{
    EARTH_J2, EARTH_J3, EARTH_J4, GMS, GMS_SQRT, GravParams, JUPITER_J2, SUN_J2, known_masses,
    register_custom_mass, register_mass, registered_masses,
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
