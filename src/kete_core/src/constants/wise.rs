//! WISE specific functions.
//!
// BSD 3-Clause License
//
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

use crate::flux::ColorCorrFn;

/// WISE cryo effective wavelengths for W1, W2, W3, W4 for stellar sources
pub const WISE_BANDS: [f64; 4] = [3352.6, 4602.8, 11560.8, 22088.3];

/// WISE cryo effective wavelengths for W1, W2, W3, W4 for 300K black body sources
pub const WISE_BANDS_300K: [f64; 4] = [3352.6, 4602.8, 11560.8 * 0.96, 22088.3 * 1.025];

/// WISE cryo effective zero magnitudes for W1, W2, W3, W4 for stellar sources
/// Magnitude can then be computed via `-2.5 log10(flux Jy / zero_point)`
pub const WISE_ZERO_MAG: [f64; 4] = [306.681, 170.663, 29.0448, 8.2839];

/// WISE cryo effective zero magnitudes for W1, W2, W3, W4 for 300K sources
pub const WISE_ZERO_MAG_300K: [f64; 4] = [306.681, 170.663, 29.0448 * 1.08, 8.2839 * 0.96];

/// Flux in the reflection model should be scaled by these values.
pub const WISE_SUN_CORRECTION: [f64; 4] = [1.0049, 1.0193, 1.0024, 1.0012];

/// The width of the WISE FOV, 47 arcminutes in radians.
pub const WISE_WIDTH: f64 = 0.01367174580728;

/// Calculate the color correction factor for the W1 band to be applied to fluxes.
///
/// # Arguments
///
/// * `temps` - Vec of temperatures in kelvin.
#[inline]
pub fn w1_color_correction(temp: f64) -> f64 {
    let temp = &temp.clamp(100.0, 400.0);
    (-3.78226591e-01 + 3.50431748e-03 * temp + 1.45866307e-05 * temp.powi(2)
        - 6.67674083e-08 * temp.powi(3)
        + 7.03651986e-11 * temp.powi(4))
    .recip()
}

/// Calculate the color correction factor for the W2 band to be applied to fluxes.
///
/// # Arguments
///
/// * `temps` - Vec of temperatures in kelvin.
#[inline]
pub fn w2_color_correction(temp: f64) -> f64 {
    let temp = &temp.clamp(100.0, 400.0);
    (-8.60229377e-01 + 1.54988562e-02 * temp - 5.10705456e-05 * temp.powi(2)
        + 7.68314436e-08 * temp.powi(3)
        - 4.32238900e-11 * temp.powi(4))
    .recip()
}

/// Calculate the color correction factor for the W3 band to be applied to fluxes.
///
/// # Arguments
///
/// * `temps` - Vec of temperatures in kelvin.
#[inline]
pub fn w3_color_correction(temp: f64) -> f64 {
    let temp = &temp.clamp(100.0, 400.0);
    (-1.29814355 + 2.43268763e-02 * temp - 9.05178737e-05 * temp.powi(2)
        + 1.50095351e-07 * temp.powi(3)
        - 9.35433316e-11 * temp.powi(4))
    .recip()
}

/// Calculate the color correction factor for the W4 band to be applied to fluxes.
///
/// # Arguments
///
/// * `temps` - Vec of temperatures in kelvin.
#[inline]
pub fn w4_color_correction(temp: f64) -> f64 {
    let temp = &temp.clamp(100.0, 400.0).ln();
    2.17247804e+01 + -1.46084733e+01 * temp + 3.85364000e+00 * temp.powi(2)
        - 4.51512551e-01 * temp.powi(3)
        + 1.98397252e-02 * temp.powi(4)
}

/// The combined color corrections for WISE.
pub const WISE_CC: [ColorCorrFn; 4] = [
    &w1_color_correction,
    &w2_color_correction,
    &w3_color_correction,
    &w4_color_correction,
];
