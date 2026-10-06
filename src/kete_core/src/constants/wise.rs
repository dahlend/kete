// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! WISE specific functions.

/// The width of the WISE FOV, 47 arcminutes in radians.
pub const WISE_WIDTH: f64 = 0.01367174580728;

/// Calculate the color correction factor for the W1 band to be applied to fluxes.
///
/// # Arguments
///
/// * `temps` - Vec of temperatures in kelvin.
#[inline]
#[must_use]
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
#[must_use]
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
#[must_use]
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
#[must_use]
pub fn w4_color_correction(temp: f64) -> f64 {
    let temp = &temp.clamp(100.0, 400.0).ln();
    2.17247804e+01 + -1.46084733e+01 * temp + 3.85364000e+00 * temp.powi(2)
        - 4.51512551e-01 * temp.powi(3)
        + 1.98397252e-02 * temp.powi(4)
}
