// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

use crate::sun::solar_flux_black_body;
use kete_core::constants::{AU_KM, C_V};
use kete_core::errors::{Error, KeteResult};

use nalgebra::Vector3;

/// This computes the phase curve correction using the IAU standard for the HG model.
///
/// Specifically page Page 550 - Equation (A4):
///
/// Asteroids II. University of Arizona Press, Tucson, pp. 524-556.
/// Bowell, E., Hapke, B., Domingue, D., Lumme, K., Peltoniemi, J., Harris,
/// A.W., 1989. Application of photometric models to asteroids, in: Binzel,
/// R.P., Gehrels, T., Matthews, M.S. (Eds.)
///
/// # Arguments
///
/// * `g_param` - The G parameter, between 0 and 1.
/// * `phase` - The phase angle in radians.
#[must_use]
pub fn hg_phase_curve_correction(g_param: f64, phase: f64) -> f64 {
    fn helper(a: f64, b: f64, c: f64, phase: f64) -> f64 {
        let phase_s = phase.sin();
        let theta_l = (-a * (0.5 * phase).tan().powf(b)).exp();
        let theta_s = 1.0 - c * phase_s / (0.119 + 1.341 * phase_s - 0.754 * phase_s.powi(2));
        let w = (-90.56 * (0.5 * phase).tan().powi(2)).exp();
        w * theta_s + (1.0 - w) * theta_l
    }

    (1.0 - g_param) * helper(3.332, 0.631, 0.986, phase)
        + g_param * helper(1.862, 1.218, 0.238, phase)
}

/// Phase correction curve for cometary dust from the following paper:
///
/// A composite phase function for cometary dust comae
/// Bertini, Ivano, et al.
/// Planetary and Space Science (2025): 106164.
///
/// This uses the fitted values from the paper for `k=0.80`, `g_f=0.944`, `g_b=-0.542`.
///
/// An additional normalization has been applied so that the value of this is 1.0 at
/// 0.0 phase angle.
///
/// # Arguments
/// * `phase_angle` - Phase angle in radians.
///
#[must_use]
pub fn cometary_dust_phase_curve_correction(phase_angle: f64) -> f64 {
    const K: f64 = 0.80;
    const G_F: f64 = 0.944;
    const G_B: f64 = -0.542;

    // normalization constant to make this function return 1.0 at 0 phase angle
    const NORM: f64 = 3.3466826486608836;
    // Equation (3) from the paper.
    // Note that phase_angle is 180 - scattering angle, so cos -> -cos
    fn hg_normalized(phase_angle: f64, g: f64) -> f64 {
        let g2 = g.powi(2);
        ((1.0 + g2) / (1.0 + g2 + 2.0 * g * phase_angle.cos())).powf(1.5)
    }
    // Equation (4)
    let v = K * hg_normalized(phase_angle, G_F) + (1.0 - K) * hg_normalized(phase_angle, G_B);
    v / NORM
}

/// Compute H magnitude from diameter and geometric albedo.
///
/// # Arguments
///
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `c_hg` - Relationship constant between H, D, and pV in km.
#[must_use]
pub fn h_mag_from_diam_albedo(diameter: f64, vis_albedo: f64, c_hg: f64) -> f64 {
    -5.0 * (diameter * vis_albedo.sqrt() / c_hg).log10()
}

/// Compute diameter from H magnitude and geometric albedo.
///
/// # Arguments
///
/// * `h_mag` - Absolute magnitude.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `c_hg` - Relationship constant between H, D, and pV in km.
#[must_use]
pub fn diam_from_h_mag_albedo(h_mag: f64, vis_albedo: f64, c_hg: f64) -> f64 {
    c_hg / vis_albedo.sqrt() * 10_f64.powf(-0.2 * h_mag)
}

/// Compute geometric albedo from H magnitude and diameter.
///
/// The result is clamped to [0, 1].
///
/// # Arguments
///
/// * `h_mag` - Absolute magnitude.
/// * `diameter` - Diameter of the object in km.
/// * `c_hg` - Relationship constant between H, D, and pV in km.
#[must_use]
pub fn albedo_from_h_mag_diam(h_mag: f64, diameter: f64, c_hg: f64) -> f64 {
    (c_hg * 10_f64.powf(-0.2 * h_mag) / diameter)
        .powi(2)
        .clamp(0.0, 1.0)
}

/// Compute the apparent magnitude of an object using the HG system.
///
/// The IAU model is not technically defined above 120 degrees phase, however this will
/// continue to return values fit to the model until 160 degrees. Phases larger than
/// 160 degrees will return nan.
///
/// Note that this typically assumes that H/G have been fit in the V band, thus this
/// will return a V band apparent magnitude.
///
/// # Arguments
///
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `sun2obj` - Vector from the sun to the object in AU.
/// * `sun2obs` - Vector from the sun to the observer in AU.
#[must_use]
pub fn hg_apparent_mag(
    g_param: f64,
    h_mag: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> f64 {
    let obj_r = sun2obj.norm();
    let obj2obs = sun2obs - sun2obj;
    let obj2obs_r = obj2obs.norm();
    let phase = sun2obj.angle(&-obj2obs);

    // 2.7925... == 160 degrees in radians
    if phase > 2.792526803190927 {
        return f64::NAN;
    }

    let correction = hg_phase_curve_correction(g_param, phase).log10();
    h_mag + 5.0 * (obj_r * obj2obs_r).log10() - 2.5 * correction
}

/// Calculate the reflected light flux from an object using the IAU HG phase correction.
///
/// This assumes that the object is an ideal disk facing the sun and applies the IAU
/// correction curve to the reflected light, returning units of Jy per unit frequency.
///
/// This flux calculation accepts a wavelength, which can be used to estimate the
/// flux outside of band definition which is implicit in the HG system when querying the
/// Minor Planet Center or JPL Horizons. The assumptions made here are that the Sun is
/// an ideal black body, and the the phase correction curve defined by the G parameter
/// is valid for the wavelength provided. Neither of these are precisely true, but are a
/// good first order approximation.
///
/// The IAU model is not technically defined above 120 degrees phase, however this will
/// continue to return values fit to the model until 160 degrees. Phases larger than
/// 160 degrees will return a flux of 0.
///
/// # Arguments
///
/// * `g_param` - The G parameter in the HG system.
/// * `diameter` - Diameter of the object in km.
/// * `sun2obj` - Vector from the sun to the object in AU.
/// * `sun2obs` - Vector from the sun to the observer in AU.
/// * `wavelength` - Central wavelength of light in nm.
/// * `albedo` - Geometric Albedo at the wavelength provided.
#[must_use]
pub fn hg_apparent_flux(
    g_param: f64,
    diameter: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    wavelength: f64,
    albedo: f64,
) -> f64 {
    let obj2obs = sun2obs - sun2obj;

    let phase = sun2obj.angle(&-obj2obs);

    // 2.7925... == 160 degrees in radians
    if phase > 2.792526803190927 {
        return 0.0;
    }

    // Jy
    let flux_at_object = solar_flux_black_body(sun2obj.norm(), wavelength);

    // total Flux from the object, treating the object as a lambertian sphere
    // Jy * km^2
    let object_flux_total = flux_at_object * (diameter / 2.0).powi(2);

    let sc2obj_r_km = obj2obs.norm() * AU_KM;

    let correction = hg_phase_curve_correction(g_param, phase) * albedo;

    // Jy
    correction * object_flux_total / sc2obj_r_km.powi(2)
}

/// Resolve H magnitude, visible albedo, and diameter from any 2-of-3, with validation.
///
/// H, Albedo, and Diameter are all related by the relation:
/// `diameter = c_hg / albedo.sqrt() * (10f64).powf(-h_mag / 5.0)`
///
/// This means if 2 are provided, the third may be computed.
///
/// This will fail in two cases:
/// - If `h_mag` is [`None`] and there is not enough information to compute it.
/// - All 3 optional parameters are provided, but not self consistent.
///
/// The `c_hg` parameter defaults to the [`C_V`] constant if not provided.
///
/// Returns `(h_mag, vis_albedo, diameter, c_hg)`.
///
/// # Arguments
///
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `diameter` - Diameter of the object in km.
/// * `c_hg` - The relationship constant of the H-D-pV conversion in km.
///
/// # Errors
/// This can fail if parameters are not self consistent between H, diameter, and albedo.
#[allow(
    clippy::missing_panics_doc,
    reason = "Unwraps are guarded by prior checks"
)]
pub fn resolve_hg_params(
    h_mag: Option<f64>,
    vis_albedo: Option<f64>,
    diameter: Option<f64>,
    c_hg: Option<f64>,
) -> KeteResult<(f64, Option<f64>, Option<f64>, f64)> {
    if h_mag.is_none() && (vis_albedo.is_none() || diameter.is_none()) {
        Err(Error::ValueError(
            "h_mag must be defined unless both vis_albedo and diameter are provided.".into(),
        ))?;
    }

    let c_hg = c_hg.unwrap_or(C_V);

    if vis_albedo.is_none() && diameter.is_none() {
        if let Some(h) = h_mag {
            return Ok((h, None, None, c_hg));
        }
    } else if h_mag.is_none() {
        let diameter = diameter.unwrap();
        let albedo = vis_albedo.unwrap();
        let h_mag = h_mag_from_diam_albedo(diameter, albedo, c_hg);
        return Ok((h_mag, Some(albedo), Some(diameter), c_hg));
    }

    let h_mag = h_mag.unwrap();

    if let Some(albedo) = vis_albedo {
        let expected_diam = diam_from_h_mag_albedo(h_mag, albedo, c_hg);
        if let Some(diameter) = diameter
            && (expected_diam - diameter).abs() > 1e-8
        {
            Err(Error::ValueError(format!(
                "Provided diameter doesn't match with computed diameter. {expected_diam} != {diameter}"
            )))?;
        }
        return Ok((h_mag, Some(albedo), Some(expected_diam), c_hg));
    }

    let diameter = diameter.unwrap();
    let expected_albedo = albedo_from_h_mag_diam(h_mag, diameter, c_hg);
    Ok((h_mag, Some(expected_albedo), Some(diameter), c_hg))
}
