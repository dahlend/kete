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

use super::sun::solar_flux_black_body;
use crate::{
    constants::{AU_KM, C_V},
    prelude::{Error, KeteResult},
};

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

/// This computes the phase curve correction using the IAU standard for the HG model.
///
/// Specifically page Page 550 - Equation (A4):
///
/// Asteroids II. University of Arizona Press, Tucson, pp. 524–556.
/// Bowell, E., Hapke, B., Domingue, D., Lumme, K., Peltoniemi, J., Harris,
/// A.W., 1989. Application of photometric models to asteroids, in: Binzel,
/// R.P., Gehrels, T., Matthews, M.S. (Eds.)
///
/// # Arguments
///
/// * `g_param` - The G parameter, between 0 and 1.
/// * `phase` - The phase angle in radians.
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
#[cfg_attr(feature = "pyo3", pyo3::pyfunction)]
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

/// Reflected light properties of an asteroid under the H/G magnitude system.
///
/// H, Albedo, and Diameter are all related by the relation:
/// `diameter = c_hg / albedo.sqrt() * (10f64).powf(-h_mag / 5.0)`
///
/// Specifically page Page 549 - Equation (A1) of:
///
/// Asteroids II. University of Arizona Press, Tucson, pp. 524–556.
/// Bowell, E., Hapke, B., Domingue, D., Lumme, K., Peltoniemi, J., Harris,
/// A.W., 1989. Application of photometric models to asteroids, in: Binzel,
/// R.P., Gehrels, T., Matthews, M.S. (Eds.)
///
/// # Arguments
/// * `desig` - Designation of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `vis_albedo` - Geometric albedo of the object.
/// * `diameter` - Diameter of the object in km.
/// * `c_hg` -  Relationship constant between H, D, and pV in km.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HGParams {
    /// Designation (name) of the object.
    pub desig: String,

    /// The G parameter of the object in the HG system.
    pub g_param: f64,

    /// The H parameter of the object in the HG system.
    pub h_mag: f64,

    /// Visible geometric albedo of the object.
    vis_albedo: Option<f64>,

    /// Diameter of the object in km.
    diam: Option<f64>,

    /// Relationship constant between H magnitudes/diameters/geometric albedo.
    ///
    /// Unit is km.
    ///
    /// See:
    ///     "Uncertainties on Asteroid Albedos Determined by Thermal Modeling"
    ///     J. R. Masiero, E. L. Wright, A. K. Mainzer - 2021
    ///     <https://iopscience.iop.org/article/10.3847/PSJ/abda4d/pdf>
    c_hg: f64,
}

impl HGParams {
    /// Create a new [`HGParams`] object without albedo or diameter.
    ///
    /// # Arguments
    ///
    /// * `desig` - Designation of the object.
    /// * `g_param` - The G parameter in the HG system.
    /// * `h_mag` - The H parameter of the object in the HG system.
    /// * `c_hg` - The relationship constant of the H-D-pV conversion in km.
    pub fn new(desig: String, g_param: f64, h_mag: f64, c_hg: Option<f64>) -> Self {
        let c_hg = c_hg.unwrap_or(C_V);
        Self {
            desig,
            g_param,
            h_mag,
            c_hg,
            vis_albedo: None,
            diam: None,
        }
    }

    /// Create a new [`HGParams`] object, attempting to fill in any missing parameters.
    ///
    /// `h_mag` must either be provided, or be computable from the albedo and diameter.
    ///
    /// H, Albedo, and Diameter are all related by the relation:
    /// `diameter = c_hg / albedo.sqrt() * (10f64).powf(-h_mag / 5.0)`
    ///
    /// This means if 2 are provided, the third may be computed, which is what this
    /// function enables.
    ///
    /// This will fail in two cases:
    /// - If `h_mag` is [`None`] and there is not enough information to compute it.
    /// - All 3 optional parameters are provided, but not self consistent.
    ///
    /// The `c_hg` parameter defaults to the [`C_V`] constant if not provided.
    ///
    /// # Arguments
    ///
    /// * `desig` - Designation of the object.
    /// * `g_param` - The G parameter in the HG system.
    /// * `h_mag` - The H parameter of the object in the HG system.
    /// * `c_hg` - The relationship constant of the H-D-pV conversion in km.
    /// * `vis_albedo` - Visible geometric albedo of the object.
    /// * `diameter` - Diameter of the object in km.
    ///
    pub fn try_new(
        desig: String,
        g_param: f64,
        h_mag: Option<f64>,
        c_hg: Option<f64>,
        vis_albedo: Option<f64>,
        diam: Option<f64>,
    ) -> KeteResult<Self> {
        let (h_mag, vis_albedo, diam, c_hg) = Self::fill(h_mag, vis_albedo, diam, c_hg)?;
        Ok(Self {
            desig,
            g_param,
            h_mag,
            c_hg,
            vis_albedo,
            diam,
        })
    }

    /// New [`HGParams`] assuming G param is `0.15` and ``c_hg`` is the V band value.
    pub fn default(desig: String, h_mag: f64) -> Self {
        Self::new(desig, 0.15, h_mag, Some(C_V))
    }

    /// Diameter of the object in km.
    pub fn diam(&self) -> Option<f64> {
        self.diam
    }

    /// Visible geometric albedo of the object.
    pub fn vis_albedo(&self) -> Option<f64> {
        self.vis_albedo
    }

    /// Try to fill in the parameters as much as possible.
    /// If there is internal data inconsistency, return an error.
    ///
    /// The `c_hg` parameter defaults to the [`C_V`] constant if not provided.
    ///
    /// This returns `h_mag`, `vis_albedo`, `diam`,  and `c_hg`.
    ///
    /// # Arguments
    ///
    /// * `h_mag` - The H parameter of the object in the HG system.
    /// * `vis_albedo` - Visible geometric albedo of the object.
    /// * `diameter` - Diameter of the object in km.
    /// * `c_hg` - The relationship constant of the H-D-pV conversion in km.
    ///
    fn fill(
        h_mag: Option<f64>,
        vis_albedo: Option<f64>,
        diam: Option<f64>,
        c_hg: Option<f64>,
    ) -> KeteResult<(f64, Option<f64>, Option<f64>, f64)> {
        if h_mag.is_none() && (vis_albedo.is_none() || diam.is_none()) {
            Err(Error::ValueError(
                "h_mag must be defined unless both vis_albedo and diam are provided.".into(),
            ))?;
        }
        // H mag is either defined, or computable.

        // first check if c_hg is given, otherwise use default assumed value
        let c_hg = c_hg.unwrap_or(C_V);

        if vis_albedo.is_none() && diam.is_none() {
            if let Some(h) = h_mag {
                // h_mag is defined, but un-computable
                return Ok((h, None, None, c_hg));
            }
        } else if h_mag.is_none() {
            // h_mag is undefined, but computable
            let diam = diam.unwrap();
            let albedo = vis_albedo.unwrap();
            let h_mag: f64 = -5.0 * (diam * albedo.sqrt() / c_hg).log10();
            return Ok((h_mag, Some(albedo), Some(diam), c_hg));
        }

        // h_mag is defined AND at least one other term is defined.
        // At this point there are 3 possibilities:
        // - diam and albedo are defined.
        // - albedo only is defined.
        // - diam only is defined.

        let h_mag = h_mag.unwrap();

        if let Some(albedo) = vis_albedo {
            // h is defined and albedo is defined, meaning diameter may be calculated.
            let expected_diam = c_hg / albedo.sqrt() * (10_f64).powf(-0.2 * h_mag);
            if let Some(diam) = diam {
                if (expected_diam - diam).abs() > 1e-8 {
                    Err(Error::ValueError(format!(
                        "Provided diameter doesn't match with computed diameter. {expected_diam} != {diam}"
                    )))?;
                }
            }
            return Ok((h_mag, Some(albedo), Some(expected_diam), c_hg));
        }

        // At this point we know that H is defined, diameter must be defined, but
        // albedo is not.

        let diam = diam.unwrap();

        let expected_albedo = (c_hg * 10_f64.powf(-0.2 * h_mag) / diam)
            .powi(2)
            .clamp(0.0, 1.0);

        // expected albedo is now calculated and possibly matches actual diameter if it exists.
        Ok((h_mag, Some(expected_albedo), Some(diam), c_hg))
    }

    /// Compute the apparent magnitude of an object using the absolute magnitude H, G, and
    /// positional information.
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
    /// * `sun2obj` - Vector from the sun to the object in AU.
    /// * `sun2obs` - Vector from the sun to the observer in AU.
    ///
    pub fn apparent_mag(&self, sun2obj: &Vector3<f64>, sun2obs: &Vector3<f64>) -> f64 {
        let obj_r = sun2obj.norm();
        let obj2obs = sun2obs - sun2obj;
        let obj2obs_r = obj2obs.norm();
        let phase = sun2obj.angle(&-obj2obs);

        // 2.7925... == 160 degrees in radians
        if phase > 2.792526803190927 {
            return f64::NAN;
        }

        let correction = hg_phase_curve_correction(self.g_param, phase).log10();
        self.h_mag + 5.0 * (obj_r * obj2obs_r).log10() - 2.5 * correction
    }

    /// Calculate the reflected light flux from an object using the IAU phase correction curve.
    ///
    /// This assumes that the object is an ideal disk facing the sun and applies the IAU
    /// correction curve to the reflected light, returning units of Jy per unit frequency.
    ///
    /// This flux calculation accepts a wavelength, which can be used to estimate the
    /// flux outside of band definition which is implicit in the HG system when querying the
    /// Minor Planet Center or JPL Horizons. The assumptions made here are that the Sun is an
    /// ideal black body, and the the phase correction curve defined by the G parameter is
    /// valid for the wavelength provided. Neither of these are precisely true, but are a
    /// good first order approximation.
    ///
    /// This treats the sun as a flat black body disk, which is a good approximation as long
    /// as the object is several solar radii away.
    ///
    /// The IAU model is not technically defined above 120 degrees phase, however this will
    /// continue to return values fit to the model until 160 degrees. Phases larger than
    /// 160 degrees will return a flux of 0.
    ///
    /// # Arguments
    ///
    /// * `sun2obj` - Vector from the sun to the object in AU.
    /// * `sun2obs` - Vector from the sun to the observer in AU.
    /// * `wavelength` - Central wavelength of light in nm.
    /// * `albedo` - Geometric Albedo at the wavelength provided.
    pub fn apparent_flux(
        &self,
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
        wavelength: f64,
        albedo: f64,
    ) -> Option<f64> {
        let obj2obs = sun2obs - sun2obj;

        let phase = sun2obj.angle(&-obj2obs);

        // 2.7925... == 160 degrees in radians
        if phase > 2.792526803190927 {
            return Some(0.0);
        }

        let diameter = self.diam()?;

        // Jy
        let flux_at_object = solar_flux_black_body(sun2obj.norm(), wavelength);

        // total Flux from the object, treating the object as a lambertian sphere
        // Jy * km^2
        let object_flux_total = flux_at_object * (diameter / 2.0).powi(2);

        let sc2obj_r_km = obj2obs.norm() * AU_KM;

        let correction = hg_phase_curve_correction(self.g_param, phase) * albedo;

        // Jy
        Some(correction * object_flux_total / sc2obj_r_km.powi(2))
    }
}
