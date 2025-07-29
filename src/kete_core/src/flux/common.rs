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

use crate::io::FileIO;
use nalgebra::{UnitVector3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

use crate::constants::{
    AU_KM, NEOS_BANDS, NEOS_SUN_CORRECTION, NEOS_ZERO_MAG, SOLAR_FLUX, STEFAN_BOLTZMANN,
    WISE_BANDS_300K, WISE_CC, WISE_SUN_CORRECTION, WISE_ZERO_MAG_300K,
};

/// A function which computes the color correction on a single facet for NEATM and FRM.
/// These functions must accept a temperature in kelvin, and return a value (typically
/// near 1.0), which scales the final flux seen by the observer for that facet.
pub type ColorCorrFn = &'static (dyn Fn(f64) -> f64 + Sync);

/// Observer specific Flux properties.
/// Such as band information, zero mags, and color corrections.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum ObserverBands {
    /// Wise specific observer properties
    Wise,

    /// NEOS specific observer properties
    Neos,

    /// Generic observer properties
    /// This cannot store color correction functions.
    Generic {
        /// Effective central band wavelengths in nm
        bands: Vec<f64>,

        /// Zero magnitudes for the listed bands, optional.
        zero_mags: Option<Vec<f64>>,

        /// Flux correction for each of the bands for reflected light from the sun.
        /// 1.0 means no change.
        solar_correction: Vec<f64>,
    },
}

impl ObserverBands {
    /// Effective central band wavelengths in nm
    pub fn band_wavelength(&self) -> &[f64] {
        match self {
            Self::Neos => &NEOS_BANDS,
            Self::Wise => &WISE_BANDS_300K,
            Self::Generic { bands, .. } => bands,
        }
    }

    /// Zero magnitudes for the wavelength bands, optional.
    pub fn zero_mags(&self) -> Option<&[f64]> {
        match self {
            Self::Neos => Some(&NEOS_ZERO_MAG),
            Self::Wise => Some(&WISE_ZERO_MAG_300K),
            Self::Generic { zero_mags, .. } => zero_mags.as_deref(),
        }
    }

    /// Flux correction for each of the bands for reflected light from the sun.
    pub fn solar_correction(&self) -> &[f64] {
        match self {
            Self::Neos => &NEOS_SUN_CORRECTION,
            Self::Wise => &WISE_SUN_CORRECTION,
            Self::Generic {
                solar_correction, ..
            } => solar_correction,
        }
    }

    /// Color correction for the thermal light for each band.
    pub fn color_correction(&self) -> Option<&[ColorCorrFn]> {
        match self {
            Self::Neos => None,
            Self::Wise => Some(&WISE_CC),
            Self::Generic { .. } => None,
        }
    }
}

/// Output of a flux calculation.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelResults {
    /// Total output fluxes in Jy, this is a sum of thermal and HG contributions.
    pub fluxes: Vec<f64>,

    /// The expected magnitude if zero magnitudes are available. These magnitudes are
    /// from the combined fluxes, not individual terms.
    pub magnitudes: Option<Vec<f64>>,

    /// Fluxes in Jy from black body emission.
    pub thermal_fluxes: Vec<f64>,

    /// Fluxes in Jy as a result of the HG model, these are computed for the specified
    /// wavelength, assuming the HG is still valid in that regime.
    pub hg_fluxes: Vec<f64>,

    /// V band magnitude as computed from the HG visible model.
    pub v_band_magnitude: f64,

    /// V band flux in Jy as computed from the HG visible model.
    /// Using Vega as the zero mag point.
    pub v_band_flux: f64,
}

impl FileIO for ModelResults {}

impl ModelResults {
    /// Compute what fraction of the total flux is due to the HG reflection model.
    pub fn reflected_fraction(&self) -> Vec<f64> {
        let mut frac = vec![0.0; self.fluxes.len()];
        frac.iter_mut()
            .zip(self.fluxes.iter())
            .zip(self.hg_fluxes.iter())
            .for_each(|((f, total_flux), vis_flux)| *f = vis_flux / total_flux);
        frac
    }
}

/// Calculate the black body radiation for a list of temperatures at the specified
/// wavelength.
///
/// Flux is always set to 0.0 if wavelength is less than 10 nm or temperature is less
/// than 30 kelvin. This improves performance of the thermal models by about 2-3x as
/// long as the wavelength stays less than around 20 um. This implementation is not
/// tuned for radio astronomy.
///
/// Flux is in units Janskys / steradian.
/// This can be converted to being per unit wavelength by multiplying the result of
/// this by `c / wavelength**2`.
///
/// # Arguments
/// * `temp` - Temperature in Kelvin.
/// * `wavelength` - Wavelength in nm.
///
/// # Returns
/// * Flux in Jy / steradian.
#[inline(always)]
pub fn black_body_flux(temp: f64, wavelength: f64) -> f64 {
    if wavelength < 10.0 || temp < 30.0 {
        return 0.0;
    }
    // juggle some scaling terms to make things more numerically stable.
    let wavelength_um = wavelength * 1e-3;

    let exponent = 1.43877688e4 / (wavelength_um * temp);

    let coef = 397.289171e17 / wavelength_um.powi(3);
    // technically exp_m1 may be a better choice here, but it is ~50% slower than pure
    // exp - 1
    // The exponent should be well behaved unless the wavelength or temperature are
    // very large
    coef / (exponent.exp() - 1.0)
}

/// Calculate the total flux visible from an observers position from a Lambertian
/// surface.
///
/// Surfaces which are facing away from the observer do not contribute any flux, and
/// there is no computation of shadowing effects.
///
/// # Arguments
///
/// * `facet_flux` - Flux in Jy / steradian.
/// * `facet_normal` - The facet normal vector, this must be unit length.
/// * `obs2obj` - The vector from the observer to the object unit length.
/// * `obs2obj_r` - Distance from the observer to the object in AU.
/// * `diameter` - Diameter of the object in km.
/// * `emissivity` - The emissivity of surface of the object.
#[inline(always)]
pub fn lambertian_flux(
    facet_normal: &UnitVector3<f64>,
    obs2obj: &UnitVector3<f64>,
    facet_flux: &f64,
    obs2obj_r: &f64,
    diameter: &f64,
    emissivity: &f64,
) -> f64 {
    lambertian_vis_scale_factor(facet_normal, obs2obj, obs2obj_r, diameter, emissivity) * facet_flux
}

/// Scale factor for emitted flux from a lambertian surface.
/// This is a helper function for [`lambertian_flux`], see that function for
/// more description.
///
/// This is broken out into its own function because of [`crate::flux::FrmParams`] and
/// [`crate::flux::NeatmParams`] running over the same geometry for different
/// wavelengths.
/// This allows this to be computed once per geometry, but then multiple wavelengths
/// be multiplied against it. This resulted in a 50% speedup in FRM and NEATM overall.
#[inline(always)]
pub fn lambertian_vis_scale_factor(
    facet_normal: &UnitVector3<f64>,
    obs2obj: &UnitVector3<f64>,
    obs2obj_r: &f64,
    diameter: &f64,
    emissivity: &f64,
) -> f64 {
    // effective scaling due to distance from the observer
    let scale = (obs2obj_r * AU_KM / diameter).powi(-2);

    // flipping direction of observer vector
    let observed = -facet_normal.dot(obs2obj);
    if observed > 0.0 {
        return observed * emissivity * PI * scale;
    }
    0.0
}

/// Calculate the subsolar temperature in Kelvin.
///
/// # Arguments
///
/// * `obj2sun` - Vector from the object to the sun in AU.
/// * `geom_albedo` - Geometric Albedo.
/// * `g_param` - The G phase parameter.
/// * `beaming` - Beaming of the object, this is geometry dependent.
/// * `emissivity` - The emissivity of the surface.
#[inline(always)]
pub fn sub_solar_temperature(
    obj2sun: &Vector3<f64>,
    geom_albedo: f64,
    g_param: f64,
    beaming: f64,
    emissivity: f64,
) -> f64 {
    let phase_integral = 0.29 + 0.684 * g_param;

    let bond_albedo = geom_albedo * phase_integral;
    let reflected = (1.0 - bond_albedo) * SOLAR_FLUX / obj2sun.norm_squared();
    let temp = reflected / (beaming * emissivity * STEFAN_BOLTZMANN);
    if temp <= 0.0 {
        return 0.0;
    }
    temp.sqrt().sqrt()
}

/// Given a magnitude and a zero point magnitude in Jy, return the flux in Jy.
///
/// # Arguments
///
/// * `mag` - Magnitude.
/// * `zero_mag_flux` - Flux in Jy at which the Magnitude is 0.
pub fn mag_to_flux(mag: f64, mag_zero_flux: f64) -> f64 {
    10_f64.powf(mag / -2.5) * mag_zero_flux
}

/// Given a flux in Hy and a zero point magnitude in Jy, return the magnitude.
///
/// # Arguments
///
/// * `flux` - Flux in Jy.
/// * `zero_mag_flux` - Flux in Jy at which the Magnitude is 0.
pub fn flux_to_mag(flux: f64, mag_zero_flux: f64) -> f64 {
    -2.5 * (flux / mag_zero_flux).log10()
}

#[cfg(test)]
mod tests {

    use crate::constants::{C_M_PER_S, SOLAR_FLUX, STEFAN_BOLTZMANN};
    use crate::flux::*;
    use std::f64::consts::E;

    #[test]
    fn test_black_body_flux() {
        // A black body at 1000 Kelvin, at wavelength 20757:
        // At this point, the 1/expm1 portion of planck's law == 1.
        let wavelength = 20757.16266505257;
        let temp = 1000.0;

        // 1474 is 2 * h * c^3 * GHz in Jy / steradian
        // (see the wikipedia entry for planck's law)
        let rad = black_body_flux(temp, wavelength) / 1474.49946;

        // within 0.25 nm of expected.
        assert!((C_M_PER_S / rad.powf(1. / 3.) - wavelength).abs() < 0.25);

        // same test as above, but scaling temp so that the output is a factor of two higher
        let temp = 1000.0 * (2.0_f64).log(E) / (3.0_f64).log(E);
        let rad = black_body_flux(temp, wavelength) / 1474.49946 * 2.0;

        // within 0.25 nm of expected.
        assert!((C_M_PER_S / rad.powf(1. / 3.) - wavelength).abs() < 0.25);

        // Edge case tests
        assert_eq!(black_body_flux(0.0, 1000.0), 0.0);
        assert_eq!(black_body_flux(1000.0, 0.0), 0.0);
        assert_eq!(black_body_flux(1e-5, 1e-5), 0.0);
    }

    #[test]
    fn test_sub_solar_temperature() {
        let obj2sun = [1.0, 0.0, 0.0].into();

        // albedo, G set to make bond_albedo == 1
        let temp = sub_solar_temperature(&obj2sun, 1.0 / 0.29, 1.0, 1.0, 1.0);
        assert_eq!(temp, 0.0);

        for range in 1..10 {
            let obj2sun = [range as f64, 0.0, 0.0].into();
            let mut temp = sub_solar_temperature(&obj2sun, 0.0, 0.0, 1.0, 1.0);
            temp = temp.powi(4);
            let expected = SOLAR_FLUX / (range as f64).powi(2) / STEFAN_BOLTZMANN;
            assert!((temp - expected).abs() < 1e-5);
        }
    }
}
