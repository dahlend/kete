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

use super::{
    BandInfo, DEFAULT_SHAPE, HGParams,
    common::{ModelResults, black_body_flux, lambertian_vis_scale_factor, sub_solar_temperature},
    flux_to_mag,
};
use crate::constants::V_MAG_ZERO;

use core::f64;
use nalgebra::{UnitVector3, Vector3};
use std::f64::consts::PI;

/// Using the FRM thermal model, calculate the temperature of each facet given the
/// direction of the sun, the subsolar temperature and the facet normal vectors.
///
/// # Arguments
///
/// * `facet_normal` - The facet normal vector, these must be unit length.
/// * `subsolar_temp` - The temperature at the sub-solar point in kelvin.
/// * `obj2sun` - The vector from the object to the sun, unit vector.
#[inline(always)]
#[must_use]
pub fn frm_facet_temperature(
    facet_normal: &UnitVector3<f64>,
    subsolar_temp: f64,
    obj2sun: &UnitVector3<f64>,
) -> f64 {
    // since the facet normals are length 1, and the sun_norm vec is length one, the
    // angle difference is arcsin(z_sun) - arcsin(z_normal)

    let tmp = (facet_normal.z.asin() - obj2sun.z.asin()).cos();
    if tmp > 0.0 {
        return tmp.sqrt().sqrt() * subsolar_temp;
    }
    0.0
}

///  FRM input
#[derive(Debug, Clone)]
pub struct FrmParams {
    /// Wavelength band information of the observer.
    pub obs_bands: Vec<BandInfo>,

    /// Albedo of the object for each band.
    pub band_albedos: Vec<f64>,

    /// HG parameters defining the HG reflected light model.
    pub hg_params: HGParams,

    /// Emissivity of the object.
    pub emissivity: f64,
}

impl FrmParams {
    /// Create new [`FrmParams`] with WISE band and zero mags
    #[must_use]
    pub fn new_wise(albedos: [f64; 4], hg_params: HGParams, emissivity: f64) -> Self {
        Self {
            obs_bands: BandInfo::new_wise().to_vec(),
            band_albedos: albedos.to_vec(),
            hg_params,
            emissivity,
        }
    }

    /// Create new [`FrmParams`] with NEOS band and zero mags
    #[must_use]
    pub fn new_neos(albedos: [f64; 2], hg_params: HGParams, emissivity: f64) -> Self {
        Self {
            obs_bands: BandInfo::new_neos().to_vec(),
            band_albedos: albedos.to_vec(),
            hg_params,
            emissivity,
        }
    }

    /// Compute the Flux visible from an object using the FRM model.
    ///
    /// # Arguments
    ///
    /// * `sun2obj` - Position of the object with respect to the Sun in AU.
    /// * `sun2obs` - Position of the Observer with respect to the Sun in AU.
    #[must_use]
    pub fn apparent_thermal_flux(
        &self,
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
    ) -> Option<Vec<f64>> {
        let obj2sun = -sun2obj;
        let obs2obj = sun2obj - sun2obs;
        let obs2obj_r = obs2obj.norm();
        let geom = &DEFAULT_SHAPE;

        let diameter = self.hg_params.diam()?;
        let ss_temp = sub_solar_temperature(
            obj2sun.norm(),
            self.hg_params.vis_albedo()?,
            self.hg_params.g_param,
            PI,
            self.emissivity,
        );

        let bands: Vec<_> = self.obs_bands.iter().map(|x| x.wavelength).collect();
        let color_correction: Vec<_> = self.obs_bands.iter().map(|x| x.color_correction).collect();
        let obj2sun = UnitVector3::new_normalize(obj2sun);
        let obs2obj = UnitVector3::new_normalize(obs2obj);

        let mut fluxes = vec![0.0; self.obs_bands.len()];
        for facet in &geom.facets {
            let temp = frm_facet_temperature(&facet.normal, ss_temp, &obj2sun);
            let obs_flux_scaling = lambertian_vis_scale_factor(
                &facet.normal,
                &obs2obj,
                &obs2obj_r,
                &diameter,
                &self.emissivity,
            );
            if temp == 0.0 || obs_flux_scaling == 0.0 {
                continue;
            }
            for (idx, (wavelength, flux)) in bands.iter().zip(&mut fluxes).enumerate() {
                let mut facet_flux = black_body_flux(temp, *wavelength);
                if let Some(func) = color_correction[idx] {
                    facet_flux *= func(temp);
                }
                facet_flux *= facet.area;

                *flux += obs_flux_scaling * facet_flux;
            }
        }
        Some(fluxes)
    }

    /// Compute FRM with an reflected reflection model added on.
    ///
    /// # Arguments
    ///
    /// * `sun2obj` - Position of the object with respect to the Sun in AU.
    /// * `sun2obs` - Position of the Observer with respect to the Sun in AU.
    #[must_use]
    pub fn apparent_total_flux(
        &self,
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
    ) -> Option<ModelResults> {
        let thermal_fluxes = self.apparent_thermal_flux(sun2obj, sun2obs)?;

        let mut hg_fluxes = Vec::with_capacity(thermal_fluxes.len());
        let mut fluxes = Vec::with_capacity(thermal_fluxes.len());
        for ((band, t_flux), albedo) in self
            .obs_bands
            .iter()
            .zip(&thermal_fluxes)
            .zip(&self.band_albedos)
        {
            let refl = self
                .hg_params
                .apparent_flux(sun2obj, sun2obs, band.wavelength, *albedo)?
                * band.solar_correction;
            hg_fluxes.push(refl);
            fluxes.push(*t_flux + refl);
        }

        let v_band_magnitude = self.hg_params.apparent_mag(sun2obj, sun2obs);
        let v_band_flux = flux_to_mag(v_band_magnitude, V_MAG_ZERO);

        let magnitudes: Vec<_> = self
            .obs_bands
            .iter()
            .zip(&fluxes)
            .map(|(band_info, flux)| flux_to_mag(*flux, band_info.zero_mag))
            .collect();

        Some(ModelResults {
            fluxes,
            magnitudes,
            thermal_fluxes,
            hg_fluxes,
            v_band_magnitude,
            v_band_flux,
        })
    }
}

#[cfg(test)]
mod tests {

    use nalgebra::UnitVector3;

    use super::*;
    use crate::flux::*;
    use std::f64::consts::PI;

    #[test]
    fn test_frm_facet_temperature() {
        let obj2sun = UnitVector3::new_unchecked([1.0, 0.0, 0.0].into());
        let t = (PI / 4.0).cos().powf(0.25);

        let temp = frm_facet_temperature(
            &UnitVector3::new_unchecked([1.0, 0.0, 0.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - 1.0).abs() < 1e-8);

        let temp = frm_facet_temperature(
            &UnitVector3::new_unchecked([0.0, 1.0, 0.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - 1.0).abs() < 1e-8);

        let temp = frm_facet_temperature(
            &UnitVector3::new_unchecked([-1.0, 0.0, 0.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - 1.0).abs() < 1e-8);

        let temp = frm_facet_temperature(
            &UnitVector3::new_normalize([1.0, 1.0, 0.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - 1.0).abs() < 1e-8);

        let temp = frm_facet_temperature(
            &UnitVector3::new_normalize([1.0, 0.0, 1.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - t).abs() < 1e-8);

        let temp = frm_facet_temperature(
            &UnitVector3::new_normalize([0.0, -1.0, 1.0].into()),
            1.0,
            &obj2sun,
        );
        assert!((temp - t).abs() < 1e-8);
        let fib_n1024 = ConvexShape::new_fibonacci_lattice(1028);
        let fib_n2048 = ConvexShape::new_fibonacci_lattice(2048);

        // Test with different geometry, answer should converge
        let t1: f64 = fib_n2048
            .facets
            .iter()
            .map(|facet| frm_facet_temperature(&facet.normal, 1.0, &obj2sun))
            .sum();
        let t2: f64 = fib_n1024
            .facets
            .iter()
            .map(|facet| frm_facet_temperature(&facet.normal, 1.0, &obj2sun))
            .sum();

        let t1: f64 = t1 / fib_n2048.facets.len() as f64;
        let t2: f64 = t2 / fib_n1024.facets.len() as f64;

        assert!((t1 - t2).abs() < 1e-2);
    }
}
