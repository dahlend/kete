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

use crate::common::ThermalGeometry;
use crate::{BandInfo, ModelResults, assemble_total};
use kete_core::geometry::ConvexShape;
use std::sync::LazyLock;

use nalgebra::{UnitVector3, Vector3};
use std::f64::consts::PI;

/// Surface facets of the FRM quadrature.
static FRM_SHAPE: LazyLock<ConvexShape> =
    LazyLock::new(|| ConvexShape::new_fibonacci_lattice(2048));

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

/// Compute the FRM thermal flux for each band.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `emissivity` - Emissivity of the object.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
#[must_use]
pub fn frm_thermal_flux(
    obs_bands: &[BandInfo],
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    emissivity: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> Vec<f64> {
    let geometry = ThermalGeometry::frm(sun2obj, sun2obs);
    // FRM ignores the beaming argument; pi is its fixed beaming.
    obs_bands
        .iter()
        .map(|band| geometry.flux(band, diameter, vis_albedo, g_param, PI, emissivity))
        .collect()
}

/// FRM surface nodes `(weight, temp_fraction)` of an object at `sun2obj` seen
/// from `sun2obs`, both in AU from the Sun.
///
/// The nodes are the facets of [`FRM_SHAPE`] that are both heated and visible to
/// the observer, with the rotation pole along z. See [`ThermalGeometry`].
pub(crate) fn frm_nodes(sun2obj: &Vector3<f64>, sun2obs: &Vector3<f64>) -> Vec<(f64, f64)> {
    let obj2sun = UnitVector3::new_normalize(-sun2obj);
    let obs2obj_hat = UnitVector3::new_normalize(sun2obj - sun2obs);
    FRM_SHAPE
        .facets
        .iter()
        .filter_map(|facet| {
            let frac = frm_facet_temperature(&facet.normal, 1.0, &obj2sun);
            let observed = -facet.normal.dot(&obs2obj_hat);
            (frac > 0.0 && observed > 0.0).then_some((observed * PI * facet.area, frac))
        })
        .collect()
}

/// Compute FRM thermal + reflected flux and magnitudes for each band.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `band_albedos` - Albedo of the object for each band.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `emissivity` - Emissivity of the object.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
#[must_use]
pub fn frm_total_flux(
    obs_bands: &[BandInfo],
    band_albedos: &[f64],
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    h_mag: f64,
    emissivity: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> ModelResults {
    let thermal_fluxes = frm_thermal_flux(
        obs_bands, diameter, vis_albedo, g_param, emissivity, sun2obj, sun2obs,
    );
    assemble_total(
        obs_bands,
        band_albedos,
        thermal_fluxes,
        diameter,
        g_param,
        h_mag,
        sun2obj,
        sun2obs,
    )
}

#[cfg(test)]
mod tests {

    use nalgebra::UnitVector3;

    use super::*;
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
