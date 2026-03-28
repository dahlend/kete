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

use crate::{
    BandInfo, ModelResults, black_body_flux, flux_to_mag, hg_apparent_flux, hg_apparent_mag,
    mag_to_flux, sub_solar_temperature,
};
use kete_core::constants::{AU_KM, V_MAG_ZERO};

use nalgebra::{UnitVector3, Vector3};
use std::f64::consts::PI;
use std::sync::LazyLock;

/// Using the NEATM thermal model, calculate the temperature of each facet given the
/// direction of the sun, the subsolar temperature and the facet normal vectors.
///
/// # Arguments
///
/// * `facet_normal` - The facet normal vector, these must be unit length.
/// * `subsolar_temp` - The temperature at the sub-solar point in kelvin.
/// * `obj2sun` - The vector from the object to the sun, unit length.
#[inline(always)]
#[must_use]
pub fn neatm_facet_temperature(
    facet_normal: &UnitVector3<f64>,
    obj2sun: &UnitVector3<f64>,
    subsolar_temp: &f64,
) -> f64 {
    let tmp = facet_normal.dot(obj2sun);
    if tmp > 0.0 {
        return tmp.sqrt().sqrt() * subsolar_temp;
    }
    0.0
}

/// Number of Gauss-Legendre quadrature points for the 1D thermal integral.
const GL_ORDER: usize = 64;

/// Precomputed Gauss-Legendre nodes and weights on [0, 1].
static GL_POINTS: LazyLock<Box<[(f64, f64)]>> = LazyLock::new(|| gauss_legendre_unit(GL_ORDER));

/// Evaluate Legendre polynomial `P_n(x)` and its derivative `P_n'(x)`.
fn legendre_pd(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    let mut p0 = 1.0;
    let mut p1 = x;
    for k in 1..n {
        let kf = k as f64;
        let p2 = ((2.0 * kf + 1.0) * x * p1 - kf * p0) / (kf + 1.0);
        p0 = p1;
        p1 = p2;
    }
    let nf = n as f64;
    let dp = nf * (x * p1 - p0) / (x * x - 1.0);
    (p1, dp)
}

/// Compute n-point Gauss-Legendre nodes and weights on [0, 1].
fn gauss_legendre_unit(n: usize) -> Box<[(f64, f64)]> {
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let theta = PI * (4 * i + 3) as f64 / (4 * n + 2) as f64;
        let mut x = theta.cos();
        for _ in 0..100 {
            let (p, dp) = legendre_pd(n, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let (_, dp) = legendre_pd(n, x);
        let w = 2.0 / ((1.0 - x * x) * dp * dp);
        result.push((f64::midpoint(x, 1.0), w / 2.0));
    }
    result.into_boxed_slice()
}

/// Azimuthal visibility integral: the integral of max(cos(alpha), 0) over
/// phi in [0, 2*pi] for a ring at cos(theta) = u, given the phase angle
/// between the sub-solar and sub-observer directions.
///
/// `cos(alpha) = a + b*cos(phi)` where `a = u*cos_phase`, `b = sqrt(1-u^2)*sin_phase`.
#[inline(always)]
fn azimuthal_visibility(u: f64, cos_phase: f64, sin_phase: f64) -> f64 {
    let a = u * cos_phase;
    let b = (1.0 - u * u).max(0.0).sqrt() * sin_phase;
    if a >= b {
        2.0 * PI * a
    } else if a + b <= 0.0 {
        0.0
    } else {
        let ratio = (-a / b).clamp(-1.0, 1.0);
        2.0 * a * ratio.acos() + 2.0 * b * (1.0 - ratio * ratio).max(0.0).sqrt()
    }
}

/// Compute the NEATM thermal flux for each band.
///
/// The NEATM temperature distribution is azimuthally symmetric about the
/// sub-solar direction, so the 2D surface integral reduces to a 1D
/// Gauss-Legendre quadrature over cos(theta) after analytically integrating
/// out the azimuthal angle.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `beaming` - Beaming parameter.
/// * `emissivity` - Emissivity of the object.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
#[must_use]
pub fn neatm_thermal_flux(
    obs_bands: &[BandInfo],
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    beaming: f64,
    emissivity: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> Vec<f64> {
    let obj2sun = -sun2obj;
    let obs2obj = sun2obj - sun2obs;
    let obs2obj_r = obs2obj.norm();

    let ss_temp = sub_solar_temperature(obj2sun.norm(), vis_albedo, g_param, beaming, emissivity);

    let obj2sun_hat = UnitVector3::new_normalize(obj2sun);
    let obj2obs_hat = UnitVector3::new_normalize(-obs2obj);
    let cos_phase = obj2sun_hat.dot(&obj2obs_hat).clamp(-1.0, 1.0);
    let sin_phase = (1.0 - cos_phase * cos_phase).max(0.0).sqrt();

    let geo_scale = emissivity / 4.0 * (diameter / (obs2obj_r * AU_KM)).powi(2);

    let bands: Vec<_> = obs_bands.iter().map(|x| x.wavelength).collect();
    let color_correction: Vec<_> = obs_bands.iter().map(|x| x.color_correction).collect();

    let mut fluxes = vec![0.0; obs_bands.len()];
    for &(cos_theta, weight) in GL_POINTS.iter() {
        let temp = ss_temp * cos_theta.sqrt().sqrt();
        if temp < 30.0 {
            continue;
        }

        let vis = azimuthal_visibility(cos_theta, cos_phase, sin_phase);
        if vis <= 0.0 {
            continue;
        }

        let scaled_weight = weight * vis;
        for (idx, (wavelength, flux)) in bands.iter().zip(&mut fluxes).enumerate() {
            let mut bb = black_body_flux(temp, *wavelength);
            if let Some(func) = color_correction[idx] {
                bb *= func(temp);
            }
            *flux += scaled_weight * bb;
        }
    }

    for flux in &mut fluxes {
        *flux *= geo_scale;
    }
    fluxes
}

/// Compute NEATM thermal + reflected flux and magnitudes for each band.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `band_albedos` - Albedo of the object for each band.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `beaming` - Beaming parameter.
/// * `emissivity` - Emissivity of the object.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
#[must_use]
pub fn neatm_total_flux(
    obs_bands: &[BandInfo],
    band_albedos: &[f64],
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    h_mag: f64,
    beaming: f64,
    emissivity: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> ModelResults {
    let thermal_fluxes = neatm_thermal_flux(
        obs_bands, diameter, vis_albedo, g_param, beaming, emissivity, sun2obj, sun2obs,
    );

    let mut hg_fluxes = Vec::with_capacity(thermal_fluxes.len());
    let mut fluxes = Vec::with_capacity(thermal_fluxes.len());
    for ((band, t_flux), albedo) in obs_bands.iter().zip(&thermal_fluxes).zip(band_albedos) {
        let refl = hg_apparent_flux(
            g_param,
            diameter,
            sun2obj,
            sun2obs,
            band.wavelength,
            *albedo,
        ) * band.solar_correction;
        hg_fluxes.push(refl);
        fluxes.push(*t_flux + refl);
    }

    let v_band_magnitude = hg_apparent_mag(g_param, h_mag, sun2obj, sun2obs);
    let v_band_flux = mag_to_flux(v_band_magnitude, V_MAG_ZERO);

    let magnitudes: Vec<_> = obs_bands
        .iter()
        .zip(&fluxes)
        .map(|(band_info, flux)| flux_to_mag(*flux, band_info.zero_mag))
        .collect();

    ModelResults {
        fluxes,
        magnitudes,
        thermal_fluxes,
        hg_fluxes,
        v_band_magnitude,
        v_band_flux,
    }
}

#[cfg(test)]
mod tests {

    use crate::*;
    use nalgebra::UnitVector3;
    use std::f64::consts::PI;

    #[test]
    fn test_neatm_facet_temperature() {
        let obj2sun = UnitVector3::new_unchecked([1.0, 0.0, 0.0].into());
        let t = (PI / 4.0).cos().powf(0.25);

        let temp = neatm_facet_temperature(
            &UnitVector3::new_unchecked([1.0, 0.0, 0.0].into()),
            &obj2sun,
            &1.0,
        );
        assert!((temp - 1.0).abs() < 1e-8);

        let temp = neatm_facet_temperature(
            &UnitVector3::new_unchecked([0.0, 1.0, 0.0].into()),
            &obj2sun,
            &1.0,
        );
        assert!(temp.abs() < 1e-8);

        let temp = neatm_facet_temperature(
            &UnitVector3::new_unchecked([-1.0, 0.0, 0.0].into()),
            &obj2sun,
            &1.0,
        );
        assert!(temp.abs() < 1e-8);

        let temp = neatm_facet_temperature(
            &UnitVector3::new_normalize([1.0, 1.0, 0.0].into()),
            &obj2sun,
            &1.0,
        );
        assert!((temp - t).abs() < 1e-8);

        let temp = neatm_facet_temperature(
            &UnitVector3::new_normalize([1.0, -1.0, 0.0].into()),
            &obj2sun,
            &1.0,
        );
        assert!((temp - t).abs() < 1e-8);
        let fib_n1024 = ConvexShape::new_fibonacci_lattice(1028);
        let fib_n2048 = ConvexShape::new_fibonacci_lattice(2048);

        // Test with different geometry, answer should converge
        let t1: f64 = fib_n2048
            .facets
            .iter()
            .map(|facet| neatm_facet_temperature(&facet.normal, &obj2sun, &1.0))
            .sum();
        let t2: f64 = fib_n1024
            .facets
            .iter()
            .map(|facet| neatm_facet_temperature(&facet.normal, &obj2sun, &1.0))
            .sum();

        let t1: f64 = t1 / fib_n2048.facets.len() as f64;
        let t2: f64 = t2 / fib_n1024.facets.len() as f64;

        assert!((t1 - t2).abs() < 1e-2);
    }
}
