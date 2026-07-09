// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

//! Surface roughness via the spherical-cap crater model (Lagerros / Spencer 1990).
//!
//! Macroscopic roughness is represented by identical spherical-cap craters with a
//! given opening half-angle `gamma` (a proxy for RMS slope). The cap is a portion of
//! a unit sphere whose center sits at `(0, 0, cos gamma)`; the crater axis is `+z`
//! (the host facet normal). `gamma -> 0` is a flat surface; `gamma = pi/2` is a
//! hemisphere (maximum roughness).
//!
//! Two properties make this geometry the standard tractable choice:
//!
//! - **Constant view factor.** Between any two micro-facets on the interior of a
//!   sphere the differential view factor is `dA_j / (4 pi R^2)`, independent of
//!   position. So all intra-crater radiative exchange (IR self-heating and solar
//!   multiple scattering) collapses to a single scalar irradiance seen identically by
//!   every micro-facet, rather than an N x N radiosity matrix.
//! - **Analytic shadowing and visibility.** A micro-facet at `P` is reached by a
//!   direction `d` iff `n . d > 0` and the second sphere intersection
//!   `P + 2 R (n . d) d` exits the opening (its z is above the rim plane).
//!
//! This module provides only the geometry and the directional shadow/visibility
//! tests; the coupled thermal solve and rendering live with the TPM solver.
use nalgebra::Vector3;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// A discretized spherical-cap crater (unit sphere, `R = 1`).
#[derive(Debug, Clone)]
pub(crate) struct Crater {
    /// Outward micro-facet normals (crater-local frame, `+z` = axis).
    normals: Vec<Vector3<f64>>,
    /// Micro-facet surface areas (sum = `2 pi (1 - cos gamma)`).
    areas: Vec<f64>,
    /// Micro-facet center positions on the cap.
    positions: Vec<Vector3<f64>>,
    /// Projected area of the opening disk, `pi sin^2 gamma` -- the area of the flat
    /// facet this crater replaces.
    opening_area: f64,
}

impl Crater {
    /// Build a crater with the given opening half-angle, discretized into
    /// `n_rings` latitude rings (in `theta`) by `n_sectors` azimuthal sectors.
    ///
    /// # Panics
    ///
    /// Panics if `opening_angle` is not in `(0, pi/2]` or a resolution is below 1.
    #[must_use]
    pub(crate) fn new(opening_angle: f64, n_rings: usize, n_sectors: usize) -> Self {
        assert!(
            opening_angle > 0.0 && opening_angle <= FRAC_PI_2,
            "opening_angle must be in (0, pi/2]"
        );
        assert!(
            n_rings >= 1 && n_sectors >= 1,
            "need at least 1 ring/sector"
        );

        let center_z = opening_angle.cos();
        let d_theta = opening_angle / n_rings as f64;
        let d_psi = TAU / n_sectors as f64;

        let mut normals = Vec::with_capacity(n_rings * n_sectors);
        let mut areas = Vec::with_capacity(n_rings * n_sectors);
        let mut positions = Vec::with_capacity(n_rings * n_sectors);

        for r in 0..n_rings {
            // ring center polar angle (0 = crater bottom, gamma = rim)
            let theta = (r as f64 + 0.5) * d_theta;
            let (s_t, c_t) = theta.sin_cos();
            // area of a micro-facet in this ring: R^2 sin(theta) dtheta dpsi
            let area = s_t * d_theta * d_psi;
            for sct in 0..n_sectors {
                let psi = (sct as f64 + 0.5) * d_psi;
                let (s_p, c_p) = psi.sin_cos();
                // outward normal points up and toward the axis (concave surface)
                let normal = Vector3::new(-s_t * c_p, -s_t * s_p, c_t);
                // position P = C - n  (C = (0, 0, cos gamma), R = 1)
                let position = Vector3::new(s_t * c_p, s_t * s_p, center_z - c_t);
                normals.push(normal);
                areas.push(area);
                positions.push(position);
            }
        }

        let opening_area = PI * opening_angle.sin().powi(2);

        Self {
            normals,
            areas,
            positions,
            opening_area,
        }
    }

    /// Number of micro-facets.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.normals.len()
    }

    /// Micro-facet outward normals (crater-local frame). Production code reads
    /// the fields through the solver/render helpers; only tests need raw access.
    #[cfg(test)]
    pub(crate) fn normals(&self) -> &[Vector3<f64>] {
        &self.normals
    }

    /// Micro-facet areas. Test-only raw access, as with `normals`.
    #[cfg(test)]
    pub(crate) fn areas(&self) -> &[f64] {
        &self.areas
    }

    /// Projected area of the opening disk (`pi sin^2 gamma`).
    #[must_use]
    pub(crate) fn opening_area(&self) -> f64 {
        self.opening_area
    }

    /// Geometric coupling for intra-crater radiation: a micro-facet receives a
    /// uniform irradiance `H = recapture_sum(J) = sum_j area_j J_j / (4 pi)` from the
    /// rest of the cap (constant view factor). For uniform radiosity `J` this is
    /// `J * total_area / (4 pi)`.
    #[must_use]
    pub(crate) fn recapture_irradiance(&self, radiosity: &[f64]) -> f64 {
        debug_assert_eq!(
            radiosity.len(),
            self.areas.len(),
            "radiosity length must match the micro-facet count"
        );
        let weighted: f64 = self.areas.iter().zip(radiosity).map(|(a, j)| a * j).sum();
        weighted / (2.0 * TAU) // 4 pi
    }

    /// Per-micro-facet direct illumination factor `max(0, n . dir)` from a unit
    /// direction `dir` (in crater-local frame), zeroed where the rim shadows the
    /// facet. Use for the Sun direction.
    #[must_use]
    pub(crate) fn direct_factor(&self, dir: &Vector3<f64>) -> Vec<f64> {
        self.normals
            .iter()
            .zip(&self.positions)
            .map(|(n, p)| {
                let c = n.dot(dir);
                if c > 0.0 && exits_opening(p, dir, c) {
                    c
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Per-micro-facet projected area toward the observer direction `dir`
    /// (`max(0, n . dir) * area`), zeroed where the rim hides the facet from the
    /// observer.
    #[must_use]
    pub(crate) fn visible_projected_area(&self, dir: &Vector3<f64>) -> Vec<f64> {
        self.normals
            .iter()
            .zip(&self.positions)
            .zip(&self.areas)
            .map(|((n, p), area)| {
                let c = n.dot(dir);
                if c > 0.0 && exits_opening(p, dir, c) {
                    c * area
                } else {
                    0.0
                }
            })
            .collect()
    }
}

/// Whether a ray from micro-facet position `p` toward `dir` exits the opening rather
/// than re-intersecting the cap. The second sphere intersection is at
/// `p + 2 (n . dir) dir` (R = 1); it exits iff its z is above the rim plane.
fn exits_opening(p: &Vector3<f64>, dir: &Vector3<f64>, cos_inc: f64) -> bool {
    p.z + 2.0 * cos_inc * dir.z > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_area_sums_to_spherical_cap() {
        // total micro-facet area should match the analytic cap area 2 pi (1 - cos g)
        for &gamma in &[0.3, 0.8, FRAC_PI_2] {
            let crater = Crater::new(gamma, 40, 60);
            let analytic = TAU * (1.0 - gamma.cos());
            let total: f64 = crater.areas().iter().sum();
            assert!(
                (total - analytic).abs() / analytic < 1e-3,
                "gamma={gamma}: total {total} vs analytic {analytic}"
            );
        }
    }

    #[test]
    fn test_projected_area_equals_opening() {
        // Viewed straight down the axis with no shadowing, the crater's projected area
        // must equal the flat opening pi sin^2(gamma) (projected-area conservation).
        for &gamma in &[0.4, 1.0, FRAC_PI_2] {
            let crater = Crater::new(gamma, 60, 90);
            let axis = Vector3::new(0.0, 0.0, 1.0);
            let proj: f64 = crater.visible_projected_area(&axis).iter().sum();
            let expected = crater.opening_area();
            assert!(
                (proj - expected).abs() / expected < 1e-3,
                "gamma={gamma}: projected {proj} vs opening {expected}"
            );
        }
    }

    #[test]
    fn test_normal_incidence_fully_lit() {
        // Sun straight down the axis: no facet is shadowed, every up-facing facet is
        // lit (exit = cos gamma + cos theta > 0 always).
        let crater = Crater::new(1.2, 30, 48);
        let axis = Vector3::new(0.0, 0.0, 1.0);
        let direct = crater.direct_factor(&axis);
        for (f, n) in direct.iter().zip(crater.normals()) {
            // illuminated factor equals the raw cosine (no shadowing)
            assert!((f - n.z.max(0.0)).abs() < 1e-12);
        }
        assert!(direct.iter().all(|&f| f >= 0.0));
        assert!(direct.iter().any(|&f| f > 0.0));
    }

    #[test]
    fn test_grazing_sun_casts_shadows() {
        // A low sun (mostly horizontal) must shadow part of the crater: the directly
        // shadowed facets have a zero factor despite facing the sun geometrically.
        let crater = Crater::new(FRAC_PI_2, 30, 48); // hemisphere
        let dir = Vector3::new(0.95_f64, 0.0, 0.312).normalize(); // ~18 deg elevation
        let direct = crater.direct_factor(&dir);
        let shadowed = crater
            .normals()
            .iter()
            .zip(&direct)
            .filter(|(n, f)| n.dot(&dir) > 1e-3 && **f == 0.0)
            .count();
        assert!(
            shadowed > 0,
            "grazing sun should shadow some sunward facets"
        );
    }

    #[test]
    fn test_recapture_uniform_radiosity() {
        // Uniform radiosity J: irradiance = J * total_area / (4 pi). For a hemisphere
        // (total_area = 2 pi) that is J / 2.
        let crater = Crater::new(FRAC_PI_2, 20, 32);
        let j = vec![1.0; crater.len()];
        let h = crater.recapture_irradiance(&j);
        assert!(
            (h - 0.5).abs() < 1e-3,
            "hemisphere recapture should be 0.5, got {h}"
        );
    }
}
