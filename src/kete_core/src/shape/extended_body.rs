//! Bundling of an extended-gravity body's geometry, rotation, units, and
//! proximity-regime metadata.
//!
//! An [`ExtendedBody`] is the runtime representation of a body that
//! participates in close-proximity propagation: one or more constant-
//! density polyhedron components, a rotation model, a body-natural unit
//! system, the body's total `GM`, and a proximity radius beyond which
//! the heliocentric N-body machinery should take over.
//!
//! `ExtendedBody` itself is frame-agnostic: the caller is responsible
//! for ensuring that the rotation model and any inertial inputs
//! (perturber positions, body ephemeris) all live in the same inertial
//! reference frame (typically ICRF / J2000 equatorial).
//!
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

use nalgebra::Vector3;

use crate::errors::{Error, KeteResult};
use crate::shape::{BodyUnits, ExtendedGravity, Polyhedron, RotationModel};

/// An extended-gravity body for close-proximity propagation.
///
/// Holds one or more polyhedron components (constant density per
/// component), a rotation model, a body-natural unit system, the body's
/// aggregate `GM`, and a proximity radius beyond which the
/// heliocentric N-body machinery should take over.
///
/// All polyhedron components share the same body-fixed frame; the
/// caller is responsible for placing them in a consistent body-centered
/// coordinate system (typically with the aggregate center of mass at
/// the origin).
#[derive(Debug, Clone)]
pub struct ExtendedBody {
    /// Polyhedron components that together model the body's mass
    /// distribution.  Vertices and `gm` of each component are in the
    /// caller's chosen length / `gm` units, **not** body-natural units.
    pub components: Vec<Polyhedron>,
    /// Rotation between inertial and body-fixed frames.
    pub rotation: RotationModel,
    /// Body-natural unit system used by [`small_body_accel`] when
    /// scaling positions / accelerations into and out of the gravity
    /// evaluator's units.
    pub units: BodyUnits,
    /// Aggregate gravitational parameter `G * M_total` (sum of
    /// component `gm`).  Stored explicitly so it does not need to be
    /// recomputed on every force call.
    pub gm: f64,
    /// Proximity radius (AU).  Beyond this distance from the body
    /// center, the heliocentric N-body propagator should be used
    /// instead of the body-centric one.  The hand-off coordinator
    /// (Phase 6) uses this as its primary regime threshold.
    pub proximity_radius_au: f64,
}

impl ExtendedBody {
    /// Construct an [`ExtendedBody`] from a list of polyhedron
    /// components, a rotation model, a body-natural unit system, and a
    /// proximity radius (AU).  The aggregate `gm` is the sum of
    /// component `gm`s.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `components` is empty or
    /// `proximity_radius_au` is non-positive.
    pub fn try_new(
        components: Vec<Polyhedron>,
        rotation: RotationModel,
        units: BodyUnits,
        proximity_radius_au: f64,
    ) -> KeteResult<Self> {
        if components.is_empty() {
            return Err(Error::ValueError(
                "ExtendedBody requires at least one polyhedron component".into(),
            ));
        }
        if !(proximity_radius_au > 0.0 && proximity_radius_au.is_finite()) {
            return Err(Error::ValueError(
                "ExtendedBody proximity_radius_au must be positive and finite".into(),
            ));
        }
        let gm = components.iter().map(|c| c.gm).sum();
        Ok(Self {
            components,
            rotation,
            units,
            gm,
            proximity_radius_au,
        })
    }

    /// Aggregate gravitational acceleration at body-fixed position
    /// `r_body` (in the same length units as the polyhedron
    /// components).  Components are summed linearly.
    ///
    /// # Errors
    /// Propagates [`Error::SurfaceImpact`] from any component.
    pub fn body_acceleration(&self, r_body: Vector3<f64>) -> KeteResult<Vector3<f64>> {
        let mut a = Vector3::zeros();
        for c in &self.components {
            a += c.acceleration(r_body)?;
        }
        Ok(a)
    }

    /// True if any component reports `r_body` as inside its surface.
    #[must_use]
    pub fn contains(&self, r_body: Vector3<f64>) -> bool {
        self.components.iter().any(|c| c.contains(r_body))
    }
}

/// Threshold below which the tidal third-body acceleration is computed
/// using the gradient expansion rather than direct subtraction.
///
/// When `|r_particle| / d_perturber` is small, the direct difference
/// `r_pp/|r_pp|^3 - r_pb/|r_pb|^3` suffers catastrophic cancellation.
/// The gradient form `-(r - 3 (d_hat . r) d_hat) / d^3` is exact at
/// leading order and unconditionally well-conditioned.
const TIDAL_GRADIENT_THRESHOLD: f64 = 1e-4;

/// Compute the body-centric acceleration of a test particle at
/// position `r_body_units` (body-natural length units, i.e. AU /
/// `body.units.length_au`) in the body's body-fixed frame, including:
///
/// 1. Aggregate gravitational acceleration from all polyhedron
///    components of `body`, evaluated in the body-fixed frame.
/// 2. Tidal perturbations from each `(gm_solar, r_inertial_au)` entry
///    in `perturbers`, where positions are inertial AU **relative to
///    the body's center**.  Use the tidal-difference form for
///    well-separated perturbers and the gradient expansion for very
///    close field points.
///
/// `jd_tdb` is the TDB Julian Day used to evaluate the body's rotation
/// model.  The returned acceleration is in the body's body-fixed frame
/// expressed in `body_length / day^2`.  Time stays in days throughout
/// (so the result can be plugged directly into an integrator whose
/// time axis is TDB Julian Day); only spatial coordinates are scaled
/// to body-natural length units.
///
/// All inertial inputs and the rotation model must share the same
/// inertial reference frame (typically ICRF / J2000 equatorial).
///
/// # Errors
/// Propagates [`Error::SurfaceImpact`] if the field point lies on the
/// surface of any polyhedron component.
pub fn small_body_accel(
    body: &ExtendedBody,
    jd_tdb: f64,
    r_body_units: Vector3<f64>,
    perturbers: &[(f64, Vector3<f64>)],
) -> KeteResult<Vector3<f64>> {
    // 1. Direct body gravity in the body-fixed frame.
    //    The polyhedron components live in caller-supplied length units
    //    (typically AU); convert the field point from body-natural
    //    lengths back to AU for the gravity evaluation.
    let r_body_au = body.units.pos_from_body(r_body_units);
    // Reject interior field points up front; the closed-form
    // polyhedron evaluator only flags exact surface coincidence, so a
    // trajectory that crosses the body would otherwise silently emit
    // an interior potential gradient.  Treat any interior evaluation
    // as a surface impact for the propagator's purposes.
    if body.contains(r_body_au) {
        return Err(Error::SurfaceImpact);
    }
    let a_body_au_per_day2 = body.body_acceleration(r_body_au)?;

    // 2. Tidal third-body perturbations.  Operate in the inertial
    //    frame (rotate the body-fixed field point to inertial first),
    //    accumulate AU/day^2 acceleration, then rotate back.
    let r_inertial_au = body.rotation.rotate_to_inertial(jd_tdb, r_body_au);
    let mut a_tidal_au_per_day2 = Vector3::zeros();
    for &(gm_p, r_pb_inertial) in perturbers {
        let d = r_pb_inertial.norm();
        if d == 0.0 {
            // Coincident perturber and body center; skip rather than
            // produce NaN.  Real propagation should never see this.
            continue;
        }
        let ratio = r_inertial_au.norm() / d;
        if ratio < TIDAL_GRADIENT_THRESHOLD {
            // Gradient expansion: a_tidal = -GM_p / d^3 * (r - 3 (d_hat . r) d_hat).
            let d_hat = r_pb_inertial / d;
            let r_dot_dhat = r_inertial_au.dot(&d_hat);
            a_tidal_au_per_day2 += -gm_p / d.powi(3) * (r_inertial_au - 3.0 * r_dot_dhat * d_hat);
        } else {
            // Direct tidal difference: a_tidal = GM_p * (r_pp / |r_pp|^3 - r_pb / |r_pb|^3),
            // where r_pp is the perturber-to-particle vector.  The
            // particle is at r_inertial_au relative to the body, and
            // the perturber is at r_pb_inertial relative to the body,
            // so r_pp = r_pb_inertial - r_inertial_au.
            let r_pp = r_pb_inertial - r_inertial_au;
            let pp3 = r_pp.norm().powi(3);
            let pb3 = d.powi(3);
            a_tidal_au_per_day2 += gm_p * (r_pp / pp3 - r_pb_inertial / pb3);
        }
    }
    let a_tidal_body_au_per_day2 = body.rotation.rotate_to_body(jd_tdb, a_tidal_au_per_day2);

    // 3. Sum direct + tidal in AU/day^2, then divide by length_au to
    //    convert acceleration to body_length/day^2.  Time stays in days.
    Ok((a_body_au_per_day2 + a_tidal_body_au_per_day2) / body.units.length_au)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use std::f64::consts::PI;

    /// Build a sphere-like polyhedron with given radius in AU and gm.
    /// (Reuses the icosphere helper from the polyhedron tests by
    /// constructing it inline.)
    fn icosphere_verts_faces(subdivisions: u32) -> (Vec<Vector3<f64>>, Vec<[u32; 3]>) {
        let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
        let mut verts: Vec<Vector3<f64>> = vec![
            Vector3::new(-1.0, phi, 0.0),
            Vector3::new(1.0, phi, 0.0),
            Vector3::new(-1.0, -phi, 0.0),
            Vector3::new(1.0, -phi, 0.0),
            Vector3::new(0.0, -1.0, phi),
            Vector3::new(0.0, 1.0, phi),
            Vector3::new(0.0, -1.0, -phi),
            Vector3::new(0.0, 1.0, -phi),
            Vector3::new(phi, 0.0, -1.0),
            Vector3::new(phi, 0.0, 1.0),
            Vector3::new(-phi, 0.0, -1.0),
            Vector3::new(-phi, 0.0, 1.0),
        ]
        .into_iter()
        .map(|v| v.normalize())
        .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..subdivisions {
            let mut new_faces = Vec::with_capacity(faces.len() * 4);
            let mut midpoints: std::collections::HashMap<(u32, u32), u32> =
                std::collections::HashMap::new();
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vector3<f64>>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&i) = midpoints.get(&key) {
                    return i;
                }
                #[allow(
                    clippy::manual_midpoint,
                    reason = "f64::midpoint does not exist for nalgebra Vector3"
                )]
                let m = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                let i = u32::try_from(verts.len()).unwrap();
                verts.push(m);
                let _ = midpoints.insert(key, i);
                i
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                new_faces.push([f[0], a, c]);
                new_faces.push([f[1], b, a]);
                new_faces.push([f[2], c, b]);
                new_faces.push([a, b, c]);
            }
            faces = new_faces;
        }
        (verts, faces)
    }

    fn sphere_body(radius_au: f64, gm: f64) -> ExtendedBody {
        let (mut v, f) = icosphere_verts_faces(2);
        for vert in &mut v {
            *vert *= radius_au;
        }
        let p = Polyhedron::try_new(v, &f, gm).unwrap();
        let units = BodyUnits::try_new(radius_au, gm).unwrap();
        ExtendedBody::try_new(vec![p], RotationModel::identity(), units, 5.0 * radius_au).unwrap()
    }

    #[test]
    fn rejects_empty_components() {
        let units = BodyUnits::try_new(1e-9, 1e-20).unwrap();
        assert!(ExtendedBody::try_new(vec![], RotationModel::identity(), units, 1e-7).is_err());
    }

    #[test]
    fn rejects_bad_proximity() {
        let body = sphere_body(1e-9, 1e-20);
        let units = body.units;
        assert!(
            ExtendedBody::try_new(body.components, RotationModel::identity(), units, 0.0).is_err()
        );
    }

    #[test]
    fn aggregate_gm_sum_of_components() {
        let (mut v, f) = icosphere_verts_faces(2);
        for vert in &mut v {
            *vert *= 1e-9;
        }
        let a = Polyhedron::try_new(v.clone(), &f, 1e-20).unwrap();
        let b = Polyhedron::try_new(v, &f, 3e-20).unwrap();
        let units = BodyUnits::try_new(1e-9, 4e-20).unwrap();
        let body =
            ExtendedBody::try_new(vec![a, b], RotationModel::identity(), units, 1e-7).unwrap();
        assert!((body.gm - 4e-20).abs() < 1e-30);
    }

    #[test]
    fn far_field_recovers_point_mass_no_perturbers() {
        // Far from the body, a sphere body's gravity is GM/r^2 toward
        // origin; small_body_accel should recover that, expressed in
        // body_length/day^2.  In that mixed unit system the
        // acceleration is `(gm / length_au^3) * (-r_body / |r_body|^3)`.
        let radius = 1e-9; // ~150 m in AU
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        // Field point at 50 body radii along +x.
        let r_body_units = Vector3::new(50.0, 0.0, 0.0);
        let a = small_body_accel(&body, 2_451_545.0, r_body_units, &[]).unwrap();
        let scale = gm / radius.powi(3);
        let expected = -scale * r_body_units / r_body_units.norm().powi(3);
        let rel = (a - expected).norm() / expected.norm();
        assert!(rel < 1e-2, "no-perturber far-field: rel={rel:e}");
    }

    #[test]
    fn tidal_near_body_center_is_negligible() {
        // Just outside the body surface (r slightly above the
        // bounding radius), the tidal contribution from a 1 AU
        // perturber must be vanishingly small compared to local self-
        // gravity, since the tidal differential at small r scales as
        // r/d^3.
        let radius = 1e-9;
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        // Position just outside the body: 1.01 body radii along +x.
        let r_test = Vector3::new(1.01, 0.0, 0.0);
        let perturbers = [(0.000_295_912_208_284_119_56, Vector3::new(1.0, 0.0, 0.0))];
        let a_with = small_body_accel(&body, 0.0, r_test, &perturbers).unwrap();
        let a_without = small_body_accel(&body, 0.0, r_test, &[]).unwrap();
        let tidal = a_with - a_without;
        // Tidal magnitude estimate: GM_sun * r_au / d^3 (in AU/day^2),
        // converted to body_length/day^2 via /length_au.
        // = 3e-4 * 1.01e-9 / 1 / 1e-9 = 3e-4 body_length/day^2.
        // Self-gravity at r=1.01: gm/length_au^3 / r^2 ~ 1e7 / 1 = 1e7.
        // So |tidal| / |self| ~ 3e-11; absolute < 1e-3 is comfortable.
        assert!(
            tidal.norm() < 1e-3,
            "near-surface tidal should be tiny: {tidal:?} (norm {})",
            tidal.norm()
        );
    }

    #[test]
    fn tidal_gradient_form_matches_direct_for_moderate_ratio() {
        // For a moderate r/d ratio that triggers the DIRECT branch,
        // the result should be very close to the gradient-expansion
        // result.  At threshold ratio (1e-4), they should agree to
        // O((r/d)^2) ~ 1e-8.
        let radius = 1e-9;
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        // Pick r and d so ratio is just above threshold (uses direct).
        // r in body units = 200 -> r in AU = 2e-7.  d = 1.5e-3 AU.
        // ratio = 2e-7 / 1.5e-3 ~ 1.3e-4 (just above threshold).
        let r_body_units = Vector3::new(200.0, 0.0, 0.0);
        let perturber = (1e-10, Vector3::new(1.5e-3, 0.0, 0.0));
        let a_direct = small_body_accel(&body, 0.0, r_body_units, &[perturber]).unwrap();

        // Now manually compute with gradient form for the same setup.
        // We can't toggle the threshold from outside, so we instead
        // compute gradient form by hand here for comparison.
        let r_au = body.units.pos_from_body(r_body_units);
        let d_vec = perturber.1;
        let d = d_vec.norm();
        let d_hat = d_vec / d;
        let r_dot_dhat = r_au.dot(&d_hat);
        let a_grad_au_per_day2 = -perturber.0 / d.powi(3) * (r_au - 3.0 * r_dot_dhat * d_hat);
        let a_self_au_per_day2 = body.body_acceleration(r_au).unwrap();
        let a_grad_body = (a_grad_au_per_day2 + a_self_au_per_day2) / body.units.length_au;
        let rel = (a_direct - a_grad_body).norm() / a_grad_body.norm();
        // The gradient form is the leading-order tidal expansion;
        // its relative error vs. the direct form is O(r/d), which is
        // ~1e-4 at this configuration.  Use a generous tolerance.
        assert!(
            rel < 1e-3,
            "direct vs gradient tidal disagreement: rel={rel:e}"
        );
    }

    #[test]
    fn rotation_round_trip_preserves_acceleration_no_tidal() {
        // With no perturbers, the result should not depend on the
        // rotation model: the body's self-gravity is computed in the
        // body-fixed frame and rotated by nothing.  Different
        // rotations should give identical acceleration.
        let radius = 1e-9;
        let gm = 1e-20;
        let mut body_a = sphere_body(radius, gm);
        let mut body_b = sphere_body(radius, gm);
        body_a.rotation = RotationModel::identity();
        body_b.rotation = RotationModel::ConstantSpin {
            pole_ra: 0.4,
            pole_dec: 0.7,
            w0: 0.2,
            w_dot: 2.0 * PI, // 1-day period
            epoch_jd: 0.0,
        };
        let r = Vector3::new(2.0, -1.0, 0.5);
        let a_a = small_body_accel(&body_a, 0.3, r, &[]).unwrap();
        let a_b = small_body_accel(&body_b, 0.3, r, &[]).unwrap();
        assert!((a_a - a_b).norm() < 1e-12);
    }
}
