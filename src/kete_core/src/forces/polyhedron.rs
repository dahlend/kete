//! Gravity field of a constant-density polyhedron.
//!
//! The closed-form field of Werner and Scheeres (1997), "Exterior gravitation of a
//! polyhedron derived and compared with harmonic and mascon gravitation
//! representations of asteroid 4769 Castalia", Celestial Mechanics and Dynamical
//! Astronomy 65, 313. For a field point, with `r_i` the vector from the field point
//! to vertex `i` and `R_i = |r_i|`:
//!
//! ```text
//! per face f, vertices i, j, k counter-clockwise seen from outside, normal n_f:
//!     F_f     = n_f n_f^T
//!     omega_f = 2 atan2( r_i . (r_j x r_k),
//!                        R_i R_j R_k + R_i (r_j . r_k) + R_j (r_k . r_i) + R_k (r_i . r_j) )
//! per edge e from vertex i to j, shared by faces A (i -> j) and B (j -> i):
//!     E_e     = n_A m_A^T + n_B m_B^T      m_X: outward in-plane edge normal of face X
//!     L_e     = ln( (R_i + R_j + |e|) / (R_i + R_j - |e|) )
//!
//! potential     U     = G s / 2 ( sum_e r_e . E_e r_e L_e - sum_f r_f . F_f r_f omega_f )
//! acceleration  g     = G s     ( sum_f F_f r_f omega_f - sum_e E_e r_e L_e )
//! gradient      dg/dr = G s     ( sum_e E_e L_e - sum_f F_f omega_f )
//! ```
//!
//! with `G s = GM / V` and `r_e`, `r_f` the vector to any vertex of the edge or face.
//! `U` is positive (`g = grad U`). `sum_f omega_f` is the solid angle the surface
//! subtends: `4 pi` inside and `0` outside. Exactly on the surface it is not well
//! defined in floating point (rounding puts the point on one side), so it is used
//! as an inside test only against a threshold between the two. The trace of `dg/dr`
//! is `-G s sum_f omega_f`, Poisson's equation.
//!
//! Far from the body the sums cancel to the much smaller point-mass value, so
//! relative precision degrades with distance, and each evaluation costs a pass over
//! every face and edge. Beyond `FAR_FIELD_RATIO` times the mesh's bounding radius
//! the field is therefore taken from the polyhedron's exact spherical harmonic
//! expansion ([`SphericalHarmonics::from_polyhedron`]), cut at the lowest degree
//! that reproduces the closed form to `FAR_FIELD_ACCEL_TOL` (acceleration) and
//! `FAR_FIELD_GRAD_TOL` (gradient) relative, checked against it at
//! `FAR_FIELD_CHECKS` points on that sphere. It is built on the first evaluation
//! beyond the sphere; if no degree up to `FAR_FIELD_MAX_DEGREE` passes, the closed
//! form is used everywhere. [`Polyhedron::without_far_field`] turns it off.
//!
//! The field contains the point-mass term, so it replaces it rather than adding to it.
//
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

use std::f64::consts::PI;
use std::sync::OnceLock;

use nalgebra::{Matrix3, Vector3};

use super::SphericalHarmonics;
use crate::errors::{Error, KeteResult};
use crate::geometry::TriMesh;

/// The far field is used at and beyond this multiple of the bounding radius.
const FAR_FIELD_RATIO: f64 = 3.0;

/// Largest relative acceleration difference from the closed form allowed for the
/// far field, on the sphere where it takes over.
const FAR_FIELD_ACCEL_TOL: f64 = 1e-11;

/// Largest relative (Frobenius) gradient difference allowed for the far field.
const FAR_FIELD_GRAD_TOL: f64 = 1e-9;

/// Highest degree tried for the far field.
const FAR_FIELD_MAX_DEGREE: usize = 40;

/// Points on the far-field sphere where it is checked against the closed form.
const FAR_FIELD_CHECKS: usize = 200;

/// Degree of the first expansion computed. The truncation error falls about as
/// `FAR_FIELD_RATIO^-(degree + 1)`, so `FAR_FIELD_ACCEL_TOL` is reached near degree
/// `-ln(1e-11) / ln(3) = 23`; this is a margin above that.
const FAR_FIELD_START_DEGREE: usize = 30;

/// A constant-density polyhedron and the precomputed terms of its gravity field.
///
/// The mesh is used where it is placed: fields are evaluated at positions relative
/// to the mesh origin, in the mesh's own (body) frame and units. The center of mass
/// is the volume centroid of the mesh, which need not be at the origin. Far from the
/// mesh the field comes from a checked spherical harmonic expansion (see the module
/// documentation).
#[derive(Debug, Clone)]
pub struct Polyhedron {
    /// The mesh.
    mesh: TriMesh,

    /// Gravitational parameter of the body.
    gm: f64,

    /// `G * density`, `gm / volume`.
    g_density: f64,

    /// Per face: vertex indices, counter-clockwise seen from outside.
    face_vertices: Vec<[usize; 3]>,

    /// Per face: `n n^T`.
    face_dyads: Vec<Matrix3<f64>>,

    /// Per edge: vertex indices.
    edge_vertices: Vec<[usize; 2]>,

    /// Per edge: `n_A m_A^T + n_B m_B^T`.
    edge_dyads: Vec<Matrix3<f64>>,

    /// Per edge: length.
    edge_lengths: Vec<f64>,

    /// Evaluations at or beyond this distance from the origin use `far_field`;
    /// `None` when turned off.
    far_field_radius: Option<f64>,

    /// The far field, built on first use; `None` inside if no degree passed.
    far_field: OnceLock<Option<SphericalHarmonics>>,
}

impl Polyhedron {
    /// Build the field of `mesh` at constant density with gravitational parameter `gm`.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `gm` is not positive and finite.
    pub fn new(mesh: TriMesh, gm: f64) -> KeteResult<Self> {
        if !gm.is_finite() || gm <= 0.0 {
            return Err(Error::ValueError(
                "Polyhedron GM must be positive and finite.".into(),
            ));
        }
        let g_density = gm / mesh.volume();
        let mesh_radius = mesh.bounding_radius();

        let face_vertices = mesh.faces().iter().map(|f| f.map(|k| k as usize)).collect();
        let normals: Vec<Vector3<f64>> = mesh
            .faces()
            .iter()
            .map(|f| mesh.face_normal(f).into_inner())
            .collect();
        let face_dyads = normals.iter().map(|n| n * n.transpose()).collect();

        let verts = mesh.vertices();
        let mut edge_vertices = Vec::with_capacity(mesh.edges().len());
        let mut edge_dyads = Vec::with_capacity(mesh.edges().len());
        let mut edge_lengths = Vec::with_capacity(mesh.edges().len());
        for edge in mesh.edges() {
            let [a, b] = edge.vertices.map(|k| k as usize);
            let d = verts[b] - verts[a];
            let n_a = normals[edge.faces[0] as usize];
            let n_b = normals[edge.faces[1] as usize];
            // Face A traverses a -> b and face B b -> a; for a counter-clockwise face
            // the outward in-plane edge normal is (edge direction) x (face normal).
            let m_a = d.cross(&n_a).normalize();
            let m_b = (-d).cross(&n_b).normalize();
            edge_vertices.push([a, b]);
            edge_dyads.push(n_a * m_a.transpose() + n_b * m_b.transpose());
            edge_lengths.push(d.norm());
        }

        Ok(Self {
            mesh,
            gm,
            g_density,
            face_vertices,
            face_dyads,
            edge_vertices,
            edge_dyads,
            edge_lengths,
            far_field_radius: Some(FAR_FIELD_RATIO * mesh_radius),
            far_field: OnceLock::new(),
        })
    }

    /// The same polyhedron with the far field turned off: the closed form is used
    /// at every distance.
    #[must_use]
    pub fn without_far_field(mut self) -> Self {
        self.far_field_radius = None;
        self.far_field = OnceLock::new();
        self
    }

    /// Distance from the origin at and beyond which the far field is used, `None`
    /// when turned off.
    #[must_use]
    pub fn far_field_radius(&self) -> Option<f64> {
        self.far_field_radius
    }

    /// The far field (building it if needed), `None` when turned off or when no
    /// degree up to `FAR_FIELD_MAX_DEGREE` met the tolerances.
    pub fn far_field(&self) -> Option<&SphericalHarmonics> {
        self.far_field_radius.and_then(|_| {
            self.far_field
                .get_or_init(|| self.build_far_field())
                .as_ref()
        })
    }

    /// The mesh.
    #[must_use]
    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    /// Gravitational parameter.
    #[must_use]
    pub fn gm(&self) -> f64 {
        self.gm
    }

    /// Gravitational potential at `pos` (relative to the mesh origin, body frame),
    /// positive, so that the acceleration is its gradient.
    #[must_use]
    pub fn potential(&self, pos: &Vector3<f64>) -> f64 {
        if let Some(Ok(value)) = self.far(pos).map(|far| far.potential(pos)) {
            return value;
        }
        self.closed_form_potential(pos)
    }

    /// Acceleration at `pos` (relative to the mesh origin, body frame), and the
    /// solid angle the surface subtends there (`4 pi` inside, `0` outside).
    #[must_use]
    pub fn field(&self, pos: &Vector3<f64>) -> (Vector3<f64>, f64) {
        if let Some(Ok(accel)) = self.far(pos).map(|far| far.field(pos)) {
            // beyond the far-field sphere the point is outside the mesh
            return (accel, 0.0);
        }
        self.closed_form_field(pos)
    }

    /// Acceleration, its gradient with respect to position, and the solid angle, at
    /// `pos` (relative to the mesh origin, body frame).
    #[must_use]
    pub fn field_and_gradient(&self, pos: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>, f64) {
        if let Some(Ok((accel, grad))) = self.far(pos).map(|far| far.field_and_gradient(pos)) {
            return (accel, grad, 0.0);
        }
        self.closed_form_field_and_gradient(pos)
    }

    /// The far field, if it is to be used at `pos`.
    fn far(&self, pos: &Vector3<f64>) -> Option<&SphericalHarmonics> {
        if pos.norm() >= self.far_field_radius? {
            self.far_field()
        } else {
            None
        }
    }

    /// The expansion cut at the lowest degree that passes the checks against the
    /// closed form on the far-field sphere, or `None`.
    fn build_far_field(&self) -> Option<SphericalHarmonics> {
        let radius = self.far_field_radius?;
        let golden = PI * (3.0 - 5_f64.sqrt());
        let checks: Vec<(Vector3<f64>, Vector3<f64>, Matrix3<f64>)> = (0..FAR_FIELD_CHECKS)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / FAR_FIELD_CHECKS as f64;
                let rho = (1.0 - z * z).sqrt();
                let phi = golden * i as f64;
                let pos = Vector3::new(rho * phi.cos(), rho * phi.sin(), z) * radius;
                let (accel, grad, _) = self.closed_form_field_and_gradient(&pos);
                (pos, accel, grad)
            })
            .collect();
        let passes = |field: &SphericalHarmonics| {
            checks.iter().all(|(pos, accel, grad)| {
                field.field_and_gradient(pos).is_ok_and(|(a, g)| {
                    (a - accel).norm() <= FAR_FIELD_ACCEL_TOL * accel.norm()
                        && (g - grad).norm() <= FAR_FIELD_GRAD_TOL * grad.norm()
                })
            })
        };
        let mut tried = 0;
        for top in [FAR_FIELD_START_DEGREE, FAR_FIELD_MAX_DEGREE] {
            let full = SphericalHarmonics::from_polyhedron(self, top).ok()?;
            for degree in tried..=top {
                let cut = full.truncated(degree).ok()?;
                if passes(&cut) {
                    return Some(cut);
                }
            }
            tried = top + 1;
        }
        None
    }

    fn closed_form_potential(&self, pos: &Vector3<f64>) -> f64 {
        let (rel, dist) = self.relative(pos);
        let mut edge_sum = 0.0;
        for (e, &[a, b]) in self.edge_vertices.iter().enumerate() {
            let log = edge_log(dist[a], dist[b], self.edge_lengths[e]);
            edge_sum += rel[a].dot(&(self.edge_dyads[e] * rel[a])) * log;
        }
        let mut face_sum = 0.0;
        for (f, &[i, j, k]) in self.face_vertices.iter().enumerate() {
            let omega = face_solid_angle(&rel, &dist, i, j, k);
            face_sum += rel[i].dot(&(self.face_dyads[f] * rel[i])) * omega;
        }
        0.5 * self.g_density * (edge_sum - face_sum)
    }

    fn closed_form_field(&self, pos: &Vector3<f64>) -> (Vector3<f64>, f64) {
        let (rel, dist) = self.relative(pos);
        let mut accel = Vector3::zeros();
        for (e, &[a, b]) in self.edge_vertices.iter().enumerate() {
            let log = edge_log(dist[a], dist[b], self.edge_lengths[e]);
            accel -= self.edge_dyads[e] * rel[a] * log;
        }
        let mut solid_angle = 0.0;
        for (f, &[i, j, k]) in self.face_vertices.iter().enumerate() {
            let omega = face_solid_angle(&rel, &dist, i, j, k);
            accel += self.face_dyads[f] * rel[i] * omega;
            solid_angle += omega;
        }
        (accel * self.g_density, solid_angle)
    }

    fn closed_form_field_and_gradient(
        &self,
        pos: &Vector3<f64>,
    ) -> (Vector3<f64>, Matrix3<f64>, f64) {
        let (rel, dist) = self.relative(pos);
        let mut accel = Vector3::zeros();
        let mut grad = Matrix3::zeros();
        for (e, &[a, b]) in self.edge_vertices.iter().enumerate() {
            let log = edge_log(dist[a], dist[b], self.edge_lengths[e]);
            accel -= self.edge_dyads[e] * rel[a] * log;
            grad += self.edge_dyads[e] * log;
        }
        let mut solid_angle = 0.0;
        for (f, &[i, j, k]) in self.face_vertices.iter().enumerate() {
            let omega = face_solid_angle(&rel, &dist, i, j, k);
            accel += self.face_dyads[f] * rel[i] * omega;
            grad -= self.face_dyads[f] * omega;
            solid_angle += omega;
        }
        (accel * self.g_density, grad * self.g_density, solid_angle)
    }

    /// Vectors from `pos` to every vertex, and their lengths.
    fn relative(&self, pos: &Vector3<f64>) -> (Vec<Vector3<f64>>, Vec<f64>) {
        let rel: Vec<Vector3<f64>> = self.mesh.vertices().iter().map(|v| v - pos).collect();
        let dist = rel.iter().map(Vector3::norm).collect();
        (rel, dist)
    }
}

/// `L_e` for an edge of length `len` whose ends are `dist_a` and `dist_b` away.
#[inline(always)]
fn edge_log(dist_a: f64, dist_b: f64, len: f64) -> f64 {
    let sum = dist_a + dist_b;
    ((sum + len) / (sum - len)).ln()
}

/// Signed solid angle of face `(i, j, k)` seen from the field point.
#[inline(always)]
fn face_solid_angle(rel: &[Vector3<f64>], dist: &[f64], i: usize, j: usize, k: usize) -> f64 {
    let (ri, rj, rk) = (&rel[i], &rel[j], &rel[k]);
    let (di, dj, dk) = (dist[i], dist[j], dist[k]);
    let num = ri.dot(&rj.cross(rk));
    let den = di * dj * dk + di * rj.dot(rk) + dj * rk.dot(ri) + dk * ri.dot(rj);
    2.0 * num.atan2(den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    /// Box `[x0, x1] x [y0, y1] x [z0, z1]` as a mesh, two faces per side.
    fn prism(lo: [f64; 3], hi: [f64; 3]) -> TriMesh {
        let v = (0..8)
            .map(|i: u32| {
                Vector3::new(
                    if i & 1 == 0 { lo[0] } else { hi[0] },
                    if (i >> 1) & 1 == 0 { lo[1] } else { hi[1] },
                    if (i >> 2) & 1 == 0 { lo[2] } else { hi[2] },
                )
            })
            .collect();
        let f = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        TriMesh::new(v, f).unwrap()
    }

    /// Potential of a homogeneous rectangular prism, Nagy, Papp and Benedek (2000),
    /// J. Geodesy 74, 552: `G rho` times the triple difference over the corners of
    /// the primitive of `1 / r`, coordinates relative to the field point.
    fn nagy_potential(lo: [f64; 3], hi: [f64; 3], g_rho: f64, p: &Vector3<f64>) -> f64 {
        let prim = |x: f64, y: f64, z: f64| {
            let r = (x * x + y * y + z * z).sqrt();
            x * y * (z + r).ln() + y * z * (x + r).ln() + z * x * (y + r).ln()
                - 0.5 * x * x * (y * z / (x * r)).atan()
                - 0.5 * y * y * (z * x / (y * r)).atan()
                - 0.5 * z * z * (x * y / (z * r)).atan()
        };
        let mut total = 0.0;
        for (i, x) in [lo[0] - p.x, hi[0] - p.x].into_iter().enumerate() {
            for (j, y) in [lo[1] - p.y, hi[1] - p.y].into_iter().enumerate() {
                for (k, z) in [lo[2] - p.z, hi[2] - p.z].into_iter().enumerate() {
                    // upper limit minus lower limit along each axis
                    let sign = if (i + j + k) % 2 == 1 { -1.0 } else { 1.0 };
                    total -= sign * prim(x, y, z);
                }
            }
        }
        g_rho * total
    }

    /// Acceleration of the same prism, the gradient of [`nagy_potential`] with
    /// respect to the field point: component `x` is `-G rho` times the triple
    /// difference of `dF/dx = y ln(z + r) + z ln(y + r) - x atan(y z / (x r))`,
    /// and cyclically for `y` and `z`.
    fn nagy_acceleration(lo: [f64; 3], hi: [f64; 3], g_rho: f64, p: &Vector3<f64>) -> Vector3<f64> {
        let d = |x: f64, y: f64, z: f64| {
            let r = (x * x + y * y + z * z).sqrt();
            y * (z + r).ln() + z * (y + r).ln() - x * (y * z / (x * r)).atan()
        };
        let mut total = Vector3::zeros();
        for (i, x) in [lo[0] - p.x, hi[0] - p.x].into_iter().enumerate() {
            for (j, y) in [lo[1] - p.y, hi[1] - p.y].into_iter().enumerate() {
                for (k, z) in [lo[2] - p.z, hi[2] - p.z].into_iter().enumerate() {
                    let sign = if (i + j + k) % 2 == 1 { -1.0 } else { 1.0 };
                    total -= sign * Vector3::new(d(x, y, z), d(y, z, x), d(z, x, y));
                }
            }
        }
        -g_rho * total
    }

    const LO: [f64; 3] = [-1.3, -0.7, -0.45];
    const HI: [f64; 3] = [1.1, 0.9, 0.35];

    /// The test prism, off-center about the origin, and its Nagy corners.
    fn prism_case() -> (Polyhedron, [f64; 3], [f64; 3], f64) {
        let gm = 2.5;
        let poly = Polyhedron::new(prism(LO, HI), gm).unwrap();
        let vol = (HI[0] - LO[0]) * (HI[1] - LO[1]) * (HI[2] - LO[2]);
        (poly, LO, HI, gm / vol)
    }

    /// Field points off every face plane and axis, inside and outside the prism,
    /// within a few prism sizes (see the module note on the far field).
    fn test_points() -> Vec<(Vector3<f64>, bool)> {
        vec![
            (Vector3::new(0.13, -0.21, 0.07), true),
            (Vector3::new(-0.9, 0.5, -0.3), true),
            (Vector3::new(2.3, 1.7, -1.1), false),
            (Vector3::new(-0.4, 0.2, 0.93), false),
            (Vector3::new(0.3, -2.9, 0.17), false),
            (Vector3::new(-4.1, 3.3, 2.6), false),
        ]
    }

    #[test]
    fn prism_potential_matches_closed_form() {
        let (poly, lo, hi, g_rho) = prism_case();
        for (p, _) in test_points() {
            let exact = nagy_potential(lo, hi, g_rho, &p);
            let got = poly.potential(&p);
            assert!(
                ((got - exact) / exact).abs() < 1e-12,
                "potential at {p:?}: {got} vs {exact}"
            );
        }
    }

    #[test]
    fn prism_acceleration_matches_closed_form() {
        let (poly, lo, hi, g_rho) = prism_case();
        for (p, _) in test_points() {
            let (g, _) = poly.field(&p);
            let exact = nagy_acceleration(lo, hi, g_rho, &p);
            assert!(
                (g - exact).norm() < 1e-12 * exact.norm(),
                "acceleration at {p:?}: {g:?} vs {exact:?}"
            );
        }
    }

    #[test]
    fn closed_form_acceleration_is_the_potential_gradient() {
        // checks the two references against each other, near the prism where the
        // finite difference of the potential is not swamped by cancellation
        let (_, lo, hi, g_rho) = prism_case();
        let h = 1e-5;
        for (p, _) in test_points().into_iter().take(5) {
            let fd = Vector3::from_fn(|i, _| {
                let dp = Vector3::ith(i, h);
                (nagy_potential(lo, hi, g_rho, &(p + dp))
                    - nagy_potential(lo, hi, g_rho, &(p - dp)))
                    / (2.0 * h)
            });
            let exact = nagy_acceleration(lo, hi, g_rho, &p);
            assert!(
                (fd - exact).norm() < 1e-8 * exact.norm(),
                "reference gradient at {p:?}"
            );
        }
    }

    #[test]
    fn solid_angle_and_poisson() {
        let (poly, ..) = prism_case();
        for (p, inside) in test_points() {
            let (_, grad, omega) = poly.field_and_gradient(&p);
            let expect = if inside { 4.0 * PI } else { 0.0 };
            assert!(
                (omega - expect).abs() < 1e-12,
                "solid angle at {p:?}: {omega}"
            );
            let expect_trace = -poly.g_density * expect;
            assert!(
                (grad.trace() - expect_trace).abs() < 1e-12 * poly.g_density,
                "trace of gradient at {p:?}"
            );
            assert!(
                (grad - grad.transpose()).norm() < 1e-12 * grad.norm().max(poly.g_density),
                "gradient symmetric at {p:?}"
            );
        }
        // just inside and just outside a face (the x = HI side), off its diagonal
        for (dx, expect) in [(-1e-9, 4.0 * PI), (1e-9, 0.0)] {
            let near_face = Vector3::new(HI[0] + dx, 0.31, -0.12);
            let (_, omega) = poly.field(&near_face);
            assert!(
                (omega - expect).abs() < 1e-9,
                "solid angle {dx} from a face: {omega}"
            );
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let (poly, ..) = prism_case();
        let h = 1e-6;
        for (p, _) in test_points() {
            let (g, grad, _) = poly.field_and_gradient(&p);
            let (g_only, _) = poly.field(&p);
            assert!((g - g_only).norm() <= 1e-15 * g.norm(), "field paths agree");
            for i in 0..3 {
                let dp = Vector3::ith(i, h);
                let col = (poly.field(&(p + dp)).0 - poly.field(&(p - dp)).0) / (2.0 * h);
                assert!(
                    (grad.column(i) - col).norm() < 1e-7 * grad.norm().max(1e-3),
                    "gradient column {i} at {p:?}"
                );
            }
        }
    }

    #[test]
    fn point_mass_far_away() {
        let (poly, ..) = prism_case();
        let com = poly.mesh().centroid();
        let dir = Vector3::new(0.3, -0.5, 0.8).normalize();
        let mut prev = f64::INFINITY;
        for dist in [10.0, 30.0, 100.0, 300.0] {
            // the point mass sits at the volume centroid, off the origin
            let p = com + dir * dist;
            let (g, _) = poly.field(&p);
            let point = -poly.gm() * dir / dist.powi(2);
            let rel = (g - point).norm() / point.norm();
            // the leading correction is the quadrupole, relative size ~ (size / r)^2
            assert!(rel < prev / 5.0, "approaches a point mass at {dist}: {rel}");
            prev = rel;
        }
        assert!(prev < 1e-4, "point mass at 300: {prev}");
    }

    #[test]
    fn rotation_invariance() {
        let (poly, ..) = prism_case();
        let rot = Rotation3::from_euler_angles(0.3, -1.1, 2.0);
        let verts: Vec<_> = poly.mesh().vertices().iter().map(|v| rot * v).collect();
        let rotated = Polyhedron::new(
            TriMesh::new(verts, poly.mesh().faces().to_vec()).unwrap(),
            poly.gm(),
        )
        .unwrap();
        for (p, _) in test_points() {
            let (g, grad, _) = poly.field_and_gradient(&p);
            let (g_r, grad_r, _) = rotated.field_and_gradient(&(rot * p));
            assert!(
                (g_r - rot * g).norm() < 1e-13 * g.norm().max(1.0),
                "rotated field"
            );
            let expect = rot.matrix() * grad * rot.matrix().transpose();
            assert!(
                (grad_r - expect).norm() < 1e-12 * grad.norm().max(1.0),
                "rotated gradient"
            );
        }
    }

    /// Split every face into four at its edge midpoints; the surface is unchanged.
    fn subdivide(mesh: &TriMesh) -> TriMesh {
        let mut verts = mesh.vertices().to_vec();
        let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vector3<f64>>| {
            *mid.entry((a.min(b), a.max(b))).or_insert_with(|| {
                verts.push((verts[a as usize] + verts[b as usize]) / 2.0);
                u32::try_from(verts.len() - 1).unwrap()
            })
        };
        let mut faces = Vec::new();
        for &[a, b, c] in mesh.faces() {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            faces.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        TriMesh::new(verts, faces).unwrap()
    }

    #[test]
    fn coplanar_subdivision_leaves_field_unchanged() {
        let (poly, ..) = prism_case();
        let fine = Polyhedron::new(subdivide(poly.mesh()), poly.gm()).unwrap();
        for (p, _) in test_points() {
            let (g, _) = poly.field(&p);
            let (g_f, _) = fine.field(&p);
            assert!(
                (g - g_f).norm() < 1e-12 * g.norm().max(1e-3),
                "subdivided field at {p:?}: {g:?} vs {g_f:?}"
            );
        }
    }

    #[test]
    fn mesh_is_used_where_placed() {
        let offset = Vector3::new(5.0, -3.0, 2.0);
        let lo = [LO[0] + offset.x, LO[1] + offset.y, LO[2] + offset.z];
        let hi = [HI[0] + offset.x, HI[1] + offset.y, HI[2] + offset.z];
        let shifted = Polyhedron::new(prism(lo, hi), 2.5).unwrap();
        let (base, ..) = prism_case();
        assert_eq!(
            shifted.mesh().vertices(),
            prism(lo, hi).vertices(),
            "vertices kept"
        );
        let p = Vector3::new(0.7, 2.2, -1.9);
        assert!(
            (shifted.field(&(p + offset)).0 - base.field(&p).0).norm() < 1e-12,
            "field moves with the mesh"
        );
    }

    /// Interior acceleration of a homogeneous triaxial ellipsoid with semi-axes `axes`:
    /// `g_i = -2 pi G rho A_i x_i`, `A_i = a b c int_0^inf du / ((a_i^2 + u) D(u))`,
    /// `D(u) = sqrt((a^2 + u)(b^2 + u)(c^2 + u))` (Chandrasekhar 1969, Ellipsoidal
    /// Figures of Equilibrium, ch. 3). The integral is taken with `u = (t / (1 - t))^2`,
    /// which makes the integrand smooth on `[0, 1]`, by composite Simpson.
    fn ellipsoid_interior(axes: [f64; 3], g_rho: f64, p: &Vector3<f64>) -> Vector3<f64> {
        let [ax, ay, az] = axes;
        let coef = |axis: f64| {
            let integrand = |t: f64| {
                if t >= 1.0 {
                    return 0.0;
                }
                let w = t / (1.0 - t);
                let u = w * w;
                let du_dt = 2.0 * t / (1.0 - t).powi(3);
                let denom = ((ax * ax + u) * (ay * ay + u) * (az * az + u)).sqrt();
                du_dt / ((axis * axis + u) * denom)
            };
            let steps = 20_000;
            let step = 1.0 / f64::from(steps);
            let mut sum = integrand(0.0) + integrand(1.0);
            for k in 1..steps {
                sum += integrand(f64::from(k) * step) * if k % 2 == 1 { 4.0 } else { 2.0 };
            }
            ax * ay * az * sum * step / 3.0
        };
        let coefs = [coef(ax), coef(ay), coef(az)];
        assert!(
            (coefs.iter().sum::<f64>() - 2.0).abs() < 1e-9,
            "ellipsoid coefficients sum to 2"
        );
        Vector3::from_fn(|i, _| -2.0 * PI * g_rho * coefs[i] * p[i])
    }

    #[test]
    fn ellipsoid_interior_converges_to_closed_form() {
        let axes = [3.0, 2.0, 1.0];
        let g_rho = 0.7;
        let points = [
            Vector3::new(0.4, -0.3, 0.2),
            Vector3::new(-1.5, 0.8, -0.3),
            Vector3::new(0.9, 1.1, 0.5),
        ];
        let mut prev = f64::INFINITY;
        for n_div in [4, 8, 16, 32] {
            let mesh = TriMesh::new_ellipsoid(n_div, axes[0], axes[1], axes[2]).unwrap();
            let gm = g_rho * mesh.volume();
            let poly = Polyhedron::new(mesh, gm).unwrap();
            let worst = points
                .iter()
                .map(|p| {
                    let exact = ellipsoid_interior(axes, g_rho, p);
                    (poly.field(p).0 - exact).norm() / exact.norm()
                })
                .fold(0.0, f64::max);
            // inscribed mesh: the error falls roughly with the facet area, 1 / n_div^2
            assert!(worst < prev / 3.0, "converging at n_div {n_div}: {worst}");
            prev = worst;
        }
        assert!(prev < 2e-3, "n_div 32 within 0.2%: {prev}");
    }

    /// Beyond the far-field sphere the field comes from the checked expansion and
    /// matches the closed form; inside the sphere it is the closed form; turned off,
    /// the closed form is used everywhere.
    #[test]
    fn far_field_matches_closed_form() {
        let poly = Polyhedron::new(prism([-0.8, -0.5, -0.3], [1.2, 0.9, 0.7]), 1.0).unwrap();
        let exact = poly.clone().without_far_field();
        let radius = poly.far_field_radius().unwrap();
        assert!((radius - FAR_FIELD_RATIO * poly.mesh().bounding_radius()).abs() < 1e-14);
        assert!(exact.far_field_radius().is_none() && exact.far_field().is_none());
        let far = poly
            .far_field()
            .expect("the far field should pass its checks");
        println!(
            "far field degree {} at {radius} (bounding radius {})",
            far.degree(),
            poly.mesh().bounding_radius()
        );
        let dirs = [
            Vector3::new(0.3, -0.8, 0.5),
            Vector3::new(-1.0, 0.1, 0.05),
            Vector3::new(0.0, 0.0, -1.0),
        ];
        for dir in dirs {
            for scale in [1.0, 1.3, 2.0] {
                let p = dir.normalize() * radius * scale;
                let (a, g, omega) = poly.field_and_gradient(&p);
                let (a0, g0, _) = exact.field_and_gradient(&p);
                assert_eq!(omega, 0.0);
                assert!((a - a0).norm() <= FAR_FIELD_ACCEL_TOL * a0.norm(), "{p:?}");
                assert!((g - g0).norm() <= FAR_FIELD_GRAD_TOL * g0.norm(), "{p:?}");
                assert!((poly.field(&p).0 - a).norm() < 1e-15 * a.norm());
                let u0 = exact.potential(&p);
                assert!((poly.potential(&p) - u0).abs() <= FAR_FIELD_ACCEL_TOL * u0);
            }
            // just inside the sphere the closed form is used, unchanged
            let p = dir.normalize() * radius * 0.99;
            assert_eq!(poly.field_and_gradient(&p), exact.field_and_gradient(&p));
        }
    }

    #[test]
    fn rejects_bad_gm() {
        for gm in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(Polyhedron::new(prism(LO, HI), gm).is_err(), "gm {gm}");
        }
    }
}
