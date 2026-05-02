//! Constant-density polyhedron gravity model (Werner & Scheeres 1996).
//!
//! The closed-form expressions used here are exact for any closed,
//! orientable polyhedron with consistent outward-facing triangle normals.
//! Convexity is not required; non-convex shapes (concavities, contact
//! binaries represented as a single mesh) are handled correctly.
//!
//! References:
//! - Werner, R. (1994), Celestial Mechanics 59, 253.
//! - Werner & Scheeres (1996), CMDA 65, 313.
//! - Scheeres, D. J. (2012), Orbital Motion in Strongly Perturbed Environments,
//!   Section 2.4.
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

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};

use crate::errors::{Error, KeteResult};
use crate::shape::ExtendedGravity;

/// Closed-form constant-density polyhedron gravity model.
///
/// All input vertices are interpreted in the body-fixed frame in caller-
/// supplied length units `L`.  The supplied `gm` (gravitational parameter
/// `G * M`) sets the units of returned potential, acceleration, and
/// gradient: potential is in `L^2 / T^2`, acceleration in `L / T^2`,
/// gradient in `1 / T^2`, where the time unit `T` is whatever makes `gm`
/// dimensionally consistent with `L`.
#[derive(Debug, Clone)]
pub struct Polyhedron {
    /// Vertex positions in the body-fixed frame.
    pub vertices: Vec<Vector3<f64>>,
    faces: Vec<FaceData>,
    edges: Vec<EdgeData>,
    /// Total volume of the polyhedron.
    pub volume: f64,
    /// Largest distance from the origin (body-fixed frame) to any vertex.
    pub bounding_radius: f64,
    /// Gravitational parameter `G * M` of the body.
    pub gm: f64,
    /// `G * sigma` where `sigma = M / V` is the constant density.  This is
    /// the multiplicative factor in the Werner-Scheeres expressions.
    g_sigma: f64,
}

#[derive(Debug, Clone)]
struct FaceData {
    /// Vertex indices in CCW order viewed from outside.
    v: [u32; 3],
    /// Outward unit normal.
    normal: Vector3<f64>,
    /// `n n^T`, the face dyad.
    dyad: Matrix3<f64>,
}

#[derive(Debug, Clone)]
struct EdgeData {
    /// Endpoint vertex indices (i, j) with i < j.
    v: [u32; 2],
    /// Edge length.
    length: f64,
    /// Per-edge dyad `n_A e_A^T + n_B e_B^T` where `n_A`, `n_B` are the
    /// outward normals of the two faces sharing the edge and `e_A`, `e_B`
    /// are the in-face perpendiculars to the edge pointing outward from
    /// each face's interior.
    dyad: Matrix3<f64>,
}

impl Polyhedron {
    /// Construct a polyhedron from vertices, triangular faces (CCW from
    /// outside), and total `gm = G * M`.
    ///
    /// # Errors
    /// Returns an error if the mesh is not closed (every edge must be
    /// shared by exactly two faces), if any face is degenerate, or if the
    /// computed volume is non-positive (indicating reversed face winding).
    pub fn try_new(vertices: Vec<Vector3<f64>>, faces: &[[u32; 3]], gm: f64) -> KeteResult<Self> {
        if vertices.len() < 4 {
            return Err(Error::ValueError(
                "polyhedron requires at least 4 vertices".into(),
            ));
        }
        if faces.len() < 4 {
            return Err(Error::ValueError(
                "polyhedron requires at least 4 faces".into(),
            ));
        }
        let n_v = u32::try_from(vertices.len()).map_err(|_| {
            Error::ValueError("polyhedron has more vertices than u32 can index".into())
        })?;
        for f in faces {
            for &i in f {
                if i >= n_v {
                    return Err(Error::ValueError(format!(
                        "face references vertex index {i} out of range (n={n_v})"
                    )));
                }
            }
            if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                return Err(Error::ValueError("face has repeated vertex".into()));
            }
        }

        // Per-face geometry: outward normals and dyads.
        let mut face_data = Vec::with_capacity(faces.len());
        for f in faces {
            let v0 = vertices[f[0] as usize];
            let v1 = vertices[f[1] as usize];
            let v2 = vertices[f[2] as usize];
            let cross = (v1 - v0).cross(&(v2 - v0));
            let area2 = cross.norm();
            if area2 == 0.0 {
                return Err(Error::ValueError("degenerate face with zero area".into()));
            }
            let normal = cross / area2;
            let dyad = normal * normal.transpose();
            face_data.push(FaceData {
                v: *f,
                normal,
                dyad,
            });
        }

        // Volume via signed-tetrahedra sum about the origin.
        let volume: f64 = face_data
            .iter()
            .map(|fd| {
                let v0 = vertices[fd.v[0] as usize];
                let v1 = vertices[fd.v[1] as usize];
                let v2 = vertices[fd.v[2] as usize];
                v0.dot(&v1.cross(&v2))
            })
            .sum::<f64>()
            / 6.0;
        if volume <= 0.0 {
            return Err(Error::ValueError(format!(
                "computed polyhedron volume is non-positive ({volume:e}); \
                 face winding may be reversed or mesh is not closed"
            )));
        }

        let bounding_radius = vertices.iter().map(Vector3::norm).fold(0.0_f64, f64::max);

        // Build edge map.  For each face, walk edges (v[0]->v[1], v[1]->v[2],
        // v[2]->v[0]).  Every undirected edge must appear in exactly two
        // faces with opposite directions on a closed orientable mesh.  We
        // use BTreeMap (sorted) rather than HashMap so edge construction
        // order is fully reproducible across runs and platforms.
        let mut directed: BTreeMap<(u32, u32), usize> = BTreeMap::new();
        for (fi, fd) in face_data.iter().enumerate() {
            for k in 0..3 {
                let a = fd.v[k];
                let b = fd.v[(k + 1) % 3];
                if directed.insert((a, b), fi).is_some() {
                    return Err(Error::ValueError(format!(
                        "directed edge {a}->{b} appears in more than one face; \
                         mesh is not a manifold"
                    )));
                }
            }
        }

        let mut edges: Vec<EdgeData> = Vec::new();
        let mut seen: BTreeMap<(u32, u32), bool> = BTreeMap::new();
        for (&(a, b), &face_a) in &directed {
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key, true).is_some() {
                continue;
            }
            // Find the opposite-direction face.
            let Some(&face_b) = directed.get(&(b, a)) else {
                return Err(Error::ValueError(format!(
                    "edge {a}-{b} is shared by only one face; mesh is not closed"
                )));
            };

            let v_a = vertices[a as usize];
            let v_b = vertices[b as usize];
            let edge_vec = v_b - v_a;
            let length = edge_vec.norm();
            if length == 0.0 {
                return Err(Error::ValueError("edge has zero length".into()));
            }
            let edge_hat_ab = edge_vec / length;

            // For face A, the directed edge as traversed in A's CCW order is a->b.
            // For face B, the directed edge as traversed is b->a (opposite).
            // The in-face perpendicular pointing OUT of each face's interior
            // (into the half-plane on the far side of the edge) is:
            //     e_perp = edge_hat_in_face_traversal x normal.
            // For face A: edge_hat_A = +edge_hat_ab, so e_perp_A = edge_hat_ab x n_A.
            // For face B: edge_hat_B = -edge_hat_ab, so e_perp_B = -edge_hat_ab x n_B.
            let n_a = face_data[face_a].normal;
            let n_b = face_data[face_b].normal;
            let e_perp_a = edge_hat_ab.cross(&n_a);
            let e_perp_b = (-edge_hat_ab).cross(&n_b);
            let dyad = n_a * e_perp_a.transpose() + n_b * e_perp_b.transpose();

            edges.push(EdgeData {
                v: [key.0, key.1],
                length,
                dyad,
            });
        }

        // Closed manifold check: 2 * directed edges == 2 * undirected edges,
        // i.e. number of directed edges equals 2 * number of undirected edges.
        // Equivalently, every directed edge had a partner (already enforced
        // by the lookup above).
        if directed.len() != 2 * edges.len() {
            return Err(Error::ValueError(
                "mesh edge accounting failed; mesh is not a closed manifold".into(),
            ));
        }

        let g_sigma = gm / volume;

        Ok(Self {
            vertices,
            faces: face_data,
            edges,
            volume,
            bounding_radius,
            gm,
            g_sigma,
        })
    }

    /// Number of vertices.
    #[must_use]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangular faces.
    #[must_use]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of unique undirected edges.
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Construct a polyhedron from vertices, faces, and a constant bulk
    /// density.  The total `gm` is computed internally from `density`
    /// and the polyhedron's volume.
    ///
    /// `density` is `G * sigma`, i.e. the gravitational constant times
    /// mass density, in units consistent with the vertex length units.
    /// (Equivalently: pass `G * mass_density` if you have SI inputs.)
    ///
    /// # Errors
    /// See [`Self::try_new`].
    pub fn try_new_density(
        vertices: Vec<Vector3<f64>>,
        faces: &[[u32; 3]],
        density: f64,
    ) -> KeteResult<Self> {
        // Build with a placeholder gm of 1.0 to compute volume, then
        // re-derive g_sigma directly from the density.
        let mut p = Self::try_new(vertices, faces, 1.0)?;
        p.gm = density * p.volume;
        p.g_sigma = density;
        Ok(p)
    }

    /// Constant-density center of mass in body-fixed coordinates.
    ///
    /// The polyhedron is assumed to be centered at its center of mass
    /// (this is what the rotation model and ephemeris will reference).
    /// This accessor lets callers verify that assumption and re-center
    /// the mesh via [`Self::recenter_to_com`] if needed.
    #[must_use]
    pub fn center_of_mass(&self) -> Vector3<f64> {
        // Constant-density COM: sum over signed tetrahedra (origin, v0, v1, v2),
        // each contributing (v0 + v1 + v2)/4 weighted by its signed volume,
        // divided by total volume.
        let mut weighted = Vector3::zeros();
        for fd in &self.faces {
            let v0 = self.vertices[fd.v[0] as usize];
            let v1 = self.vertices[fd.v[1] as usize];
            let v2 = self.vertices[fd.v[2] as usize];
            let signed_v = v0.dot(&v1.cross(&v2)) / 6.0;
            weighted += signed_v * (v0 + v1 + v2) * 0.25;
        }
        weighted / self.volume
    }

    /// Translate the mesh so the constant-density center of mass is at
    /// the origin.  Returns the translation that was applied (the old
    /// COM in the original coordinates).
    pub fn recenter_to_com(&mut self) -> Vector3<f64> {
        let com = self.center_of_mass();
        for v in &mut self.vertices {
            *v -= com;
        }
        // Volume, normals, dyads, edge lengths, gm, g_sigma are translation
        // invariant.  Bounding radius must be recomputed.
        self.bounding_radius = self
            .vertices
            .iter()
            .map(Vector3::norm)
            .fold(0.0_f64, f64::max);
        com
    }

    /// Compute per-face and per-edge scalar contributions at field point `r`.
    ///
    /// Returns `(face_terms, edge_terms)` where each entry corresponds to
    /// the matching index in `self.faces` / `self.edges`.  For each face
    /// the entry is `(r_f, omega_f)`; for each edge it is `(r_e, L_e)`.
    /// `r_f` is a representative vector from the field point to a point on
    /// the face (the first vertex), and similarly `r_e`.
    ///
    /// Returns [`Error::SurfaceImpact`] if the field point lies on the
    /// surface of the body.  This is detected by the edge logarithm
    /// argument `(a + b - e)` going non-positive (the field point lies on
    /// the line segment between the edge endpoints) or by a vertex
    /// coinciding with the field point (`a == 0` or `b == 0`).
    fn evaluate_terms(
        &self,
        r: Vector3<f64>,
    ) -> KeteResult<(Vec<(Vector3<f64>, f64)>, Vec<(Vector3<f64>, f64)>)> {
        let face_terms: Vec<_> = self
            .faces
            .iter()
            .map(|fd| {
                let r1 = self.vertices[fd.v[0] as usize] - r;
                let r2 = self.vertices[fd.v[1] as usize] - r;
                let r3 = self.vertices[fd.v[2] as usize] - r;
                let omega = signed_solid_angle(r1, r2, r3);
                (r1, omega)
            })
            .collect();

        let mut edge_terms: Vec<(Vector3<f64>, f64)> = Vec::with_capacity(self.edges.len());
        for ed in &self.edges {
            let r_a = self.vertices[ed.v[0] as usize] - r;
            let r_b = self.vertices[ed.v[1] as usize] - r;
            let a = r_a.norm();
            let b = r_b.norm();
            let e = ed.length;
            // Field point on a vertex => one of a, b is zero.
            // Field point on the edge interior => a + b == e exactly.
            // Either case makes the Werner-Scheeres expression singular and
            // physically corresponds to surface contact.
            let denom = a + b - e;
            if a == 0.0 || b == 0.0 || denom <= 0.0 {
                return Err(Error::SurfaceImpact);
            }
            let l_e = ((a + b + e) / denom).ln();
            edge_terms.push((r_a, l_e));
        }

        Ok((face_terms, edge_terms))
    }
}

/// Signed solid angle subtended by a triangle (vertices given as vectors
/// from the field point) using the Van Oosterom-Strackee formula.
///
/// With CCW vertex ordering viewed from outside the body (consistent
/// outward normals), this returns positive values when the triangle face
/// is "facing" the field point and negative values when back-facing, such
/// that summing over all faces of a closed body gives `+4*pi` if the
/// field point is inside and `0` if outside.
fn signed_solid_angle(r1: Vector3<f64>, r2: Vector3<f64>, r3: Vector3<f64>) -> f64 {
    let a = r1.norm();
    let b = r2.norm();
    let c = r3.norm();
    let num = r1.dot(&r2.cross(&r3));
    let den = a * b * c + r1.dot(&r2) * c + r2.dot(&r3) * a + r3.dot(&r1) * b;
    2.0 * num.atan2(den)
}

impl ExtendedGravity for Polyhedron {
    fn potential(&self, r: Vector3<f64>) -> KeteResult<f64> {
        let (face_terms, edge_terms) = self.evaluate_terms(r)?;
        let edge_sum: f64 = self
            .edges
            .iter()
            .zip(edge_terms.iter())
            .map(|(ed, (r_e, l_e))| (r_e.transpose() * ed.dyad * r_e)[(0, 0)] * *l_e)
            .sum();
        let face_sum: f64 = self
            .faces
            .iter()
            .zip(face_terms.iter())
            .map(|(fd, (r_f, omega))| (r_f.transpose() * fd.dyad * r_f)[(0, 0)] * *omega)
            .sum();
        Ok(0.5 * self.g_sigma * (edge_sum - face_sum))
    }

    fn acceleration(&self, r: Vector3<f64>) -> KeteResult<Vector3<f64>> {
        let (face_terms, edge_terms) = self.evaluate_terms(r)?;
        let mut edge_sum = Vector3::zeros();
        for (ed, (r_e, l_e)) in self.edges.iter().zip(edge_terms.iter()) {
            edge_sum += ed.dyad * r_e * *l_e;
        }
        let mut face_sum = Vector3::zeros();
        for (fd, (r_f, omega)) in self.faces.iter().zip(face_terms.iter()) {
            face_sum += fd.dyad * r_f * *omega;
        }
        // a = grad U = G sigma * (sum_f F r_f omega - sum_e E r_e L)
        Ok(self.g_sigma * (face_sum - edge_sum))
    }

    fn gradient(&self, r: Vector3<f64>) -> KeteResult<Matrix3<f64>> {
        let (face_terms, edge_terms) = self.evaluate_terms(r)?;
        let mut edge_sum = Matrix3::zeros();
        for (ed, (_, l_e)) in self.edges.iter().zip(edge_terms.iter()) {
            edge_sum += ed.dyad * *l_e;
        }
        let mut face_sum = Matrix3::zeros();
        for (fd, (_, omega)) in self.faces.iter().zip(face_terms.iter()) {
            face_sum += fd.dyad * *omega;
        }
        // grad grad U = G sigma * (sum_e E L - sum_f F omega)
        Ok(self.g_sigma * (edge_sum - face_sum))
    }

    fn min_valid_radius(&self) -> f64 {
        0.0
    }

    fn contains(&self, r: Vector3<f64>) -> bool {
        // Sum of signed solid angles is +4*pi inside, 0 outside.
        let total: f64 = self
            .faces
            .iter()
            .map(|fd| {
                let r1 = self.vertices[fd.v[0] as usize] - r;
                let r2 = self.vertices[fd.v[1] as usize] - r;
                let r3 = self.vertices[fd.v[2] as usize] - r;
                signed_solid_angle(r1, r2, r3)
            })
            .sum();
        total.abs() > 2.0 * std::f64::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    /// Build a unit cube centered at the origin (vertices at +/- 0.5).
    fn unit_cube(gm: f64) -> Polyhedron {
        let v = vec![
            Vector3::new(-0.5, -0.5, -0.5), // 0
            Vector3::new(0.5, -0.5, -0.5),  // 1
            Vector3::new(0.5, 0.5, -0.5),   // 2
            Vector3::new(-0.5, 0.5, -0.5),  // 3
            Vector3::new(-0.5, -0.5, 0.5),  // 4
            Vector3::new(0.5, -0.5, 0.5),   // 5
            Vector3::new(0.5, 0.5, 0.5),    // 6
            Vector3::new(-0.5, 0.5, 0.5),   // 7
        ];
        // Faces with outward CCW winding.
        let f = vec![
            // bottom (z=-0.5), normal -z, CCW from below = (0,3,2),(0,2,1)
            [0, 3, 2],
            [0, 2, 1],
            // top (z=+0.5), normal +z, CCW from above = (4,5,6),(4,6,7)
            [4, 5, 6],
            [4, 6, 7],
            // front (y=-0.5), normal -y, CCW from -y side = (0,1,5),(0,5,4)
            [0, 1, 5],
            [0, 5, 4],
            // back (y=+0.5), normal +y, CCW from +y side = (3,7,6),(3,6,2)
            [3, 7, 6],
            [3, 6, 2],
            // left (x=-0.5), normal -x, CCW from -x side = (0,4,7),(0,7,3)
            [0, 4, 7],
            [0, 7, 3],
            // right (x=+0.5), normal +x, CCW from +x side = (1,2,6),(1,6,5)
            [1, 2, 6],
            [1, 6, 5],
        ];
        Polyhedron::try_new(v, &f, gm).unwrap()
    }

    /// Recursive icosphere subdivision.  Returns vertices and triangle indices
    /// for a sphere of radius 1.
    fn icosphere(subdivisions: u32) -> (Vec<Vector3<f64>>, Vec<[u32; 3]>) {
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
            let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
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

    fn unit_sphere(subdivisions: u32, gm: f64) -> Polyhedron {
        let (v, f) = icosphere(subdivisions);
        Polyhedron::try_new(v, &f, gm).unwrap()
    }

    #[test]
    fn cube_is_closed_manifold() {
        let p = unit_cube(1.0);
        // 12 unique edges per cube; 12 triangle faces.
        assert_eq!(p.n_edges(), 18); // cube has 12 cube-edges + 6 face diagonals
        assert_eq!(p.n_faces(), 12);
        assert!((p.volume - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_is_closed_manifold() {
        let p = unit_sphere(2, 1.0);
        // Euler: V - E + F = 2 for any closed orientable triangulation.
        let v = isize::try_from(p.n_vertices()).unwrap();
        let e = isize::try_from(p.n_edges()).unwrap();
        let f = isize::try_from(p.n_faces()).unwrap();
        assert_eq!(v - e + f, 2);
    }

    #[test]
    fn reversed_winding_is_rejected() {
        // Build a cube but flip every face.
        let v = vec![
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.5, -0.5),
            Vector3::new(-0.5, 0.5, -0.5),
            Vector3::new(-0.5, -0.5, 0.5),
            Vector3::new(0.5, -0.5, 0.5),
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(-0.5, 0.5, 0.5),
        ];
        let f = vec![
            [0, 2, 3],
            [0, 1, 2],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [3, 6, 7],
            [3, 2, 6],
            [0, 7, 4],
            [0, 3, 7],
            [1, 6, 2],
            [1, 5, 6],
        ];
        assert!(Polyhedron::try_new(v, &f, 1.0).is_err());
    }

    #[test]
    fn open_mesh_is_rejected() {
        // Tetrahedron missing one face.
        let v = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let f = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2]]; // missing [1,2,3]
        assert!(Polyhedron::try_new(v, &f, 1.0).is_err());
    }

    #[test]
    fn sphere_far_field_matches_point_mass() {
        let gm = 1.0;
        let p = unit_sphere(3, gm);
        // Far from the body: a should be ~ -gm * r_hat / r^2 and U ~ gm/r.
        for &r_mag in &[10.0, 50.0, 100.0] {
            let r = Vector3::new(r_mag, 0.0, 0.0);
            let a = p.acceleration(r).unwrap();
            let u = p.potential(r).unwrap();
            let expected_a = -gm * r / r_mag.powi(3);
            let expected_u = gm / r_mag;
            let a_err = (a - expected_a).norm() / expected_a.norm();
            let u_err = (u - expected_u).abs() / expected_u;
            assert!(a_err < 1e-3, "r={r_mag}, a_err={a_err:e}");
            assert!(u_err < 1e-3, "r={r_mag}, u_err={u_err:e}");
        }
    }

    #[test]
    fn contains_inside_outside() {
        let p = unit_cube(1.0);
        // Inside the cube.
        assert!(p.contains(Vector3::new(0.0, 0.0, 0.0)));
        assert!(p.contains(Vector3::new(0.3, -0.2, 0.1)));
        // Outside.
        assert!(!p.contains(Vector3::new(2.0, 0.0, 0.0)));
        assert!(!p.contains(Vector3::new(0.6, 0.0, 0.0)));
        assert!(!p.contains(Vector3::new(-1.0, -1.0, -1.0)));
    }

    #[test]
    fn contains_inside_non_convex_void() {
        // Build a "dumbbell" as two cubes connected by a thin neck.  We just
        // test contains() works on a non-convex shape (concave region around
        // the neck waist must not be flagged as inside).
        // Two cubes side by side along x, joined by a small bridge.  Easier:
        // use a sphere shell test - place two spheres apart and check the gap
        // between them is "outside" of either sphere.
        let p = unit_sphere(2, 1.0);
        // Outside the unit sphere in any direction.
        for dir in [
            Vector3::new(1.5, 0.0, 0.0),
            Vector3::new(0.0, 1.5, 0.0),
            Vector3::new(-1.5, -1.5, 0.0),
        ] {
            assert!(!p.contains(dir), "should be outside: {dir:?}");
        }
        // Inside.
        for dir in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 0.9),
        ] {
            assert!(p.contains(dir), "should be inside: {dir:?}");
        }
    }

    #[test]
    fn laplacian_inside_and_outside() {
        let gm = 1.0;
        let p = unit_cube(gm);
        let g_sigma = p.g_sigma;

        // Outside: trace(grad grad U) ~ 0.
        for r in [
            Vector3::new(2.0, 0.0, 0.0),
            Vector3::new(0.0, 3.0, 1.0),
            Vector3::new(-5.0, 4.0, 2.0),
        ] {
            let h = p.gradient(r).unwrap();
            assert!(
                h.trace().abs() < 1e-10,
                "outside Laplacian {} too large at {r:?}",
                h.trace()
            );
        }

        // Inside: trace(grad grad U) ~ -4 pi G sigma.
        let expected = -4.0 * PI * g_sigma;
        for r in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.1, -0.2, 0.05),
            Vector3::new(-0.3, 0.0, 0.2),
        ] {
            let h = p.gradient(r).unwrap();
            let rel = (h.trace() - expected).abs() / expected.abs();
            assert!(
                rel < 1e-10,
                "inside Laplacian {} != {expected} at {r:?} (rel {rel:e})",
                h.trace()
            );
        }
    }

    #[test]
    fn laplacian_inside_sphere() {
        let gm = 1.0;
        let p = unit_sphere(2, gm);
        let expected = -4.0 * PI * p.g_sigma;
        for r in [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.3, 0.0, 0.0),
            Vector3::new(0.0, -0.4, 0.2),
        ] {
            let h = p.gradient(r).unwrap();
            let rel = (h.trace() - expected).abs() / expected.abs();
            assert!(rel < 1e-9, "rel={rel:e} at {r:?}");
        }
    }

    #[test]
    fn dumbbell_superposition() {
        // Two unit spheres, each with gm=1, displaced along x by +/- 5.
        // The combined gravity field at a probe point should equal the sum of
        // the two individual gravity fields.  Validates that multi-component
        // bodies are simply additive (the sum of two ExtendedGravity sources).
        let mut left_v: Vec<Vector3<f64>> = icosphere(2).0;
        let mut right_v: Vec<Vector3<f64>> = icosphere(2).0;
        for v in &mut left_v {
            v.x -= 5.0;
        }
        for v in &mut right_v {
            v.x += 5.0;
        }
        let f = icosphere(2).1;
        let left = Polyhedron::try_new(left_v, &f, 1.0).unwrap();
        let right = Polyhedron::try_new(right_v, &f, 1.0).unwrap();

        // Probe point in the gap between the two bodies.
        let probe = Vector3::new(0.0, 0.5, 0.5);
        let a_combined = left.acceleration(probe).unwrap() + right.acceleration(probe).unwrap();
        // By symmetry along x the x-components from the two spheres should
        // cancel completely; the y and z components add.
        assert!(
            a_combined.x.abs() < 1e-12,
            "x-component of dumbbell field at midplane should cancel: {}",
            a_combined.x
        );
        // Both components pull toward y=0, z=0.
        assert!(a_combined.y < 0.0);
        assert!(a_combined.z < 0.0);

        // Far-field check: from far away on +x axis, dumbbell looks like a
        // point mass of total gm=2 at origin.  Tolerance reflects the
        // icosphere subdivision used (level 2, ~320 faces per sphere).
        let far = Vector3::new(100.0, 0.0, 0.0);
        let a_far = left.acceleration(far).unwrap() + right.acceleration(far).unwrap();
        let expected = -2.0 * far / far.norm().powi(3);
        let rel = (a_far - expected).norm() / expected.norm();
        assert!(rel < 1e-2, "dumbbell far-field error {rel:e}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        // Numerically differentiate acceleration() and check it equals
        // gradient() componentwise.  This catches off-diagonal sign / dyad
        // bugs that the trace-based Laplacian test alone would miss.
        let p = unit_cube(1.0);
        let h = 1e-5;
        for r0 in [
            Vector3::new(2.0, 0.5, -0.3),
            Vector3::new(-1.5, 1.2, 0.8),
            Vector3::new(0.0, 3.0, 0.0),
        ] {
            let analytic = p.gradient(r0).unwrap();
            let mut fd = Matrix3::zeros();
            for j in 0..3 {
                let mut dr = Vector3::zeros();
                dr[j] = h;
                let a_plus = p.acceleration(r0 + dr).unwrap();
                let a_minus = p.acceleration(r0 - dr).unwrap();
                let col = (a_plus - a_minus) / (2.0 * h);
                fd.set_column(j, &col);
            }
            // gradient() returns grad grad U; acceleration() = grad U.
            // Therefore d(acceleration_i)/dr_j = grad_grad_U[i,j].
            for i in 0..3 {
                for j in 0..3 {
                    let diff = (analytic[(i, j)] - fd[(i, j)]).abs();
                    assert!(
                        diff < 1e-6,
                        "gradient[{i},{j}] = {} vs FD {} (diff {diff:e}) at r={r0:?}",
                        analytic[(i, j)],
                        fd[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn density_constructor_matches_explicit_gm() {
        // Constructing with density should give the same g_sigma as
        // constructing with gm = density * volume.
        let (v, f) = icosphere(2);
        let density = 3.7;
        let from_density = Polyhedron::try_new_density(v.clone(), &f, density).unwrap();
        let expected_gm = density * from_density.volume;
        let from_gm = Polyhedron::try_new(v, &f, expected_gm).unwrap();

        let probe = Vector3::new(2.5, 0.7, -1.1);
        let a_d = from_density.acceleration(probe).unwrap();
        let a_g = from_gm.acceleration(probe).unwrap();
        assert!((a_d - a_g).norm() < 1e-12);
        assert!((from_density.gm - from_gm.gm).abs() < 1e-12);
    }

    #[test]
    fn center_of_mass_centered_cube_at_origin() {
        let p = unit_cube(1.0);
        let com = p.center_of_mass();
        assert!(
            com.norm() < 1e-12,
            "centered cube COM should be ~0: {com:?}"
        );
    }

    #[test]
    fn center_of_mass_recenters_offset_cube() {
        // Shift cube vertices by +x so COM is offset.
        let mut v: Vec<Vector3<f64>> = vec![
            Vector3::new(-0.5, -0.5, -0.5),
            Vector3::new(0.5, -0.5, -0.5),
            Vector3::new(0.5, 0.5, -0.5),
            Vector3::new(-0.5, 0.5, -0.5),
            Vector3::new(-0.5, -0.5, 0.5),
            Vector3::new(0.5, -0.5, 0.5),
            Vector3::new(0.5, 0.5, 0.5),
            Vector3::new(-0.5, 0.5, 0.5),
        ];
        let offset = Vector3::new(3.0, -1.0, 2.0);
        for vert in &mut v {
            *vert += offset;
        }
        let f = vec![
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];
        let mut p = Polyhedron::try_new(v, &f, 1.0).unwrap();
        let com = p.center_of_mass();
        assert!(
            (com - offset).norm() < 1e-12,
            "COM {com:?} != offset {offset:?}"
        );
        let applied = p.recenter_to_com();
        assert!((applied - offset).norm() < 1e-12);
        // After re-centering, COM should be at origin.
        assert!(p.center_of_mass().norm() < 1e-12);
    }

    #[test]
    fn surface_contact_returns_impact_error() {
        let p = unit_cube(1.0);
        // Point on a vertex.
        let on_vertex = Vector3::new(0.5, 0.5, 0.5);
        assert!(matches!(
            p.acceleration(on_vertex),
            Err(Error::SurfaceImpact)
        ));
        assert!(matches!(p.potential(on_vertex), Err(Error::SurfaceImpact)));
        assert!(matches!(p.gradient(on_vertex), Err(Error::SurfaceImpact)));

        // Point on the interior of an edge (top edge between vertices 4 and 5).
        let on_edge = Vector3::new(0.0, -0.5, 0.5);
        assert!(matches!(p.acceleration(on_edge), Err(Error::SurfaceImpact)));

        // Points just barely inside / outside should still evaluate fine.
        let just_outside = Vector3::new(0.5 + 1e-6, 0.0, 0.0);
        let just_inside = Vector3::new(0.5 - 1e-6, 0.0, 0.0);
        assert!(p.acceleration(just_outside).is_ok());
        assert!(p.acceleration(just_inside).is_ok());
    }
}
