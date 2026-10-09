// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Closed triangle meshes.
//!
//! A [`TriMesh`] is an indexed triangle mesh: a list of vertices and a list of
//! faces, each face three indices into the vertex list. Vertices are shared
//! between faces, so the mesh carries its edge topology.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI};

use nalgebra::{Unit, UnitVector3, Vector3};

use crate::errors::{Error, KeteResult};

/// A face is degenerate when twice its area is below this fraction of the square of
/// its longest edge, i.e. its vertices are collinear to within rounding.
const DEGENERATE_TOLERANCE: f64 = 1e-12;

/// An edge of a [`TriMesh`] and the two faces that share it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    /// Vertex indices `[a, b]`, with `a < b`.
    pub vertices: [u32; 2],

    /// The two faces sharing the edge: `faces[0]` traverses it from `a` to `b`,
    /// `faces[1]` from `b` to `a`.
    pub faces: [u32; 2],
}

/// A closed, consistently wound, outward-facing triangle mesh.
///
/// [`TriMesh::new`] validates the mesh. Every edge is shared by exactly two
/// faces, which traverse it in opposite directions. The faces are wound
/// counter-clockwise seen from outside, and no face is degenerate. Vertex
/// manifoldness and self-intersection are not checked.
#[derive(Debug, Clone)]
pub struct TriMesh {
    /// Vertex positions.
    vertices: Vec<Vector3<f64>>,

    /// Faces as indices into `vertices`, counter-clockwise seen from outside.
    faces: Vec<[u32; 3]>,

    /// Every edge once, sorted by vertex indices.
    edges: Vec<Edge>,
}

impl TriMesh {
    /// Build a mesh from vertices and faces, validating it.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] in these cases:
    /// - There are fewer than 4 vertices or faces.
    /// - A vertex is not finite.
    /// - A face index is out of range or repeated within a face.
    /// - A face is degenerate.
    /// - An edge is not shared by exactly two faces that traverse it in
    ///   opposite directions: the mesh is open or wound inconsistently.
    /// - The enclosed volume is not positive: the faces are wound inward.
    pub fn new(vertices: Vec<Vector3<f64>>, faces: Vec<[u32; 3]>) -> KeteResult<Self> {
        if vertices.len() < 4 || faces.len() < 4 {
            return Err(Error::ValueError(
                "A closed triangle mesh needs at least 4 vertices and 4 faces.".into(),
            ));
        }
        if u32::try_from(vertices.len()).is_err() || u32::try_from(faces.len()).is_err() {
            return Err(Error::ValueError(
                "Mesh has more vertices or faces than fit in a u32 index.".into(),
            ));
        }
        if let Some(idx) = vertices
            .iter()
            .position(|v| !v.iter().all(|x| x.is_finite()))
        {
            return Err(Error::ValueError(format!("Vertex {idx} is not finite.")));
        }
        for (idx, face) in faces.iter().enumerate() {
            if face.iter().any(|&k| k as usize >= vertices.len()) {
                return Err(Error::ValueError(format!(
                    "Face {idx} has a vertex index out of range."
                )));
            }
            if face[0] == face[1] || face[1] == face[2] || face[2] == face[0] {
                return Err(Error::ValueError(format!(
                    "Face {idx} repeats a vertex index."
                )));
            }
            let [a, b, c] = face.map(|k| vertices[k as usize]);
            let longest = (b - a)
                .norm_squared()
                .max((c - b).norm_squared())
                .max((a - c).norm_squared());
            if (b - a).cross(&(c - a)).norm() <= DEGENERATE_TOLERANCE * longest {
                return Err(Error::ValueError(format!("Face {idx} is degenerate.")));
            }
        }

        let edges = build_edges(&faces)?;
        let mesh = Self {
            vertices,
            faces,
            edges,
        };
        let volume = mesh.volume();
        if volume.is_nan() || volume <= 0.0 {
            return Err(Error::ValueError(
                "Mesh encloses a non-positive volume: the faces are wound inward \
                 (clockwise seen from outside)."
                    .into(),
            ));
        }
        Ok(mesh)
    }

    /// Triangulated ellipsoid with semi-axes `x_scale`, `y_scale`, `z_scale` along the
    /// coordinate axes, centered on the origin.
    ///
    /// `n_div` is the number of divisions from pole to equator, at least 1. The
    /// vertices lie on `2 * n_div - 1` rings of constant colatitude between the
    /// poles, with `4 * min(l, 2 * n_div - l)` vertices on ring `l`, and the
    /// rings are joined by triangle strips. The mesh has `8 * n_div * n_div`
    /// faces and `4 * n_div * n_div + 2` vertices. Every vertex lies on the
    /// ellipsoid, so the mesh is inscribed in it.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `n_div` is 0 or a semi-axis is not positive and finite.
    pub fn new_ellipsoid(n_div: u32, x_scale: f64, y_scale: f64, z_scale: f64) -> KeteResult<Self> {
        if n_div == 0 {
            return Err(Error::ValueError("n_div must be at least 1.".into()));
        }
        if [x_scale, y_scale, z_scale]
            .iter()
            .any(|s| !s.is_finite() || *s <= 0.0)
        {
            return Err(Error::ValueError(
                "Ellipsoid semi-axes must be positive and finite.".into(),
            ));
        }
        let n = n_div as usize;

        // Vertex rings from the north pole (level 0) to the south pole (level
        // 2n). Level l lies at colatitude l * pi / (2n) and holds 4k points
        // evenly spaced in longitude from 0, where k = min(l, 2n - l). A pole is
        // a single point. Every quarter of longitude holds k + 1 points of the
        // ring, counting the points on both of its boundary meridians.
        let ring_size = |level: usize| {
            let k = level.min(2 * n - level);
            if k == 0 { 1 } else { 4 * k }
        };
        let mut ring_start = Vec::with_capacity(2 * n + 1);
        let mut points: Vec<Vector3<f64>> = Vec::with_capacity(4 * n * n + 2);
        for level in 0..=2 * n {
            ring_start.push(points.len());
            let theta = level as f64 * FRAC_PI_2 / n as f64;
            let size = ring_size(level);
            for j in 0..size {
                let phi = j as f64 * 2.0 * PI / size as f64;
                points.push(Vector3::new(
                    x_scale * theta.sin() * phi.cos(),
                    y_scale * theta.sin() * phi.sin(),
                    z_scale * theta.cos(),
                ));
            }
        }

        // Point index of position `p` along the ring at `level`, counted from
        // longitude 0 and wrapping around the ring.
        let index = |level: usize, p: usize| ring_start[level] + p % ring_size(level);

        // Each band joins two adjacent rings, a smaller one with k + 1 points
        // per quarter and a larger one with k + 2. In each quarter the band is
        // a strip of k + 1 triangles with an edge on the larger ring and k
        // triangles with an edge on the smaller ring.
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(8 * n * n);
        for upper in 0..2 * n {
            let lower = upper + 1;
            let (small, large) = if upper < n {
                (upper, lower)
            } else {
                (lower, upper)
            };
            let k = small.min(2 * n - small);
            for quarter in 0..4 {
                let s = |a: usize| index(small, quarter * k + a);
                let l = |b: usize| index(large, quarter * (k + 1) + b);
                for a in 0..=k {
                    tris.push([s(a), l(a), l(a + 1)]);
                }
                for a in 0..k {
                    tris.push([s(a), l(a + 1), s(a + 1)]);
                }
            }
        }

        // The ellipsoid is convex and contains the origin, so a face is wound
        // outward exactly when its normal points away from the origin.
        let faces = tris
            .into_iter()
            .map(|[a, b, c]| {
                let (pa, pb, pc) = (points[a], points[b], points[c]);
                let outward = pa.dot(&(pb - pa).cross(&(pc - pa))) > 0.0;
                let [a, b, c] = if outward { [a, b, c] } else { [a, c, b] };
                Ok([to_index(a)?, to_index(b)?, to_index(c)?])
            })
            .collect::<KeteResult<Vec<_>>>()?;
        Self::new(points, faces)
    }

    /// Vertex positions.
    #[must_use]
    pub fn vertices(&self) -> &[Vector3<f64>] {
        &self.vertices
    }

    /// Faces as vertex indices, counter-clockwise seen from outside.
    #[must_use]
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Every edge once, sorted by vertex indices.
    #[must_use]
    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    /// The three vertex positions of a face of this mesh, in winding order.
    ///
    /// # Panics
    /// Panics if an index of `face` is not a vertex index of this mesh.
    #[must_use]
    pub fn face_vertices(&self, face: &[u32; 3]) -> [Vector3<f64>; 3] {
        face.map(|k| self.vertices[k as usize])
    }

    /// Outward unit normal of a face of this mesh.
    ///
    /// # Panics
    /// Panics if an index of `face` is not a vertex index of this mesh.
    #[must_use]
    pub fn face_normal(&self, face: &[u32; 3]) -> UnitVector3<f64> {
        let [a, b, c] = self.face_vertices(face);
        Unit::new_normalize((b - a).cross(&(c - a)))
    }

    /// Area of a face of this mesh.
    ///
    /// # Panics
    /// Panics if an index of `face` is not a vertex index of this mesh.
    #[must_use]
    pub fn face_area(&self, face: &[u32; 3]) -> f64 {
        let [a, b, c] = self.face_vertices(face);
        (b - a).cross(&(c - a)).norm() / 2.0
    }

    /// Total surface area.
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        self.faces.iter().map(|f| self.face_area(f)).sum()
    }

    /// Enclosed volume.
    ///
    /// Sum of the signed volumes of the tetrahedra joining each face to the origin;
    /// for a closed mesh the result does not depend on the origin.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = self.face_vertices(f);
                a.dot(&b.cross(&c))
            })
            .sum::<f64>()
            / 6.0
    }

    /// Centroid of the enclosed volume, the center of mass at uniform density.
    #[must_use]
    pub fn centroid(&self) -> Vector3<f64> {
        let mut weighted = Vector3::zeros();
        let mut total = 0.0;
        for face in &self.faces {
            let [a, b, c] = self.face_vertices(face);
            // six times the signed volume of the tetrahedron (origin, a, b, c)
            let vol6 = a.dot(&b.cross(&c));
            weighted += (a + b + c) * vol6;
            total += vol6;
        }
        // the tetrahedron centroid is (0 + a + b + c) / 4
        weighted / (4.0 * total)
    }

    /// Largest distance of a vertex from the origin: the radius of the smallest
    /// origin-centered sphere containing the mesh.
    #[must_use]
    pub fn bounding_radius(&self) -> f64 {
        self.vertices.iter().map(Vector3::norm).fold(0.0, f64::max)
    }

    /// Move every vertex by `offset`.
    ///
    /// # Errors
    /// [`Error::ValueError`] if `offset` is not finite. The mesh is unchanged.
    pub fn translate(&mut self, offset: &Vector3<f64>) -> KeteResult<()> {
        if !offset.iter().all(|x| x.is_finite()) {
            return Err(Error::ValueError("Mesh offset must be finite.".into()));
        }
        self.vertices.iter_mut().for_each(|v| *v += offset);
        Ok(())
    }

    /// The mesh moved so its volume centroid is at the origin, and the centroid it
    /// had before the move (the vector that was subtracted from every vertex).
    #[must_use]
    pub fn centered(mut self) -> (Self, Vector3<f64>) {
        // The centroid is finite, because the vertices are finite and the
        // volume is positive.
        let centroid = self.centroid();
        self.vertices.iter_mut().for_each(|v| *v -= centroid);
        (self, centroid)
    }
}

/// Vertex index as stored in a face.
fn to_index(idx: usize) -> KeteResult<u32> {
    u32::try_from(idx).map_err(|_| Error::ValueError("Vertex index does not fit in a u32.".into()))
}

/// Pair every directed edge of the faces with its reverse.
///
/// In a closed, consistently wound 2-manifold each edge is traversed exactly once
/// in each direction, by its two faces.
fn build_edges(faces: &[[u32; 3]]) -> KeteResult<Vec<Edge>> {
    let mut directed: HashMap<(u32, u32), u32> = HashMap::with_capacity(3 * faces.len());
    for (idx, face) in faces.iter().enumerate() {
        let face_idx = to_index(idx)?;
        for k in 0..3 {
            let (a, b) = (face[k], face[(k + 1) % 3]);
            if directed.insert((a, b), face_idx).is_some() {
                return Err(Error::ValueError(format!(
                    "Edge ({a}, {b}) is traversed in the same direction by two faces: the \
                     faces are wound inconsistently or the mesh is not a 2-manifold."
                )));
            }
        }
    }
    let mut edges = Vec::with_capacity(directed.len() / 2);
    for (&(a, b), &face_ab) in &directed {
        let Some(&face_ba) = directed.get(&(b, a)) else {
            return Err(Error::ValueError(format!(
                "Edge ({a}, {b}) belongs to only one face: the mesh is not closed."
            )));
        };
        if a < b {
            edges.push(Edge {
                vertices: [a, b],
                faces: [face_ab, face_ba],
            });
        }
    }
    edges.sort_unstable_by_key(|e| e.vertices);
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit cube [0, 1]^3, faces counter-clockwise seen from outside.
    fn cube() -> (Vec<Vector3<f64>>, Vec<[u32; 3]>) {
        let v = (0..8)
            .map(|i| {
                Vector3::new(
                    f64::from(i & 1),
                    f64::from((i >> 1) & 1),
                    f64::from((i >> 2) & 1),
                )
            })
            .collect();
        // two faces per side: z = 0, z = 1, y = 0, y = 1, x = 0, x = 1
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
        (v, f)
    }

    #[test]
    fn cube_properties() {
        let (v, f) = cube();
        let mesh = TriMesh::new(v, f).unwrap();
        assert!((mesh.volume() - 1.0).abs() < 1e-15, "volume");
        assert!((mesh.surface_area() - 6.0).abs() < 1e-14, "area");
        assert!(
            (mesh.centroid() - Vector3::repeat(0.5)).norm() < 1e-15,
            "centroid"
        );
        assert_eq!(mesh.edges().len(), 18, "edge count");
        assert!(
            (mesh.bounding_radius() - 3_f64.sqrt()).abs() < 1e-15,
            "bounding radius"
        );
    }

    #[test]
    fn edge_faces_traverse_opposite_directions() {
        let (v, f) = cube();
        let mesh = TriMesh::new(v, f).unwrap();
        let traverses = |face: &[u32; 3], a: u32, b: u32| {
            (0..3).any(|k| face[k] == a && face[(k + 1) % 3] == b)
        };
        for e in mesh.edges() {
            let [a, b] = e.vertices;
            assert!(a < b, "edge vertices sorted");
            assert!(
                traverses(&mesh.faces()[e.faces[0] as usize], a, b),
                "face 0 goes a -> b"
            );
            assert!(
                traverses(&mesh.faces()[e.faces[1] as usize], b, a),
                "face 1 goes b -> a"
            );
        }
    }

    #[test]
    fn outward_normals() {
        let (verts, faces) = cube();
        let mesh = TriMesh::new(verts, faces).unwrap();
        let center = mesh.centroid();
        for face in mesh.faces() {
            let [p0, p1, p2] = mesh.face_vertices(face);
            let out = (p0 + p1 + p2) / 3.0 - center;
            assert!(mesh.face_normal(face).dot(&out) > 0.0, "normal points out");
        }
    }

    #[test]
    fn centering() {
        let (mut v, f) = cube();
        let offset = Vector3::new(3.0, -2.0, 7.5);
        for p in &mut v {
            *p += offset;
        }
        let (mesh, shift) = TriMesh::new(v, f).unwrap().centered();
        assert!(
            (shift - (offset + Vector3::repeat(0.5))).norm() < 1e-14,
            "reported shift"
        );
        assert!(mesh.centroid().norm() < 1e-14, "centroid at origin");
        assert!((mesh.volume() - 1.0).abs() < 1e-14, "volume unchanged");
    }

    #[test]
    fn translate_rejects_non_finite_offsets() {
        let (v, f) = cube();
        let mut mesh = TriMesh::new(v, f).unwrap();
        assert!(mesh.translate(&Vector3::new(f64::NAN, 0.0, 0.0)).is_err());
        assert!((mesh.centroid() - Vector3::repeat(0.5)).norm() < 1e-15);
        mesh.translate(&Vector3::new(1.0, 0.0, 0.0)).unwrap();
        assert!((mesh.centroid() - Vector3::new(1.5, 0.5, 0.5)).norm() < 1e-15);
    }

    #[test]
    fn rejects_bad_meshes() {
        let (v, f) = cube();
        let err = |v: Vec<Vector3<f64>>, f: Vec<[u32; 3]>| TriMesh::new(v, f).is_err();

        let mut open = f.clone();
        let _ = open.pop();
        assert!(err(v.clone(), open), "open mesh");

        let mut flipped = f.clone();
        flipped[3].swap(1, 2);
        assert!(err(v.clone(), flipped), "one face wound the other way");

        let inward: Vec<_> = f.iter().map(|&[a, b, c]| [a, c, b]).collect();
        assert!(err(v.clone(), inward), "all faces wound inward");

        let mut repeated = f.clone();
        repeated[0] = [0, 0, 1];
        assert!(err(v.clone(), repeated), "repeated index");

        let mut out_of_range = f.clone();
        out_of_range[0][0] = 8;
        assert!(err(v.clone(), out_of_range), "index out of range");

        let mut not_finite = v.clone();
        not_finite[0].x = f64::NAN;
        assert!(err(not_finite, f.clone()), "non-finite vertex");

        let mut collinear = v.clone();
        // vertex 3 moved onto the segment 1-2, so face [1, 2, 3] has no area
        collinear[3] = (collinear[1] + collinear[2]) / 2.0;
        assert!(err(collinear, f.clone()), "degenerate face");

        let mut doubled = f.clone();
        doubled.extend_from_slice(&f);
        assert!(err(v, doubled), "every edge used twice per direction");
    }

    #[test]
    fn ellipsoid() {
        let (ax, ay, az) = (3.0, 2.0, 1.0);
        let exact = 4.0 / 3.0 * PI * ax * ay * az;
        let mut prev_err = f64::INFINITY;
        for n_div in [1, 2, 4, 8, 16] {
            let mesh = TriMesh::new_ellipsoid(n_div, ax, ay, az).unwrap();
            let n = n_div as usize;
            assert_eq!(mesh.faces().len(), 8 * n * n, "face count");
            assert_eq!(mesh.vertices().len(), 4 * n * n + 2, "vertex count");
            assert_eq!(mesh.edges().len(), 12 * n * n, "edge count");
            assert!(mesh.centroid().norm() < 1e-12, "centroid at origin");
            // inscribed, so the volume is below the ellipsoid's and converges to it
            let err = exact - mesh.volume();
            assert!(err > 0.0 && err < prev_err, "volume converges from below");
            prev_err = err;
            for p in mesh.vertices() {
                let r = (p.x / ax).powi(2) + (p.y / ay).powi(2) + (p.z / az).powi(2);
                assert!((r - 1.0).abs() < 1e-12, "vertex on the ellipsoid");
            }
        }
        assert!(prev_err / exact < 0.01, "fine mesh within 1% of the volume");
    }

    #[test]
    fn ellipsoid_rejects_bad_input() {
        assert!(TriMesh::new_ellipsoid(0, 1.0, 1.0, 1.0).is_err(), "n_div 0");
        assert!(
            TriMesh::new_ellipsoid(2, 1.0, 0.0, 1.0).is_err(),
            "zero axis"
        );
        assert!(
            TriMesh::new_ellipsoid(2, 1.0, f64::NAN, 1.0).is_err(),
            "NaN axis"
        );
    }
}
