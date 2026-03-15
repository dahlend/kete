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

//! Basic Shape Models
use kete_core::constants::GOLDEN_RATIO;
use nalgebra::{Unit, UnitVector3, Vector3};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Pre-compute a default shape.
pub static DEFAULT_SHAPE: std::sync::LazyLock<ConvexShape> =
    std::sync::LazyLock::new(|| ConvexShape::new_fibonacci_lattice(2048));

/// Facet of a shape.
#[derive(Debug, Clone)]
pub struct Facet {
    /// Normal unit vector defining the facets face
    pub normal: UnitVector3<f64>,

    /// Surface area of the facet
    pub area: f64,
}

/// Convex shape made up of individual facets.
#[derive(Debug, Clone)]
pub struct ConvexShape {
    /// The facets which make up this shape.
    pub facets: Box<[Facet]>,
}

impl ConvexShape {
    /// Construct a new [`ConvexShape`] using fibonacci lattice spacing.
    ///
    /// Evenly place points on a sphere using the Fibonacci Lattice algorithm.
    ///
    /// This uses a slightly modified method where an epsilon term is added to shift the
    /// points slightly, causing the average spacing between the points to be minimized.
    ///
    /// See:
    /// <http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/>
    ///
    ///
    /// Total surface area is set to 1.
    #[must_use]
    pub fn new_fibonacci_lattice(n_facets: u32) -> Self {
        const EPSILON: f64 = 0.36;

        let mut facets: Vec<Facet> = Vec::with_capacity(n_facets as usize);

        let n_normals = f64::from(n_facets);
        let area = n_normals.recip();

        for idx in 0..n_facets {
            let theta: f64 = TAU * f64::from(idx) / GOLDEN_RATIO;
            let phi: f64 =
                (1.0 - 2.0 * (f64::from(idx) + EPSILON) / (n_normals - 1.0 + 2.0 * EPSILON)).acos();
            let normal = Unit::new_unchecked(Vector3::new(
                theta.cos() * phi.sin(),
                theta.sin() * phi.sin(),
                phi.cos(),
            ));

            facets.push(Facet { normal, area });
        }

        Self {
            facets: facets.into(),
        }
    }

    /// Rescale the total areas to sum to 1.
    pub fn normalize_areas(&mut self) {
        let total_area_inv = self.facets.iter().map(|f| f.area).sum::<f64>().recip();
        self.facets
            .iter_mut()
            .for_each(|x| x.area *= total_area_inv);
    }
}

/// A single triangle facet defined by three vertices, used for visualization.
#[derive(Debug, Clone)]
pub struct TriangleFacet {
    /// The three vertices defining this triangle.
    pub vertices: [Vector3<f64>; 3],

    /// Normal unit vector of the triangle surface, oriented away from the origin.
    pub normal: UnitVector3<f64>,

    /// Surface area of the triangle.
    pub area: f64,
}

impl TriangleFacet {
    /// Create a triangle facet from three vertices.
    ///
    /// The normal is computed via the cross product and oriented to face away from
    /// the origin.
    #[must_use]
    pub fn new(v1: Vector3<f64>, v2: Vector3<f64>, v3: Vector3<f64>) -> Self {
        let cross = (v2 - v1).cross(&(v3 - v1));
        let area = cross.norm() / 2.0;
        let mut normal_vec = cross;
        if v1.dot(&normal_vec) < 0.0 {
            normal_vec = -normal_vec;
        }
        let normal = Unit::new_normalize(normal_vec);
        Self {
            vertices: [v1, v2, v3],
            normal,
            area,
        }
    }
}

/// A shape made up of triangle facets, suitable for visualization.
///
/// This stores the full triangle vertex information needed for 3D rendering,
/// along with precomputed normals and areas.
#[derive(Debug, Clone)]
pub struct TriangleShape {
    /// The triangle facets making up this shape.
    pub facets: Box<[TriangleFacet]>,
}

impl TriangleShape {
    /// Construct a triangle-faceted ellipsoid (or sphere when all scales are equal).
    ///
    /// Uses the algorithm described in <https://arxiv.org/abs/1502.04816> to generate
    /// a nearly-uniform triangulation of a sphere, then applies axis scaling.
    ///
    /// Total surface area is normalized to 1.
    ///
    /// # Arguments
    ///
    /// * `n_div` - Number of divisions from pole to equator. Total facet count is
    ///   `8 * n_div * n_div`. Must be at least 1.
    /// * `x_scale` - Scale factor along the x-axis.
    /// * `y_scale` - Scale factor along the y-axis.
    /// * `z_scale` - Scale factor along the z-axis.
    ///
    /// # Panics
    ///
    /// Panics if `n_div` is 0.
    #[must_use]
    pub fn new_ellipsoid(n_div: u32, x_scale: f64, y_scale: f64, z_scale: f64) -> Self {
        assert!(n_div >= 1, "n_div must be at least 1");
        let n = n_div as usize;
        let dth = FRAC_PI_2 / n as f64;

        // Generate points on the unit sphere from (colatitude, longitude) -> Cartesian.
        let mut points: Vec<Vector3<f64>> = Vec::new();

        // North pole
        points.push(Vector3::new(0.0, 0.0, 1.0));

        // Northern hemisphere rings
        for i in 1..=n {
            let dphi = FRAC_PI_2 / i as f64;
            for j in 0..4 * i {
                let theta = i as f64 * dth;
                let phi = j as f64 * dphi;
                points.push(Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ));
            }
        }

        // Southern hemisphere rings
        for i in (1..n).rev() {
            let dphi = FRAC_PI_2 / i as f64;
            for j in 0..4 * i {
                let theta = PI - i as f64 * dth;
                let phi = j as f64 * dphi;
                points.push(Vector3::new(
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ));
            }
        }

        // South pole
        points.push(Vector3::new(0.0, 0.0, -1.0));

        // Apply axis scaling before computing normals/areas.
        for p in &mut points {
            p.x *= x_scale;
            p.y *= y_scale;
            p.z *= z_scale;
        }

        // Build the vertex-index matrix (variable row lengths).
        // Row i has 4*k + 1 elements where k mirrors around the equator.
        let mut matrix: Vec<Vec<usize>> = Vec::with_capacity(2 * n + 1);
        for i in 0..=n {
            matrix.push(vec![0; 4 * i + 1]);
        }
        for i in (0..n).rev() {
            matrix.push(vec![0; 4 * i + 1]);
        }

        // Fill matrix entries: map each ring position to its point index.
        let mut count: usize = 0;

        for i in 1..=n {
            for j in 0..4 * i {
                count += 1;
                matrix[i][j] = count;
                if j == 0 {
                    // wrap-around
                    matrix[i][4 * i] = count;
                }
            }
        }

        for i in (1..n).rev() {
            for j in 0..4 * i {
                count += 1;
                matrix[2 * n - i][j] = count;
                if j == 0 {
                    matrix[2 * n - i][4 * i] = count;
                }
            }
        }
        // south pole
        matrix[2 * n][0] = count + 1;

        // Generate triangle facets by connecting adjacent rings.
        let expected = 8 * n * n;
        let mut facets: Vec<TriangleFacet> = Vec::with_capacity(expected);

        // Northern hemisphere
        for j1 in 1..=n {
            for j3 in 1..=4_usize {
                let j0 = (j3 - 1) * j1;
                facets.push(TriangleFacet::new(
                    points[matrix[j1 - 1][j0 - (j3 - 1)]],
                    points[matrix[j1][j0]],
                    points[matrix[j1][j0 + 1]],
                ));

                for j2 in (j0 + 1)..(j0 + j1) {
                    facets.push(TriangleFacet::new(
                        points[matrix[j1][j2]],
                        points[matrix[j1 - 1][j2 - (j3 - 1)]],
                        points[matrix[j1 - 1][(j2 - 1) - (j3 - 1)]],
                    ));
                    facets.push(TriangleFacet::new(
                        points[matrix[j1][j2]],
                        points[matrix[j1 - 1][j2 - (j3 - 1)]],
                        points[matrix[j1][j2 + 1]],
                    ));
                }
            }
        }

        // Southern hemisphere
        for j1 in (n + 1)..=(2 * n) {
            for j3 in 1..=4_usize {
                let j0 = (j3 - 1) * (2 * n - j1);
                facets.push(TriangleFacet::new(
                    points[matrix[j1][j0]],
                    points[matrix[j1 - 1][(j0 + 1) + (j3 - 1)]],
                    points[matrix[j1 - 1][j0 + (j3 - 1)]],
                ));

                for j2 in (j0 + 1)..=(j0 + 2 * n - j1) {
                    facets.push(TriangleFacet::new(
                        points[matrix[j1][j2]],
                        points[matrix[j1 - 1][j2 + (j3 - 1)]],
                        points[matrix[j1][j2 - 1]],
                    ));
                    facets.push(TriangleFacet::new(
                        points[matrix[j1][j2]],
                        points[matrix[j1 - 1][(j2 + 1) + (j3 - 1)]],
                        points[matrix[j1 - 1][j2 + (j3 - 1)]],
                    ));
                }
            }
        }

        debug_assert_eq!(facets.len(), expected, "facet count mismatch");

        // Normalize total area to 1.
        let total_area_inv = facets.iter().map(|f| f.area).sum::<f64>().recip();
        for facet in &mut facets {
            facet.area *= total_area_inv;
        }

        Self {
            facets: facets.into(),
        }
    }

    /// Total number of facets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.facets.len()
    }

    /// Returns true if there are no facets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.facets.is_empty()
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_convex_shape() {
        let n1024 = ConvexShape::new_fibonacci_lattice(1024);

        assert!(n1024.facets.len() == 1024);
        assert!(n1024.facets.iter().all(|x| x.area == (1024_f64).recip()));
    }

    #[test]
    fn test_triangle_shape_sphere() {
        let shape = TriangleShape::new_ellipsoid(6, 1.0, 1.0, 1.0);
        // 8 * n_div^2
        assert_eq!(shape.len(), 8 * 36);

        let total_area: f64 = shape.facets.iter().map(|f| f.area).sum();
        assert!((total_area - 1.0).abs() < 1e-10);

        // All normals should point outward for a convex shape around the origin.
        for facet in shape.facets {
            let centroid = (facet.vertices[0] + facet.vertices[1] + facet.vertices[2]) / 3.0;
            assert!(centroid.dot(&facet.normal) > 0.0);
        }
    }

    #[test]
    fn test_triangle_shape_ellipsoid() {
        let shape = TriangleShape::new_ellipsoid(4, 2.0, 1.0, 0.5);
        // 8 * 4^2
        assert_eq!(shape.len(), 8 * 16);

        let total_area: f64 = shape.facets.iter().map(|f| f.area).sum();
        assert!((total_area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_shape_n_div_1() {
        // Minimal case: 8 facets (octahedron)
        let shape = TriangleShape::new_ellipsoid(1, 1.0, 1.0, 1.0);
        assert_eq!(shape.len(), 8);
    }
}
