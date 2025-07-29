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

//! Basic Shape Models
use crate::constants::GOLDEN_RATIO;
use nalgebra::{Unit, UnitVector3, Vector3};
use std::f64::consts::TAU;

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
    pub fn new_fibonacci_lattice(n_facets: usize) -> Self {
        let mut facets: Vec<Facet> = Vec::with_capacity(n_facets);

        const EPSILON: f64 = 0.36;

        let n_normals = n_facets as f64;
        let area = n_normals.recip();

        for idx in 0..n_facets {
            let theta: f64 = TAU * (idx as f64) / GOLDEN_RATIO;
            let phi: f64 =
                (1.0 - 2.0 * ((idx as f64) + EPSILON) / (n_normals - 1.0 + 2.0 * EPSILON)).acos();
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_convex_shape() {
        let n1024 = ConvexShape::new_fibonacci_lattice(1024);

        assert!(n1024.facets.len() == 1024);
        assert!(n1024.facets.iter().all(|x| x.area == (1024_f64).recip()));
    }
}
