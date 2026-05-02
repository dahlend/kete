//! # Extended Gravity Models
//!
//! Gravity sources that are not point masses.  The current implementation
//! provides a constant-density polyhedron model (Werner & Scheeres 1996),
//! suitable for irregular small bodies such as asteroids and comet nuclei.
//!
//! All evaluation here is performed in a **body-fixed frame** in whatever
//! length units the caller supplied to the constructor.  Frame rotation,
//! body-natural unit conversion, and integration with kete's force machinery
//! are handled in later phases by separate modules.
//!
//! Multiple polyhedron components can be combined to model contact binaries
//! or bilobate bodies with per-component density by summing the outputs of
//! several [`Polyhedron`] instances.
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

mod extended_body;
mod io;
mod polyhedron;
mod rotation;
mod units;

pub use extended_body::{ExtendedBody, small_body_accel};
pub use io::load_obj;
pub use polyhedron::Polyhedron;
pub use rotation::RotationModel;
pub use units::BodyUnits;

use crate::errors::KeteResult;
use nalgebra::{Matrix3, Vector3};

/// A gravitational source that is not a point mass.
///
/// Implementors evaluate the gravitational potential, acceleration, and
/// gravity gradient (Hessian of the potential) at a field point expressed
/// in the source's body-fixed frame.
///
/// Sign conventions follow Werner & Scheeres 1996: the potential `U` is
/// defined such that `U -> GM/r` in the far field, and gravitational
/// acceleration on a test particle is `a = grad U` (the vector points
/// toward the attracting mass).
///
/// All evaluation methods return [`KeteResult`].  Implementations may
/// return [`crate::errors::Error::SurfaceImpact`] when the field point
/// coincides with the body surface; propagators should catch this and
/// translate it to a propagation-level [`crate::errors::Error::Impact`].
pub trait ExtendedGravity {
    /// Gravitational potential at body-fixed position `r`.
    ///
    /// Far from the body this approaches `GM/|r|`.
    ///
    /// # Errors
    /// Returns [`crate::errors::Error::SurfaceImpact`] if `r` coincides
    /// with the body surface.
    fn potential(&self, r: Vector3<f64>) -> KeteResult<f64>;

    /// Gravitational acceleration at body-fixed position `r`.
    ///
    /// The returned vector is the acceleration experienced by a test
    /// particle, pointing toward the attracting mass.
    ///
    /// # Errors
    /// Returns [`crate::errors::Error::SurfaceImpact`] if `r` coincides
    /// with the body surface.
    fn acceleration(&self, r: Vector3<f64>) -> KeteResult<Vector3<f64>>;

    /// Gravity gradient tensor `grad grad U` at body-fixed position `r`.
    ///
    /// The trace of this matrix equals `-4 pi G rho` inside the body
    /// and `0` outside (Poisson's equation).
    ///
    /// # Errors
    /// Returns [`crate::errors::Error::SurfaceImpact`] if `r` coincides
    /// with the body surface.
    fn gradient(&self, r: Vector3<f64>) -> KeteResult<Matrix3<f64>>;

    /// Smallest radius (in the same length units as the constructor) at
    /// which the model is considered well-defined.
    ///
    /// Polyhedron models return `0` because the closed-form Werner-Scheeres
    /// expressions are valid everywhere outside, on, and inside the body.
    /// Spherical-harmonic models return the Brillouin sphere radius.
    fn min_valid_radius(&self) -> f64;

    /// True if `r` lies inside the body.
    ///
    /// Default implementation returns `false`; types that can answer this
    /// (such as polyhedra, via the Laplacian sign test) should override.
    fn contains(&self, _r: Vector3<f64>) -> bool {
        false
    }
}
