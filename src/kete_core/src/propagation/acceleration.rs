//! Compute the acceleration of test particles with regard to massive objects.
//! This is used by the propagation code along with the numerical integrator to
//! calculated orbital dynamics.
//!
//! There are several functions defined here, which enable various levels of accuracy.
//!
//! These functions have a strict function signature, which is defined inside of the
//! radau integrator class. This function signature contains 4 terms:
//!
//! `(time, x, x_der, &mut MetaData, exact_eval) -> NeosResult<x_der_der>`
//!
//! Where `x` and its derivative `x_der` are vectors. This also accepts a mutable
//! reference to a metadata collection. Metadata may include things like object
//! specific orbit parameters such as the non-grav terms, or keep track of close
//!
//! `exact_eval` is a bool which is passed when the integrator is passing values where
//! the `x` and `x_der` are being evaluated at true locations. IE: where the integrator
//! thinks that the object should actually be. These times are when close encounter
//! information should be recorded.
//!
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

use crate::prelude::KeteResult;
use crate::time::{TDB, Time};
use crate::{constants, errors::Error, propagation::nongrav::NonGravModel};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix3, OVector, U1, U2, Vector3};
use std::ops::AddAssign;

/// Metadata object used by the [`central_accel`] function below.
#[derive(Debug, Clone)]
pub struct CentralAccelMeta {
    /// A vector of times where the central accel function was evaluated at.
    pub times: Vec<Time<TDB>>,

    /// The position where the central accel function was evaluated.
    pub pos: Vec<Vector3<f64>>,

    /// The velocity where the central accel function was evaluated.
    pub vel: Vec<Vector3<f64>>,

    /// Scaling factor for central mass.
    pub mass_scaling: f64,

    /// Total number of function evaluations.
    pub eval_count: usize,
}

impl Default for CentralAccelMeta {
    fn default() -> Self {
        Self {
            times: Vec::new(),
            pos: Vec::new(),
            vel: Vec::new(),
            mass_scaling: 1.0,
            eval_count: 0,
        }
    }
}

/// Compute the accel on an object which experiences acceleration due to the Sun only.
/// Integrating this with Radau should result in the same values as two-body analytic
/// integration.
///
/// # Arguments
///
/// * `time` - Time of the evaluation. This is saved in the metadata but otherwise
///   unused.
/// * `pos` - A vector which defines the position with respect to the Sun in AU.
/// * `vel` - A vector which defines the velocity with respect to the Sun in AU/Day.
/// * `meta` - Metadata object which records values at integration steps.
///
/// # Errors
/// This is actually infallible, but must have this signature for the integrator.
pub fn central_accel(
    time: Time<TDB>,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut CentralAccelMeta,
    exact_eval: bool,
) -> KeteResult<Vector3<f64>> {
    meta.eval_count += 1;
    if exact_eval {
        meta.times.push(time);
        meta.pos.push(*pos);
        meta.vel.push(*vel);
    }

    Ok(-pos * pos.norm().powi(-3) * constants::GMS)
}

/// Metadata for the [`vec_accel`] function defined below.
#[derive(Debug)]
pub struct AccelVecMeta<'a> {
    /// The non-gravitational forces.
    /// If this is not provided, only standard gravitational model is applied.
    /// If these values are provided, then the effects of the Non-Grav terms are added.
    pub non_gravs: Vec<Option<NonGravModel>>,

    /// The list of massive objects to apply during SPK computation.
    /// This list contains the ID of the object in the SPK along with the mass and
    /// radius of the object. Mass is given in fractions of solar mass and radius is
    /// in AU.
    pub massive_obj: &'a [constants::GravParams],
}

/// Compute the accel on an object which experiences acceleration due to all massive
/// objects. This assumes that the first N objects match the objects in the metadata
/// list in order. IE: if `MASSIVE_OBJECTS` from the constants file is used in the meta
/// data, then those objects are assumed to be in the same order in the pos/vel vectors
/// provided.
///
/// # Arguments
///
/// * `time` - Time is not used in this.
/// * `pos` - A vector which defines the position with respect to the Sun in AU.
/// * `vel` - A vector which defines the velocity with respect to the Sun in AU/Day.
/// * `meta` - Metadata.
///
/// # Errors
/// Fails during an impact.
pub fn vec_accel<D: Dim>(
    time: Time<TDB>,
    pos: &OVector<f64, D>,
    vel: &OVector<f64, D>,
    meta: &mut AccelVecMeta<'_>,
    exact_eval: bool,
) -> KeteResult<OVector<f64, D>>
where
    DefaultAllocator: Allocator<D> + Allocator<D, U2>,
{
    // objects in the pos/vel vectors are setup like so
    // (x, y, z, x, y, z, ...

    let n_objects = pos.len() / 3;
    let n_massive = meta.massive_obj.len();

    let (dim, _) = pos.shape_generic();
    let mut accel = OVector::<f64, D>::zeros_generic(dim, U1);

    let mut accel_working = Vector3::zeros();

    for idx in 0..n_objects {
        let pos_idx = pos.fixed_rows::<3>(idx * 3);
        let vel_idx = vel.fixed_rows::<3>(idx * 3);

        for (idy, grav_params) in meta.massive_obj.iter().enumerate() {
            if idx == idy {
                continue;
            }
            accel_working.fill(0.0);
            let radius = grav_params.radius;
            let pos_idy = pos.fixed_rows::<3>(idy * 3);
            let vel_idy = vel.fixed_rows::<3>(idy * 3);

            let rel_pos = pos_idx - pos_idy;
            let rel_vel = vel_idx - vel_idy;

            if exact_eval & (rel_pos.norm() as f32 <= radius) {
                Err(Error::Impact(grav_params.naif_id, time))?;
            }
            grav_params.add_acceleration(&mut accel_working, &rel_pos, &rel_vel);

            // If the center is the sun, add non-gravitational forces
            if (grav_params.naif_id == 10)
                && (idx > n_massive)
                && let Some(non_grav) = &meta.non_gravs[idx - n_massive]
            {
                non_grav.add_acceleration(&mut accel_working, &rel_pos, &rel_vel);
            }

            accel.fixed_rows_mut::<3>(idx * 3).add_assign(accel_working);
        }
    }

    Ok(accel)
}

/// Calculate the Jacobian for the [`central_accel`] function.
///
/// This enables the computation of the STM.
pub fn central_accel_grad(
    _time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut CentralAccelMeta,
) -> Matrix3<f64> {
    let zeros = Vector3::<f64>::zeros();
    accel_grad(pos, vel, &zeros, &zeros, meta.mass_scaling)
}

/// Calculate the Jacobian for the [`central_accel`] function.
///
/// This enables the computation of the two body STM.
#[must_use]
pub fn accel_grad(
    obj_pos: &Vector3<f64>,
    _obj_vel: &Vector3<f64>,
    mass_pos: &Vector3<f64>,
    _mass_vel: &Vector3<f64>,
    mass: f64,
) -> Matrix3<f64> {
    let pos = obj_pos - mass_pos;
    let r = pos.norm();
    let r_2 = r.powi(2);
    let r_5_inv = r.powi(5);
    Matrix3::<f64>::new(
        r_2 - 3.0 * pos.x.powi(2),
        -3.0 * pos.x * pos.y,
        -3.0 * pos.x * pos.z,
        -3.0 * pos.x * pos.y,
        r_2 - 3.0 * pos.y.powi(2),
        -3.0 * pos.y * pos.z,
        -3.0 * pos.x * pos.z,
        -3.0 * pos.y * pos.z,
        r_2 - 3.0 * pos.z.powi(2),
    ) / (2.0 * r_5_inv)
        * mass
}
