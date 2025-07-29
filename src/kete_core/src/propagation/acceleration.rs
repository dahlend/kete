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
// Copyright (c) 2025, Dar Dahlen
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

use crate::frames::Equatorial;
use crate::prelude::KeteResult;
use crate::spice::LOADED_SPK;
use crate::{constants, errors::Error, propagation::nongrav::NonGravModel};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix3, OVector, U1, U2, Vector3};
use std::ops::AddAssign;

/// Metadata object used by the [`central_accel`] function below.
#[derive(Debug)]
pub struct CentralAccelMeta {
    /// A vector of times where the central accel function was evaluated at.
    pub times: Vec<f64>,

    /// The position where the central accel function was evaluated.
    pub pos: Vec<Vector3<f64>>,

    /// The velocity where the central accel function was evaluated.
    pub vel: Vec<Vector3<f64>>,

    /// Scaling factor for central mass.
    pub mass_scaling: f64,
}

impl Default for CentralAccelMeta {
    fn default() -> Self {
        Self {
            times: Vec::new(),
            pos: Vec::new(),
            vel: Vec::new(),
            mass_scaling: 1.0,
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
pub fn central_accel(
    time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut CentralAccelMeta,
    exact_eval: bool,
) -> KeteResult<Vector3<f64>> {
    if exact_eval {
        meta.times.push(time);
        meta.pos.push(*pos);
        meta.vel.push(*vel);
    }

    Ok(-pos * pos.norm().powi(-3) * constants::GMS)
}

/// Metadata for the [`spk_accel`] function defined below.
#[derive(Debug)]
pub struct AccelSPKMeta<'a> {
    /// Closest approach to a massive object.
    /// This records the ID of the object, time, and distance in AU.
    pub close_approach: Option<(i32, f64, f64)>,

    /// The non-gravitational forces.
    /// If this is not provided, only standard gravitational model is applied.
    /// If this are provided, then the effects of the Non-Grav terms are added.
    pub non_grav_model: Option<NonGravModel>,

    /// The list of massive objects to apply during SPK computation.
    /// This list contains the ID of the object in the SPK along with the mass and
    /// radius of the object. Mass is given in fractions of solar mass and radius is
    /// in AU.
    pub massive_obj: &'a [constants::GravParams],
}

/// Compute the accel on an object which experiences acceleration due to all massive
/// objects contained within the Spice Kernel SPKs. This uses the planets and the Moon,
/// and applies a General Relativity correction for the Sun and Jupiter.
///
/// Whatever objects are present in the metadata struct will be used as sources of mass.
/// These objects' states are queried from the loaded SPICE kernels, so this will fail
/// if the SPICE kernels are not present/loaded.
///
/// Typically this relies on DE440s, which contains the years ~1800-2200.
///
/// Metadata:
///     Metadata here records the closest approach to any planet. This value is updated
///     every time the function is called.
///
/// # Arguments
///
/// * `time` - Time of the evaluation in JD in TDB scaled time multiplied by `SUN_GMS_SQRT`.
/// * `pos` - A vector which defines the position with respect to the Sun in AU.
/// * `vel` - A vector which defines the velocity with respect to the Sun in AU/Day multiplied by `SUN_GMS_SQRT`.
/// * `meta` - Metadata object [`AccelSPKMeta`] which records values at each integration step.
pub fn spk_accel(
    time: f64,
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    meta: &mut AccelSPKMeta<'_>,
    exact_eval: bool,
) -> KeteResult<Vector3<f64>> {
    let mut accel = Vector3::<f64>::zeros();

    if exact_eval {
        if let Some(close_approach) = meta.close_approach.as_mut() {
            if close_approach.2 == 0.0 {
                *close_approach = (-1, 1000000.0, 0.0);
            }
        }
    }

    let spk = &LOADED_SPK.try_read().unwrap();

    for grav_params in meta.massive_obj {
        let id = grav_params.naif_id;
        let radius = grav_params.radius;
        let state = spk.try_get_state_with_center::<Equatorial>(id, time, 0)?;
        let rel_pos: Vector3<f64> = pos - Vector3::from(state.pos);
        let rel_vel: Vector3<f64> = vel - Vector3::from(state.vel);

        if exact_eval {
            let r = rel_pos.norm();
            if let Some(close_approach) = meta.close_approach.as_mut() {
                if close_approach.2 >= r {
                    *close_approach = (id, r, time);
                }
            }

            if r as f32 <= radius {
                Err(Error::Impact(id, time))?;
            }
        }
        grav_params.add_acceleration(&mut accel, &rel_pos, &rel_vel);

        // If the center is the sun, add non-gravitational forces
        if grav_params.naif_id == 10 {
            if let Some(non_grav) = &meta.non_grav_model {
                non_grav.add_acceleration(&mut accel, &rel_pos, &rel_vel);
            }
        }
    }
    Ok(accel)
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
pub fn vec_accel<D: Dim>(
    time: f64,
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
            if (grav_params.naif_id == 10) & (idx > n_massive) {
                if let Some(non_grav) = &meta.non_gravs[idx - n_massive] {
                    non_grav.add_acceleration(&mut accel_working, &rel_pos, &rel_vel);
                }
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
/// This enables the computation of the STM.
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

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn check_accelerations_equal() {
        let spk = &LOADED_SPK.try_read().unwrap();
        let jd = 2451545.0;
        let mut pos: Vec<f64> = Vec::new();
        let mut vel: Vec<f64> = Vec::new();

        for obj in constants::GravParams::planets() {
            let planet = spk
                .try_get_state_with_center::<Equatorial>(obj.naif_id, jd, 0)
                .unwrap();
            pos.append(&mut planet.pos.into());
            vel.append(&mut planet.vel.into());
        }

        pos.append(&mut [0.0, 0.0, 0.5].into());
        vel.append(&mut [0.0, 0.0, 1.0].into());

        let accel = vec_accel(
            jd,
            &pos.into(),
            &vel.into(),
            &mut AccelVecMeta {
                non_gravs: vec![None],
                massive_obj: &constants::GravParams::planets(),
            },
            false,
        )
        .unwrap()
        .iter()
        .copied()
        .skip(constants::GravParams::planets().len() * 3)
        .collect_vec();

        let accel2 = spk_accel(
            jd,
            &[0.0, 0.0, 0.5].into(),
            &[0.0, 0.0, 1.0].into(),
            &mut AccelSPKMeta {
                close_approach: None,
                non_grav_model: None,
                massive_obj: &constants::GravParams::planets(),
            },
            false,
        )
        .unwrap();
        assert!((accel[0] - accel2[0]).abs() < 1e-10);
        assert!((accel[1] - accel2[1]).abs() < 1e-10);
        assert!((accel[2] - accel2[2]).abs() < 1e-10);
    }
}
