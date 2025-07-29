//! Inertial and Non-Inertial coordinate frames.
//!
//! The Equatorial frame is considered the base inertial frame, and all other frames
//! provide conversions to and from this frame. If you have a choice of frame,
//! it is recommended to use the Equatorial frame as a result.
//!
//! Equatorial is the fundamental frame as it is what is used in the DE440 ephemeris
//! file. This file is the primary limiting factor for speed when computing orbital
//! integration, so any reduction in friction in reading those states improves
//! performance.
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

use crate::errors::{Error, KeteResult};
use crate::spice::{CkArray, LOADED_CK};
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, Rotation3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt::Debug;

use super::earth::OBLIQUITY;
use super::euler_rotation;

/// Frame which supports vector conversion
pub trait InertialFrame: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// Convert a vector from input frame to equatorial frame.
    #[inline(always)]
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().transform_vector(&vec)
    }

    /// Convert a vector from the equatorial frame to this frame.
    #[inline(always)]
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().inverse_transform_vector(&vec)
    }

    /// Rotation matrix from the inertial frame to the equatorial frame.
    fn rotation_to_equatorial() -> &'static Rotation3<f64>;

    /// Convert between frames.
    #[inline(always)]
    fn convert<Target: InertialFrame>(vec: Vector3<f64>) -> Vector3<f64> {
        Target::from_equatorial(Self::to_equatorial(vec))
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Equatorial {}

impl InertialFrame for Equatorial {
    #[inline(always)]
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        // equatorial is a special case, so we can skip the rotation
        // and just return the vector as is.
        vec
    }

    #[inline(always)]
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        // equatorial is a special case, so we can skip the rotation
        // and just return the vector as is.
        vec
    }

    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &IDENTITY_ROT
    }
}

/// Ecliptic frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ecliptic {}

impl InertialFrame for Ecliptic {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &ECLIPTIC_EQUATORIAL_ROT
    }
}

/// Galactic frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Galactic {}

impl InertialFrame for Galactic {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &GALACTIC_EQUATORIAL_ROT
    }
}

/// FK4 frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FK4 {}

impl InertialFrame for FK4 {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &FK4_EQUATORIAL_ROT
    }
}

/// General representation of a non-inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonInertialFrame {
    /// Time of the frame in TDB.
    pub time: Time<TDB>,

    /// Rotation matrix from this frame to the reference frame.
    pub rotation: Rotation3<f64>,

    /// Rotation rate of this frame, if not defined, this is assumed to be zero.
    pub rotation_rate: Option<Matrix3<f64>>,

    /// The frame that this frame is defined relative to.
    pub reference_frame_id: i32,

    /// The frame ID of this frame.
    pub frame_id: i32,
}

impl NonInertialFrame {
    /// Create a new non-inertial frame from the provided rotation and rotation rate.
    pub fn from_euler<const E1: char, const E2: char, const E3: char>(
        time: Time<TDB>,
        angles: [f64; 3],
        rates: [f64; 3],
        reference_frame_id: i32,
        frame_id: i32,
    ) -> Self {
        let (rot_p, rot_dp) = euler_rotation::<E1, E2, E3>(&angles, &rates);
        Self {
            time,
            rotation: rot_p,
            rotation_rate: Some(rot_dp),
            reference_frame_id,
            frame_id,
        }
    }

    /// Create non-inertial from from rotations
    pub fn from_rotations(
        time: Time<TDB>,
        rotation: Rotation3<f64>,
        rotation_rate: Option<Matrix3<f64>>,
        reference_frame_id: i32,
        frame_id: i32,
    ) -> Self {
        Self {
            time,
            rotation,
            rotation_rate,
            reference_frame_id,
            frame_id,
        }
    }

    /// Return the rotation matrix and rotation rate for this frame in the equatorial frame.
    pub fn rotations_to_equatorial(&self) -> KeteResult<(Rotation3<f64>, Matrix3<f64>)> {
        if self.reference_frame_id == 1 {
            // Equatorial frame
            Ok((
                self.rotation,
                self.rotation_rate.unwrap_or_else(Matrix3::identity),
            ))
        } else if self.reference_frame_id == 17 {
            // Ecliptic frame
            let rot = self.rotation;
            let dt_rot = self.rotation_rate.unwrap_or_else(Matrix3::identity);
            Ok((
                *ECLIPTIC_EQUATORIAL_ROT * rot,
                *ECLIPTIC_EQUATORIAL_ROT * dt_rot,
            ))
        } else if self.reference_frame_id < 0 {
            let cks = LOADED_CK.read().unwrap();

            // find the segment in the ck data which matches the reference frame id, with its own reference frame id being either 1 or 17.
            for segment in cks.segments.iter() {
                let array: &CkArray = segment.into();
                if array.instrument_id == self.reference_frame_id {
                    let frame = segment.try_get_orientation(self.reference_frame_id, self.time);
                    if frame.is_err() {
                        continue;
                    }
                    let (time, frame) = frame.unwrap();
                    if (time.jd - self.time.jd).abs() > 1e-8 {
                        continue; // time mismatch, skip this frame
                    }
                    let (rot, vel) = frame.rotations_to_equatorial()?;
                    return Ok((
                        rot * self.rotation,
                        vel * self.rotation_rate.unwrap_or_else(Matrix3::identity),
                    ));
                }
            }
            Err(Error::DAFLimits(format!(
                "Reference frame ID {} not found in CK data.",
                self.reference_frame_id
            )))
        } else {
            // Unsupported frame
            Err(Error::DAFLimits(format!(
                "Reference frame ID {} is not supported.",
                self.reference_frame_id
            )))
        }
    }

    /// Convert a vector from the equatorial frame to this frame.
    #[allow(
        clippy::wrong_self_convention,
        reason = "Always need position and velocity together"
    )]
    pub fn from_equatorial(
        &self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let (rot_p, rot_dp) = self.rotations_to_equatorial()?;

        let new_pos = rot_p.inverse_transform_vector(&pos);
        let new_vel = rot_dp.transpose() * pos + rot_p.inverse_transform_vector(&vel);

        Ok((new_pos, new_vel))
    }

    /// Convert a vector from input frame to equatorial frame.
    pub fn to_equatorial(
        &self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let (rot_p, rot_dp) = self.rotations_to_equatorial()?;

        let new_pos = rot_p.transform_vector(&pos);
        let new_vel = rot_dp * pos + rot_p.transform_vector(&vel);
        Ok((new_pos, new_vel))
    }
}

static IDENTITY_ROT: std::sync::LazyLock<Rotation3<f64>> =
    std::sync::LazyLock::new(Rotation3::identity);

static ECLIPTIC_EQUATORIAL_ROT: std::sync::LazyLock<Rotation3<f64>> =
    std::sync::LazyLock::new(|| {
        let x = nalgebra::Unit::new_unchecked(Vector3::x_axis());
        Rotation3::from_axis_angle(&x, OBLIQUITY)
    });

static FK4_EQUATORIAL_ROT: std::sync::LazyLock<Rotation3<f64>> = std::sync::LazyLock::new(|| {
    let y = nalgebra::Unit::new_unchecked(Vector3::y_axis());
    let z = nalgebra::Unit::new_unchecked(Vector3::z_axis());
    let r1 = Rotation3::from_axis_angle(&z, (1152.84248596724 + 0.525) / 3600.0 * PI / 180.0);
    let r2 = Rotation3::from_axis_angle(&y, -1002.26108439117 / 3600.0 * PI / 180.0);
    let r3 = Rotation3::from_axis_angle(&z, 1153.04066200330 / 3600.0 * PI / 180.0);
    r3 * r2 * r1
});

static GALACTIC_EQUATORIAL_ROT: std::sync::LazyLock<Rotation3<f64>> =
    std::sync::LazyLock::new(|| {
        let x = nalgebra::Unit::new_unchecked(Vector3::x_axis());
        let z = nalgebra::Unit::new_unchecked(Vector3::z_axis());
        let r1 = Rotation3::from_axis_angle(&z, 1177200.0 / 3600.0 * PI / 180.0);
        let r2 = Rotation3::from_axis_angle(&x, 225360.0 / 3600.0 * PI / 180.0);
        let r3 = Rotation3::from_axis_angle(&z, 1016100.0 / 3600.0 * PI / 180.0);
        (*FK4_EQUATORIAL_ROT) * r3 * r2 * r1
    });

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ecliptic_rot_roundtrip() {
        let vec = Ecliptic::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = Ecliptic::from_equatorial(vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_fk4_roundtrip() {
        let vec = FK4::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = FK4::from_equatorial(vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_galactic_rot_roundtrip() {
        let vec = Galactic::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = Galactic::from_equatorial(vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }

    #[test]
    fn test_noninertial_rot_roundtrip() {
        let angles = [0.11, 0.21, 0.31];
        let rates = [0.41, 0.51, 0.61];
        let pos = [1.0, 2.0, 3.0].into();
        let vel = [0.1, 0.2, 0.3].into();
        let frame =
            NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(0_f64.into(), angles, rates, 17, 100);
        let (r_pos, r_vel) = frame.to_equatorial(pos, vel).unwrap();
        let (pos_return, vel_return) = frame.from_equatorial(r_pos, r_vel).unwrap();

        assert!((1.0 - pos_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - pos_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - pos_return[2]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.1 - vel_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.2 - vel_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.3 - vel_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
}
