// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

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

use crate::errors::{Error, KeteResult};
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, Rotation3, Vector3};
use std::f64::consts::PI;
use std::fmt::Debug;

use super::earth::OBLIQUITY;
use super::euler_rotation;

/// Frame which supports vector conversion
pub trait InertialFrame: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// Convert a vector from input frame to equatorial frame.
    #[inline(always)]
    #[must_use]
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().transform_vector(&vec)
    }

    /// Convert a vector from the equatorial frame to this frame.
    #[inline(always)]
    #[must_use]
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        Self::rotation_to_equatorial().inverse_transform_vector(&vec)
    }

    /// Rotation matrix from the inertial frame to the equatorial frame.
    fn rotation_to_equatorial() -> &'static Rotation3<f64>;

    /// Convert between frames.
    #[inline(always)]
    #[must_use]
    fn convert<Target: InertialFrame>(vec: Vector3<f64>) -> Vector3<f64> {
        Target::from_equatorial(Self::to_equatorial(vec))
    }

    /// Rotation matrix from this frame to another inertial frame.
    ///
    /// [`Self::convert`] applies this to a single vector. The matrix itself is what is
    /// needed to rotate a Jacobian or a covariance, where the same rotation acts on every
    /// column rather than on one vector.
    #[inline(always)]
    #[must_use]
    fn rotation_to_frame<Target: InertialFrame>() -> Rotation3<f64> {
        Target::rotation_to_equatorial().inverse() * *Self::rotation_to_equatorial()
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecliptic {}

impl InertialFrame for Ecliptic {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &ECLIPTIC_EQUATORIAL_ROT
    }
}

/// Galactic frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Galactic {}

impl InertialFrame for Galactic {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &GALACTIC_EQUATORIAL_ROT
    }
}

/// FK4 frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FK4 {}

impl InertialFrame for FK4 {
    #[inline(always)]
    fn rotation_to_equatorial() -> &'static Rotation3<f64> {
        &FK4_EQUATORIAL_ROT
    }
}

/// SPICE frame ID of a reference frame, such as [`Self::J2000`].
///
/// An [`Ephemeris`](crate::ephemeris::Ephemeris) resolves a body frame from its
/// ID. The inertial frames below convert to equatorial directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameId(pub i32);

impl FrameId {
    /// The SPICE J2000 frame, the equatorial frame.
    pub const J2000: Self = Self(1);

    /// The SPICE FK4 frame.
    pub const FK4: Self = Self(3);

    /// The SPICE GALACTIC frame.
    pub const GALACTIC: Self = Self(13);

    /// The SPICE ECLIPJ2000 frame, the ecliptic frame.
    pub const ECLIPJ2000: Self = Self(17);
}

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// General representation of a non-inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct NonInertialFrame {
    /// Time of the frame in TDB.
    pub time: Time<TDB>,

    /// Rotation matrix from this frame to the reference frame.
    pub rotation: Rotation3<f64>,

    /// Time derivative of `rotation`, per day.
    ///
    /// It is `None` when the source of the frame holds no rate, such as a CK
    /// segment without angular velocity. A frame without a rate gives its
    /// rotation, but not a velocity transformation.
    pub rotation_rate: Option<Matrix3<f64>>,

    /// The frame that this frame is defined relative to.
    pub reference_frame_id: FrameId,
}

impl NonInertialFrame {
    /// Create a new non-inertial frame from the provided rotation and rotation rate.
    pub fn from_euler<const E1: char, const E2: char, const E3: char>(
        time: impl Into<Time<TDB>>,
        angles: [f64; 3],
        rates: [f64; 3],
        reference_frame_id: FrameId,
    ) -> Self {
        let (rot_p, rot_dp) = euler_rotation::<E1, E2, E3>(&angles, &rates);
        Self {
            time: time.into(),
            rotation: rot_p,
            rotation_rate: Some(rot_dp),
            reference_frame_id,
        }
    }

    /// Create non-inertial from from rotations
    ///
    /// # Arguments
    /// * `time` - Time of the frame in TDB.
    /// * `rotation` - Rotation matrix from this frame to the reference frame.
    /// * `rotation_rate` - Time derivative of `rotation` per day, or `None` if unknown.
    /// * `reference_frame_id` - The frame that this frame is defined relative to.
    pub fn from_rotations(
        time: impl Into<Time<TDB>>,
        rotation: Rotation3<f64>,
        rotation_rate: Option<Matrix3<f64>>,
        reference_frame_id: FrameId,
    ) -> Self {
        Self {
            time: time.into(),
            rotation,
            rotation_rate,
            reference_frame_id,
        }
    }

    /// The rotation from this frame to the equatorial frame.
    ///
    /// The reference frame must be one of the SPICE inertial frames J2000 (1),
    /// FK4 (3), GALACTIC (13) or ECLIPJ2000 (17). The provider of a frame
    /// relative to another body frame resolves it, see
    /// [`Ephemeris::try_frame`](crate::ephemeris::Ephemeris::try_frame).
    ///
    /// # Errors
    /// [`Error::Bounds`] if the reference frame is not one of those four.
    pub fn rotation_to_equatorial(&self) -> KeteResult<Rotation3<f64>> {
        Ok(self.reference_to_equatorial()? * self.rotation)
    }

    /// The rotation from this frame to the equatorial frame, and its time
    /// derivative per day.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if the reference frame is not supported, as for
    ///   [`Self::rotation_to_equatorial`].
    /// - [`Error::ValueError`] if the frame has no rotation rate.
    pub fn rotations_to_equatorial(&self) -> KeteResult<(Rotation3<f64>, Matrix3<f64>)> {
        let to_equatorial = self.reference_to_equatorial()?;
        let rate = self.rotation_rate.ok_or_else(|| {
            Error::ValueError(
                "The frame has no rotation rate, so it rotates positions but cannot \
                 transform velocities. A CK segment without angular velocity gives such a \
                 frame."
                    .into(),
            )
        })?;
        Ok((to_equatorial * self.rotation, to_equatorial * rate))
    }

    /// The rotation from the reference frame to the equatorial frame.
    ///
    /// # Errors
    /// [`Error::Bounds`] if the reference frame is not J2000, FK4, GALACTIC or
    /// ECLIPJ2000.
    fn reference_to_equatorial(&self) -> KeteResult<&'static Rotation3<f64>> {
        let to_equatorial: &'static Rotation3<f64> = match self.reference_frame_id {
            FrameId::J2000 => &IDENTITY_ROT,
            FrameId::FK4 => &FK4_EQUATORIAL_ROT,
            FrameId::GALACTIC => &GALACTIC_EQUATORIAL_ROT,
            FrameId::ECLIPJ2000 => &ECLIPTIC_EQUATORIAL_ROT,
            id => {
                return Err(Error::Bounds(format!(
                    "Reference frame ID {id} is not supported. Supported inertial references \
                     are J2000 (1), FK4 (3), GALACTIC (13) and ECLIPJ2000 (17); a frame \
                     relative to a body frame is resolved by the ephemeris that provides it \
                     (Ephemeris::try_frame)."
                )));
            }
        };
        Ok(to_equatorial)
    }

    /// Convert a position and velocity from the equatorial frame to this frame.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if the reference frame is not supported.
    /// - [`Error::ValueError`] if the frame has no rotation rate.
    pub fn from_equatorial(
        &self,
        pos: impl Into<Vector3<f64>>,
        vel: impl Into<Vector3<f64>>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let pos = pos.into();
        let (rot_p, rot_dp) = self.rotations_to_equatorial()?;

        let new_pos = rot_p.inverse_transform_vector(&pos);
        let new_vel = rot_dp.transpose() * pos + rot_p.inverse_transform_vector(&vel.into());

        Ok((new_pos, new_vel))
    }

    /// Convert a position and velocity from this frame to the equatorial frame.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if the reference frame is not supported.
    /// - [`Error::ValueError`] if the frame has no rotation rate.
    pub fn to_equatorial(
        &self,
        pos: impl Into<Vector3<f64>>,
        vel: impl Into<Vector3<f64>>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let pos = pos.into();
        let (rot_p, rot_dp) = self.rotations_to_equatorial()?;

        let new_pos = rot_p.transform_vector(&pos);
        let new_vel = rot_dp * pos + rot_p.transform_vector(&vel.into());
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
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_fk4_roundtrip() {
        let vec = FK4::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = FK4::from_equatorial(vec);
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_galactic_rot_roundtrip() {
        let vec = Galactic::to_equatorial([1.0, 2.0, 3.0].into());
        let vec_return = Galactic::from_equatorial(vec);
        assert!((1.0 - vec_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return.z).abs() <= 10.0 * f64::EPSILON);
    }

    #[test]
    fn test_noninertial_rot_roundtrip() {
        let angles = [0.11, 0.21, 0.31];
        let rates = [0.41, 0.51, 0.61];
        let pos = [1.0, 2.0, 3.0];
        let vel = [0.1, 0.2, 0.3];
        let frame = NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            0_f64,
            angles,
            rates,
            FrameId::ECLIPJ2000,
        );
        let (r_pos, r_vel) = frame.to_equatorial(pos, vel).unwrap();
        let (pos_return, vel_return) = frame.from_equatorial(r_pos, r_vel).unwrap();

        assert!((1.0 - pos_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - pos_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - pos_return.z).abs() <= 10.0 * f64::EPSILON);
        assert!((0.1 - vel_return.x).abs() <= 10.0 * f64::EPSILON);
        assert!((0.2 - vel_return.y).abs() <= 10.0 * f64::EPSILON);
        assert!((0.3 - vel_return.z).abs() <= 10.0 * f64::EPSILON);
    }

    /// A frame with no rotation rate gives its rotation, but a velocity transformation
    /// is an error rather than one that assumes the frame does not rotate.
    #[test]
    fn missing_rate_rotates_but_does_not_transform_velocity() {
        let rotation = euler_rotation::<'Z', 'X', 'Z'>(&[0.11, 0.21, 0.31], &[0.0; 3]).0;
        let pos = Vector3::new(1.0, 2.0, 3.0);
        let vel = Vector3::new(0.1, 0.2, 0.3);
        for reference in [FrameId::J2000, FrameId::ECLIPJ2000] {
            let frame = NonInertialFrame::from_rotations(0_f64, rotation, None, reference);
            let with_rate = NonInertialFrame::from_rotations(
                0_f64,
                rotation,
                Some(Matrix3::zeros()),
                reference,
            );
            assert_eq!(
                frame.rotation_to_equatorial().unwrap(),
                with_rate.rotations_to_equatorial().unwrap().0
            );
            assert!(frame.rotations_to_equatorial().is_err());
            assert!(frame.to_equatorial(pos, vel).is_err());
            assert!(frame.from_equatorial(pos, vel).is_err());
        }
    }

    /// Each supported inertial reference applies its frame's rotation to equatorial;
    /// any other reference is an error.
    #[test]
    fn rotations_to_equatorial_by_reference() {
        let rot = Rotation3::from_euler_angles(0.1, -0.2, 0.3);
        let rate = Matrix3::new(0.0, -1e-3, 0.0, 1e-3, 0.0, 0.0, 0.0, 0.0, 0.0);
        let frame = |id| {
            NonInertialFrame::from_rotations(Time::<TDB>::new(2_451_545.0), rot, Some(rate), id)
        };
        let cases = [
            (FrameId::J2000, Equatorial::rotation_to_equatorial()),
            (FrameId::FK4, FK4::rotation_to_equatorial()),
            (FrameId::GALACTIC, Galactic::rotation_to_equatorial()),
            (FrameId::ECLIPJ2000, Ecliptic::rotation_to_equatorial()),
        ];
        for (id, to_eq) in cases {
            let (r, dr) = frame(id).rotations_to_equatorial().unwrap();
            assert!((r.matrix() - (to_eq * rot).matrix()).norm() < 1e-15);
            assert!((dr - to_eq * rate).norm() < 1e-15);
        }
        assert!(frame(FrameId(2)).rotations_to_equatorial().is_err());
        assert!(frame(FrameId(-1000)).rotations_to_equatorial().is_err());
    }
}
