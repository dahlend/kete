//! Define coordinate frames and object states.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.
//!
use lazy_static::lazy_static;
use nalgebra::{Matrix3, Rotation3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt::Debug;

/// Frame which supports vector conversion
pub trait InertialFrame: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// Convert a vector from input frame to equatorial frame.
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64>;

    /// Convert a vector from the equatorial frame to this frame.
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64>;

    /// Convert between frames.
    fn convert<Target: InertialFrame>(vec: Vector3<f64>) -> Vector3<f64> {
        Target::from_equatorial(Self::to_equatorial(vec))
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Equatorial {}

impl InertialFrame for Equatorial {
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        vec
    }
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        vec
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ecliptic {}
impl InertialFrame for Ecliptic {
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        ECLIPTIC_EQUATORIAL_ROT.inverse_transform_vector(&vec)
    }
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        ECLIPTIC_EQUATORIAL_ROT.transform_vector(&vec)
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Galactic {}
impl InertialFrame for Galactic {
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        GALACTIC_ECLIPTIC_ROT.inverse_transform_vector(&vec)
    }
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        GALACTIC_ECLIPTIC_ROT.transform_vector(&vec)
    }
}

/// Equatorial frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FK4 {}
impl InertialFrame for FK4 {
    fn from_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        FK4_ECLIPTIC_ROT.inverse_transform_vector(&vec)
    }
    fn to_equatorial(vec: Vector3<f64>) -> Vector3<f64> {
        FK4_ECLIPTIC_ROT.transform_vector(&vec)
    }
}

/// Frame which supports vector conversion
pub trait NonInertialFrame {
    /// Convert a vector from input frame to equatorial frame.
    fn to_equatorial(&self, pos: Vector3<f64>, vel: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>);

    /// Convert a vector from the equatorial frame to this frame.
    #[allow(clippy::wrong_self_convention)]
    fn from_equatorial(&self, pos: Vector3<f64>, vel: Vector3<f64>)
        -> (Vector3<f64>, Vector3<f64>);

    /// Convert between frames.
    fn convert<T: NonInertialFrame>(
        &self,
        target_frame: &T,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let (pos, vel) = self.to_equatorial(pos, vel);
        target_frame.from_equatorial(pos, vel)
    }
}

/// NonInertial rotation frame defined by rotations to and from the Ecliptic Inertial Frame.
#[derive(Debug, Clone)]
pub struct EclipticNonInertial(pub [f64; 6]);

impl NonInertialFrame for EclipticNonInertial {
    fn to_equatorial(&self, pos: Vector3<f64>, vel: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        let (rot_p, rot_dp) = noninertial_rotation(&self.0);

        let new_pos = rot_p.transpose() * pos;
        let new_vel = rot_dp.transpose() * pos + rot_p.transpose() * vel;

        (new_pos, new_vel)
    }

    fn from_equatorial(
        &self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let (rot_p, rot_dp) = noninertial_rotation(&self.0);

        let new_pos = rot_p * pos;
        let new_vel = rot_dp * pos + rot_p * vel;

        (new_pos, new_vel)
    }
}

/// Ecliptic obliquity angle in radians at the J2000 epoch. This is using the definition
/// from the 1984 JPL DE Series. These constants allow the conversion between Ecliptic
/// and Equatorial frames. Note that there are more modern definitions for these values,
/// however these are used for compatibility with JPL Horizons and Spice.
///
/// See:
///     - https://en.wikipedia.org/wiki/Axial_tilt#Short_term
///     - https://ssd.jpl.nasa.gov/horizons/manual.html#defs
const OBLIQUITY: f64 = 0.40909280422232897;

lazy_static! {
    static ref ECLIPTIC_EQUATORIAL_ROT: Rotation3<f64> = {
        let x = nalgebra::Unit::new_unchecked(Vector3::new(1.0, 0.0, 0.0));
        Rotation3::from_axis_angle(&x, OBLIQUITY)
    };
    static ref FK4_ECLIPTIC_ROT: Rotation3<f64> = {
        let y = nalgebra::Unit::new_unchecked(Vector3::new(0.0, 1.0, 0.0));
        let z = nalgebra::Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0));
        let r1 = Rotation3::from_axis_angle(&z, (1152.84248596724 + 0.525) / 3600.0 * PI / 180.0);
        let r2 = Rotation3::from_axis_angle(&y, -1002.26108439117 / 3600.0 * PI / 180.0);
        let r3 = Rotation3::from_axis_angle(&z, 1153.04066200330 / 3600.0 * PI / 180.0);
        (*ECLIPTIC_EQUATORIAL_ROT).inverse() * r3 * r2 * r1
    };
    static ref GALACTIC_ECLIPTIC_ROT: Rotation3<f64> = {
        let x = nalgebra::Unit::new_unchecked(Vector3::new(1.0, 0.0, 0.0));
        let z = nalgebra::Unit::new_unchecked(Vector3::new(0.0, 0.0, 1.0));
        let r1 = Rotation3::from_axis_angle(&z, 1177200.0 / 3600.0 * PI / 180.0);
        let r2 = Rotation3::from_axis_angle(&x, 225360.0 / 3600.0 * PI / 180.0);
        let r3 = Rotation3::from_axis_angle(&z, 1016100.0 / 3600.0 * PI / 180.0);
        (*FK4_ECLIPTIC_ROT) * r3 * r2 * r1
    };
}

/// Derivative of the z rotation matrix with respect to the rotation angle.
fn rot_z_der(angle: f64) -> Matrix3<f64> {
    let (sin_a, cos_a) = angle.sin_cos();
    Matrix3::<f64>::from([[-sin_a, cos_a, 0.0], [-cos_a, -sin_a, 0.0], [0.0, 0.0, 0.0]])
}

/// Derivative of the x rotation matrix with respect to the rotation angle.
fn rot_x_der(angle: f64) -> Matrix3<f64> {
    let (sin_a, cos_a) = angle.sin_cos();
    Matrix3::<f64>::from([[0.0, 0.0, 0.0], [0.0, -sin_a, cos_a], [0.0, -cos_a, -sin_a]])
}

/// Compute two rotation matrices from a target inertial frame to the frame defined by
/// the provided angles. The first 3 angles here define the rotation ZXZ, the second
/// three values define the derivative of the 3 angles.
///
/// This then provides two rotation matrices, one is the 3x3 rotation matrix, and the
/// second is the derivative of the 3x3 matrix with respect to time. These two matrices
/// may be used to compute the new position and velocities when moving from one frame
/// to another.
pub fn noninertial_rotation(frame_angles: &[f64; 6]) -> (Matrix3<f64>, Matrix3<f64>) {
    let r_z1 = Rotation3::from_axis_angle(&Vector3::z_axis(), frame_angles[0]);
    let r_x = Rotation3::from_axis_angle(&Vector3::x_axis(), frame_angles[1]);
    let r_z0 = Rotation3::from_axis_angle(&Vector3::z_axis(), frame_angles[2]);
    let dr_z1 = rot_z_der(frame_angles[0]);
    let dr_x = rot_x_der(frame_angles[1]);
    let dr_z0 = rot_z_der(frame_angles[2]);

    // math for computing the derivative:
    // r = rot_z(z1) * rot_x(x) * rot_z(z0)
    // dr / dt =
    //  (d rot_z(z1) / d z1 * d z1 / dt) * rot_x(x) * rot_z(z0) +
    //  rot_z(z1) * (d rot_x(x) / d x * d x / dt) * rot_z(z0) +
    //  rot_z(z0) * rot_x(x) * (d rot_z(z1) / d z1 * d z1 / dt)
    let mut dr_dt = dr_z1 * r_x * r_z0 * frame_angles[3];
    dr_dt += r_z1 * dr_x * r_z0 * frame_angles[4];
    dr_dt += r_z1 * r_x * dr_z0 * frame_angles[5];

    ((r_z1 * r_x * r_z0).into(), dr_dt)
}

// #[cfg(test)]
// mod tests {
//     use crate::frames::*;

//     #[test]
//     fn test_ecliptic_rot_roundtrip() {
//         let vec = ecliptic_to_equatorial(&[1.0, 2.0, 3.0].into());
//         let vec_return = equatorial_to_ecliptic(&vec);
//         assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
//         assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
//         assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
//     }
//     #[test]
//     fn test_fk4_roundtrip() {
//         let vec = ecliptic_to_fk4(&[1.0, 2.0, 3.0].into());
//         let vec_return = fk4_to_ecliptic(&vec);
//         assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
//         assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
//         assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
//     }
//     #[test]
//     fn test_galactic_rot_roundtrip() {
//         let vec = ecliptic_to_galactic(&[1.0, 2.0, 3.0].into());
//         let vec_return = galactic_to_ecliptic(&vec);
//         assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
//         assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
//         assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
//     }

//     #[test]
//     fn test_noninertial_rot_roundtrip() {
//         let angles = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61];
//         let pos = [1.0, 2.0, 3.0].into();
//         let vel = [0.1, 0.2, 0.3].into();
//         let (r_pos, r_vel) = inertial_to_noninertial(&angles, &pos, &vel);
//         let (pos_return, vel_return) = noninertial_to_inertial(&angles, &r_pos, &r_vel);

//         assert!((1.0 - pos_return[0]).abs() <= 10.0 * f64::EPSILON);
//         assert!((2.0 - pos_return[1]).abs() <= 10.0 * f64::EPSILON);
//         assert!((3.0 - pos_return[2]).abs() <= 10.0 * f64::EPSILON);
//         assert!((0.1 - vel_return[0]).abs() <= 10.0 * f64::EPSILON);
//         assert!((0.2 - vel_return[1]).abs() <= 10.0 * f64::EPSILON);
//         assert!((0.3 - vel_return[2]).abs() <= 10.0 * f64::EPSILON);
//     }
// }
