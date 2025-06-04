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
use crate::time::Time;
use lazy_static::lazy_static;
use nalgebra::{Matrix3, Rotation3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::fmt::Debug;

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

/// Frame which supports vector conversion
pub trait NonInertialFrame: Sized + Sync + Send + Clone + Copy + Debug + PartialEq {
    /// Construct a new non-inertial frame from the provided angles.
    fn new(angles: [f64; 6]) -> Self;

    /// Convert a vector from the equatorial frame to this frame.
    #[allow(clippy::wrong_self_convention)]
    fn from_equatorial(
        &self,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let (rot_p, rot_dp) = self.rotations_to_equatorial();

        let new_pos = rot_p.inverse_transform_vector(&pos);
        let new_vel = rot_dp.transpose() * pos + rot_p.inverse_transform_vector(&vel);

        (new_pos, new_vel)
    }

    /// Convert a vector from input frame to equatorial frame.
    fn to_equatorial(&self, pos: Vector3<f64>, vel: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        let (rot_p, rot_dp) = self.rotations_to_equatorial();

        let new_pos = rot_p.transform_vector(&pos);
        let new_vel = rot_dp * pos + rot_p.transform_vector(&vel);
        (new_pos, new_vel)
    }

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

    /// Rotation matrix from the non-inertial frame to the reference frame.
    /// The second rotation is the derivative of the first rotation with respect to time.
    fn rotations_to_equatorial(&self) -> (Rotation3<f64>, Matrix3<f64>) {
        euler_rotation::<'Z', 'X', 'Z'>(self.angles(), self.rates())
    }

    /// Euler angles for this non-inertial frame.
    fn angles(&self) -> &[f64; 3];

    /// Euler rates for this non-inertial frame.
    fn rates(&self) -> &[f64; 3];

    /// Return the SPICE frame ID for this non-inertial frame.
    fn spice_frame_id() -> i32;
}

/// NonInertial rotation frame defined by rotations to and from the Equatorial Inertial Frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EclipticNonInertial(pub [f64; 3], pub [f64; 3]);

impl NonInertialFrame for EclipticNonInertial {
    fn new(angles: [f64; 6]) -> Self {
        Self(
            angles[..3].try_into().unwrap(),
            angles[3..].try_into().unwrap(),
        )
    }

    /// Rotation matrix from the non-inertial frame to the reference frame.
    /// The second rotation is the derivative of the first rotation with respect to time.
    fn rotations_to_equatorial(&self) -> (Rotation3<f64>, Matrix3<f64>) {
        let (rot_p, rot_dp) = euler_rotation::<'Z', 'X', 'Z'>(self.angles(), self.rates());
        (
            *ECLIPTIC_EQUATORIAL_ROT * rot_p,
            *ECLIPTIC_EQUATORIAL_ROT * rot_dp,
        )
    }

    fn spice_frame_id() -> i32 {
        17
    }

    fn angles(&self) -> &[f64; 3] {
        &self.0
    }

    fn rates(&self) -> &[f64; 3] {
        &self.1
    }
}

/// NonInertial rotation frame defined by rotations to and from the Equatorial Inertial Frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EquatorialNonInertial(pub [f64; 3], pub [f64; 3]);

impl NonInertialFrame for EquatorialNonInertial {
    fn new(angles: [f64; 6]) -> Self {
        Self(
            angles[..3].try_into().unwrap(),
            angles[3..].try_into().unwrap(),
        )
    }

    fn spice_frame_id() -> i32 {
        1
    }

    fn angles(&self) -> &[f64; 3] {
        &self.0
    }

    fn rates(&self) -> &[f64; 3] {
        &self.1
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

/// Compute the angle of obliquity of Earth.
///
/// This is only valid for several centuries near J2000.
///
/// The equation here is from the 2010 Astronomical Almanac.
///
#[inline(always)]
pub fn calc_obliquity(jd: f64) -> f64 {
    // centuries from j2000
    let c = (jd - Time::j2000().jd) / 365.25 / 100.0;
    (23.439279444444444
        + c * (-0.013010213611111
            + c * (-5.08611111111111e-08
                + c * (5.565e-07 - c * (1.6e-10 + -1.1777777777777779e-11 * c)))))
        .to_radians()
}

/// Rotation which transforms a vector from the J2000 Equatorial frame to the
/// desired epoch.
///
/// Earth's north pole precesses at a rate of about 50 arcseconds per year.
/// This means there was an approximately 20 arcminute rotation of the Equatorial
/// axis from the year 2000 to 2025.
///
/// This implementation is valid for around 200 years on either side of 2000 to
/// within sub micro-arcsecond accuracy.
///
/// This function is an implementation equation (21) from this paper:
///     "Expressions for IAU 2000 precession quantities"
///     Capitaine, N. ; Wallace, P. T. ; Chapront, J.
///     Astronomy and Astrophysics, v.412, p.567-586 (2003)
///
/// It is recommended to first look at the following paper, as it provides useful
/// discussion to help understand the above model. This defines the model used
/// by JPL Horizons:
///     "Precession matrix based on IAU (1976) system of astronomical constants."
///     Lieske, J. H.
///     Astronomy and Astrophysics, vol. 73, no. 3, Mar. 1979, p. 282-284.
///
/// The IAU 2000 model paper improves accuracy by approximately ~300 mas/century over
/// the 1976 model.
///
/// # Arguments
///
/// * `tdb_time` - Time in TDB scaled Julian Days.
///
#[inline(always)]
pub fn earth_precession_rotation(tdb_time: f64) -> Rotation3<f64> {
    // centuries since 2000
    let t = (tdb_time - 2451545.0) / 36525.0;

    // angles as defined in the cited paper, equations (21)
    // Note that equation 45 is an even more developed model, which takes into
    // account frame bias in addition to simple precession, however more clarity
    // on the DE source and interpretation is probably required to take advantage
    // of this increased precision.
    let angle_c = -((2.5976176
        + (2306.0809506 + (0.3019015 + (0.0179663 + (-0.0000327 - 0.0000002 * t) * t) * t) * t)
            * t)
        / 3600.0)
        .to_radians();
    let angle_a = -((-2.5976176
        + (2306.0803226 + (1.094779 + (0.0182273 + (0.000047 - 0.0000003 * t) * t) * t) * t) * t)
        / 3600.0)
        .to_radians();
    let angle_b = ((2004.1917476
        + (-0.4269353 + (-0.0418251 + (-0.0000601 - 0.0000001 * t) * t) * t) * t)
        * t
        / 3600.0)
        .to_radians();
    let z_axis = Vector3::z_axis();
    Rotation3::from_axis_angle(&z_axis, angle_a)
        * Rotation3::from_axis_angle(&Vector3::y_axis(), angle_b)
        * Rotation3::from_axis_angle(&z_axis, angle_c)
}

lazy_static! {
    static ref IDENTITY_ROT: Rotation3<f64> = Rotation3::identity();
    static ref ECLIPTIC_EQUATORIAL_ROT: Rotation3<f64> = {
        let x = nalgebra::Unit::new_unchecked(Vector3::x_axis());
        Rotation3::from_axis_angle(&x, OBLIQUITY)
    };
    static ref FK4_EQUATORIAL_ROT: Rotation3<f64> = {
        let y = nalgebra::Unit::new_unchecked(Vector3::y_axis());
        let z = nalgebra::Unit::new_unchecked(Vector3::z_axis());
        let r1 = Rotation3::from_axis_angle(&z, (1152.84248596724 + 0.525) / 3600.0 * PI / 180.0);
        let r2 = Rotation3::from_axis_angle(&y, -1002.26108439117 / 3600.0 * PI / 180.0);
        let r3 = Rotation3::from_axis_angle(&z, 1153.04066200330 / 3600.0 * PI / 180.0);
        r3 * r2 * r1
    };
    static ref GALACTIC_EQUATORIAL_ROT: Rotation3<f64> = {
        let x = nalgebra::Unit::new_unchecked(Vector3::x_axis());
        let z = nalgebra::Unit::new_unchecked(Vector3::z_axis());
        let r1 = Rotation3::from_axis_angle(&z, 1177200.0 / 3600.0 * PI / 180.0);
        let r2 = Rotation3::from_axis_angle(&x, 225360.0 / 3600.0 * PI / 180.0);
        let r3 = Rotation3::from_axis_angle(&z, 1016100.0 / 3600.0 * PI / 180.0);
        (*FK4_EQUATORIAL_ROT) * r3 * r2 * r1
    };
}

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
        let frame = EclipticNonInertial(angles, rates);
        let (r_pos, r_vel) = frame.to_equatorial(pos, vel);
        let (pos_return, vel_return) = frame.from_equatorial(r_pos, r_vel);

        assert!((1.0 - pos_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - pos_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - pos_return[2]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.1 - vel_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.2 - vel_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.3 - vel_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
}
