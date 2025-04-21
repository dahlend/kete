//! Define coordinate frames and object states.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.
//!
use lazy_static::lazy_static;
use nalgebra::{Matrix3, Rotation3, UnitVector3, Vector3};
use serde::{Deserialize, Serialize};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::fmt::{self, Debug, Display};

use crate::prelude::{Error, KeteResult};
use crate::time::Time;

/// Coordinate frames.
///
/// Frames define the orientation of the coordinate system.
/// These distance units are always AU, and time is measured in days with TDB scaling.
///
#[derive(Debug, PartialEq, Clone, Copy, Deserialize, Serialize)]
pub enum Frame {
    /// Equivalent to Ecliptic J2000
    Ecliptic,

    /// Equivalent to Equatorial J2000
    Equatorial,

    /// FK4 Frame
    FK4,

    /// Galactic Frame as defined by SPICE
    Galactic,

    /// Unknown is to allow SPK files to be loaded with other frames.
    Unknown(i32),

    /// Non-inertial frame as defined by rotations from the ecliptic frame.
    ///
    /// - i32 value represents the frame identifier.
    /// - array of 6 floats represent the euler angles and their derivatives to move to
    ///   this frame from the ecliptic frame.
    ///
    /// Rotation is done with a ZXZ set of chained rotations.
    EclipticNonInertial(i32, [f64; 6]),
    // Other non inertial frames will require multi-step conversions
}

impl From<Frame> for i32 {
    fn from(value: Frame) -> Self {
        match value {
            Frame::Equatorial => 1,
            Frame::Ecliptic => 2,
            Frame::FK4 => 3,
            Frame::Galactic => 4,
            Frame::EclipticNonInertial(..) => 5,
            Frame::Unknown(_) => 0,
        }
    }
}

impl From<i32> for Frame {
    fn from(value: i32) -> Self {
        match value {
            1 => Frame::Equatorial,
            2 => Frame::Ecliptic,
            3 => Frame::FK4,
            4 => Frame::Galactic,
            i => Frame::Unknown(i),
        }
    }
}

impl Frame {
    /// Change a vector from the current frame into the target frame.
    pub fn try_vec_frame_change(
        &self,
        mut vec: Vector3<f64>,
        target: Frame,
    ) -> KeteResult<Vector3<f64>> {
        match self {
            Frame::Equatorial => {
                vec = equatorial_to_ecliptic(&vec);
            }
            Frame::Ecliptic => {}
            Frame::EclipticNonInertial(_, _) => {
                 Err(Error::ValueError(
                    "Cannot convert bare vector between non-inertial frames.  This may only be done with a State".into(),
                ))?
            }
            Frame::FK4 => {
                vec = fk4_to_ecliptic(&vec);
            }
            Frame::Galactic => {
                vec = galactic_to_ecliptic(&vec);
            }
            Frame::Unknown(id) =>  Err(Error::UnknownFrame(*id))?,
        }

        // new_vec is now in ecliptic.
        match target {
            Frame::Equatorial => {
                vec = ecliptic_to_equatorial(&vec);
            }
            Frame::Ecliptic => {}
            Frame::EclipticNonInertial(_, _) => {
                 Err(Error::ValueError(
                    "Cannot convert bare vector between non-inertial frames. This may only be done with a State.".into(),
                ))?
            }
            Frame::FK4 => {
                vec = ecliptic_to_fk4(&vec);
            }
            Frame::Galactic => {
                vec = ecliptic_to_galactic(&vec);
            }
            Frame::Unknown(id) =>  Err(Error::UnknownFrame(id))?,
        }
        Ok(vec)
    }
}

impl Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
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

/// Convert a vector in the fk4 frame to ecliptic frame.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn fk4_to_ecliptic(vec: &Vector3<f64>) -> Vector3<f64> {
    FK4_ECLIPTIC_ROT.transform_vector(vec)
}

/// Convert a vector in the ecliptic to the fk4 frame.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn ecliptic_to_fk4(vec: &Vector3<f64>) -> Vector3<f64> {
    FK4_ECLIPTIC_ROT.inverse().transform_vector(vec)
}

/// Convert a vector in the galactic frame to ecliptic frame.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn galactic_to_ecliptic(vec: &Vector3<f64>) -> Vector3<f64> {
    GALACTIC_ECLIPTIC_ROT.transform_vector(vec)
}

/// Convert a vector in the ecliptic to the galactic frame.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn ecliptic_to_galactic(vec: &Vector3<f64>) -> Vector3<f64> {
    GALACTIC_ECLIPTIC_ROT.inverse().transform_vector(vec)
}

/// Convert a vector from ecliptic to equatorial J2000 frames assuming the 1984 JPL DE
/// definition.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn ecliptic_to_equatorial(vec: &Vector3<f64>) -> Vector3<f64> {
    ECLIPTIC_EQUATORIAL_ROT.transform_vector(vec)
}

/// Convert a vector from equatorial to ecliptic J2000 frames assuming the 1984 JPL DE
/// definition.
///
/// # Arguments
///
/// * `vec` - Vector, arbitrary units.
#[inline(always)]
pub fn equatorial_to_ecliptic(vec: &Vector3<f64>) -> Vector3<f64> {
    ECLIPTIC_EQUATORIAL_ROT.inverse().transform_vector(vec)
}

/// Derivative of the z rotation matrix with respect to the rotation angle.
#[inline(always)]
fn rot_z_der(angle: f64) -> Matrix3<f64> {
    let (sin_a, cos_a) = angle.sin_cos();
    Matrix3::<f64>::from([[-sin_a, cos_a, 0.0], [-cos_a, -sin_a, 0.0], [0.0, 0.0, 0.0]])
}

/// Derivative of the x rotation matrix with respect to the rotation angle.
#[inline(always)]
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
#[inline(always)]
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

/// Convert positions and velocities from a non-inertial frame to an inertial frame
/// defined by the provided angles. The first 3 angles here define the rotation ZXZ, the
/// second three values define the derivative of the 3 angles. These angles define the
/// rotation from the inertial to the non-inertial frame.
#[inline(always)]
pub fn noninertial_to_inertial(
    frame_angles: &[f64; 6],
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let (rot_p, rot_dp) = noninertial_rotation(frame_angles);

    let new_pos = rot_p * pos;
    let new_vel = rot_dp * pos + rot_p * vel;

    (new_pos, new_vel)
}

/// Convert positions and velocities from a non-inertial frame to an inertial frame
/// defined by the provided angles. The first 3 angles here define the rotation ZXZ, the
/// second three values define the derivative of the 3 angles. These angles define the
/// rotation from the inertial to the non-inertial frame.
#[inline(always)]
pub fn inertial_to_noninertial(
    frame_angles: &[f64; 6],
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let (rot_p, rot_dp) = noninertial_rotation(frame_angles);

    let new_pos = rot_p.transpose() * pos;
    let new_vel = rot_dp.transpose() * pos + rot_p.transpose() * vel;

    (new_pos, new_vel)
}

/// Create a unit vector from latitude and longitude.
///
/// In the Equatorial frame, lat = dec, lon = ra
///
/// <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>
#[inline(always)]
pub fn from_lat_lon(lat: f64, lon: f64) -> [f64; 3] {
    let (lat_sin, lat_cos) = lat.sin_cos();
    let (lon_sin, lon_cos) = lon.sin_cos();
    [lat_cos * lon_cos, lat_cos * lon_sin, lat_sin]
}

/// Convert a vector to latitude and longitude in the current coordinate frame.
///
/// In the Equatorial frame, lat = dec, lon = ra
///
/// <https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates>
#[inline(always)]
pub fn to_lat_lon(x: f64, y: f64, z: f64) -> (f64, f64) {
    let r = Vector3::from([x, y, z]).norm();
    if r < 1e-10 {
        return (0.0, 0.0);
    }
    let lon = (3.0 * FRAC_PI_2 - (z / r).clamp(-1.0, 1.0).acos()).rem_euclid(TAU) - PI;
    let lat = y.atan2(x).rem_euclid(TAU);
    (lon, lat)
}

/// Rotate a collection of vectors around the specified rotation vector.
///
/// # Arguments
///
/// * `vectors` - A Matrix containing N vectors of length 3.
/// * `rotation_vec` - The single vector around which to rotate the vectors.
/// * `angle` - The angle in radians to rotate the vectors.
///
#[inline(always)]
pub fn rotate_around(
    vector: &Vector3<f64>,
    rotation_vec: Vector3<f64>,
    angle: f64,
) -> Vector3<f64> {
    let rot = Rotation3::from_axis_angle(&UnitVector3::new_normalize(rotation_vec), angle);
    rot.transform_vector(vector)
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

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, TAU};

    use nalgebra::Matrix3;

    use crate::frames::*;

    #[test]
    fn test_earth_precession() {
        let rot = earth_precession_rotation(2451545.0);
        assert!((rot.matrix() - Matrix3::identity()).norm() < 1e-16);

        let rot = earth_precession_rotation(2433282.42345905);

        let expected = Matrix3::new(
            0.9999257168056067,
            -0.011178271729385899,
            -0.004858715050078201,
            0.011178271443436222,
            0.9999375208015913,
            -2.72158794420131e-05,
            0.004858715707952332,
            -2.70981779365330e-05,
            0.9999881960040119,
        );
        assert!((rot.matrix() - expected).norm() < 1e-16);
    }

    #[test]
    fn test_ecliptic_rot_roundtrip() {
        let vec = ecliptic_to_equatorial(&[1.0, 2.0, 3.0].into());
        let vec_return = equatorial_to_ecliptic(&vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_fk4_roundtrip() {
        let vec = ecliptic_to_fk4(&[1.0, 2.0, 3.0].into());
        let vec_return = fk4_to_ecliptic(&vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_galactic_rot_roundtrip() {
        let vec = ecliptic_to_galactic(&[1.0, 2.0, 3.0].into());
        let vec_return = galactic_to_ecliptic(&vec);
        assert!((1.0 - vec_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - vec_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - vec_return[2]).abs() <= 10.0 * f64::EPSILON);
    }

    #[test]
    fn test_noninertial_rot_roundtrip() {
        let angles = [0.11, 0.21, 0.31, 0.41, 0.51, 0.61];
        let pos = [1.0, 2.0, 3.0].into();
        let vel = [0.1, 0.2, 0.3].into();
        let (r_pos, r_vel) = inertial_to_noninertial(&angles, &pos, &vel);
        let (pos_return, vel_return) = noninertial_to_inertial(&angles, &r_pos, &r_vel);

        assert!((1.0 - pos_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((2.0 - pos_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((3.0 - pos_return[2]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.1 - vel_return[0]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.2 - vel_return[1]).abs() <= 10.0 * f64::EPSILON);
        assert!((0.3 - vel_return[2]).abs() <= 10.0 * f64::EPSILON);
    }
    #[test]
    fn test_frame_conversion() {
        let f: Frame = 2i32.into();
        assert!(f == Frame::Ecliptic);
    }

    #[test]
    fn test_lat_lon() {
        // Several cardinal axis checks.
        let [x, y, z] = from_lat_lon(0.0, 0.0);
        assert!((x - 1.0).abs() < 10.0 * f64::EPSILON);
        assert!(y.abs() < 10.0 * f64::EPSILON);
        assert!(z.abs() < 10.0 * f64::EPSILON);

        let [x, y, z] = from_lat_lon(0.0, FRAC_PI_2);
        assert!(x.abs() < 10.0 * f64::EPSILON);
        assert!((y - 1.0).abs() < 10.0 * f64::EPSILON);
        assert!(z.abs() < 10.0 * f64::EPSILON);

        let [x, y, z] = from_lat_lon(0.0, -FRAC_PI_2);
        assert!(x.abs() < 10.0 * f64::EPSILON);
        assert!((y + 1.0).abs() < 10.0 * f64::EPSILON);
        assert!(z.abs() < 10.0 * f64::EPSILON);

        let [x, y, z] = from_lat_lon(FRAC_PI_2, 0.0);
        assert!(x.abs() < 10.0 * f64::EPSILON);
        assert!(y.abs() < 10.0 * f64::EPSILON);
        assert!((z - 1.0).abs() < 10.0 * f64::EPSILON);

        let [x, y, z] = from_lat_lon(-FRAC_PI_2, 0.0);
        assert!(x.abs() < 10.0 * f64::EPSILON);
        assert!(y.abs() < 10.0 * f64::EPSILON);
        assert!((z + 1.0).abs() < 10.0 * f64::EPSILON);

        // sample conversion from around the sphere
        for idx in -10..10 {
            let lat = idx as f64 / 11.0 * FRAC_PI_2;
            for idy in 0..11 {
                let lon = idy as f64 / 11. * TAU;
                let [x, y, z] = from_lat_lon(lat, lon);
                let (new_lat, new_lon) = to_lat_lon(x, y, z);

                let [x_new, y_new, z_new] = from_lat_lon(new_lat, new_lon);
                assert!((new_lat - lat).abs() < 10.0 * f64::EPSILON);
                assert!((new_lon - lon).abs() < 10.0 * f64::EPSILON);

                assert!((x - x_new).abs() < 10.0 * f64::EPSILON);
                assert!((y - y_new).abs() < 10.0 * f64::EPSILON);
                assert!((z - z_new).abs() < 10.0 * f64::EPSILON);
            }
        }
    }
}
