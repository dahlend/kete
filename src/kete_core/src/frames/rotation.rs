// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Rotation related math utilities.

use std::f64::consts::PI;

use nalgebra::{Matrix3, Quaternion, Rotation3, Unit, Vector3};

/// Convert a quaternion to specified euler angles.
///
/// Implementation of:
///     "Quaternion to Euler angles conversion: A direct,
///     general and computationally efficient method"
///     Evandro Bernardes, Stephane Viollet 2022
///     10.1371/journal.pone.0276302
///
/// The const generics of this function are used to specify the output axis.
/// For example:
///
/// ```rust
///     use nalgebra::UnitQuaternion;
///     use kete_core::frames::quaternion_to_euler;
///     let quat = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3).into_inner();
///     let euler = quaternion_to_euler::<'X', 'Y', 'Z'>(quat);
///     assert!((euler[0] - 0.1).abs() < 1e-12);
///     assert!((euler[1] - 0.2).abs() < 1e-12);
///     assert!((euler[2] - 0.3).abs() < 1e-12);
/// ```
///
#[must_use]
#[allow(clippy::cast_sign_loss, reason = "Casts are positive by construction")]
pub fn quaternion_to_euler<const E1: char, const E2: char, const E3: char>(
    quat: Quaternion<f64>,
) -> [f64; 3] {
    const {
        // compile time checks to make sure the axes are valid
        check_axis::<E1>();
        check_axis::<E2>();
        check_axis::<E3>();

        assert!(E1 != E2 && E2 != E3, "Middle axis must not match outer.");
    }

    let idi = const { char_to_index::<E1>() };
    let idj = const { char_to_index::<E2>() };
    let idk = const {
        if E1 == E3 {
            // 1 + 2 + 3 = 6, so the remaining axis is 6 - i - j
            6 - char_to_index::<E1>() - char_to_index::<E2>()
        } else {
            char_to_index::<E3>()
        }
    };

    let proper = const { E1 == E3 };

    let epsilon = f64::from((idi - idj) * (idj - idk) * (idk - idi) / 2);

    let idi = idi as usize;
    let idj = idj as usize;
    let idk = idk as usize;

    let quat = [quat.w, quat.i, quat.j, quat.k];

    let [a, b, c, d] = if proper {
        [quat[0], quat[idi], quat[idj], quat[idk] * epsilon]
    } else {
        [
            quat[0] - quat[idj],
            quat[idi] + quat[idk] * epsilon,
            quat[idj] + quat[0],
            quat[idk] * epsilon - quat[idi],
        ]
    };

    let normal = a.powi(2) + b.powi(2) + c.powi(2) + d.powi(2);

    let mut theta_2 = (2.0 * ((a.powi(2) + b.powi(2)) / normal) - 1.0).acos();
    let theta_plus = b.atan2(a);
    let theta_minus = d.atan2(c);

    let (theta_1, mut theta_3) = match theta_2 {
        tmp if tmp.abs() < 1e-10 => (0.0, 2.0 * theta_plus),
        tmp if (tmp - PI / 2.0).abs() < 1e-10 => (0.0, 2.0 * theta_minus),
        _ => (theta_plus - theta_minus, theta_plus + theta_minus),
    };

    if !proper {
        theta_3 *= epsilon;
        theta_2 -= PI / 2.0;
    }

    [
        theta_1.rem_euclid(2.0 * PI),
        theta_2,
        theta_3.rem_euclid(2.0 * PI),
    ]
}

/// Rotation from a target inertial frame to the frame defined by the Euler angles
/// `angles`, along with its time derivative given the angle `rates`.
///
/// The rotation transforms positions between the frames, and together with its
/// derivative it also transforms velocities.
#[must_use]
pub fn euler_rotation<const E1: char, const E2: char, const E3: char>(
    angles: &[f64; 3],
    rates: &[f64; 3],
) -> (Rotation3<f64>, Matrix3<f64>) {
    let r_e1 = rotation::<E1>(angles[0]);
    let r_e2 = rotation::<E2>(angles[1]);
    let r_e3 = rotation::<E3>(angles[2]);
    let dr_e1 = rotation_derivative::<E1>(angles[0]);
    let dr_e2 = rotation_derivative::<E2>(angles[1]);
    let dr_e3 = rotation_derivative::<E3>(angles[2]);

    // math for computing the derivative:
    // r = rot_z(z1) * rot_x(x) * rot_z(z0)
    // dr / dt =
    //  (d rot_z(z1) / d z1 * d z1 / dt) * rot_x(x) * rot_z(z0) +
    //  rot_z(z1) * (d rot_x(x) / d x * d x / dt) * rot_z(z0) +
    //  rot_z(z0) * rot_x(x) * (d rot_z(z1) / d z1 * d z1 / dt)
    let mut dr_dt = dr_e1 * r_e2 * r_e3 * rates[0];
    dr_dt += r_e1 * dr_e2 * r_e3 * rates[1];
    dr_dt += r_e1 * r_e2 * dr_e3 * rates[2];

    ((r_e1 * r_e2 * r_e3), dr_dt)
}

/// Convert the character axis to an index X=1, Y=2, Z=3.
const fn char_to_index<const E: char>() -> i8 {
    const { check_axis::<E>() }
    match E {
        'X' => 1,
        'Y' => 2,
        'Z' => 3,
        _ => unreachable!(),
    }
}

/// Convert the character axis to a unit vector.
#[inline(always)]
const fn char_to_vector<const E: char>() -> [f64; 3] {
    const { check_axis::<E>() }
    match E {
        'X' => [1.0, 0.0, 0.0],
        'Y' => [0.0, 1.0, 0.0],
        'Z' => [0.0, 0.0, 1.0],
        _ => unreachable!(),
    }
}

/// ensure the axis is in the set 'XYZ'
const fn check_axis<const E: char>() {
    assert!(E == 'X' || E == 'Y' || E == 'Z', "Axis must be one of XYZ.");
}

/// Derivative of the rotation matrix with respect to the rotation angle.
fn rotation_derivative<const E: char>(angle: f64) -> Matrix3<f64> {
    const { check_axis::<E>() }
    let (sin_a, cos_a) = angle.sin_cos();
    match E {
        'X' => Matrix3::<f64>::from([[0.0, 0.0, 0.0], [0.0, -sin_a, cos_a], [0.0, -cos_a, -sin_a]]),
        'Y' => Matrix3::<f64>::from([[-sin_a, 0.0, cos_a], [0.0, 0.0, 0.0], [-cos_a, 0.0, -sin_a]]),
        'Z' => Matrix3::<f64>::from([[-sin_a, cos_a, 0.0], [-cos_a, -sin_a, 0.0], [0.0, 0.0, 0.0]]),
        _ => unreachable!(),
    }
}

/// Construct a rotation around the specified axis.
#[inline(always)]
fn rotation<const E: char>(angle: f64) -> Rotation3<f64> {
    let axis: [f64; 3] = const { char_to_vector::<E>() };
    Rotation3::from_axis_angle(&Unit::new_unchecked(Vector3::from(axis)), angle)
}

// tests
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    #[test]
    fn test_quaternion_to_euler() {
        let quat = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let euler = quaternion_to_euler::<'X', 'Y', 'Z'>(quat.into_inner() * 5.0);
        assert!((euler[0] - 0.1).abs() < 1e-12);
        assert!((euler[1] - 0.2).abs() < 1e-12);
        assert!((euler[2] - 0.3).abs() < 1e-12);

        let quat = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.8);
        let euler = quaternion_to_euler::<'Z', 'X', 'Z'>(quat.into_inner());
        assert!((euler[0] - 0.0).abs() < 1e-12);
        assert!((euler[1] - 0.0).abs() < 1e-12);
        assert!((euler[2] - 0.8).abs() < 1e-12);
    }
}
