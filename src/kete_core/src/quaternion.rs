//! Quaternion math utilities.
//!

use std::f64::consts::PI;

use nalgebra::Quaternion;

/// Convert a quaternion to specified euler angles.
///
/// Implementation of:
///     "Quaternion to Euler angles conversion: A direct,
///     general and computationally efficient method"
///     Evandro Bernardes, Stéphane Viollet 2022
///     10.1371/journal.pone.0276302
///
/// The const generics of this function are used to specify the output axis.
/// For example:
///
/// ```rust
///     use nalgebra::UnitQuaternion;
///     use kete_core::quaternion::quaternion_to_euler;
///     let quat = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3).into_inner();
///     let euler = quaternion_to_euler::<'x', 'y', 'z'>(quat);
///     assert!((euler[0] - 0.1).abs() < 1e-12);
///     assert!((euler[1] - 0.2).abs() < 1e-12);
///     assert!((euler[2] - 0.3).abs() < 1e-12);
/// ```
///
pub fn quaternion_to_euler<const E1: char, const E2: char, const E3: char>(
    quat: Quaternion<f64>,
) -> [f64; 3] {
    const {
        // compile time checks to make sure the axes are valid
        assert!(check_axis::<E1>(), "Axis must be one of x, y, z.");
        assert!(check_axis::<E2>(), "Axis must be one of x, y, z.");
        assert!(check_axis::<E3>(), "Axis must be one of x, y, z.");

        assert!(E1 != E2 && E2 != E3, "Middle axis must not match outer.");
    }

    let i = const { char_to_index::<E1>() };
    let j = const { char_to_index::<E2>() };
    let k = const {
        if E1 == E3 {
            // 1 + 2 + 3 = 6, so the remaining axis is 6 - i - j
            6 - char_to_index::<E1>() - char_to_index::<E2>()
        } else {
            char_to_index::<E3>()
        }
    };

    let proper = const { E1 == E3 };

    let epsilon = ((i - j) * (j - k) * (k - i) / 2) as f64;

    let i = i as usize;
    let j = j as usize;
    let k = k as usize;

    let q = [quat.w, quat.i, quat.j, quat.k];

    let [a, b, c, d] = if proper {
        [q[0], q[i], q[j], q[k] * epsilon]
    } else {
        [
            q[0] - q[j],
            q[i] + q[k] * epsilon,
            q[j] + q[0],
            q[k] * epsilon - q[i],
        ]
    };

    let n = a.powi(2) + b.powi(2) + c.powi(2) + d.powi(2);

    let mut theta_2 = (2.0 * ((a.powi(2) + b.powi(2)) / n) - 1.0).acos();
    let theta_plus = b.atan2(a);
    let theta_minus = d.atan2(c);

    let (theta_1, mut theta_3) = match theta_2 {
        t if t.abs() < 1e-10 => (0.0, 2.0 * theta_plus),
        t if (t - PI / 2.0).abs() < 1e-10 => (0.0, 2.0 * theta_minus),
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

/// Convert the character axis to an index X=1, Y=2, Z=3.
const fn char_to_index<const E: char>() -> i8 {
    const {
        assert!(check_axis::<E>(), "Axis must be one of x, y, z.");
    }
    match E {
        'x' | 'X' => 1,
        'y' | 'Y' => 2,
        'z' | 'Z' => 3,
        _ => unreachable!(),
    }
}

/// ensure the axis is in the set 'xXyYzZ'
const fn check_axis<const E: char>() -> bool {
    E == 'x' || E == 'y' || E == 'z' || E == 'X' || E == 'Y' || E == 'Z'
}
// tests
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    #[test]
    fn test_quaternion_to_euler() {
        let quat = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let euler = quaternion_to_euler::<'x', 'y', 'z'>(quat.into_inner() * 5.0);
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
