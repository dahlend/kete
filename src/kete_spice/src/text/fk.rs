//! Frame definitions from frames kernels (FK).
//!
//! A frames kernel defines a frame by these variables:
//!
//! - `FRAME_<name> = <id>`, for the lookup of the ID from the name;
//! - `FRAME_<id>_NAME`, `FRAME_<id>_CLASS` and `FRAME_<id>_CLASS_ID`;
//! - for a TK frame, the `TKFRAME_*` variables, named by either the frame ID
//!   or the frame name.
//!
//! Dynamic (class 5) and switch (class 6) frames, and inertial frames defined
//! in a frames kernel, are not supported. A request for one is an error.

use super::TextKernelVars;
use super::sclk::ClockId;
use crate::frames::{FrameDef, frame_name, id_from_name};
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::FrameId;
use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::f64::consts::PI;

/// The definition of the frame `id`, whose class a frames kernel sets.
///
/// # Errors
/// [`Error::ValueError`] if the frame is of a class kete does not support, or
/// its definition is incomplete or malformed.
fn definition(vars: &TextKernelVars, id: FrameId) -> KeteResult<FrameDef> {
    let class = vars
        .integer(&format!("FRAME_{id}_CLASS"))?
        .ok_or_else(|| Error::ValueError(format!("Frame {id} has no FRAME_{id}_CLASS.")))?;
    let name = frame_name(vars, id).unwrap_or_else(|| id.to_string());
    let class_id = vars
        .integer(&format!("FRAME_{id}_CLASS_ID"))?
        .ok_or_else(|| {
            Error::ValueError(format!("Frame {name} ({id}) has no FRAME_{id}_CLASS_ID."))
        })?;
    let def = match class {
        2 => FrameDef::Pck { class_id },
        3 => FrameDef::Ck {
            class_id,
            clock_id: ck_clock_id(vars, class_id)?,
        },
        4 => tk_definition(vars, id, &name, frame_name(vars, id).as_deref())?,
        1 => {
            return Err(Error::ValueError(format!(
                "Frame {name} ({id}) is an inertial frame defined in a frames kernel, \
                 which is not supported."
            )));
        }
        5 | 6 => {
            return Err(Error::ValueError(format!(
                "Frame {name} ({id}) is a {} frame, which is not supported.",
                if class == 5 { "dynamic" } else { "switch" }
            )));
        }
        other => {
            return Err(Error::ValueError(format!(
                "Frame {name} ({id}) has an unknown class {other}."
            )));
        }
    };
    Ok(def)
}

/// The spacecraft clock ID of the CK frame with class ID `ck_id`.
///
/// The clock ID is the value of `CK_<ck_id>_SCLK` if it is set. Otherwise it is
/// `ck_id / 1000`.
///
/// # Errors
/// [`Error::ValueError`] if `CK_<ck_id>_SCLK` is not one integer.
pub fn ck_clock_id(vars: &TextKernelVars, ck_id: i32) -> KeteResult<ClockId> {
    Ok(ClockId(
        vars.integer(&format!("CK_{ck_id}_SCLK"))?
            .unwrap_or(ck_id / 1000),
    ))
}

/// The SPK ID of the spacecraft of the CK frame with class ID `ck_id`.
///
/// It is the value of `CK_<ck_id>_SPK` if it is set. Otherwise it is
/// `ck_id / 1000` if `ck_id` is -1000 or less.
///
/// # Errors
/// [`Error::ValueError`] if `CK_<ck_id>_SPK` is not one integer, or it is not
/// set and `ck_id` is above -1000.
pub fn ck_spk_id(vars: &TextKernelVars, ck_id: i32) -> KeteResult<i32> {
    match vars.integer(&format!("CK_{ck_id}_SPK"))? {
        Some(id) => Ok(id),
        None if ck_id <= -1000 => Ok(ck_id / 1000),
        None => Err(Error::ValueError(format!(
            "CK frame class ID {ck_id} gives no SPK ID: CK_{ck_id}_SPK is not set."
        ))),
    }
}

/// The definition of the TK frame `id`.
///
/// `name` labels the frame in messages. `frame_name` is its `FRAME_<id>_NAME`,
/// which can name the `TKFRAME_*` variables.
///
/// # Errors
/// [`Error::ValueError`] if a `TKFRAME_*` variable is missing or malformed, if
/// both the ID and the name name the variables, or if the relative frame has
/// no ID.
pub(crate) fn tk_definition(
    vars: &TextKernelVars,
    id: FrameId,
    name: &str,
    frame_name: Option<&str>,
) -> KeteResult<FrameDef> {
    // The TKFRAME variables are named by either the frame ID or the frame name, not
    // both.
    let by_id = format!("TKFRAME_{id}_");
    let by_name = format!("TKFRAME_{}_", frame_name.unwrap_or_default());
    let prefix = match (
        vars.get(&format!("{by_id}RELATIVE")).is_some(),
        frame_name.is_some() && vars.get(&format!("{by_name}RELATIVE")).is_some(),
    ) {
        (true, false) => by_id,
        (false, true) => by_name,
        (true, true) => {
            return Err(Error::ValueError(format!(
                "TK frame {name} ({id}) is defined twice, by {by_id}* and {by_name}* \
                 variables."
            )));
        }
        (false, false) => {
            return Err(Error::ValueError(format!(
                "TK frame {name} ({id}) has no {by_id}RELATIVE."
            )));
        }
    };
    let get_str = |key: &str| -> KeteResult<&str> {
        vars.string(&format!("{prefix}{key}"))?
            .ok_or_else(|| Error::ValueError(format!("TK frame {name} has no {prefix}{key}.")))
    };
    let get_numbers = |key: &str, n: usize| -> KeteResult<&[f64]> {
        let v = vars
            .numbers(&format!("{prefix}{key}"))?
            .ok_or_else(|| Error::ValueError(format!("TK frame {name} has no {prefix}{key}.")))?;
        if v.len() == n {
            Ok(v)
        } else {
            Err(Error::ValueError(format!(
                "TK frame {name}: {prefix}{key} must hold {n} numbers, found {}.",
                v.len()
            )))
        }
    };
    let relative = id_from_name(vars, get_str("RELATIVE")?)?;
    let rotation = match get_str("SPEC")?.to_uppercase().as_str() {
        "MATRIX" => {
            let m = Matrix3::from_column_slice(get_numbers("MATRIX", 9)?);
            sharpened(&m).ok_or_else(|| {
                Error::ValueError(format!("TK frame {name}: the matrix is singular."))
            })?
        }
        "ANGLES" => {
            let angles = get_numbers("ANGLES", 3)?;
            let axes = vars
                .integers(&format!("{prefix}AXES"))?
                .filter(|v| v.len() == 3)
                .ok_or_else(|| {
                    Error::ValueError(format!(
                        "TK frame {name}: {prefix}AXES must hold 3 integers."
                    ))
                })?;
            let units = vars.string(&format!("{prefix}UNITS"))?.unwrap_or("RADIANS");
            let scale = angle_unit(units).ok_or_else(|| {
                Error::ValueError(format!("TK frame {name}: unknown angle unit {units}."))
            })?;
            let mut rotation = Rotation3::identity();
            for (angle, axis) in angles.iter().zip(axes) {
                let axis = match axis {
                    1 => Vector3::x_axis(),
                    2 => Vector3::y_axis(),
                    3 => Vector3::z_axis(),
                    _ => {
                        return Err(Error::ValueError(format!(
                            "TK frame {name}: axis {axis} is not 1, 2 or 3."
                        )));
                    }
                };
                // An angle rotates the axes, which is the opposite sense to a
                // rotation of vectors.
                rotation *= Rotation3::from_axis_angle(&axis, -angle * scale);
            }
            rotation
        }
        "QUATERNION" => {
            let q = get_numbers("Q", 4)?;
            let q = Quaternion::new(q[0], q[1], q[2], q[3]);
            if q.norm() == 0.0 {
                return Err(Error::ValueError(format!(
                    "TK frame {name}: the quaternion is zero."
                )));
            }
            UnitQuaternion::from_quaternion(q).to_rotation_matrix()
        }
        other => {
            return Err(Error::ValueError(format!(
                "TK frame {name}: unknown {prefix}SPEC {other}."
            )));
        }
    };
    Ok(FrameDef::Tk { relative, rotation })
}

/// The definitions of all frames that the frames kernels define.
///
/// There is one definition for each `FRAME_<id>_CLASS` variable. A frame whose
/// definition is unsupported or malformed keeps its error. The error comes when
/// the frame is used.
pub(crate) fn definitions(vars: &TextKernelVars) -> HashMap<FrameId, KeteResult<FrameDef>> {
    vars.names()
        .filter_map(|name| {
            name.strip_prefix("FRAME_")?
                .strip_suffix("_CLASS")?
                .parse::<i32>()
                .ok()
        })
        .map(|id| (FrameId(id), definition(vars, FrameId(id))))
        .collect()
}

/// The rotation from the TK matrix `m`.
///
/// The first column of the rotation is the first column of `m`, normalized.
/// The third is the normalized cross product of the first two columns of `m`.
/// The second is the cross product of the third and the first. The result is
/// `None` if `m` is singular.
fn sharpened(m: &Matrix3<f64>) -> Option<Rotation3<f64>> {
    let c1 = m.column(0).try_normalize(0.0)?;
    let c3 = c1.cross(&m.column(1)).try_normalize(0.0)?;
    let c2 = c3.cross(&c1);
    Some(Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[
        c1, c2, c3,
    ])))
}

/// Radians per unit of the SPICE angle unit `units`, matched in upper case.
pub(crate) fn angle_unit(units: &str) -> Option<f64> {
    let deg = PI / 180.0;
    Some(match units.trim().to_uppercase().as_str() {
        "RADIANS" => 1.0,
        "DEGREES" => deg,
        "ARCMINUTES" => deg / 60.0,
        "ARCSECONDS" => deg / 3600.0,
        "HOURANGLE" => PI / 12.0,
        "MINUTEANGLE" => PI / 12.0 / 60.0,
        "SECONDANGLE" => PI / 12.0 / 3600.0,
        _ => return None,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::text::TextKernels;

    /// A chain of TK frames, one of each specification; the expected rotations
    /// are from CSPICE `pxform(frame, 'J2000', 0)` for the same text.
    pub(crate) const TK_CHAIN: &str = r"KPL/FK
\begindata
FRAME_TK_MAT = 1400201
FRAME_1400201_NAME = 'TK_MAT'
FRAME_1400201_CLASS = 4
FRAME_1400201_CLASS_ID = 1400201
FRAME_1400201_CENTER = 399
TKFRAME_1400201_RELATIVE = 'J2000'
TKFRAME_1400201_SPEC = 'MATRIX'
TKFRAME_1400201_MATRIX = ( 0.9 0.1 0.0
                           -0.1 0.9 0.05
                           0.0 -0.05 1.0 )
FRAME_TK_ANG = 1400202
FRAME_1400202_NAME = 'TK_ANG'
FRAME_1400202_CLASS = 4
FRAME_1400202_CLASS_ID = 1400202
FRAME_1400202_CENTER = 399
TKFRAME_TK_ANG_RELATIVE = 'TK_MAT'
TKFRAME_TK_ANG_SPEC = 'ANGLES'
TKFRAME_TK_ANG_UNITS = 'DEGREES'
TKFRAME_TK_ANG_AXES = ( 3, 1, 2 )
TKFRAME_TK_ANG_ANGLES = ( 10.0, 20.0, 30.0 )
FRAME_TK_Q = 1400203
FRAME_1400203_NAME = 'TK_Q'
FRAME_1400203_CLASS = 4
FRAME_1400203_CLASS_ID = 1400203
FRAME_1400203_CENTER = 399
TKFRAME_1400203_RELATIVE = 'TK_ANG'
TKFRAME_1400203_SPEC = 'QUATERNION'
TKFRAME_1400203_Q = ( 0.9, 0.1, -0.2, 0.3 )
FRAME_TK_AS = 1400204
FRAME_1400204_NAME = 'TK_AS'
FRAME_1400204_CLASS = 4
FRAME_1400204_CLASS_ID = 1400204
FRAME_1400204_CENTER = 399
TKFRAME_1400204_RELATIVE = 'ECLIPJ2000'
TKFRAME_1400204_SPEC = 'ANGLES'
TKFRAME_1400204_UNITS = 'ARCSECONDS'
TKFRAME_1400204_AXES = ( 1, 2, 3 )
TKFRAME_1400204_ANGLES = ( 3600.0, -7200.0, 360.0 )
\begintext
";

    /// `pxform(name, 'J2000', 0)` from CSPICE N0067 for the frames of
    /// [`TK_CHAIN`].
    pub(crate) const TK_CHAIN_EXPECTED: [(&str, [[f64; 3]; 3]); 4] = [
        (
            "TK_MAT",
            [
                [
                    9.938_837_346_736_189e-1,
                    -1.102_635_692_839_942_6e-1,
                    6.088_287_113_245_53e-3,
                ],
                [
                    1.104_315_260_748_465_5e-1,
                    9.923_721_235_559_483e-1,
                    -5.479_458_401_920_977e-2,
                ],
                [0.0, 5.513_178_464_199_712_5e-2, 9.984_790_865_722_668e-1],
            ],
        ),
        (
            "TK_ANG",
            [
                [
                    8.780_388_162_301_459e-1,
                    5.805_583_215_910_58e-2,
                    -4.750_551_100_088_051e-1,
                ],
                [
                    8.960_866_292_346_625e-2,
                    9.551_182_314_475_183e-1,
                    2.823_463_325_167_042e-1,
                ],
                [
                    4.701_256_478_030_450_5e-1,
                    -2.904_800_927_927_392e-1,
                    8.334_285_758_053_228e-1,
                ],
            ],
        ),
        (
            "TK_Q",
            [
                [
                    4.582_647_927_101_636e-1,
                    -5.202_357_325_207_195e-1,
                    -7.206_581_452_886_947e-1,
                ],
                [
                    6.926_047_924_189_123e-1,
                    7.171_646_621_484_533e-1,
                    -7.728_809_018_115_457e-2,
                ],
                [
                    5.570_385_815_010_18e-1,
                    -4.637_128_744_968_433e-1,
                    6.889_690_767_699_291e-1,
                ],
            ],
        ),
        (
            "TK_AS",
            [
                [
                    9.993_893_048_602_067e-1,
                    1.744_265_159_014_997_8e-3,
                    3.489_949_670_250_097e-2,
                ],
                [
                    1.170_808_781_884_929_6e-2,
                    9.243_063_359_604_74e-1,
                    -3.814_717_787_503_414e-1,
                ],
                [
                    -3.292_321_385_677_502e-2,
                    3.816_474_221_613_167_5e-1,
                    9.237_214_445_637_618e-1,
                ],
            ],
        ),
    ];

    fn loaded(text: &str) -> TextKernels {
        let mut frames = TextKernels::default();
        frames.load_text(text).unwrap();
        frames
    }

    /// Each TK specification gives the rotation from the frame to its relative
    /// frame.
    #[test]
    fn tk_definitions() {
        let frames = loaded(TK_CHAIN);
        let Some(FrameDef::Tk { relative, rotation }) =
            frames.frame_definition(FrameId(1_400_202)).unwrap()
        else {
            panic!("not a TK frame");
        };
        assert_eq!(relative, FrameId(1_400_201));
        // SPICE rotates the axes: [a1]ax1 [a2]ax2 [a3]ax3, with [a]ax a rotation of the
        // axes by a, takes vectors from the TK frame to the relative frame.
        let a = [
            10_f64.to_radians(),
            20_f64.to_radians(),
            30_f64.to_radians(),
        ];
        let expected = Rotation3::from_axis_angle(&Vector3::z_axis(), -a[0])
            * Rotation3::from_axis_angle(&Vector3::x_axis(), -a[1])
            * Rotation3::from_axis_angle(&Vector3::y_axis(), -a[2]);
        assert!((rotation.matrix() - expected.matrix()).abs().max() < 1e-15);
    }

    /// CK frames take their clock from `CK_<id>_SCLK`, else the ID divided by
    /// 1000.
    #[test]
    fn ck_clocks() {
        let frames = loaded(
            "\\begindata\nFRAME_C = -1000012000\nFRAME_-1000012000_NAME = 'C'\n\
             FRAME_-1000012000_CLASS = 3\nFRAME_-1000012000_CLASS_ID = -1000012000\n\
             CK_-1000012000_SCLK = -1000012\n",
        );
        assert_eq!(
            frames.frame_definition(FrameId(-1_000_012_000)).unwrap(),
            Some(FrameDef::Ck {
                class_id: -1_000_012_000,
                clock_id: ClockId(-1_000_012)
            })
        );
        assert_eq!(frames.ck_clock_id(-226_000).unwrap(), ClockId(-226));
    }

    /// CK frames take their spacecraft from `CK_<id>_SPK`, else the ID divided
    /// by 1000 if it is -1000 or less.
    #[test]
    fn ck_spacecraft() {
        let frames = loaded("\\begindata\nCK_-1000012000_SPK = -5\n");
        assert_eq!(frames.ck_spk_id(-1_000_012_000).unwrap(), -5);
        assert_eq!(frames.ck_spk_id(-226_000).unwrap(), -226);
        assert_eq!(frames.ck_spk_id(-1999).unwrap(), -1);
        assert!(frames.ck_spk_id(-999).is_err());
        assert!(frames.ck_spk_id(399_000).is_err());
    }

    /// Unsupported and malformed definitions are errors; an undefined ID is
    /// `None`.
    #[test]
    fn unsupported_and_malformed() {
        let frames = loaded(
            "\\begindata\n\
             FRAME_1400301_CLASS = 5\nFRAME_1400301_CLASS_ID = 1400301\n\
             FRAME_1400302_CLASS = 4\nFRAME_1400302_CLASS_ID = 1400302\n\
             FRAME_1400303_CLASS = 4\nFRAME_1400303_CLASS_ID = 1400303\n\
             FRAME_1400303_NAME = 'BOTH'\n\
             TKFRAME_1400303_RELATIVE = 'J2000'\nTKFRAME_BOTH_RELATIVE = 'J2000'\n\
             FRAME_1400304_CLASS = 4\nFRAME_1400304_CLASS_ID = 1400304\n\
             TKFRAME_1400304_RELATIVE = 'J2000'\nTKFRAME_1400304_SPEC = 'ANGLES'\n\
             TKFRAME_1400304_AXES = ( 1 2 4 )\nTKFRAME_1400304_ANGLES = ( 0 0 0 )\n\
             FRAME_1400305_CLASS = 3\n",
        );
        for id in [1_400_301, 1_400_302, 1_400_303, 1_400_304, 1_400_305, 2, 18] {
            assert!(frames.frame_definition(FrameId(id)).is_err(), "{id}");
        }
        assert_eq!(frames.frame_definition(FrameId(1_400_399)).unwrap(), None);
    }

    /// A missing UNITS means radians.
    #[test]
    fn angles_default_to_radians() {
        let frames = loaded(
            "\\begindata\nFRAME_1400401_CLASS = 4\nFRAME_1400401_CLASS_ID = 1400401\n\
             TKFRAME_1400401_RELATIVE = 'J2000'\nTKFRAME_1400401_SPEC = 'ANGLES'\n\
             TKFRAME_1400401_AXES = ( 1 2 3 )\nTKFRAME_1400401_ANGLES = ( 0.1 0 0 )\n",
        );
        let Some(FrameDef::Tk { rotation, .. }) =
            frames.frame_definition(FrameId(1_400_401)).unwrap()
        else {
            panic!("not a TK frame");
        };
        assert!((rotation.angle() - 0.1).abs() < 1e-15);
    }
}
