//! Field of view definitions from instrument kernels (IK).
//!
//! An instrument kernel defines the field of view of instrument `<id>` by these
//! variables:
//!
//! - `INS<id>_FOV_FRAME`: the name of the frame the vectors are given in.
//! - `INS<id>_FOV_SHAPE`: `CIRCLE`, `ELLIPSE`, `RECTANGLE` or `POLYGON`.
//! - `INS<id>_BORESIGHT`: the boresight vector.
//! - `INS<id>_FOV_CLASS_SPEC`: `CORNERS` (the default) or `ANGLES`.
//!
//! A `CORNERS` definition gives the boundary vectors in
//! `INS<id>_FOV_BOUNDARY_CORNERS`, or in `INS<id>_FOV_BOUNDARY`. An `ANGLES`
//! definition gives `INS<id>_FOV_REF_VECTOR`, `INS<id>_FOV_REF_ANGLE`,
//! `INS<id>_FOV_CROSS_ANGLE` and `INS<id>_FOV_ANGLE_UNITS`, and the boundary
//! vectors follow from them:
//!
//! - The reference direction is the reference vector without its component
//!   along the boresight. The cross direction is the boresight crossed with the
//!   reference direction.
//! - A circle has one boundary vector, at the reference angle from the
//!   boresight toward the reference direction.
//! - An ellipse has two: one at the reference angle toward the reference
//!   direction, and one at the cross angle toward the cross direction.
//! - A rectangle has four corners, along the boresight plus or minus the
//!   tangent of each angle times its direction. The order is (+, +), (-, +),
//!   (-, -), (+, -) in (reference, cross).
//!
//! The boundary vectors of an `ANGLES` definition have the length of the
//! boresight. Each angle must be above 0 and below 90 degrees.

use super::TextKernelVars;
use super::fk::angle_unit;
use crate::frames::id_from_name;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::FrameId;
use nalgebra::Vector3;

/// The shape of a field of view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FovShape {
    /// A circle, with one boundary vector.
    Circle,

    /// An ellipse, with two boundary vectors.
    Ellipse,

    /// A rectangle, with four corners.
    Rectangle,

    /// A polygon, with three or more corners.
    Polygon,
}

/// The field of view of an instrument, in its own frame.
#[derive(Debug, Clone, PartialEq)]
pub struct InstrumentFov {
    /// The shape of the field of view.
    pub shape: FovShape,

    /// The name of the frame of the vectors, as the kernel gives it.
    pub frame_name: String,

    /// The SPICE frame ID of the frame of the vectors.
    pub frame: FrameId,

    /// The boresight vector, as the kernel gives it.
    pub boresight: Vector3<f64>,

    /// The boundary vectors: one for a circle, two for an ellipse, and the
    /// corners, in order, for a rectangle or a polygon.
    pub bounds: Vec<Vector3<f64>>,
}

/// The field of view of instrument `id` from the variables.
///
/// # Errors
/// [`Error::ValueError`] if a variable of the field of view is missing or
/// malformed, if the frame has no ID, if an angle is outside the range above 0
/// and below 90 degrees, if the reference vector is parallel to the boresight,
/// or if the number of boundary vectors does not fit the shape.
pub fn instrument_fov(vars: &TextKernelVars, id: i32) -> KeteResult<InstrumentFov> {
    let key = |name: &str| format!("INS{id}_{name}");
    let missing = |name: &str| Error::ValueError(format!("{} is missing.", key(name)));
    let string =
        |name: &str| -> KeteResult<&str> { vars.string(&key(name))?.ok_or_else(|| missing(name)) };
    let vector = |name: &str| -> KeteResult<Vector3<f64>> {
        match vars.numbers(&key(name))? {
            Some(&[x, y, z]) => Ok(Vector3::new(x, y, z)),
            Some(_) => Err(Error::ValueError(format!(
                "{} must hold 3 numbers.",
                key(name)
            ))),
            None => Err(missing(name)),
        }
    };

    let frame_name = string("FOV_FRAME")?.to_string();
    let frame = id_from_name(vars, &frame_name)?;
    let shape = match string("FOV_SHAPE")?.trim().to_uppercase().as_str() {
        "CIRCLE" => FovShape::Circle,
        "ELLIPSE" => FovShape::Ellipse,
        "RECTANGLE" => FovShape::Rectangle,
        "POLYGON" => FovShape::Polygon,
        other => {
            return Err(Error::ValueError(format!(
                "{} {other} is not CIRCLE, ELLIPSE, RECTANGLE or POLYGON.",
                key("FOV_SHAPE")
            )));
        }
    };
    let boresight = vector("BORESIGHT")?;
    let spec = vars
        .string(&key("FOV_CLASS_SPEC"))?
        .unwrap_or("CORNERS")
        .trim()
        .to_uppercase();

    let bounds = match spec.as_str() {
        "CORNERS" => {
            let values = match vars.numbers(&key("FOV_BOUNDARY_CORNERS"))? {
                Some(values) => values,
                None => vars
                    .numbers(&key("FOV_BOUNDARY"))?
                    .ok_or_else(|| missing("FOV_BOUNDARY_CORNERS"))?,
            };
            if values.len() % 3 != 0 {
                return Err(Error::ValueError(format!(
                    "The boundary of instrument {id} must hold 3 numbers per vector."
                )));
            }
            values
                .chunks(3)
                .map(|v| Vector3::new(v[0], v[1], v[2]))
                .collect()
        }
        "ANGLES" => angle_bounds(vars, id, shape, &boresight, vector("FOV_REF_VECTOR")?)?,
        other => {
            return Err(Error::ValueError(format!(
                "{} {other} is not CORNERS or ANGLES.",
                key("FOV_CLASS_SPEC")
            )));
        }
    };

    let fits = match shape {
        FovShape::Circle => bounds.len() == 1,
        FovShape::Ellipse => bounds.len() == 2,
        FovShape::Rectangle => bounds.len() == 4,
        FovShape::Polygon => bounds.len() >= 3,
    };
    if !fits {
        return Err(Error::ValueError(format!(
            "Instrument {id} has {} boundary vectors, which do not fit its shape {shape:?}.",
            bounds.len()
        )));
    }
    Ok(InstrumentFov {
        shape,
        frame_name,
        frame,
        boresight,
        bounds,
    })
}

/// The boundary vectors of an `ANGLES` definition; see the module docs.
///
/// # Errors
/// [`Error::ValueError`] if an angle or the units are missing or malformed, if
/// an angle is outside the range above 0 and below 90 degrees, if the reference
/// vector is parallel to the boresight, or if the shape is a polygon.
fn angle_bounds(
    vars: &TextKernelVars,
    id: i32,
    shape: FovShape,
    boresight: &Vector3<f64>,
    reference: Vector3<f64>,
) -> KeteResult<Vec<Vector3<f64>>> {
    let key = |name: &str| format!("INS{id}_{name}");
    let units_name = key("FOV_ANGLE_UNITS");
    let units = vars
        .string(&units_name)?
        .ok_or_else(|| Error::ValueError(format!("{units_name} is missing.")))?;
    let scale = angle_unit(units)
        .ok_or_else(|| Error::ValueError(format!("{units_name} {units} is not an angle unit.")))?;
    let angle = |name: &str| -> KeteResult<f64> {
        let value = match vars.numbers(&key(name))? {
            Some(&[value]) => value * scale,
            _ => {
                return Err(Error::ValueError(format!(
                    "{} must hold one number.",
                    key(name)
                )));
            }
        };
        if value > 0.0 && value < std::f64::consts::FRAC_PI_2 {
            Ok(value)
        } else {
            Err(Error::ValueError(format!(
                "{} must be above 0 and below 90 degrees.",
                key(name)
            )))
        }
    };

    let length = boresight.norm();
    let along = boresight
        .try_normalize(0.0)
        .ok_or_else(|| Error::ValueError(format!("The boresight of instrument {id} is zero.")))?;
    let ref_dir = (reference - along * reference.dot(&along))
        .try_normalize(1e-12)
        .ok_or_else(|| {
            Error::ValueError(format!(
                "The reference vector of instrument {id} is parallel to its boresight."
            ))
        })?;
    let cross_dir = along.cross(&ref_dir);
    let toward =
        |dir: &Vector3<f64>, angle: f64| (along * angle.cos() + dir * angle.sin()) * length;

    match shape {
        FovShape::Circle => Ok(vec![toward(&ref_dir, angle("FOV_REF_ANGLE")?)]),
        FovShape::Ellipse => Ok(vec![
            toward(&ref_dir, angle("FOV_REF_ANGLE")?),
            toward(&cross_dir, angle("FOV_CROSS_ANGLE")?),
        ]),
        FovShape::Rectangle => {
            let ref_tan = angle("FOV_REF_ANGLE")?.tan();
            let cross_tan = angle("FOV_CROSS_ANGLE")?.tan();
            Ok([(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
                .iter()
                .map(|(r, c)| {
                    (along + ref_dir * (r * ref_tan) + cross_dir * (c * cross_tan)).normalize()
                        * length
                })
                .collect())
        }
        FovShape::Polygon => Err(Error::ValueError(format!(
            "Instrument {id} is a POLYGON, which an ANGLES definition cannot give."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two `ANGLES` definitions with a boresight and reference vector that are
    /// neither unit nor perpendicular, and CSPICE `getfov` for them.
    const ANGLES_IK: &str = r"KPL/IK
\begindata
INS-9001_FOV_FRAME = 'J2000'
INS-9001_FOV_SHAPE = 'RECTANGLE'
INS-9001_BORESIGHT = ( 0.1 -0.2 2.0 )
INS-9001_FOV_CLASS_SPEC = 'ANGLES'
INS-9001_FOV_REF_VECTOR = ( 1.0 0.3 0.2 )
INS-9001_FOV_REF_ANGLE = 3.0
INS-9001_FOV_CROSS_ANGLE = 1.5
INS-9001_FOV_ANGLE_UNITS = 'DEGREES'
INS-9002_FOV_FRAME = 'J2000'
INS-9002_FOV_SHAPE = 'ELLIPSE'
INS-9002_BORESIGHT = ( 0.1 -0.2 2.0 )
INS-9002_FOV_CLASS_SPEC = 'ANGLES'
INS-9002_FOV_REF_VECTOR = ( 1.0 0.3 0.2 )
INS-9002_FOV_REF_ANGLE = 180.0
INS-9002_FOV_CROSS_ANGLE = 90.0
INS-9002_FOV_ANGLE_UNITS = 'ARCMINUTES'
\begintext
";

    /// CSPICE N0067 `getfov` boundary vectors of the definitions of
    /// [`ANGLES_IK`].
    const ANGLES_EXPECTED: [(i32, &[[f64; 3]]); 2] = [
        (
            -9001,
            &[
                [
                    1.838_574_590_440_703_6e-1,
                    -1.173_404_441_606_791_3e-1,
                    2.000_606_821_671_372_6,
                ],
                [
                    -1.636_458_167_780_042_7e-2,
                    -1.824_650_869_616_271e-1,
                    2.004_105_459_427_371_3,
                ],
                [
                    1.580_019_419_441_31e-2,
                    -2.819_748_623_162_877e-1,
                    1.992_546_243_098_295,
                ],
                [
                    2.160_222_349_162_838_6e-1,
                    -2.168_502_195_153_397e-1,
                    1.989_047_605_342_295_8,
                ],
            ],
        ),
        (
            -9002,
            &[
                [
                    2.000_081_971_615_901_8e-1,
                    -1.671_524_539_995_84e-1,
                    1.995_509_152_619_974_2,
                ],
                [
                    8.386_128_907_386_743e-2,
                    -1.501_083_431_530_757_8e-1,
                    2.005_102_184_306_502_4,
                ],
            ],
        ),
    ];

    fn vars(text: &str) -> TextKernelVars {
        let mut vars = TextKernelVars::default();
        vars.load_text(text).unwrap();
        vars
    }

    /// `ANGLES` definitions give the boundary vectors of CSPICE `getfov`.
    #[test]
    fn angles_match_spice() {
        let vars = vars(ANGLES_IK);
        for (id, expected) in ANGLES_EXPECTED {
            let fov = instrument_fov(&vars, id).unwrap();
            assert_eq!(fov.frame, FrameId::J2000);
            assert_eq!(fov.bounds.len(), expected.len());
            for (got, want) in fov.bounds.iter().zip(expected) {
                let err = (got - Vector3::from(*want)).abs().max();
                assert!(err < 1e-15, "{id}: {err:e}");
            }
        }
    }

    /// A `CORNERS` polygon keeps its corners as given, with `FOV_BOUNDARY` as
    /// the variable name and no `FOV_CLASS_SPEC`.
    #[test]
    fn corners_as_given() {
        let vars = vars(
            "\\begindata\nINS-9010_FOV_FRAME = 'ECLIPJ2000'\nINS-9010_FOV_SHAPE = 'polygon'\n\
             INS-9010_BORESIGHT = ( 0 0 3 )\n\
             INS-9010_FOV_BOUNDARY = ( 1 0 5  0 1 5  -1 0 5 )\n",
        );
        let fov = instrument_fov(&vars, -9010).unwrap();
        assert_eq!(fov.shape, FovShape::Polygon);
        assert_eq!(fov.frame, FrameId::ECLIPJ2000);
        assert_eq!(fov.bounds[1], Vector3::new(0.0, 1.0, 5.0));
    }

    /// Malformed definitions are errors.
    #[test]
    fn malformed_definitions() {
        let base = "INS-9020_FOV_FRAME = 'J2000'\nINS-9020_BORESIGHT = ( 0 0 1 )\n";
        for body in [
            // An angle of 90 degrees.
            "INS-9020_FOV_SHAPE = 'CIRCLE'\nINS-9020_FOV_CLASS_SPEC = 'ANGLES'\n\
             INS-9020_FOV_REF_VECTOR = ( 1 0 0 )\nINS-9020_FOV_REF_ANGLE = 90\n\
             INS-9020_FOV_ANGLE_UNITS = 'DEGREES'\n",
            // A reference vector parallel to the boresight.
            "INS-9020_FOV_SHAPE = 'CIRCLE'\nINS-9020_FOV_CLASS_SPEC = 'ANGLES'\n\
             INS-9020_FOV_REF_VECTOR = ( 0 0 2 )\nINS-9020_FOV_REF_ANGLE = 1\n\
             INS-9020_FOV_ANGLE_UNITS = 'DEGREES'\n",
            // A rectangle with 3 corners.
            "INS-9020_FOV_SHAPE = 'RECTANGLE'\n\
             INS-9020_FOV_BOUNDARY_CORNERS = ( 1 0 5  0 1 5  -1 0 5 )\n",
            // An unknown shape.
            "INS-9020_FOV_SHAPE = 'SQUARE'\nINS-9020_FOV_BOUNDARY_CORNERS = ( 1 0 5 )\n",
        ] {
            let vars = vars(&format!("\\begindata\n{base}{body}"));
            assert!(instrument_fov(&vars, -9020).is_err(), "{body}");
        }
        assert!(instrument_fov(&vars("\\begindata\nA = 1\n"), -9020).is_err());
    }
}
