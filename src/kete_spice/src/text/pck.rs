//! Body orientation from text PCK constants.
//!
//! A text PCK gives the right ascension (RA) and declination (DEC) of the pole
//! of a body, and the angle (W) of its prime meridian, in degrees:
//!
//! ```text
//! RA  = RA0 + RA1 T + RA2 T^2 + sum_i a_i sin(theta_i)
//! DEC = DEC0 + DEC1 T + DEC2 T^2 + sum_i d_i cos(theta_i)
//! W   = W0 + W1 d + W2 d^2 + sum_i w_i sin(theta_i)
//! ```
//!
//! `T` is in Julian centuries and `d` in days, from the epoch of the
//! constants. The rotation from the reference frame to the body frame is
//! `[W]3 [pi/2 - DEC]1 [pi/2 + RA]3`.
//!
//! These variables give the model of body `<id>`:
//!
//! - `BODY<id>_POLE_RA`, `_POLE_DEC` and `_PM`: the polynomials.
//! - `BODY<id>_NUT_PREC_RA`, `_NUT_PREC_DEC` and `_NUT_PREC_PM`: the `a_i`,
//!   `d_i` and `w_i`.
//! - `BODY<sys>_NUT_PREC_ANGLES`: the coefficients of each nutation and
//!   precession angle `theta_i`, a polynomial in `T`.
//! - `BODY<sys>_MAX_PHASE_DEGREE`: the degree of those polynomials (default
//!   1).
//! - `BODY<sys>_CONSTANTS_JED_EPOCH`: the epoch (default J2000).
//! - `BODY<sys>_CONSTANTS_REF_FRAME`: the reference frame (default J2000).
//!
//! The last two may be spelled `CONSTS` instead of `CONSTANTS`, but not both.
//! `sys` is the planetary system: the ID divided by 100 for a 3 digit ID, the
//! ID divided by 10000 for a 5 digit ID, and the body itself for any other ID.
use super::TextKernelVars;
use kete_core::errors::{Error, KeteResult};
use kete_core::frames::{FrameId, NonInertialFrame};
use kete_core::time::{TDB, Time};
use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

/// Days in a Julian century.
const DAYS_PER_CENTURY: f64 = 36525.0;

/// The orientation model of one body from text PCK constants.
#[derive(Debug, Clone, PartialEq)]
pub struct BodyRotation {
    /// Pole right ascension polynomial in centuries, degrees.
    ra: [f64; 3],
    /// Pole declination polynomial in centuries, degrees.
    dec: [f64; 3],
    /// Prime meridian polynomial in days, degrees.
    pm: [f64; 3],
    /// Nutation and precession amplitudes of RA, DEC and W, degrees.
    nut_ra: Vec<f64>,
    nut_dec: Vec<f64>,
    nut_pm: Vec<f64>,
    /// Polynomial coefficients in centuries of each nutation and precession
    /// angle, degrees.
    angles: Vec<Vec<f64>>,
    /// Epoch of the constants.
    epoch: Time<TDB>,
    /// Frame the pole is given in.
    reference: FrameId,
}

impl BodyRotation {
    /// The body frame at `time`.
    ///
    /// The frame holds the rotation from the body frame to the reference frame
    /// of the constants, and its rate.
    pub fn frame(&self, time: Time<TDB>) -> NonInertialFrame {
        let days = (time - self.epoch).elapsed;
        let t = days / DAYS_PER_CENTURY;

        // Each angle and its rate in degrees per day.
        let poly = |c: &[f64], x: f64| c.iter().rev().fold(0.0, |acc, c| acc * x + c);
        let poly_rate = |c: &[f64], x: f64| {
            c.iter()
                .enumerate()
                .skip(1)
                .rev()
                .fold(0.0, |acc, (k, c)| acc * x + c * k as f64)
        };
        let mut ra = (poly(&self.ra, t), poly_rate(&self.ra, t) / DAYS_PER_CENTURY);
        let mut dec = (
            poly(&self.dec, t),
            poly_rate(&self.dec, t) / DAYS_PER_CENTURY,
        );
        let mut pm = (poly(&self.pm, days), poly_rate(&self.pm, days));
        for (idx, coefs) in self.angles.iter().enumerate() {
            let theta = poly(coefs, t).to_radians();
            let theta_rate = (poly_rate(coefs, t) / DAYS_PER_CENTURY).to_radians();
            let (sin, cos) = theta.sin_cos();
            if let Some(a) = self.nut_ra.get(idx) {
                ra.0 += a * sin;
                ra.1 += a * cos * theta_rate;
            }
            if let Some(d) = self.nut_dec.get(idx) {
                dec.0 += d * cos;
                dec.1 -= d * sin * theta_rate;
            }
            if let Some(w) = self.nut_pm.get(idx) {
                pm.0 += w * sin;
                pm.1 += w * cos * theta_rate;
            }
        }
        NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [
                FRAC_PI_2 + ra.0.to_radians(),
                FRAC_PI_2 - dec.0.to_radians(),
                pm.0.to_radians(),
            ],
            [ra.1.to_radians(), -dec.1.to_radians(), pm.1.to_radians()],
            self.reference,
        )
    }

    /// The model of body `id` from the variables.
    ///
    /// The model is `None` if the variables hold no `BODY<id>_POLE_RA`.
    ///
    /// # Errors
    /// [`Error::ValueError`] if a variable of the model is missing or has the
    /// wrong type or size, if the phase degree is not 1, 2 or 3, or if both
    /// spellings of a system constant are set (see [`system_constant`]).
    fn from_vars(vars: &TextKernelVars, id: i32) -> KeteResult<Option<Self>> {
        let body = |key: &str| format!("BODY{id}_{key}");
        let Some(ra) = vars.numbers(&body("POLE_RA"))? else {
            return Ok(None);
        };
        let polynomial = |name: String, values: Option<&[f64]>| -> KeteResult<[f64; 3]> {
            let values = values.ok_or_else(|| Error::ValueError(format!("{name} is missing.")))?;
            if values.is_empty() || values.len() > 3 {
                return Err(Error::ValueError(format!(
                    "{name} must hold 1 to 3 numbers, found {}.",
                    values.len()
                )));
            }
            let mut out = [0.0; 3];
            out[..values.len()].copy_from_slice(values);
            Ok(out)
        };
        let ra = polynomial(body("POLE_RA"), Some(ra))?;
        let dec = polynomial(body("POLE_DEC"), vars.numbers(&body("POLE_DEC"))?)?;
        let pm = polynomial(body("PM"), vars.numbers(&body("PM"))?)?;
        let terms = |key: &str| -> KeteResult<Vec<f64>> {
            Ok(vars.numbers(&body(key))?.unwrap_or(&[]).to_vec())
        };
        let nut_ra = terms("NUT_PREC_RA")?;
        let nut_dec = terms("NUT_PREC_DEC")?;
        let nut_pm = terms("NUT_PREC_PM")?;

        let sys = system(id);
        let system_var = |key: &str| format!("BODY{sys}_{key}");
        let n_terms = nut_ra.len().max(nut_dec.len()).max(nut_pm.len());
        let angles = if n_terms == 0 {
            Vec::new()
        } else {
            let degree = vars.integer(&system_var("MAX_PHASE_DEGREE"))?.unwrap_or(1);
            let per_angle = usize::try_from(degree)
                .ok()
                .filter(|d| (1..=3).contains(d))
                .ok_or_else(|| {
                    Error::ValueError(format!(
                        "{} must be 1, 2 or 3, found {degree}.",
                        system_var("MAX_PHASE_DEGREE")
                    ))
                })?
                + 1;
            let name = system_var("NUT_PREC_ANGLES");
            let values = vars.numbers(&name)?.ok_or_else(|| {
                Error::ValueError(format!(
                    "Body {id} has nutation and precession terms, but {name} is missing."
                ))
            })?;
            if values.len() % per_angle != 0 || values.len() / per_angle < n_terms {
                return Err(Error::ValueError(format!(
                    "{name} must hold {per_angle} numbers for each of at least {n_terms} \
                     angles, found {} numbers.",
                    values.len()
                )));
            }
            values.chunks(per_angle).map(<[f64]>::to_vec).collect()
        };

        let epoch = system_constant(vars, sys, "JED_EPOCH")?
            .map(|name| match vars.numbers(&name)? {
                Some(&[jd]) => Ok(Time::<TDB>::new(jd)),
                _ => Err(Error::ValueError(format!("{name} must hold one number."))),
            })
            .transpose()?
            .unwrap_or_else(Time::j2000);
        let reference = system_constant(vars, sys, "REF_FRAME")?
            .map(|name| {
                vars.integer(&name)?
                    .map(FrameId)
                    .ok_or_else(|| Error::ValueError(format!("{name} must hold one integer.")))
            })
            .transpose()?
            .unwrap_or(FrameId::J2000);

        Ok(Some(Self {
            ra,
            dec,
            pm,
            nut_ra,
            nut_dec,
            nut_pm,
            angles,
            epoch,
            reference,
        }))
    }
}

/// The planetary system of body `id`.
///
/// The variables of the system hold the nutation and precession angles, the
/// epoch and the reference frame. The system is the barycenter of a planet or
/// satellite: ID / 100 for a 3 digit ID, and ID / 10000 for a 5 digit ID. It is
/// the body itself for any other ID.
fn system(id: i32) -> i32 {
    match id {
        100..=999 => id / 100,
        10_000..=99_999 => id / 10_000,
        _ => id,
    }
}

/// The name of the variable `BODY<sys>_CONSTANTS_<key>` or
/// `BODY<sys>_CONSTS_<key>` that is set, or `None` if neither is.
///
/// # Errors
/// [`Error::ValueError`] if both are set.
fn system_constant(vars: &TextKernelVars, sys: i32, key: &str) -> KeteResult<Option<String>> {
    let long = format!("BODY{sys}_CONSTANTS_{key}");
    let short = format!("BODY{sys}_CONSTS_{key}");
    match (vars.get(&long).is_some(), vars.get(&short).is_some()) {
        (true, true) => Err(Error::ValueError(format!(
            "Both {long} and {short} are set; only one may be."
        ))),
        (true, false) => Ok(Some(long)),
        (false, true) => Ok(Some(short)),
        (false, false) => Ok(None),
    }
}

/// The orientation models of all bodies that the variables give a pole for.
///
/// There is one model for each `BODY<id>_POLE_RA`. A malformed model keeps its
/// error. The error comes when the model is used.
pub(crate) fn body_rotations(vars: &TextKernelVars) -> HashMap<i32, KeteResult<BodyRotation>> {
    vars.names()
        .filter_map(|name| {
            name.strip_prefix("BODY")?
                .strip_suffix("_POLE_RA")?
                .parse::<i32>()
                .ok()
        })
        .filter_map(|id| {
            BodyRotation::from_vars(vars, id)
                .transpose()
                .map(|m| (id, m))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model with nutation terms and a degree 2 phase gives a rate that is
    /// the derivative of its rotation.
    #[test]
    fn rate_is_the_derivative() {
        let mut vars = TextKernelVars::default();
        vars.load_text(
            "\\begindata\n\
             BODY401_POLE_RA = ( 10 2 3 )\nBODY401_POLE_DEC = ( 60 -1 0.5 )\n\
             BODY401_PM = ( 30 100 1e-6 )\n\
             BODY401_NUT_PREC_RA = ( 0.5 0.2 )\nBODY401_NUT_PREC_DEC = ( 0.3 0.1 )\n\
             BODY401_NUT_PREC_PM = ( 0.7 -0.4 )\n\
             BODY4_NUT_PREC_ANGLES = ( 40 1000 5 120 -300 7 )\n\
             BODY4_MAX_PHASE_DEGREE = 2\n",
        )
        .unwrap();
        let model = body_rotations(&vars).remove(&401).unwrap().unwrap();
        // Offsets are added to one Time, which keeps them exact; a float JD would not.
        let time = Time::<TDB>::new(2_451_545.0) + 0.37 * DAYS_PER_CENTURY;
        // Five-point stencil.
        let h = 1e-3;
        let at = |dt: f64| *model.frame(time + dt).rotation.matrix();
        let numeric = (at(-2.0 * h) - 8.0 * at(-h) + 8.0 * at(h) - at(2.0 * h)) / (12.0 * h);
        let rate = model.frame(time).rotation_rate.unwrap();
        let err = (rate - numeric).abs().max();
        assert!(err < 1e-8, "rate error {err:e}");
    }

    /// Missing or malformed constants are errors; a body without a pole has no
    /// model.
    #[test]
    fn malformed_constants() {
        let mut vars = TextKernelVars::default();
        vars.load_text(
            "\\begindata\n\
             BODY2000001_POLE_RA = ( 10 )\nBODY2000001_POLE_DEC = ( 60 )\n\
             BODY2000002_POLE_RA = ( 10 )\nBODY2000002_POLE_DEC = ( 60 )\n\
             BODY2000002_PM = ( 1 2 3 4 )\n\
             BODY2000003_POLE_RA = ( 10 )\nBODY2000003_POLE_DEC = ( 60 )\n\
             BODY2000003_PM = ( 1 )\nBODY2000003_NUT_PREC_RA = ( 1 )\n\
             BODY2000004_PM = ( 1 )\n",
        )
        .unwrap();
        let models = body_rotations(&vars);
        for id in [2_000_001, 2_000_002, 2_000_003] {
            assert!(models[&id].is_err(), "{id}");
        }
        assert!(!models.contains_key(&2_000_004));
    }

    /// A 5 digit satellite ID takes its system variables from its barycenter,
    /// ID / 10000, as a 3 digit ID does from ID / 100.
    #[test]
    fn extended_satellite_ids_use_their_system() {
        assert_eq!(system(65_002), 6);
        assert_eq!(system(401), 4);
        assert_eq!(system(2_000_001), 2_000_001);
        let mut vars = TextKernelVars::default();
        vars.load_text(
            "\\begindata\n\
             BODY65002_POLE_RA = ( 40 )\nBODY65002_POLE_DEC = ( 80 )\nBODY65002_PM = ( 10 100 )\n\
             BODY65002_NUT_PREC_RA = ( 0.1 )\n\
             BODY6_NUT_PREC_ANGLES = ( 350 75000 )\nBODY6_CONSTANTS_REF_FRAME = 17\n",
        )
        .unwrap();
        let model = body_rotations(&vars).remove(&65_002).unwrap().unwrap();
        assert_eq!(model.reference, FrameId::ECLIPJ2000);
        assert_eq!(model.angles, vec![vec![350.0, 75_000.0]]);
    }

    /// Both spellings of a system constant is an error, as in CSPICE `bodeul`.
    #[test]
    fn competing_system_constants() {
        let mut vars = TextKernelVars::default();
        vars.load_text(
            "\\begindata\n\
             BODY2000005_POLE_RA = ( 10 )\nBODY2000005_POLE_DEC = ( 60 )\nBODY2000005_PM = ( 1 )\n\
             BODY2000005_CONSTANTS_JED_EPOCH = 2451545\nBODY2000005_CONSTS_JED_EPOCH = 2451545\n\
             BODY2000006_POLE_RA = ( 10 )\nBODY2000006_POLE_DEC = ( 60 )\nBODY2000006_PM = ( 1 )\n\
             BODY2000006_CONSTANTS_REF_FRAME = 1\nBODY2000006_CONSTS_REF_FRAME = 17\n\
             BODY2000007_POLE_RA = ( 10 )\nBODY2000007_POLE_DEC = ( 60 )\nBODY2000007_PM = ( 1 )\n\
             BODY2000007_CONSTS_REF_FRAME = 17\n",
        )
        .unwrap();
        let models = body_rotations(&vars);
        assert!(
            models[&2_000_005]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("CONSTS_JED_EPOCH")
        );
        assert!(
            models[&2_000_006]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("CONSTS_REF_FRAME")
        );
        assert_eq!(
            models[&2_000_007].as_ref().unwrap().reference,
            FrameId::ECLIPJ2000
        );
    }
}
