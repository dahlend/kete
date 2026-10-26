// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The IAU model of a spinning body: its pole and prime meridian over time.

use super::{DynFrameId, InstantFrame};
use crate::errors::{Error, KeteResult};
use crate::time::{TDB, Time};
use std::f64::consts::FRAC_PI_2;

/// The orientation of a spinning body over time, from its pole and prime meridian.
///
/// The model gives the right ascension (RA) and declination (DEC) of the pole of a
/// body, and the angle (W) of its prime meridian, in degrees:
///
/// ```text
/// RA  = RA0 + RA1 T + RA2 T^2 + sum_i a_i sin(theta_i)
/// DEC = DEC0 + DEC1 T + DEC2 T^2 + sum_i d_i cos(theta_i)
/// W   = W0 + W1 d + W2 d^2 + sum_i w_i sin(theta_i)
/// ```
///
/// `T` is in Julian centuries and `d` in days, from the epoch of the model, and each
/// nutation and precession angle `theta_i` is a polynomial in `T`. The rotation from
/// the reference frame to the body frame is `[W]3 [pi/2 - DEC]1 [pi/2 + RA]3`.
///
/// This is the model of the IAU reports on cartographic coordinates and rotational
/// elements, and of the text PCK kernels of SPICE. A body that spins at a constant
/// rate about a fixed pole is [`Self::uniform`].
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
    /// Epoch of the model.
    epoch: Time<TDB>,
    /// Frame the pole is given in.
    reference: DynFrameId,
}

impl BodyRotation {
    /// A model from the polynomials of the pole and the prime meridian, in degrees.
    ///
    /// - `ra` and `dec`: the right ascension and declination of the pole, as
    ///   polynomials in Julian centuries from `epoch`.
    /// - `pm`: the prime meridian angle, as a polynomial in days from `epoch`.
    /// - `reference`: the frame the pole is given in.
    #[must_use]
    pub fn new(
        ra: [f64; 3],
        dec: [f64; 3],
        pm: [f64; 3],
        epoch: Time<TDB>,
        reference: DynFrameId,
    ) -> Self {
        Self {
            ra,
            dec,
            pm,
            nut_ra: Vec::new(),
            nut_dec: Vec::new(),
            nut_pm: Vec::new(),
            angles: Vec::new(),
            epoch,
            reference,
        }
    }

    /// A body that spins at a constant rate about a fixed pole.
    ///
    /// The pole is at right ascension `pole_ra` and declination `pole_dec` in
    /// `reference`, and the prime meridian is at `w0` at `epoch` and turns at
    /// `w_rate` degrees per day. Angles are in degrees.
    #[must_use]
    pub fn uniform(
        pole_ra: f64,
        pole_dec: f64,
        w0: f64,
        w_rate: f64,
        epoch: Time<TDB>,
        reference: DynFrameId,
    ) -> Self {
        Self::new(
            [pole_ra, 0.0, 0.0],
            [pole_dec, 0.0, 0.0],
            [w0, w_rate, 0.0],
            epoch,
            reference,
        )
    }

    /// The model with the nutation and precession terms, in degrees.
    ///
    /// `ra`, `dec` and `pm` are the amplitudes `a_i`, `d_i` and `w_i`, and `angles`
    /// the polynomial in Julian centuries of each angle `theta_i`.
    ///
    /// # Errors
    /// [`Error::ValueError`] if there are more amplitudes than angles.
    pub fn with_nutation_precession(
        self,
        ra: Vec<f64>,
        dec: Vec<f64>,
        pm: Vec<f64>,
        angles: Vec<Vec<f64>>,
    ) -> KeteResult<Self> {
        let n_terms = ra.len().max(dec.len()).max(pm.len());
        if n_terms > angles.len() {
            return Err(Error::ValueError(format!(
                "{n_terms} nutation and precession terms need as many angles, found {}.",
                angles.len()
            )));
        }
        Ok(Self {
            nut_ra: ra,
            nut_dec: dec,
            nut_pm: pm,
            angles,
            ..self
        })
    }

    /// The frame the pole is given in.
    #[must_use]
    pub fn reference(&self) -> DynFrameId {
        self.reference
    }

    /// The body frame at `time`.
    ///
    /// The frame holds the rotation from the body frame to the reference frame of
    /// the model, and its rate. The model does not name the body frame, so the frame
    /// is [`DynFrameId::UNNAMED`]; a [`FrameSource`](super::FrameSource) names it.
    pub fn frame(&self, time: Time<TDB>) -> InstantFrame {
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
        InstantFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [
                FRAC_PI_2 + ra.0.to_radians(),
                FRAC_PI_2 - dec.0.to_radians(),
                pm.0.to_radians(),
            ],
            [ra.1.to_radians(), -dec.1.to_radians(), pm.1.to_radians()],
            DynFrameId::UNNAMED,
            self.reference,
        )
    }
}

/// Days in a Julian century.
const DAYS_PER_CENTURY: f64 = 36525.0;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    /// A uniform spin keeps its pole at the given RA and Dec, turns its prime
    /// meridian at the given rate, and has the derivative of its rotation as its rate.
    #[test]
    fn uniform_spin() {
        let epoch = Time::<TDB>::new(2_451_545.0);
        let model = BodyRotation::uniform(70.0, 50.0, 20.0, 540.0, epoch, DynFrameId::J2000);
        let (ra, dec) = (70_f64.to_radians(), 50_f64.to_radians());
        let pole = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        for dt in [0.0, 0.3, 17.25] {
            let frame = model.frame(epoch + dt);
            assert!((frame.rotation * Vector3::z() - pole).norm() < 1e-14);
        }

        // A quarter turn of the prime meridian about the pole.
        let quarter = 90.0 / 540.0;
        let x0 = model.frame(epoch).rotation * Vector3::x();
        let x1 = model.frame(epoch + quarter).rotation * Vector3::x();
        assert!((x0.cross(&x1) - pole).norm() < 1e-12);

        // Five-point stencil, as the spin is fast.
        let h = 1e-3;
        let at = |dt: f64| *model.frame(epoch + 0.2 + dt).rotation.matrix();
        let numeric = (at(-2.0 * h) - 8.0 * at(-h) + 8.0 * at(h) - at(2.0 * h)) / (12.0 * h);
        let rate = model.frame(epoch + 0.2).rotation_rate.unwrap();
        let err = (rate - numeric).abs().max();
        assert!(err < 1e-7, "rate error {err:e}");
    }

    /// More amplitudes than angles is an error.
    #[test]
    fn nutation_terms_need_angles() {
        let model = BodyRotation::uniform(0.0, 90.0, 0.0, 1.0, Time::j2000(), DynFrameId::J2000);
        assert!(
            model
                .clone()
                .with_nutation_precession(vec![0.1, 0.2], vec![], vec![], vec![vec![1.0, 2.0]])
                .is_err()
        );
        assert!(
            model
                .with_nutation_precession(vec![0.1], vec![], vec![0.3], vec![vec![1.0, 2.0]])
                .is_ok()
        );
    }
}
