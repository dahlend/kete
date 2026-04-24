//! Observation types, predicted measurements, and geometric partial derivatives.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::frames::{Equatorial, SSB};
use kete_core::prelude::{Error, KeteResult, State};
use nalgebra::{DVector, Matrix2x3, Matrix3x1, RowVector6};

/// A single astrometric or radar observation.
///
/// Each variant carries the observer geometry, measured values, and
/// uncertainties. The observation epoch is taken from `observer.epoch`.
#[derive(Debug, Clone)]
pub enum AstrometricObservation {
    /// Optical astrometry: RA and Dec on the sky.
    Optical {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial, SSB>,
        /// Right ascension (radians).
        ra: f64,
        /// Declination (radians).
        dec: f64,
        /// 1-sigma RA uncertainty (radians, includes cos(dec) factor).
        sigma_ra: f64,
        /// 1-sigma Dec uncertainty (radians).
        sigma_dec: f64,
        /// 1-sigma timing uncertainty (seconds). When non-zero, the
        /// along-track positional uncertainty is inflated by
        /// `time_sigma * apparent_speed`. Set to 0 to disable.
        time_sigma: f64,
    },

    /// Radar range measurement.
    RadarRange {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial, SSB>,
        /// Measured range (AU).
        range: f64,
        /// 1-sigma range uncertainty (AU).
        sigma_range: f64,
    },

    /// Radar range-rate (Doppler) measurement.
    RadarRate {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial, SSB>,
        /// Measured range-rate (AU/day, positive = receding).
        range_rate: f64,
        /// 1-sigma range-rate uncertainty (AU/day).
        sigma_range_rate: f64,
    },
}

impl AstrometricObservation {
    /// Reference to the observer state (carries the observation epoch).
    pub fn observer(&self) -> &State<Equatorial, SSB> {
        match self {
            Self::Optical { observer, .. }
            | Self::RadarRange { observer, .. }
            | Self::RadarRate { observer, .. } => observer,
        }
    }

    /// Observation epoch (shorthand for `self.observer().epoch`).
    pub fn epoch(&self) -> kete_core::time::Time<kete_core::time::TDB> {
        self.observer().epoch
    }

    /// Extract RA, Dec, and observer state from an Optical observation.
    ///
    /// # Errors
    /// Returns an error if the observation is not Optical.
    pub fn as_optical(&self) -> KeteResult<(f64, f64, &State<Equatorial, SSB>)> {
        match self {
            Self::Optical {
                observer, ra, dec, ..
            } => Ok((*ra, *dec, observer)),
            Self::RadarRange { .. } | Self::RadarRate { .. } => {
                Err(Error::ValueError("Expected an Optical observation".into()))
            }
        }
    }

    /// Returns true if this is a radar observation (range or range-rate).
    #[must_use]
    pub fn is_radar(&self) -> bool {
        matches!(self, Self::RadarRange { .. } | Self::RadarRate { .. })
    }

    /// Number of measurement components (2 for optical, 1 for radar).
    #[must_use]
    pub fn measurement_dim(&self) -> usize {
        match self {
            Self::Optical { .. } => 2,
            Self::RadarRange { .. } | Self::RadarRate { .. } => 1,
        }
    }

    /// Diagonal weight vector (1 / sigma^2 for each measurement component).
    ///
    /// This does not include the timing correction; use [`Self::weight_matrix`]
    /// for the full weight matrix with timing.
    #[must_use]
    pub fn weights(&self) -> DVector<f64> {
        match self {
            Self::Optical {
                sigma_ra,
                sigma_dec,
                ..
            } => DVector::from_column_slice(&[
                1.0 / (sigma_ra * sigma_ra),
                1.0 / (sigma_dec * sigma_dec),
            ]),
            Self::RadarRange { sigma_range, .. } => {
                DVector::from_column_slice(&[1.0 / (sigma_range * sigma_range)])
            }
            Self::RadarRate {
                sigma_range_rate, ..
            } => DVector::from_column_slice(&[1.0 / (sigma_range_rate * sigma_range_rate)]),
        }
    }

    /// Full weight matrix (inverse noise covariance) for this observation.
    ///
    /// For optical observations with a non-zero `time_sigma`, the along-track
    /// uncertainty is inflated by the timing displacement
    /// `t = time_sigma * apparent_velocity`, producing a non-diagonal 2x2
    /// matrix via Sherman-Morrison:
    ///
    ///   `W_new` = W - (Wu)(Wu)^T / (1 + u^T W u)
    ///
    /// where `u = (motion_ra_rad_per_day * time_sigma_days,
    ///             motion_dec_rad_per_day * time_sigma_days)` is the timing
    /// displacement in radians.
    ///
    /// When `time_sigma == 0` the result is `diag(weights())`.
    /// For radar observations, `motion_ra` and `motion_dec` are ignored.
    #[must_use]
    pub fn weight_matrix(
        &self,
        motion_ra_rad_per_day: f64,
        motion_dec_rad_per_day: f64,
    ) -> nalgebra::DMatrix<f64> {
        match self {
            Self::Optical {
                sigma_ra,
                sigma_dec,
                time_sigma,
                ..
            } => {
                let w_ra = 1.0 / (sigma_ra * sigma_ra);
                let w_dec = 1.0 / (sigma_dec * sigma_dec);
                if *time_sigma == 0.0 {
                    return nalgebra::DMatrix::from_diagonal(&DVector::from_column_slice(&[
                        w_ra, w_dec,
                    ]));
                }
                // Timing displacement vector u = time_sigma_days * motion.
                let t_days = time_sigma / 86_400.0;
                let u_ra = t_days * motion_ra_rad_per_day;
                let u_dec = t_days * motion_dec_rad_per_day;
                // Wu = W * u (diagonal case).
                let wu_ra = w_ra * u_ra;
                let wu_dec = w_dec * u_dec;
                // Denominator 1 + u^T W u is always >= 1.
                let denom = 1.0 + u_ra * wu_ra + u_dec * wu_dec;
                nalgebra::DMatrix::from_row_slice(
                    2,
                    2,
                    &[
                        w_ra - wu_ra * wu_ra / denom,
                        -wu_ra * wu_dec / denom,
                        -wu_ra * wu_dec / denom,
                        w_dec - wu_dec * wu_dec / denom,
                    ],
                )
            }
            Self::RadarRange { sigma_range, .. } => {
                nalgebra::DMatrix::from_column_slice(1, 1, &[1.0 / (sigma_range * sigma_range)])
            }
            Self::RadarRate {
                sigma_range_rate, ..
            } => nalgebra::DMatrix::from_column_slice(
                1,
                1,
                &[1.0 / (sigma_range_rate * sigma_range_rate)],
            ),
        }
    }
}

impl AstrometricObservation {
    /// Compute the predicted measurement and residual (observed - computed).
    ///
    /// The input `obj_state` is the object at the observation epoch (before
    /// light-time correction). Light-time correction is applied internally.
    ///
    /// Returns `(residual, predicted)` where both are [`DVector`].
    ///
    /// # Errors
    /// Fails if two-body propagation for light-time correction fails.
    pub fn residual(
        &self,
        obj_state: State<Equatorial>,
    ) -> KeteResult<(DVector<f64>, DVector<f64>)> {
        let obs = self.observer();
        let spk = kete_spice::prelude::LOADED_SPK.try_read()?;
        // Convert object to heliocentric for two-body light-time correction.
        let obj_sun = spk.try_to_sun(obj_state)?;
        // Convert observer to heliocentric so the light-travel offset is
        // computed from the correct observer position.
        let obs_helio_sun = spk.try_to_sun(obs.clone().into())?;
        // Apply light-time correction (heliocentric in, heliocentric out).
        let obj_lt_helio = kete_core::kepler::light_time_correct(&obj_sun, &obs_helio_sun.pos)?;
        // Convert the light-time-corrected object state back to SSB so that
        // residual_predicted_from_corrected sees consistent SSB positions.
        let obj_lt_ssb = spk.try_to_ssb(obj_lt_helio.into())?;
        Ok(self.residual_predicted_from_corrected(&obj_lt_ssb))
    }

    /// Compute residual from an already light-time-corrected object state.
    ///
    /// This avoids a redundant two-body propagation when the caller has
    /// already applied `light_time_correct`.
    #[must_use]
    pub fn residual_from_corrected(&self, obj_lt: &State<Equatorial, SSB>) -> DVector<f64> {
        self.residual_predicted_from_corrected(obj_lt).0
    }

    /// Core residual computation from a light-time-corrected state.
    ///
    /// Returns `(residual, predicted)`.
    fn residual_predicted_from_corrected(
        &self,
        obj_lt: &State<Equatorial, SSB>,
    ) -> (DVector<f64>, DVector<f64>) {
        let obs = self.observer();

        match self {
            Self::Optical { ra, dec, .. } => {
                let (ra_pred, dec_pred) = (obj_lt.pos - obs.pos).to_ra_dec();
                // Wrap RA residual to [-pi, pi]
                let mut d_ra = ra - ra_pred;
                if d_ra > std::f64::consts::PI {
                    d_ra -= 2.0 * std::f64::consts::PI;
                } else if d_ra < -std::f64::consts::PI {
                    d_ra += 2.0 * std::f64::consts::PI;
                }
                (
                    DVector::from_column_slice(&[d_ra, dec - dec_pred]),
                    DVector::from_column_slice(&[ra_pred, dec_pred]),
                )
            }
            Self::RadarRange { range, .. } => {
                // Monostatic round-trip model: measured = (outgoing + incoming) / 2.
                // obj_lt is the object at bounce time t_b = t_rx - incoming/c;
                // obs is the observer at receive time t_rx.
                // Transmitter position at t_tx ≈ t_rx - 2*incoming/c:
                //   r_obs_tx ≈ r_obs_rx - v_obs * 2 * incoming / c.
                let incoming = (obj_lt.pos - obs.pos).norm();
                let obs_tx_pos = obs.pos - obs.vel * (2.0 * incoming * C_AU_PER_DAY_INV);
                let outgoing = (obj_lt.pos - obs_tx_pos).norm();
                let pred = f64::midpoint(incoming, outgoing);
                (
                    DVector::from_column_slice(&[range - pred]),
                    DVector::from_column_slice(&[pred]),
                )
            }
            Self::RadarRate { range_rate, .. } => {
                // Two-way Doppler: average of receive-path and transmit-path range rates.
                // Use receiver velocity for both paths (observer acceleration over the
                // ~10 s round-trip light time is negligible compared to measurement sigma).
                let d_pos_rx = obj_lt.pos - obs.pos;
                let range = d_pos_rx.norm();
                let obs_tx_pos = obs.pos - obs.vel * (2.0 * range * C_AU_PER_DAY_INV);
                let d_pos_tx = obj_lt.pos - obs_tx_pos;
                let v_rel = obj_lt.vel - obs.vel;
                let rr_rx = d_pos_rx.dot(&v_rel) / range;
                let rr_tx = d_pos_tx.dot(&v_rel) / d_pos_tx.norm();
                let pred = f64::midpoint(rr_rx, rr_tx);
                (
                    DVector::from_column_slice(&[range_rate - pred]),
                    DVector::from_column_slice(&[pred]),
                )
            }
        }
    }
}

/// Optical partials: d(RA,Dec)/d(pos) as a 2x3 matrix.
///
/// Velocity partials are zero (RA/Dec do not depend on velocity at the
/// instant of observation, neglecting light-time rate corrections).
fn optical_partials_pos(
    obj: &State<Equatorial, SSB>,
    obs: &State<Equatorial, SSB>,
) -> Matrix2x3<f64> {
    let d = obj.pos - obs.pos;
    let dx = d[0];
    let dy = d[1];
    let dz = d[2];
    let rho2 = d.norm_squared();
    let xy2 = dx * dx + dy * dy;

    // Guard against the pole singularity (dec near +/-90 deg).
    // When xy2 -> 0 the RA partial is undefined and the Dec partial
    // diverges.  Clamp to a small floor so the Jacobian stays finite;
    // the residual itself is still well-defined, and the solver will
    // not be driven by a single near-pole observation.
    let xy2_safe = xy2.max(1e-30);
    let xy = xy2_safe.sqrt();

    // dRA/d(pos)
    let dra_dx = -dy / xy2_safe;
    let dra_dy = dx / xy2_safe;

    // dDec/d(pos)
    let ddec_dx = -dx * dz / (rho2 * xy);
    let ddec_dy = -dy * dz / (rho2 * xy);
    let ddec_dz = xy / rho2;

    Matrix2x3::new(dra_dx, dra_dy, 0.0, ddec_dx, ddec_dy, ddec_dz)
}

/// Radar range partials: d(range)/d(pos) as a 3x1 column vector.
///
/// Uses the round-trip average `(incoming + outgoing) / 2`.
fn range_partials_pos(
    obj: &State<Equatorial, SSB>,
    obs: &State<Equatorial, SSB>,
) -> Matrix3x1<f64> {
    let d_rx = obj.pos - obs.pos;
    let incoming = d_rx.norm();
    let obs_tx_pos = obs.pos - obs.vel * (2.0 * incoming * C_AU_PER_DAY_INV);
    let d_tx = obj.pos - obs_tx_pos;
    let outgoing = d_tx.norm();
    // Indirect correction: d(outgoing)/d(r) includes the change in obs_tx_pos
    // through d(incoming)/d(r) = unit_rx.
    // alpha = (unit_tx . v_obs) * 2/c
    let alpha = d_tx.dot(&obs.vel) / outgoing * 2.0 * C_AU_PER_DAY_INV;
    Matrix3x1::new(
        ((1.0 + alpha) * d_rx[0] / incoming + d_tx[0] / outgoing) * 0.5,
        ((1.0 + alpha) * d_rx[1] / incoming + d_tx[1] / outgoing) * 0.5,
        ((1.0 + alpha) * d_rx[2] / incoming + d_tx[2] / outgoing) * 0.5,
    )
}

/// Radar range-rate partials: `d(range_rate)/d(pos,vel)` as a 1x6 row vector.
///
/// Two-way Doppler partial: average of receive-path and transmit-path
/// contributions.  The transmitter position at `t_tx ≈ t_rx - 2*range/c`
/// depends on `range` (and hence on `r_obj`), producing an indirect correction
/// to the position partial:
///
/// `d(rr_tx)/d(r) = (d_vel - d_hat_tx * rr_tx) / range_tx
///                 + beta_tx * unit_rx`
///
/// where `beta_tx = 2/c * [(d_vel . v_obs) - rr_tx * (unit_tx . v_obs)] / range_tx`.
///
/// The velocity partial is `(d_hat_rx + d_hat_tx) / 2` (no indirect term).
fn range_rate_partials(
    obj: &State<Equatorial, SSB>,
    obs: &State<Equatorial, SSB>,
) -> RowVector6<f64> {
    let d_pos_rx = obj.pos - obs.pos;
    let d_vel = obj.vel - obs.vel;
    let range = d_pos_rx.norm();

    // Transmitter position at t_tx ≈ t_rx - 2*range/c.
    let obs_tx_pos = obs.pos - obs.vel * (2.0 * range * C_AU_PER_DAY_INV);
    let d_pos_tx = obj.pos - obs_tx_pos;
    let range_tx = d_pos_tx.norm();

    let rr_rx = d_pos_rx.dot(&d_vel) / range;
    let rr_tx = d_pos_tx.dot(&d_vel) / range_tx;
    let d_hat_rx = d_pos_rx.normalize();
    let d_hat_tx = d_pos_tx.normalize();

    // d(rr_rx)/d(r) = (d_vel - d_hat_rx * rr_rx) / range
    let dr_rx = (d_vel - d_hat_rx * rr_rx) * (1.0 / range);

    // d(rr_tx)/d(r) = (d_vel - d_hat_tx * rr_tx) / range_tx  [direct term]
    //               + beta_tx * unit_rx                        [indirect term]
    // beta_tx = 2/c * [(d_vel . v_obs) - rr_tx * (unit_tx . v_obs)] / range_tx
    let dr_tx_direct = (d_vel - d_hat_tx * rr_tx) * (1.0 / range_tx);
    let beta_tx =
        2.0 * C_AU_PER_DAY_INV * (d_vel.dot(&obs.vel) - rr_tx * d_hat_tx.dot(&obs.vel)) / range_tx;
    let dr = (dr_rx + dr_tx_direct + d_hat_rx * beta_tx) * 0.5;

    // d(rr)/d(v) = (d_hat_rx + d_hat_tx) / 2  (no indirect term)
    let dv = (d_hat_rx + d_hat_tx) * 0.5;

    RowVector6::new(dr[0], dr[1], dr[2], dv[0], dv[1], dv[2])
}

impl AstrometricObservation {
    /// Compute the local geometric partial derivatives (`H_local`).
    ///
    /// Returns an m x 6 matrix where m is the measurement dimension
    /// (2 for optical, 1 for radar). Columns correspond to
    /// `[dx, dy, dz, dvx, dvy, dvz]` of the object state.
    ///
    /// The input `obj_state` should already be light-time-corrected.
    #[must_use]
    pub fn partials(&self, obj_state: &State<Equatorial, SSB>) -> nalgebra::DMatrix<f64> {
        let obs = self.observer();
        match self {
            Self::Optical { .. } => {
                let h = optical_partials_pos(obj_state, obs);
                let mut out = nalgebra::DMatrix::zeros(2, 6);
                out.view_mut((0, 0), (2, 3)).copy_from(&h);
                out
            }
            Self::RadarRange { .. } => {
                let h = range_partials_pos(obj_state, obs);
                let mut out = nalgebra::DMatrix::zeros(1, 6);
                out[(0, 0)] = h[0];
                out[(0, 1)] = h[1];
                out[(0, 2)] = h[2];
                out
            }
            Self::RadarRate { .. } => {
                let h = range_rate_partials(obj_state, obs);
                let mut out = nalgebra::DMatrix::zeros(1, 6);
                for j in 0..6 {
                    out[(0, j)] = h[j];
                }
                out
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::constants::C_AU_PER_DAY_INV;
    use kete_core::desigs::Desig;
    use kete_core::frames::{Equatorial, SunCenter};
    use kete_core::kepler::propagate_two_body;
    use kete_core::prelude::State;

    /// Helper: build a simple state at the given position/velocity.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SSB> {
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: SSB,
        }
    }

    /// Finite-difference helper: perturb component `idx` of the object state
    /// by `eps`, recompute the predicted measurement, return the numerical
    /// partial derivative.
    fn fd_partial(
        obs: &AstrometricObservation,
        obj: &State<Equatorial, SSB>,
        idx: usize,
        eps: f64,
    ) -> DVector<f64> {
        let mut pos_p = [obj.pos[0], obj.pos[1], obj.pos[2]];
        let mut vel_p = [obj.vel[0], obj.vel[1], obj.vel[2]];
        let mut pos_m = pos_p;
        let mut vel_m = [obj.vel[0], obj.vel[1], obj.vel[2]];

        if idx < 3 {
            pos_p[idx] += eps;
            pos_m[idx] -= eps;
        } else {
            vel_p[idx - 3] += eps;
            vel_m[idx - 3] -= eps;
        }

        let obj_p = make_state(pos_p, vel_p, obj.epoch.jd);
        let obj_m = make_state(pos_m, vel_m, obj.epoch.jd);

        let pred_p = predict_for_fd(obs, &obj_p);
        let pred_m = predict_for_fd(obs, &obj_m);

        (pred_p - pred_m) / (2.0 * eps)
    }

    /// Direct prediction without light-time (for FD tests where we want to
    /// test the geometric partials in isolation).
    fn predict_for_fd(obs: &AstrometricObservation, obj: &State<Equatorial, SSB>) -> DVector<f64> {
        let observer = obs.observer();
        match obs {
            AstrometricObservation::Optical { .. } => {
                let (ra, dec) = (obj.pos - observer.pos).to_ra_dec();
                DVector::from_vec(vec![ra, dec])
            }
            AstrometricObservation::RadarRange { .. } => {
                // Round-trip range: (incoming + outgoing) / 2.
                let incoming = (obj.pos - observer.pos).norm();
                let obs_tx_pos = observer.pos - observer.vel * (2.0 * incoming * C_AU_PER_DAY_INV);
                let outgoing = (obj.pos - obs_tx_pos).norm();
                DVector::from_vec(vec![f64::midpoint(incoming, outgoing)])
            }
            AstrometricObservation::RadarRate { .. } => {
                // Two-way Doppler: average of rx and tx path range rates.
                let d_pos_rx = obj.pos - observer.pos;
                let range = d_pos_rx.norm();
                let obs_tx_pos = observer.pos - observer.vel * (2.0 * range * C_AU_PER_DAY_INV);
                let d_pos_tx = obj.pos - obs_tx_pos;
                let v_rel = obj.vel - observer.vel;
                let rr_rx = d_pos_rx.dot(&v_rel) / range;
                let rr_tx = d_pos_tx.dot(&v_rel) / d_pos_tx.norm();
                DVector::from_vec(vec![f64::midpoint(rr_rx, rr_tx)])
            }
        }
    }

    #[test]
    fn test_optical_partials_vs_fd() {
        // Object ~1 AU from observer, not on any axis to exercise all terms
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let obj = make_state([1.5, 0.8, 0.3], [0.001, -0.002, 0.0005], 2460000.5);

        let obs = AstrometricObservation::Optical {
            observer: observer.clone(),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
            time_sigma: 0.0,
        };

        let h = obs.partials(&obj);
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(&obs, &obj, idx, eps);
            for row in 0..2 {
                let analytic = h[(row, idx)];
                let numeric = fd[row];
                let abs_err = (analytic - numeric).abs();
                let scale = analytic.abs().max(numeric.abs()).max(1e-15);
                assert!(
                    abs_err < 1e-7 || abs_err / scale < 1e-5,
                    "Optical partial ({row}, {idx}) mismatch: analytic={analytic}, fd={numeric}",
                );
            }
        }
    }

    #[test]
    fn test_range_partials_vs_fd() {
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let obj = make_state([2.3, 0.5, -0.2], [0.002, -0.001, 0.001], 2460000.5);

        let obs = AstrometricObservation::RadarRange {
            observer: observer.clone(),
            range: 1.0,
            sigma_range: 1e-6,
        };

        let h = obs.partials(&obj);
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(&obs, &obj, idx, eps);
            let analytic = h[(0, idx)];
            let numeric = fd[0];
            let abs_err = (analytic - numeric).abs();
            let scale = analytic.abs().max(numeric.abs()).max(1e-15);
            assert!(
                abs_err < 1e-7 || abs_err / scale < 1e-5,
                "Range partial ({idx}) mismatch: analytic={analytic}, fd={numeric}",
            );
        }
    }

    #[test]
    fn test_range_rate_partials_vs_fd() {
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let obj = make_state([2.3, 0.5, -0.2], [0.002, -0.001, 0.001], 2460000.5);

        let obs = AstrometricObservation::RadarRate {
            observer: observer.clone(),
            range_rate: 0.01,
            sigma_range_rate: 1e-6,
        };

        let h = obs.partials(&obj);
        let eps = 1e-8;

        for idx in 0..6 {
            let fd = fd_partial(&obs, &obj, idx, eps);
            let analytic = h[(0, idx)];
            let numeric = fd[0];
            let abs_err = (analytic - numeric).abs();
            let scale = analytic.abs().max(numeric.abs()).max(1e-15);
            assert!(
                abs_err < 1e-7 || abs_err / scale < 1e-5,
                "Range-rate partial ({idx}) mismatch: analytic={analytic}, fd={numeric}",
            );
        }
    }

    #[test]
    fn test_light_time_correction() {
        // Object at 2 AU from observer. Light time ~ 2 * C_AU_PER_DAY_INV ~ 0.01155 days
        let observer = make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5);
        let obj = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: 2460000.5.into(),
            pos: [2.0, 0.0, 0.0].into(),
            vel: [0.0, 0.01, 0.0].into(),
            center: SunCenter,
        };

        let tau_lt = (obj.pos - observer.pos).norm() * C_AU_PER_DAY_INV;
        let corrected = propagate_two_body(&obj, obj.epoch - tau_lt).unwrap();

        // Corrected epoch should be earlier
        let tau = 2.0 * C_AU_PER_DAY_INV;
        assert!((corrected.epoch.jd - (2460000.5 - tau)).abs() < 1e-12);

        // Position should be slightly different due to back-propagation
        assert!((corrected.pos[0] - obj.pos[0]).abs() < 1e-4);
        // dy should shift: the object moves in y at 0.01 AU/day, backed by ~0.01 day
        assert!(corrected.pos[1] < obj.pos[1]);
    }

    #[test]
    fn test_residual_optical() {
        // Observer at ~1 AU (Earth-like), object at ~2 AU along +x.
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.017, 0.0], 2460000.5);
        let obj = make_state([2.0, 0.0, 0.0], [0.0, 0.012, 0.0], 2460000.5);

        // True RA/Dec for object along +x from observer is RA~0, Dec~0.
        let d = obj.pos - observer.pos;
        let (true_ra, true_dec) = d.to_ra_dec();

        let obs = AstrometricObservation::Optical {
            observer,
            ra: true_ra + 0.01,
            dec: true_dec + 0.005,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
            time_sigma: 0.0,
        };

        let (resid, _pred) = obs.residual(obj.into()).unwrap();
        // Residual should be close to the injected offset.
        // (Predicted RA may wrap by 2*pi; the residual handles wrapping.)
        assert!((resid[0] - 0.01).abs() < 0.01);
        assert!((resid[1] - 0.005).abs() < 0.01);
    }

    #[test]
    fn test_weights() {
        let obs = AstrometricObservation::Optical {
            observer: make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 0.5,
            sigma_dec: 0.25,
            time_sigma: 0.0,
        };
        let w = obs.weights();
        // 1/0.5^2 = 4
        assert!((w[0] - 4.0).abs() < 1e-12);
        // 1/0.25^2 = 16
        assert!((w[1] - 16.0).abs() < 1e-12);
    }
}
