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

use kete_core::frames::Equatorial;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::propagation::light_time_correct;
use nalgebra::{DVector, Matrix2x3, Matrix3x1, RowVector6};

/// A single astrometric or radar observation.
///
/// Each variant carries the observer geometry, measured values, and
/// uncertainties. The observation epoch is taken from `observer.epoch`.
#[derive(Debug, Clone)]
pub enum Observation {
    /// Optical astrometry: RA and Dec on the sky.
    Optical {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial>,
        /// Right ascension (radians).
        ra: f64,
        /// Declination (radians).
        dec: f64,
        /// 1-sigma RA uncertainty (radians, includes cos(dec) factor).
        sigma_ra: f64,
        /// 1-sigma Dec uncertainty (radians).
        sigma_dec: f64,
    },

    /// Radar range measurement.
    RadarRange {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial>,
        /// Measured range (AU).
        range: f64,
        /// 1-sigma range uncertainty (AU).
        sigma_range: f64,
    },

    /// Radar range-rate (Doppler) measurement.
    RadarRate {
        /// Observer state (SSB-centered, Equatorial).
        observer: State<Equatorial>,
        /// Measured range-rate (AU/day, positive = receding).
        range_rate: f64,
        /// 1-sigma range-rate uncertainty (AU/day).
        sigma_range_rate: f64,
    },
}

impl Observation {
    /// Reference to the observer state (carries the observation epoch).
    pub fn observer(&self) -> &State<Equatorial> {
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
    pub fn as_optical(&self) -> KeteResult<(f64, f64, &State<Equatorial>)> {
        match self {
            Self::Optical {
                observer, ra, dec, ..
            } => Ok((*ra, *dec, observer)),
            Self::RadarRange { .. } | Self::RadarRate { .. } => {
                Err(Error::ValueError("Expected an Optical observation".into()))
            }
        }
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
    #[must_use]
    pub fn weights(&self) -> DVector<f64> {
        match self {
            Self::Optical {
                sigma_ra,
                sigma_dec,
                ..
            } => DVector::from_vec(vec![
                1.0 / (sigma_ra * sigma_ra),
                1.0 / (sigma_dec * sigma_dec),
            ]),
            Self::RadarRange { sigma_range, .. } => {
                DVector::from_vec(vec![1.0 / (sigma_range * sigma_range)])
            }
            Self::RadarRate {
                sigma_range_rate, ..
            } => DVector::from_vec(vec![1.0 / (sigma_range_rate * sigma_range_rate)]),
        }
    }
}

impl Observation {
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
        obj_state: &State<Equatorial>,
    ) -> KeteResult<(DVector<f64>, DVector<f64>)> {
        let obs = self.observer();
        let obj_lt = light_time_correct(obj_state, &obs.pos)?;
        Ok(self.residual_predicted_from_corrected(&obj_lt))
    }

    /// Compute residual from an already light-time-corrected object state.
    ///
    /// This avoids a redundant two-body propagation when the caller has
    /// already applied `light_time_correct`.
    #[must_use]
    pub fn residual_from_corrected(&self, obj_lt: &State<Equatorial>) -> DVector<f64> {
        self.residual_predicted_from_corrected(obj_lt).0
    }

    /// Core residual computation from a light-time-corrected state.
    ///
    /// Returns `(residual, predicted)`.
    fn residual_predicted_from_corrected(
        &self,
        obj_lt: &State<Equatorial>,
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
                    DVector::from_vec(vec![d_ra, dec - dec_pred]),
                    DVector::from_vec(vec![ra_pred, dec_pred]),
                )
            }
            Self::RadarRange { range, .. } => {
                let pred = (obj_lt.pos - obs.pos).norm();
                (
                    DVector::from_vec(vec![range - pred]),
                    DVector::from_vec(vec![pred]),
                )
            }
            Self::RadarRate { range_rate, .. } => {
                let d_pos = obj_lt.pos - obs.pos;
                let pred = d_pos.dot(&(obj_lt.vel - obs.vel)) / d_pos.norm();
                (
                    DVector::from_vec(vec![range_rate - pred]),
                    DVector::from_vec(vec![pred]),
                )
            }
        }
    }
}

/// Optical partials: d(RA,Dec)/d(pos) as a 2x3 matrix.
///
/// Velocity partials are zero (RA/Dec do not depend on velocity at the
/// instant of observation, neglecting light-time rate corrections).
fn optical_partials_pos(obj: &State<Equatorial>, obs: &State<Equatorial>) -> Matrix2x3<f64> {
    let d = obj.pos - obs.pos;
    let dx = d[0];
    let dy = d[1];
    let dz = d[2];
    let rho2 = d.norm_squared();
    let xy2 = dx * dx + dy * dy;
    let xy = xy2.sqrt();

    // dRA/d(pos)
    let dra_dx = -dy / xy2;
    let dra_dy = dx / xy2;

    // dDec/d(pos)
    let ddec_dx = -dx * dz / (rho2 * xy);
    let ddec_dy = -dy * dz / (rho2 * xy);
    let ddec_dz = xy / rho2;

    Matrix2x3::new(dra_dx, dra_dy, 0.0, ddec_dx, ddec_dy, ddec_dz)
}

/// Radar range partials: d(range)/d(pos) as a 1x3 row vector (unit vector).
fn range_partials_pos(obj: &State<Equatorial>, obs: &State<Equatorial>) -> Matrix3x1<f64> {
    let d = obj.pos - obs.pos;
    let range_inv = 1.0 / d.norm();
    Matrix3x1::new(d[0] * range_inv, d[1] * range_inv, d[2] * range_inv)
}

/// Radar range-rate partials: `d(range_rate)/d(pos,vel)` as a 1x6 row vector.
///
/// `range_rate = v_rel . d_hat`
///
/// `d(range_rate)/d(r) = (v_rel - d_hat * range_rate) / range`
/// `d(range_rate)/d(v) = d_hat`
fn range_rate_partials(obj: &State<Equatorial>, obs: &State<Equatorial>) -> RowVector6<f64> {
    let d_pos = obj.pos - obs.pos;
    let d_vel = obj.vel - obs.vel;
    let range = d_pos.norm();
    let range_inv = 1.0 / range;

    let rr = d_pos.dot(&d_vel) * range_inv;
    let d_hat = d_pos.normalize();

    // d(range_rate)/d(r) = (v_rel - d_hat * rr) / range
    let dr = (d_vel - d_hat * rr) * range_inv;

    RowVector6::new(dr[0], dr[1], dr[2], d_hat[0], d_hat[1], d_hat[2])
}

impl Observation {
    /// Compute the local geometric partial derivatives (`H_local`).
    ///
    /// Returns an m x 6 matrix where m is the measurement dimension
    /// (2 for optical, 1 for radar). Columns correspond to
    /// `[dx, dy, dz, dvx, dvy, dvz]` of the object state.
    ///
    /// The input `obj_state` should already be light-time-corrected.
    #[must_use]
    pub fn partials(&self, obj_state: &State<Equatorial>) -> nalgebra::DMatrix<f64> {
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
    use kete_core::frames::Equatorial;
    use kete_core::prelude::State;
    use kete_core::propagation::propagate_two_body;

    /// Helper: build a simple state at the given position/velocity.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial> {
        State::new(Desig::Empty, jd.into(), pos.into(), vel.into(), 0)
    }

    /// Finite-difference helper: perturb component `idx` of the object state
    /// by `eps`, recompute the predicted measurement, return the numerical
    /// partial derivative.
    fn fd_partial(
        obs: &Observation,
        obj: &State<Equatorial>,
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
    fn predict_for_fd(obs: &Observation, obj: &State<Equatorial>) -> DVector<f64> {
        let observer = obs.observer();
        match obs {
            Observation::Optical { .. } => {
                let (ra, dec) = (obj.pos - observer.pos).to_ra_dec();
                DVector::from_vec(vec![ra, dec])
            }
            Observation::RadarRange { .. } => {
                DVector::from_vec(vec![(obj.pos - observer.pos).norm()])
            }
            Observation::RadarRate { .. } => {
                let d_pos = obj.pos - observer.pos;
                DVector::from_vec(vec![d_pos.dot(&(obj.vel - observer.vel)) / d_pos.norm()])
            }
        }
    }

    #[test]
    fn test_optical_partials_vs_fd() {
        // Object ~1 AU from observer, not on any axis to exercise all terms
        let observer = make_state([1.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);
        let obj = make_state([1.5, 0.8, 0.3], [0.001, -0.002, 0.0005], 2460000.5);

        let obs = Observation::Optical {
            observer: observer.clone(),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
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

        let obs = Observation::RadarRange {
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

        let obs = Observation::RadarRate {
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
        let obj = make_state([2.0, 0.0, 0.0], [0.0, 0.01, 0.0], 2460000.5);

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

        let obs = Observation::Optical {
            observer,
            ra: true_ra + 0.01,
            dec: true_dec + 0.005,
            sigma_ra: 1e-6,
            sigma_dec: 1e-6,
        };

        let (resid, _pred) = obs.residual(&obj).unwrap();
        // Residual should be close to the injected offset.
        // (Predicted RA may wrap by 2*pi; the residual handles wrapping.)
        assert!((resid[0] - 0.01).abs() < 0.01);
        assert!((resid[1] - 0.005).abs() < 0.01);
    }

    #[test]
    fn test_weights() {
        let obs = Observation::Optical {
            observer: make_state([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2460000.5),
            ra: 0.0,
            dec: 0.0,
            sigma_ra: 0.5,
            sigma_dec: 0.25,
        };
        let w = obs.weights();
        assert!((w[0] - 4.0).abs() < 1e-12); // 1/0.5^2 = 4
        assert!((w[1] - 16.0).abs() < 1e-12); // 1/0.25^2 = 16
    }
}
