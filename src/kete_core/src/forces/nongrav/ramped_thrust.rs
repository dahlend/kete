// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Outgassing thrust along a direction fixed in the radial / transverse / normal
//! frame, with a strength that changes linearly in time and an optional part that
//! turns at a fixed period.

use std::f64::consts::TAU;

use nalgebra::{Matrix3xX, Vector3};

use super::jpl_comet::rtn_dirs;
use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::time::{TDB, Time};

/// Thrust fixed in the radial / transverse / normal (RTN) frame, with a strength
/// that ramps linearly in time, and optionally a part that turns at a fixed period.
///
/// ```text
/// accel = f(t) (A . rtn)
/// A     = a + b cos(phi) + c sin(phi),   phi = 2 pi (t - t0) / period
/// f(t)  = max(0, 1 + rate (t - t0))
/// ```
///
/// where `a . rtn = a1 r_hat + a2 t_hat + a3 n_hat`, likewise for `b` and `c`.
///
/// Free parameters, in order: `a1`, `a2`, `a3` (AU/day^2, the steady thrust at
/// `t0`), `rate` (1/day, the fractional change of the thrust per day), and, only
/// when a period is given, `b1`, `b2`, `b3`, `c1`, `c2`, `c3` (AU/day^2, the
/// turning part at `t0`). Without a period there are four parameters and the
/// thrust keeps its direction in the RTN frame; only its size changes. The ramp
/// scales the whole thrust, and the thrust is zero, not reversed, where
/// `1 + rate (t - t0)` is negative.
///
/// The turning part is the lowest-order signature of a spinning body whose
/// outgassing is not symmetric about its spin axis: `b` and `c` span the plane
/// the thrust turns in, and the period is the spin period (or the period of any
/// other modulation). It is linear in `b` and `c`; the period is not fitted, and
/// is meant to be scanned.
///
/// There is no dependence on heliocentric distance. The model is meant for arcs
/// short enough that the distance to the Sun, and so the RTN frame, change little,
/// such as a body near a comet nucleus over hours to days. With `rate = 0` and no
/// period it is the JPL comet model with `g(r) = 1`.
///
/// `accel` expects `pos`/`vel` Sun-relative.
#[derive(Debug, Clone)]
pub struct RampedThrustNonGrav {
    /// Reference epoch of the ramp and of the phase.
    pub t0: Time<TDB>,

    /// Period of the turning part, days; `None` for a thrust without one.
    pub period: Option<f64>,
}

impl RampedThrustNonGrav {
    /// Build with the reference epoch `t0` and the period of the turning part in
    /// days, if any.
    ///
    /// # Errors
    /// Fails if `t0` is not finite, or the period is not positive and finite.
    pub fn new(t0: Time<TDB>, period: Option<f64>) -> KeteResult<Self> {
        if !t0.jd().is_finite() {
            return Err(Error::ValueError(format!(
                "RampedThrustNonGrav reference epoch must be finite, found {}.",
                t0.jd()
            )));
        }
        if let Some(p) = period
            && !(p.is_finite() && p > 0.0)
        {
            return Err(Error::ValueError(format!(
                "RampedThrustNonGrav period must be positive and finite, found {p}."
            )));
        }
        Ok(Self { t0, period })
    }

    /// The ramp factor `1 + rate (t - t0)`, before clamping at zero.
    fn ramp(&self, time: Time<TDB>, rate: f64) -> f64 {
        1.0 + rate * (time - self.t0).elapsed
    }

    /// `(cos(phi), sin(phi))` of the turning part at `time`; `(0, 0)` without a period.
    fn phase(&self, time: Time<TDB>) -> (f64, f64) {
        match self.period {
            Some(p) => {
                let (s, c) = (TAU * (time - self.t0).elapsed / p).sin_cos();
                (c, s)
            }
            None => (0.0, 0.0),
        }
    }

    /// Check the parameter count and return `(A, rate)` at `time`, with `A` the RTN
    /// components of the thrust before the ramp.
    fn unpack(&self, time: Time<TDB>, free_params: &[f64]) -> KeteResult<(Vector3<f64>, f64)> {
        if free_params.len() != self.n_free_params() {
            return Err(Error::ValueError(format!(
                "RampedThrustNonGrav expects {} free parameters ({}), got {}",
                self.n_free_params(),
                self.free_param_names().join(", "),
                free_params.len()
            )));
        }
        let mut a = Vector3::new(free_params[0], free_params[1], free_params[2]);
        if self.period.is_some() {
            let (cos, sin) = self.phase(time);
            let b = Vector3::new(free_params[4], free_params[5], free_params[6]);
            let c = Vector3::new(free_params[7], free_params[8], free_params[9]);
            a += b * cos + c * sin;
        }
        Ok((a, free_params[3]))
    }
}

impl ParameterizedForce for RampedThrustNonGrav {
    type Frame = Equatorial;
    type Center = SunCenter;
    type Meta = ();

    fn n_free_params(&self) -> usize {
        if self.period.is_some() { 10 } else { 4 }
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        let mut names = vec!["a1", "a2", "a3", "rate"];
        if self.period.is_some() {
            names.extend(["b1", "b2", "b3", "c1", "c2", "c3"]);
        }
        names
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        _meta: &mut Self::Meta,
        _exact_eval: bool,
    ) -> KeteResult<Vector<Equatorial>> {
        let (a, rate) = self.unpack(time, free_params)?;
        let factor = self.ramp(time, rate).max(0.0);
        let pos: Vector3<f64> = (*pos).into();
        let vel: Vector3<f64> = (*vel).into();
        let (r_hat, t_hat, n_hat) = rtn_dirs(&pos, &vel);
        let result = (r_hat * a.x + t_hat * a.y + n_hat * a.z) * factor;
        Ok(Vector::<Equatorial>::new(result.into()))
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        _meta: &mut Self::Meta,
    ) -> KeteResult<Matrix3xX<f64>> {
        // Linear in a, b and c at a fixed rate; the rate column is the thrust before
        // the ramp times (t - t0) where the ramp is positive, and zero where it is
        // clamped.
        let (a, rate) = self.unpack(time, free_params)?;
        let ramp = self.ramp(time, rate);
        let pos: Vector3<f64> = (*pos).into();
        let vel: Vector3<f64> = (*vel).into();
        let dirs = rtn_dirs(&pos, &vel);
        let dirs = [dirs.0, dirs.1, dirs.2];
        let factor = ramp.max(0.0);
        let (cos, sin) = self.phase(time);
        let mut out = Matrix3xX::<f64>::zeros(self.n_free_params());
        for (i, dir) in dirs.iter().enumerate() {
            out.set_column(i, &(dir * factor));
            if self.period.is_some() {
                out.set_column(4 + i, &(dir * (factor * cos)));
                out.set_column(7 + i, &(dir * (factor * sin)));
            }
        }
        if ramp > 0.0 {
            let thrust = dirs[0] * a.x + dirs[1] * a.y + dirs[2] * a.z;
            out.set_column(3, &(thrust * (time - self.t0).elapsed));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::JplCometNonGrav;
    use super::*;

    fn pos() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([1.5, 0.3, 0.1])
    }
    fn vel() -> Vector<Equatorial> {
        Vector::<Equatorial>::new([-0.005, 0.012, 0.001])
    }

    fn accel_at(f: &RampedThrustNonGrav, jd: f64, params: &[f64]) -> Vector3<f64> {
        f.accel(
            Time::<TDB>::new(jd),
            &pos(),
            &vel(),
            params,
            &mut Default::default(),
            false,
        )
        .unwrap()
        .into()
    }

    /// With the rate at zero this is the JPL comet model with g(r) = 1.
    #[test]
    fn zero_rate_matches_jpl_comet() {
        let ramped = RampedThrustNonGrav::new(2_451_545.0.into(), None).unwrap();
        let comet = JplCometNonGrav::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let a = [1e-8, -3e-9, 5e-10];
        for jd in [2_451_500.0, 2_451_545.0, 2_451_600.0] {
            let got = accel_at(&ramped, jd, &[a[0], a[1], a[2], 0.0]);
            let want: Vector3<f64> = comet
                .accel(
                    Time::<TDB>::new(jd),
                    &pos(),
                    &vel(),
                    &a,
                    &mut Default::default(),
                    false,
                )
                .unwrap()
                .into();
            assert!((got - want).norm() <= 1e-15 * want.norm(), "jd {jd}");
        }
    }

    /// The thrust scales by 1 + rate (t - t0) and keeps its direction.
    #[test]
    fn ramp_scales_the_thrust() {
        let t0 = 2_451_545.0;
        let f = RampedThrustNonGrav::new(t0.into(), None).unwrap();
        let params = [1e-8, -3e-9, 5e-10, 0.5];
        let at_t0 = accel_at(&f, t0, &params);
        let later = accel_at(&f, t0 + 2.0, &params);
        assert!((later - at_t0 * 2.0).norm() <= 1e-15 * later.norm());
    }

    /// Before the ramp crosses zero the thrust is off, not reversed.
    #[test]
    fn thrust_is_off_before_the_ramp_starts() {
        let t0 = 2_451_545.0;
        let f = RampedThrustNonGrav::new(t0.into(), Some(0.5)).unwrap();
        let params = [1e-8, -3e-9, 5e-10, 0.5, 1e-9, 0.0, 0.0, 0.0, 2e-9, 0.0];
        assert_eq!(accel_at(&f, t0 - 3.0, &params).norm(), 0.0);
        let jac = f
            .parameter_jacobian(
                Time::<TDB>::new(t0 - 3.0),
                &pos(),
                &vel(),
                &params,
                &mut Default::default(),
            )
            .unwrap();
        assert_eq!(jac.norm(), 0.0);
    }

    /// The turning part is b at t0, c a quarter period later, -b half a period
    /// later, and back to b after a full period; with b and c zero the period changes
    /// nothing.
    #[test]
    fn turning_part_follows_its_phase() {
        let t0 = 2_451_545.0;
        let period = 0.5;
        let f = RampedThrustNonGrav::new(t0.into(), Some(period)).unwrap();
        let steady = RampedThrustNonGrav::new(t0.into(), None).unwrap();
        let a = [1e-8, -3e-9, 5e-10];
        let b = [2e-9, 1e-9, 0.0];
        let c = [0.0, -1e-9, 3e-9];
        let with =
            |b: [f64; 3], c: [f64; 3]| [a[0], a[1], a[2], 0.0, b[0], b[1], b[2], c[0], c[1], c[2]];
        let only_a = accel_at(&steady, t0, &[a[0], a[1], a[2], 0.0]);
        let only_b = accel_at(&f, t0, &with(b, [0.0; 3])) - only_a;
        let only_c = accel_at(&f, t0 + period / 4.0, &with([0.0; 3], c)) - only_a;
        let params = with(b, c);
        for (dt, want) in [
            (0.0, only_b),
            (period / 4.0, only_c),
            (period / 2.0, -only_b),
            (period, only_b),
        ] {
            let got = accel_at(&f, t0 + dt, &params) - only_a;
            assert!((got - want).norm() <= 1e-9 * want.norm(), "dt {dt}");
        }
        for jd in [t0 - 0.3, t0 + 0.1] {
            let got = accel_at(&f, jd, &with([0.0; 3], [0.0; 3]));
            let want = accel_at(&steady, jd, &[a[0], a[1], a[2], 0.0]);
            assert!((got - want).norm() <= 1e-15 * want.norm(), "jd {jd}");
        }
    }

    #[test]
    fn analytic_jacobian_matches_finite_difference() {
        let t0 = 2_451_545.0;
        let cases: [(Option<f64>, &[f64]); 2] = [
            (None, &[1e-8, -3e-9, 5e-10, 0.2]),
            (
                Some(0.52),
                &[
                    1e-8, -3e-9, 5e-10, 0.2, 2e-9, 1e-9, -1e-9, 5e-10, -2e-9, 3e-9,
                ],
            ),
        ];
        for (period, params) in cases {
            let f = RampedThrustNonGrav::new(t0.into(), period).unwrap();
            for jd in [t0 - 1.0, t0, t0 + 0.37, t0 + 3.0] {
                let jac = f
                    .parameter_jacobian(
                        Time::<TDB>::new(jd),
                        &pos(),
                        &vel(),
                        params,
                        &mut Default::default(),
                    )
                    .unwrap();
                for col in 0..params.len() {
                    let h = if col == 3 { 1e-6 } else { 1e-12 };
                    let analytic = Vector3::new(jac[(0, col)], jac[(1, col)], jac[(2, col)]);
                    let mut p = params.to_vec();
                    p[col] += h;
                    let a_plus = accel_at(&f, jd, &p);
                    p[col] = params[col] - h;
                    let a_minus = accel_at(&f, jd, &p);
                    let fd = (a_plus - a_minus) / (2.0 * h);
                    assert!(
                        (analytic - fd).norm() <= 1e-7 * fd.norm().max(1e-30),
                        "period {period:?} jd {jd} column {col}: analytic {analytic:?} vs fd {fd:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_epoch_and_period() {
        assert!(RampedThrustNonGrav::new(f64::NAN.into(), None).is_err());
        for p in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(RampedThrustNonGrav::new(2_451_545.0.into(), Some(p)).is_err());
        }
    }

    #[test]
    fn rejects_wrong_parameter_count() {
        let jd = Time::<TDB>::new(2_451_545.0);
        let f = RampedThrustNonGrav::new(2_451_545.0.into(), None).unwrap();
        assert!(
            f.accel(
                jd,
                &pos(),
                &vel(),
                &[1.0, 2.0, 3.0],
                &mut Default::default(),
                false
            )
            .is_err()
        );
        let f = RampedThrustNonGrav::new(2_451_545.0.into(), Some(0.5)).unwrap();
        assert_eq!(f.free_param_names().len(), 10);
        assert!(
            f.accel(
                jd,
                &pos(),
                &vel(),
                &[1.0, 2.0, 3.0, 0.0],
                &mut Default::default(),
                false
            )
            .is_err()
        );
    }
}
