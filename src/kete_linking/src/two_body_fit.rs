// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! A damped least-squares fit of a two-body orbit about the Sun to lines of
//! sight.
//!
//! A candidate carries the state of one unit under one hypothesis. On a short
//! arc, hypotheses far from the truth group the same sights, so that state
//! can be far off along the line of sight while it still predicts the sights
//! within the grouping tolerance. This fit moves it to the orbit that fits
//! the sights, so that the orbit fit which confirms the candidate starts near
//! its solution, and it picks out sights which do not fit.

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::kepler::propagate_two_body;
use kete_core::prelude::State;
use nalgebra::{Matrix6, Vector3, Vector6};

use crate::Sight;
use crate::linalg::{chi2, inverse, scaled_solve};
use crate::observe::{Seen, seen};
use crate::sight::first_and_last;

/// Most trial steps of one fit of a set of sights.
///
/// From a start which predicts the sights within the grouping tolerance, a
/// fit takes a handful of accepted steps. Rejected trials in a row raise the
/// damping by factors of two, four, eight and so on, so from
/// [`START_DAMPING`] it reaches [`MAX_DAMPING`] after nine rejections in a
/// row. A hundred covers both several times, and also the longer curved
/// paths of starts a few times too near or too far along the lines of sight
/// (test `far_starts_reach_the_minimum`).
const STEPS: usize = 100;

/// The fit ends when the undamped step from an accepted one is predicted to
/// lower the chi-square by less than this, or a rejected step is. It is the
/// chi-square of a step of about a hundredth of a sigma.
const CHI2_TOLERANCE: f64 = 1e-4;

/// Starting damping, added to the unit diagonal of the scaled normal matrix.
/// It is small, so the first step is nearly a Gauss-Newton step, as the start
/// already predicts the sights within the grouping tolerance.
///
/// The damping follows the schedule of Nielsen (1999), with a deeper fall
/// after good steps: after an accepted step it is scaled by
/// `max(SMALLEST_SCALE, 1 - (2 q - 1)^3)`, where `q` is the gain over the
/// gain the linear model predicted, so it falls when the model is good and
/// stays when it is poor; after rejected steps in a row it doubles, then
/// quadruples, and so on.
const START_DAMPING: f64 = 1e-3;

/// Smallest factor by which an accepted step lowers the damping. Nielsen
/// (1999) falls by a third at most, which after a run of rejected trials
/// takes many good steps to return to nearly Gauss-Newton steps; the fixed
/// tenfold schedule falls by a tenth, but it also raises tenfold on every
/// rejection and so swings between a step too long and an accepted one.
const SMALLEST_SCALE: f64 = 0.1;

/// Smallest damping.
const MIN_DAMPING: f64 = 1e-12;

/// Largest damping. A step damped this much is ten orders of magnitude
/// shorter than a Gauss-Newton step on the scaled parameters. When no such
/// step lowers the chi-square, the fit is at its minimum to rounding.
const MAX_DAMPING: f64 = 1e10;

/// Fewest sights the fit keeps when it drops outliers: four give eight
/// measurements for six parameters.
const MIN_KEPT: usize = 4;

/// Fewest of `count` sights the fit keeps when it drops outliers: four, and
/// at least two thirds of them, the floor of the two-thirds rule of
/// [`crate::confirm()`], see there for the reason. The orbit fit of
/// `kete_fitting` likewise widens its rejection threshold rather than keep
/// less than two thirds of its observations.
pub(crate) fn fewest_kept(count: usize) -> usize {
    MIN_KEPT.max((2 * count).div_ceil(3))
}

/// The two-body orbit about the Sun, as its state at the middle of the time
/// `sights` span, which best fits them in the least-squares sense starting
/// from `start`, the indices of the sights it kept, in order, and the reduced
/// root mean square residual of those sights, see [`reduced_rms`].
///
/// The residual of a sight is the predicted unit direction, with light
/// travel time, less the observed one, over the sigma of the sight. The fit
/// steps by Levenberg-Marquardt, see [`settle`], on the state at the middle
/// of the time the sights span. A candidate's state is at the reference time
/// of its hypothesis, which can be far from its own sights; there its
/// position and velocity are strongly correlated and the steps are poor.
/// The start is therefore moved to the middle on its two-body orbit, and the
/// fit is given there: an orbit fit under more forces than the Sun, started
/// from it, starts where the two-body orbit is tied to the sights, rather
/// than where the forces it leaves out have pulled it away from them.
///
/// Once on a bound orbit the fit takes only steps to bound orbits, as only
/// bound orbits are in the scope of the crate. A short arc constrains the
/// distance along the line of sight poorly, and without this the fit can
/// follow that freedom far out to orbits which fit the sights no better than
/// bound ones. A start which is not bound, as a candidate found under a
/// hypothesis far from the truth can be, steps freely until it is.
///
/// After the fit settles, each sight is scored as the orbit fit of
/// `kete_fitting` scores observations for rejection: its chi-square over its
/// expected value, two less its leverage, the share of its two measurements
/// that the six parameters absorb, with that expected value at least a half.
/// The sight with the largest score is dropped if the score is above `reject`
/// and the sights left after the drop would still number at least
/// [`fewest_kept`] of the count of `sights`: at least four, and at least two
/// thirds of them. The rest are fit again from `start`, so that the drop of a
/// stray sight leaves no trace of it in the orbit.
///
/// Returns `None` if `start` cannot be moved to the middle of the sights, or
/// the residuals cannot be evaluated there, such as for a sight without a
/// positive uncertainty. A trial step whose residuals cannot be evaluated is
/// rejected, and a fit whose normal matrix is singular scores its sights
/// without leverage.
pub(crate) fn fit_two_body(
    start: &State<Equatorial, SunCenter>,
    sights: &[Sight],
    reject: f64,
) -> Option<(State<Equatorial, SunCenter>, Vec<usize>, f64)> {
    let (first, last) = first_and_last(sights.iter().map(|sight| sight.time.jd()));
    let middle = propagate_two_body(start, f64::midpoint(first, last).into()).ok()?;
    let mut kept: Vec<usize> = (0..sights.len()).collect();
    let fewest = fewest_kept(sights.len());
    loop {
        let chosen: Vec<&Sight> = kept.iter().map(|&i| &sights[i]).collect();
        let (params, current) = settle(&middle, &chosen)?;
        let state = from_params(&middle, &params);
        let rms = reduced_rms(&current);
        if kept.len() <= fewest {
            return Some((state, kept, rms));
        }
        let leverage = leverages(&current).unwrap_or_else(|| vec![0.0; chosen.len()]);
        let scores = current
            .iter()
            .zip(&leverage)
            .map(|(seen, h)| seen.residual.norm_squared() / (2.0 - h).max(0.5));
        let (worst, largest) = scores.enumerate().max_by(|a, b| a.1.total_cmp(&b.1))?;
        if largest <= reject {
            return Some((state, kept, rms));
        }
        let _ = kept.remove(worst);
    }
}

/// Whether a state is on a bound orbit about the Sun.
fn is_bound(params: &Vector6<f64>) -> bool {
    let r = params.fixed_rows::<3>(0).norm();
    let v = params.fixed_rows::<3>(3).norm();
    0.5 * v * v < GMS / r
}

/// Position and velocity of a state as six numbers.
fn to_params(state: &State<Equatorial, SunCenter>) -> Vector6<f64> {
    let (pos, vel) = (Vector3::from(state.pos), Vector3::from(state.vel));
    Vector6::new(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2])
}

/// The state at the epoch of `like` with these six numbers.
fn from_params(
    like: &State<Equatorial, SunCenter>,
    p: &Vector6<f64>,
) -> State<Equatorial, SunCenter> {
    State {
        pos: Vector::new([p[0], p[1], p[2]]),
        vel: Vector::new([p[3], p[4], p[5]]),
        ..like.clone()
    }
}

/// Each sight under the orbit of `params` at the epoch of `like`, if every
/// residual and its derivative can be evaluated.
fn evaluate(
    like: &State<Equatorial, SunCenter>,
    params: &Vector6<f64>,
    sights: &[&Sight],
) -> Option<Vec<Seen>> {
    let state = from_params(like, params);
    sights.iter().map(|sight| seen(&state, sight)).collect()
}

/// Reduced root mean square of the residuals, as the orbit fit of
/// `kete_fitting` reports it: the square root of the chi-square over two
/// measurements for each sight less the six parameters, or over the
/// measurements alone when there are no more of them than parameters.
fn reduced_rms(seen: &[Seen]) -> f64 {
    let measurements = 2 * seen.len();
    let dof = if measurements > 6 {
        measurements - 6
    } else {
        measurements
    };
    (chi2(seen.iter().map(|seen| &seen.residual)) / dof.max(1) as f64).sqrt()
}

/// The parameters where Levenberg-Marquardt steps from `like` stop lowering
/// the chi-square of `sights`, staying bound once bound, and the sights
/// under them.
///
/// Each step is on the exact Jacobian of [`seen`], which comes with the
/// residuals of each trial, so the fit's minimum is that of the exact
/// residuals. The fit ends when the undamped step from the current orbit is
/// predicted to gain less than [`CHI2_TOLERANCE`], not when a damped step
/// gains that little: where the sights fix one direction weakly, as the
/// distance on a short arc, a damped step barely moves along it while the
/// minimum is still far along it.
fn settle(
    like: &State<Equatorial, SunCenter>,
    sights: &[&Sight],
) -> Option<(Vector6<f64>, Vec<Seen>)> {
    let mut params = to_params(like);
    let mut current = evaluate(like, &params, sights)?;
    let mut current_chi2 = chi2(current.iter().map(|seen| &seen.residual));
    let mut damping = START_DAMPING;
    let mut raise = 2.0;
    let mut normal = normal_equations(&current);
    for _ in 0..STEPS {
        // Damped on the scaled unit diagonal, so that the damping is the same
        // for every parameter whatever its units.
        let (matrix, gradient) = &normal;
        let step = scaled_solve(matrix, gradient, damping);
        // The gain of the linear model, `2 d.g - d.N.d` for the step `d`.
        let predicted = step.map_or(f64::INFINITY, |step| {
            2.0 * step.dot(gradient) - step.dot(&(matrix * step))
        });
        let better = step.and_then(|step| {
            let trial = params - step;
            if is_bound(&params) && !is_bound(&trial) {
                return None;
            }
            let found = evaluate(like, &trial, sights)?;
            let found_chi2 = chi2(found.iter().map(|seen| &seen.residual));
            (found_chi2 < current_chi2).then_some((trial, found, found_chi2))
        });
        let Some((trial, found, found_chi2)) = better else {
            // More damping only shortens the step and lowers what it can
            // gain, so a step whose linear model gains less than the
            // tolerance is the end of the fit.
            if predicted < CHI2_TOLERANCE {
                break;
            }
            damping *= raise;
            raise *= 2.0;
            if damping > MAX_DAMPING {
                break;
            }
            continue;
        };
        let gain = current_chi2 - found_chi2;
        params = trial;
        current = found;
        current_chi2 = found_chi2;
        normal = normal_equations(&current);
        // The gain of the linear model of the undamped step, `g.N^-1.g`.
        let (matrix, gradient) = &normal;
        let left =
            scaled_solve(matrix, gradient, 0.0).map_or(f64::INFINITY, |step| step.dot(gradient));
        if left < CHI2_TOLERANCE {
            break;
        }
        let scale = (1.0 - (2.0 * gain / predicted - 1.0).powi(3)).max(SMALLEST_SCALE);
        damping = (damping * scale).max(MIN_DAMPING);
        raise = 2.0;
    }
    Some((params, current))
}

/// The chi-square of `sights` under the two-body orbit of `state`, with the
/// normal matrix and gradient of their residuals, see [`normal_equations`].
/// `None` if a residual or its derivative cannot be evaluated.
pub(crate) fn normal_at(
    state: &State<Equatorial, SunCenter>,
    sights: &[Sight],
) -> Option<(f64, Matrix6<f64>, Vector6<f64>)> {
    let seen: Vec<Seen> = sights
        .iter()
        .map(|sight| seen(state, sight))
        .collect::<Option<_>>()?;
    let (matrix, gradient) = normal_equations(&seen);
    Some((
        chi2(seen.iter().map(|seen| &seen.residual)),
        matrix,
        gradient,
    ))
}

/// The normal matrix `J^T J` and the gradient `J^T r` of the residuals `r`
/// of the sights, with their Jacobian `J`.
fn normal_equations(seen: &[Seen]) -> (Matrix6<f64>, Vector6<f64>) {
    let mut matrix = Matrix6::zeros();
    let mut gradient = Vector6::zeros();
    for Seen { residual, jacobian } in seen {
        matrix += jacobian.tr_mul(jacobian);
        gradient += jacobian.tr_mul(residual);
    }
    (matrix, gradient)
}

/// Leverage of each sight, `trace(J_s (J^T J)^-1 J_s^T)` for its rows `J_s`
/// of the Jacobian. `None` if `J^T J` is singular.
fn leverages(seen: &[Seen]) -> Option<Vec<f64>> {
    let inverse = inverse(&normal_equations(seen).0)?;
    Some(
        seen.iter()
            .map(|Seen { jacobian, .. }| (jacobian * inverse * jacobian.transpose()).trace())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::kepler::light_time_correct;

    use crate::tests::synthetic::{Rng, circular, sights};
    use crate::unit_fit::unit_rows;

    /// Sights of `object` from an observer on a circular orbit at 1 AU, at
    /// these days from `epoch`, with about a tenth of an arcsecond of noise.
    fn noisy_sights(object: &State<Equatorial, SunCenter>, epoch: f64, days: &[f64]) -> Vec<Sight> {
        let observer = circular(epoch.into(), 1.0, 0.0, 0.0);
        let sigma = (0.1_f64 / 3600.0).to_radians();
        let mut rng = Rng(11);
        sights(object, &observer, epoch, days, sigma, || {
            rng.normal() * sigma
        })
    }

    #[test]
    fn far_starts_reach_the_minimum() {
        // A main belt object at opposition and at quadrature, a near-Earth
        // object at 1.05 AU, an object 0.02 AU from the observer, and a
        // distant one, each seen nightly for 15 and for 30 nights. From the
        // orbit through the first and last sight at half to three times
        // their true ranges, the fit reaches the minimum that Gauss-Newton
        // steps reach from the truth. Nearer starts can settle on a second
        // minimum: at a third of the range, the object at quadrature lands on
        // an orbit near the observer's, whose residuals confirmation turns
        // down as hopeless.
        let epoch = 2460000.5;
        let earth = circular(epoch.into(), 1.0, 0.0, 0.0);
        let close = State {
            pos: earth.pos + Vector::new([0.3, 0.8, 0.2]).normalize() * 0.02,
            vel: earth.vel
                + Vector::new([-0.5, 0.2, 0.9]).normalize() * (10.0 * 86400.0 / 1.495_978_707e8),
            ..earth.clone()
        };
        let objects = [
            (
                "main belt at opposition",
                circular(epoch.into(), 2.5, 10.0, 0.5),
            ),
            (
                "main belt at quadrature",
                circular(epoch.into(), 2.5, 10.0, 70.0),
            ),
            ("near-Earth", circular(epoch.into(), 1.05, 10.0, 2.0)),
            ("close", close),
            ("distant", circular(epoch.into(), 40.0, 10.0, 10.0)),
        ];
        for (name, object) in &objects {
            for nights in [15, 30] {
                let days: Vec<f64> = (0..nights)
                    .map(|n| f64::from(n) - f64::from(nights / 2))
                    .collect();
                let sights = noisy_sights(object, epoch, &days);
                let chosen: Vec<&Sight> = sights.iter().collect();
                let middle = f64::midpoint(days[0], days[days.len() - 1]) + epoch;
                let truth = propagate_two_body(object, middle.into()).unwrap();
                let mut params = to_params(&truth);
                let mut minimum = evaluate(&truth, &params, &chosen).unwrap();
                for _ in 0..20 {
                    let (matrix, gradient) = normal_equations(&minimum);
                    let step = scaled_solve(&matrix, &gradient, 0.0).unwrap();
                    let moved = params - step;
                    let found = evaluate(&truth, &moved, &chosen).unwrap();
                    let chi2_of = |seen: &[Seen]| chi2(seen.iter().map(|seen| &seen.residual));
                    if chi2_of(&found) >= chi2_of(&minimum) {
                        break;
                    }
                    (params, minimum) = (moved, found);
                }
                let minimum = chi2(minimum.iter().map(|seen| &seen.residual));
                let range = |sight: &Sight| {
                    let at = propagate_two_body(object, sight.time).unwrap();
                    (light_time_correct(&at, &sight.observer).unwrap().pos - sight.observer).norm()
                };
                let (first, last) = (&sights[0], &sights[sights.len() - 1]);
                for k in [0.5, 2.0, 3.0] {
                    let Ok(start) = crate::from_ranges::orbit_through(
                        first,
                        last,
                        k * range(first),
                        k * range(last),
                    ) else {
                        continue;
                    };
                    let (state, kept, _) = fit_two_body(&start, &sights, f64::INFINITY).unwrap();
                    assert_eq!(kept.len(), sights.len());
                    let found = unit_rows(&chosen, &state).unwrap();
                    let found = chi2(found.iter().map(|row| &row.residual));
                    assert!(
                        found < minimum + 0.01 * minimum.max(1.0),
                        "{name}, {nights} nights, {k} of the range: {found} against {minimum}"
                    );
                }
            }
        }
    }

    #[test]
    fn strays_are_dropped() {
        // An object 2.5 AU from the Sun near opposition, seen from 1 AU in
        // paired visits on five nights with 0.1 arcseconds of noise, and two
        // sights 30 arcseconds off its track between the visits. From the true
        // state, the fit keeps every sight of the object and drops the two.
        let epoch = 2460000.5;
        let object = circular(epoch.into(), 2.5, 10.0, 0.5);
        let mut times: Vec<f64> = (-2..=2)
            .flat_map(|night| [0.0, 0.028].map(|t| f64::from(night) + t))
            .collect();
        times.extend([-1.014, 1.014]);
        let mut sights = noisy_sights(&object, epoch, &times);
        for sight in &mut sights[10..] {
            let across = sight.direction.cross(&Vector::new([0.0, 0.0, 1.0]));
            sight.direction =
                (sight.direction + across.normalize() * (300.0 * sight.sigma)).normalize();
        }
        let (_, kept, _) = fit_two_body(&object, &sights, 9.0).unwrap();
        assert_eq!(kept, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn strays_past_a_third_are_not_all_dropped() {
        // An object seen nightly for six nights, and four sights 30
        // arcseconds off its track between them, more than a third of the
        // ten. The fit drops strays only down to two thirds of the sights,
        // so it keeps seven, a stray among them.
        let epoch = 2460000.5;
        let object = circular(epoch.into(), 2.5, 10.0, 0.5);
        let mut sights = noisy_sights(&object, epoch, &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]);
        let strays = noisy_sights(&object, epoch, &[-2.5, -0.5, 0.5, 1.5]);
        sights.extend(strays.into_iter().map(|sight| {
            let across = sight.direction.cross(&Vector::new([0.0, 0.0, 1.0]));
            Sight {
                direction: (sight.direction + across.normalize() * (300.0 * sight.sigma))
                    .normalize(),
                ..sight
            }
        }));
        let (_, kept, _) = fit_two_body(&object, &sights, 9.0).unwrap();
        assert_eq!(kept.len(), fewest_kept(sights.len()));
        assert_eq!(kept.len(), 7);
        assert!(kept.iter().any(|&k| k >= 6), "{kept:?}");
    }

    #[test]
    fn bound_fits_stay_bound() {
        // An object on an unbound orbit seen for ten days without noise,
        // started from its own position with its velocity cut to just below
        // escape speed. The best orbit for the sights is the true one, which
        // is unbound, but from a bound start the fit keeps to bound orbits,
        // and ends on one which fits worse.
        let epoch = 2460000.5;
        let earth = circular(epoch.into(), 1.0, 0.0, 0.0);
        let base = circular(epoch.into(), 1.8, 10.0, 20.0);
        let escape = (2.0 * GMS / 1.8_f64).sqrt();
        let direction = (base.vel.normalize() + base.pos.normalize() * 0.3).normalize();
        let object = State {
            vel: direction * (1.3 * escape),
            ..base.clone()
        };
        let days: Vec<f64> = (-5..=5).map(f64::from).collect();
        let sights = sights(&object, &earth, epoch, &days, 1e-6, || 0.0);
        let start = State {
            vel: direction * (0.99 * escape),
            ..base
        };
        let (state, kept, rms) = fit_two_body(&start, &sights, f64::INFINITY).unwrap();
        assert_eq!(kept.len(), sights.len());
        assert!(is_bound(&to_params(&state)), "{state:?}");
        let truth = propagate_two_body(&object, state.epoch).unwrap();
        let chosen: Vec<&Sight> = sights.iter().collect();
        let best = reduced_rms(&evaluate(&truth, &to_params(&truth), &chosen).unwrap());
        assert!(best < 1e-3 && rms > 1.0, "{best} against {rms}");
    }
}
