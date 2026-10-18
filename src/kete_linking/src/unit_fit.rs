// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The least-squares orbit of a unit under one hypothesis.

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::time::{TDB, Time};

use crate::hypothesis::{Solution, bound_within_noise, solve};
use crate::linalg::{chi2, dot, inverse, scaled_solve};
use crate::observe::residual_at;
use crate::{Hypothesis, Sight};

/// Most Gauss-Newton steps of the fit of a unit, see [`solve_unit`].
const UNIT_STEPS: usize = 2;

/// The fit of a unit takes another step when the chi-square evaluated after
/// a step differs from the one its linear model predicted by more than this
/// share of the predicted one, or of one if that is less, see
/// [`solve_unit`]. A smaller difference moves no residual by a share of a
/// sigma that the grouping thresholds of several sigma can tell.
const UNIT_AGREEMENT: f64 = 0.1;

/// Two unit vectors spanning the plane normal to `normal`.
pub(crate) fn plane_axes(normal: &Vector<Equatorial>) -> [Vector<Equatorial>; 2] {
    let other = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let across = normal.cross(&Vector::new(other)).normalize();
    let along = normal.cross(&across);
    [across, along]
}

/// The orbit of a unit under a hypothesis, with what grouping needs of it.
#[derive(Debug)]
pub(crate) struct Fit {
    /// State at the reference time.
    pub(crate) state: State<Equatorial, SunCenter>,

    /// Each sight under the orbit, in the order of the unit.
    pub(crate) rows: Vec<Row>,

    /// Covariance of the four numbers the hypothesis leaves free, at the
    /// reference time: the position along the two [`plane_axes`] in AU, then
    /// the velocity along them in AU per day.
    pub(crate) covariance: [[f64; 4]; 4],
}

/// One sight of a unit under an orbit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Row {
    /// Predicted less observed direction, over the sigma of the sight.
    pub(crate) residual: [f64; 3],

    /// Time light left the object.
    pub(crate) time: Time<TDB>,

    /// Range along the sight, in AU.
    pub(crate) range: f64,
}

/// The orbit of a unit, two or more sights in time order, under a
/// hypothesis.
///
/// Two sights give four measurements for the four numbers the hypothesis
/// leaves free, and their orbit is the exact one of [`solve`]. More give
/// more, and the orbit is the least-squares fit of the directions of all of
/// them, with exact two-body motion, light time and the observer of each. It
/// starts from the exact orbit of the first and last sight, and takes at
/// most [`UNIT_STEPS`] Gauss-Newton steps within the hypothesis, so that only
/// the position on its plane and the velocity along the plane move.
///
/// The Jacobian is that of the motion from the position and velocity at the
/// reference time to each sight's direction at the range and time it is seen,
/// to second order in the time, see [`columns`]. When the reference time is
/// near the sights it is the full one to well within the uncertainty of a unit,
/// the residuals are linear in the step to the same degree, and one step
/// reaches the least-squares orbit. Far from the sights, and near the Sun, the
/// terms it leaves out move the sights by more than their misfit. So the
/// residuals are evaluated at the stepped orbit, and when their chi-square
/// differs from the one the step predicted by more than [`UNIT_AGREEMENT`] of
/// it, another step is taken from there, up to [`UNIT_STEPS`] in all. A step
/// that does not lower the chi-square is not taken, so a fit whose first step
/// is spoiled by those terms keeps the exact orbit of the first and last sight.
/// The returned orbit and residuals always agree. The covariance is the inverse
/// of the normal matrix of that Jacobian at the returned orbit.
///
/// # Errors
///
/// * The errors of [`solve`] for the first and last sight: an
///   [`Error::ValueError`] for a sight too little aligned with the direction
///   or a plane too far away, and an [`Error::Convergence`] for no orbit.
/// * [`Error::Convergence`] if the orbit after the steps is not bound by the
///   rule of [`solve`] for the first and last sight, or the residuals or the
///   covariance cannot be evaluated.
pub(crate) fn solve_unit(
    sights: &[Sight],
    unit: &[usize],
    hypothesis: &Hypothesis,
) -> KeteResult<Fit> {
    let fail = || Error::Convergence("Unit under hypothesis: no fit".into());
    let (first, last) = (&sights[unit[0]], &sights[unit[unit.len() - 1]]);
    let Solution { mut state, ranges } = solve(first, last, hypothesis)?;
    let axes = plane_axes(&hypothesis.direction);
    // The normal matrix at the returned orbit, and its rows.
    let (normal, rows) = if unit.len() == 2 {
        let rows = [(first, ranges[0]), (last, ranges[1])].map(|(sight, range)| Row {
            residual: [0.0; 3],
            time: sight.emission_time(range),
            range,
        });
        let (normal, _) = normal_equations(&[first, last], &rows, &state, &axes);
        (normal, rows.to_vec())
    } else {
        let chosen: Vec<&Sight> = unit.iter().map(|&i| &sights[i]).collect();
        let mut rows = unit_rows(&chosen, &state).ok_or_else(fail)?;
        for _ in 0..UNIT_STEPS {
            let (normal, gradient) = normal_equations(&chosen, &rows, &state, &axes);
            let step = scaled_solve(&normal, gradient.map(|g| -g), 0.0).ok_or_else(fail)?;
            let predicted: f64 = chosen
                .iter()
                .zip(&rows)
                .map(|(sight, row)| {
                    let columns = columns(sight, row, &state, &axes);
                    let residual: [f64; 3] = std::array::from_fn(|k| {
                        row.residual[k] + (0..4).map(|i| columns[i][k] * step[i]).sum::<f64>()
                    });
                    dot(&residual, &residual)
                })
                .sum();
            let mut moved = state.clone();
            moved.pos += axes[0] * step[0] + axes[1] * step[1];
            moved.vel += axes[0] * step[2] + axes[1] * step[3];
            let Some(found) = unit_rows(&chosen, &moved) else {
                break;
            };
            let after = chi2(found.iter().map(|row| &row.residual));
            if after >= chi2(rows.iter().map(|row| &row.residual)) {
                break;
            }
            (state, rows) = (moved, found);
            if (after - predicted).abs() <= UNIT_AGREEMENT * predicted.max(1.0) {
                break;
            }
        }
        // The steps can carry the orbit past the speed the end pair allows.
        let (start, end) = (&rows[0], &rows[rows.len() - 1]);
        let between = (end.time - start.time).elapsed.abs();
        if !bound_within_noise(&state, first, last, [start.range, end.range], between) {
            return Err(Error::Convergence(
                "Unit under hypothesis: the orbit is not bound to the Sun".into(),
            ));
        }
        let (normal, _) = normal_equations(&chosen, &rows, &state, &axes);
        (normal, rows)
    };
    Ok(Fit {
        covariance: inverse(&normal).ok_or_else(fail)?,
        state,
        rows,
    })
}

/// Each sight under the orbit of `state`, if every residual can be
/// evaluated.
pub(crate) fn unit_rows(
    sights: &[&Sight],
    state: &State<Equatorial, SunCenter>,
) -> Option<Vec<Row>> {
    sights
        .iter()
        .map(|sight| {
            let (residual, at) = residual_at(state, sight)?;
            Some(Row {
                residual,
                time: at.epoch,
                range: (at.pos - sight.observer).norm(),
            })
        })
        .collect()
}

/// The Jacobian of the residual of one sight over the four free numbers of
/// `state` within the hypothesis, see [`Fit`], as one column of three for
/// each, at the time and range of its row.
///
/// The motion from `state` to the sight is taken to second order in the
/// time: a change of position `dr` and velocity `dv` at the reference time
/// moves the object by `(I + G t^2 / 2) dr + (I + G t^2 / 6) t dv`, with `G`
/// the gradient of the pull of the Sun there. The pull matters when the
/// reference time is days from the sights: a change of the four numbers
/// which barely moves the sights can move the position and the velocity a
/// lot each, and the pull then parts their effects.
fn columns(
    sight: &Sight,
    row: &Row,
    state: &State<Equatorial, SunCenter>,
    axes: &[Vector<Equatorial>; 2],
) -> [[f64; 3]; 4] {
    let dt = (row.time - state.epoch).elapsed;
    let dir = sight.direction;
    let weight = 1.0 / (row.range * sight.sigma);
    let distance = state.pos.norm();
    let unit = state.pos / distance;
    let pull = |e: Vector<Equatorial>| (unit * (3.0 * unit.dot(&e)) - e) * (GMS / distance.powi(3));
    let seen =
        |moved: Vector<Equatorial>| <[f64; 3]>::from((moved - dir * dir.dot(&moved)) * weight);
    let [a, b] = axes.map(|e| {
        (
            seen(e + pull(e) * (0.5 * dt * dt)),
            seen((e + pull(e) * (dt * dt / 6.0)) * dt),
        )
    });
    [a.0, b.0, a.1, b.1]
}

/// The normal matrix and the gradient of the residuals of `sights` over the
/// four free numbers of `state` within the hypothesis, with the Jacobian of
/// [`columns`].
fn normal_equations(
    sights: &[&Sight],
    rows: &[Row],
    state: &State<Equatorial, SunCenter>,
    axes: &[Vector<Equatorial>; 2],
) -> ([[f64; 4]; 4], [f64; 4]) {
    let mut normal = [[0.0; 4]; 4];
    let mut gradient = [0.0; 4];
    for (sight, row) in sights.iter().zip(rows) {
        let columns = columns(sight, row, state, axes);
        for i in 0..4 {
            gradient[i] += dot(&columns[i], &row.residual);
            for j in 0..4 {
                normal[i][j] += dot(&columns[i], &columns[j]);
            }
        }
    }
    (normal, gradient)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::kepler::propagate_two_body;

    use crate::hypothesis::axes;
    use crate::tests::synthetic::{Rng, circular, sights};

    #[test]
    fn tracklet_fit_finds_the_truth() {
        // A main belt object near opposition, and an object 0.3 AU from the Sun
        // moving fast, each seen from 1 AU in a tracklet of nine sights over
        // two hours with 0.1 arcseconds of noise, solved under the true
        // hypothesis with the reference time in the tracklet and up to fifteen
        // days before it. The returned residuals are those of the returned
        // orbit. That orbit is on the plane of the hypothesis, fits the sights
        // about as well as the truth and, with the reference time in the
        // tracklet, better than the orbit through the first and last sight
        // alone, and has the truth within four sigma on each free number by its
        // own covariance.
        let epoch = 2460000.5;
        let earth = circular(epoch.into(), 1.0, 0.0, 0.0);
        let speed = (1.9 * GMS / 0.3).sqrt();
        let near_sun = State {
            pos: Vector::new([0.3, 0.05, 0.02]),
            vel: Vector::new([-0.002, 1.0, 0.1]).normalize() * speed,
            ..earth.clone()
        };
        let sigma = (0.1_f64 / 3600.0).to_radians();
        let days: Vec<f64> = (0..9).map(|k| f64::from(k) / 96.0).collect();
        let unit: Vec<usize> = (0..9).collect();
        for (name, truth) in [
            ("main belt", circular(epoch.into(), 2.5, 10.0, 0.5)),
            ("near the Sun", near_sun),
        ] {
            let mut rng = Rng(5);
            let sights = sights(&truth, &earth, epoch, &days, sigma, || rng.normal() * sigma);
            let chosen: Vec<&Sight> = sights.iter().collect();
            let rows_chi2 = |rows: &[Row]| chi2(rows.iter().map(|row| &row.residual));
            let truth_chi2 = rows_chi2(&unit_rows(&chosen, &truth).unwrap());
            let direction = axes()
                .into_iter()
                .max_by(|a, b| {
                    a.dot(&sights[0].direction)
                        .total_cmp(&b.dot(&sights[0].direction))
                })
                .unwrap();
            for ahead in [0.0, -5.0, -10.0, -15.0] {
                let reference: Time<TDB> = (epoch + 1.0 / 24.0 + ahead).into();
                let origin = propagate_two_body(&earth, reference).unwrap();
                let at = propagate_two_body(&truth, reference).unwrap();
                let hypothesis = Hypothesis {
                    depth: direction.dot(&(at.pos - origin.pos)),
                    depth_rate: direction.dot(&(at.vel - origin.vel)),
                    origin: origin.clone(),
                    direction,
                };
                let case = format!("{name}, {ahead} days");
                let fit = solve_unit(&sights, &unit, &hypothesis).unwrap();
                let evaluated = unit_rows(&chosen, &fit.state).unwrap();
                for (row, exact) in fit.rows.iter().zip(&evaluated) {
                    assert_eq!(row.residual, exact.residual, "{case}");
                }
                let found = rows_chi2(&fit.rows);
                let ends = solve_unit(&sights, &[0, 8], &hypothesis).unwrap();
                let through_ends = rows_chi2(&unit_rows(&chosen, &ends.state).unwrap());
                // Far from the tracklet the steps can be spoiled, and the fit
                // then keeps the orbit through the ends, however near the truth
                // that is.
                let near_truth = found < truth_chi2 + 1.0;
                if ahead == 0.0 {
                    assert!(near_truth, "{case}: {found} against {truth_chi2}");
                    assert!(
                        found < through_ends,
                        "{case}: {found} against {through_ends}"
                    );
                } else {
                    assert!(
                        near_truth || found == through_ends,
                        "{case}: {found} against {truth_chi2} and {through_ends}"
                    );
                }

                let depth = direction.dot(&(fit.state.pos - origin.pos));
                assert!((depth - hypothesis.depth).abs() < 1e-9, "{case}");
                let axes = plane_axes(&direction);
                let miss = [
                    axes[0].dot(&(fit.state.pos - at.pos)),
                    axes[1].dot(&(fit.state.pos - at.pos)),
                    axes[0].dot(&(fit.state.vel - at.vel)),
                    axes[1].dot(&(fit.state.vel - at.vel)),
                ];
                for (k, miss) in miss.iter().enumerate() {
                    let error = fit.covariance[k][k].sqrt();
                    assert!(error.is_finite() && error > 0.0, "{case}, number {k}");
                    assert!(
                        miss.abs() < 4.0 * error,
                        "{case}, number {k}: {miss} against {error}"
                    );
                }
            }
        }
    }
}
