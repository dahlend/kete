// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The orbit through two lines of sight under one hypothesis.

use std::f64::consts::SQRT_2;

use kete_core::constants::{C_AU_PER_DAY_INV, GMS};
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::kepler::propagate_two_body;
use kete_core::prelude::{Error, KeteResult, State};

use crate::Sight;
use crate::from_ranges::orbit_through;
use crate::sight::{MAX_DAYS, is_unit};

/// Smallest cosine of the angle between a line of sight and the direction of
/// a hypothesis it can be solved under.
///
/// A depth becomes a range along the sight by dividing by this cosine, so
/// errors grow as it falls. Every direction is within a cosine of 0.577 of
/// one of the six coordinate axes of the Equatorial frame, both ways, the
/// directions of the hypotheses of [`crate::hypothesis_grid`], and the margin
/// below that lets an object which moves on the sky keep one axis for all of
/// its sights.
pub const MIN_ALIGNMENT: f64 = 0.45;

/// The six hypothesis directions which together cover the sky: both ways
/// along each coordinate axis.
pub(crate) fn axes() -> [Vector<Equatorial>; 6] {
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
    .map(Vector::new)
}

/// An assertion of the two things a pair of sights does not measure.
///
/// At the reference time, which is the epoch of `origin`, the object is on
/// the plane `direction . (position - origin position) = depth`, and moves
/// through it at `direction . (velocity - origin velocity) = depth_rate`.
///
/// Both values are nearly linear in the two ranges of a pair, so a pair has
/// one orbit under a hypothesis. The depth can have either sign.
#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    /// Point the depth is measured from, with its velocity, with respect to
    /// the Sun in AU and AU per day. Its epoch is the reference time.
    pub origin: State<Equatorial, SunCenter>,

    /// Unit vector the depth is measured along.
    pub direction: Vector<Equatorial>,

    /// Distance of the object from the origin along the direction, in AU.
    pub depth: f64,

    /// Rate of change of the depth, in AU per day.
    pub depth_rate: f64,
}

/// Most steps of the solve.
const STEPS: usize = 30;

/// How far the speed of the straight line which meets both sights under a
/// hypothesis may pass the escape speed, as a multiple of how far the pull of
/// the Sun between the sights and the reference time can move that line. A
/// hypothesis whose line is faster than that is refused before any
/// propagation. The test `unbound_prefilter_margin` prints the largest
/// share of it which the bound orbits of the scenarios use.
const PULL_MARGIN: f64 = 8.0;

/// Limit on how far the orbit of a step of the solve may be past the speed
/// [`solve`] allows, for the solve to go on, see [`past_escape`].
///
/// The limit is a multiple of how much the next step can change that speed.
/// The steps after the next one shrink with the miss, so ten times the next
/// step covers all of them with a wide margin. An orbit further past the
/// allowed speed than that cannot come back under it, and the hypothesis is
/// refused. Most hypotheses which pass [`PULL_MARGIN`] but have no bound
/// orbit for a pair are refused this way after a step or two, rather than
/// after the solve converges.
const REMAINING_MARGIN: f64 = 10.0;

/// Largest miss of an accepted orbit: in depth as a fraction of the range,
/// and in depth rate as a fraction of the speed.
///
/// A range error of this fraction moves the object by as much as an error in
/// direction of the same number of radians. That is 0.02 milliarcseconds,
/// below the error of any astrometry.
const TOLERANCE: f64 = 1e-10;

/// The orbit of a pair under a hypothesis, and the ranges it was built from.
#[derive(Debug)]
pub(crate) struct Solution {
    /// State at the reference time.
    pub(crate) state: State<Equatorial, SunCenter>,

    /// Ranges along the two sights, in AU.
    pub(crate) ranges: [f64; 2],
}

/// An error unless the hypothesis is finite, with a reference time within
/// 9e15 days of zero, and its direction is a unit vector.
pub(crate) fn check_hypothesis(hypothesis: &Hypothesis) -> KeteResult<()> {
    let Hypothesis {
        origin,
        direction,
        depth,
        depth_rate,
    } = hypothesis;
    let finite = depth.is_finite() && depth_rate.is_finite() && origin.is_finite();
    if !finite {
        return Err(Error::ValueError("the hypothesis must be finite".into()));
    }
    if origin.epoch.jd().abs() >= MAX_DAYS {
        return Err(Error::ValueError(
            "the reference time must be within 9e15 days of JD 0".into(),
        ));
    }
    if !is_unit(direction) {
        return Err(Error::ValueError(
            "the direction must be a unit vector".into(),
        ));
    }
    Ok(())
}

/// The two-body orbit about the Sun of the object seen in two lines of sight,
/// if a hypothesis holds, as its state at the reference time, and the ranges
/// along the two sights it was built from.
///
/// Only bound orbits are returned. An object on an unbound orbit about the
/// Sun is out of scope, and a hypothesis which needs one is counted as wrong.
/// The speed may pass the escape speed by three times the velocity error
/// that astrometric noise gives the pair, `sqrt(2) sigma range / gap` on
/// each axis of the sky, so that noise does not refuse a bound object while
/// a hypothesis far from any bound orbit still fails. A hypothesis whose
/// orbit without gravity, a straight line, is faster than any bound orbit
/// near it is refused before the first step, and one whose orbit after a
/// step is past that speed by far more than the steps still to come can
/// change is refused there.
///
/// The unknowns are the two ranges. Each step builds the exact two-body
/// orbit through them, and corrects them by its miss in depth and depth rate
/// at the reference time, moved back to each sight in a straight line. A
/// step shrinks the miss by about the square of the angle the object sweeps
/// about the Sun between the sights and the reference time, so the search
/// window has to be short against the orbital period. Sights from observers
/// far apart converge more slowly. The tests `hypothesis_convergence` and
/// `hypothesis_cost` print where the solve stops finding the orbit and what
/// it costs.
///
/// The hypothesis has to pass [`check_hypothesis`] and both sights have to be
/// [`Sight::is_usable`], as [`crate::link()`] checks of its inputs.
///
/// # Arguments
///
/// * `a` - First line of sight.
/// * `b` - Second line of sight, at any other time and from any observer.
/// * `hypothesis` - The assertion to solve under.
///
/// # Errors
///
/// * [`Error::ValueError`] if either sight is less aligned with the direction
///   than [`MIN_ALIGNMENT`], or the plane meets a sight farther away than
///   light travels in 9e15 days.
/// * [`Error::Convergence`] if there is no orbit: the misses stop shrinking,
///   the object would be behind an observer, the orbit is unbound, the two
///   positions are farther apart than light travels between them, or the
///   two-body propagation fails.
pub(crate) fn solve(a: &Sight, b: &Sight, hypothesis: &Hypothesis) -> KeteResult<Solution> {
    accepted(a, b, converge(a, b, hypothesis, true)?)
}

/// The orbit of [`converge`] if it is in front of both observers and bound.
fn accepted(a: &Sight, b: &Sight, solution: Solution) -> KeteResult<Solution> {
    let Solution { state, ranges } = &solution;
    if !(ranges[0] > 0.0 && ranges[1] > 0.0) {
        return Err(Error::Convergence(
            "Orbit from hypothesis: the object would be behind an observer".into(),
        ));
    }
    let between = (b.emission_time(ranges[1]) - a.emission_time(ranges[0]))
        .elapsed
        .abs();
    if !bound_within_noise(state, a, b, *ranges, between) {
        return Err(Error::Convergence(
            "Orbit from hypothesis: the orbit is not bound to the Sun".into(),
        ));
    }
    Ok(solution)
}

/// Whether the speed of `state` is under the escape speed plus three times
/// the velocity error that astrometric noise gives the two sights `a` and
/// `b` at `ranges`, whose light left them `between` days apart, see
/// [`velocity_noise`].
pub(crate) fn bound_within_noise(
    state: &State<Equatorial, SunCenter>,
    a: &Sight,
    b: &Sight,
    ranges: [f64; 2],
    between: f64,
) -> bool {
    let noise = velocity_noise(a, b, ranges, between);
    let escape = (2.0 * GMS / state.pos.norm()).sqrt();
    state.vel.norm() < escape + 3.0 * noise
}

/// The velocity error that astrometric noise gives the orbit through two
/// sights `a` and `b` at `ranges`, whose light left them `between` days
/// apart, on each axis of the sky: `sqrt(2)` times the larger uncertainty
/// times the larger range over the time between. The ranges can be negative
/// during the solve, so their sizes are compared.
fn velocity_noise(a: &Sight, b: &Sight, ranges: [f64; 2], between: f64) -> f64 {
    SQRT_2 * a.sigma.max(b.sigma) * ranges[0].abs().max(ranges[1].abs()) / between
}

/// The starting ranges of the solve, and the straight line through them.
#[derive(Debug)]
struct Start {
    /// Ranges along the two sights of the straight line, in AU.
    ranges: [f64; 2],

    /// Cosine between each sight and the direction.
    aligned: [f64; 2],

    /// Speed of the straight line, in AU per day.
    speed: f64,

    /// Escape speed at the plane plus the velocity noise allowed for, in AU
    /// per day.
    limit: f64,

    /// How far the pull of the Sun can move the line, in AU per day.
    pull: f64,
}

/// The checks of [`solve`] on its inputs, and the start of its solve.
fn start(a: &Sight, b: &Sight, hypothesis: &Hypothesis) -> KeteResult<Start> {
    let Hypothesis {
        origin,
        direction,
        depth,
        depth_rate,
    } = hypothesis;
    let sights = [a, b];
    let aligned = sights.map(|sight| direction.dot(&sight.direction));
    if !(aligned[0] >= MIN_ALIGNMENT && aligned[1] >= MIN_ALIGNMENT) {
        return Err(Error::ValueError(format!(
            "Orbit from hypothesis: a sight is less aligned with the direction than {MIN_ALIGNMENT}"
        )));
    }

    // Each range starts where the sight crosses the plane at the time light
    // left that point, with the origin and the depth moved in a straight
    // line: the orbit without gravity, a straight line which meets each
    // sight where the plane, moved at the rate of the hypothesis, is then.
    // The object is no nearer the Sun than the plane is at the reference
    // time, so the escape speed there bounds the speed of a bound orbit.
    // Gravity moves the velocity of the orbit from that of the line by the
    // pull of the Sun over the time from the sights to the reference. It
    // also moves each point of the line along its own sight, by that pull
    // times the square of the time, which turns the line by the angle
    // between the two sights over the time between them; for sights from
    // two observers that term is the larger. The solve allows three times
    // the velocity noise past escape, at its final ranges, and half again
    // covers the ranges growing on the way. A line faster than all of that
    // has no bound orbit near it.
    let rate = depth_rate + direction.dot(&origin.vel);
    let ranges = [0, 1].map(|i| {
        let ahead = (origin.epoch - sights[i].time).elapsed;
        let offset = sights[i].observer - (origin.pos - origin.vel * ahead);
        (depth - depth_rate * ahead - direction.dot(&offset))
            / (aligned[i] + rate * C_AU_PER_DAY_INV)
    });
    // Times are only taken of points light reaches the observer from within
    // the days this crate does arithmetic on.
    let near = ranges
        .iter()
        .all(|range| range.abs() * C_AU_PER_DAY_INV < MAX_DAYS);
    if !near {
        return Err(Error::ValueError(
            "Orbit from hypothesis: the plane meets a sight farther away than light travels in \
             9e15 days"
                .into(),
        ));
    }
    let [(time_a, at_a), (time_b, at_b)] = [0, 1].map(|i| {
        (
            sights[i].emission_time(ranges[i]),
            sights[i].position(ranges[i]),
        )
    });
    let gap = (time_b - time_a).elapsed.abs();
    let lever = (origin.epoch - time_a)
        .elapsed
        .abs()
        .max((origin.epoch - time_b).elapsed.abs());
    let plane = (depth + direction.dot(&origin.pos)).abs();
    let nearest = at_a.norm().min(at_b.norm()).min(plane);
    let turn = (b.direction - a.direction).norm();
    let pull = GMS / (nearest * nearest) * (lever + gap + turn * lever * lever / gap);
    let vel_noise = velocity_noise(a, b, ranges, gap);
    Ok(Start {
        ranges,
        aligned,
        speed: (at_b - at_a).norm() / gap,
        limit: (2.0 * GMS / plane).sqrt() + 3.0 * 1.5 * vel_noise,
        pull,
    })
}

/// Share of [`PULL_MARGIN`] a pair uses under a hypothesis: how far the speed
/// of the straight line of the start of the solve passes the escape speed and
/// the noise, over the margin times the pull of the Sun. The solve refuses
/// the hypothesis before its first step when this passes one.
#[cfg(test)]
pub(crate) fn margin_share(a: &Sight, b: &Sight, hypothesis: &Hypothesis) -> KeteResult<f64> {
    let Start {
        speed, limit, pull, ..
    } = start(a, b, hypothesis)?;
    Ok((speed - limit) / (PULL_MARGIN * pull))
}

/// The orbit of [`solve`] before it is checked to be bound and in front of
/// both observers. With `early`, hypotheses whose orbit is far from bound
/// are refused here too: before the first step, see [`PULL_MARGIN`], and
/// after a step, see [`past_escape`].
fn converge(a: &Sight, b: &Sight, hypothesis: &Hypothesis, early: bool) -> KeteResult<Solution> {
    let Start {
        mut ranges,
        aligned,
        speed,
        limit,
        pull,
    } = start(a, b, hypothesis)?;
    if early && speed > limit + PULL_MARGIN * pull {
        return Err(Error::Convergence(
            "Orbit from hypothesis: the orbit is not bound to the Sun".into(),
        ));
    }
    let Hypothesis {
        origin,
        direction,
        depth,
        depth_rate,
    } = hypothesis;
    let sights = [a, b];

    // Sizes of the last two misses, in units of the goal.
    let mut before = [f64::INFINITY; 2];
    for _ in 0..STEPS {
        // A pair of ranges without an orbit is a hypothesis without one.
        let orbit = orbit_through(a, b, ranges[0], ranges[1])
            .map_err(|err| Error::Convergence(err.to_string()))?;
        let state = propagate_two_body(&orbit, origin.epoch)
            .map_err(|err| Error::Convergence(err.to_string()))?;
        let miss = direction.dot(&(state.pos - origin.pos)) - depth;
        let rate_miss = direction.dot(&(state.vel - origin.vel)) - depth_rate;

        // Days from the time light left each range to the reference time.
        let ahead = [0, 1].map(|i| (origin.epoch - sights[i].emission_time(ranges[i])).elapsed);
        // Taken as a difference of times rather than of `ahead`, which can
        // round to zero when the reference time is far from the sights.
        let between = (b.emission_time(ranges[1]) - a.emission_time(ranges[0]))
            .elapsed
            .abs();
        let apart = between > 0.0;
        if !apart {
            return Err(Error::Convergence(
                "Orbit from hypothesis: the two emission times are the same".into(),
            ));
        }
        // Rounding of the positions, which over the time between the two
        // emissions is the best the velocity can be known. Building a
        // position from observer, direction and range, and moving it with
        // the two-body propagation, rounds it a few times, each by up to
        // `f64::EPSILON` of its distance from the Sun. Sixteen such roundings
        // bound that, so no miss within it is left for the steps to remove.
        let rounding = 16.0 * f64::EPSILON * state.pos.norm();
        let depth_goal = TOLERANCE * ranges[0].abs().min(ranges[1].abs())
            + rounding * (1.0 + ahead[0].abs().max(ahead[1].abs()) / between);
        let rate_goal = TOLERANCE * state.vel.norm() + rounding / between;
        let size = (miss / depth_goal).abs().max((rate_miss / rate_goal).abs());

        if size <= 1.0 {
            return Ok(Solution { state, ranges });
        }
        // A miss which is not under half the larger of the two before it
        // will not reach the goal within the steps allowed. One step can
        // grow a little on the way to a solution, so two are compared.
        let shrinking = size < 0.5 * before[0].max(before[1]);
        if !shrinking {
            break;
        }
        before = [before[1], size];

        // The miss moved in a straight line to the time light left each
        // range, as a change of that range.
        let step = [0, 1].map(|i| (miss - rate_miss * ahead[i]) / aligned[i]);
        if early && past_escape(a, b, &state, ranges, step, between) {
            return Err(Error::Convergence(
                "Orbit from hypothesis: the orbit is not bound to the Sun".into(),
            ));
        }
        for i in 0..2 {
            ranges[i] -= step[i];
        }
    }
    Err(Error::Convergence(
        "Orbit from hypothesis: the misses do not shrink, the reference time is too far from \
         the sights for this orbit"
            .into(),
    ))
}

/// Whether the orbit `state` of a step of [`converge`] is so far past the
/// speed [`solve`] allows that the steps still to come cannot bring it back,
/// so that the hypothesis can be refused without them.
///
/// The next step changes the ranges by `step`. That moves the two positions
/// along their sights by those amounts, and so the velocity of the orbit
/// through them by at most their sum over the time `between` them, the
/// escape speed by at most its share of the larger over the distance from
/// the Sun, and the allowed noise by its share of the larger over the
/// smaller range. Each step shrinks the miss, and with it the step after
/// it, so all the steps to come change the speed by a few times the next
/// one at most. The orbit is refused when it is past the allowed speed by
/// more than [`REMAINING_MARGIN`] times what the next step can change.
fn past_escape(
    a: &Sight,
    b: &Sight,
    state: &State<Equatorial, SunCenter>,
    ranges: [f64; 2],
    step: [f64; 2],
    between: f64,
) -> bool {
    let noise = 3.0 * velocity_noise(a, b, ranges, between);
    let distance = state.pos.norm();
    let escape = (2.0 * GMS / distance).sqrt();
    let largest = step[0].abs().max(step[1].abs());
    let change = (step[0].abs() + step[1].abs()) / between
        + escape * largest / distance
        + noise * largest / ranges[0].abs().min(ranges[1].abs());
    state.vel.norm() - escape - noise > REMAINING_MARGIN * change
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::synthetic::{Rng, circular, sights};
    use crate::{DEPTH_STEP, RATE_STEP};

    #[test]
    fn early_refusals_keep_every_bound_orbit() {
        // Random objects, from well bound to just past escape, seen from one
        // observer or from two a hundredth of an AU apart, in pairs from
        // minutes to days long at both ends of a ten day window, with and
        // without an arcsecond of noise, solved under the true hypothesis
        // and one grid step off it in depth or rate, along every axis that
        // holds the pair. Every orbit the solve finds bound without its
        // early refusals it also finds with them.
        let epoch = 2460000.5;
        let earth = circular(epoch.into(), 1.0, 0.0, 0.0);
        let mut far = earth.clone();
        far.pos += Vector::new([0.0, 0.0, 0.01]);
        let mut rng = Rng(29);
        let mut uniform = || rng.uniform();
        let arcsec = (1.0_f64 / 3600.0).to_radians();
        let (mut found, mut refused_early) = (0, 0);
        for trial in 0..400 {
            let radius = 0.3 * 130.0_f64.powf(uniform());
            let unit = |uniform: &mut dyn FnMut() -> f64| {
                Vector::<Equatorial>::new([0; 3].map(|_| 2.0 * uniform() - 1.0)).normalize()
            };
            let share = if trial % 3 == 0 {
                0.97 + 0.05 * uniform()
            } else {
                0.3 + 0.7 * uniform()
            };
            let speed = share * (2.0 * GMS / radius).sqrt();
            let pos = unit(&mut uniform) * radius;
            let vel = unit(&mut uniform) * speed;
            let truth = State {
                pos,
                vel,
                ..earth.clone()
            };
            let second = if trial % 2 == 0 { &earth } else { &far };
            for gap in [0.003, 0.05, 1.0, 3.0] {
                for start in [-5.0, 5.0 - gap] {
                    for noise in [0.0, arcsec] {
                        let mut shift = || noise * (2.0 * uniform() - 1.0);
                        let a = sights(&truth, &earth, epoch, &[start], arcsec, &mut shift)[0];
                        let b = sights(&truth, second, epoch, &[start + gap], arcsec, shift)[0];
                        for direction in axes() {
                            let aligned =
                                direction.dot(&a.direction).min(direction.dot(&b.direction));
                            if aligned < MIN_ALIGNMENT {
                                continue;
                            }
                            let depth = direction.dot(&(truth.pos - earth.pos));
                            let depth_rate = direction.dot(&(truth.vel - earth.vel));
                            for (depth_step, rate_step) in
                                [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
                            {
                                let hypothesis = Hypothesis {
                                    origin: earth.clone(),
                                    direction,
                                    depth: depth * (depth_step * DEPTH_STEP).exp(),
                                    depth_rate: depth_rate + rate_step * RATE_STEP * depth.abs(),
                                };
                                let full = converge(&a, &b, &hypothesis, false)
                                    .and_then(|solution| accepted(&a, &b, solution));
                                let early = solve(&a, &b, &hypothesis);
                                if full.is_ok() {
                                    found += 1;
                                    assert!(early.is_ok(), "trial {trial}: {early:?}");
                                } else if converge(&a, &b, &hypothesis, true).is_err()
                                    && converge(&a, &b, &hypothesis, false).is_ok()
                                {
                                    refused_early += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        println!("orbits found {found}, refused early and unbound {refused_early}");
        assert!(
            found > 1000 && refused_early > 100,
            "{found} {refused_early}"
        );
    }

    #[test]
    fn far_planes_are_refused() {
        // A hypothesis whose plane meets the sights farther away than light
        // travels in the days a time can hold, by its depth or by its origin,
        // is refused rather than overflowing a time.
        let epoch = 2460000.5;
        let observer = circular(epoch.into(), 1.0, 0.0, 0.0);
        let direction = Vector::new([1.0, 0.0, 0.0]);
        let sight = |days: f64| Sight {
            time: (epoch + days).into(),
            observer: observer.pos,
            direction,
            sigma: 1e-6,
        };
        let (a, b) = (sight(0.0), sight(1.0));
        let near = Hypothesis {
            origin: observer.clone(),
            direction,
            depth: 1.0,
            depth_rate: 0.0,
        };
        let deep = Hypothesis {
            depth: 1e300,
            ..near.clone()
        };
        let mut far = near.clone();
        far.origin.pos = Vector::new([-1e300, 0.0, 0.0]);
        for hypothesis in [deep, far] {
            let found = solve(&a, &b, &hypothesis).map(|found| found.state);
            assert!(matches!(found, Err(Error::ValueError(_))), "{found:?}");
        }
    }
}
