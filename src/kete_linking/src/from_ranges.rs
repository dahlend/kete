// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The orbit through two lines of sight, given a range along each.

use kete_core::constants::{C_AU_PER_DAY, C_AU_PER_DAY_INV, GMS};
use kete_core::desigs::Desig;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::kepler::analytic_2_body_delta;
use kete_core::prelude::{Error, KeteResult, State};

use crate::Sight;
use crate::sight::MAX_DAYS;

/// Most passes of the velocity iteration.
const PASSES: usize = 30;

/// Largest miss of an accepted velocity, as a fraction of the distance
/// between the two positions.
const TOLERANCE: f64 = 1e-12;

/// The two-body orbit about the Sun of an object at given ranges along two
/// lines of sight.
///
/// Each range gives a position and, after removing the light travel time, the
/// time the object was there. The orbit through both is returned as the state
/// at the emission time of `a`.
///
/// The sights may be in either time order, any time apart, and from different
/// observers. Everything here is smooth through zero range, and the steps of
/// the hypothesis solve can pass through points behind an observer, so a
/// range may have either sign. The orbit goes the short way around the Sun,
/// and the arc it sweeps has to be short, see the tests
/// `orbit_through_long_arcs` and `orbit_through_accuracy`. Both sights have
/// to be [`Sight::is_usable`], as [`crate::link()`] checks of its inputs.
///
/// # Arguments
///
/// * `a` - Line of sight the state is returned at.
/// * `b` - Second line of sight.
/// * `range_a` - Distance from the observer of `a` to the object, in AU.
/// * `range_b` - Distance from the observer of `b` to the object, in AU.
///
/// # Errors
///
/// * [`Error::ValueError`] if a range is not finite or too large, or if the
///   two positions are the same point or farther apart than light travels
///   between their emission times, which includes equal emission times.
/// * [`Error::Convergence`] if the arc about the Sun is too long, or the
///   two-body propagation fails.
pub(crate) fn orbit_through(
    a: &Sight,
    b: &Sight,
    range_a: f64,
    range_b: f64,
) -> KeteResult<State<Equatorial, SunCenter>> {
    let near = range_a.abs().max(range_b.abs()) * C_AU_PER_DAY_INV < MAX_DAYS;
    if !near {
        return Err(Error::ValueError(
            "Orbit from ranges: a range is too large".into(),
        ));
    }
    let epoch = a.emission_time(range_a);
    let pos = a.position(range_a);
    let dt = (b.emission_time(range_b) - epoch).elapsed;
    let vel = velocity(&pos, &b.position(range_b), dt)?;
    Ok(State::new(Desig::Empty, epoch, pos, vel, SunCenter))
}

/// Velocity at `start` of the two-body orbit about the Sun which is at `end`
/// after `dt` days. `dt` may be negative.
///
/// The velocity starts as that of straight-line motion, less half the pull
/// of the Sun at `start` times `dt`, the motion to second order in the time.
/// Each pass moves the start along the exact two-body orbit and adds the
/// miss at the end, divided by `dt`, to the velocity, which shrinks the miss
/// by about the square of the angle swept about the Sun. The arc is limited
/// by the need to shrink it from the chord to [`TOLERANCE`] within
/// [`PASSES`] passes.
///
/// This needs no direction of motion to pick the way around the Sun, as a
/// Lambert solver does: the straight line it starts from already goes the
/// short way, which is the way of every short arc.
fn velocity(
    start: &Vector<Equatorial>,
    end: &Vector<Equatorial>,
    dt: f64,
) -> KeteResult<Vector<Equatorial>> {
    let chord = end - start;
    let length = chord.norm();
    // Nothing moves faster than light, and nearly equal emission times give
    // a velocity made of rounding error.
    if !(length > 0.0 && length < C_AU_PER_DAY * dt.abs()) {
        return Err(Error::ValueError(
            "Orbit from ranges: the two positions are the same point, or farther apart than \
             light travels between their emission times"
                .into(),
        ));
    }
    let pull = -*start * (GMS / start.norm().powi(3));
    let mut vel = chord / dt - pull * (0.5 * dt);
    let start = (*start).into();
    for _ in 0..PASSES {
        let (moved, _) = analytic_2_body_delta(dt, &start, &vel.into(), GMS)?;
        let miss = chord - Vector::from(moved);
        // Taken also when the miss meets the tolerance, as it is free.
        vel += miss / dt;
        if miss.norm() <= TOLERANCE * length {
            return Ok(vel);
        }
    }
    Err(Error::Convergence(format!(
        "Orbit from ranges: no velocity found in {PASSES} passes, the arc about the Sun between \
         the two positions is too long"
    )))
}
