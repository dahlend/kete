// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! What an observer sees of a two-body orbit about the Sun: the direction,
//! with light time, the residual of a line of sight, and their derivatives
//! over the state of the orbit, through its state transition matrix from
//! [`kete_core::kepler::propagate_two_body_with_stm`].

use kete_core::constants::{C_AU_PER_DAY, GMS};
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::kepler::{propagate_two_body, propagate_two_body_with_stm};
use kete_core::prelude::State;
use kete_core::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3x6, Vector3};

use crate::Sight;

/// Predicted less observed direction of one sight, over its sigma, and the
/// state of the object when it sent the light the sight received.
///
/// The state is propagated to the time of the sight, and then moved back by
/// the light travel time, see [`emitted`]. So this is the light-time
/// solution with one propagation, see the test `light_time_to_rounding`.
/// `None` for a state at or above the speed of light, which has no light
/// sent before the sight to solve for, or if a residual cannot be evaluated.
pub(crate) fn residual_at(
    state: &State<Equatorial, SunCenter>,
    sight: &Sight,
) -> Option<(Vector3<f64>, State<Equatorial, SunCenter>)> {
    let sent = sent_at(state, sight.time, &sight.observer)?;
    Some((miss(&sent, sight)?, sent))
}

/// The direction in which an observer at `observer` sees the two-body orbit
/// of `state` at `time`, with light time, as [`residual_at`] finds it.
/// `None` if it cannot be evaluated.
pub(crate) fn direction_at(
    state: &State<Equatorial, SunCenter>,
    time: Time<TDB>,
    observer: Vector<Equatorial>,
) -> Option<Vector<Equatorial>> {
    let sent = sent_at(state, time, &observer)?;
    let direction = (sent.pos - observer).normalize();
    direction.is_finite().then_some(direction)
}

/// The direction in which an observer at `observer` sees the two-body orbit
/// of `state` at `time`, with light time, its distance from the observer,
/// and the derivative of the direction over the position and velocity of
/// `state`, see [`seen`]. `None` if it cannot be evaluated.
pub(crate) fn predicted(
    state: &State<Equatorial, SunCenter>,
    time: Time<TDB>,
    observer: Vector<Equatorial>,
) -> Option<(Vector<Equatorial>, f64, Matrix3x6<f64>)> {
    let (sent, jacobian) = sent_and_jacobian(state, time, &observer, 1.0)?;
    let offset = sent.pos - observer;
    let direction = offset.normalize();
    direction
        .is_finite()
        .then_some((direction, offset.norm(), jacobian))
}

/// The state of the object when it sent the light an observer at `observer`
/// received at `time`, by [`emitted`] from the two-body orbit of `state`.
fn sent_at(
    state: &State<Equatorial, SunCenter>,
    time: Time<TDB>,
    observer: &Vector<Equatorial>,
) -> Option<State<Equatorial, SunCenter>> {
    let at = propagate_two_body(state, time).ok()?;
    Some(emitted(&at, observer)?.0)
}

/// The state of an object when it sent the light an observer at `observer`
/// received, from its state `at` the time of reception, and the light travel
/// time.
///
/// The light travel time is solved for exactly along the straight line
/// through the state at the time of reception, and the state is then moved
/// back by it on the Taylor series of its two-body motion to third order in
/// the time, with the pull of the Sun and its rate of change. The pull bends
/// the line over the travel time by far less than light travels in the
/// rounding of a time. The first term the series leaves out is about the
/// fourth power of the angle the object sweeps about the Sun during the
/// travel time, times its distance from the Sun, which over a travel time of
/// minutes to hours is far below any astrometric error. `None` for a state at
/// or above the speed of light.
fn emitted(
    at: &State<Equatorial, SunCenter>,
    observer: &Vector<Equatorial>,
) -> Option<(State<Equatorial, SunCenter>, f64)> {
    // The travel time `tau` solves `|d - v tau| = c tau` for the offset `d`
    // from the observer and the velocity `v`, a quadratic in `tau`.
    let (d, v) = (at.pos - observer, at.vel);
    let a = C_AU_PER_DAY * C_AU_PER_DAY - v.norm_squared();
    let b = d.dot(&v);
    let tau = (b.mul_add(b, a * d.norm_squared()).sqrt() - b) / a;
    if !(a > 0.0 && tau >= 0.0) {
        return None;
    }
    let (pos, vel) = (at.pos, at.vel);
    let squared = pos.norm_squared();
    let strength = GMS / (squared * squared.sqrt());
    let pull = -pos * strength;
    let change = (pos * (3.0 * pos.dot(&vel) / squared) - vel) * strength;
    let sent = State {
        epoch: at.epoch - tau,
        pos: pos - vel * tau + pull * (0.5 * tau * tau) - change * (tau * tau * tau / 6.0),
        vel: vel - pull * tau + change * (0.5 * tau * tau),
        ..at.clone()
    };
    Some((sent, tau))
}

/// Predicted less observed direction of a sight, over its sigma, for the
/// state `sent` of the object when it sent the light. `None` if not finite.
fn miss(sent: &State<Equatorial, SunCenter>, sight: &Sight) -> Option<Vector3<f64>> {
    let miss: Vector3<f64> =
        (((sent.pos - sight.observer).normalize() - sight.direction) / sight.sigma).into();
    miss.iter().all(|x| x.is_finite()).then_some(miss)
}

/// One sight under an orbit.
pub(crate) struct Seen {
    /// Predicted less observed direction, over the sigma of the sight, as
    /// [`residual_at`] evaluates it.
    pub(crate) residual: Vector3<f64>,

    /// The derivative of the residual over the position and velocity of the
    /// orbit's state.
    pub(crate) jacobian: Matrix3x6<f64>,
}

/// One sight under the orbit of `state`: its residual, as [`residual_at`]
/// evaluates it, from one Kepler solve, and the exact derivative of that
/// residual over the position and velocity of `state`, see
/// [`sent_and_jacobian`]. The test `jacobian_matches_differences` compares
/// it with finite differences.
pub(crate) fn seen(state: &State<Equatorial, SunCenter>, sight: &Sight) -> Option<Seen> {
    let (sent, jacobian) = sent_and_jacobian(state, sight.time, &sight.observer, sight.sigma)?;
    Some(Seen {
        residual: miss(&sent, sight)?,
        jacobian,
    })
}

/// The state of the object when it sent the light an observer at `observer`
/// received at `time`, on the two-body orbit of `state`, and the exact
/// derivative of the direction the observer sees over the position and
/// velocity of `state`, over `sigma`. `None` if they cannot be evaluated.
///
/// The derivative is the chain of three. The state transition matrix of
/// [`propagate_two_body_with_stm`] takes a change of `state` to one of the
/// position `x` and velocity `v` at `time`. [`emitted`] moves the object back
/// by the light travel time `tau`, which solves `|x - v tau - o| = c tau` for the
/// observer `o`, so it changes by `n.(dx - tau dv) / (c + n.v)` with `n` the
/// unit vector of `x - v tau - o`. The point the light left then changes by
/// `(I + A tau^2 / 2) dx - tau (I + A tau^2 / 6) dv - w dtau`, with `A` the
/// gradient of the pull of the Sun at `x` and `w` the velocity when the light
/// left. Left out is the change of the gradient across `dx` times `tau^3 /
/// 6`, smaller than the gradient term kept by the angle the object sweeps
/// about the Sun during the travel time. Last, the direction changes by the
/// move across it over the range.
fn sent_and_jacobian(
    state: &State<Equatorial, SunCenter>,
    time: Time<TDB>,
    observer: &Vector<Equatorial>,
    sigma: f64,
) -> Option<(State<Equatorial, SunCenter>, Matrix3x6<f64>)> {
    let (at, stm) = propagate_two_body_with_stm(state, time).ok()?;
    let (sent, tau) = emitted(&at, observer)?;

    let line: Vector3<f64> = (at.pos - at.vel * tau - observer).normalize().into();
    let speed = C_AU_PER_DAY + line.dot(&Vector3::from(at.vel));
    let distance = at.pos.norm();
    let unit: Vector3<f64> = (at.pos / distance).into();
    let gradient = (unit * unit.transpose() * 3.0 - Matrix3::identity()) * (GMS / distance.powi(3));
    let delay = Vector3::from(sent.vel) * line.transpose() / speed;
    // The derivative of the point the light left over `x` and `v`.
    let mut leaves = Matrix3x6::zeros();
    leaves
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&(Matrix3::identity() + gradient * (0.5 * tau * tau) - delay));
    leaves
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&((Matrix3::identity() + gradient * (tau * tau / 6.0)) * -tau + delay * tau));
    let offset = sent.pos - observer;
    let range = offset.norm();
    let dir: Vector3<f64> = (offset / range).into();
    let across = (Matrix3::identity() - dir * dir.transpose()) * (1.0 / (range * sigma));
    let jacobian = across * leaves * stm;
    jacobian
        .iter()
        .all(|x| x.is_finite())
        .then_some((sent, jacobian))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::kepler::light_time_correct;

    use crate::tests::synthetic::{circular, sights};

    #[test]
    fn jacobian_matches_differences() {
        // Objects from 0.003 to 100 AU from the observer, arcs from hours to
        // 60 days, an eccentric orbit through perihelion 0.1 AU from the
        // Sun, and orbits just bound and just past escape, with the state
        // off the sights. Every column of the Jacobian is that of central
        // differences of the residuals, to the noise of the differences.
        let epoch = 2460000.5;
        let earth = circular(epoch.into(), 1.0, 0.0, 0.0);
        let km_per_s = 86400.0 / 1.495_978_707e8;
        let near = |distance: f64, speed: f64| State {
            pos: earth.pos + Vector::new([0.3, 0.8, 0.2]).normalize() * distance,
            vel: earth.vel + Vector::new([-0.5, 0.2, 0.9]).normalize() * (speed * km_per_s),
            ..earth.clone()
        };
        // At 1.5 AU, moving partly outward at `share` of the escape speed.
        let at_escape = |share: f64| {
            let base = circular(epoch.into(), 1.5, 10.0, 115.0);
            let speed = share * (2.0 * GMS / 1.5_f64).sqrt();
            State {
                vel: (base.vel.normalize() + base.pos.normalize() * 0.3).normalize() * speed,
                ..base
            }
        };
        let perihelion = {
            let base = circular(epoch.into(), 0.1, 10.0, 145.0);
            let speed = (GMS * 1.9 / 0.1_f64).sqrt();
            State {
                vel: base.vel.normalize() * speed,
                ..base
            }
        };
        let cases = [
            (
                "main belt, hours",
                circular(epoch.into(), 2.5, 10.0, 0.5),
                0.05,
            ),
            (
                "main belt, two weeks",
                circular(epoch.into(), 2.5, 10.0, 30.0),
                14.0,
            ),
            (
                "near-Earth, a week",
                circular(epoch.into(), 1.3, 10.0, 60.0),
                7.0,
            ),
            ("0.003 AU, hours", near(0.003, 10.0), 0.1),
            ("0.02 AU, two days", near(0.02, 10.0), 2.0),
            (
                "100 AU, 60 days",
                circular(epoch.into(), 100.0, 10.0, 10.0),
                60.0,
            ),
            ("perihelion 0.1 AU, 10 days", perihelion.clone(), 10.0),
            ("perihelion 0.1 AU, 60 days", perihelion, 60.0),
            ("just bound, 30 days", at_escape(0.999), 30.0),
            ("just past escape, 30 days", at_escape(1.001), 30.0),
            ("past escape, 60 days", at_escape(1.2), 60.0),
        ];
        for (name, object, span) in cases {
            let days: Vec<f64> = (-3..=3).map(|k| f64::from(k) * span / 6.0).collect();
            let sights = sights(&object, &earth, epoch, &days, 1e-6, || 0.0);
            let range = (object.pos - earth.pos).norm();
            let mut state = object.clone();
            state.pos += Vector::new([1.0, -2.0, 3.0]) * (1e-4 * range);
            // The state with its `k`th number of position and velocity moved
            // by `h`, and every sight under it.
            let under = |k: usize, h: f64| -> Vec<Seen> {
                let mut moved = state.clone();
                if k < 3 {
                    moved.pos[k] += h;
                } else {
                    moved.vel[k - 3] += h;
                }
                sights.iter().map(|s| seen(&moved, s).unwrap()).collect()
            };
            let base = under(0, 0.0);
            // Each move shifts the sights by about a millionth of a radian:
            // smaller moves show the rounding of the residuals, larger ones
            // their curvature.
            let position = 1e-6 * range;
            let velocity = position / span.max(1.0 / 24.0);
            for k in 0..6 {
                let h = if k < 3 { position } else { velocity };
                let (up, down) = (under(k, h), under(k, -h));
                let (mut worst, mut size) = (0.0_f64, 0.0_f64);
                for ((a, b), seen) in up.iter().zip(&down).zip(&base) {
                    for i in 0..3 {
                        let differences = (a.residual[i] - b.residual[i]) / (2.0 * h);
                        worst = worst.max((differences - seen.jacobian[(i, k)]).abs());
                        size = size.max(seen.jacobian[(i, k)].abs());
                    }
                }
                println!("{name}, column {k}: {:.1e}", worst / size);
                assert!(worst < 1e-6 * size, "{name}, column {k}: {worst} of {size}");
            }
        }
    }

    #[test]
    fn light_time_to_rounding() {
        // Objects from 0.02 to 40 AU from the observer, and far from the
        // epoch of their state: the state light left is that of the full
        // light-time iteration to within its own convergence.
        let epoch = 2460000.5;
        let observer = circular(epoch.into(), 1.0, 0.0, 0.0);
        for (radius, phase) in [(1.02, 0.0), (1.3, 30.0), (2.5, 170.0), (40.0, 60.0)] {
            let object = circular(epoch.into(), radius, 10.0, phase);
            let time: Time<TDB> = (epoch + 12.0).into();
            let seen_from = propagate_two_body(&observer, time).unwrap().pos;
            let iterated =
                light_time_correct(&propagate_two_body(&object, time).unwrap(), &seen_from)
                    .unwrap();
            let sight = Sight {
                time,
                observer: seen_from,
                direction: (iterated.pos - seen_from).normalize(),
                sigma: 1e-6,
            };
            let (miss, at) = residual_at(&object, &sight).unwrap();
            let range = (iterated.pos - seen_from).norm();
            assert!((at.pos - iterated.pos).norm() < 1e-12 * range, "{radius}");
            assert!(miss.norm() * sight.sigma < 1e-12, "{radius}");
        }
    }

    #[test]
    fn light_speed_states_are_refused() {
        // A state moving at or above the speed of light, toward or away
        // from the observer, has no light-time solution.
        let epoch = 2460000.5;
        let observer = circular(epoch.into(), 1.0, 0.0, 0.0);
        let sight = Sight {
            time: epoch.into(),
            observer: observer.pos,
            direction: Vector::new([1.0, 0.0, 0.0]),
            sigma: 1e-6,
        };
        for speed in [1.0, 1.5] {
            for sign in [1.0, -1.0] {
                let state = State {
                    pos: observer.pos + Vector::new([1.0, 0.0, 0.0]),
                    vel: Vector::new([sign * speed * C_AU_PER_DAY, 0.0, 0.0]),
                    ..observer.clone()
                };
                assert!(residual_at(&state, &sight).is_none(), "{speed} {sign}");
            }
        }
    }
}
