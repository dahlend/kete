// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Accuracy and limits of the orbit through two ranges.

use std::f64::consts::PI;

use kete_core::constants::C_AU_PER_DAY;
use kete_core::kepler::propagate_two_body;
use kete_core::prelude::Error;

use crate::from_ranges::orbit_through;

use super::reference;
use super::synthetic::{CASES, Case, GAPS, Observer, sight};

/// Relative velocity error of the orbit through one pair at its true ranges,
/// and the relative error that rounding of the two positions alone causes.
///
/// The first sight is from `first` and the second from `second`. The pair is
/// used in both orders, so the orbit is checked at its early and late end.
fn velocity_error(first: &Observer, second: &Observer, case: &Case, gap: f64) -> (f64, f64) {
    let epoch = reference();
    let object = case.object(&first.state(epoch));
    let (sight_a, range_a) = sight(first, &object, epoch - 0.5 * gap);
    let (sight_b, range_b) = sight(second, &object, epoch + 0.5 * gap);

    let (mut worst, mut floor) = (0.0_f64, 0.0_f64);
    for (a, b, range_a, range_b) in [
        (&sight_a, &sight_b, range_a, range_b),
        (&sight_b, &sight_a, range_b, range_a),
    ] {
        let dt = (b.emission_time(range_b) - a.emission_time(range_a)).elapsed;
        let truth = propagate_two_body(&object, a.emission_time(range_a)).unwrap();
        let speed = truth.vel.norm();
        let orbit = orbit_through(a, b, range_a, range_b).unwrap();
        worst = worst.max((orbit.vel - truth.vel).norm() / speed);
        floor = floor.max(f64::EPSILON * truth.pos.norm() / (dt.abs() * speed));
    }
    (worst, floor)
}

#[test]
fn orbit_through_accuracy() {
    // The error is bounded by the rounding of the positions on short gaps
    // and by the convergence tolerance on long ones, for every object, every
    // kind of observer, and pairs of observers near and far apart.
    let epoch = reference();
    let mut setups: Vec<_> = Observer::all(epoch)
        .into_iter()
        .zip(Observer::all(epoch))
        .collect();
    setups.push((Observer::ground(epoch), Observer::sunward(epoch)));
    setups.push((Observer::ground(epoch), Observer::outer(epoch)));
    setups.push((Observer::low_orbit(epoch), Observer::polar(epoch)));
    setups.push((Observer::inner(epoch), Observer::planet(epoch)));
    println!("\nWorst relative velocity error of orbit_through at the true ranges, over");
    println!("all objects, as a multiple of the rounding floor.");
    print!("{:>24}", "gap (days)");
    for gap in GAPS {
        print!("{gap:9.1e}");
    }
    println!();
    for (first, second) in &setups {
        print!("{:>24}", format!("{} + {}", first.name, second.name));
        for gap in GAPS {
            let mut worst = 0.0_f64;
            for case in &CASES {
                let (err, floor) = velocity_error(first, second, case, gap);
                assert!(
                    err < 1e-11 + 10.0 * floor,
                    "{} from {} and {} at gap {gap}: error {err:e}, floor {floor:e}",
                    case.name,
                    first.name,
                    second.name
                );
                worst = worst.max(err / floor);
            }
            print!("{worst:9.1e}");
        }
        println!();
    }
}

#[test]
fn orbit_through_long_arcs() {
    // Long gaps, to find how far around the Sun a pair can reach. A pair is
    // either solved to the true orbit or fails. It is never solved to a wrong
    // orbit, unless the object went more than half way around the Sun, where
    // the orbit the short way around is a different one.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let (mut solved, mut failed) = (0.0_f64, f64::INFINITY);
    for case in &CASES {
        let object = case.object(&planet.state(epoch));
        for gap in [20.0, 40.0, 80.0, 160.0, 320.0] {
            let (a, range_a) = sight(&planet, &object, epoch);
            let (b, range_b) = sight(&planet, &object, epoch + gap);
            // Angle swept about the Sun, summed over short steps.
            let at = |k: i32| propagate_two_body(&object, epoch + gap * f64::from(k) / 64.0);
            let swept: f64 = (0..64)
                .map(|k| at(k).unwrap().pos.angle(&at(k + 1).unwrap().pos))
                .sum();
            let truth = propagate_two_body(&object, a.emission_time(range_a)).unwrap();
            match orbit_through(&a, &b, range_a, range_b) {
                Ok(orbit) if swept < PI => {
                    let err = (orbit.vel - truth.vel).norm() / truth.vel.norm();
                    assert!(err < 1e-9, "{} gap {gap}: error {err:e}", case.name);
                    solved = solved.max(swept);
                }
                Ok(_) => {}
                Err(err) => {
                    assert!(matches!(err, Error::Convergence(_)), "{err}");
                    failed = failed.min(swept);
                }
            }
        }
    }
    // The limit is near one radian. It moving either way is a change in
    // what pairs can be linked.
    assert!((0.9..1.3).contains(&solved), "solved up to {solved} rad");
    assert!((0.8..1.2).contains(&failed), "failed from {failed} rad");
}

#[test]
fn orbit_through_rejects() {
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let object = Case::named("main belt e180").object(&planet.state(epoch));
    let (a, range_a) = sight(&planet, &object, epoch);
    let (b, range_b) = sight(&planet, &object, epoch + 0.1);
    assert!(orbit_through(&a, &b, range_a, range_b).is_ok());

    let invalid = |range_a: f64, range_b: f64| {
        let result = orbit_through(&a, &b, range_a, range_b);
        matches!(result, Err(Error::ValueError(_)))
    };
    // Ranges which are not finite.
    assert!(invalid(f64::NAN, range_b), "range not a number");
    assert!(invalid(range_a, f64::INFINITY), "infinite range");

    // The same sight twice is one point, which has no velocity.
    let same = orbit_through(&a, &a, range_a, range_a);
    assert!(matches!(same, Err(Error::ValueError(_))), "same point");

    // Ranges whose light travel times nearly cancel the gap between the
    // sights put two distant positions at nearly the same time.
    let gap = (b.time - a.time).elapsed;
    for offset in [0.0, 1e-13, -1e-13, 1e-9, 1e-3 * gap] {
        let far = range_a + (gap - offset) * C_AU_PER_DAY;
        assert!(invalid(range_a, far), "emission times {offset} days apart");
    }

    // An arc too far around the Sun is a failure to converge.
    let (late, range_late) = sight(&planet, &object, epoch + 700.0);
    let long = orbit_through(&a, &late, range_a, range_late);
    assert!(matches!(long, Err(Error::Convergence(_))), "long arc");
}
