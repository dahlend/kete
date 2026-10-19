// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The solve of a pair under a hypothesis: convergence at the truth,
//! hypotheses without an answer, false pairs, and bad input.

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, Vector};
use kete_core::kepler::propagate_two_body;
use kete_core::prelude::Error;
use kete_core::time::Time;

use crate::from_ranges::orbit_through;
use crate::hypothesis::{axes, margin_share};
use crate::{DEPTH_STEP, Hypothesis, MIN_ALIGNMENT, RATE_STEP, Sight, form_units};

use super::synthetic::{
    ARCSEC, CASES, Case, EXACT, Field, GAPS, HOUR, Observer, Population, Rng, SunState, WINDOWS,
    add_noise, emitted, nearest_axis, observe, sight, sun, true_hypothesis,
};
use super::{orbit_from_hypothesis, reference};

/// One solve at the true hypothesis.
struct Solve {
    /// How hard the solve is expected to be: the contraction of one step,
    /// from the square of the angle swept about the Sun on the way to the
    /// reference time, times one plus the angle between the two sights over
    /// the fraction of that time which lies between the two emissions.
    difficulty: f64,

    /// Position error of the orbit found, as a fraction of the distance from
    /// the observer to the object. `None` if the solve failed.
    error: Option<f64>,

    /// That error over the error which rounding and the tolerance of the
    /// solve allow a pair from one observer.
    excess: f64,
}

/// Solve the pairs of one object at both ends of a window under every axis
/// which can hold them, at the true hypothesis about the first observer.
/// Also the number of pairs no axis can hold.
///
/// The first sight of each pair is from `first` and the second from `second`.
fn solve_window(
    first: &Observer,
    second: &Observer,
    case: &Case,
    window: f64,
) -> (Vec<Solve>, usize) {
    let epoch = reference();
    let origin = first.state(epoch);
    let truth = case.object(&origin);
    let mut solves = Vec::new();
    let mut uncovered = 0;
    for gap in GAPS.into_iter().filter(|gap| *gap <= 2.0 * window) {
        for start in [-window, window - gap] {
            let (a, range_a) = sight(first, &truth, epoch + start);
            let (b, range_b) = sight(second, &truth, epoch + (start + gap));
            let then = propagate_two_body(&truth, a.emission_time(range_a)).unwrap();
            let ahead = (epoch - then.epoch).elapsed;
            let lever = (ahead / (b.emission_time(range_b) - then.epoch).elapsed).abs();
            let apart = a.direction.angle(&b.direction);
            let swept_sq = GMS / then.pos.norm().powi(3) * ahead * ahead;
            // What the tolerance of the solve and the rounding of the
            // positions allow, for sights from one observer. The rounding
            // reaches the reference time through the velocity.
            let allowed = 1e-6 * case.dist + 1e3 * f64::EPSILON * then.pos.norm() * (1.0 + lever);

            let before = solves.len();
            for axis in axes() {
                if axis.dot(&a.direction).min(axis.dot(&b.direction)) < MIN_ALIGNMENT {
                    continue;
                }
                let hypothesis = true_hypothesis(&truth, &origin, axis);
                let error = orbit_from_hypothesis(&a, &b, &hypothesis)
                    .ok()
                    .map(|state| (state.pos - truth.pos).norm());
                solves.push(Solve {
                    difficulty: 0.5 * swept_sq * (1.0 + apart * lever),
                    error: error.map(|e| e / case.dist),
                    excess: error.unwrap_or(0.0) / allowed,
                });
            }
            uncovered += usize::from(solves.len() == before);
        }
    }
    (solves, uncovered)
}

/// Solve every object over every window for pairs of observers, print the
/// outcomes, and check what holds for any pair of observers.
///
/// Returns, over the bound objects, the difficulty of the hardest solve
/// found and of the easiest solve lost, and the largest [`Solve::error`] and
/// [`Solve::excess`] of the solves found.
fn convergence_table(title: &str, observers: &[(Observer, Observer)]) -> [f64; 4] {
    println!("\n{title}");
    print!("{:>20}", "window half-length");
    for window in WINDOWS {
        print!("{:>14}", format!("{window} d"));
    }
    println!();
    let (mut hardest_found, mut easiest_lost) = (0.0_f64, f64::INFINITY);
    let (mut largest_error, mut largest_excess) = (0.0_f64, 0.0_f64);
    for case in &CASES {
        print!("{:>20}", case.name);
        let bound = case.population != Population::Hyperbolic;
        for window in WINDOWS {
            let (mut found, mut lost, mut uncovered_all) = (0, 0, 0);
            for (first, second) in observers {
                let (solves, uncovered) = solve_window(first, second, case, window);
                uncovered_all += uncovered;
                // An object does not leave every axis within the time it
                // takes to move its own distance from the observer.
                let origin = first.state(reference());
                let encounter = case.dist / (case.object(&origin).vel - origin.vel).norm();
                if first.name == second.name && bound && window <= encounter {
                    assert_eq!(uncovered, 0, "{} from {}", case.name, first.name);
                }
                for solve in &solves {
                    let Some(error) = solve.error else {
                        lost += 1;
                        if bound {
                            easiest_lost = easiest_lost.min(solve.difficulty);
                        }
                        continue;
                    };
                    found += 1;
                    // One pair has one orbit under a hypothesis, so a solve
                    // never returns an orbit other than the true one, which
                    // would be off by about the distance to the object.
                    assert!(error < 0.01, "{} at {window} days", case.name);
                    // Unbound orbits are out of scope and are all refused.
                    assert!(bound, "{} at {window} days", case.name);
                    hardest_found = hardest_found.max(solve.difficulty);
                    largest_error = largest_error.max(error);
                    largest_excess = largest_excess.max(solve.excess);
                }
            }
            print!("{:>14}", format!("{found}/{lost} -{uncovered_all}"));
        }
        println!();
    }
    println!(
        "Difficulty of the hardest solve found {hardest_found:.2}, of the easiest lost \
         {easiest_lost:.2}. Largest error {largest_error:.1e} of the distance to the object."
    );
    [hardest_found, easiest_lost, largest_error, largest_excess]
}

#[test]
fn hypothesis_convergence() {
    let epoch = reference();
    println!("\nPairs solved under the true hypothesis about the first observer, counted over");
    println!("gaps from one second to ten days, both ends of the window, and every axis");
    println!("which can hold the pair: found / lost -pairs without an axis.");

    let alone: Vec<_> = Observer::all(epoch)
        .into_iter()
        .zip(Observer::all(epoch))
        .collect();
    let [found, lost, _, excess] =
        convergence_table("Both sights from one observer, all seven kinds.", &alone);
    // Every solve easier than this is found. The solve stops working where
    // the reference time is too far for the orbit, not before.
    assert!(lost > 0.4, "lost a solve of difficulty {lost}");
    assert!(lost < 1.0 && found < 8.0, "found {found}, lost {lost}");
    // The orbits found are as exact as the tolerance and rounding allow.
    assert!(excess < 1.0, "error {excess} times what is allowed");

    // The two sights from different observers. The time between the two
    // emissions is then what the velocity rests on, and the solve is lost
    // sooner as that time shrinks against the time to the reference.
    let near = [
        (Observer::ground(epoch), Observer::sunward(epoch)),
        (Observer::low_orbit(epoch), Observer::ground(epoch)),
    ];
    let apart = [
        (Observer::ground(epoch), Observer::outer(epoch)),
        (Observer::polar(epoch), Observer::low_orbit(epoch)),
    ];
    for (title, observers) in [
        ("Sights from two observers within 0.01 AU.", near),
        ("Sights from two observers over 1 AU apart.", apart),
    ] {
        let [found, lost, error, _] = convergence_table(title, &observers);
        assert!(lost > 0.2, "lost a solve of difficulty {lost}");
        assert!(lost < 1.0 && found < 8.0, "found {found}, lost {lost}");
        assert!(error < 1e-3, "error {error} of the distance");
    }
}

/// A pair of sights of a main belt object an hour apart, and the hypothesis
/// which is true of it about the observer.
fn main_belt_pair() -> (Sight, Sight, Hypothesis) {
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("main belt e180").object(&origin);
    let a = sight(&planet, &truth, epoch + 2.0).0;
    let b = sight(&planet, &truth, epoch + (2.0 + HOUR)).0;
    let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&a.direction));
    (a, b, hypothesis)
}

#[test]
fn orbit_from_hypothesis_rejects() {
    let (a, b, good) = main_belt_pair();
    assert!(orbit_from_hypothesis(&a, &b, &good).is_ok());

    // A direction the sights are not aligned with. Hypotheses which are not
    // valid whatever the sights are refused by `link`, see
    // `link_rejects_bad_input`.
    let invalid = |change: fn(&mut Hypothesis)| {
        let mut hypothesis = good.clone();
        change(&mut hypothesis);
        let result = orbit_from_hypothesis(&a, &b, &hypothesis);
        matches!(result, Err(Error::ValueError(_)))
    };
    assert!(
        invalid(|h| h.direction = -h.direction),
        "direction behind the sights"
    );
    assert!(
        invalid(|h| {
            let side = h.direction.cross(&Vector::new([0.6, 0.0, 0.8])).normalize();
            h.direction = (h.direction * (MIN_ALIGNMENT - 0.1) + side).normalize();
        }),
        "direction to the side of the sights"
    );

    // Each of these is a valid hypothesis which no orbit satisfies.
    let no_orbit = |change: fn(&mut Hypothesis)| {
        let mut hypothesis = good.clone();
        change(&mut hypothesis);
        let result = orbit_from_hypothesis(&a, &b, &hypothesis);
        matches!(result, Err(Error::Convergence(_)))
    };
    assert!(
        no_orbit(|h| h.origin.pos -= h.direction * (h.depth + 1.0)),
        "plane behind the observer"
    );
    assert!(no_orbit(|h| h.depth_rate = 0.5), "unbound outward");
    assert!(no_orbit(|h| h.depth_rate = -0.5), "unbound inward");
    assert!(no_orbit(|h| h.origin.epoch += 900.0), "far reference time");
}

#[test]
fn plane_through_the_observer_returns() {
    // The plane through the observer at the time of the first sight, so
    // that the first range starts at zero, with sights pointed well apart.
    // The solve returns, without an orbit.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let truth = Case::named("main belt e90").object(&planet.state(epoch));
    let a = sight(&planet, &truth, epoch).0;
    let mut b = sight(&planet, &truth, epoch + HOUR).0;
    let axis = nearest_axis(&a.direction);
    // Turn the second sight well away from the first, still within reach of
    // the axis.
    b.direction = (b.direction + axis.cross(&b.direction) * 0.5).normalize();
    for depth_rate in [0.0, 1e-3, -1e-3, 0.02] {
        let hypothesis = Hypothesis {
            origin: sun(epoch),
            direction: axis,
            depth: axis.dot(&a.observer).abs(),
            depth_rate,
        };
        assert!(orbit_from_hypothesis(&a, &b, &hypothesis).is_err());
        assert!(orbit_from_hypothesis(&b, &a, &hypothesis).is_err());
    }
}

#[test]
fn close_object_has_one_orbit() {
    // An object 0.001 AU from the observer at the reference time, seen a day
    // earlier. The pair has one orbit under every axis which holds it, and
    // the solve finds the true one.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let home = planet.state(epoch);
    let truth = Case::named("close 0.001 e120").object(&home);
    let a = sight(&planet, &truth, epoch - 1.0).0;
    let b = sight(&planet, &truth, epoch - (1.0 - HOUR)).0;
    // The object passes the observer between the pair and the reference
    // time, so along some axes its depth is negative, which is allowed.
    for origin in [home, sun(epoch)] {
        let mut solved = 0;
        for axis in axes() {
            let hypothesis = true_hypothesis(&truth, &origin, axis);
            let aligned = axis.dot(&a.direction).min(axis.dot(&b.direction));
            if aligned < MIN_ALIGNMENT {
                continue;
            }
            let state = orbit_from_hypothesis(&a, &b, &hypothesis).unwrap();
            let error = (state.pos - truth.pos).norm();
            assert!(error < 1e-9, "error {error:e} AU");
            solved += 1;
        }
        assert_eq!(solved, 2, "axes which hold the pair");
    }
}

#[test]
fn noise_does_not_make_an_object_unbound() {
    // A distant object seen in pairs an hour long with 0.1 arcsec of noise.
    // The noise moves the speed of a pair by a sixth of the escape speed, and
    // the object moves at three quarters of it. With the noise recorded in
    // the sights the orbits are kept; told the sights are exact, the solve
    // refuses some of them as unbound.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("tno e180").object(&origin);
    let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&(truth.pos - origin.pos)));
    let mut rng = Rng(71);
    let (mut kept, mut refused_exact) = (0, 0);
    let trials = 200;
    for _ in 0..trials {
        let mut a = sight(&planet, &truth, epoch).0;
        let mut b = sight(&planet, &truth, epoch + HOUR).0;
        add_noise(&mut a, 0.1 * ARCSEC, &mut rng);
        add_noise(&mut b, 0.1 * ARCSEC, &mut rng);
        kept += usize::from(orbit_from_hypothesis(&a, &b, &hypothesis).is_ok());
        a.sigma = EXACT;
        b.sigma = EXACT;
        let exact = orbit_from_hypothesis(&a, &b, &hypothesis);
        refused_exact += usize::from(matches!(exact, Err(Error::Convergence(_))));
    }
    assert!(kept >= trials - 2, "{kept} of {trials} noisy pairs kept");
    assert!(
        refused_exact >= trials / 20,
        "the margin was not needed: {refused_exact} refused"
    );
}

/// The hypotheses about `origin` near the truth of `object` along `axis`:
/// the true one, and one grid step off in depth or in rate.
fn near_truth(object: &SunState, origin: &SunState, axis: Vector<Equatorial>) -> [Hypothesis; 5] {
    let truth = true_hypothesis(object, origin, axis);
    let step = |depth: f64, rate: f64| Hypothesis {
        depth: truth.depth * (depth * DEPTH_STEP).exp(),
        depth_rate: truth.depth_rate + rate * RATE_STEP * truth.depth.abs(),
        ..truth.clone()
    };
    [
        step(0.0, 0.0),
        step(1.0, 0.0),
        step(-1.0, 0.0),
        step(0.0, 1.0),
        step(0.0, -1.0),
    ]
}

/// The largest share of the margin of the unbound refusal, see
/// [`margin_share`], of the orbits the solve finds for `object`,
/// seen in pairs from `first` and `second` at both ends of every window,
/// with every gap, with and without an arcsecond of noise, under the
/// hypotheses [`near_truth`] about `first` along every axis which holds the
/// pair. Also the number of orbits found.
fn largest_share(
    first: &Observer,
    second: &Observer,
    object: &SunState,
    rng: &mut Rng,
) -> (f64, usize) {
    let epoch = reference();
    let origin = first.state(epoch);
    let (mut largest, mut found) = (f64::NEG_INFINITY, 0);
    for window in WINDOWS {
        for gap in GAPS.into_iter().filter(|gap| *gap <= 2.0 * window) {
            for start in [-window, window - gap] {
                for noise in [0.0, ARCSEC] {
                    let mut a = sight(first, object, epoch + start).0;
                    let mut b = sight(second, object, epoch + (start + gap)).0;
                    if noise > 0.0 {
                        add_noise(&mut a, noise, rng);
                        add_noise(&mut b, noise, rng);
                    }
                    for axis in axes() {
                        if axis.dot(&a.direction).min(axis.dot(&b.direction)) < MIN_ALIGNMENT {
                            continue;
                        }
                        for hypothesis in near_truth(object, &origin, axis) {
                            if orbit_from_hypothesis(&a, &b, &hypothesis).is_ok() {
                                largest = largest.max(margin_share(&a, &b, &hypothesis).unwrap());
                                found += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    (largest, found)
}

#[test]
fn unbound_prefilter_margin() {
    // Before its first step, the solve refuses a hypothesis whose straight
    // line through the starting ranges is faster than escape by more than a
    // margin of the pull of the Sun. Every orbit it finds used some share of
    // that margin. If the largest share over every population, observer and
    // geometry is well under one, the margin is not what decides whether a
    // bound orbit is found.
    let epoch = reference();
    // Pairs from two observers use the most of the margin.
    let observers = [
        (Observer::planet(epoch), Observer::planet(epoch)),
        (Observer::inner(epoch), Observer::inner(epoch)),
        (Observer::polar(epoch), Observer::polar(epoch)),
        (Observer::ground(epoch), Observer::sunward(epoch)),
        (Observer::low_orbit(epoch), Observer::ground(epoch)),
        (Observer::ground(epoch), Observer::outer(epoch)),
        (Observer::polar(epoch), Observer::low_orbit(epoch)),
    ];
    let mut rng = Rng(83);
    println!("\nLargest share of the margin of the unbound refusal used by an orbit found.");
    println!("{:<44}{:>10}{:>12}", "objects", "orbits", "share");
    let mut worst = f64::NEG_INFINITY;
    let mut row = |name: String, cases: Vec<(&Observer, &Observer, SunState)>, rng: &mut Rng| {
        let (mut largest, mut found) = (f64::NEG_INFINITY, 0);
        for (first, second, object) in cases {
            let (share, count) = largest_share(first, second, &object, rng);
            largest = largest.max(share);
            found += count;
        }
        println!("{name:<44}{found:>10}{largest:>12.3}");
        assert!(found > 0, "{name}: no orbits found");
        worst = worst.max(largest);
    };

    // The described objects, from single observers and pairs of observers.
    for population in Population::BOUND {
        let cases = CASES
            .iter()
            .filter(|case| case.population == population)
            .flat_map(|case| {
                observers
                    .iter()
                    .map(move |(first, second)| (first, second, case.object(&first.state(epoch))))
            })
            .collect();
        row(format!("{population:?}, described"), cases, &mut rng);
    }

    // Random members of every bound population at opposition and at
    // quadrature, and random interior and near-Earth objects close to the
    // Sun on the sky, from the ground site alone and with the sunward point.
    let ground = Observer::ground(epoch);
    let sunward = Observer::sunward(epoch);
    let home = ground.state(epoch);
    for (population, elongs) in [
        (Population::MainBelt, &[90.0, 180.0][..]),
        (Population::Close, &[90.0, 180.0]),
        (Population::NearEarth, &[90.0, 180.0]),
        (Population::Trojan, &[90.0, 180.0]),
        (Population::Tno, &[90.0, 180.0]),
        (Population::Tilted, &[90.0, 180.0]),
        (Population::NearEarth, &[20.0, 25.0, 30.0]),
        (Population::Interior, &[20.0, 30.0]),
    ] {
        let mut cases = Vec::new();
        for &elong in elongs {
            let field = Field::new(&home.pos, elong, 0.0, 5.0);
            for _ in 0..3 {
                let distances = (population == Population::Close).then_some((0.003, 0.1));
                let object = population.in_field(&mut rng, &home, &field, distances);
                cases.push((&ground, &ground, object.clone()));
                cases.push((&ground, &sunward, object));
            }
        }
        row(
            format!("{population:?}, random, elong {elongs:?}"),
            cases,
            &mut rng,
        );
    }
    assert!(worst < 0.5, "an orbit used {worst} of the margin");
}

#[test]
fn hypothesis_fuzz() {
    // Random objects of every kind seen by two observers over two weeks, and
    // every pair of sights up to three days and a fifth of a radian a day
    // apart, under random hypotheses. Most of the pairs are of two different
    // objects. Whatever is returned is an orbit of the pair under the
    // hypothesis: bound, on the plane, and seen along both sights.
    let epoch = reference();
    let mut rng = Rng(3);
    let objects: Vec<_> = Population::ALL
        .into_iter()
        .flat_map(|population| [population; 8])
        .map(|population| population.draw(&mut rng, epoch))
        .collect();
    let observers = [Observer::planet(epoch), Observer::ground(epoch)];
    let times: Vec<_> = (0..8)
        .map(|k| epoch + (f64::from(k) * 1.93 - 7.0))
        .collect();
    let (sights, labels) = observe(&observers, &objects, &times);
    let pairs = form_units(&sights, 3.0, 0.2, 0.0, 2, true, 0.0).unwrap();
    let origin = observers[0].state(epoch);

    // Orbits, failures to converge, and other errors, for pairs of two
    // objects and of one.
    let mut counts = [[0_usize; 3]; 2];
    for pair in &pairs {
        let (i, j) = (pair[0], pair[1]);
        let (a, b) = (&sights[i], &sights[j]);
        let real = usize::from(labels[i] == labels[j]);
        for _ in 0..3 {
            let hypothesis = Hypothesis {
                origin: origin.clone(),
                direction: nearest_axis(&a.direction),
                depth: 10.0_f64.powf(rng.between(-3.0, 1.7)),
                depth_rate: rng.between(-0.03, 0.03),
            };
            let result = orbit_from_hypothesis(a, b, &hypothesis);
            let kind = match &result {
                Ok(_) => 0,
                Err(Error::Convergence(_)) => 1,
                Err(_) => 2,
            };
            counts[real][kind] += 1;
            let Ok(state) = result else { continue };
            assert!(
                state.vel.norm_squared() * state.pos.norm() < 2.0 * GMS,
                "unbound"
            );
            let miss = hypothesis.direction.dot(&(state.pos - origin.pos)) - hypothesis.depth;
            let rate_miss =
                hypothesis.direction.dot(&(state.vel - origin.vel)) - hypothesis.depth_rate;
            assert!(
                miss.abs() < 1e-7 && rate_miss.abs() < 1e-7,
                "miss {miss:e} {rate_miss:e}"
            );
            for seen in [a, b] {
                let offset = emitted(&state, &seen.observer, seen.time).pos - seen.observer;
                let angle = offset.angle(&seen.direction);
                assert!(angle < 1e-6, "sight missed by {angle:e} rad");
            }
        }
    }
    for kinds in counts {
        assert!(kinds[0] > 100 && kinds[1] > 100, "too few: {counts:?}");
    }
}

#[test]
fn bad_input_returns() {
    // Every function must return, without a panic, whatever numbers it is
    // given, and an answer is finite. Values are drawn from the kinds of
    // number which break things.
    let mut rng = Rng(4);
    let value = |rng: &mut Rng| {
        let kinds = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1e-300,
            -1e-300,
            1e300,
            -1e300,
            f64::EPSILON,
            1.0,
            -1.0,
        ];
        let _ = rng.uniform();
        let kind = (rng.0 % 24) as usize;
        kinds.get(kind).copied().unwrap_or_else(|| rng.normal())
    };
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let truth = CASES[0].object(&planet.state(epoch));
    let good = sight(&planet, &truth, epoch).0;
    for _ in 0..20_000 {
        let mut draw = || value(&mut rng);
        let mut sights = [good; 3];
        for sight in &mut sights {
            if draw() > 0.0 {
                sight.time = Time::new(2460000.5 + draw());
            }
            if draw() > 0.0 {
                sight.observer = Vector::new([draw(), draw(), draw()]);
            }
            if draw() > 0.0 {
                sight.direction = Vector::new([draw(), draw(), draw()]);
            }
        }
        let mut hypothesis =
            true_hypothesis(&truth, &planet.state(epoch), nearest_axis(&good.direction));
        if draw() > 0.0 {
            hypothesis.depth = draw();
        }
        if draw() > 0.0 {
            hypothesis.depth_rate = draw();
        }
        if draw() > 0.0 {
            hypothesis.direction = Vector::new([draw(), draw(), draw()]);
        }
        if draw() > 0.0 {
            hypothesis.origin.pos = Vector::new([draw(), draw(), draw()]);
        }
        if draw() > 0.0 {
            hypothesis.origin.epoch = Time::new(2460000.5 + draw());
        }

        let [a, b, _] = &sights;
        for state in [
            orbit_from_hypothesis(a, b, &hypothesis),
            orbit_through(a, b, draw(), draw()),
        ]
        .into_iter()
        .flatten()
        {
            assert!(state.is_finite(), "an answer which is not finite");
        }
        let _ = form_units(&sights, draw(), draw(), draw(), 2, true, draw());
    }
}
