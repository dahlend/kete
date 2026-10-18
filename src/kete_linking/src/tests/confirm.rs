// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Confirming candidates with orbit fits, and searching in passes, on
//! observations from the Earth of the test kernels of objects moving under
//! the n-body model.

use kete_core::constants::GMS;
use kete_core::frames::Vector;
use kete_core::prelude::{Error, State};
use kete_core::propagation::NBody;
use kete_fitting::{AstrometricObservation, fit_orbit};

use crate::confirm::sun_centered;
use crate::{
    Candidate, ConfirmRules, SearchPass, SearchSettings, Sight, confirm, search,
    sight_from_observation,
};
use kete_spice::prelude::SpiceEphemeris;

use super::harness::{
    CLUTTER, MAX_RMS, SPAN, Score, Search, Sky, earth, nightly, on_threads, optical, run,
};
use super::reference;
use super::synthetic::{ARCSEC, Population, Rng, SunState, add_noise, emitted};

/// Near-Earth objects of `search_in_passes`.
const NEAR: usize = 6;

/// Least distance of the near-Earth objects of `search_in_passes` from the
/// Earth, in AU.
const NEAR_DEPTH: f64 = 0.05;

/// Whether `score` has every one of `objects` found whole by a pure
/// candidate of its own, and nothing else.
fn all_whole(score: &Score, objects: usize) -> bool {
    score.whole == objects && score.groups == objects && score.pure == objects
}

/// `object`, a state at the reference time, moved under the n-body model to
/// `days` from it, about the Sun.
fn moved(object: &SunState, days: f64) -> SunState {
    let eph = SpiceEphemeris::loaded().unwrap();
    let start = eph.spk().try_to_ssb(object.clone()).unwrap();
    let state = start
        .propagate_with(&NBody::new(&eph, false), reference() + days)
        .unwrap();
    eph.spk().try_to_sun(state).unwrap()
}

/// Observations from the Earth of `object`, a state at the reference time
/// moving under the n-body model, at `times` days from the reference time,
/// with `noise` radians of error on each axis of the sky.
fn observed(object: &SunState, times: &[f64], noise: f64) -> Vec<AstrometricObservation> {
    let mut rng = Rng(3);
    times
        .iter()
        .map(|&days| {
            let time = reference() + days;
            let observer = earth(time);
            let at = sun_centered(&observer).unwrap().pos;
            let mut sight = Sight {
                time,
                observer: at,
                direction: (emitted(&moved(object, days), &at, time).pos - at).normalize(),
                sigma: noise,
            };
            add_noise(&mut sight, noise, &mut rng);
            optical(observer, sight.direction, noise)
        })
        .collect()
}

/// The rules of the tests below: six observations, two on each side of
/// their largest gap, and two time groups at most a day and a half apart.
fn rules() -> ConfirmRules {
    ConfirmRules {
        min_sights: 6,
        max_rms: MAX_RMS,
        min_sights_per_side: 2,
        group_span: SPAN,
        max_gap: 1.5,
    }
}

#[test]
fn link_and_confirm() {
    // Main belt objects in paired visits on five nights, one or three days
    // apart, with twice as much clutter, linked from the pairs of each
    // night. Linking finds every object, and confirmation keeps each with
    // all of its observations and nothing else. Each candidate with a
    // clutter observation from the exposure of its last one added is
    // confirmed without it.
    for spacing in [3.0, 1.0] {
        let times: Vec<f64> = (-2..=2)
            .flat_map(|night| [0.0, 1.0 / 48.0].map(|t| spacing * f64::from(night) + t))
            .collect();
        let mut sky = Sky::new(Population::MainBelt, 4, &times).on_earth();
        sky.clutter = 8;
        let scene = sky.scene();
        let search = Search {
            confirm: Some(6),
            ..Search::new((0.01, 0.1), 0.5)
        };
        let run = run(&scene, &search);
        assert_eq!(run.linked.found, 4, "{spacing}: found by link");
        assert!(
            all_whole(&run.confirmed_score, 4),
            "{spacing}: {:?}",
            run.confirmed_score
        );

        let salted: Vec<Candidate> = run
            .candidates
            .iter()
            .map(|candidate| {
                let mut sights = candidate.sights.clone();
                let last = sights[sights.len() - 1];
                sights.extend((last + 1..scene.labels.len()).find(|&i| scene.labels[i] == CLUTTER));
                Candidate {
                    sights,
                    state: candidate.state.clone(),
                }
            })
            .collect();
        let confirmed = confirm(
            &scene.observations,
            &salted,
            &ConfirmRules {
                min_sights: 6,
                max_rms: MAX_RMS,
                min_sights_per_side: 2,
                group_span: SPAN,
                max_gap: 0.1,
            },
        )
        .unwrap();
        let score = Score::new(
            confirmed.iter().map(|c| c.sights.as_slice()),
            &scene.labels,
            4,
            6,
        );
        assert!(all_whole(&score, 4), "{spacing}, salted: {score:?}");
    }
}

#[test]
fn confirm_rejects_fixed_sources() {
    // Main belt objects with as much clutter, and false sources fixed on the
    // sky in every exposure, which link groups into candidates under a grid
    // reaching in to half an AU. Confirmation keeps no false object, on five
    // nights, three nights, and two nights four days apart.
    // Two nights are confirmed only within the largest gap of a pair, so
    // the search of two nights four days apart pairs across them.
    let four_days = [-2.0, -2.0 + 0.028, 2.0, 2.0 + 0.028];
    for (name, times, min_sights, max_gap) in [
        ("paired 5 nights", nightly(-2, 2, 2, 0.028), 6, 0.1),
        ("paired 3 nights", nightly(-1, 1, 2, 0.028), 6, 0.1),
        ("2 nights 4 days apart", four_days.to_vec(), 4, 4.1),
    ] {
        let mut sky = Sky::new(Population::MainBelt, 8, &times).on_earth();
        sky.clutter = 8;
        sky.stationary = 6;
        let mut search = Search::new((0.01, max_gap), 0.5).span(SPAN);
        search.depths = Some((0.5, 3.0));
        search.min_sights = min_sights;
        search.confirm = Some(min_sights);
        let run = run(&sky.scene(), &search);
        let confirmed = run.confirmed_score;
        assert!(
            run.linked.false_groups > 0,
            "{name}: no fixed sources linked"
        );
        assert_eq!(confirmed.false_groups, 0, "{name}: false objects");
        assert_eq!(confirmed.pure, confirmed.groups, "{name}: purity");
        assert!(
            confirmed.found + 2 >= confirmed.findable,
            "{name}: found by confirm {confirmed:?}"
        );
    }
}

#[test]
fn confirm_short_arc() {
    // Main belt objects on three nights of paired visits, each given as a
    // candidate of its own observations and true state. The orbit fit
    // refuses some of these arcs even from the true state, as their
    // observations do not constrain every orbit parameter. Every object is
    // confirmed with all of its observations, those on the residual of the
    // start, with a fit which reports that it did not converge.
    let scene = Sky::new(Population::MainBelt, 8, &nightly(-1, 1, 2, 0.028))
        .on_earth()
        .scene();
    let eph = SpiceEphemeris::loaded().unwrap();
    let mut refused = 0;
    for (label, object) in scene.objects.iter().enumerate() {
        let own: Vec<usize> = (0..scene.labels.len())
            .filter(|&i| scene.labels[i] == label)
            .collect();
        let chosen: Vec<_> = own.iter().map(|&i| scene.observations[i].clone()).collect();
        let start = eph.spk().try_to_ssb(object.clone()).unwrap();
        let fit = fit_orbit(&start, &chosen, false, None, 50, 9.0, 3, None);
        refused += usize::from(fit.is_err());
        let candidate = Candidate {
            sights: own.clone(),
            state: object.clone(),
        };
        let confirmed = confirm(
            &scene.observations,
            &[candidate],
            &ConfirmRules {
                min_sights: 6,
                max_rms: MAX_RMS,
                min_sights_per_side: 2,
                group_span: SPAN,
                max_gap: 1.5,
            },
        )
        .unwrap();
        assert_eq!(confirmed.len(), 1, "object {label}");
        assert_eq!(confirmed[0].sights, own, "object {label}");
        if fit.is_err() {
            assert!(!confirmed[0].fit.converged, "object {label}");
        }
    }
    assert!(refused > 0, "the fit refused none of the arcs");
}

#[test]
fn confirm_from_far_along_the_sight() {
    // On five nights, the state of a candidate found under a hypothesis far
    // from the truth: the true state seen from the Earth at twice the range,
    // with the same direction and rate of motion on the sky. It predicts the
    // observations to within the grouping tolerance, and the orbit fit alone
    // does not reach the solution from it. The object is confirmed with all
    // of its observations.
    let scene = Sky::new(Population::MainBelt, 1, &nightly(-2, 2, 2, 1.0 / 48.0))
        .on_earth()
        .scene();
    let object = &scene.objects[0];
    let home = sun_centered(&earth(reference())).unwrap();
    let state = State {
        pos: home.pos + (object.pos - home.pos) * 2.0,
        vel: home.vel + (object.vel - home.vel) * 2.0,
        ..object.clone()
    };
    let count = scene.observations.len();
    let candidate = Candidate {
        sights: (0..count).collect(),
        state,
    };
    let confirmed = confirm(
        &scene.observations,
        &[candidate],
        &ConfirmRules {
            min_sights: 6,
            max_rms: MAX_RMS,
            min_sights_per_side: 2,
            group_span: SPAN,
            max_gap: 1.5,
        },
    )
    .unwrap();
    assert_eq!(confirmed.len(), 1);
    assert_eq!(confirmed[0].sights.len(), count);
}

#[test]
fn confirm_with_tied_and_reversed_times() {
    // A main belt object seen on five nights, listed latest first, with two
    // clutter observations at the very times of two of its own, one listed
    // before it and one after. The fit rejects the clutter, and what it kept
    // maps back to the indices of the object.
    let scene = Sky::new(Population::MainBelt, 1, &nightly(-2, 2, 2, 1.0 / 48.0))
        .on_earth()
        .scene();
    let mut rng = Rng(62);
    let mut observations = Vec::new();
    let mut own = Vec::new();
    for (visit, (observation, sight)) in scene.observations.iter().zip(&scene.sights).enumerate() {
        let mut clutter = || {
            let off = (sight.direction + rng.direction() * (30.0 * ARCSEC)).normalize();
            optical(earth(sight.time), off, sight.sigma)
        };
        if visit == 3 {
            observations.push(clutter());
        }
        own.push(observations.len());
        observations.push(observation.clone());
        if visit == 6 {
            observations.push(clutter());
        }
    }
    let count = observations.len();
    observations.reverse();
    let mut own: Vec<usize> = own.into_iter().map(|i| count - 1 - i).collect();
    own.sort_unstable();

    let candidate = Candidate {
        sights: (0..count).collect(),
        state: scene.objects[0].clone(),
    };
    let confirmed = confirm(
        &observations,
        &[candidate],
        &ConfirmRules {
            min_sights: 6,
            max_rms: MAX_RMS,
            min_sights_per_side: 2,
            group_span: SPAN,
            max_gap: 1.5,
        },
    )
    .unwrap();
    assert_eq!(confirmed.len(), 1);
    assert_eq!(confirmed[0].sights, own);
}

#[test]
fn confirm_rejects_bad_settings() {
    for max_rms in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
        let result = confirm(
            &[],
            &[],
            &ConfirmRules {
                min_sights: 6,
                max_rms,
                min_sights_per_side: 2,
                group_span: SPAN,
                max_gap: 1.5,
            },
        );
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "max_rms {max_rms}"
        );
    }
    for min_sights in [0, 3] {
        let result = confirm(
            &[],
            &[],
            &ConfirmRules {
                min_sights,
                max_rms: MAX_RMS,
                min_sights_per_side: 2,
                group_span: SPAN,
                max_gap: 1.5,
            },
        );
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "min_sights {min_sights}"
        );
    }
    let result = confirm(
        &[],
        &[],
        &ConfirmRules {
            min_sights_per_side: 0,
            ..ConfirmRules::default()
        },
    );
    assert!(
        matches!(result, Err(Error::ValueError(_))),
        "min_sights_per_side 0"
    );
    for (group_span, max_gap) in [
        (f64::NAN, 1.5),
        (f64::INFINITY, 1.5),
        (-1.0, 1.5),
        (SPAN, f64::NAN),
        (SPAN, -1.0),
    ] {
        let result = confirm(
            &[],
            &[],
            &ConfirmRules {
                min_sights: 4,
                max_rms: MAX_RMS,
                min_sights_per_side: 2,
                group_span,
                max_gap,
            },
        );
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "group_span {group_span}, max_gap {max_gap}"
        );
    }
    assert!(
        confirm(
            &[],
            &[],
            &ConfirmRules {
                min_sights: 4,
                max_rms: MAX_RMS,
                min_sights_per_side: 2,
                group_span: SPAN,
                max_gap: 1.5
            }
        )
        .unwrap()
        .is_empty()
    );
    let unlimited = confirm(
        &[],
        &[],
        &ConfirmRules {
            min_sights: 4,
            max_rms: MAX_RMS,
            min_sights_per_side: 2,
            group_span: 0.0,
            max_gap: f64::INFINITY,
        },
    );
    assert!(unlimited.unwrap().is_empty());
}

#[test]
fn confirm_needs_two_sights_across_the_gap() {
    // Two main belt objects, each seen four times in forty minutes, twice a
    // day later, and once two days after that. With one observation needed
    // on each side of the largest gap, every candidate of one object is
    // confirmed whole. With two, the observations on a side with fewer are
    // dropped and the rest confirmed: the object without its single
    // observation across the gap. Four observations of one object and one of
    // the other are never confirmed together. Every distinct time is a time
    // group of its own here, with no gap limit, so that this rule is the one
    // tested.
    let times = [0.0, 0.01, 0.02, 0.03, 1.0, 1.01, 3.0];
    let scene = Sky::new(Population::MainBelt, 2, &times).on_earth().scene();
    // Observations `ks` of object `label`.
    let of = |label: usize, ks: &[usize]| ks.iter().map(|&k| 2 * k + label).collect::<Vec<_>>();
    assert!((0..times.len()).all(|k| scene.labels[2 * k] == 0 && scene.labels[2 * k + 1] == 1));
    let confirmed = |sights: Vec<usize>, per_side: usize| {
        let candidate = Candidate {
            sights,
            state: scene.objects[0].clone(),
        };
        let rules = ConfirmRules {
            min_sights: 4,
            max_rms: MAX_RMS,
            min_sights_per_side: per_side,
            group_span: 0.0,
            max_gap: f64::INFINITY,
        };
        let found = confirm(&scene.observations, &[candidate], &rules).unwrap();
        found.into_iter().map(|c| c.sights).collect::<Vec<_>>()
    };
    let cases = [
        (
            "4 and 1",
            1,
            of(0, &[0, 1, 2, 3, 4]),
            of(0, &[0, 1, 2, 3, 4]),
        ),
        ("4 and 1", 2, of(0, &[0, 1, 2, 3, 4]), of(0, &[0, 1, 2, 3])),
        (
            "4, 2 and 1",
            1,
            of(0, &[0, 1, 2, 3, 4, 5, 6]),
            of(0, &[0, 1, 2, 3, 4, 5, 6]),
        ),
        (
            "4, 2 and 1",
            2,
            of(0, &[0, 1, 2, 3, 4, 5, 6]),
            of(0, &[0, 1, 2, 3, 4, 5]),
        ),
        ("2 and 2", 2, of(0, &[2, 3, 4, 5]), of(0, &[2, 3, 4, 5])),
    ];
    for (name, per_side, sights, want) in cases {
        assert_eq!(
            confirmed(sights, per_side),
            vec![want],
            "{name}, {per_side} per side"
        );
    }
    let mixed = [of(0, &[0, 1, 2, 3]), of(1, &[4])].concat();
    let found = confirmed(mixed, 2);
    assert!(
        found.iter().flatten().all(|&i| scene.labels[i] == 0),
        "two objects: {found:?}"
    );

    // The orbit fit of the object with one observation across the gap is
    // good, so the rule alone sets the 4 and 1 apart.
    let chosen: Vec<_> = of(0, &[0, 1, 2, 3, 4])
        .iter()
        .map(|&i| scene.observations[i].clone())
        .collect();
    let start = SpiceEphemeris::loaded()
        .unwrap()
        .spk()
        .try_to_ssb(scene.objects[0].clone())
        .unwrap();
    let fit = fit_orbit(&start, &chosen, false, None, 50, 9.0, 3, None)
        .or_else(|_| fit_orbit(&start, &chosen, false, None, 0, 9.0, 3, None))
        .unwrap();
    assert!(fit.rms <= MAX_RMS, "rms {}", fit.rms);
}

#[test]
fn confirm_needs_two_time_groups() {
    // A main belt object seen in pairs forty minutes apart on nights zero,
    // four and eight, given as candidates of its own observations and true
    // state, confirmed with time groups of up to a tenth of a day and two
    // groups at most a day and a half apart.
    let times = [0.0, 0.028, 4.0, 4.028, 8.0, 8.028];
    let scene = Sky::new(Population::MainBelt, 1, &times).on_earth().scene();
    let confirmed = |sights: &[usize], group_span: f64, max_gap: f64| {
        let candidate = Candidate {
            sights: sights.to_vec(),
            state: scene.objects[0].clone(),
        };
        let rules = ConfirmRules {
            min_sights: 4,
            max_rms: MAX_RMS,
            min_sights_per_side: 2,
            group_span,
            max_gap,
        };
        let found = confirm(&scene.observations, &[candidate], &rules).unwrap();
        found.len() == 1
    };
    // Three groups have no gap limit.
    assert!(confirmed(&[0, 1, 2, 3, 4, 5], SPAN, 1.5), "three groups");
    // Two groups four days apart are refused, unless the gap limit
    // reaches them.
    assert!(!confirmed(&[0, 1, 2, 3], SPAN, 1.5), "two groups");
    assert!(confirmed(&[0, 1, 2, 3], SPAN, 4.1), "two groups, wider gap");
    assert!(
        confirmed(&[0, 1, 2, 3], SPAN, f64::INFINITY),
        "no gap limit"
    );

    // Four observations of one night, a single run, are one group, and
    // confirm nothing. As four groups of their own, they are confirmed.
    let night = [0.0, 0.01, 0.02, 0.03];
    let scene = Sky::new(Population::MainBelt, 1, &night).on_earth().scene();
    let candidate = Candidate {
        sights: vec![0, 1, 2, 3],
        state: scene.objects[0].clone(),
    };
    let count = |group_span: f64| {
        let rules = ConfirmRules {
            min_sights: 4,
            max_rms: MAX_RMS,
            min_sights_per_side: 2,
            group_span,
            max_gap: 1.5,
        };
        let found = confirm(
            &scene.observations,
            std::slice::from_ref(&candidate),
            &rules,
        );
        found.unwrap().len()
    };
    assert_eq!(count(SPAN), 0, "a lone run");
    assert_eq!(count(0.0), 1, "four groups");
}

#[test]
fn search_in_passes() {
    // Main belt and near-Earth objects in paired visits on three nights,
    // with clutter. One pass at the default grid steps misses near-Earth
    // objects, and the default second pass at three tenths of the steps, over
    // the observations the first did not confirm, finds them. Nothing false
    // is confirmed. A search with no passes is refused.
    let times = nightly(-1, 1, 2, 0.028);
    let belt = Sky {
        clutter: 4,
        ..Sky::new(Population::MainBelt, 4, &times).on_earth()
    };
    let near = Sky {
        clutter: 4,
        elong: 120.0,
        radius: 3.0,
        distances: Some((NEAR_DEPTH, 0.3)),
        seed: 11,
        ..Sky::new(Population::NearEarth, NEAR, &times).on_earth()
    };
    let (belt, near) = (belt.scene(), near.scene());
    let objects = belt.objects.len() + near.objects.len();
    let shift = |label: usize| {
        if label == CLUTTER {
            label
        } else {
            label + belt.objects.len()
        }
    };
    let labels: Vec<usize> = belt
        .labels
        .iter()
        .copied()
        .chain(near.labels.iter().map(|&label| shift(label)))
        .collect();
    let observations = [belt.observations, near.observations].concat();
    let settings = SearchSettings {
        min_gap: 0.01,
        min_depth: NEAR_DEPTH / 1.5,
        max_depth: 4.0,
        confirm: ConfirmRules {
            max_gap: 0.1,
            min_sights: 6,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    };
    // Passes at these grid factors, each with a rate limit above the motion
    // of every object, so that the grids alone make the difference.
    let score = |factors: Vec<f64>| {
        let passes = factors
            .into_iter()
            .map(|grid_factor| SearchPass {
                grid_factor,
                max_rate: 20.0_f64.to_radians(),
            })
            .collect();
        let found = search(
            &observations,
            None,
            &SearchSettings {
                passes,
                ..settings.clone()
            },
        )
        .unwrap();
        Score::new(
            found.iter().map(|c| c.sights.as_slice()),
            &labels,
            objects,
            6,
        )
    };
    let one = score(vec![1.0]);
    let two = score(
        SearchSettings::default()
            .passes
            .iter()
            .map(|pass| pass.grid_factor)
            .collect(),
    );
    for (name, score) in [("one pass", one), ("two passes", two)] {
        assert_eq!(score.false_groups, 0, "{name}: {score:?}");
        assert_eq!(score.pure, score.groups, "{name}: {score:?}");
    }
    assert!(one.found < objects, "one pass found all: {one:?}");
    assert_eq!(two.found, objects, "two passes: {two:?}");
    let none = SearchSettings {
        passes: Vec::new(),
        ..settings
    };
    assert!(
        matches!(
            search(&observations, None, &none),
            Err(Error::ValueError(_))
        ),
        "no passes"
    );
}

#[test]
fn search_names_an_unusable_observation() {
    // An optical observation without an uncertainty, after a radar
    // observation, is refused by its index among all the observations.
    let time = reference();
    let radar = AstrometricObservation::RadarRange {
        xmit_lat_rad: 0.0,
        xmit_lon_rad: 0.0,
        xmit_height_km: 0.0,
        rcvr_lat_rad: 0.0,
        rcvr_lon_rad: 0.0,
        rcvr_height_km: 0.0,
        epoch: time,
        range: 1.0,
        sigma_range: 1e-8,
    };
    let direction = Vector::new([1.0, 0.0, 0.0]);
    let good = optical(earth(time), direction, ARCSEC);
    let exact = optical(earth(time + 1.0), direction, 0.0);
    assert!(sight_from_observation(&exact).is_err());
    let refused = search(&[radar, good, exact], None, &SearchSettings::default()).unwrap_err();
    assert!(refused.to_string().contains("observation 2"), "{refused}");
}

#[test]
fn confirm_is_the_same_on_any_threads() {
    // Main belt objects in three visits on each of three nights. The first
    // half are each a candidate of all nine of their observations and of
    // five, the second half a candidate of five only. Candidates of the two
    // outer nights, two days apart, more than the largest gap of two time
    // groups, are turned down after their two-body fits, and put every
    // candidate of five past the first batch, where those of the first half
    // are skipped for their overlap and those of the second half confirmed.
    // Each object is confirmed once, by its own observations, and the result
    // is the same on one thread and on eight.
    const OBJECTS: usize = 12;
    const OUTER: usize = 520;
    let scene = Sky::new(Population::MainBelt, OBJECTS, &nightly(-1, 1, 3, 0.028))
        .on_earth()
        .scene();
    let own = |label: usize, keep: &[usize]| Candidate {
        sights: (0..scene.labels.len())
            .filter(|&i| scene.labels[i] == label)
            .enumerate()
            .filter(|(k, _)| keep.contains(k))
            .map(|(_, i)| i)
            .collect(),
        state: scene.objects[label].clone(),
    };
    let half = OBJECTS / 2;
    let mut candidates: Vec<Candidate> = (0..half)
        .map(|label| own(label, &[0, 1, 2, 3, 4, 5, 6, 7, 8]))
        .collect();
    candidates.extend((0..OUTER).map(|k| own(k % OBJECTS, &[0, 1, 2, 6, 7, 8])));
    for label in 0..OBJECTS {
        candidates.push(own(label, &[0, 1, 4, 7, 8]));
        if label < half {
            candidates.push(own(label, &[0, 2, 3, 6, 8]));
        }
    }
    assert!(half + OUTER > 512);
    let on = |threads: usize| {
        let found = on_threads(threads, || {
            confirm(
                &scene.observations,
                &candidates,
                &ConfirmRules {
                    min_sights: 4,
                    max_rms: MAX_RMS,
                    min_sights_per_side: 2,
                    group_span: SPAN,
                    max_gap: 1.5,
                },
            )
        })
        .unwrap();
        found
            .into_iter()
            .map(|c| (c.sights, c.fit.rms.to_bits()))
            .collect::<Vec<_>>()
    };
    let one = on(1);
    let mut labels: Vec<usize> = one
        .iter()
        .map(|(sights, _)| {
            let label = scene.labels[sights[0]];
            assert!(sights.iter().all(|&i| scene.labels[i] == label));
            label
        })
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, (0..OBJECTS).collect::<Vec<_>>());
    assert_eq!(one, on(8));
}

#[test]
fn each_pass_has_its_own_rate_limit() {
    // Main belt objects at opposition, moving a few tenths of a degree a
    // day, in paired visits on three nights. A pass whose rate limit is
    // below their motion forms none of their units and finds none of them;
    // a second pass with a larger limit, over what the first left, finds
    // them all, and so does that pass alone.
    let scene = Sky::new(Population::MainBelt, 6, &nightly(-1, 1, 2, 0.028))
        .on_earth()
        .scene();
    let pass = |rate: f64| SearchPass {
        grid_factor: 1.0,
        max_rate: rate.to_radians(),
    };
    let whole = |passes: Vec<SearchPass>| {
        let settings = SearchSettings {
            min_gap: 0.01,
            passes,
            confirm: ConfirmRules {
                max_gap: 0.1,
                min_sights: 6,
                ..ConfirmRules::default()
            },
            ..SearchSettings::default()
        };
        let found = search(&scene.observations, None, &settings).unwrap();
        let score = Score::new(
            found.iter().map(|c| c.sights.as_slice()),
            &scene.labels,
            scene.objects.len(),
            6,
        );
        assert_eq!(score.false_groups, 0, "{score:?}");
        score.whole
    };
    assert_eq!(whole(vec![pass(0.02)]), 0);
    assert_eq!(whole(vec![pass(0.02), pass(1.0)]), 6);
    assert_eq!(whole(vec![pass(1.0)]), 6);
}

#[test]
fn confirm_turns_down_unbound_orbits() {
    // An object two AU from the Sun near opposition, moving at twice the
    // escape speed there, seen in paired visits on eleven nights. The orbit
    // fit from its true state fits its observations to their noise with an
    // unbound orbit, and the candidate of all of them is turned down: only
    // bound orbits are in the scope of the crate.
    let home = sun_centered(&earth(reference())).unwrap();
    let outward = home.pos.normalize();
    let along = home.vel.normalize();
    let pos = home.pos + outward;
    let speed = 2.0 * (2.0 * GMS / pos.norm()).sqrt();
    let object = SunState {
        pos,
        vel: (along + outward * 0.5).normalize() * speed,
        ..home
    };
    let observations = observed(&object, &nightly(-5, 5, 2, 0.028), 0.1 * ARCSEC);
    let start = SpiceEphemeris::loaded()
        .unwrap()
        .spk()
        .try_to_ssb(object.clone())
        .unwrap();
    let fit = fit_orbit(&start, &observations, false, None, 50, 9.0, 3, None).unwrap();
    let elements = &fit.uncertain_state.elements;
    assert!(fit.rms <= MAX_RMS, "rms {}", fit.rms);
    assert!(elements.ecc_f.hypot(elements.ecc_g) > 1.0, "{elements:?}");

    let candidate = Candidate {
        sights: (0..observations.len()).collect(),
        state: object,
    };
    assert!(
        confirm(&observations, &[candidate], &rules())
            .unwrap()
            .is_empty()
    );
}

#[test]
fn confirm_turns_down_candidates_over_a_third_strays() {
    // A main belt object seen on five nights, ten observations, with
    // clutter observations 30 arcseconds off it at the times of the first
    // few of them. With five, a third of the fifteen, the object is
    // confirmed without them. With six, more than a third of the sixteen,
    // the fits drop strays only down to two thirds of the candidate, and it
    // is turned down.
    let scene = Sky::new(Population::MainBelt, 1, &nightly(-2, 2, 2, 1.0 / 48.0))
        .on_earth()
        .scene();
    let mut rng = Rng(19);
    let mut confirmed = |strays: usize| {
        let mut observations = scene.observations.clone();
        observations.extend(scene.sights[..strays].iter().map(|sight| {
            let off = (sight.direction + rng.direction() * (30.0 * ARCSEC)).normalize();
            optical(earth(sight.time), off, sight.sigma)
        }));
        let candidate = Candidate {
            sights: (0..observations.len()).collect(),
            state: scene.objects[0].clone(),
        };
        let found = confirm(&observations, &[candidate], &rules()).unwrap();
        found.into_iter().map(|c| c.sights).collect::<Vec<_>>()
    };
    let own: Vec<usize> = (0..scene.observations.len()).collect();
    assert_eq!(confirmed(5), vec![own]);
    assert!(confirmed(6).is_empty());
}

#[test]
fn confirm_short_arc_far_from_the_reference_time() {
    // Main belt objects on three nights of paired visits with a hundredth of
    // an arcsecond of noise, each a candidate of its own observations and
    // its true state fifteen days after the middle night, as a hypothesis
    // with that reference time gives it. The orbit fit refuses some of these
    // arcs, so their start is scored as it is. The start is at the middle of
    // the observations, where leaving out the planets over the time to the
    // reference time does not show in its residuals, and every object is
    // confirmed with all of its observations.
    let scene = Sky {
        noise: 0.01 * ARCSEC,
        ..Sky::new(Population::MainBelt, 8, &nightly(-1, 1, 2, 0.028)).on_earth()
    }
    .scene();
    let mut refused = 0;
    for (label, object) in scene.objects.iter().enumerate() {
        let own: Vec<usize> = (0..scene.labels.len())
            .filter(|&i| scene.labels[i] == label)
            .collect();
        let chosen: Vec<_> = own.iter().map(|&i| scene.observations[i].clone()).collect();
        let start = SpiceEphemeris::loaded()
            .unwrap()
            .spk()
            .try_to_ssb(object.clone())
            .unwrap();
        refused += usize::from(fit_orbit(&start, &chosen, false, None, 50, 9.0, 3, None).is_err());
        let candidate = Candidate {
            sights: own.clone(),
            state: moved(object, 15.0),
        };
        let confirmed = confirm(&scene.observations, &[candidate], &rules()).unwrap();
        assert_eq!(confirmed.len(), 1, "object {label}");
        assert_eq!(confirmed[0].sights, own, "object {label}");
    }
    assert!(refused > 0, "the fit refused none of the arcs");
}
