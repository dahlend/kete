// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Grouping units into candidates, alone and end to end.

use kete_core::constants::GMS;
use kete_core::desigs::Desig;
use kete_core::frames::{SunCenter, Vector};
use kete_core::prelude::{Error, State};
use kete_core::time::Time;

use crate::{Candidate, Hypothesis, LinkSettings, Sight, form_units, link};

use super::harness::{Search, Sky, nightly, print_header, run};
use super::reference;
use super::synthetic::{
    ARCSEC, Case, Field, HOUR, Observer, Population, Rng, SECOND, add_noise, nearest_axis, sight,
    true_hypothesis,
};

#[test]
fn link_cadences() {
    // Main belt objects with some clutter, seen from the ground at cadences
    // from a minute to a day, at the default settings. At the default grid
    // steps every object is found by a candidate of its own sights alone,
    // though one seen once a night may be split into several. At steps an
    // eighth of the default, as a second search would take, every object is
    // also found whole.
    let rows = [
        (
            "paired 40 min, 5 nights",
            nightly(-2, 2, 2, 0.028),
            Search::new((0.01, 0.1), 0.5),
        ),
        (
            "4 a minute apart, 3 nights",
            nightly(-1, 1, 4, 60.0 * SECOND),
            Search::new((30.0 * SECOND, 0.1), 0.5),
        ),
        (
            "1 a night, 15 nights",
            nightly(-7, 7, 1, 0.0),
            Search::new((0.5, 1.1), 0.5),
        ),
    ];
    print_header("Main belt, 6 objects and 3 clutter per exposure");
    for (name, times, search) in rows {
        let mut sky = Sky::new(Population::MainBelt, 6, &times);
        sky.clutter = 3;
        let scene = sky.scene();
        let coarse = run(&scene, &search);
        coarse.print(name);
        let score = coarse.linked;
        assert_eq!(score.findable, 6, "{name}");
        assert_eq!(score.pure_found, 6, "{name}: found pure");
        let fine = run(&scene, &search.coarse(0.125));
        fine.print(&format!("{name}, steps x1/8"));
        let score = fine.linked;
        assert_eq!(score.whole, 6, "{name}, steps x1/8: found whole");
        assert_eq!(score.pure_found, 6, "{name}, steps x1/8: found pure");
    }
}

#[test]
fn link_observers() {
    // Main belt objects in paired visits on five nights, by one observer of
    // each kind away from the ground, and by two observers together. Every
    // object is found, and nearly all candidates are pure.
    let epoch = reference();
    // Name, observers, and elongation and latitude of the field.
    let rows = [
        ("low orbit", vec![Observer::low_orbit(epoch)], 180.0, 0.0),
        ("L1-like", vec![Observer::sunward(epoch)], 180.0, 0.0),
        ("0.72 AU", vec![Observer::inner(epoch)], 150.0, 0.0),
        ("polar 1.5 AU", vec![Observer::polar(epoch)], 60.0, -40.0),
        (
            "ground and low orbit",
            vec![Observer::ground(epoch), Observer::low_orbit(epoch)],
            180.0,
            0.0,
        ),
    ];
    let times = nightly(-2, 2, 2, 0.028);
    print_header("Main belt, 6 objects, paired visits on five nights");
    for (name, observers, elong, lat) in rows {
        let mut sky = Sky::new(Population::MainBelt, 6, &[]);
        sky.exposures = (0..observers.len())
            .flat_map(|who| times.iter().map(move |&t| (who, t + 0.003 * who as f64)))
            .collect();
        sky.observers = observers;
        sky.elong = elong;
        sky.lat = lat;
        let run = run(&sky.scene(), &Search::new((0.01, 0.1), 0.5));
        run.print(name);
        assert_eq!(run.linked.found, run.linked.findable, "{name}");
        assert!(run.linked.purity() >= 90.0, "{name}");
    }
}

#[test]
fn link_false_detections() {
    // Clutter equal to the objects, fixed false sources, and one detection in
    // five missed.
    let mut sky = Sky::new(Population::MainBelt, 15, &nightly(-2, 2, 2, 0.028));
    sky.clutter = 15;
    sky.stationary = 10;
    sky.keep = 0.8;
    sky.noise = 0.3 * ARCSEC;
    let score = run(&sky.scene(), &Search::new((0.01, 0.1), 0.5)).linked;
    assert!(score.completeness() >= 90.0, "completeness {score:?}");
    assert!(score.purity() >= 50.0, "purity {score:?}");
}

#[test]
fn link_near_earth() {
    // Near-Earth objects 0.05 to 0.5 AU away, paired visits on three nights,
    // at grid steps an eighth of the default. The default steps lose some of
    // them, see `matrix_grid_steps`.
    let mut sky = Sky::new(Population::NearEarth, 6, &nightly(-1, 1, 2, 0.028));
    sky.distances = Some((0.05, 0.5));
    sky.elong = 120.0;
    sky.radius = 3.0;
    let search = Search::new((0.01, 0.1), 20.0).coarse(0.125);
    let score = run(&sky.scene(), &search).linked;
    assert!(score.completeness() >= 90.0, "completeness {score:?}");
}

#[test]
fn link_distant_observers_combine() {
    // Paired visits on two nights by the ground site and by a spacecraft
    // 0.3 AU away, linked under the true hypothesis, form one candidate of
    // all eight sights. On a grid they do not, see the row
    // `ground + trailing 0.3 AU` of `matrix_observers`: a depth off by part
    // of a step moves the solved positions of the two observers apart.
    let epoch = reference();
    let ground = Observer::ground(epoch);
    let trailing = Observer::trailing(epoch);
    let origin = ground.state(epoch);
    let mut rng = Rng(3);
    let field = Field::new(&origin.pos, 180.0, 0.0, 1.0);
    for _ in 0..3 {
        let object = Population::MainBelt.in_field(&mut rng, &origin, &field, None);
        let mut sights = Vec::new();
        for observer in [&ground, &trailing] {
            for t in [-2.0, -1.97, 2.0, 2.03] {
                let mut sight = sight(observer, &object, epoch + t).0;
                sight.sigma = 0.1 * ARCSEC;
                sights.push(sight);
            }
        }
        let units = [vec![0, 1], vec![2, 3], vec![4, 5], vec![6, 7]];
        let axis = nearest_axis(&(object.pos - origin.pos));
        let truth = true_hypothesis(&object, &origin, axis);
        let found = link(&sights, &units, &[truth], 4, &LinkSettings::default()).unwrap();
        assert!(
            found.iter().any(|c| c.sights.len() == 8),
            "no candidate of both observers"
        );
    }
}

#[test]
fn match_noise_calibration() {
    // Two pairs an hour long of one object, three days on either side of the
    // reference time, solved under the true hypothesis, along an axis close
    // to the sights and along one at a cosine of one half. They form one
    // group when all four comparisons pass. Noise alone fails one comparison
    // with probability about exp(-threshold^2 / 2), so a match is lost at
    // most about four times as often.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let toward = Vector::new([0.5, 0.85, 0.0275_f64.sqrt()]);
    let pos = origin.pos + toward * 2.2;
    let along = Vector::new([0.0, -0.398, 0.917]).cross(&pos).normalize();
    let speed = (GMS / pos.norm()).sqrt();
    let truth = State::new(Desig::Empty, epoch, pos, along * speed, SunCenter);
    let exact: Vec<Sight> = [-3.0, -3.0 + HOUR, 3.0, 3.0 + HOUR]
        .map(|days| sight(&planet, &truth, epoch + days).0)
        .to_vec();
    let trials: u32 = 1000;
    for axis in [Vector::new([0.0, 1.0, 0.0]), Vector::new([1.0, 0.0, 0.0])] {
        let hypothesis = true_hypothesis(&truth, &origin, axis);
        for threshold in [2.0, 3.0, 4.0] {
            let settings = LinkSettings {
                threshold,
                grid_allowance: 0.0,
                ..LinkSettings::default()
            };
            let mut rng = Rng(51);
            let mut lost: u32 = 0;
            for _ in 0..trials {
                let mut sights = exact.clone();
                for sight in &mut sights {
                    add_noise(sight, 0.1 * ARCSEC, &mut rng);
                }
                let units = [vec![0, 1], vec![2, 3]];
                let found = link(
                    &sights,
                    &units,
                    std::slice::from_ref(&hypothesis),
                    4,
                    &settings,
                );
                lost += u32::from(!found.unwrap().iter().any(|c| c.sights.len() == 4));
            }
            let share = f64::from(lost) / f64::from(trials);
            // Within the count's own scatter of the bound of four comparisons.
            let four = 4.0 * (-threshold * threshold / 2.0).exp();
            let scatter = (four / f64::from(trials)).sqrt();
            assert!(
                share <= four + 3.0 * scatter,
                "axis {axis:?}, threshold {threshold}: lost {share}"
            );
        }
    }
}

#[test]
fn same_time_sights_conflict() {
    // An object seen once a day, and a second detection in one exposure a
    // little off the first. The pairs of both match, but one exposure cannot
    // show one object twice, and the closer detection is kept.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("main belt e180").object(&origin);
    let mut sights: Vec<Sight> = (-3..=3)
        .map(|day| sight(&planet, &truth, epoch + f64::from(day)).0)
        .collect();
    let mut twin = sights[3];
    let side = twin
        .direction
        .cross(&Vector::new([0.0, 0.0, 1.0]))
        .normalize();
    twin.direction = (twin.direction + side * (2.0 * ARCSEC)).normalize();
    sights.push(twin);
    for sight in &mut sights {
        sight.sigma = 0.5 * ARCSEC;
    }
    let units = form_units(&sights, 0.5, 1.1, 0.01, 0.0).unwrap();
    let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&(truth.pos - origin.pos)));
    let found = link(&sights, &units, &[hypothesis], 4, &LinkSettings::default()).unwrap();
    assert!(!found.is_empty());
    let both = |c: &Candidate| c.sights.contains(&3) && c.sights.contains(&7);
    assert!(!found.iter().any(both), "a candidate has both detections");
    let whole = found
        .iter()
        .find(|c| c.sights.len() == 7)
        .expect("no candidate of all days");
    assert_eq!(
        whole.sights,
        (0..7).collect::<Vec<_>>(),
        "the twin was kept"
    );
}

#[test]
fn uncertain_pairs_are_left_out() {
    // An object seen once a day for a week about the reference time, and a
    // pair of its sights at day seven. A pair a second long with one
    // arcsecond of noise knows nothing of where the object is at the
    // reference time, and is left out. The same pair an hour long with a
    // tenth of an arcsecond is grouped with the rest.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("main belt e180").object(&origin);
    let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&(truth.pos - origin.pos)));
    for (gap, sigma, kept) in [(SECOND, ARCSEC, false), (HOUR, 0.1 * ARCSEC, true)] {
        let mut sights: Vec<Sight> = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 7.0, 7.0 + gap]
            .map(|days| sight(&planet, &truth, epoch + days).0)
            .to_vec();
        for sight in &mut sights {
            sight.sigma = sigma;
        }
        let units: Vec<Vec<usize>> = (0..8).map(|i| vec![i, i + 1]).collect();
        let found = link(
            &sights,
            &units,
            std::slice::from_ref(&hypothesis),
            4,
            &LinkSettings::default(),
        )
        .unwrap();
        let has = |index| found.iter().any(|c| c.sights.contains(&index));
        assert!(has(0) && has(6), "the week is not linked");
        assert_eq!(has(8), kept, "gap {gap}, sigma {sigma}");
    }
}

#[test]
fn link_rejects_bad_input() {
    // An object seen once a day for five days, which links. Each change
    // below makes the input wrong in one way, and `link` names it.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("main belt e180").object(&origin);
    let sights: Vec<Sight> = (-2..=2)
        .map(|day| sight(&planet, &truth, epoch + f64::from(day)).0)
        .collect();
    let units: Vec<Vec<usize>> = (0..4).map(|i| vec![i, i + 1]).collect();
    let good = true_hypothesis(&truth, &origin, nearest_axis(&(truth.pos - origin.pos)));
    let defaults = LinkSettings::default();
    let refusal = |sights: &[Sight],
                   units: &[Vec<usize>],
                   hypothesis: &Hypothesis,
                   settings: &LinkSettings| {
        match link(sights, units, std::slice::from_ref(hypothesis), 4, settings) {
            Err(Error::ValueError(message)) => message,
            Err(other) => panic!("not a value error: {other}"),
            Ok(_) => String::new(),
        }
    };
    let found = link(&sights, &units, std::slice::from_ref(&good), 4, &defaults).unwrap();
    assert!(
        found.iter().any(|c| c.sights.len() == 5),
        "the object is not linked"
    );

    // A sight which is not usable, whether or not a unit holds it.
    let mut bad = sights.clone();
    bad[3].direction = bad[3].direction * 2.0;
    assert!(refusal(&bad, &units, &good, &defaults).contains("sight 3"));
    let mut bad = sights.clone();
    bad[4].sigma = f64::NAN;
    assert!(refusal(&bad, &units[..2], &good, &defaults).contains("sight 4"));

    // Units which are not two or more sights in time order spanning some
    // time.
    for unit in [vec![1, 9], vec![2, 2], vec![2], vec![3, 2], vec![0, 2, 1]] {
        let wrong = [vec![0, 1], unit];
        assert!(refusal(&sights, &wrong, &good, &defaults).contains("unit 1"));
    }

    // Settings under which nothing can match.
    let nothing = LinkSettings {
        threshold: 0.0,
        grid_allowance: 0.0,
        ..LinkSettings::default()
    };
    assert!(refusal(&sights, &units, &good, &nothing).contains("both zero"));
    let certain = LinkSettings {
        max_uncertainty: 0.0,
        ..LinkSettings::default()
    };
    assert!(refusal(&sights, &units, &good, &certain).contains("max_uncertainty"));
    let negative = LinkSettings {
        max_uncertainty: -1.0,
        ..LinkSettings::default()
    };
    assert!(!refusal(&sights, &units, &good, &negative).is_empty());

    // Hypotheses which are not finite, whose reference time is past what a
    // time holds, or whose direction is not a unit vector.
    let changes: [fn(&mut Hypothesis); 8] = [
        |h| h.depth = f64::NAN,
        |h| h.depth = f64::INFINITY,
        |h| h.depth_rate = f64::NAN,
        |h| h.depth_rate = f64::INFINITY,
        |h| h.origin.pos = Vector::new([f64::NAN; 3]),
        |h| h.origin.epoch = Time::new(1e16),
        |h| h.direction = h.direction * 1.001,
        |h| h.direction = Vector::new([0.0; 3]),
    ];
    for (k, change) in changes.iter().enumerate() {
        let mut wrong = good.clone();
        change(&mut wrong);
        let message = refusal(&sights, &units, &wrong, &defaults);
        assert!(message.contains("hypothesis 0"), "change {k}: {message}");
    }
    // A reference time farther outside the four days of the sights of the
    // units than four days.
    let mut late = good.clone();
    late.origin.epoch += 5.9;
    assert!(refusal(&sights, &units, &late, &defaults).is_empty());
    late.origin.epoch += 0.2;
    assert!(refusal(&sights, &units, &late, &defaults).contains("outside the span"));
    // Without units there is no span, and nothing to find.
    assert!(
        link(&sights, &[], &[late], 4, &defaults)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn runs_across_two_objects_are_left_out() {
    // A main belt object seen five times in an hour, linked as one run
    // under the true hypothesis, is a candidate on its own. With its last two
    // sights a minute of arc to one side, as when a run is joined across two
    // objects, the run fits no orbit within its noise and the grid allowance,
    // and is left out.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    let truth = Case::named("main belt e180").object(&origin);
    let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&(truth.pos - origin.pos)));
    let mut sights: Vec<Sight> = (0..5)
        .map(|k| sight(&planet, &truth, epoch + f64::from(k) * 15.0 * SECOND * 60.0).0)
        .collect();
    for sight in &mut sights {
        sight.sigma = 0.1 * ARCSEC;
    }
    let run = [(0..5).collect::<Vec<usize>>()];
    let linked = |sights: &[Sight]| {
        let found = link(
            sights,
            &run,
            std::slice::from_ref(&hypothesis),
            4,
            &LinkSettings::default(),
        );
        found
            .unwrap()
            .into_iter()
            .map(|c| c.sights)
            .collect::<Vec<_>>()
    };
    assert_eq!(linked(&sights), vec![run[0].clone()]);
    let side = sights[0]
        .direction
        .cross(&Vector::new([0.0, 0.0, 1.0]))
        .normalize();
    for sight in &mut sights[3..] {
        sight.direction = (sight.direction + side * (60.0 * ARCSEC)).normalize();
    }
    assert!(linked(&sights).is_empty());
}
