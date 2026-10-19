// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Unit forming: pairs against a brute-force search and on moving objects, and
//! tracklets in crowds of false detections.

use kete_core::frames::Vector;
use kete_core::prelude::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::Time;

use crate::{Sight, form_units};

use super::harness::on_threads;
use super::reference;
use super::synthetic::{ARCSEC, EXACT, Observer, Population, Rng, observe};

/// The units without tracklets, which are pairs, as index pairs.
fn pairs_only(sights: &[Sight], max_gap: f64, max_rate: f64) -> KeteResult<Vec<(usize, usize)>> {
    let units = form_units(sights, max_gap, max_rate, 0.0, 2, true, 0.0)?;
    Ok(units.iter().map(|unit| (unit[0], unit[1])).collect())
}

/// Pairs by comparing every sight with every other.
fn brute_force(sights: &[Sight], max_gap: f64, max_rate: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, first) in sights.iter().enumerate() {
        for (j, second) in sights.iter().enumerate() {
            let gap = (second.time - first.time).elapsed;
            let angle = first.direction.angle(&second.direction);
            if gap > 0.0 && gap <= max_gap && angle <= max_rate * gap {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

/// Sights clumped around a few directions over the whole sky, so that every
/// limit has pairs just inside and just outside of it.
fn clumped_sights(rng: &mut Rng) -> Vec<Sight> {
    let mut sights = Vec::new();
    for _ in 0..30 {
        let middle = rng.direction();
        for _ in 0..20 {
            sights.push(Sight {
                time: reference() + 3.0 * rng.normal(),
                observer: Vector::new([0.0; 3]),
                direction: (middle + rng.direction() * 0.03).normalize(),
                sigma: EXACT,
            });
        }
    }
    sights
}

/// Gap and rate limits to compare over: narrow, wide, and a rate past the
/// whole sky.
const LIMITS: [(f64, f64); 5] = [
    (0.3, 0.5),
    (1.0, 0.02),
    (3.0, 0.001),
    (10.0, 0.5),
    (2.0, 5.0),
];

#[test]
fn form_pairs_matches_brute_force() {
    let sights = clumped_sights(&mut Rng(7));
    for (max_gap, max_rate) in LIMITS {
        let mut found = pairs_only(&sights, max_gap, max_rate).unwrap();
        found.sort_unstable();
        let expected = brute_force(&sights, max_gap, max_rate);
        assert_eq!(found, expected, "gap {max_gap}, rate {max_rate}");
        assert!(
            found.len() > 10,
            "too few pairs to compare for rate {max_rate}"
        );
    }
}

#[test]
fn form_pairs_rejects_sights_which_are_not_usable() {
    // Each kind of unusable sight, the first one among them, is an error
    // which names its index.
    let good = clumped_sights(&mut Rng(8));
    let changes: [fn(&mut Sight); 9] = [
        |s| s.time = Time::new(f64::NAN),
        |s| s.time = Time::new(f64::INFINITY),
        |s| s.time = Time::new(1e16),
        |s| s.direction = Vector::new([f64::NAN, 0.0, 1.0]),
        |s| s.direction = s.direction * 2.0,
        |s| s.observer = Vector::new([0.0, f64::NAN, 0.0]),
        |s| s.sigma = -1.0,
        |s| s.sigma = 0.0,
        |s| s.sigma = f64::NAN,
    ];
    for (k, change) in changes.iter().enumerate() {
        let index = k * 97 % good.len();
        let mut sights = good.clone();
        change(&mut sights[index]);
        assert!(!sights[index].is_usable(), "change {k}");
        match pairs_only(&sights, 1.0, 0.5) {
            Err(Error::ValueError(message)) => assert!(
                message.contains(&format!("sight {index} ")),
                "change {k}: {message}"
            ),
            other => panic!("change {k}: {other:?}"),
        }
    }

    // A direction off unit length by rounding is usable.
    let mut sights = good.clone();
    sights[0].direction = sights[0].direction * (1.0 + 1e-12);
    assert!(pairs_only(&sights, 1.0, 0.5).is_ok());

    // Nothing to pair is not an error.
    assert!(pairs_only(&[], 1.0, 1.0).unwrap().is_empty());
    assert!(pairs_only(&good[..1], 1.0, 1.0).unwrap().is_empty());
}

#[test]
fn form_pairs_rejects_bad_limits() {
    let sights = clumped_sights(&mut Rng(9));
    for (max_gap, max_rate) in [
        (-1.0, 1.0),
        (1.0, -1.0),
        (f64::NAN, 1.0),
        (1.0, f64::NAN),
        (1.0, 0.0),
        (1.0, f64::INFINITY),
    ] {
        let result = pairs_only(&sights, max_gap, max_rate);
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "gap {max_gap}, rate {max_rate}"
        );
    }
    for margin in [-1.0, f64::NAN, f64::INFINITY] {
        let result = form_units(&sights, 1.0, 1.0, 0.0, 2, true, margin);
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "margin {margin}"
        );
    }
    // A gap without an upper end, and a rate past the whole sky, are allowed.
    let all = pairs_only(&sights, f64::INFINITY, 10.0).unwrap();
    assert_eq!(all.len(), brute_force(&sights, f64::INFINITY, 10.0).len());
}

#[test]
fn form_pairs_links_moving_objects() {
    // Random objects of every kind, seen by two observers at three times.
    let epoch = reference();
    let mut rng = Rng(10);
    let objects: Vec<_> = Population::ALL
        .into_iter()
        .flat_map(|population| [population; 5])
        .map(|population| population.draw(&mut rng, epoch))
        .collect();
    let observers = [Observer::planet(epoch), Observer::sunward(epoch)];
    let times = [0.0, 0.02, 0.9].map(|days| epoch + days);
    let (sights, labels) = observe(&observers, &objects, &times);
    let count = objects.len() * observers.len();

    // With limits wide enough for the fastest and for the parallax between
    // the observers, every two sights of one object are paired.
    let pairs = pairs_only(&sights, 1.0, 400.0).unwrap();
    let same = pairs.iter().filter(|(i, j)| labels[*i] == labels[*j]);
    assert_eq!(same.count(), objects.len() * 12, "pairs of one object");

    // A limit of two arcseconds a minute keeps the pairs of main belt
    // objects from one observer, and drops those of the close objects.
    let slow = pairs_only(&sights, 1.0, 2880.0 * ARCSEC).unwrap();
    let kept = |population: usize| {
        slow.iter()
            .filter(|(i, j)| labels[*i] == labels[*j] && labels[*i] / 5 == population)
            .filter(|(i, j)| i % count / objects.len() == j % count / objects.len())
            .count()
    };
    assert_eq!(kept(0), 5 * 2 * 3, "main belt pairs from one observer");
    assert!(kept(1) < 5 * 2 * 3, "close objects should be too fast");
}

/// Largest rate of the tracklets of [`crowd_with_object`], in radians per day.
const RATE: f64 = 0.01;

/// Longest tracklet of [`crowd_with_object`], in days.
const SPAN: f64 = 0.1;

/// Sights of an object moving at half of [`RATE`] every hundredth of a day
/// within [`SPAN`], and then `count` false detections at times uniform over a
/// day and directions uniform over a square 0.004 radians wide about the
/// object, all with an uncertainty of 0.2 arcseconds, and the number of
/// sights of the object.
fn crowd_with_object(rng: &mut Rng, count: usize) -> (Vec<Sight>, usize) {
    let sight = |days: f64, y: f64, z: f64| Sight {
        time: reference() + days,
        observer: Vector::new([1.0, 0.0, 0.0]),
        direction: Vector::new([1.0, y, z]).normalize(),
        sigma: 1e-6,
    };
    let mut sights: Vec<Sight> = (0..10)
        .map(|k| {
            let days = 0.45 + 0.01 * f64::from(k);
            sight(days, 0.5 * RATE * (days - 0.5), 0.0)
        })
        .collect();
    let object = sights.len();
    for _ in 0..count {
        let days = rng.uniform();
        sights.push(sight(
            days,
            rng.between(-0.002, 0.002),
            rng.between(-0.002, 0.002),
        ));
    }
    (sights, object)
}

#[test]
fn crowds_form_no_tracklets() {
    // A crowd in which each false detection has on average more than one
    // neighbor within the rate limit and the span, and an object in it.
    let (sights, object) = crowd_with_object(&mut Rng(11), 500);
    let neighbors = form_units(&sights[object..], SPAN, RATE, 0.0, 2, true, 0.0).unwrap();
    assert!(neighbors.len() > 500, "{} neighbors", neighbors.len());
    // The false detections form no tracklet of three, and the object keeps
    // its own.
    let units = form_units(&sights, 0.0, RATE, SPAN, 3, true, 0.0).unwrap();
    assert_eq!(units, vec![(0..object).collect::<Vec<_>>()]);
}

#[test]
fn units_do_not_depend_on_threads() {
    // A crowd dense enough for many tracklets of false detections, whose
    // tracklets and pairs are the same, in the same order, on any number of
    // threads.
    let (sights, _) = crowd_with_object(&mut Rng(12), 2000);
    let units = |threads: usize| {
        on_threads(threads, || {
            form_units(&sights, 1.0, RATE, SPAN, 3, false, 0.0).unwrap()
        })
    };
    let one = units(1);
    assert!(one.iter().filter(|unit| unit.len() >= 3).count() > 10);
    assert_eq!(one, units(8));
}
