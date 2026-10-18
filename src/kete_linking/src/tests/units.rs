// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Unit forming without tracklets, as pairs, against a brute-force search and
//! on moving objects.

use kete_core::frames::Vector;
use kete_core::prelude::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::Time;

use crate::{Sight, form_units};

use super::reference;
use super::synthetic::{ARCSEC, EXACT, Observer, Population, Rng, observe};

/// The units without tracklets, which are pairs, as index pairs.
fn pairs_only(
    sights: &[Sight],
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
) -> KeteResult<Vec<(usize, usize)>> {
    let units = form_units(sights, min_gap, max_gap, max_rate, 0.0, 2, true)?;
    Ok(units.iter().map(|unit| (unit[0], unit[1])).collect())
}

/// Pairs by comparing every sight with every other.
fn brute_force(sights: &[Sight], min_gap: f64, max_gap: f64, max_rate: f64) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (i, first) in sights.iter().enumerate() {
        for (j, second) in sights.iter().enumerate() {
            let gap = (second.time - first.time).elapsed;
            let angle = first.direction.angle(&second.direction);
            if gap > 0.0 && gap >= min_gap && gap <= max_gap && angle <= max_rate * gap {
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
const LIMITS: [(f64, f64, f64); 5] = [
    (0.0, 0.3, 0.5),
    (0.01, 1.0, 0.02),
    (0.5, 3.0, 0.001),
    (0.0, 10.0, 0.5),
    (1.0, 2.0, 5.0),
];

#[test]
fn form_pairs_matches_brute_force() {
    let sights = clumped_sights(&mut Rng(7));
    for (min_gap, max_gap, max_rate) in LIMITS {
        let mut found = pairs_only(&sights, min_gap, max_gap, max_rate).unwrap();
        found.sort_unstable();
        let expected = brute_force(&sights, min_gap, max_gap, max_rate);
        assert_eq!(
            found, expected,
            "gap {min_gap} to {max_gap}, rate {max_rate}"
        );
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
        match pairs_only(&sights, 0.0, 1.0, 0.5) {
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
    assert!(pairs_only(&sights, 0.0, 1.0, 0.5).is_ok());

    // Nothing to pair is not an error.
    assert!(pairs_only(&[], 0.0, 1.0, 1.0).unwrap().is_empty());
    assert!(pairs_only(&good[..1], 0.0, 1.0, 1.0).unwrap().is_empty());
}

#[test]
fn form_pairs_rejects_bad_limits() {
    let sights = clumped_sights(&mut Rng(9));
    for (min_gap, max_gap, max_rate) in [
        (-1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (0.0, 1.0, -1.0),
        (f64::NAN, 1.0, 1.0),
        (0.0, f64::NAN, 1.0),
        (0.0, 1.0, f64::NAN),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, f64::INFINITY),
    ] {
        let result = pairs_only(&sights, min_gap, max_gap, max_rate);
        assert!(
            matches!(result, Err(Error::ValueError(_))),
            "gap {min_gap} to {max_gap}, rate {max_rate}"
        );
    }
    // A gap without an upper end, and a rate past the whole sky, are allowed.
    let all = pairs_only(&sights, 0.0, f64::INFINITY, 10.0).unwrap();
    assert_eq!(
        all.len(),
        brute_force(&sights, 0.0, f64::INFINITY, 10.0).len()
    );
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
    let pairs = pairs_only(&sights, 0.01, 1.0, 400.0).unwrap();
    let same = pairs.iter().filter(|(i, j)| labels[*i] == labels[*j]);
    assert_eq!(same.count(), objects.len() * 12, "pairs of one object");

    // A limit of two arcseconds a minute keeps the pairs of main belt
    // objects from one observer, and drops those of the close objects.
    let slow = pairs_only(&sights, 0.01, 1.0, 2880.0 * ARCSEC).unwrap();
    let kept = |population: usize| {
        slow.iter()
            .filter(|(i, j)| labels[*i] == labels[*j] && labels[*i] / 5 == population)
            .filter(|(i, j)| i % count / objects.len() == j % count / objects.len())
            .count()
    };
    assert_eq!(kept(0), 5 * 2 * 3, "main belt pairs from one observer");
    assert!(kept(1) < 5 * 2 * 3, "close objects should be too fast");
}
