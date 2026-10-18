// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Searching in windows of time against searching the whole span, on
//! observations from the Earth of the test kernels of objects moving under
//! the n-body model.

use kete_core::frames::{Equatorial, Vector};
use kete_fitting::AstrometricObservation;

use crate::{ConfirmRules, Confirmed, SearchPass, SearchSettings, search, search_windows};

use super::harness::{CLUTTER, Score, Sky, earth, nightly, on_threads, optical, sorted};
use super::reference;
use super::synthetic::{ARCSEC, Population};

#[test]
fn objects_across_windows_are_merged_whole() {
    // Main belt objects in paired visits on seven nights, searched in
    // windows of four days starting two days apart, the last open-ended:
    // nights -3 to 0, -1 to 2, and 1 to 3. The first window confirms each
    // object on its first four nights, the second has only two nights of it
    // left, too few to confirm, and the third confirms its last three
    // nights. No window holds an object whole, and each comes out once,
    // whole, merged from the pieces of the first and third, as from a search
    // of all the observations at once, alike on one thread and on eight.
    let scene = Sky::new(Population::MainBelt, 2, &nightly(-3, 3, 2, 0.028))
        .on_earth()
        .scene();
    let settings = SearchSettings {
        confirm: ConfirmRules {
            min_sights: 6,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    };
    let whole = search_windows(&scene.observations, None, &settings, None, 0.0, None).unwrap();
    let score = Score::new(
        whole.iter().map(|c| c.sights.as_slice()),
        &scene.labels,
        scene.objects.len(),
        6,
    );
    assert_eq!(score.false_groups, 0, "{score:?}");
    assert_eq!(score.whole, scene.objects.len(), "{score:?}");
    let on = |threads: usize| {
        on_threads(threads, || {
            search_windows(&scene.observations, None, &settings, Some(2.0), 0.0, None)
        })
        .unwrap()
    };
    let windowed = on(1);
    assert_eq!(sorted(&windowed), sorted(&whole));
    let fits =
        |found: &[Confirmed]| -> Vec<u64> { found.iter().map(|c| c.fit.rms.to_bits()).collect() };
    let again = on(8);
    assert_eq!(sorted(&again), sorted(&windowed));
    assert_eq!(fits(&again), fits(&windowed));
}

/// Main belt objects in paired visits on nights 0 to 3, a thirty-second of a
/// day apart, which gives exact Julian dates, with a lone detection far from
/// them at each of `lone` days from the reference time, which sets where the
/// windows start; their labels; and settings which confirm an object only
/// from all eight of its detections.
fn four_nights(
    objects: usize,
    lone: &[f64],
) -> (Vec<AstrometricObservation>, Vec<usize>, SearchSettings) {
    let scene = Sky::new(Population::MainBelt, objects, &nightly(0, 3, 2, 1.0 / 32.0))
        .on_earth()
        .scene();
    let mut observations = scene.observations;
    let mut labels = scene.labels;
    for &days in lone {
        let time = reference() + days;
        let direction = Vector::<Equatorial>::from_ra_dec(0.3, 0.6);
        observations.push(optical(earth(time), direction, 0.1 * ARCSEC));
        labels.push(CLUTTER);
    }
    let settings = SearchSettings {
        confirm: ConfirmRules {
            min_sights: 8,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    };
    (observations, labels, settings)
}

#[test]
fn objects_across_the_start_of_a_window_are_whole() {
    // Objects seen over three days and a little, from day 0, in windows of
    // seven days starting three and a half days apart from a lone detection
    // on day -6, with another on day 8: days -6 to 1, -2.5 to 4.5, and 1 on.
    // The objects cross the start of the last window and the end of the
    // first, and lie whole in the second, so each is found whole though no
    // fewer than all its detections confirm it. Windows holding only the
    // longest unit past their end would have cut every object.
    let (observations, labels, settings) = four_nights(3, &[-6.0, 8.0]);
    let found = search_windows(&observations, None, &settings, Some(3.5), 0.0, None).unwrap();
    let score = Score::new(found.iter().map(|c| c.sights.as_slice()), &labels, 3, 8);
    assert_eq!(score.false_groups, 0, "{score:?}");
    assert_eq!(score.whole, 3, "{score:?}");
}

#[test]
fn last_observation_at_the_end_of_a_window_is_searched() {
    // A lone detection ten and a half days before the last detection of the
    // objects, in windows of seven days starting three and a half days
    // apart: two windows, the second of which would end exactly at the last
    // detection. The last window holds everything from its start on, so the
    // objects, which confirm only with that detection, are found whole.
    let (observations, labels, settings) = four_nights(2, &[3.0 + 1.0 / 32.0 - 10.5]);
    let found = search_windows(&observations, None, &settings, Some(3.5), 0.0, None).unwrap();
    let score = Score::new(found.iter().map(|c| c.sights.as_slice()), &labels, 2, 8);
    assert_eq!(score.false_groups, 0, "{score:?}");
    assert_eq!(score.whole, 2, "{score:?}");
}

#[test]
fn tiles_of_the_longest_window_are_checked_first() {
    // Objects seen on day 0 and on days 7 to 10, in windows of seven days
    // starting three and a half days apart: the first holds day 0 alone, and
    // the second days 7 to 10. At the given nside the tiles are fine for the
    // first window, which can be searched alone, and too small for the
    // second, which is an error before any window is searched.
    let times: Vec<f64> = [0, 7, 8, 9, 10]
        .into_iter()
        .flat_map(|day| [f64::from(day), f64::from(day) + 1.0 / 32.0])
        .collect();
    let scene = Sky::new(Population::MainBelt, 4, &times).on_earth().scene();
    let settings = SearchSettings::default();
    let nside = Some(2048);
    let message = search_windows(&scene.observations, None, &settings, Some(3.5), 0.0, nside)
        .unwrap_err()
        .to_string();
    assert!(message.contains("smaller nside"), "{message}");
    let first: Vec<AstrometricObservation> = scene
        .observations
        .iter()
        .filter(|o| o.epoch().jd() < reference().jd() + 7.0)
        .cloned()
        .collect();
    assert!(search_windows(&first, None, &settings, None, 0.0, nside).is_ok());
}

#[test]
fn search_windows_refuses_bad_settings() {
    // A window which is not finite and positive, shorter than a unit, or so
    // short that it makes too many windows. A window longer than the span is
    // one window over all of it.
    let scene = Sky::new(Population::MainBelt, 2, &nightly(-1, 1, 2, 0.028))
        .on_earth()
        .scene();
    let settings = SearchSettings::default();
    let message = |window: f64, settings: &SearchSettings| {
        search_windows(&scene.observations, None, settings, Some(window), 0.0, None)
            .unwrap_err()
            .to_string()
    };
    for window in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(message(window, &settings).contains("window"));
    }
    // Shorter than the longest unit, the max_gap of the confirmation rules.
    assert!(message(1.0, &settings).contains("max_gap"));
    // A window so short that it would make more windows than are searched.
    let short = SearchSettings {
        min_gap: 0.0,
        run_span: 0.0,
        confirm: ConfirmRules {
            max_gap: 1e-5,
            ..ConfirmRules::default()
        },
        ..settings.clone()
    };
    assert!(message(1e-5, &short).contains("longer window"));
    let one = search_windows(&scene.observations, None, &settings, Some(10.0), 0.0, None).unwrap();
    let all = search_windows(&scene.observations, None, &settings, None, 0.0, None).unwrap();
    assert_eq!(sorted(&one), sorted(&all));
    assert!(
        search_windows(&[], None, &settings, Some(10.0), 0.0, None)
            .unwrap()
            .is_empty()
    );
    // The size of the grid is checked also without optical observations.
    let fine = SearchSettings {
        passes: vec![SearchPass {
            grid_factor: 1e-6,
            ..settings.passes[0]
        }],
        ..settings.clone()
    };
    assert!(
        search(&[], None, &fine)
            .unwrap_err()
            .to_string()
            .contains("hypotheses")
    );
    let message = search_windows(&[], None, &fine, None, 0.0, None)
        .unwrap_err()
        .to_string();
    assert!(message.contains("hypotheses"), "{message}");
}
