// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Uncertainties scaled by station in confirmation, on observations from the
//! Earth of the test kernels of objects moving under the n-body model, by two
//! stations, one of which states uncertainties smaller than its scatter.

use std::collections::HashMap;

use kete_core::frames::Vector;
use kete_fitting::AstrometricObservation;

use crate::{ConfirmRules, Confirmed, SearchSettings, Sight, search, search_windows};

use super::harness::{Score, Sky, on_threads, optical, sorted_with_fits};
use super::reference;
use super::synthetic::{ARCSEC, Population, Rng, add_noise};

/// Scatter of the cautious station on each axis of the sky, in radians.
const CAUTIOUS: f64 = 0.1 * ARCSEC;

/// Uncertainty the cautious station states, a little more than its scatter,
/// in radians.
const CAUTIOUS_STATED: f64 = 0.125 * ARCSEC;

/// Uncertainty the optimistic station states, in radians.
const OPTIMISTIC_STATED: f64 = 0.1 * ARCSEC;

/// Scatter of the optimistic station over what it states.
const OPTIMISM: f64 = 2.0;

/// Label of the cautious station.
const CAUTIOUS_LABEL: usize = 0;

/// Label of the optimistic station.
const OPTIMISTIC_LABEL: usize = 7;

/// Objects in the field both stations see.
const MIXED: usize = 30;

/// Objects in the field only the optimistic station sees.
const ALONE: usize = 10;

/// Observations by two stations, with the label of each one's station and of
/// its object.
struct Stations {
    observations: Vec<AstrometricObservation>,
    station: Vec<usize>,
    labels: Vec<usize>,
}

impl Stations {
    /// Main belt objects on four nights, six exposures a night a fiftieth of a
    /// day apart, from the Earth: in a field at opposition, [`MIXED`] objects
    /// of which the cautious station takes four exposures a night and the
    /// optimistic two; and in a field 40 degrees away, [`ALONE`] objects only
    /// the optimistic station sees.
    fn new() -> Self {
        let times: Vec<f64> = [-2.0, -1.0, 0.0, 1.0]
            .iter()
            .flat_map(|night| (0..6).map(move |k| night + 0.02 * f64::from(k)))
            .collect();
        let field = |elong: f64, objects: usize, seed: u64| {
            Sky {
                elong,
                seed,
                noise: CAUTIOUS,
                ..Sky::new(Population::MainBelt, objects, &times).on_earth()
            }
            .scene()
        };
        let (mixed, alone) = (field(180.0, MIXED, 3), field(140.0, ALONE, 4));
        // The optimistic station takes the third and sixth exposure of each
        // night in the field both see.
        let optimistic_in_mixed = |observation: &AstrometricObservation| {
            let days = observation.epoch().jd() - reference().jd();
            let k = times.iter().position(|&t| (t - days).abs() < 1e-6);
            k.unwrap() % 3 == 2
        };
        let mut rng = Rng(11);
        let mut observations = Vec::new();
        let mut stations = Vec::new();
        let seen = mixed
            .observations
            .iter()
            .map(|o| (o, optimistic_in_mixed(o)))
            .chain(alone.observations.iter().map(|o| (o, true)));
        for (observation, optimistic) in seen {
            let AstrometricObservation::Optical {
                observer, ra, dec, ..
            } = observation
            else {
                unreachable!()
            };
            let mut sight = Sight {
                time: observer.epoch,
                observer: observer.pos,
                direction: Vector::from_ra_dec(*ra, *dec),
                sigma: CAUTIOUS,
            };
            let (stated, station) = if optimistic {
                // More noise on top of the scene's, to the optimistic scatter.
                let scatter = OPTIMISM * OPTIMISTIC_STATED;
                let extra = (scatter.powi(2) - CAUTIOUS.powi(2)).sqrt();
                add_noise(&mut sight, extra, &mut rng);
                (OPTIMISTIC_STATED, OPTIMISTIC_LABEL)
            } else {
                (CAUTIOUS_STATED, CAUTIOUS_LABEL)
            };
            observations.push(optical(observer.clone(), sight.direction, stated));
            stations.push(station);
        }
        let labels = mixed
            .labels
            .iter()
            .copied()
            .chain(alone.labels.iter().map(|&label| label + MIXED))
            .collect();
        Self {
            observations,
            station: stations,
            labels,
        }
    }

    /// The score of a search.
    fn score(&self, found: &[Confirmed]) -> Score {
        Score::new(
            found.iter().map(|c| c.sights.as_slice()),
            &self.labels,
            MIXED + ALONE,
            6,
        )
    }

    /// The factors on the uncertainties of each station that the
    /// observations of the fits of `found` carry, sorted and distinct.
    fn applied(&self, found: &[Confirmed]) -> HashMap<usize, Vec<f64>> {
        // Observations by time and direction, which the fits keep as given.
        let key = |o: &AstrometricObservation| {
            let (ra, dec, _) = o.as_optical().unwrap();
            (o.epoch().jd().to_bits(), ra.to_bits(), dec.to_bits())
        };
        let index: HashMap<_, usize> = self
            .observations
            .iter()
            .enumerate()
            .map(|(i, o)| (key(o), i))
            .collect();
        let mut factors: HashMap<usize, Vec<f64>> = HashMap::new();
        for fitted in found.iter().flat_map(|object| &object.fit.observations) {
            let AstrometricObservation::Optical { sigma_dec, .. } = fitted else {
                unreachable!()
            };
            let station = self.station[index[&key(fitted)]];
            let stated = if station == OPTIMISTIC_LABEL {
                OPTIMISTIC_STATED
            } else {
                CAUTIOUS_STATED
            };
            factors.entry(station).or_default().push(sigma_dec / stated);
        }
        for values in factors.values_mut() {
            values.sort_by(f64::total_cmp);
            values.dedup();
        }
        factors
    }
}

/// The settings of the searches below: six observations and a residual of
/// at most one and a half times the uncertainties confirm.
fn settings() -> SearchSettings {
    SearchSettings {
        confirm: ConfirmRules {
            min_sights: 6,
            max_rms: 1.5,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    }
}

#[test]
fn optimistic_station_is_scaled_in_confirmation() {
    // Objects of the optimistic station alone fail the residual limit with
    // its stated uncertainties, and so do some of the field both stations
    // see, or lose observations to the outlier rule. With the station
    // labels, the first pass confirms enough objects of both to estimate
    // their factors, and with the optimistic station's uncertainties scaled
    // every object is confirmed, nearly all whole, with nothing false.
    //
    // The factor reads below the scatter: the first pass sees only the
    // objects and observations which pass with the stated uncertainties, and
    // the overweighted observations pull their fits toward themselves. The
    // cautious station, which states more than its scatter, keeps its stated
    // uncertainties. The fits of the confirmed objects carry the factors.
    let scene = Stations::new();
    let settings = settings();
    let stated = search(&scene.observations, None, &settings).unwrap();
    let scaled = search(&scene.observations, Some(&scene.station), &settings).unwrap();
    let (before, after) = (scene.score(&stated), scene.score(&scaled));
    assert!(before.found <= MIXED, "{before:?}");
    assert!(after.found >= MIXED + ALONE - 2, "{after:?}");
    assert!(after.whole >= MIXED + ALONE - 4, "{after:?}");
    assert_eq!(after.false_groups, 0, "{after:?}");
    let factors = scene.applied(&scaled);
    assert_eq!(factors[&CAUTIOUS_LABEL], vec![1.0], "{factors:?}");
    let optimistic = &factors[&OPTIMISTIC_LABEL];
    assert_eq!(optimistic.len(), 1, "{factors:?}");
    assert!(
        (1.3..OPTIMISM).contains(&optimistic[0]),
        "{factors:?} for a scatter of {OPTIMISM}"
    );
    // Without stations every fit is made with the stated uncertainties.
    let factors = scene.applied(&stated);
    assert_eq!(factors[&CAUTIOUS_LABEL], vec![1.0], "{factors:?}");
    assert_eq!(factors[&OPTIMISTIC_LABEL], vec![1.0], "{factors:?}");
}

#[test]
fn stations_too_small_change_nothing() {
    // Each observation a station of its own: no station reaches the fewest
    // observations and objects of an estimate, so every search is exactly
    // as without stations, to the bits of every residual.
    let scene = Stations::new();
    let settings = settings();
    let own: Vec<usize> = (0..scene.observations.len()).map(|i| 3 * i + 1).collect();
    let observations = &scene.observations;
    let searches = |stations: Option<&[usize]>| {
        [
            search(observations, stations, &settings).unwrap(),
            search_windows(observations, stations, &settings, None, 0.0, None).unwrap(),
            search_windows(observations, stations, &settings, Some(1.5), 0.0, None).unwrap(),
        ]
        .map(|found| sorted_with_fits(&found))
    };
    assert_eq!(searches(Some(&own)), searches(None));
}

#[test]
fn scaled_searches_are_the_same_on_any_threads() {
    // With stations, each search gives the same objects and fits on one thread
    // and on eight. Tiles scale the optimistic station as a search of all the
    // observations does, and a window longer than the span is the tiled
    // search of the whole span.
    // Each window estimates its own factors from its own first pass: in
    // windows of three days the first, of three nights, scales the
    // optimistic station, and the second holds only what the first left, too
    // few to estimate from, and keeps its stated uncertainties.
    let scene = Stations::new();
    let settings = settings();
    let (observations, stations) = (&scene.observations, Some(scene.station.as_slice()));
    let on = |threads: usize| {
        on_threads(threads, || {
            [
                search(observations, stations, &settings).unwrap(),
                search_windows(observations, stations, &settings, None, 0.0, Some(8)).unwrap(),
                search_windows(observations, stations, &settings, Some(1.5), 0.0, Some(8)).unwrap(),
                search_windows(observations, stations, &settings, Some(10.0), 0.0, Some(8))
                    .unwrap(),
            ]
        })
    };
    let one = on(1);
    let eight = on(8);
    for (a, b) in one.iter().zip(&eight) {
        assert_eq!(sorted_with_fits(a), sorted_with_fits(b));
    }
    let [_, tiled, windowed, single] = &one;
    assert_eq!(sorted_with_fits(single), sorted_with_fits(tiled));
    let score = scene.score(tiled);
    assert!(score.found >= MIXED + ALONE - 2, "{score:?}");
    assert_eq!(score.false_groups, 0, "{score:?}");
    let factors = scene.applied(tiled);
    assert_eq!(factors[&CAUTIOUS_LABEL], vec![1.0], "{factors:?}");
    let optimistic = &factors[&OPTIMISTIC_LABEL];
    assert!(optimistic.len() == 1 && optimistic[0] > 1.3, "{factors:?}");
    let score = scene.score(windowed);
    assert_eq!(score.false_groups, 0, "{score:?}");
    let factors = scene.applied(windowed);
    assert_eq!(factors[&CAUTIOUS_LABEL], vec![1.0], "{factors:?}");
    let optimistic = &factors[&OPTIMISTIC_LABEL];
    assert!(
        optimistic.len() == 2 && optimistic[0] == 1.0 && optimistic[1] > 1.3,
        "{factors:?}"
    );
}

#[test]
fn stations_must_be_one_per_observation() {
    let scene = Stations::new();
    let settings = settings();
    let observations = &scene.observations;
    let short = &scene.station[1..];
    let messages = [
        search(observations, Some(short), &settings).unwrap_err(),
        search_windows(observations, Some(short), &settings, None, 0.0, None).unwrap_err(),
        search_windows(observations, Some(short), &settings, Some(1.5), 0.0, None).unwrap_err(),
    ];
    for message in messages.map(|error| error.to_string()) {
        assert!(message.contains("stations"), "{message}");
    }
}
