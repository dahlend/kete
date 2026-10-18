// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Searching tile by tile against searching everything at once, on
//! observations of objects moving under the n-body model, from the Earth and
//! from a point near the Sun-Earth L1 point.

use std::f64::consts::TAU;
use std::time::Instant;

use kete_core::ephemeris::Ephemeris;
use kete_core::frames::{Equatorial, SSB, Vector};
use kete_core::prelude::State;
use kete_core::propagation::NBody;
use kete_core::time::{TDB, Time};
use kete_fitting::AstrometricObservation;

use crate::search::Gathered;
use crate::{
    ConfirmRules, Confirmed, SearchPass, SearchSettings, merge_pieces, parallax_margin, search,
    search_windows, sight_from_observation, tiles,
};
use kete_spice::prelude::SpiceEphemeris;
use rayon::prelude::*;

use super::harness::{CLUTTER, Score, Sky, earth, nightly, on_threads, optical, sorted};
use super::reference;
use super::synthetic::{ARCSEC, Population, Rng, SunState, add_noise, emitted};

/// The default passes with these rate limits, in degrees a day.
fn with_rates(rates: &[f64]) -> Vec<SearchPass> {
    SearchSettings::default()
        .passes
        .iter()
        .zip(rates)
        .map(|(pass, rate)| SearchPass {
            max_rate: rate.to_radians(),
            ..*pass
        })
        .collect()
}

/// Search each tile of each pass in turn with [`search`], merge them as
/// [`search_windows`] does, and merge the pieces of the passes, with the
/// number of objects all tiles confirmed before the merges and the number of
/// tiles. Unlike [`search_windows`], each tile's search takes the origin
/// nearest the middle of its own units.
fn by_tile(
    observations: &[AstrometricObservation],
    settings: &SearchSettings,
    margin: f64,
    nside: Option<u32>,
) -> (Vec<Confirmed>, usize, usize) {
    let sights: Vec<_> = observations
        .iter()
        .map(|o| sight_from_observation(o).unwrap())
        .collect();
    let mut remaining: Vec<usize> = (0..observations.len()).collect();
    let (mut merged, mut before, mut count) = (Vec::new(), 0, 0);
    for pass in &settings.passes {
        let own: Vec<_> = remaining.iter().map(|&i| sights[i]).collect();
        let tiles: Vec<Vec<usize>> = tiles(&own, pass.max_rate, margin, nside, None)
            .unwrap()
            .into_iter()
            .map(|tile| tile.into_iter().map(|k| remaining[k]).collect())
            .collect();
        let alone = SearchSettings {
            passes: vec![*pass],
            ..settings.clone()
        };
        let found: Vec<Vec<Confirmed>> = tiles
            .iter()
            .map(|tile| {
                let own: Vec<_> = tile.iter().map(|&i| observations[i].clone()).collect();
                search(&own, None, &alone).unwrap()
            })
            .collect();
        before += found.iter().map(Vec::len).sum::<usize>();
        count += tiles.len();
        let mut gathered = Gathered::default();
        for (tile, objects) in tiles.iter().zip(found) {
            gathered.add(objects.into_iter().map(|mut object| {
                object.sights = object.sights.iter().map(|&k| tile[k]).collect();
                object.sights.sort_unstable();
                object
            }));
        }
        let confirmed = gathered.merge();
        let taken: Vec<usize> = confirmed.iter().flat_map(|c| c.sights.clone()).collect();
        remaining.retain(|i| !taken.contains(i));
        merged.extend(confirmed);
    }
    let merged = merge_pieces(observations, merged, &settings.confirm).unwrap();
    (merged, before, count)
}

#[test]
fn tiled_search_equals_untiled() {
    // Main belt objects over a field 15 degrees in radius, in paired visits
    // on three nights, with clutter, cut into tiles about 7 degrees across.
    // Objects near the edge of a core are also whole, or in part, in the
    // tiles next to it, and are confirmed there too. The merge keeps each
    // object once, and the result is the search of all the observations.
    // The second pass has the larger rate limit, so its tiles have larger
    // halos than those of the first. With the parallax margin of two sites
    // on opposite sides of the Earth, for objects down to a tenth of an AU,
    // the result is the same, on one thread and on eight.
    let sky = Sky {
        radius: 15.0,
        clutter: 2,
        seed: 3,
        ..Sky::new(Population::MainBelt, 40, &nightly(-1, 1, 2, 0.028)).on_earth()
    };
    let scene = sky.scene();
    let settings = SearchSettings {
        min_gap: 0.01,
        passes: with_rates(&[0.4, 0.5]),
        confirm: ConfirmRules {
            max_gap: 0.1,
            min_sights: 6,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    };
    let all = search(&scene.observations, None, &settings).unwrap();
    let score = Score::new(
        all.iter().map(|c| c.sights.as_slice()),
        &scene.labels,
        scene.objects.len(),
        6,
    );
    assert_eq!(score.false_groups, 0, "{score:?}");
    assert_eq!(score.whole, score.findable, "{score:?}");
    let (merged, before, count) = by_tile(&scene.observations, &settings, 0.0, Some(8));
    assert!(count > 4, "{count} tiles");
    assert!(
        before > merged.len(),
        "no object was confirmed in two tiles"
    );
    assert_eq!(sorted(&merged), sorted(&all));
    let direct = search_windows(&scene.observations, None, &settings, None, 0.0, Some(8)).unwrap();
    assert_eq!(sorted(&direct), sorted(&all));
    let margin = parallax_margin(2.0 * EARTH_RADIUS, 0.1).unwrap();
    assert!(margin > 0.0);
    let on = |threads: usize| {
        let found = on_threads(threads, || {
            search_windows(&scene.observations, None, &settings, None, margin, Some(8))
        })
        .unwrap();
        sorted(&found)
    };
    let one = on(1);
    assert_eq!(one, sorted(&all));
    assert_eq!(one, on(8));
}

#[test]
fn settings_of_every_pass_are_checked_first() {
    // A bad setting of the second pass is an error of both searches, also
    // over observations which form no units, and so is a second pass whose
    // tiles at the given nside would be too small against its halo.
    let scene = Sky::new(Population::MainBelt, 4, &nightly(-1, 1, 2, 0.028))
        .on_earth()
        .scene();
    let bad = |pass: SearchPass| SearchSettings {
        passes: vec![SearchSettings::default().passes[0], pass],
        ..SearchSettings::default()
    };
    let flat = bad(SearchPass {
        grid_factor: 0.0,
        max_rate: 0.01,
    });
    let still = bad(SearchPass {
        grid_factor: 1.0,
        max_rate: 0.0,
    });
    for observations in [&scene.observations[..], &scene.observations[..1], &[]] {
        for settings in [&flat, &still] {
            assert!(search(observations, None, settings).is_err());
            assert!(search_windows(observations, None, settings, None, 0.0, None).is_err());
        }
    }
    let fast = bad(SearchPass {
        grid_factor: 1.0,
        max_rate: 1.0,
    });
    let message = search_windows(&scene.observations, None, &fast, None, 0.0, Some(256))
        .unwrap_err()
        .to_string();
    assert!(message.contains("smaller nside"), "{message}");
}

/// State of the Sun with respect to the Solar System Barycenter, from the
/// test kernels.
fn sun(time: Time<TDB>) -> State<Equatorial, SSB> {
    let state = SpiceEphemeris::loaded()
        .unwrap()
        .try_get_state_with_center(10, time, 0)
        .unwrap();
    State::try_from(state).unwrap()
}

/// Distance of the L1-like point from the Earth toward the Sun, in AU.
const L1: f64 = 0.01;

/// Radius of the Earth, in AU.
const EARTH_RADIUS: f64 = 4.26e-5;

/// An observer of a survey.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Site {
    /// A point 0.01 AU from the Earth toward the Sun.
    L1,

    /// A site on the Earth at 30 degrees latitude, turning once a day.
    Ground,
}

impl Site {
    /// State of the observer with respect to the Solar System Barycenter.
    fn state(self, time: Time<TDB>) -> State<Equatorial, SSB> {
        let earth = earth(time);
        let mut state = earth.clone();
        match self {
            Self::L1 => {
                let sun = sun(time);
                let toward = (sun.pos - earth.pos).normalize();
                state.pos = earth.pos + toward * L1;
                state.vel = sun.vel + (earth.vel - sun.vel) * (1.0 - L1);
            }
            Self::Ground => {
                let turn = TAU * (time - reference()).elapsed;
                let (sin, cos) = turn.sin_cos();
                let (sin_lat, cos_lat) = 30.0_f64.to_radians().sin_cos();
                let offset = Vector::new([cos_lat * cos, cos_lat * sin, sin_lat]);
                let motion = Vector::new([-cos_lat * sin, cos_lat * cos, 0.0]);
                state.pos = earth.pos + offset * EARTH_RADIUS;
                state.vel = earth.vel + motion * (EARTH_RADIUS * TAU);
            }
        }
        state
    }

    /// Range of solar elongation the site observes, in degrees.
    fn band(self) -> (f64, f64) {
        match self {
            Self::L1 => (45.0, 120.0),
            Self::Ground => (90.0, 180.0),
        }
    }
}

/// A survey of the whole sky within each site's band of solar elongation.
#[derive(Debug)]
struct Survey {
    /// The observations, labeled.
    observations: Vec<AstrometricObservation>,

    /// Index of the object of each observation.
    labels: Vec<usize>,

    /// Site of each observation.
    sites: Vec<Site>,

    /// Number of objects.
    objects: usize,
}

/// Exposures from `site` on days `first` to `last` from the reference time,
/// in visits starting `starts` days into each day, each of `per_visit`
/// exposures `spacing` days apart.
fn visits(
    site: Site,
    (first, last): (i32, i32),
    starts: &[f64],
    per_visit: usize,
    spacing: f64,
) -> Vec<(Site, f64)> {
    let mut out = Vec::new();
    for day in first..=last {
        for &start in starts {
            for k in 0..per_visit {
                out.push((site, f64::from(day) + start + k as f64 * spacing));
            }
        }
    }
    out
}

impl Survey {
    /// `counts` objects of each population, drawn over the whole sky, seen in
    /// the exposures, each a site and a time in days from the reference
    /// time. Each object in the band of the site is detected with
    /// probability 0.9, with 0.1 arcseconds of noise.
    fn new(counts: &[(Population, usize)], exposures: &[(Site, f64)], seed: u64) -> Self {
        let mut rng = Rng(seed);
        let objects: Vec<SunState> = counts
            .iter()
            .flat_map(|&(population, count)| vec![population; count])
            .map(|population| population.draw(&mut rng, reference()))
            .collect();
        Self::of(&objects, exposures, (0.9, 0.1 * ARCSEC), &mut rng)
    }

    /// These objects, at the reference time, seen in the exposures as by
    /// [`Survey::new`], each detected with probability `keep` and with
    /// `noise` radians of noise.
    fn of(
        objects: &[SunState],
        exposures: &[(Site, f64)],
        (keep, noise): (f64, f64),
        rng: &mut Rng,
    ) -> Self {
        // The test kernels are loaded before the ephemeris is held, and it
        // is released before the observers, which take it again, are placed.
        let _ = earth(reference());
        let eph = SpiceEphemeris::loaded().unwrap();
        let moved: Vec<Vec<SunState>> = objects
            .par_iter()
            .map(|object| {
                let start = eph.spk().try_to_ssb(object.clone()).unwrap();
                exposures
                    .iter()
                    .map(|&(_, days)| {
                        let moved = start
                            .clone()
                            .propagate_with(&NBody::new(&eph, false), reference() + days)
                            .unwrap();
                        eph.spk().try_to_sun(moved).unwrap()
                    })
                    .collect()
            })
            .collect();
        drop(eph);
        let mut observations = Vec::new();
        let mut labels = Vec::new();
        let mut sites = Vec::new();
        for (k, &(site, days)) in exposures.iter().enumerate() {
            let time = reference() + days;
            let observer = site.state(time);
            let at = observer.pos - sun(time).pos;
            let (low, high) = site.band();
            for (label, states) in moved.iter().enumerate() {
                let toward = (emitted(&states[k], &at, time).pos - at).normalize();
                let elong = toward.angle(&(-at)).to_degrees();
                if elong < low || elong > high || rng.uniform() > keep {
                    continue;
                }
                let mut sight = crate::Sight {
                    time,
                    observer: at,
                    direction: toward,
                    sigma: 0.0,
                };
                add_noise(&mut sight, noise, rng);
                observations.push(optical(observer.clone(), sight.direction, sight.sigma));
                labels.push(label);
                sites.push(site);
            }
        }
        Self {
            observations,
            labels,
            sites,
            objects: objects.len(),
        }
    }

    /// The observations of only these sites, with their labels.
    fn only(&self, sites: &[Site]) -> (Vec<AstrometricObservation>, Vec<usize>) {
        let kept = (0..self.observations.len()).filter(|&k| sites.contains(&self.sites[k]));
        kept.map(|k| (self.observations[k].clone(), self.labels[k]))
            .unzip()
    }
}

#[test]
fn close_object_seen_from_two_places_merges() {
    // An object about a tenth of an AU from the Earth, seen on six nights
    // from the L1-like point and from the ground in the same quarter of a
    // day, 0.01 AU apart, which puts several degrees of parallax between the
    // two. Confirmed in two pieces, its first three nights and its last three,
    // it is merged whole: each observer has its own bins in the index of the
    // merge, so the parallax between them does not widen the reach.
    let earth = crate::confirm::sun_centered(&earth(reference())).unwrap();
    let out = earth.pos.normalize();
    let ahead = earth.vel.normalize();
    let toward = ahead * 10.0_f64.to_radians().cos() + out * 10.0_f64.to_radians().sin();
    let north = out.cross(&ahead).normalize();
    let object = SunState::new(
        kete_core::desigs::Desig::Empty,
        reference(),
        earth.pos + toward * 0.1,
        earth.vel * 0.97 + north * 0.002,
        kete_core::frames::SunCenter,
    );
    let mut exposures = visits(Site::L1, (-3, 2), &[0.0], 2, 0.02);
    exposures.extend(visits(Site::Ground, (-3, 2), &[0.05], 2, 0.02));
    let survey = Survey::of(
        std::slice::from_ref(&object),
        &exposures,
        (1.0, ARCSEC),
        &mut Rng(5),
    );
    assert_eq!(survey.observations.len(), exposures.len());
    let night = |i: usize| (survey.observations[i].epoch().jd() - reference().jd()).floor();
    let piece = |early: bool| {
        let sights: Vec<usize> = (0..survey.observations.len())
            .filter(|&i| (night(i) < 0.0) == early)
            .collect();
        let candidate = crate::Candidate {
            sights,
            state: object.clone(),
        };
        let mut found =
            crate::confirm(&survey.observations, &[candidate], &ConfirmRules::default()).unwrap();
        assert_eq!(found.len(), 1);
        found.remove(0)
    };
    let pieces = vec![piece(true), piece(false)];
    let settings = SearchSettings::default();
    let merged = merge_pieces(&survey.observations, pieces, &settings.confirm).unwrap();
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].sights.len(), survey.observations.len());
}

#[test]
#[ignore = "prints a table; minutes in release"]
fn space_and_ground_survey() {
    // A survey from the L1-like point of every object at 45 to 120 degrees
    // of solar elongation, in two visits a day of two exposures each, over
    // two weeks; and the same with a site on the Earth observing the night
    // sky in a visit of three exposures each night. Main belt and near-Earth
    // objects. Each is searched at once and tile by tile with the default
    // nside, the second with the parallax margin of the two observers. The
    // table gives completeness and purity of both, the objects the merge
    // dropped, and whether the two give the same objects.
    let start = Instant::now();
    let mut exposures = visits(Site::L1, (-7, 6), &[0.0, 0.5], 2, 0.02);
    // The ground site's local midnight, with the site turning from the
    // reference time.
    exposures.extend(visits(Site::Ground, (-7, 6), &[0.25], 3, 0.02));
    let belt = 800;
    let survey = Survey::new(
        &[(Population::MainBelt, belt), (Population::NearEarth, 200)],
        &exposures,
        17,
    );
    println!(
        "\nsurvey: {} observations of {} objects in {:.0} s",
        survey.observations.len(),
        survey.objects,
        start.elapsed().as_secs_f64()
    );
    let min_depth = 0.2;
    let settings = SearchSettings {
        min_gap: 0.4,
        passes: with_rates(&[0.6, 2.0]),
        min_depth,
        max_depth: 4.5,
        confirm: ConfirmRules {
            max_gap: 1.5,
            ..ConfirmRules::default()
        },
        ..SearchSettings::default()
    };
    println!(
        "{:<22} {:>7} {:>6} {:>8} {:>6} {:>6} {:>8} {:>6} {:>8} {:>8} {:>6}",
        "case",
        "sights",
        "tiles",
        "found",
        "whole",
        "false",
        "NEOs",
        "dupes",
        "search s",
        "tiled s",
        "equal"
    );
    for (name, sites, separation) in [
        ("L1", vec![Site::L1], 0.0),
        (
            "L1 and ground",
            vec![Site::L1, Site::Ground],
            L1 + EARTH_RADIUS,
        ),
    ] {
        let (observations, labels) = survey.only(&sites);
        let margin = parallax_margin(separation, min_depth).unwrap();
        let at = Instant::now();
        let all = search(&observations, None, &settings).unwrap();
        let untiled = at.elapsed().as_secs_f64();
        println!("  {name}: searched at once in {untiled:.0} s");
        let at = Instant::now();
        let (merged, before, count) = by_tile(&observations, &settings, margin, None);
        let tiled = at.elapsed().as_secs_f64();
        let score = |found: &[Confirmed]| {
            Score::new(
                found.iter().map(|c| c.sights.as_slice()),
                &labels,
                survey.objects,
                settings.confirm.min_sights,
            )
        };
        // The near-Earth objects alone, with the main belt as clutter.
        let near: Vec<usize> = labels
            .iter()
            .map(|&l| if l < belt { CLUTTER } else { l })
            .collect();
        let near_found = |found: &[Confirmed]| {
            let s = Score::new(
                found.iter().map(|c| c.sights.as_slice()),
                &near,
                survey.objects,
                settings.confirm.min_sights,
            );
            format!("{}/{}", s.found, s.findable)
        };
        let equal = sorted(&all) == sorted(&merged);
        for (kind, found) in [("search", &all), ("tiles", &merged)] {
            let s = score(found);
            println!(
                "{:<22} {:>7} {:>6} {:>4}/{:<4} {:>5} {:>6} {:>8} {:>6} {:>8.1} {:>8.1} {:>6}",
                format!("{name}, {kind}"),
                observations.len(),
                count,
                s.found,
                s.findable,
                s.whole,
                s.false_groups,
                near_found(found),
                before - merged.len(),
                untiled,
                tiled,
                equal
            );
        }
    }
}
