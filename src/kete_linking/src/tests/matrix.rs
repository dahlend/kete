// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The scenario matrix and the cost of the solves, the regression benchmark
//! for changes to speed and quality.
//!
//! Every test here is ignored, since together they take many minutes. Each
//! prints one table and asserts nothing. Run them in release, one at a time:
//!
//! `cargo test -p kete_linking --release -- --ignored --nocapture
//! --test-threads 1 matrix_cadence`
//!
//! The columns of most tables are described at
//! [`super::harness::Run::print`] and [`super::harness::Score`]. Unless a
//! row says otherwise, objects are seen by the ground site with 0.1
//! arcseconds of noise, the grid has the default steps and is aimed at the
//! objects, runs are up to the run span of `search` and time groups up to a
//! tenth of a day, or the largest gap if shorter, and the settings of `link`
//! are the defaults. Searches for close objects take runs and time groups
//! of a hundredth of a day. The tables of confirmation are seen from the
//! Earth of the test kernels, by objects moving under the n-body model.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use kete_core::frames::Equatorial;
use kete_fitting::{AstrometricObservation, fit_orbit};

use crate::from_ranges::orbit_through;
use crate::{
    Candidate, ConfirmRules, DEPTH_STEP, Hypothesis, LinkSettings, RATE_STEP, Sight, confirm,
    form_units, hypothesis_grid,
};
use kete_spice::prelude::SpiceEphemeris;

use super::harness::{
    CLOSE_SPAN, CLUTTER, MAX_RMS, SPAN, Search, Sky, majority, nightly, print_header, run,
};
use super::synthetic::{
    ARCSEC, CASES, GAPS, HOUR, Observer, Population, Rng, SECOND, SunState, add_noise,
    nearest_axis, observe, sight, true_hypothesis,
};
use super::{orbit_from_hypothesis, reference};

/// Search the scene of a sky and print its row.
fn row(name: &str, sky: &Sky, search: &Search) {
    run(&sky.scene(), search).print(name);
}

/// Paired visits 40 minutes apart on five nights.
fn paired() -> Vec<f64> {
    nightly(-2, 2, 2, 0.028)
}

/// Paired visits 40 minutes apart on the given nights.
fn paired_nights(nights: &[i32]) -> Vec<f64> {
    nights
        .iter()
        .flat_map(|&night| [0.0, 0.028].map(|t| f64::from(night) + t))
        .collect()
}

/// One sight a night for fifteen nights.
fn daily() -> Vec<f64> {
    nightly(-7, 7, 1, 0.0)
}

/// Nine sights evenly over `window` days centered on the reference time.
fn nine_over(window: f64) -> Vec<f64> {
    (0..9)
        .map(|k| window * (f64::from(k) / 8.0 - 0.5))
        .collect()
}

/// Pairs within a night only, for main belt rates.
fn within_night() -> Search {
    Search::new((0.01, 0.1), 0.5)
}

/// Pairs on neighboring nights, for main belt rates.
fn next_night() -> Search {
    Search::new((0.5, 1.1), 0.5)
}

/// `times` for each observer, offset by `offset` days per observer.
fn each(observers: usize, times: &[f64], offset: f64) -> Vec<(usize, f64)> {
    (0..observers)
        .flat_map(|who| times.iter().map(move |&t| (who, t + offset * who as f64)))
        .collect()
}

/// A sky of `objects` of a population at `distances` from the observer,
/// with the field at `elong` degrees from the Sun.
fn sky(
    population: Population,
    objects: usize,
    distances: Option<(f64, f64)>,
    elong: f64,
    times: &[f64],
) -> Sky {
    Sky {
        elong,
        distances,
        ..Sky::new(population, objects, times)
    }
}

/// `objects` near-Earth objects 0.05 to 0.5 AU away, 120 degrees from the
/// Sun.
fn near_earth(objects: usize, times: &[f64]) -> Sky {
    sky(
        Population::NearEarth,
        objects,
        Some((0.05, 0.5)),
        120.0,
        times,
    )
}

/// `objects` close objects at `distances`, 120 degrees from the Sun, seen
/// nine times over `window` days.
fn close(objects: usize, distances: (f64, f64), window: f64) -> Sky {
    sky(
        Population::Close,
        objects,
        Some(distances),
        120.0,
        &nine_over(window),
    )
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_observers() {
    print_header("Main belt, 30 objects, by observer");
    let epoch = reference();
    // Name, observers, and elongation and latitude of the field.
    let observers: Vec<(&str, Vec<Observer>, f64, f64)> = vec![
        ("planet center", vec![Observer::planet(epoch)], 180.0, 0.0),
        ("ground", vec![Observer::ground(epoch)], 180.0, 0.0),
        ("low orbit", vec![Observer::low_orbit(epoch)], 180.0, 0.0),
        ("L1-like", vec![Observer::sunward(epoch)], 180.0, 0.0),
        ("0.72 AU", vec![Observer::inner(epoch)], 150.0, 0.0),
        ("5.2 AU, elong 30", vec![Observer::outer(epoch)], 30.0, 0.0),
        (
            "polar 1.5 AU, elong 60 lat -40",
            vec![Observer::polar(epoch)],
            60.0,
            -40.0,
        ),
        (
            "two ground sites",
            vec![
                Observer::ground(epoch),
                Observer::ground_at("ground 120", epoch, 120.0),
            ],
            180.0,
            0.0,
        ),
        (
            "ground + low orbit",
            vec![Observer::ground(epoch), Observer::low_orbit(epoch)],
            180.0,
            0.0,
        ),
        (
            "ground + trailing 0.3 AU",
            vec![Observer::ground(epoch), Observer::trailing(epoch)],
            180.0,
            0.0,
        ),
    ];
    for (name, observers, elong, lat) in observers {
        for (cadence, times, search) in [
            ("paired", paired(), within_night()),
            ("daily", daily(), next_night()),
        ] {
            let mut sky = Sky::new(Population::MainBelt, 30, &times);
            sky.exposures = each(observers.len(), &times, 0.1 * HOUR);
            sky.observers = observers.iter().map(Observer::clone).collect();
            sky.elong = elong;
            sky.lat = lat;
            row(&format!("{name}, {cadence}"), &sky, &search);
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_geometry() {
    print_header(
        "By solar elongation and ecliptic latitude, paired visits, 30 objects, rate limit 1 deg/day for the main belt",
    );
    for (population, distances, rate) in [
        (Population::MainBelt, None, 1.0),
        (Population::NearEarth, Some((0.05, 0.5)), 20.0),
        (Population::Tilted, None, 3.0),
    ] {
        let fields = [
            (180.0_f64, 0.0_f64),
            (120.0, 0.0),
            (90.0, 0.0),
            (60.0, 0.0),
            (45.0, 0.0),
            (30.0, 0.0),
            (20.0, 0.0),
            (120.0, 30.0),
            (90.0, 60.0),
        ];
        for (elong, lat) in fields {
            // The main belt does not reach that far from the ecliptic.
            let lat = if population == Population::MainBelt {
                lat.min(20.0)
            } else {
                lat
            };
            let mut sky = sky(population, 30, distances, elong, &paired());
            sky.lat = lat;
            row(
                &format!("{population:?} elong {elong} lat {lat}"),
                &sky,
                &Search::new((0.01, 0.1), rate),
            );
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_cadence() {
    print_header("Main belt, 30 objects, by cadence");
    let seconds = nightly(-1, 1, 4, 10.0 * SECOND);
    let runs = nightly(-1, 1, 10, 3.0 * 60.0 * SECOND);
    let mut rng = Rng(5);
    let mut irregular: Vec<f64> = (0..15).map(|_| rng.between(-5.0, 5.0)).collect();
    irregular.sort_by(f64::total_cmp);
    let cases: Vec<(&str, Vec<f64>, Search)> = vec![
        (
            "4 x 10 s, 3 nights",
            seconds.clone(),
            Search::new((5.0 * SECOND, 0.1), 0.5),
        ),
        (
            "4 x 3 min, 3 nights",
            nightly(-1, 1, 4, 3.0 * 60.0 * SECOND),
            Search::new((60.0 * SECOND, 0.1), 0.5),
        ),
        ("paired 40 min, 5 nights", paired(), within_night()),
        (
            "paired 60 min, 5 nights",
            nightly(-2, 2, 2, HOUR),
            within_night(),
        ),
        (
            "paired 40 min, 5 nights, all pairs",
            paired(),
            Search::new((0.01, 1.1), 0.5),
        ),
        (
            "10 x 3 min, 3 nights, all pairs",
            runs.clone(),
            Search::new((60.0 * SECOND, 0.1), 0.5),
        ),
        (
            "10 x 3 min, 3 nights, pairs over 20 min",
            runs,
            Search::new((20.0 * 60.0 * SECOND, 0.1), 0.5),
        ),
        ("1 a night, 15 nights", daily(), next_night()),
        (
            "1 every 2 nights, 15 nights",
            (-7..=7).step_by(2).map(f64::from).collect(),
            Search::new((1.5, 2.1), 0.5),
        ),
        (
            "15 at random over 10 days",
            irregular,
            Search::new((0.0, 2.0), 0.5),
        ),
        (
            "paired, nights -6..-4 and 4..6",
            paired_nights(&[-6, -5, -4, 4, 5, 6]),
            within_night(),
        ),
        (
            "1 a night, nights -6..-4 and 4..6",
            [-6, -5, -4, 4, 5, 6].map(f64::from).to_vec(),
            Search::new((0.5, 8.1), 0.5),
        ),
        (
            "10 over 2 hours, one night",
            nightly(0, 0, 10, 12.0 * 60.0 * SECOND),
            Search::new((0.0, 0.1), 0.5),
        ),
        (
            "paired, two nights",
            nightly(0, 1, 2, 0.028),
            within_night(),
        ),
    ];
    for (name, times, search) in cases {
        row(name, &Sky::new(Population::MainBelt, 30, &times), &search);
    }
    for noise in [0.5, 1.0] {
        let mut sky = Sky::new(Population::MainBelt, 30, &seconds);
        sky.noise = noise * ARCSEC;
        let search = Search::new((5.0 * SECOND, 0.1), 0.5);
        row(
            &format!("4 x 10 s, 3 nights, {noise} arcsec"),
            &sky,
            &search,
        );
    }

    // Two sites 120 degrees apart, 30 seconds apart in time. A rate of six
    // degrees a day covers their parallax in 30 seconds.
    for (name, search) in [
        ("two sites 30 s apart, no cross pairs", within_night()),
        (
            "two sites 30 s apart, cross pairs",
            Search::new((10.0 * SECOND, 0.1), 6.0),
        ),
    ] {
        let mut sky = Sky::new(Population::MainBelt, 30, &[]);
        sky.observers = vec![
            Observer::ground(reference()),
            Observer::ground_at("ground 120", reference(), 120.0),
        ];
        sky.exposures = each(2, &nightly(-1, 1, 2, 0.028), 30.0 * SECOND);
        row(name, &sky, &search);
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_populations() {
    use Population::{Centaur, Interior, MainBelt, Tilted, Tno, Trojan};
    print_header("By population, 30 objects, 15 for close ones");
    let short = nightly(-1, 1, 2, 0.028);
    row(
        "main belt, paired",
        &Sky::new(MainBelt, 30, &paired()),
        &within_night(),
    );
    row(
        "main belt, daily",
        &Sky::new(MainBelt, 30, &daily()),
        &next_night(),
    );
    let search = Search::new((0.01, 0.1), 20.0);
    row(
        "near-Earth 0.05-0.5, paired 3 nights",
        &near_earth(30, &short),
        &search,
    );
    row(
        "near-Earth 0.05-0.5, paired 3 nights, steps x1/8",
        &near_earth(30, &short),
        &search.coarse(0.125),
    );
    // Close objects pair with each other across the whole field, so there
    // are fewer of them.
    row(
        "close 0.02-0.05, 9 over 8 h, steps x1/2",
        &close(15, (0.02, 0.05), 8.0 * HOUR),
        &Search::new((0.0, 0.1), 100.0).coarse(0.5).span(CLOSE_SPAN),
    );
    row(
        "close 0.003-0.01, 9 over 2 h, steps x1.25",
        &close(15, (0.003, 0.01), 2.0 * HOUR),
        &Search::new((0.0, 0.1), 400.0).coarse(1.25).span(CLOSE_SPAN),
    );
    let interior = sky(Interior, 30, None, 45.0, &short);
    let search = Search::new((0.01, 0.1), 5.0);
    row("interior, elong 45, paired 3 nights", &interior, &search);
    row(
        "interior, elong 45, paired 3 nights, steps x1/8",
        &interior,
        &search.coarse(0.125),
    );
    row(
        "trojan, paired",
        &Sky::new(Trojan, 30, &paired()),
        &within_night(),
    );
    row(
        "trojan, daily",
        &Sky::new(Trojan, 30, &daily()),
        &next_night(),
    );
    row(
        "centaur, daily",
        &Sky::new(Centaur, 30, &daily()),
        &next_night(),
    );
    row(
        "TNO, paired",
        &Sky::new(Tno, 30, &paired()),
        &within_night(),
    );
    row("TNO, daily", &Sky::new(Tno, 30, &daily()), &next_night());
    row(
        "high incl and retrograde, paired",
        &sky(Tilted, 30, None, 150.0, &paired()),
        &Search::new((0.01, 0.1), 3.0),
    );
    row(
        "high incl and retrograde, daily",
        &sky(Tilted, 30, None, 150.0, &daily()),
        &Search::new((0.5, 1.1), 3.0),
    );
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_realism() {
    print_header(
        "Main belt, paired visits within a night: noise, false detections, and realistic density",
    );
    let base = || Sky::new(Population::MainBelt, 30, &paired());
    for noise in [0.05, 0.3, 1.0] {
        let mut sky = base();
        sky.noise = noise * ARCSEC;
        row(&format!("noise {noise} arcsec"), &sky, &within_night());
    }
    for clutter in [30, 150, 600] {
        let mut sky = base();
        sky.clutter = clutter;
        row(
            &format!("{clutter} clutter per exposure"),
            &sky,
            &within_night(),
        );
    }
    let mut sky = base();
    sky.stationary = 50;
    row("50 fixed sources", &sky, &within_night());
    for keep in [0.7, 0.5] {
        let mut sky = base();
        sky.keep = keep;
        row(&format!("{keep} of detections kept"), &sky, &within_night());
    }
    let mut sky = base();
    sky.keep = 0.7;
    sky.clutter = 60;
    sky.stationary = 20;
    sky.noise = 0.3 * ARCSEC;
    row(
        "0.7 kept, 60 clutter, 20 fixed, 0.3 arcsec",
        &sky,
        &within_night(),
    );

    // About 100 per square degree.
    for (count, clutter) in [(3000, 0), (3000, 3000)] {
        let mut sky = Sky::new(Population::MainBelt, count, &paired());
        sky.radius = 3.1;
        sky.clutter = clutter;
        let name = format!(
            "{count} in {:.0} deg2, {clutter} clutter, paired",
            sky.field().area()
        );
        row(&name, &sky, &within_night());
    }
    for count in [100, 300] {
        let mut sky = Sky::new(Population::MainBelt, count, &daily());
        sky.radius = 3.1;
        let name = format!("{count} in {:.0} deg2, daily", sky.field().area());
        row(&name, &sky, &next_night());
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_gap() {
    print_header("100 main belt objects in 13 deg2, one sight a visit");
    for (every, max_gap) in [(1, 1.1), (1, 2.1), (1, 4.1), (2, 2.1), (3, 3.1), (4, 4.1)] {
        let times: Vec<f64> = (-8..=8).step_by(every).map(f64::from).collect();
        let mut sky = Sky::new(Population::MainBelt, 100, &times);
        sky.radius = 2.0;
        row(
            &format!("every {every} nights, gaps up to {max_gap} d"),
            &sky,
            &Search::new((0.5, max_gap), 0.5),
        );
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_runs() {
    print_header("runs of ten sights over 27 minutes, 30 main belt objects");
    for noise in [0.1, 0.5, 1.0] {
        for nights in [2, 3] {
            let runs = nightly(1 - nights, 0, 10, 3.0 * 60.0 * SECOND);
            let ends = nightly(1 - nights, 0, 2, 27.0 * 60.0 * SECOND);
            for (name, times, min_gap) in [
                ("all pairs", &runs, 60.0 * SECOND),
                ("pairs over 20 min", &runs, 20.0 * 60.0 * SECOND),
                ("ends of runs only", &ends, 0.01),
            ] {
                let mut sky = Sky::new(Population::MainBelt, 30, times);
                sky.noise = noise * ARCSEC;
                row(
                    &format!("{noise} arcsec, {nights} nights, {name}"),
                    &sky,
                    &Search::new((min_gap, 0.1), 0.5),
                );
            }
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_close_objects() {
    print_header("close objects by distance and window, 9 sights, 10 objects");
    for (near, far) in [(0.003_f64, 0.006), (0.01, 0.02), (0.03, 0.05)] {
        for window in [1.0 * HOUR, 3.0 * HOUR, 8.0 * HOUR, 1.0, 2.0] {
            let rate = 0.02 / near;
            let coarse = 0.00375 / near;
            let search = Search::new((0.0, window / 3.5), rate.to_degrees())
                .coarse(coarse)
                .span(CLOSE_SPAN);
            let name = format!(
                "{near}-{far} AU, window {:.1} h, steps x{coarse}",
                window / HOUR
            );
            row(&name, &close(10, (near, far), window), &search);
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_grid_allowance() {
    use Population::{MainBelt, Tno, Trojan};
    print_header("Grid allowance, as a multiple of the default, by population and window");
    let month: Vec<f64> = (-14..=14).step_by(2).map(f64::from).collect();
    let cases = [
        (
            "main belt, daily 15 nights",
            Sky::new(MainBelt, 30, &daily()),
            next_night(),
        ),
        (
            "main belt, every 2 nights over 28",
            Sky::new(MainBelt, 30, &month),
            Search::new((1.5, 2.1), 0.5),
        ),
        (
            "near-Earth, paired 3 nights",
            near_earth(30, &nightly(-1, 1, 2, 0.028)),
            Search::new((0.01, 0.1), 20.0),
        ),
        (
            "trojan, daily 15 nights",
            Sky::new(Trojan, 30, &daily()),
            next_night(),
        ),
        (
            "TNO, daily 15 nights",
            Sky::new(Tno, 30, &daily()),
            next_night(),
        ),
    ];
    for (name, sky, search) in cases {
        let sky = Sky { clutter: 30, ..sky };
        for factor in [0.25, 0.5, 1.0, 2.0, 4.0] {
            let mut search = search.clone();
            search.settings.grid_allowance = factor * LinkSettings::default().grid_allowance;
            row(&format!("{name}, x{factor}"), &sky, &search);
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_supersets() {
    print_header("Candidates removed as subsets: found against found in a pure candidate");
    for (cadence, times, search) in [
        ("paired", paired(), within_night()),
        ("daily", daily(), next_night()),
    ] {
        for clutter in [0, 30, 90, 300] {
            let mut sky = Sky::new(Population::MainBelt, 30, &times);
            sky.clutter = clutter;
            row(&format!("{cadence}, {clutter} clutter"), &sky, &search);
        }
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_hash_cells() {
    print_header("Hash cells sized by the least certain pair: a few pairs seconds long");
    for (count, burst) in [(60, 0), (60, 6), (200, 0), (200, 20)] {
        let mut sky = Sky::new(Population::MainBelt, count, &daily());
        sky.radius = 2.0;
        let mut scene = sky.scene();
        // A second sight five seconds after the first, for `burst` objects.
        let mut rng = Rng(3);
        let extra: Vec<usize> = (0..scene.sights.len())
            .filter(|&i| scene.labels[i] < burst)
            .collect();
        for i in extra {
            let mut twin = scene.sights[i];
            twin.time += 5.0 * SECOND;
            twin.observer = sky.observers[0].state(twin.time).pos;
            add_noise(&mut twin, sky.noise, &mut rng);
            scene.sights.push(twin);
            scene.labels.push(scene.labels[i]);
        }
        let search = Search::new((1.0 * SECOND, 1.1), 0.5);
        run(&scene, &search).print(&format!("{count} objects daily, {burst} with 5 s pairs"));
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_grid_size() {
    println!("\nHypotheses in a grid about the planet at 1 AU, default steps, whole bound rates");
    println!("{:>8}{:>8}{:>12}", "near", "far", "hypotheses");
    let origin = Observer::planet(reference()).state(reference());
    for (near, far) in [(0.003, 0.05), (0.05, 0.5), (0.5, 5.0), (5.0, 50.0)] {
        let grid = hypothesis_grid(&origin, near, far, DEPTH_STEP, RATE_STEP).unwrap();
        println!("{near:>8}{far:>8}{:>12}", grid.len());
    }
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_grid_steps() {
    use Population::{Centaur, MainBelt, Tno, Trojan};
    print_header("Grid steps as a multiple of the defaults, by population and window");
    let short = nightly(-1, 1, 2, 0.028);
    let fine = [0.125, 0.25, 0.5, 1.0, 2.0];
    // Close objects are too costly at the finest steps, and pair with each
    // other across the field, so there are fewer of them.
    let cases = [
        (
            "main belt, daily 15 nights",
            Sky::new(MainBelt, 30, &daily()),
            next_night(),
            fine,
        ),
        (
            "main belt, paired 5 nights",
            Sky::new(MainBelt, 30, &paired()),
            within_night(),
            fine,
        ),
        (
            "near-Earth, paired 3 nights",
            near_earth(30, &short),
            Search::new((0.01, 0.1), 20.0),
            fine,
        ),
        (
            "close 0.02-0.05, 9 over 8 h",
            close(10, (0.02, 0.05), 8.0 * HOUR),
            Search::new((0.0, 0.1), 100.0).span(CLOSE_SPAN),
            [0.5, 1.0, 2.0, 4.0, 8.0],
        ),
        (
            "trojan, daily 15 nights",
            Sky::new(Trojan, 30, &daily()),
            next_night(),
            fine,
        ),
        (
            "TNO, daily 15 nights",
            Sky::new(Tno, 30, &daily()),
            next_night(),
            fine,
        ),
    ];
    for (name, sky, search, factors) in cases {
        for factor in factors {
            let search = search.clone().coarse(factor);
            row(&format!("{name}, both x{factor}"), &sky, &search);
        }
    }

    // The rate step alone, for distant objects.
    for population in [Trojan, Centaur, Tno] {
        for factor in [1.0, 0.25, 0.0625] {
            let mut search = next_night();
            search.steps.1 *= factor;
            row(
                &format!("{population:?}, daily 15 nights, rate x{factor}"),
                &Sky::new(population, 30, &daily()),
                &search,
            );
        }
    }

    // Near-Earth objects over every bound rate, not only theirs.
    let whole = Search {
        rates: Some(f64::INFINITY),
        ..Search::new((0.01, 0.1), 20.0).coarse(0.5)
    };
    row(
        "near-Earth, paired 3 nights, all rates, both x0.5",
        &near_earth(30, &short),
        &whole,
    );
}

/// Print the header of the table [`confirm_row`] writes rows of.
fn print_confirm_header(title: &str) {
    println!("\n{title}");
    println!("Lost: why objects found by link were not confirmed, as fit failed or a");
    println!("rule refused it / residual / too few kept / overlap.");
    println!(
        "{:<44} {:>7} {:>7} {:>6} {:>9} {:>5} {:>5} {:>6} {:>9} {:>5} {:>5} {:>6} {:>7} {:>9}",
        "scenario",
        "units",
        "hyps",
        "cands",
        "found",
        "pure%",
        "false",
        "conf",
        "found",
        "pure%",
        "false",
        "link s",
        "conf s",
        "lost"
    );
}

/// Why `confirm` turned down a candidate, as an index into the lost column
/// of [`confirm_row`]. The candidate is confirmed alone with no limit on the
/// residual and the fewest observations `confirm` allows, which gives its
/// fit unless the fit failed or a rule of `confirm` refused it.
fn why_lost(
    observations: &[AstrometricObservation],
    candidate: &Candidate,
    min_sights: usize,
    search: &Search,
) -> usize {
    let rules = ConfirmRules {
        min_sights: 4,
        max_rms: f64::MAX,
        min_sights_per_side: 2,
        group_span: search.group_span,
        max_gap: search.gaps.1,
    };
    let alone = confirm(observations, std::slice::from_ref(candidate), &rules).unwrap();
    match alone.first() {
        None => 0,
        Some(found) if found.fit.rms > MAX_RMS => 1,
        Some(found) if found.sights.len() < min_sights => 2,
        Some(_) => 3,
    }
}

/// Link and confirm the scene of a sky seen from the Earth, and print its
/// row. `min_sights` is the fewest sights of a candidate and of a confirmed
/// fit.
fn confirm_row(name: &str, sky: &Sky, search: &Search, min_sights: usize) {
    let scene = sky.scene();
    let mut search = search.clone();
    search.min_sights = min_sights.min(search.min_sights);
    search.confirm = Some(min_sights);
    let run = run(&scene, &search);
    let confirmed: HashSet<usize> = run
        .confirmed
        .iter()
        .filter_map(|c| majority(&c.sights, &scene.labels, min_sights))
        .map(|(label, _)| label)
        .collect();
    let mut lost = [0; 4];
    let mut seen = HashSet::new();
    for candidate in &run.candidates {
        let Some((label, _)) = majority(&candidate.sights, &scene.labels, min_sights) else {
            continue;
        };
        if !confirmed.contains(&label) && seen.insert(label) {
            lost[why_lost(&scene.observations, candidate, min_sights, &search)] += 1;
        }
    }
    let (l, c) = (&run.linked, &run.confirmed_score);
    println!(
        "{:<44} {:>7} {:>7} {:>6} {:>4}/{:<4} {:>5.0} {:>5} {:>6} {:>4}/{:<4} {:>5.0} {:>5} {:>6.2} {:>7.2} {:>9}",
        name,
        run.units,
        run.hypotheses,
        l.groups,
        l.found,
        l.findable,
        l.purity(),
        l.false_groups,
        c.groups,
        c.found,
        c.findable,
        c.purity(),
        c.false_groups,
        run.seconds[0] + run.seconds[1],
        run.seconds[2],
        lost.map(|n| n.to_string()).join("/")
    );
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_confirm() {
    use Population::{Interior, MainBelt, Tilted, Tno, Trojan};
    print_confirm_header("Linked and confirmed, 30 objects, 30 clutter per exposure");
    let earth = |sky: Sky| Sky {
        clutter: 30,
        ..sky.on_earth()
    };
    let short = nightly(-1, 1, 2, 0.028);
    let main_belt = |times: &[f64]| earth(Sky::new(MainBelt, 30, times));
    let four_days = [-2.0, -2.0 + 0.028, 2.0, 2.0 + 0.028];
    confirm_row(
        "main belt, paired 5 nights",
        &main_belt(&paired()),
        &within_night(),
        6,
    );
    confirm_row(
        "main belt, paired 5 nights, 4 sights",
        &main_belt(&paired()),
        &within_night(),
        4,
    );
    // Two nights are confirmed only within the largest gap of a pair.
    confirm_row(
        "main belt, 2 nights 4 days apart",
        &main_belt(&four_days),
        &Search::new((0.01, 4.1), 0.5).span(SPAN),
        4,
    );
    confirm_row(
        "main belt, paired 3 nights",
        &main_belt(&short),
        &within_night(),
        6,
    );
    confirm_row(
        "main belt, daily 15 nights",
        &main_belt(&daily()),
        &next_night(),
        6,
    );
    confirm_row(
        "near-Earth 0.05-0.5, paired 3 nights",
        &earth(near_earth(30, &short)),
        &Search::new((0.01, 0.1), 20.0),
        6,
    );
    confirm_row(
        "close 0.02-0.05, 9 over 8 h, steps x1/2",
        &earth(close(30, (0.02, 0.05), 8.0 * HOUR)),
        &Search::new((0.0, 0.1), 100.0).coarse(0.5).span(CLOSE_SPAN),
        6,
    );
    confirm_row(
        "close 0.003-0.01, 9 over 3 h, steps x1.25",
        &earth(close(30, (0.003, 0.01), 3.0 * HOUR)),
        &Search::new((0.0, 0.05), 400.0)
            .coarse(1.25)
            .span(CLOSE_SPAN),
        6,
    );
    confirm_row(
        "interior, elong 45, paired 3 nights",
        &earth(sky(Interior, 30, None, 45.0, &short)),
        &Search::new((0.01, 0.1), 5.0),
        6,
    );
    confirm_row(
        "trojan, paired 5 nights",
        &earth(Sky::new(Trojan, 30, &paired())),
        &within_night(),
        6,
    );
    confirm_row(
        "TNO, daily 15 nights",
        &earth(Sky::new(Tno, 30, &daily())),
        &next_night(),
        6,
    );
    confirm_row(
        "high incl and retrograde, paired 5 nights",
        &earth(sky(Tilted, 30, None, 150.0, &paired())),
        &Search::new((0.01, 0.1), 2.0),
        6,
    );
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn matrix_density() {
    print_confirm_header("main belt at about 100 per square degree, paired visits");
    for (count, clutter, noise) in [
        (1000, 0, 0.1),
        (1000, 1000, 0.1),
        (3000, 3000, 0.1),
        (3000, 3000, 0.5),
    ] {
        let sky = Sky {
            radius: if count > 1000 { 3.1 } else { 1.8 },
            noise: noise * ARCSEC,
            clutter,
            ..Sky::new(Population::MainBelt, count, &paired()).on_earth()
        };
        confirm_row(
            &format!("{count} objects, {clutter} clutter, {noise} arcsec"),
            &sky,
            &within_night(),
            6,
        );
    }
}

/// Range error of `state` at the reference time, along the line of sight
/// from `origin` to `truth`, as a fraction of the true range.
fn range_error(state: &SunState, truth: &SunState, origin: &SunState) -> f64 {
    let sight = truth.pos - origin.pos;
    ((state.pos - truth.pos).dot(&sight) / sight.norm_squared()).abs()
}

/// The 10th, 50th and 90th percentiles and the largest of `values`.
fn quantiles(mut values: Vec<f64>) -> String {
    if values.is_empty() {
        return "-".into();
    }
    values.sort_by(f64::total_cmp);
    let at = |percent: usize| values[(values.len() - 1) * percent / 100];
    format!("{:.3} {:.3} {:.3} {:.3}", at(10), at(50), at(90), at(100))
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn diagnose_confirm_losses() {
    // For each object with a pure candidate, the range error of the state the
    // candidate carries and of the orbit `confirm` found, at the reference
    // time. Pure candidates `confirm` turns down are refit from the true
    // state, the way `confirm` fits: `fit_orbit`, then the start scored as it
    // is if the fit fails.
    println!("\nRange error along the line of sight over the range, at the reference time, as");
    println!("percentiles 10/50/90/100, of the state each pure candidate carries and of the");
    println!("orbit confirm found. Pure candidates turned down are refit from the truth; from");
    println!("truth counts those which reach a residual of at most 2.");
    println!(
        "{:<36} {:>5} {:>5} {:>8} {:>5} {:>5} {:>27} {:>27}",
        "scenario",
        "pure",
        "conf",
        "not conv",
        "down",
        "truth",
        "candidate state",
        "confirmed orbit"
    );
    let cases: Vec<(&str, Sky)> = vec![
        (
            "main belt, paired 5 nights, 300",
            Sky {
                radius: 1.8,
                ..Sky::new(Population::MainBelt, 300, &paired()).on_earth()
            },
        ),
        (
            "main belt, paired 3 nights, 30",
            Sky::new(Population::MainBelt, 30, &nightly(-1, 1, 2, 0.028)).on_earth(),
        ),
        (
            "close 0.02-0.05, 9 over 8 h, 10",
            close(10, (0.02, 0.05), 8.0 * HOUR).on_earth(),
        ),
    ];
    let searches = [
        within_night(),
        within_night(),
        Search::new((0.0, 0.1), 100.0).coarse(0.5).span(CLOSE_SPAN),
    ];
    let eph = SpiceEphemeris::loaded().unwrap();
    for ((name, sky), search) in cases.into_iter().zip(searches) {
        let scene = sky.scene();
        let search = Search {
            confirm: Some(6),
            ..search
        };
        let run = run(&scene, &search);
        let mut orbits = vec![None; scene.objects.len()];
        let mut unconverged = 0;
        for found in &run.confirmed {
            if let Some((label, _)) = majority(&found.sights, &scene.labels, 6) {
                let state = found.fit.uncertain_state.state::<Equatorial>().unwrap();
                orbits[label] =
                    Some(SunState::try_from(eph.spk().try_to_sun(state).unwrap()).unwrap());
                unconverged += usize::from(!found.fit.converged);
            }
        }
        let mut tried = HashSet::new();
        let (mut before, mut after) = (Vec::new(), Vec::new());
        let (mut turned_down, mut from_truth) = (0, 0);
        for candidate in &run.candidates {
            let label = scene.labels[candidate.sights[0]];
            let pure = candidate.sights.iter().all(|&i| scene.labels[i] == label);
            if !pure || label == CLUTTER || !tried.insert(label) {
                continue;
            }
            let truth = &scene.objects[label];
            before.push(range_error(&candidate.state, truth, &scene.origin));
            if let Some(orbit) = &orbits[label] {
                after.push(range_error(orbit, truth, &scene.origin));
                continue;
            }
            turned_down += 1;
            let chosen: Vec<_> = candidate
                .sights
                .iter()
                .map(|&i| scene.observations[i].clone())
                .collect();
            let start = eph.spk().try_to_ssb(truth.clone()).unwrap();
            let fit = fit_orbit(&start, &chosen, false, None, 50, 9.0, 3, None)
                .or_else(|_| fit_orbit(&start, &chosen, false, None, 0, 9.0, 3, None));
            from_truth += usize::from(fit.is_ok_and(|fit| fit.rms <= MAX_RMS));
        }
        println!(
            "{:<36} {:>5} {:>5} {:>8} {:>5} {:>5} {:>27} {:>27}",
            name,
            tried.len(),
            after.len(),
            unconverged,
            turned_down,
            from_truth,
            quantiles(before),
            quantiles(after)
        );
    }
}

/// Mean nanoseconds per call of `solve` over `repeats` passes of `inputs`.
fn nanos_per_call<T>(inputs: &[T], repeats: usize, solve: impl Fn(&T)) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        for input in inputs {
            solve(black_box(input));
        }
    }
    start.elapsed().as_nanos() as f64 / (repeats * inputs.len()).max(1) as f64
}

#[test]
#[ignore = "benchmark, see the module documentation"]
fn hypothesis_cost() {
    // Most solves of a search fail, so failing must not cost much more than
    // succeeding.
    let epoch = reference();
    let planet = Observer::planet(epoch);
    let origin = planet.state(epoch);
    println!("\nNanoseconds per call, mean over the inputs, and how many of the inputs fail.");
    println!("{:<52}{:>10}{:>12}", "call", "ns", "failed");
    let print = |name: &str, nanos: f64, failed: usize, of: usize| {
        println!("{name:<52}{nanos:>10.0}{:>12}", format!("{failed}/{of}"));
    };

    // The orbit through two ranges, at the true ranges, of every object.
    for gap in GAPS {
        let inputs: Vec<_> = CASES
            .iter()
            .map(|case| {
                let object = case.object(&origin);
                let (a, range_a) = sight(&planet, &object, epoch - 0.5 * gap);
                let (b, range_b) = sight(&planet, &object, epoch + 0.5 * gap);
                (a, b, range_a, range_b)
            })
            .collect();
        let call = |(a, b, ra, rb): &(Sight, Sight, f64, f64)| orbit_through(a, b, *ra, *rb);
        let failed = inputs.iter().filter(|i| call(i).is_err()).count();
        let nanos = nanos_per_call(&inputs, 200, |i| {
            let _ = black_box(call(i));
        });
        print(
            &format!("orbit_through, gap {gap:.1e} d"),
            nanos,
            failed,
            inputs.len(),
        );
    }

    // Pairs an hour apart of every bound object past 0.15 AU, a given time
    // before the reference, under their true hypothesis and wrong ones.
    let pairs_at = |ahead: f64| -> Vec<(Sight, Sight, Hypothesis)> {
        CASES
            .iter()
            .filter(|case| case.population != Population::Hyperbolic && case.dist > 0.15)
            .map(|case| {
                let truth = case.object(&origin);
                let a = sight(&planet, &truth, epoch - ahead).0;
                let b = sight(&planet, &truth, epoch - (ahead - HOUR)).0;
                let hypothesis = true_hypothesis(&truth, &origin, nearest_axis(&a.direction));
                (a, b, hypothesis)
            })
            .collect()
    };
    let solve = |(a, b, h): &(Sight, Sight, Hypothesis)| orbit_from_hypothesis(a, b, h);
    let time = |name: &str, inputs: &[(Sight, Sight, Hypothesis)]| {
        let failed = inputs.iter().filter(|i| solve(i).is_err()).count();
        let nanos = nanos_per_call(inputs, 50, |i| {
            let _ = black_box(solve(i));
        });
        print(name, nanos, failed, inputs.len());
    };
    for ahead in [1.0, 7.0, 15.0] {
        time(
            &format!("orbit_from_hypothesis, true, {ahead} days ahead"),
            &pairs_at(ahead),
        );
    }
    let failures: [(&str, fn(&mut Hypothesis)); 5] = [
        ("unbound, rate 0.1 AU/day too high", |h| h.depth_rate += 0.1),
        ("behind the observer", |h| {
            h.origin.pos -= h.direction * (2.0 * h.depth);
        }),
        ("reference 400 days on", |h| h.origin.epoch += 400.0),
        ("direction reversed", |h| h.direction = -h.direction),
        ("depth 100 times too small", |h| h.depth *= 0.01),
    ];
    for (name, change) in failures {
        let mut inputs = pairs_at(15.0);
        for (_, _, hypothesis) in &mut inputs {
            change(hypothesis);
        }
        time(&format!("orbit_from_hypothesis, {name}, 15 days"), &inputs);
    }

    // Random hypotheses on pairs of random objects of every kind, most of
    // them pairs of two objects, split by whether the solve finds an orbit.
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
    let (sights, _) = observe(&observers, &objects, &times);
    let (found, lost): (Vec<_>, Vec<_>) = form_units(&sights, 0.0, 3.0, 0.2, 0.0)
        .unwrap()
        .into_iter()
        .map(|pair| {
            let (i, j) = (pair[0], pair[1]);
            let hypothesis = Hypothesis {
                origin: origin.clone(),
                direction: nearest_axis(&sights[i].direction),
                depth: 10.0_f64.powf(rng.between(-3.0, 1.7)),
                depth_rate: rng.between(-0.03, 0.03),
            };
            (sights[i], sights[j], hypothesis)
        })
        .partition(|input| solve(input).is_ok());
    time("orbit_from_hypothesis, random, orbit found", &found);
    time("orbit_from_hypothesis, random, no orbit", &lost);
}

#[test]
#[ignore = "matrix, see the module documentation"]
fn units_close_observers() {
    print_header(
        "Close objects in runs of 5 sights 2 minutes apart, from the ground and from low orbit, 10 objects",
    );
    let runs = |every: f64, window: f64| -> Vec<f64> {
        let count = (window / every).round() as i32 + 1;
        (0..count)
            .flat_map(|r| {
                (0..5).map(move |k| {
                    -0.5 * window + f64::from(r) * every + f64::from(k) * 120.0 * SECOND
                })
            })
            .collect()
    };
    for (who, observer) in [
        ("ground", Observer::ground(reference())),
        ("low orbit", Observer::low_orbit(reference())),
    ] {
        for (near, far, every, window) in [
            (0.003, 0.006, 0.5 * HOUR, 3.0 * HOUR),
            (0.01, 0.02, 0.5 * HOUR, 3.0 * HOUR),
            (0.01, 0.02, HOUR, 8.0 * HOUR),
            (0.03, 0.05, HOUR, 8.0 * HOUR),
        ] {
            let mut sky = close(10, (near, far), window);
            sky.observers = vec![observer.clone()];
            sky.exposures = runs(every, window).into_iter().map(|t| (0, t)).collect();
            let rate = 0.02 / near;
            let coarse = 0.00375 / near;
            let search = Search::new((0.0, window / 3.5), rate.to_degrees())
                .coarse(coarse)
                .span(CLOSE_SPAN);
            row(
                &format!(
                    "{who}, {near}-{far} AU, runs every {:.1} h over {:.0} h",
                    every / HOUR,
                    window / HOUR
                ),
                &sky,
                &search,
            );
        }
    }
}
