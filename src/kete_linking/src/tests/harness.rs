// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Labeled scenarios, linked end to end and scored.
//!
//! A [`Sky`] says what objects there are, who sees them and when. Its
//! [`Scene`] holds the sights, each labeled with the object it shows or as
//! [`CLUTTER`]. A [`Search`] says how the sights are linked, and [`run`]
//! forms the units, builds the grid, links, optionally confirms, and scores
//! the result against the labels with [`Score`].

use std::collections::HashMap;
use std::time::Instant;

use kete_core::Band;
use kete_core::ephemeris::Ephemeris;
use kete_core::frames::{Equatorial, SSB, Vector};
use kete_core::prelude::State;
use kete_core::propagation::NBody;
use kete_core::time::{TDB, Time};
use kete_fitting::AstrometricObservation;

use crate::confirm::sun_centered;
use crate::hypothesis::axes;
use crate::{
    Candidate, ConfirmRules, Confirmed, DEPTH_STEP, LinkSettings, MIN_ALIGNMENT, RATE_STEP,
    SearchSettings, Sight, confirm, form_units, hypothesis_grid, link, sight_from_observation,
};
use kete_spice::prelude::SpiceEphemeris;
use kete_spice::test_data::ensure_test_spk;
use rayon::prelude::*;

use super::reference;
use super::synthetic::{
    ARCSEC, EXACT, Field, Observer, Population, Rng, SunState, add_noise, emitted,
};

/// Label of a sight which shows no object.
pub(crate) const CLUTTER: usize = usize::MAX;

/// Largest reduced residual of a confirmed fit, in units of the observation
/// uncertainties.
pub(crate) const MAX_RMS: f64 = 2.0;

/// State of the center of the Earth with respect to the Solar System
/// Barycenter, from the test kernels.
pub(crate) fn earth(time: Time<TDB>) -> State<Equatorial, SSB> {
    ensure_test_spk();
    let state = SpiceEphemeris::loaded()
        .unwrap()
        .try_get_state_with_center(399, time, 0)
        .unwrap();
    State::try_from(state).unwrap()
}

/// An optical observation from `observer` in `direction`, with an
/// uncertainty of `sigma` radians on each axis of the sky.
pub(crate) fn optical(
    observer: State<Equatorial, SSB>,
    direction: Vector<Equatorial>,
    sigma: f64,
) -> AstrometricObservation {
    let (ra, dec) = direction.to_ra_dec();
    AstrometricObservation::Optical {
        observer,
        ra,
        dec,
        sigma_ra: sigma / dec.cos(),
        sigma_dec: sigma,
        sigma_corr: 0.0,
        time_sigma: 0.0,
        is_occultation: false,
        band: Band::Unknown([0; 8]),
        mag: f64::NAN,
    }
}

/// Times in days of `per_night` sights `spacing` days apart on each night,
/// for nights `first` to `last` from the reference time.
pub(crate) fn nightly(first: i32, last: i32, per_night: usize, spacing: f64) -> Vec<f64> {
    (first..=last)
        .flat_map(|night| (0..per_night).map(move |k| f64::from(night) + k as f64 * spacing))
        .collect()
}

/// The result of `work` run on a thread pool of `threads` threads.
pub(crate) fn on_threads<T: Send>(threads: usize, work: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
        .install(work)
}

/// The sights of confirmed objects, sorted, for comparison.
pub(crate) fn sorted(found: &[Confirmed]) -> Vec<Vec<usize>> {
    let mut sets: Vec<Vec<usize>> = found.iter().map(|c| c.sights.clone()).collect();
    sets.sort();
    sets
}

/// The sights of confirmed objects, each with the bits of the reduced
/// residual of its fit, sorted, for comparison.
pub(crate) fn sorted_with_fits(found: &[Confirmed]) -> Vec<(Vec<usize>, u64)> {
    let mut sets: Vec<(Vec<usize>, u64)> = found
        .iter()
        .map(|c| (c.sights.clone(), c.fit.rms.to_bits()))
        .collect();
    sets.sort();
    sets
}

/// What there is to see, and who sees it when.
#[derive(Debug)]
pub(crate) struct Sky {
    /// The observers. Depths are measured from the first, and the field is
    /// placed as it sees the sky. Unused when seen from the Earth.
    pub(crate) observers: Vec<Observer>,

    /// Each exposure, as the index of its observer and its time in days from
    /// the reference time.
    pub(crate) exposures: Vec<(usize, f64)>,

    /// Seen from the center of the Earth of the test kernels, by objects
    /// moving under the n-body model of `kete_core`, which is what `confirm`
    /// fits with. The scene then has observations to confirm. Otherwise the
    /// observers are analytic and the objects move on two-body orbits.
    pub(crate) from_earth: bool,

    /// Kind of object.
    pub(crate) population: Population,

    /// Number of objects.
    pub(crate) objects: usize,

    /// Elongation of the field center from the Sun, in degrees.
    pub(crate) elong: f64,

    /// Ecliptic latitude of the field center, in degrees.
    pub(crate) lat: f64,

    /// Radius of the field the objects start in, in degrees.
    pub(crate) radius: f64,

    /// Range of distances of the objects from the first observer at the
    /// reference time, see [`Population::in_field`].
    pub(crate) distances: Option<(f64, f64)>,

    /// Astrometric noise on each axis of the sky, in radians.
    pub(crate) noise: f64,

    /// False detections in each exposure, uniform over the patch of sky the
    /// objects cover in it.
    pub(crate) clutter: usize,

    /// False sources fixed on the sky, seen in every exposure.
    pub(crate) stationary: usize,

    /// Chance that an object is detected in an exposure.
    pub(crate) keep: f64,

    /// Seed of the random numbers.
    pub(crate) seed: u64,
}

impl Sky {
    /// `objects` of a population at opposition in a field one degree in
    /// radius, seen by the ground site at `times` days from the reference
    /// time, with 0.1 arcseconds of noise and nothing false.
    pub(crate) fn new(population: Population, objects: usize, times: &[f64]) -> Self {
        Self {
            observers: vec![Observer::ground(reference())],
            exposures: times.iter().map(|&time| (0, time)).collect(),
            from_earth: false,
            population,
            objects,
            elong: 180.0,
            lat: 0.0,
            radius: 1.0,
            distances: None,
            noise: 0.1 * ARCSEC,
            clutter: 0,
            stationary: 0,
            keep: 1.0,
            seed: 7,
        }
    }

    /// This sky seen from the Earth of the test kernels.
    pub(crate) fn on_earth(self) -> Self {
        Self {
            observers: Vec::new(),
            from_earth: true,
            ..self
        }
    }

    /// Position with respect to the Sun of observer `who` at `time`.
    fn place(&self, who: usize, time: Time<TDB>) -> Vector<Equatorial> {
        if self.from_earth {
            sun_centered(&earth(time)).unwrap().pos
        } else {
            self.observers[who].state(time).pos
        }
    }

    /// The first observer at the reference time, from which depths are
    /// measured.
    pub(crate) fn origin(&self) -> SunState {
        if self.from_earth {
            sun_centered(&earth(reference())).unwrap()
        } else {
            self.observers[0].state(reference())
        }
    }

    /// The field the objects start in.
    pub(crate) fn field(&self) -> Field {
        Field::new(&self.origin().pos, self.elong, self.lat, self.radius)
    }

    /// The objects moved under the n-body model to each exposure, about the
    /// Sun, indexed by object and then exposure.
    fn n_body(&self, objects: &[SunState]) -> Vec<Vec<SunState>> {
        let eph = SpiceEphemeris::loaded().unwrap();
        objects
            .par_iter()
            .map(|object| {
                let start = eph.spk().try_to_ssb(object.clone()).unwrap();
                self.exposures
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
            .collect()
    }

    /// The labeled sights.
    pub(crate) fn scene(&self) -> Scene {
        let epoch = reference();
        let origin = self.origin();
        let field = self.field();
        let mut rng = Rng(self.seed);
        let objects: Vec<SunState> = (0..self.objects)
            .map(|_| {
                self.population
                    .in_field(&mut rng, &origin, &field, self.distances)
            })
            .collect();
        let fixed: Vec<Vector<Equatorial>> = (0..self.stationary)
            .map(|_| field.direction(&mut rng))
            .collect();
        let moved = if self.from_earth {
            self.n_body(&objects)
        } else {
            Vec::new()
        };

        let mut sights = Vec::new();
        let mut labels = Vec::new();
        for (k, &(who, days)) in self.exposures.iter().enumerate() {
            let time = epoch + days;
            let observer = self.place(who, time);
            let seen = |direction| Sight {
                time,
                observer,
                direction,
                sigma: EXACT,
            };
            let start = sights.len();
            for (label, object) in objects.iter().enumerate() {
                if rng.uniform() <= self.keep {
                    let near = if self.from_earth {
                        &moved[label][k]
                    } else {
                        object
                    };
                    sights.push(seen(
                        (emitted(near, &observer, time).pos - observer).normalize(),
                    ));
                    labels.push(label);
                }
            }
            let directions: Vec<_> = sights[start..].iter().map(|s| s.direction).collect();
            let patch = cover(&directions).unwrap_or(field);
            let mut fakes: Vec<_> = (0..self.clutter)
                .map(|_| patch.direction(&mut rng))
                .collect();
            fakes.extend(fixed.iter().copied());
            sights.extend(fakes.into_iter().map(seen));
            labels.resize(sights.len(), CLUTTER);
        }
        for sight in &mut sights {
            add_noise(sight, self.noise, &mut rng);
        }

        // Seen from the Earth, the sights are those of the observations.
        let mut observations = Vec::new();
        if self.from_earth {
            observations = sights
                .iter()
                .map(|s| optical(earth(s.time), s.direction, s.sigma))
                .collect();
            sights = observations
                .iter()
                .map(|observation| sight_from_observation(observation).unwrap())
                .collect();
        }
        Scene {
            sights,
            labels,
            objects,
            origin,
            observations,
        }
    }
}

/// The smallest round field about the mean of `directions` which holds them
/// all, with a margin of three arcminutes. `None` if there are none.
fn cover(directions: &[Vector<Equatorial>]) -> Option<Field> {
    if directions.is_empty() {
        return None;
    }
    let sum = directions
        .iter()
        .fold(Vector::new([0.0; 3]), |sum, d| sum + *d);
    let center = sum.normalize();
    let radius = directions
        .iter()
        .map(|d| d.angle(&center))
        .fold(0.0, f64::max);
    Some(Field {
        center,
        radius: radius + 0.05_f64.to_radians(),
    })
}

/// Labeled sights to link.
#[derive(Debug)]
pub(crate) struct Scene {
    /// Every sight, objects and clutter mixed.
    pub(crate) sights: Vec<Sight>,

    /// Index of the object of each sight, or [`CLUTTER`].
    pub(crate) labels: Vec<usize>,

    /// The objects at the reference time, about the Sun.
    pub(crate) objects: Vec<SunState>,

    /// The first observer at the reference time, from which depths are
    /// measured.
    pub(crate) origin: SunState,

    /// The observation of each sight, when seen from the Earth.
    pub(crate) observations: Vec<AstrometricObservation>,
}

/// How a scene's sights are searched.
#[derive(Debug, Clone)]
pub(crate) struct Search {
    /// Smallest and largest time between the sights of a pair, in days.
    pub(crate) gaps: (f64, f64),

    /// Largest apparent rate of motion, in radians per day.
    pub(crate) max_rate: f64,

    /// Longest run, in days, see `form_units`.
    pub(crate) run_span: f64,

    /// Largest time between consecutive sights of one time group of a
    /// confirmation, in days.
    pub(crate) group_span: f64,

    /// Smallest and largest depth of the grid, in AU. `None` takes them from
    /// the true depths of the objects, as a search aimed at their population
    /// would, see [`aimed`].
    pub(crate) depths: Option<(f64, f64)>,

    /// Largest depth rate over depth of the grid, per day. `None` takes it
    /// from the objects, see [`aimed`].
    pub(crate) rates: Option<f64>,

    /// Steps of the grid in log depth and in depth rate over depth.
    pub(crate) steps: (f64, f64),

    /// Settings of [`link`].
    pub(crate) settings: LinkSettings,

    /// Fewest distinct sights of a candidate of [`link`].
    pub(crate) min_sights: usize,

    /// Fewest observations a confirmed fit keeps, to confirm the candidates
    /// with residuals up to [`MAX_RMS`]. `None` stops at [`link`].
    pub(crate) confirm: Option<usize>,
}

impl Search {
    /// A search with the default grid steps and settings aimed at the
    /// objects, with these gaps in days and largest rate in degrees a day,
    /// runs up to the run span of `search` and time groups up to [`SPAN`],
    /// or the largest gap if shorter.
    pub(crate) fn new(gaps: (f64, f64), max_rate: f64) -> Self {
        Self {
            gaps,
            max_rate: max_rate.to_radians(),
            run_span: SearchSettings::default().run_span.min(gaps.1),
            group_span: SPAN.min(gaps.1),
            depths: None,
            rates: None,
            steps: (DEPTH_STEP, RATE_STEP),
            settings: LinkSettings::default(),
            min_sights: 4,
            confirm: None,
        }
    }

    /// This search with runs and time groups up to `span` days.
    pub(crate) fn span(mut self, span: f64) -> Self {
        self.run_span = span;
        self.group_span = span;
        self
    }

    /// This search with both grid steps `factor` times the defaults.
    pub(crate) fn coarse(mut self, factor: f64) -> Self {
        self.steps = (DEPTH_STEP * factor, RATE_STEP * factor);
        self
    }
}

/// Largest time between consecutive sights of one time group of a search
/// unless it says otherwise, in days, as that of `search`.
pub(crate) const SPAN: f64 = 0.1;

/// Longest run and time group of a search aimed at objects within a few
/// hundredths of an AU, in days. They move too much within a longer run for
/// one orbit to fit it, and a window of a few hours of their sights is one
/// time group at [`SPAN`].
pub(crate) const CLOSE_SPAN: f64 = 0.01;

/// The depths and depth rate over depth a search aimed at `objects` would
/// cover, from `origin`: every depth and rate the objects have along an axis
/// their direction from the origin is aligned with, widened by two grid
/// steps.
fn aimed(objects: &[SunState], origin: &SunState, steps: (f64, f64)) -> (f64, f64, f64) {
    let mut near = f64::INFINITY;
    let mut far = 0.0_f64;
    let mut rate = 0.0_f64;
    for object in objects {
        let offset = object.pos - origin.pos;
        let motion = object.vel - origin.vel;
        for axis in axes() {
            if axis.dot(&offset.normalize()) < MIN_ALIGNMENT {
                continue;
            }
            let depth = axis.dot(&offset);
            near = near.min(depth);
            far = far.max(depth);
            rate = rate.max((axis.dot(&motion) / depth).abs());
        }
    }
    let widen = (2.0 * steps.0).exp();
    (near / widen, far * widen, rate + 2.0 * steps.1)
}

/// The object a group of sights counts for, and how many of the sights show
/// it: the object which has at least `min_sights` of them, as their
/// majority.
pub(crate) fn majority(
    group: &[usize],
    labels: &[usize],
    min_sights: usize,
) -> Option<(usize, usize)> {
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &sight in group {
        *counts.entry(labels[sight]).or_default() += 1;
    }
    let (&label, &count) = counts.iter().max_by_key(|(_, count)| **count)?;
    (label != CLUTTER && count >= min_sights && 2 * count > group.len()).then_some((label, count))
}

/// How well groups of sights match the labels.
///
/// An object is found when some group holds at least `min_sights` of its
/// sights and they are the majority of that group, and whole when that group
/// holds all of its sights. A group is pure when all its sights show one
/// object, and false when it counts for no object.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Score {
    /// Objects with at least the fewest sights a group needs.
    pub(crate) findable: usize,

    /// Groups.
    pub(crate) groups: usize,

    /// Objects found.
    pub(crate) found: usize,

    /// Objects found whole.
    pub(crate) whole: usize,

    /// Objects found by a pure group.
    pub(crate) pure_found: usize,

    /// Pure groups.
    pub(crate) pure: usize,

    /// False groups.
    pub(crate) false_groups: usize,
}

impl Score {
    /// Score `groups` of sight indices against `labels`.
    pub(crate) fn new<'a>(
        groups: impl Iterator<Item = &'a [usize]>,
        labels: &[usize],
        objects: usize,
        min_sights: usize,
    ) -> Self {
        let mut own = vec![0; objects];
        for &label in labels {
            if label != CLUTTER {
                own[label] += 1;
            }
        }
        let mut found = vec![false; objects];
        let mut whole = vec![false; objects];
        let mut pure_found = vec![false; objects];
        let mut score = Self::default();
        for group in groups {
            score.groups += 1;
            let first = labels[group[0]];
            let pure = first != CLUTTER && group.iter().all(|&i| labels[i] == first);
            score.pure += usize::from(pure);
            if let Some((label, count)) = majority(group, labels, min_sights) {
                found[label] = true;
                whole[label] |= count == own[label];
                pure_found[label] |= pure;
            } else {
                score.false_groups += 1;
            }
        }
        let count = |flags: &[bool]| flags.iter().filter(|&&f| f).count();
        score.findable = own.iter().filter(|&&n| n >= min_sights).count();
        score.found = count(&found);
        score.whole = count(&whole);
        score.pure_found = count(&pure_found);
        score
    }

    /// Found over findable, as a percentage.
    pub(crate) fn completeness(&self) -> f64 {
        100.0 * self.found as f64 / self.findable.max(1) as f64
    }

    /// Pure groups over all groups, as a percentage. 100 with no groups.
    pub(crate) fn purity(&self) -> f64 {
        if self.groups == 0 {
            100.0
        } else {
            100.0 * self.pure as f64 / self.groups as f64
        }
    }
}

/// The outcome of searching one scene.
#[derive(Debug)]
pub(crate) struct Run {
    /// Sights, objects and clutter.
    pub(crate) sights: usize,

    /// Units formed.
    pub(crate) units: usize,

    /// Units formed whose sights all show the same object.
    pub(crate) true_units: usize,

    /// Hypotheses searched.
    pub(crate) hypotheses: usize,

    /// The candidates of `link`.
    pub(crate) candidates: Vec<Candidate>,

    /// Score of the candidates.
    pub(crate) linked: Score,

    /// The confirmed candidates, if the search confirms.
    pub(crate) confirmed: Vec<Confirmed>,

    /// Score of the confirmed candidates, with the fewest observations of a
    /// confirmed fit.
    pub(crate) confirmed_score: Score,

    /// Seconds to form the units and build the grid, to link, and to confirm.
    pub(crate) seconds: [f64; 3],
}

/// Search `scene` and score the outcome.
pub(crate) fn run(scene: &Scene, search: &Search) -> Run {
    let (near, far, rate) = aimed(&scene.objects, &scene.origin, search.steps);
    let (near, far) = search.depths.unwrap_or((near, far));
    let rates = search.rates.unwrap_or(rate);
    let start = Instant::now();
    let units = form_units(
        &scene.sights,
        search.gaps.0,
        search.gaps.1,
        search.max_rate,
        search.run_span,
    )
    .unwrap();
    let hypotheses: Vec<_> =
        hypothesis_grid(&scene.origin, near, far, search.steps.0, search.steps.1)
            .unwrap()
            .into_iter()
            .filter(|h| (h.depth_rate / h.depth).abs() <= rates)
            .collect();
    let setup = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let candidates = link(
        &scene.sights,
        &units,
        &hypotheses,
        search.min_sights,
        &search.settings,
    )
    .unwrap();
    let linking = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (confirmed, min_sights) = match search.confirm {
        Some(min_sights) => (
            confirm(
                &scene.observations,
                &candidates,
                &ConfirmRules {
                    min_sights,
                    max_rms: MAX_RMS,
                    min_sights_per_side: 2,
                    group_span: search.group_span,
                    max_gap: search.gaps.1,
                },
            )
            .unwrap(),
            min_sights,
        ),
        None => (Vec::new(), search.min_sights),
    };
    let confirming = start.elapsed().as_secs_f64();
    let score = |groups: Vec<&[usize]>, min_sights| {
        Score::new(
            groups.into_iter(),
            &scene.labels,
            scene.objects.len(),
            min_sights,
        )
    };
    let true_units = units
        .iter()
        .filter(|unit| {
            let label = scene.labels[unit[0]];
            label != CLUTTER && unit.iter().all(|&i| scene.labels[i] == label)
        })
        .count();
    Run {
        sights: scene.sights.len(),
        units: units.len(),
        true_units,
        hypotheses: hypotheses.len(),
        linked: score(
            candidates.iter().map(|c| c.sights.as_slice()).collect(),
            search.min_sights,
        ),
        confirmed_score: score(
            confirmed.iter().map(|c| c.sights.as_slice()).collect(),
            min_sights,
        ),
        candidates,
        confirmed,
        seconds: [setup, linking, confirming],
    }
}

/// Print the title and header of the table [`Run::print`] writes rows of.
pub(crate) fn print_header(title: &str) {
    println!("\n{title}");
    println!(
        "{:<46} {:>6} {:>8} {:>8} {:>7} {:>6} {:>9} {:>5} {:>5} {:>5} {:>5} {:>6} {:>7}",
        "scenario",
        "sights",
        "units",
        "false u",
        "hyps",
        "cands",
        "found",
        "whole",
        "pure",
        "pure%",
        "false",
        "unit s",
        "link s"
    );
}

impl Run {
    /// Print one row of the table: found is over findable objects, whole the
    /// objects found whole, pure the objects found by a pure candidate, false
    /// u the units of two objects or of clutter, see [`Score`].
    pub(crate) fn print(&self, name: &str) {
        let s = &self.linked;
        println!(
            "{:<46} {:>6} {:>8} {:>8} {:>7} {:>6} {:>4}/{:<4} {:>5} {:>5} {:>5.0} {:>5} {:>6.2} {:>7.2}",
            name,
            self.sights,
            self.units,
            self.units - self.true_units,
            self.hypotheses,
            s.groups,
            s.found,
            s.findable,
            s.whole,
            s.pure_found,
            s.purity(),
            s.false_groups,
            self.seconds[0],
            self.seconds[1]
        );
    }
}
