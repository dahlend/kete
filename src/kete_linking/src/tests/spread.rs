// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! How well the pairs of one object agree under a hypothesis which is wrong,
//! and under astrometric noise.

use kete_core::frames::{Equatorial, Vector};
use kete_core::kepler::propagate_two_body;

use crate::{Hypothesis, MIN_ALIGNMENT, Sight};

use super::synthetic::{
    ARCSEC, CASES, Case, HOUR, Observer, Population, Rng, SECOND, SunState, WINDOWS, add_noise,
    emitted, nearest_axis, sight, true_hypothesis,
};
use super::{orbit_from_hypothesis, reference};

/// Number of pairs of one object, spread evenly over the search window.
const PAIRS: usize = 9;

/// A way for a hypothesis to be wrong.
#[derive(Clone, Copy)]
enum Wrong {
    /// Depth and depth rate both too large by this fraction, as if the whole
    /// scene were scaled about the origin.
    Scale(f64),

    /// Depth too large by this fraction, with the true depth rate.
    Depth(f64),

    /// Depth rate too large by this many AU per day.
    Rate(f64),
}

impl Wrong {
    /// The hypothesis with this error.
    fn apply(self, mut hypothesis: Hypothesis) -> Hypothesis {
        match self {
            Self::Scale(by) => {
                hypothesis.depth *= 1.0 + by;
                hypothesis.depth_rate *= 1.0 + by;
            }
            Self::Depth(by) => hypothesis.depth *= 1.0 + by,
            Self::Rate(by) => hypothesis.depth_rate += by,
        }
        hypothesis
    }
}

/// One object seen by one observer, with the hypothesis which is true of it
/// about the observer along the axis nearest to it at the reference time.
struct Tracked {
    /// Who sees the object.
    observer: Observer,

    /// State of the object at the reference time.
    object: SunState,

    /// The hypothesis which is true of it.
    hypothesis: Hypothesis,
}

impl Tracked {
    /// The object of `case` seen by `observer`.
    fn new(observer: Observer, case: &Case) -> Self {
        let origin = observer.state(reference());
        let object = case.object(&origin);
        let axis = nearest_axis(&(object.pos - origin.pos));
        let hypothesis = true_hypothesis(&object, &origin, axis);
        Self {
            observer,
            object,
            hypothesis,
        }
    }

    /// The pair of sights starting `start` days from the reference time and
    /// `gap` days apart, with astrometric noise of `noise` radians. `None`
    /// if the axis of the hypothesis cannot hold the pair.
    fn pair(&self, start: f64, gap: f64, noise: f64, rng: &mut Rng) -> Option<(Sight, Sight)> {
        let mut a = sight(&self.observer, &self.object, reference() + start).0;
        let mut b = sight(&self.observer, &self.object, reference() + (start + gap)).0;
        add_noise(&mut a, noise, rng);
        add_noise(&mut b, noise, rng);
        let axis = self.hypothesis.direction;
        (axis.dot(&a.direction).min(axis.dot(&b.direction)) >= MIN_ALIGNMENT).then_some((a, b))
    }
}

/// How the pairs of one object land at the reference time.
struct Group {
    /// Largest distance in AU of a pair from the mean position of the pairs.
    spread: f64,

    /// Number of pairs the axis holds.
    held: usize,

    /// Number of those with no orbit.
    lost: usize,
}

/// Solve [`PAIRS`] pairs of an object, spread evenly over a window of
/// half-length `window` days with `gap` days between the sights of each.
fn group(tracked: &Tracked, window: f64, gap: f64, wrong: Wrong, noise: f64) -> Group {
    let hypothesis = wrong.apply(tracked.hypothesis.clone());
    let mut rng = Rng(11);
    let mut held = 0;
    let mut states = Vec::new();
    for k in 0..PAIRS {
        let start = -window + (2.0 * window - gap) * k as f64 / (PAIRS - 1) as f64;
        let Some((a, b)) = tracked.pair(start, gap, noise, &mut rng) else {
            continue;
        };
        held += 1;
        states.extend(orbit_from_hypothesis(&a, &b, &hypothesis));
    }
    let sum = states
        .iter()
        .fold(Vector::<Equatorial>::new([0.0; 3]), |sum, s| sum + s.pos);
    let mean = sum / states.len() as f64;
    Group {
        spread: states
            .iter()
            .map(|s| (s.pos - mean).norm())
            .fold(0.0, f64::max),
        held,
        lost: held - states.len(),
    }
}

/// Print a table with a row per bound population and a column per window.
fn print_table(title: &str, values: &[[f64; 4]; 7]) {
    println!("\n{title}");
    print!("{:>12}", "window");
    for window in WINDOWS {
        print!("{:>10}", format!("{window} d"));
    }
    println!();
    for (population, row) in Population::BOUND.iter().zip(values) {
        print!("{:>12}", format!("{population:?}"));
        for value in row {
            print!("{value:>10.1e}");
        }
        println!();
    }
}

#[test]
fn group_spread_by_hypothesis_error() {
    // Spread is the largest distance in AU of a pair from the mean of the
    // pairs of an object at the reference time. Pairs are one hour long, and
    // the hypothesis is about the observer along the axis nearest the
    // object. Each value is the worst over the objects of a population, over
    // all seven kinds of observer.
    //
    // Without noise, every pair an axis holds has an orbit within a week of
    // the reference time. Past that, interior objects seen from inside 1 AU
    // sweep too far around the Sun, see `hypothesis_convergence`. Noise does
    // not change that, since the limit on unbound orbits allows for it. The
    // spread grows with the window for every kind of error.
    let kinds = [
        (
            "Depth and depth rate both 1 percent too large",
            Wrong::Scale(0.01),
            0.0,
        ),
        ("Depth rate 1e-4 AU/day too large", Wrong::Rate(1e-4), 0.0),
        (
            "Depth 1 percent too large, true depth rate",
            Wrong::Depth(0.01),
            0.0,
        ),
        (
            "True hypothesis, 0.1 arcsec noise",
            Wrong::Scale(0.0),
            0.1 * ARCSEC,
        ),
    ];
    // Worst spreads per kind, over every observer and from the planet alone.
    let mut all = [[[0.0_f64; 4]; 7]; 4];
    let mut planet = [[[0.0_f64; 4]; 7]; 4];
    for case in &CASES {
        let Some(row) = Population::BOUND.iter().position(|p| *p == case.population) else {
            continue;
        };
        for observer in Observer::all(reference()) {
            let alone = observer.name == Observer::planet(reference()).name;
            let tracked = Tracked::new(observer, case);
            for (kind, &(_, wrong, noise)) in kinds.iter().enumerate() {
                for (column, window) in WINDOWS.into_iter().enumerate() {
                    let group = group(&tracked, window, HOUR, wrong, noise);
                    if window < 10.0 {
                        assert_eq!(
                            group.lost, 0,
                            "{} from {}",
                            case.name, tracked.observer.name
                        );
                    }
                    if group.held > 1 {
                        let cell = &mut all[kind][row][column];
                        *cell = cell.max(group.spread);
                        if alone {
                            let cell = &mut planet[kind][row][column];
                            *cell = cell.max(group.spread);
                        }
                    }
                }
            }
        }
    }
    for ((title, _, _), table) in kinds.iter().zip(&all) {
        print_table(&format!("{title}, all observers."), table);
        for (population, row) in Population::BOUND.iter().zip(table) {
            assert!(
                row.windows(2).all(|w| w[0] < w[1]),
                "{title}, {population:?}: {row:?}"
            );
        }
    }

    // The grid step at which the spread from a wrong hypothesis matches the
    // spread from the noise, for the observer at 1 AU. The error of a grid is
    // at most half a step, and the spread grows in proportion to it.
    let steps = |kind: usize, error: f64| {
        let mut out = [[0.0; 4]; 7];
        for row in 0..7 {
            for column in 0..4 {
                let slope = planet[kind][row][column] / error;
                out[row][column] = 2.0 * planet[3][row][column] / slope;
            }
        }
        out
    };
    print_table(
        "Largest step in log depth, scaling depth and rate together, before the spread\n\
         of a wrong hypothesis passes the spread of 0.1 arcsec noise. `planet 1 AU`.",
        &steps(0, 0.01),
    );
    print_table(
        "Largest step in depth rate, AU per day, by the same measure.",
        &steps(1, 1e-4),
    );
}

#[test]
fn group_spread_is_proportional() {
    // A wrong hypothesis spreads the pairs in proportion to its error, up to
    // errors of tens of percent in scale and far past any grid step in rate,
    // over a 7 day window.
    for case in CASES
        .iter()
        .filter(|case| Population::BOUND.contains(&case.population))
    {
        let tracked = Tracked::new(Observer::planet(reference()), case);
        if group(&tracked, 7.0, HOUR, Wrong::Scale(0.0), 0.0).held < 2 {
            continue;
        }
        let kinds: [(fn(f64) -> Wrong, f64); 2] = [(Wrong::Scale, 1e-2), (Wrong::Rate, 1e-4)];
        for (kind, error) in kinds {
            let spread = |error| group(&tracked, 7.0, HOUR, kind(error), 0.0).spread;
            let middle = spread(error);
            let ratios = [spread(0.1 * error) / middle, spread(10.0 * error) / middle];
            assert!(
                (0.08..0.12).contains(&ratios[0]),
                "{}: {ratios:?}",
                case.name
            );
            assert!(
                (5.0..15.0).contains(&ratios[1]),
                "{}: {ratios:?}",
                case.name
            );
        }
    }
}

/// Miss in AU between two pairs of an object solved under one hypothesis:
/// the orbit of the first, carried to the time light left the object for
/// the first sight of the second, against the orbit of the second there.
///
/// The pairs start `start_i` and `start_j` days from the reference time and
/// have `gap` days between their sights. `None` if the axis of the
/// hypothesis cannot hold both pairs, or either has no orbit.
fn pair_miss(
    tracked: &Tracked,
    [start_i, start_j]: [f64; 2],
    gap: f64,
    noise: f64,
    rng: &mut Rng,
) -> Option<f64> {
    let hypothesis = &tracked.hypothesis;
    let pair_i = tracked.pair(start_i, gap, noise, rng)?;
    let pair_j = tracked.pair(start_j, gap, noise, rng)?;
    let state_i = orbit_from_hypothesis(&pair_i.0, &pair_i.1, hypothesis).ok()?;
    let state_j = orbit_from_hypothesis(&pair_j.0, &pair_j.1, hypothesis).ok()?;
    let seen = pair_j.0;
    let there = emitted(&state_j, &seen.observer, seen.time);
    let carried = propagate_two_body(&state_i, there.epoch).unwrap();
    Some((carried.pos - there.pos).norm())
}

#[test]
fn pair_miss_by_hypothesis_error_and_noise() {
    // Grouping compares pairs this way: carry one to the time of the other
    // and measure the miss there, against the astrometric error. With noise
    // and the true hypothesis, the root mean square miss per axis of the sky
    // is the velocity error of the first pair over the time to the second,
    // plus the position error of the second:
    // sqrt(2) sigma range_i / gap |t_j - t_i| + sigma range_j.
    let sigma = 0.1 * ARCSEC;
    let mut rng = Rng(2);
    for case in CASES
        .iter()
        .filter(|case| Population::BOUND.contains(&case.population))
    {
        let tracked = Tracked::new(Observer::planet(reference()), case);
        for gap in [60.0 * SECOND, HOUR, 0.25] {
            for apart in [1.0, 7.0, 15.0] {
                let starts = [-0.5 * apart, 0.5 * apart - gap];
                let range_at =
                    |start: f64| sight(&tracked.observer, &tracked.object, reference() + start).1;
                let between = starts[1] - starts[0];
                let predicted = 2.0_f64.sqrt() * sigma * range_at(starts[0]) / gap * between
                    + sigma * range_at(starts[1]);
                let misses: Vec<f64> = (0..60)
                    .filter_map(|_| pair_miss(&tracked, starts, gap, sigma, &mut rng))
                    .collect();
                if misses.len() < 50 {
                    continue;
                }
                let rms = (misses.iter().map(|m| m * m).sum::<f64>() / misses.len() as f64).sqrt();
                // The prediction is per axis of the sky, and the miss has two.
                let ratio = rms / 2.0_f64.sqrt() / predicted;
                assert!(
                    (0.6..1.6).contains(&ratio),
                    "{} gap {gap} apart {apart}: {ratio}",
                    case.name
                );
            }
        }
    }
}
