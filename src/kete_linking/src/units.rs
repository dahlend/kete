// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Forming the units of linking from a list of lines of sight: runs of
//! sights close in time, and pairs across gaps.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, SQRT_2};

use kete_core::constants::{AU_KM, GMS};
use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult};

use crate::Sight;
use crate::sight::UNUSABLE;

/// Speed an observer may have over the escape speed from the Sun at its
/// distance, 60 km/s, in AU per day. It is generous for any observer on or
/// about a planet, and only has to part observers far apart seen at nearly
/// the same time.
const OBSERVER_SPEED: f64 = 60.0 * 86400.0 / AU_KM;

/// Errors allowed in the miss of a sight paired with an end of a run, both
/// of position and of the rate of the run, see [`form_units`]. It equals the
/// default `threshold` of [`crate::LinkSettings`], but is fixed: it does not
/// follow a caller's `threshold`.
const TRACK_SIGMAS: f64 = 4.0;

/// Change of the rate of a run allowed in the miss of a sight paired with one
/// of its ends, as a fraction of `max_rate`, see [`form_units`].
const TRACK_SLACK: f64 = 0.05;

/// The units of linking: groups of two or more sights which could be one
/// object, each in time order.
///
/// A unit is a run or a pair. Runs are short arcs of sights close in time,
/// up to `run_span` days long, see below. Pairs are formed among the sights
/// in no run and the first and last sight of each run: sights `i` and `j`
/// are paired when `j` is later than `i` by between `min_gap` and `max_gap`
/// days, the angle between their directions is at most `max_rate` times
/// that gap, and each of the two which ends a run has the other on its
/// track, see below. Two sights of one run are not paired, as the run holds
/// them already. With `run_span` zero there are no runs, and the units are
/// the pairs of all the sights. The order of the units is not specified.
///
/// The ends of a run are paired so that an object which moves too much
/// during a run for one orbit to fit it within its noise, as an object close
/// to the observer does, is still found through pairs, and so that a run is
/// joined to sights in no run. The sights within a run are not, since each
/// of their pairs adds little to the run and costs as much to link.
///
/// A run measures its own motion on the sky, so its ends are paired only along
/// it. The track of a run carries the end on along the great circle from its
/// first to its last sight, at the rate of the run, the angle between them over
/// its span, to the time of the other sight. The other sight is on the track
/// when it misses it by at most `4 s + r t`, where `s` is the uncertainties of
/// the end and of the other sight added in quadrature, `t` the time between the
/// two, and `r` an allowed error of rate: 4 errors of the rate of the run, each
/// the spread of the run over its span, plus 0.05 times `max_rate` for the
/// change of the rate over the gap, from the curvature of the motion and the
/// parallax between places, and no more than `max_rate`. The 4 is fixed, and
/// does not follow the `threshold` of [`crate::LinkSettings`]. The spread is
/// `sqrt(2)` times the larger uncertainty of the first and last sight, or the
/// largest distance of a sight within the run from the great circle through
/// those two at its share of the span, if that is larger. A run which holds
/// sights of two places has its rate off by their parallax over its span, and
/// its sights within stray from its track by most of that parallax, so the
/// track widens to keep the pairs of its object; a run of one place widens only
/// by its noise. In a crowded field pairs of two objects far outnumber pairs of
/// one, and nearly all of them join ends of runs, so the track leaves out most
/// pairs and few of one object.
///
/// A run of two has no sight within, so its spread is that of its ends.
/// Two sights always lie on one track, so a run of two joined across two
/// objects nearby looks like one object, and a run of two from two places
/// has a track off by their parallax over its span; either track can leave
/// out the pairs of both ends. Joining only mutual nearest neighbors makes
/// such runs rare, and the objects are still linked through their other
/// units. The test `run_ends_pair_along_their_track` shows a sight off the
/// track left out, for runs of three and of two, and
/// `runs_of_two_places_widen_their_track` a run whose last sight is from
/// another place.
///
/// Two sights are neighbors in a run when the later is more than zero and at
/// most `run_span` days after the earlier, their directions are within
/// `max_rate` times that gap, and their observers are no farther apart than
/// one observer moves in that time: the escape speed from the Sun at the
/// observer's distance plus 60 km/s. That last rule only parts observers far
/// apart seen at nearly the same time. Positions alone cannot tell a second
/// place from the motion of one observer, such as one in low orbit about the
/// Earth, so a run may hold sights of two places near each other, such as
/// two sites on the Earth. Every sight keeps its own observer, so
/// [`crate::link()`] solves such a run exactly, and its track is used only as
/// above.
///
/// Each sight is joined to its next neighbor, the one at the earliest later
/// time and of those the nearest in direction, when the sight is likewise the
/// neighbor of it at the latest earlier time nearest in direction. Joining
/// only such mutual nearest neighbors keeps each sight in at most one run and
/// leaves out sights of a crowd, which no one rule could share out. Two
/// objects closer together than one step of their motion can still be joined
/// across, which makes a run that fits no orbit; [`crate::link()`] refuses such
/// a run. Chains of joined sights are cut where they would span more than
/// `run_span`, and those of two or more sights are the runs.
///
/// Observers are not compared for pairs, so `max_rate` has to cover the
/// parallax between observers at different places if their pairs are
/// wanted. Parallax larger than the track allows also keeps a sight off the
/// track of a run from another place; such sights are joined to the run
/// through the other runs of their own object, which [`crate::link()`]
/// matches at any time apart.
///
/// Each unit direction is hashed into a cubic cell as wide as the largest
/// separation allowed, and the 27 cells around each sight are searched in
/// time order.
///
/// # Arguments
///
/// * `sights` - Lines of sight, in any order.
/// * `min_gap` - Smallest time between the sights of a pair, in days.
/// * `max_gap` - Largest time between the sights of a pair, in days.
/// * `max_rate` - Largest apparent rate of motion, in radians per day.
/// * `run_span` - Longest run, in days. Zero forms pairs only.
///
/// # Errors
///
/// [`Error::ValueError`] unless `0 <= min_gap <= max_gap`, `max_rate` is
/// finite and positive, and `run_span` is finite and not negative, or if a
/// sight is not [`Sight::is_usable`].
pub fn form_units(
    sights: &[Sight],
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
    run_span: f64,
) -> KeteResult<Vec<Vec<usize>>> {
    check_limits(min_gap, max_gap, max_rate, run_span)?;
    if let Some(bad) = sights.iter().position(|sight| !sight.is_usable()) {
        return Err(Error::ValueError(format!(
            "Form units: sight {bad} has {UNUSABLE}"
        )));
    }
    let runs = form_runs(sights, run_span, max_rate);
    let mut paired = vec![true; sights.len()];
    let mut run_of = vec![usize::MAX; sights.len()];
    for (k, run) in runs.iter().enumerate() {
        for &i in run {
            paired[i] = false;
            run_of[i] = k;
        }
        paired[run[0]] = true;
        paired[run[run.len() - 1]] = true;
    }
    let ends: Vec<usize> = (0..sights.len()).filter(|&i| paired[i]).collect();
    let chosen: Vec<Sight> = ends.iter().map(|&i| sights[i]).collect();
    let pairs = form_pairs(&chosen, min_gap, max_gap, max_rate);
    // How far each run's sights stray from its track.
    let bends: Vec<f64> = runs.iter().map(|run| bend(sights, run)).collect();
    // Whether sight `b` is on the track of the run that sight `a` ends, if
    // it ends one.
    let on_track = |a: usize, b: usize| {
        let Some(run) = runs.get(run_of[a]) else {
            return true;
        };
        let (first, last) = (&sights[run[0]], &sights[run[run.len() - 1]]);
        let span = (last.time - first.time).elapsed;
        let spread = (SQRT_2 * first.sigma.max(last.sigma)).max(bends[run_of[a]]);
        let noise = spread / span;
        let (a, b) = (&sights[a], &sights[b]);
        let gap = (b.time - a.time).elapsed;
        let ahead = turn(first, last, &a.direction, gap / span);
        let allowed = (TRACK_SIGMAS * noise + TRACK_SLACK * max_rate).min(max_rate) * gap.abs()
            + TRACK_SIGMAS * a.sigma.hypot(b.sigma);
        (b.direction - ahead).norm() <= allowed
    };
    let kept = |&(i, j): &(usize, usize)| {
        let (a, b) = (ends[i], ends[j]);
        let apart = run_of[a] == usize::MAX || run_of[a] != run_of[b];
        apart && on_track(a, b) && on_track(b, a)
    };
    let pairs: Vec<Vec<usize>> = pairs
        .into_iter()
        .filter(kept)
        .map(|(i, j)| vec![ends[i], ends[j]])
        .collect();
    Ok(runs.into_iter().chain(pairs).collect())
}

/// The errors of [`form_units`] for its limits.
pub(crate) fn check_limits(
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
    run_span: f64,
) -> KeteResult<()> {
    if !(0.0 <= min_gap && min_gap <= max_gap) {
        return Err(Error::ValueError(format!(
            "Form units: need 0 <= min_gap <= max_gap, got {min_gap} and {max_gap}"
        )));
    }
    if !(max_rate > 0.0 && max_rate.is_finite()) {
        return Err(Error::ValueError(format!(
            "Form units: max_rate must be finite and positive, got {max_rate}"
        )));
    }
    if !(run_span >= 0.0 && run_span.is_finite()) {
        return Err(Error::ValueError(format!(
            "Form units: run_span must be finite and not negative, got {run_span}"
        )));
    }
    Ok(())
}

/// `start`, the direction of sight `first` or `last`, turned about the pole
/// of the great circle through the two by `share` of the angle from the
/// direction of `first` to that of `last`, back when `share` is negative.
fn turn(first: &Sight, last: &Sight, start: &Vector<Equatorial>, share: f64) -> Vector<Equatorial> {
    let pole = first.direction.cross(&last.direction);
    let sin = pole.norm();
    if sin == 0.0 {
        return *start;
    }
    let angle = share * sin.atan2(first.direction.dot(&last.direction));
    *start * angle.cos() + pole.cross(start) * (angle.sin() / sin)
}

/// Largest distance of a sight within `run` from the track from its first
/// to its last sight, zero for a run of two, see [`form_units`].
fn bend(sights: &[Sight], run: &[usize]) -> f64 {
    let (first, last) = (&sights[run[0]], &sights[run[run.len() - 1]]);
    let span = (last.time - first.time).elapsed;
    run[1..run.len() - 1]
        .iter()
        .map(|&k| {
            let sight = &sights[k];
            let share = (sight.time - first.time).elapsed / span;
            (sight.direction - turn(first, last, &first.direction, share)).norm()
        })
        .fold(0.0, f64::max)
}

/// Index pairs of the sights which could be one object, as for
/// [`form_units`] without runs, as `(i, j)` with `j` the later. The
/// arguments are those [`form_units`] accepts.
fn form_pairs(sights: &[Sight], min_gap: f64, max_gap: f64, max_rate: f64) -> Vec<(usize, usize)> {
    // Largest distance between the unit directions of a pair `gap` days apart.
    let reach = |gap: f64| 2.0 * (0.5 * max_rate * gap).min(FRAC_PI_2).sin();
    // No finer than the rounding of a unit vector, which also keeps a zero
    // rate from dividing by zero.
    let width = reach(max_gap).max(f64::EPSILON);
    let cell =
        |sight: &Sight| <[f64; 3]>::from(sight.direction).map(|x| (x / width).floor() as i64);

    // Days from the first sight, so that gaps are plain differences.
    let Some(first) = sights.first() else {
        return Vec::new();
    };
    let days: Vec<f64> = sights
        .iter()
        .map(|sight| (sight.time - first.time).elapsed)
        .collect();
    let mut order: Vec<usize> = (0..sights.len()).collect();
    order.sort_by(|&i, &j| days[i].total_cmp(&days[j]));
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for &i in &order {
        cells.entry(cell(&sights[i])).or_default().push(i);
    }

    let mut pairs = Vec::new();
    for &i in &order {
        let [x, y, z] = cell(&sights[i]);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(list) = cells.get(&[x + dx, y + dy, z + dz]) else {
                        continue;
                    };
                    // Only later sights, as two at one time cannot both be
                    // one object.
                    let start = list.partition_point(|&j| {
                        let gap = days[j] - days[i];
                        gap < min_gap || gap <= 0.0
                    });
                    for &j in &list[start..] {
                        let gap = days[j] - days[i];
                        if gap > max_gap {
                            break;
                        }
                        let apart = (sights[j].direction - sights[i].direction).norm();
                        if apart <= reach(gap) {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        }
    }
    pairs
}

/// The runs of [`form_units`], each in time order, in the order of their
/// first sight. None when `run_span` is zero. The arguments are those
/// [`form_units`] accepts.
fn form_runs(sights: &[Sight], run_span: f64, max_rate: f64) -> Vec<Vec<usize>> {
    let Some(first) = sights.first().filter(|_| run_span > 0.0) else {
        return Vec::new();
    };
    let close = form_pairs(sights, 0.0, run_span, max_rate);
    let days: Vec<f64> = sights
        .iter()
        .map(|sight| (sight.time - first.time).elapsed)
        .collect();
    // For each sight, the nearest later and earlier neighbor, by time and
    // then by direction, as (gap, distance between directions, index).
    let none = (f64::INFINITY, f64::INFINITY, usize::MAX);
    let mut next = vec![none; sights.len()];
    let mut prev = next.clone();
    let keep = |best: &mut (f64, f64, usize), candidate: (f64, f64, usize)| {
        if candidate < *best {
            *best = candidate;
        }
    };
    for (i, j) in close {
        let gap = days[j] - days[i];
        let speed = (2.0 * GMS / sights[i].observer.norm()).sqrt() + OBSERVER_SPEED;
        if (sights[j].observer - sights[i].observer).norm() > speed * gap {
            continue;
        }
        let apart = (sights[j].direction - sights[i].direction).norm();
        keep(&mut next[i], (gap, apart, j));
        keep(&mut prev[j], (gap, apart, i));
    }
    let joined = |i: usize| {
        let j = next[i].2;
        (j != usize::MAX && prev[j].2 == i).then_some(j)
    };
    let mut has_prev = vec![false; sights.len()];
    for i in 0..sights.len() {
        if let Some(j) = joined(i) {
            has_prev[j] = true;
        }
    }
    let mut order: Vec<usize> = (0..sights.len()).filter(|&i| !has_prev[i]).collect();
    order.sort_by(|&i, &j| days[i].total_cmp(&days[j]).then(i.cmp(&j)));
    let mut runs = Vec::new();
    for start in order {
        let mut run = vec![start];
        let mut at = start;
        while let Some(j) = joined(at) {
            // Every link is within the span, so a run cut here has at least
            // two sights.
            if days[j] - days[run[0]] > run_span {
                runs.push(std::mem::take(&mut run));
            }
            run.push(j);
            at = j;
        }
        if run.len() >= 2 {
            runs.push(run);
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::frames::{Equatorial, Vector};
    use kete_core::time::{TDB, Time};

    /// A sight at `days` after JD 2460000.5 from `observer`, toward the unit
    /// vector nearest `direction`.
    fn sight(days: f64, observer: [f64; 3], direction: [f64; 3]) -> Sight {
        Sight {
            time: Time::<TDB>::new(2460000.5 + days),
            observer: Vector::<Equatorial>::new(observer),
            direction: Vector::<Equatorial>::new(direction).normalize(),
            sigma: 1e-6,
        }
    }

    /// One minute, in days.
    const MINUTE: f64 = 1.0 / 1440.0;

    /// Each run, sorted, for comparison.
    fn sorted(mut runs: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        runs.sort();
        runs
    }

    #[test]
    fn runs_join_mutual_nearest_neighbors() {
        let at = [1.0, 0.0, 0.0];
        // Two objects moving apart on the sky, seen together every five
        // minutes, and a third sight of a crowd near the first object at its
        // second time.
        let a = |k: f64| [1.0, 1e-4 * k, 0.0];
        let b = |k: f64| [1.0, 0.01 + 1e-4 * k, 0.0];
        let sights = vec![
            sight(0.0, at, a(0.0)),
            sight(0.0, at, b(0.0)),
            sight(5.0 * MINUTE, at, a(1.0)),
            sight(5.0 * MINUTE, at, b(1.0)),
            sight(10.0 * MINUTE, at, a(2.0)),
            sight(10.0 * MINUTE, at, b(2.0)),
        ];
        let rate = 1.0;
        let runs = sorted(form_runs(&sights, 0.1, rate));
        assert_eq!(runs, vec![vec![0, 2, 4], vec![1, 3, 5]]);

        // A second sight near the first object at the middle time, farther
        // from both of its neighbors than the object is, joins nothing.
        let mut crowded = sights.clone();
        crowded.push(sight(5.0 * MINUTE, at, [1.0, 1e-4, 2e-4]));
        let runs = sorted(form_runs(&crowded, 0.1, rate));
        assert_eq!(runs, vec![vec![0, 2, 4], vec![1, 3, 5]]);

        // One nearer to the first sight and farther from the last is the
        // nearest neighbor of the first, and the object's own middle sight
        // that of the last. The run is cut in two rather than mixed.
        crowded[6] = sight(5.0 * MINUTE, at, [1.0, 0.5e-4, 0.0]);
        let runs = sorted(form_runs(&crowded, 0.1, rate));
        assert_eq!(runs, vec![vec![0, 6], vec![1, 3, 5], vec![2, 4]]);
    }

    #[test]
    fn runs_keep_to_the_span_and_the_rate() {
        let at = [1.0, 0.0, 0.0];
        // One sight every ten minutes for an hour, moving 1e-4 radians each.
        let sights: Vec<Sight> = (0..7)
            .map(|k| {
                sight(
                    10.0 * MINUTE * f64::from(k),
                    at,
                    [1.0, 1e-4 * f64::from(k), 0.0],
                )
            })
            .collect();
        // A span of half an hour cuts the chain into runs of at most four
        // sights.
        let runs = form_runs(&sights, 30.0 * MINUTE + 1e-6, 1.0);
        assert_eq!(runs, vec![vec![0, 1, 2, 3], vec![4, 5, 6]]);
        // A span shorter than the step makes no runs, as does zero.
        assert!(form_runs(&sights, 5.0 * MINUTE, 1.0).is_empty());
        assert!(form_runs(&sights, 0.0, 1.0).is_empty());
        // A rate below the motion makes no runs.
        let slow = 0.5e-4 / (10.0 * MINUTE);
        assert!(form_runs(&sights, 0.1, slow).is_empty());
    }

    #[test]
    fn runs_part_observers_far_apart() {
        // The same direction seen a minute apart from two places: the Earth,
        // and a point a tenth of an AU away, which no observer reaches in a
        // minute. Seen from places 1000 km apart, the sights are joined.
        let near = [1.0 + 1000.0 / AU_KM, 0.0, 0.0];
        let far = [1.1, 0.0, 0.0];
        let dir = [0.0, 1.0, 0.0];
        let moved = [0.0, 1.0, 1e-6];
        let from = |other: [f64; 3]| {
            let sights = [
                sight(0.0, [1.0, 0.0, 0.0], dir),
                sight(MINUTE, other, moved),
            ];
            form_runs(&sights, 0.1, 1.0)
        };
        assert_eq!(from(near), vec![vec![0, 1]]);
        assert!(from(far).is_empty());
    }

    #[test]
    fn run_ends_pair_along_their_track() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A run of three sights, and a day later one sight on its track and
        // one off it by twice what the track allows, both well within the
        // rate limit.
        let run = [0.0, 10.0, 20.0].map(|m| sight(m * MINUTE, at, track(m * MINUTE)));
        let on = sight(1.0, at, track(1.0));
        let off = sight(1.0, at, [1.0, 1e-3, 2e-3]);
        let rate = 0.01;
        let mut sights = run.to_vec();
        sights.extend([on, off]);
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);

        // A run of two over the same span has the same track.
        let sights = [run[0], run[2], on, off];
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);

        // A sight in no run has no track, and pairs with both.
        let sights = [run[0], on, off];
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2]]);
    }

    #[test]
    fn runs_of_two_places_widen_their_track() {
        // An object 0.5 AU away seen from one site three times over twenty
        // minutes, then from a second site 6000 km away five minutes later,
        // which joins the run, and from the first site a day later. The
        // parallax between the sites over the span of the run puts the rate
        // of the run off by more than the noise of its ends allows, and the
        // sights within stray from its track by about as much, which widens
        // it so that the last sight pairs with both ends of the run.
        let site = [1.0, 0.0, 0.0];
        let other = [1.0, 0.0, 6000.0 / AU_KM];
        let seen = |days: f64, observer: [f64; 3]| {
            let object = [1.0 + 0.005 * days, 0.5, 0.0];
            sight(
                days,
                observer,
                std::array::from_fn(|k| object[k] - observer[k]),
            )
        };
        let late = seen(1.0, site);
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0].map(|m| seen(m * MINUTE, site)).to_vec();
        sights.extend([seen(25.0 * MINUTE, other), late]);
        let rate = 0.05;
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.05).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2, 3], vec![0, 4], vec![3, 4]]);
        let runs = form_runs(&sights, 0.05, rate);
        assert!(bend(&sights, &runs[0]) > 10.0 * SQRT_2 * sights[0].sigma);

        // From the first site alone the run is straight, and its track
        // holds the late sight but not one off it by five thousandths of a
        // radian.
        sights[3] = seen(25.0 * MINUTE, site);
        let mut off = late;
        off.direction = (off.direction + Vector::new([0.0, 0.0, 5e-3])).normalize();
        sights.push(off);
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.05).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2, 3], vec![0, 4], vec![3, 4]]);
    }

    #[test]
    fn fast_runs_pair_along_great_circles() {
        // A run moving ten degrees a day along the equator, and a sight on
        // its great circle three days later, thirty degrees on, which a
        // straight line in space carried on from the run misses by more
        // than the track allows.
        let at = [1.0, 0.0, 0.0];
        let rate = 10.0_f64.to_radians();
        let track = |days: f64| [(rate * days).cos(), (rate * days).sin(), 0.0];
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        sights.push(sight(3.0, at, track(3.0)));
        let mut units = form_units(&sights, 1.0, 4.0, 1.1 * rate, 0.05).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);
    }

    #[test]
    fn units_are_runs_and_pairs_across_gaps() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A run of three sights on each of two nights, and a lone sight on
        // a third night.
        let mut times = Vec::new();
        for night in [0.0, 1.0] {
            times.extend([0.0, 10.0, 20.0].map(|m| night + m * MINUTE));
        }
        times.push(2.0);
        let sights: Vec<Sight> = times.iter().map(|&t| sight(t, at, track(t))).collect();
        let mut units = form_units(&sights, 0.5, 1.5, 0.01, 0.1).unwrap();
        units.sort();
        // The runs, then the pairs of their ends across the nights and with
        // the lone sight, and none within a run.
        let expected = vec![
            vec![0, 1, 2],
            vec![0, 3],
            vec![0, 5],
            vec![2, 3],
            vec![2, 5],
            vec![3, 4, 5],
            vec![3, 6],
            vec![5, 6],
        ];
        assert_eq!(units, expected);

        // Without runs the units are every pair.
        let mut units = form_units(&sights, 0.0, 1.5, 0.01, 0.0).unwrap();
        units.sort();
        let mut pairs: Vec<Vec<usize>> = form_pairs(&sights, 0.0, 1.5, 0.01)
            .into_iter()
            .map(|(i, j)| vec![i, j])
            .collect();
        pairs.sort();
        assert_eq!(units, pairs);

        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(form_units(&sights, 0.5, 1.5, 0.01, bad).is_err());
        }
        // Each argument is named in its own error.
        let message = |result: KeteResult<Vec<Vec<usize>>>| result.unwrap_err().to_string();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(message(form_units(&sights, 0.5, 1.5, bad, 0.1)).contains("max_rate"));
        }
        assert!(message(form_units(&sights, 2.0, 1.5, 0.01, 0.1)).contains("min_gap"));
        assert!(message(form_units(&sights, 0.5, f64::NAN, 0.01, 0.1)).contains("max_gap"));
        let mut exact = sights.clone();
        exact[2].sigma = 0.0;
        assert!(message(form_units(&exact, 0.5, 1.5, 0.01, 0.1)).contains("sight 2"));
    }
}
