// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Forming the units of linking from a list of lines of sight: tracklets of
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

/// Errors allowed in the miss of a sight from the track of a tracklet, both
/// of position and of the rate of the tracklet, see [`form_units`]. It equals
/// the default `threshold` of [`crate::LinkSettings`], but is fixed: it does
/// not follow a caller's `threshold`.
const TRACK_SIGMAS: f64 = 4.0;

/// Change of the rate of a tracklet allowed in the miss of a sight from its
/// track, as a fraction of `max_rate`, see [`form_units`].
const TRACK_SLACK: f64 = 0.05;

/// The units of linking: groups of two or more sights which could be one
/// object, each in time order.
///
/// A unit is a tracklet or a pair. Tracklets are short arcs of sights close
/// in time which move steadily along a great circle, up to `tracklet_span`
/// days long, see below. Pairs are formed among the lone sights, those in no
/// tracklet, and the first and last sight of each tracklet: sights `i` and
/// `j` are paired when `j` is later than `i` by between `min_gap` and
/// `max_gap` days, the angle between their directions is at most `max_rate`
/// times that gap, and each of the two which ends a tracklet has the other on
/// its track, see below. Two sights of one tracklet are not paired, as the
/// tracklet holds them already. With `tracklet_span` zero there are no
/// tracklets, and the units are the pairs of all the sights. The order of the
/// units is not specified.
///
/// The ends of a tracklet are paired so that an object which moves too much
/// during a tracklet for one orbit to fit it within its noise, as an object
/// close to the observer does, is still found through pairs, and so that a
/// tracklet is joined to lone sights. The sights within a tracklet are not,
/// since each of their pairs adds little to the tracklet and costs as much
/// to link.
///
/// # Tracks
///
/// A tracklet measures its own motion on the sky. Its track carries one of
/// its sights on along the great circle from its first to its last sight, at
/// the rate of the tracklet, the angle between them over its span, to the
/// time of another sight. The other sight is on the track when it misses it
/// by at most `4 s + r t`, where `s` is the uncertainties of the two sights
/// added in quadrature, `t` the time between them, and `r` an allowed error
/// of rate: 4 errors of the rate of the tracklet, each `sqrt(2)` times the
/// larger uncertainty of its first and last sight over its span, plus 0.05
/// times `max_rate` for the change of the rate over the gap, from the
/// curvature of the motion and the parallax between places, and no more than
/// `max_rate`. The 4 is fixed, and does not follow the `threshold` of
/// [`crate::LinkSettings`].
///
/// The ends of a tracklet are paired only along its track. In a crowded field
/// pairs of two objects far outnumber pairs of one, and nearly all of them
/// join ends of tracklets, so the track leaves out most pairs and few of one
/// object. The test `tracklet_ends_pair_along_their_track` shows a sight off
/// the track left out, for tracklets of three and of two.
///
/// # Tracklets
///
/// Two sights are neighbors when the later is more than zero and at most
/// `tracklet_span` days after the earlier, their directions are within
/// `max_rate` times that gap, and their observers are no farther apart than
/// one observer moves in that time: the escape speed from the Sun at the
/// observer's distance plus 60 km/s. That last rule only parts observers far
/// apart seen at nearly the same time. Positions alone cannot tell a second
/// place from the motion of one observer, such as one in low orbit about the
/// Earth, so sights of two places near each other, such as two sites on the
/// Earth, can be neighbors. Every sight keeps its own observer, so
/// [`crate::link()`] solves a tracklet which holds sights of two places
/// exactly, and its track is used only as above.
///
/// Each sight is joined to its next neighbor, the one at the earliest later
/// time and of those the nearest in direction, when the sight is likewise the
/// neighbor of it at the latest earlier time nearest in direction. Joining
/// only such mutual nearest neighbors keeps each sight in at most one chain
/// and leaves out sights of a crowd, which no one rule could share out.
///
/// Each chain of joined sights is cut into tracklets in time order. The first
/// two sights of a tracklet need only be joined. A later sight joins the
/// tracklet when it lies on the track of the tracklet carried on from its
/// last sight, and the tracklet then spans at most `tracklet_span`. A sight
/// off the track is left out, as a lone sight, when the sight after it is on
/// the track, so that one stray sight does not cut a tracklet; otherwise the
/// tracklet ends, and a new one starts at that sight. Tracklets
/// of fewer than `min_tracklet` sights are left out, and their sights are
/// lone.
///
/// A tracklet measures a direction and a rate, and sights which do not share
/// one are not the tracklet of one object. False detections joined by chance
/// seldom line up within their noise, so a chain of them breaks into short
/// pieces. A chain which holds the sights of one object from two places,
/// such as two sites on the Earth, whose parallax is larger than the track
/// allows, breaks where the place changes, and each piece pairs along its own
/// track. Two objects closer together than one step of their motion can
/// still be joined across, into a tracklet which fits no orbit;
/// [`crate::link()`] refuses such a tracklet. The cost is an object whose
/// motion on the sky curves beyond the track within `tracklet_span`, such as
/// a very close object seen from the ground: it is cut into shorter
/// tracklets, which still link. The tests `tracklets_keep_to_their_track`
/// and `tracklets_split_where_the_place_changes` show these.
///
/// Two sights always lie on one track, so a tracklet of two joined across two
/// objects nearby looks like one object, and a tracklet of two from two
/// places has a track off by their parallax over its span; either track can
/// leave out the pairs of both ends. Joining only mutual nearest neighbors
/// makes such tracklets rare, and the objects are still linked through their
/// other units.
///
/// Observers are not compared for pairs, so `max_rate` has to cover the
/// parallax between observers at different places if their pairs are
/// wanted. Parallax larger than the track allows also keeps a sight off the
/// track of a tracklet from another place; such sights are joined to the
/// tracklet through the other tracklets of their own object, which
/// [`crate::link()`] matches at any time apart.
///
/// Each unit direction is hashed into a cubic cell as wide as the largest
/// separation allowed, and the 27 cells around each sight are searched in
/// time order.
///
/// # False detections
///
/// The end of a tracklet pairs only along its track, but a lone sight pairs
/// with every sight in reach, `max_rate` times the gap, so where most sights
/// are false detections, which seldom form tracklets, the pairs of two lone
/// sights grow as the square of their density and outnumber all other
/// units. Two false detections within `tracklet_span` and the rate limit of
/// each other also form a tracklet of two, whose track any two sights lie
/// on. Two settings trade the objects found only through such units for
/// fewer units:
///
/// * `min_tracklet` - Tracklets of fewer sights than this are left out, and
///   their sights are lone. Two, the least, keeps every tracklet. Three or
///   more suits a survey which sees each object three or more times in a
///   visit, such as a spacecraft which revisits a field every orbit: it
///   leaves out the tracklets of two that false detections form, and the
///   short pieces a chain of them breaks into. An object seen fewer times in
///   a visit than `min_tracklet` has no tracklet there.
/// * `lone_pairs` - Whether two lone sights may be paired. When false, every
///   pair holds the end of a tracklet, and the track of that tracklet keeps
///   the pair only along it, so the pairs grow with the density of the
///   sights rather than its square; pairs of the ends of two tracklets
///   remain. It suits a survey which sees each object at least
///   `min_tracklet` times in a visit, such as one on the ground which takes
///   its visits in pairs, or a spacecraft which revisits a field every orbit,
///   with many false detections. An object seen in a visit fewer times than
///   `min_tracklet` is then found only through the pairs of those sights
///   with the ends of its tracklets in other visits, and an object with no
///   tracklet is not found. With `tracklet_span` zero there are no tracklets
///   and so no units.
///
/// Neither changes the tracklets of `min_tracklet` or more sights, or the
/// pairs of their ends with each other.
///
/// # Arguments
///
/// * `sights` - Lines of sight, in any order.
/// * `min_gap` - Smallest time between the sights of a pair, in days.
/// * `max_gap` - Largest time between the sights of a pair, in days.
/// * `max_rate` - Largest apparent rate of motion, in radians per day.
/// * `tracklet_span` - Longest tracklet, in days. Zero forms pairs only.
/// * `min_tracklet` - Fewest sights of a tracklet, at least 2, see above.
/// * `lone_pairs` - Whether two lone sights may be paired, see above.
///
/// # Errors
///
/// [`Error::ValueError`] unless `0 <= min_gap <= max_gap`, `max_rate` is
/// finite and positive, `tracklet_span` is finite and not negative, and
/// `min_tracklet` is at least 2, or if a sight is not [`Sight::is_usable`].
pub fn form_units(
    sights: &[Sight],
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
    tracklet_span: f64,
    min_tracklet: usize,
    lone_pairs: bool,
) -> KeteResult<Vec<Vec<usize>>> {
    check_limits(min_gap, max_gap, max_rate, tracklet_span, min_tracklet)?;
    if let Some(bad) = sights.iter().position(|sight| !sight.is_usable()) {
        return Err(Error::ValueError(format!(
            "Form units: sight {bad} has {UNUSABLE}"
        )));
    }
    let tracklets = form_tracklets(sights, tracklet_span, max_rate, min_tracklet);
    let mut paired = vec![true; sights.len()];
    let mut tracklet_of = vec![usize::MAX; sights.len()];
    for (k, tracklet) in tracklets.iter().enumerate() {
        for &i in tracklet {
            paired[i] = false;
            tracklet_of[i] = k;
        }
        paired[tracklet[0]] = true;
        paired[tracklet[tracklet.len() - 1]] = true;
    }
    let ends: Vec<usize> = (0..sights.len()).filter(|&i| paired[i]).collect();
    let chosen: Vec<Sight> = ends.iter().map(|&i| sights[i]).collect();
    // Two lone sights are left out as they are found, so that their pairs
    // are never held.
    let lone = |i: usize| tracklet_of[ends[i]] == usize::MAX;
    let allowed = |i: usize, j: usize| lone_pairs || !(lone(i) && lone(j));
    let pairs = form_pairs(&chosen, min_gap, max_gap, max_rate, allowed);
    // Whether sight `b` is on the track of the tracklet that sight `a` ends,
    // if it ends one.
    let along = |a: usize, b: usize| {
        let Some(tracklet) = tracklets.get(tracklet_of[a]) else {
            return true;
        };
        let (first, last) = (&sights[tracklet[0]], &sights[tracklet[tracklet.len() - 1]]);
        on_track(first, last, &sights[a], &sights[b], max_rate)
    };
    let kept = |&(i, j): &(usize, usize)| {
        let (a, b) = (ends[i], ends[j]);
        let apart = tracklet_of[a] == usize::MAX || tracklet_of[a] != tracklet_of[b];
        apart && along(a, b) && along(b, a)
    };
    let pairs: Vec<Vec<usize>> = pairs
        .into_iter()
        .filter(kept)
        .map(|(i, j)| vec![ends[i], ends[j]])
        .collect();
    Ok(tracklets.into_iter().chain(pairs).collect())
}

/// The errors of [`form_units`] for its limits.
pub(crate) fn check_limits(
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
    tracklet_span: f64,
    min_tracklet: usize,
) -> KeteResult<()> {
    if min_tracklet < 2 {
        return Err(Error::ValueError(format!(
            "Form units: min_tracklet must be at least 2, got {min_tracklet}"
        )));
    }
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
    if !(tracklet_span >= 0.0 && tracklet_span.is_finite()) {
        return Err(Error::ValueError(format!(
            "Form units: tracklet_span must be finite and not negative, got {tracklet_span}"
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

/// Whether `other` lies on the track of the tracklet from `first` to `last`,
/// carried on from its sight `from`, see [`form_units`].
fn on_track(first: &Sight, last: &Sight, from: &Sight, other: &Sight, max_rate: f64) -> bool {
    let span = (last.time - first.time).elapsed;
    let noise = SQRT_2 * first.sigma.max(last.sigma) / span;
    let gap = (other.time - from.time).elapsed;
    let ahead = turn(first, last, &from.direction, gap / span);
    let allowed = (TRACK_SIGMAS * noise + TRACK_SLACK * max_rate).min(max_rate) * gap.abs()
        + TRACK_SIGMAS * from.sigma.hypot(other.sigma);
    (other.direction - ahead).norm() <= allowed
}
/// Index pairs of the sights which could be one object, as for
/// [`form_units`] without tracklets, as `(i, j)` with `j` the later, of
/// those for which `allowed(i, j)` holds. The other arguments are those
/// [`form_units`] accepts.
fn form_pairs(
    sights: &[Sight],
    min_gap: f64,
    max_gap: f64,
    max_rate: f64,
    allowed: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
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
                        if apart <= reach(gap) && allowed(i, j) {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        }
    }
    pairs
}

/// The tracklets of [`form_units`] of at least `min_tracklet` sights, each in
/// time order, in the order of their first sight. None when `tracklet_span`
/// is zero. The other arguments are those [`form_units`] accepts.
fn form_tracklets(
    sights: &[Sight],
    tracklet_span: f64,
    max_rate: f64,
    min_tracklet: usize,
) -> Vec<Vec<usize>> {
    let Some(first) = sights.first().filter(|_| tracklet_span > 0.0) else {
        return Vec::new();
    };
    let close = form_pairs(sights, 0.0, tracklet_span, max_rate, |_, _| true);
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
    // Whether sight `j` may join a tracklet beginning at `begin` and ending
    // at `at`.
    let fits = |begin: usize, at: usize, j: usize, len: usize| {
        days[j] - days[begin] <= tracklet_span
            && (len < 2
                || on_track(
                    &sights[begin],
                    &sights[at],
                    &sights[at],
                    &sights[j],
                    max_rate,
                ))
    };
    let mut tracklets = Vec::new();
    for start in order {
        let mut tracklet = vec![start];
        let mut at = start;
        let mut step = joined(at);
        while let Some(j) = step {
            if fits(tracklet[0], at, j, tracklet.len()) {
                tracklet.push(j);
                at = j;
                step = joined(j);
                continue;
            }
            // A sight off the track of a tracklet of two or more is left out
            // when the one after it is on the track, so that one stray sight
            // does not cut the tracklet; otherwise the tracklet ends there.
            let after = joined(j);
            if let Some(k) =
                after.filter(|&k| tracklet.len() >= 2 && fits(tracklet[0], at, k, tracklet.len()))
            {
                tracklet.push(k);
                at = k;
                step = joined(k);
                continue;
            }
            tracklets.push(std::mem::take(&mut tracklet));
            tracklet.push(j);
            at = j;
            step = after;
        }
        tracklets.push(tracklet);
    }
    tracklets.retain(|tracklet| tracklet.len() >= min_tracklet.max(2));
    tracklets
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

    /// Each tracklet, sorted, for comparison.
    fn sorted(mut tracklets: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        tracklets.sort();
        tracklets
    }

    #[test]
    fn tracklets_join_mutual_nearest_neighbors() {
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
        let tracklets = sorted(form_tracklets(&sights, 0.1, rate, 2));
        assert_eq!(tracklets, vec![vec![0, 2, 4], vec![1, 3, 5]]);

        // A second sight near the first object at the middle time, farther
        // from both of its neighbors than the object is, joins nothing.
        let mut crowded = sights.clone();
        crowded.push(sight(5.0 * MINUTE, at, [1.0, 1e-4, 2e-4]));
        let tracklets = sorted(form_tracklets(&crowded, 0.1, rate, 2));
        assert_eq!(tracklets, vec![vec![0, 2, 4], vec![1, 3, 5]]);

        // One nearer to the first sight and farther from the last is the
        // nearest neighbor of the first, and the object's own middle sight
        // that of the last. The tracklet is cut in two rather than mixed.
        crowded[6] = sight(5.0 * MINUTE, at, [1.0, 0.5e-4, 0.0]);
        let tracklets = sorted(form_tracklets(&crowded, 0.1, rate, 2));
        assert_eq!(tracklets, vec![vec![0, 6], vec![1, 3, 5], vec![2, 4]]);
    }

    #[test]
    fn tracklets_keep_to_the_span_and_the_rate() {
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
        // A span of half an hour cuts the chain into tracklets of at most
        // four sights.
        let tracklets = form_tracklets(&sights, 30.0 * MINUTE + 1e-6, 1.0, 2);
        assert_eq!(tracklets, vec![vec![0, 1, 2, 3], vec![4, 5, 6]]);
        // A span shorter than the step makes no tracklets, as does zero.
        assert!(form_tracklets(&sights, 5.0 * MINUTE, 1.0, 2).is_empty());
        assert!(form_tracklets(&sights, 0.0, 1.0, 2).is_empty());
        // A rate below the motion makes no tracklets.
        let slow = 0.5e-4 / (10.0 * MINUTE);
        assert!(form_tracklets(&sights, 0.1, slow, 2).is_empty());
    }

    #[test]
    fn tracklets_part_observers_far_apart() {
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
            form_tracklets(&sights, 0.1, 1.0, 2)
        };
        assert_eq!(from(near), vec![vec![0, 1]]);
        assert!(from(far).is_empty());
    }

    #[test]
    fn tracklet_ends_pair_along_their_track() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of three sights, and a day later one sight on its track
        // and one off it by twice what the track allows, both well within
        // the rate limit.
        let tracklet = [0.0, 10.0, 20.0].map(|m| sight(m * MINUTE, at, track(m * MINUTE)));
        let on = sight(1.0, at, track(1.0));
        let off = sight(1.0, at, [1.0, 1e-3, 2e-3]);
        let rate = 0.01;
        let mut sights = tracklet.to_vec();
        sights.extend([on, off]);
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1, 2, true).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);

        // A tracklet of two over the same span has the same track.
        let sights = [tracklet[0], tracklet[2], on, off];
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1, 2, true).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);

        // A lone sight has no track, and pairs with both.
        let sights = [tracklet[0], on, off];
        let mut units = form_units(&sights, 0.5, 1.5, rate, 0.1, 2, true).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2]]);
    }

    #[test]
    fn short_tracklets_can_be_left_out() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of two over twenty minutes, and a day later one sight on
        // its track and one off it, as in
        // `tracklet_ends_pair_along_their_track`.
        let mut sights: Vec<Sight> = [0.0, 20.0 * MINUTE, 1.0]
            .map(|t| sight(t, at, track(t)))
            .to_vec();
        sights.push(sight(1.0, at, [1.0, 1e-3, 2e-3]));
        let units = |min_tracklet: usize, lone_pairs: bool| {
            sorted(form_units(&sights, 0.5, 1.5, 0.01, 0.1, min_tracklet, lone_pairs).unwrap())
        };
        assert_eq!(units(2, true), vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
        // Below `min_tracklet` the tracklet is two lone sights, which have no
        // track, and pair with both later sights.
        let lone = vec![vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3]];
        assert_eq!(units(3, true), lone);
        // A tracklet of two is kept at a `min_tracklet` of two whatever
        // `lone_pairs`.
        assert_eq!(units(2, false), units(2, true));
        // With no tracklet left, no two sights pair without `lone_pairs`.
        assert!(units(3, false).is_empty());
    }

    #[test]
    fn lone_pairs_can_be_left_out() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of three, a day later one sight on its track and one off
        // it, and a day after that one more on it, too late to pair with the
        // tracklet.
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        sights.extend([
            sight(1.0, at, track(1.0)),
            sight(1.0, at, [1.0, 1e-3, 2e-3]),
            sight(2.0, at, track(2.0)),
        ]);
        let units = |lone_pairs: bool| {
            sorted(form_units(&sights, 0.5, 1.5, 0.01, 0.1, 2, lone_pairs).unwrap())
        };
        // The lone sights pair with each other across the second gap.
        let tracklet_and_track = vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]];
        let mut all = tracklet_and_track.clone();
        all.extend([vec![3, 5], vec![4, 5]]);
        assert_eq!(units(true), all);
        // Without `lone_pairs` only the pairs along the track of the tracklet
        // are kept.
        assert_eq!(units(false), tracklet_and_track);
    }

    #[test]
    fn tracklets_keep_to_their_track() {
        let at = [1.0, 0.0, 0.0];
        // Six sights ten minutes apart, moving 1e-4 radians each along the
        // equator, each `side(k)` off it to the north.
        let chain = |side: &dyn Fn(u32) -> f64| -> Vec<Sight> {
            (0..6)
                .map(|k| {
                    let step = f64::from(k);
                    sight(10.0 * MINUTE * step, at, [1.0, 1e-4 * step, side(k)])
                })
                .collect()
        };
        let tracklets =
            |sights: &[Sight], min_tracklet: usize| form_tracklets(sights, 0.1, 0.03, min_tracklet);
        // A straight chain is one tracklet.
        assert_eq!(tracklets(&chain(&|_| 0.0), 2), vec![vec![0, 1, 2, 3, 4, 5]]);
        // A last sight off the track by 1e-4, many times what it allows, is
        // lone.
        let off = |at: u32| move |k: u32| if k == at { 1e-4 } else { 0.0 };
        assert_eq!(tracklets(&chain(&off(5)), 2), vec![vec![0, 1, 2, 3, 4]]);
        // One off it within the chain, with the sight after it on the track,
        // is left out and the tracklet goes on.
        assert_eq!(tracklets(&chain(&off(3)), 2), vec![vec![0, 1, 2, 4, 5]]);
        // Two in a row off it end the tracklet before them.
        let two_off = |k: u32| if k == 3 || k == 4 { 1e-4 } else { 0.0 };
        assert_eq!(
            tracklets(&chain(&two_off), 2),
            vec![vec![0, 1, 2], vec![3, 4]]
        );
        // A chain which zig-zags as noise does breaks into tracklets of two,
        // which a `min_tracklet` of three leaves out.
        let zig_zag = chain(&|k| if k % 2 == 0 { 5e-5 } else { -5e-5 });
        let expected = vec![vec![0, 1], vec![2, 3], vec![4, 5]];
        assert_eq!(tracklets(&zig_zag, 2), expected);
        assert!(tracklets(&zig_zag, 3).is_empty());
    }

    #[test]
    fn tracklets_split_where_the_place_changes() {
        // An object 0.5 AU away seen from one site three times over twenty
        // minutes, then from a second site 6000 km away three times, and
        // from the first site a day later, with one more sight then off the
        // track by five thousandths of a radian. The parallax between the
        // sites is many times the noise, so the chain of six sights is cut
        // where the site changes, and the late sight pairs with both ends of
        // each tracklet along its own track.
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
        let mut off = late;
        off.direction = (off.direction + Vector::new([0.0, 0.0, 5e-3])).normalize();
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0].map(|m| seen(m * MINUTE, site)).to_vec();
        sights.extend([25.0, 35.0, 45.0].map(|m| seen(m * MINUTE, other)));
        sights.extend([late, off]);
        let units =
            |sights: &[Sight]| sorted(form_units(sights, 0.5, 1.5, 0.05, 0.05, 2, true).unwrap());
        let expected = vec![
            vec![0, 1, 2],
            vec![0, 6],
            vec![2, 6],
            vec![3, 4, 5],
            vec![3, 6],
            vec![5, 6],
        ];
        assert_eq!(units(&sights), expected);

        // From the first site alone the chain is one tracklet.
        for (k, m) in [(3, 25.0), (4, 35.0), (5, 45.0)] {
            sights[k] = seen(m * MINUTE, site);
        }
        let expected = vec![vec![0, 1, 2, 3, 4, 5], vec![0, 6], vec![5, 6]];
        assert_eq!(units(&sights), expected);
    }

    #[test]
    fn fast_tracklets_pair_along_great_circles() {
        // A tracklet moving ten degrees a day along the equator, and a sight
        // on its great circle three days later, thirty degrees on, which a
        // straight line in space carried on from the tracklet misses by more
        // than the track allows.
        let at = [1.0, 0.0, 0.0];
        let rate = 10.0_f64.to_radians();
        let track = |days: f64| [(rate * days).cos(), (rate * days).sin(), 0.0];
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        sights.push(sight(3.0, at, track(3.0)));
        let mut units = form_units(&sights, 1.0, 4.0, 1.1 * rate, 0.05, 2, true).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);
    }

    #[test]
    fn units_are_tracklets_and_pairs_across_gaps() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of three sights on each of two nights, and a lone sight on
        // a third night.
        let mut times = Vec::new();
        for night in [0.0, 1.0] {
            times.extend([0.0, 10.0, 20.0].map(|m| night + m * MINUTE));
        }
        times.push(2.0);
        let sights: Vec<Sight> = times.iter().map(|&t| sight(t, at, track(t))).collect();
        let mut units = form_units(&sights, 0.5, 1.5, 0.01, 0.1, 2, true).unwrap();
        units.sort();
        // The tracklets, then the pairs of their ends across the nights and
        // with the lone sight, and none within a tracklet.
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

        // Without tracklets the units are every pair.
        let mut units = form_units(&sights, 0.0, 1.5, 0.01, 0.0, 2, true).unwrap();
        units.sort();
        let mut pairs: Vec<Vec<usize>> = form_pairs(&sights, 0.0, 1.5, 0.01, |_, _| true)
            .into_iter()
            .map(|(i, j)| vec![i, j])
            .collect();
        pairs.sort();
        assert_eq!(units, pairs);

        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(form_units(&sights, 0.5, 1.5, 0.01, bad, 2, true).is_err());
        }
        // Each argument is named in its own error.
        let message = |result: KeteResult<Vec<Vec<usize>>>| result.unwrap_err().to_string();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(message(form_units(&sights, 0.5, 1.5, bad, 0.1, 2, true)).contains("max_rate"));
        }
        assert!(message(form_units(&sights, 2.0, 1.5, 0.01, 0.1, 2, true)).contains("min_gap"));
        assert!(
            message(form_units(&sights, 0.5, f64::NAN, 0.01, 0.1, 2, true)).contains("max_gap")
        );
        let mut exact = sights.clone();
        exact[2].sigma = 0.0;
        assert!(message(form_units(&exact, 0.5, 1.5, 0.01, 0.1, 2, true)).contains("sight 2"));
        for bad in [0, 1] {
            let result = form_units(&sights, 0.5, 1.5, 0.01, 0.1, bad, true);
            assert!(message(result).contains("min_tracklet"));
        }
    }
}
