// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Forming the units of linking from a list of lines of sight: tracklets of
//! sights close in time, and pairs across gaps.

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, SQRT_2};

use kete_core::constants::{AU_KM, GMS};
use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult};
use rayon::prelude::*;

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

/// Most pairs the end of a tracklet keeps on each side of it in time, those
/// nearest its track, see [`form_units`].
const PAIRS_PER_SIDE: usize = 2;

/// The units of linking: groups of two or more sights which could be one
/// object, each in time order.
///
/// A unit is a tracklet or a pair. Tracklets are short arcs of sights close
/// in time which move steadily along a great circle, up to `tracklet_span`
/// days long, see below. Pairs are formed among the lone sights, those in no
/// tracklet, and the first and last sight of each tracklet: sights `i` and
/// `j` are paired when `j` is later than `i` by at most `max_gap` days, the
/// angle between their directions is at most `max_rate` times that gap plus
/// `parallax_margin`, and each of the two which ends a tracklet has the other
/// on its track and among the nearest to it, see below. Pairs have no least
/// gap, so a sight is paired however soon after a tracklet it falls, such as
/// one just past the span of a tracklet. Two sights of one tracklet are not
/// paired, as the tracklet holds them already. With `tracklet_span` zero
/// there are no tracklets, and the units are the pairs of all the sights with
/// `lone_pairs`, and none without. The order of the units is not specified.
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
/// curvature of the motion, and no more than `max_rate`, plus
/// `parallax_margin` for the parallax between the observers of the two
/// sights. The 4 is fixed, and does not follow the `threshold` of
/// [`crate::LinkSettings`].
///
/// The ends of a tracklet are paired only along its track. In a crowded field
/// pairs of two objects far outnumber pairs of one, and nearly all of them
/// join ends of tracklets, so the track leaves out most pairs and few of one
/// object. The test `tracklet_ends_pair_along_their_track` shows a sight off
/// the track left out, for tracklets of three and of two.
///
/// Of the pairs on its track, each end of a tracklet keeps the two nearest
/// it on each side in time, nearest by the miss over what the track allows,
/// ties to the pair found first; a pair of two tracklet ends has to be kept
/// by both. The track allows for the curvature of every motion up to
/// `max_rate`, which among many false detections admits many sights of
/// nothing; the true sight of the object is nearly always the nearest, and
/// the other units of the object join it in [`crate::link()`] at any time
/// apart. This bounds the pairs at four for each tracklet end however dense
/// the false detections. The test `crowded_ends_keep_their_nearest_pairs`
/// shows it.
///
/// # Tracklets
///
/// A tracklet is two or more sights, within `tracklet_span` days, of one
/// smooth motion on the sky that one observer could have seen. Two sights
/// are neighbors when the later is more than zero and at most
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
/// Every two neighbors at most `tracklet_span / (min_tracklet - 1)` days
/// apart seed a tracklet: a tracklet of `min_tracklet` sights within
/// `tracklet_span` has two sights in a row at most that far apart, so a
/// larger `min_tracklet` means fewer seeds. A tracklet grows from its seed
/// along its track, which runs from its first to its last sight as it
/// grows. Of the neighbors of its last sight which are later and within
/// `tracklet_span` of its first, it takes the earliest on its track, of
/// those at one time the one with the least miss in units of what the track
/// allows, then the lowest index, until none is on the track. It then grows
/// back in time the same way from its first sight, within `tracklet_span` of
/// its last. Sights off the track between those it takes are passed over.
/// What the track allows within a tracklet is the noise of the sights and of
/// the rate the tracklet measures, with no allowance for a change of rate,
/// unlike the track gate of a pair, as an object's rate does not change
/// beyond its noise within a tracklet; that keeps false detections from
/// lining up by chance.
///
/// Every seed grows a tracklet. Those of at least `min_tracklet` sights are
/// taken in order of the most sights, then the least sum of the misses of
/// the sights they grew by, then the lowest first index. Each keeps its
/// sights which no tracklet kept before it holds, when at least
/// `min_tracklet` of them remain: a subset of a tracklet lies on its track.
/// Sights in no kept tracklet are lone.
///
/// A tracklet measures a direction and a rate, and sights which do not share
/// one are not the tracklet of one object. A false detection seldom lies on
/// the track of an object, so it neither takes the place of the object's own
/// sight nor ends its tracklet, even when it is nearer in direction, and a
/// missed detection does not end a tracklet either, as the track carries on
/// past it. Sights of one object from two places, such as two sites on the
/// Earth, whose parallax is larger than the track allows, are off each
/// other's tracks, and form a tracklet for each place, each paired along its
/// own track. The costs:
///
/// * An object whose motion on the sky curves beyond the track within
///   `tracklet_span`, such as a very close object seen from the ground, is cut
///   into several shorter tracklets, which still link.
/// * Two objects whose tracks cross can swap sights where they meet. Of two
///   tracklets which share a sight, the rule of most sights first keeps one;
///   a tracklet joined across two objects fits no orbit, and
///   [`crate::link()`] refuses it.
/// * A few sights of an object seen over longer than `tracklet_span` can be
///   left lone between its tracklets, when fewer than `min_tracklet` of them
///   remain.
///
/// The tests `tracklets_grow_along_their_track`,
/// `missed_sights_do_not_cut_tracklets`,
/// `crossing_tracklets_share_out_their_sights` and
/// `tracklets_split_where_the_place_changes` show these.
///
/// Two sights always lie on one track, so a tracklet of two across two
/// objects nearby looks like one object, and a tracklet of two from two
/// places has a track off by their parallax over its span; either track can
/// leave out the pairs of both ends. The objects are still linked through
/// their other units.
///
/// Observers are not compared for pairs: `parallax_margin` bounds the
/// angle between the directions of one object seen at one time by two of the
/// observers, see [`crate::parallax_margin`], and is added to both the reach
/// of a pair and what a track allows. Without it, a sight seen from a place
/// far from the observer of a tracklet misses its track by their parallax,
/// and an object seen there only in sights of no tracklet of their own is
/// not found.
///
/// Each unit direction is hashed into a cubic cell as wide as the largest
/// separation allowed, and the 27 cells around each sight are searched in
/// time order.
///
/// # False detections
///
/// The end of a tracklet pairs only along its track, and keeps at most two
/// pairs on each side of it in time, so the pairs which hold the end of a
/// tracklet grow with the tracklets, not with the false detections around
/// them. With `lone_pairs` two lone sights pair whenever they are in reach,
/// `max_rate` times the gap, so where most sights are false detections, which
/// seldom form tracklets, the pairs of two lone sights grow as the square of
/// their density and outnumber all other units. Two false detections within
/// `tracklet_span` and the rate limit of each other also form a tracklet of
/// two, whose track any two sights lie on. Three or more line up on a track
/// only by chance, but every seed is grown, so in a crowd where each sight has
/// many neighbors such tracklets grow about as the cube of the density of the
/// false detections. Two settings trade the objects found only through such
/// units for fewer units:
///
/// * `min_tracklet` - Tracklets of fewer sights than this are left out, and
///   their sights are lone. Two, the least, keeps every tracklet. Three or
///   more suits a survey which sees each object three or more times in a
///   visit, such as a spacecraft which revisits a field every orbit: it
///   leaves out the tracklets of two that false detections form, and seeds
///   only neighbors closer in time, see above. An object seen fewer times in
///   a visit than `min_tracklet` has no tracklet there.
/// * `lone_pairs` - Whether two lone sights may be paired. When false, every
///   pair holds the end of a tracklet, and at most four pairs hold each end,
///   so the pairs grow with the tracklets rather than the square of the
///   density of the sights; pairs of the ends of two tracklets remain. It
///   suits a survey which sees each object at least `min_tracklet` times in a
///   visit, such as one on the ground which takes its visits in pairs, or a
///   spacecraft which revisits a field every orbit, with many false
///   detections. An object seen in a visit fewer times than `min_tracklet` is
///   then found only through the pairs of those sights with the ends of its
///   tracklets in other visits, and an object with no tracklet is not found.
///   With `tracklet_span` zero there are no tracklets and so no units.
///
/// `lone_pairs` changes neither the tracklets nor the pairs of the ends of
/// two tracklets.
///
/// # Arguments
///
/// * `sights` - Lines of sight, in any order.
/// * `max_gap` - Largest time between the sights of a pair, in days.
/// * `max_rate` - Largest apparent rate of motion, in radians per day.
/// * `tracklet_span` - Longest tracklet, in days. Zero forms no tracklets.
/// * `min_tracklet` - Fewest sights of a tracklet, at least 2, see above.
/// * `lone_pairs` - Whether two lone sights may be paired, see above.
/// * `parallax_margin` - Largest angle between the directions of one object
///   seen at one time by two of the observers, in radians, see above.
///
/// # Errors
///
/// [`Error::ValueError`] unless `max_gap` is not negative, `max_rate` is
/// finite and positive, `tracklet_span` and `parallax_margin` are finite and
/// not negative, and `min_tracklet` is at least 2, or if a sight is not
/// [`Sight::is_usable`].
pub fn form_units(
    sights: &[Sight],
    max_gap: f64,
    max_rate: f64,
    tracklet_span: f64,
    min_tracklet: usize,
    lone_pairs: bool,
    parallax_margin: f64,
) -> KeteResult<Vec<Vec<usize>>> {
    check_limits(
        max_gap,
        max_rate,
        tracklet_span,
        min_tracklet,
        parallax_margin,
    )?;
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
    let pairs = form_pairs(&chosen, max_gap, max_rate, parallax_margin, allowed);
    // The miss of sight `b` from the track of the tracklet that sight `a`
    // ends, over what the track allows, if `a` ends one.
    let miss = |a: usize, b: usize| {
        let tracklet = tracklets.get(tracklet_of[a])?;
        let (first, last) = (&sights[tracklet[0]], &sights[tracklet[tracklet.len() - 1]]);
        Some(track_miss(
            first,
            last,
            &sights[a],
            &sights[b],
            max_rate,
            parallax_margin,
        ))
    };
    // Each pair on the tracks of all the tracklets it ends, with its miss at
    // each end, by the end, which side of it in time the other sight is, and
    // the pair, and how many ends it holds. A pair off any of the tracks, or
    // of two sights of one tracklet, holds none of the places of either end.
    let mut at_ends: Vec<(usize, bool, f64, usize)> = Vec::new();
    let mut needed: Vec<Option<u8>> = vec![None; pairs.len()];
    for (k, &(i, j)) in pairs.iter().enumerate() {
        let (a, b) = (ends[i], ends[j]);
        if tracklet_of[a] != usize::MAX && tracklet_of[a] == tracklet_of[b] {
            continue;
        }
        let misses = [(a, b), (b, a)].map(|(end, other)| (end, other, miss(end, other)));
        if misses.iter().any(|&(_, _, m)| m.is_some_and(|m| m > 1.0)) {
            continue;
        }
        let mut count = 0;
        for (end, other, m) in misses {
            if let Some(m) = m {
                count += 1;
                at_ends.push((end, sights[other].time > sights[end].time, m, k));
            }
        }
        needed[k] = Some(count);
    }
    // The nearest pairs on each side of each end, ties to the lower pair.
    at_ends.sort_by(|x, y| {
        (x.0, x.1)
            .cmp(&(y.0, y.1))
            .then(x.2.total_cmp(&y.2))
            .then(x.3.cmp(&y.3))
    });
    let mut kept_at = vec![0_u8; pairs.len()];
    for same in at_ends.chunk_by(|x, y| (x.0, x.1) == (y.0, y.1)) {
        for &(_, _, _, k) in same.iter().take(PAIRS_PER_SIDE) {
            kept_at[k] += 1;
        }
    }
    let pairs: Vec<Vec<usize>> = pairs
        .into_iter()
        .enumerate()
        .filter(|&(k, _)| needed[k] == Some(kept_at[k]))
        .map(|(_, (i, j))| vec![ends[i], ends[j]])
        .collect();
    Ok(tracklets.into_iter().chain(pairs).collect())
}

/// The errors of [`form_units`] for its limits.
pub(crate) fn check_limits(
    max_gap: f64,
    max_rate: f64,
    tracklet_span: f64,
    min_tracklet: usize,
    parallax_margin: f64,
) -> KeteResult<()> {
    if min_tracklet < 2 {
        return Err(Error::ValueError(format!(
            "Form units: min_tracklet must be at least 2, got {min_tracklet}"
        )));
    }
    if max_gap.is_nan() || max_gap < 0.0 {
        return Err(Error::ValueError(format!(
            "Form units: max_gap must not be negative, got {max_gap}"
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
    if !(parallax_margin >= 0.0 && parallax_margin.is_finite()) {
        return Err(Error::ValueError(format!(
            "Form units: parallax_margin must be finite and not negative, got {parallax_margin}"
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

/// The miss of `other` from the track of the tracklet from `first` to
/// `last`, carried on from its sight `from`, over what the track allows with
/// `margin` radians for parallax, so that `other` is on the track when this
/// is at most one, see [`form_units`].
fn track_miss(
    first: &Sight,
    last: &Sight,
    from: &Sight,
    other: &Sight,
    max_rate: f64,
    margin: f64,
) -> f64 {
    let gap = (other.time - from.time).elapsed;
    let ahead = carry(first, last, from, gap);
    (other.direction - ahead).norm()
        / (allowance(first, last, from, other, gap, max_rate, TRACK_SLACK) + margin)
}

/// The direction of the track of the tracklet from `first` to `last`,
/// carried on from its sight `from` by `gap` days, see [`form_units`].
fn carry(first: &Sight, last: &Sight, from: &Sight, gap: f64) -> Vector<Equatorial> {
    let span = (last.time - first.time).elapsed;
    turn(first, last, &from.direction, gap / span)
}

/// The largest distance between the unit direction of `other`, `gap` days
/// from `from`, and the track of the tracklet from `first` to `last` carried
/// on from `from`, see [`form_units`].
fn allowance(
    first: &Sight,
    last: &Sight,
    from: &Sight,
    other: &Sight,
    gap: f64,
    max_rate: f64,
    slack: f64,
) -> f64 {
    let span = (last.time - first.time).elapsed;
    let noise = SQRT_2 * first.sigma.max(last.sigma) / span;
    (TRACK_SIGMAS * noise + slack * max_rate).min(max_rate) * gap.abs()
        + TRACK_SIGMAS * from.sigma.hypot(other.sigma)
}

/// Index pairs of the sights which could be one object, as `(i, j)` with `j`
/// later than `i` by at most `max_gap` days, and their directions within
/// `max_rate` times that plus `margin` radians, of those for which
/// `allowed(i, j)` holds.
fn form_pairs(
    sights: &[Sight],
    max_gap: f64,
    max_rate: f64,
    margin: f64,
    allowed: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize)> {
    // Largest distance between the unit directions of a pair `gap` days apart.
    let reach = |gap: f64| 2.0 * (0.5 * max_rate.mul_add(gap, margin)).min(FRAC_PI_2).sin();
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
                    let start = list.partition_point(|&j| days[j] <= days[i]);
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
    let days: Vec<f64> = sights
        .iter()
        .map(|sight| (sight.time - first.time).elapsed)
        .collect();
    // Neighbors: later sights within the span, the rate limit, and the reach
    // of the observer.
    let reachable = |i: usize, j: usize| {
        let speed = (2.0 * GMS / sights[i].observer.norm()).sqrt() + OBSERVER_SPEED;
        (sights[j].observer - sights[i].observer).norm() <= speed * (days[j] - days[i])
    };
    let mut later = vec![Vec::new(); sights.len()];
    let mut earlier = vec![Vec::new(); sights.len()];
    for (i, j) in form_pairs(sights, tracklet_span, max_rate, 0.0, reachable) {
        later[i].push(j);
        earlier[j].push(i);
    }
    // Nearest in time first, then by index.
    for list in &mut later {
        list.sort_by(|&a, &b| days[a].total_cmp(&days[b]).then(a.cmp(&b)));
    }
    for list in &mut earlier {
        list.sort_by(|&a, &b| days[b].total_cmp(&days[a]).then(a.cmp(&b)));
    }
    let growth = Growth {
        sights,
        days: &days,
        later: &later,
        earlier: &earlier,
        tracklet_span,
        max_rate,
    };

    // A tracklet of `min_tracklet` sights within the span has two in a row
    // at most this far apart, which seed it.
    let seed_gap = tracklet_span / (min_tracklet - 1) as f64;
    let mut seeds = Vec::new();
    for (i, list) in later.iter().enumerate() {
        let close = list.iter().take_while(|&&j| days[j] - days[i] <= seed_gap);
        seeds.extend(close.map(|&j| (i, j)));
    }
    let mut grown: Vec<(Vec<usize>, f64)> = seeds
        .par_iter()
        .map_init(Vec::new, |tracklet, &seed| {
            let miss = growth.grow(seed, tracklet);
            (tracklet.len() >= min_tracklet).then(|| (tracklet.clone(), miss))
        })
        .flatten()
        .collect();

    // Most sights first, then the least miss, then the earliest sights.
    grown.sort_by(|(a, a_miss), (b, b_miss)| {
        b.len()
            .cmp(&a.len())
            .then(a_miss.total_cmp(b_miss))
            .then_with(|| a.cmp(b))
    });
    let mut taken = vec![false; sights.len()];
    let mut tracklets = Vec::new();
    for (tracklet, _) in grown {
        // The sights of a tracklet not taken by a longer one, when enough
        // remain; a subset of a tracklet lies on its track.
        let free: Vec<usize> = tracklet.iter().copied().filter(|&i| !taken[i]).collect();
        if free.len() >= min_tracklet {
            for &i in &free {
                taken[i] = true;
            }
            tracklets.push(free);
        }
    }
    tracklets.sort_by(|a, b| days[a[0]].total_cmp(&days[b[0]]).then(a[0].cmp(&b[0])));
    tracklets
}

/// The growth of tracklets from their seeds, see [`form_units`].
struct Growth<'a> {
    /// The sights.
    sights: &'a [Sight],

    /// Days of each sight after the first sight.
    days: &'a [f64],

    /// Neighbors of each sight later in time, nearest in time first.
    later: &'a [Vec<usize>],

    /// Neighbors of each sight earlier in time, nearest in time first.
    earlier: &'a [Vec<usize>],

    /// Longest tracklet, in days.
    tracklet_span: f64,

    /// Largest apparent rate of motion, in radians per day.
    max_rate: f64,
}

impl Growth<'_> {
    /// The tracklet grown from the two sights of `seed`, into `tracklet`, and
    /// the sum of the misses of the sights it added, each in units of what
    /// its track allowed.
    fn grow(&self, seed: (usize, usize), tracklet: &mut Vec<usize>) -> f64 {
        tracklet.clear();
        tracklet.extend([seed.0, seed.1]);
        let mut total = 0.0;
        while let Some((k, miss)) = self.step(tracklet, true) {
            tracklet.push(k);
            total += miss;
        }
        while let Some((k, miss)) = self.step(tracklet, false) {
            tracklet.insert(0, k);
            total += miss;
        }
        total
    }

    /// The sight which carries `tracklet` on, later in time when `forward`
    /// and earlier otherwise, with its miss in units of what the track
    /// allows: of the neighbors of its end on its track within the span of
    /// the tracklet, the nearest in time, then the least miss, then the
    /// lowest index.
    fn step(&self, tracklet: &[usize], forward: bool) -> Option<(usize, f64)> {
        let (first, last) = (tracklet[0], tracklet[tracklet.len() - 1]);
        let (end, neighbors) = if forward {
            (last, &self.later[last])
        } else {
            (first, &self.earlier[first])
        };
        let (a, b, from) = (&self.sights[first], &self.sights[last], &self.sights[end]);
        let mut best: Option<(usize, f64)> = None;
        // The track at the time of the last neighbor, which neighbors at one
        // time share.
        let mut ahead = (f64::NAN, from.direction);
        for &k in neighbors {
            let (low, high) = if forward { (first, k) } else { (k, last) };
            if self.days[high] - self.days[low] > self.tracklet_span
                || best.is_some_and(|(j, _)| self.days[j] != self.days[k])
            {
                break;
            }
            let other = &self.sights[k];
            let gap = (other.time - from.time).elapsed;
            if gap != ahead.0 {
                ahead = (gap, carry(a, b, from, gap));
            }
            let distance = (other.direction - ahead.1).norm();
            // Within a tracklet the rate does not change beyond its noise, so
            // no slack for a change of rate is allowed, unlike for a pair.
            let allowed = allowance(a, b, from, other, gap, self.max_rate, 0.0);
            if distance <= allowed && best.is_none_or(|(_, miss)| distance / allowed < miss) {
                best = Some((k, distance / allowed));
            }
        }
        best
    }
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
    fn tracklets_grow_along_their_track() {
        let at = [1.0, 0.0, 0.0];
        // Two objects moving apart on the sky, seen together every five
        // minutes.
        let a = |k: f64| [1.0, 1e-4 * k, 0.0];
        let b = |k: f64| [1.0, 0.01 + 1e-4 * k, 0.0];
        let mut sights = Vec::new();
        for k in 0..5 {
            let step = f64::from(k);
            sights.extend([
                sight(5.0 * MINUTE * step, at, a(step)),
                sight(5.0 * MINUTE * step, at, b(step)),
            ]);
        }
        let rate = 0.05;
        let tracklets = |sights: &[Sight]| sorted(form_tracklets(sights, 0.1, rate, 2));
        let expected = vec![vec![0, 2, 4, 6, 8], vec![1, 3, 5, 7, 9]];
        assert_eq!(tracklets(&sights), expected);

        // A false detection at the second time, nearer to the first sight of
        // the first object than its own second sight is, and off its track
        // by over a hundred times the uncertainty, joins nothing.
        let mut crowded = sights.clone();
        crowded.push(sight(5.0 * MINUTE, at, [1.0, 0.3e-4, 0.3e-4]));
        assert!(
            (crowded[10].direction - crowded[0].direction).norm()
                < (crowded[2].direction - crowded[0].direction).norm()
        );
        assert_eq!(tracklets(&crowded), expected);
    }

    #[test]
    fn missed_sights_do_not_cut_tracklets() {
        let at = [1.0, 0.0, 0.0];
        let track = |k: f64| sight(10.0 * MINUTE * k, at, [1.0, 1e-4 * k, 0.0]);
        // Six sights of an object ten minutes apart, the fourth missed, and
        // a false detection at its time thirty times the uncertainty off the
        // track.
        let mut sights: Vec<Sight> = [0.0, 1.0, 2.0, 4.0, 5.0].map(track).to_vec();
        let mut false_sight = track(3.0);
        false_sight.direction = (false_sight.direction + Vector::new([0.0, 0.0, 3e-5])).normalize();
        sights.push(false_sight);
        assert_eq!(
            form_tracklets(&sights, 0.1, 0.03, 3),
            vec![vec![0, 1, 2, 3, 4]]
        );
        // Without the false detection the tracklet is the same.
        assert_eq!(
            form_tracklets(&sights[..5], 0.1, 0.03, 3),
            vec![vec![0, 1, 2, 3, 4]]
        );
    }

    #[test]
    fn crossing_tracklets_share_out_their_sights() {
        let at = [1.0, 0.0, 0.0];
        // Two objects seen every ten minutes, one along the equator and one
        // across it, which pass within two uncertainties of each other at
        // the third time.
        let mut sights = Vec::new();
        for k in 0..5 {
            let step = f64::from(k);
            let time = 10.0 * MINUTE * step;
            sights.extend([
                sight(time, at, [1.0, 1e-4 * step, 0.0]),
                sight(time, at, [1.0, 1e-4 * step, 1e-4 * (2.0 - step) + 2e-6]),
            ]);
        }
        let tracklets = sorted(form_tracklets(&sights, 0.1, 0.03, 3));
        assert_eq!(tracklets.len(), 2, "{tracklets:?}");
        for tracklet in &tracklets {
            // Each holds one object away from the crossing.
            let away: Vec<usize> = tracklet
                .iter()
                .filter(|&&i| i / 2 != 2)
                .map(|&i| i % 2)
                .collect();
            assert!(away.windows(2).all(|w| w[0] == w[1]), "{tracklets:?}");
        }
    }

    #[test]
    fn tracklets_keep_to_the_span_and_the_rate() {
        let at = [1.0, 0.0, 0.0];
        // One sight every ten minutes for half an hour, moving 1e-4 radians
        // each.
        let sights: Vec<Sight> = (0..4)
            .map(|k| {
                sight(
                    10.0 * MINUTE * f64::from(k),
                    at,
                    [1.0, 1e-4 * f64::from(k), 0.0],
                )
            })
            .collect();
        assert_eq!(
            form_tracklets(&sights, 30.0 * MINUTE + 1e-6, 1.0, 2),
            vec![vec![0, 1, 2, 3]]
        );
        // A span of twenty minutes holds three of them, the first three or
        // the last three, and leaves one lone.
        let tracklets = form_tracklets(&sights, 20.0 * MINUTE + 1e-6, 1.0, 2);
        assert!(
            tracklets == vec![vec![0, 1, 2]] || tracklets == vec![vec![1, 2, 3]],
            "{tracklets:?}"
        );
        // A span shorter than the step makes no tracklets, as does zero.
        assert!(form_tracklets(&sights, 5.0 * MINUTE, 1.0, 2).is_empty());
        assert!(form_tracklets(&sights, 0.0, 1.0, 2).is_empty());
        // A rate below the motion makes no tracklets.
        let slow = 0.5e-4 / (10.0 * MINUTE);
        assert!(form_tracklets(&sights, 0.1, slow, 2).is_empty());
    }

    #[test]
    fn tracklets_are_seeded_by_their_closest_sights() {
        let at = [1.0, 0.0, 0.0];
        // Three sights of an object within a tenth of a day, two of them
        // farther apart than the twentieth of a day a tracklet of three
        // needs between two of its sights in a row. A tracklet of three
        // grows from the close pair, before it or after it.
        let three = |times: [f64; 3]| -> Vec<Sight> {
            times.map(|t| sight(t, at, [1.0, 1e-3 * t, 0.0])).to_vec()
        };
        for times in [[0.0, 0.07, 0.09], [0.0, 0.02, 0.09]] {
            assert_eq!(
                form_tracklets(&three(times), 0.1, 0.03, 3),
                vec![vec![0, 1, 2]],
                "{times:?}"
            );
        }
        // Two sights a little more than that apart seed nothing at a
        // `min_tracklet` of three, so a third sight beyond the span from the
        // first cannot reach them.
        let sights = three([0.0, 0.06, 0.11]);
        assert!(form_tracklets(&sights, 0.1, 0.03, 3).is_empty());
        // At two they are a tracklet.
        assert_eq!(sorted(form_tracklets(&sights, 0.1, 0.03, 2)).len(), 1);
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
        let mut units = form_units(&sights, 1.5, rate, 0.1, 2, true, 0.0).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);

        // A tracklet of two over the same span has the same track.
        let sights = [tracklet[0], tracklet[2], on, off];
        let mut units = form_units(&sights, 1.5, rate, 0.1, 2, true, 0.0).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);

        // A lone sight has no track, and pairs with both.
        let sights = [tracklet[0], on, off];
        let mut units = form_units(&sights, 1.5, rate, 0.1, 2, true, 0.0).unwrap();
        units.sort();
        assert_eq!(units, vec![vec![0, 1], vec![0, 2]]);
    }

    #[test]
    fn crowded_ends_keep_their_nearest_pairs() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of three, and a day later its own sight on its track
        // among twenty false ones which are on the track too, within its
        // slack, but farther from it, and one more a day before.
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        let late = 1.0 + 20.0 * MINUTE;
        sights.push(sight(late, at, track(late)));
        for k in 1..=20 {
            let off = 1e-6 * f64::from(k);
            sights.push(sight(late, at, [1.0, 1e-3 * late, off]));
        }
        sights.push(sight(-1.0, at, track(-1.0)));
        let units = sorted(form_units(&sights, 1.5, 0.01, 0.1, 2, false, 0.0).unwrap());
        let pairs: Vec<&Vec<usize>> = units.iter().filter(|unit| unit.len() == 2).collect();
        // Each end keeps two later pairs, the object's own sight first, and
        // the earlier sight.
        for end in [0, 2] {
            let later: Vec<usize> = pairs
                .iter()
                .filter(|pair| pair[0] == end)
                .map(|pair| pair[1])
                .collect();
            assert_eq!(later, vec![3, 4], "end {end}: {units:?}");
            assert!(pairs.contains(&&vec![24, end]), "{units:?}");
        }
        assert_eq!(pairs.len(), 6, "{units:?}");
    }

    #[test]
    fn pairs_off_the_other_track_hold_no_place() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet of three, and a day later a false tracklet of two which
        // crosses its track, nearer to it than the object's own sight then,
        // but whose own track misses the first tracklet. The pairs of the
        // false tracklet are refused by its track, and leave the places of
        // the first tracklet's ends to the object's sight.
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        sights.push(sight(1.0, at, [1.0, 1e-3, 1e-4]));
        sights.push(sight(1.0 + 10.0 * MINUTE, at, [1.0, 1e-3, 1.4e-4]));
        sights.push(sight(1.0, at, [1.0, 1e-3, -4.4e-4]));
        let units = sorted(form_units(&sights, 1.5, 0.01, 0.1, 2, false, 0.0).unwrap());
        assert_eq!(
            units,
            vec![vec![0, 1, 2], vec![0, 5], vec![2, 5], vec![3, 4]],
        );
    }

    #[test]
    fn parallax_margin_widens_the_pairs() {
        let at = [1.0, 0.0, 0.0];
        let track = |days: f64| [1.0, 1e-3 * days, 0.0];
        // A tracklet, and a day later a sight off its track by a parallax of
        // two thousandths of a radian, which a margin of that much pairs.
        let mut sights: Vec<Sight> = [0.0, 10.0, 20.0]
            .map(|m| sight(m * MINUTE, at, track(m * MINUTE)))
            .to_vec();
        sights.push(sight(1.0, at, [1.0, 1e-3, 2e-3]));
        let units =
            |margin: f64| sorted(form_units(&sights, 1.5, 0.01, 0.1, 2, false, margin).unwrap());
        assert_eq!(units(0.0), vec![vec![0, 1, 2]]);
        assert_eq!(units(2e-3), vec![vec![0, 1, 2], vec![0, 3], vec![2, 3]]);
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
            sorted(form_units(&sights, 1.5, 0.01, 0.1, min_tracklet, lone_pairs, 0.0).unwrap())
        };
        assert_eq!(units(2, true), vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
        // Below `min_tracklet` the tracklet is two lone sights, which have no
        // track, and pair with each other and with both later sights.
        let lone = vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3]];
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
            sorted(form_units(&sights, 1.5, 0.01, 0.1, 2, lone_pairs, 0.0).unwrap())
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
        let row = |side: &dyn Fn(u32) -> f64| -> Vec<Sight> {
            (0..6)
                .map(|k| {
                    let step = f64::from(k);
                    sight(10.0 * MINUTE * step, at, [1.0, 1e-4 * step, side(k)])
                })
                .collect()
        };
        let tracklets =
            |sights: &[Sight], min_tracklet: usize| form_tracklets(sights, 0.1, 0.03, min_tracklet);
        // A straight row is one tracklet.
        assert_eq!(tracklets(&row(&|_| 0.0), 2), vec![vec![0, 1, 2, 3, 4, 5]]);
        // A last sight off the track by 1e-4, many times what it allows, is
        // lone.
        let off = |at: u32| move |k: u32| if k == at { 1e-4 } else { 0.0 };
        assert_eq!(tracklets(&row(&off(5)), 2), vec![vec![0, 1, 2, 3, 4]]);
        // One off it within the row is left out and the tracklet goes on.
        assert_eq!(tracklets(&row(&off(3)), 2), vec![vec![0, 1, 2, 4, 5]]);
        // So are two in a row, which are a tracklet of two of their own.
        let two_off = |k: u32| if k == 3 || k == 4 { 1e-4 } else { 0.0 };
        assert_eq!(
            tracklets(&row(&two_off), 2),
            vec![vec![0, 1, 2, 5], vec![3, 4]]
        );
        assert_eq!(tracklets(&row(&two_off), 3), vec![vec![0, 1, 2, 5]]);
    }

    #[test]
    fn tracklets_split_where_the_place_changes() {
        // An object 0.5 AU away seen from one site three times over twenty
        // minutes, then from a second site 6000 km away three times, and
        // from the first site a day later, with one more sight then off the
        // track by five thousandths of a radian. The parallax between the
        // sites is many times the noise, so the six sights are cut into two
        // where the site changes, and the late sight pairs with both ends of
        // each tracklet along its own track. Of the ends of the two tracklets,
        // only the two 45 minutes apart pair: over that time the slack of the
        // tracks covers the parallax, over the minutes between the others it
        // does not.
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
            |sights: &[Sight]| sorted(form_units(sights, 1.5, 0.05, 0.05, 2, true, 0.0).unwrap());
        let expected = vec![
            vec![0, 1, 2],
            vec![0, 5],
            vec![0, 6],
            vec![2, 6],
            vec![3, 4, 5],
            vec![3, 6],
            vec![5, 6],
        ];
        assert_eq!(units(&sights), expected);

        // From the first site alone the six are one tracklet.
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
        let mut units = form_units(&sights, 4.0, 1.1 * rate, 0.05, 2, true, 0.0).unwrap();
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
        let mut units = form_units(&sights, 1.5, 0.01, 0.1, 2, true, 0.0).unwrap();
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
        let mut units = form_units(&sights, 1.5, 0.01, 0.0, 2, true, 0.0).unwrap();
        units.sort();
        let mut pairs: Vec<Vec<usize>> = form_pairs(&sights, 1.5, 0.01, 0.0, |_, _| true)
            .into_iter()
            .map(|(i, j)| vec![i, j])
            .collect();
        pairs.sort();
        assert_eq!(units, pairs);

        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(form_units(&sights, 1.5, 0.01, bad, 2, true, 0.0).is_err());
        }
        // Each argument is named in its own error.
        let message = |result: KeteResult<Vec<Vec<usize>>>| result.unwrap_err().to_string();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(message(form_units(&sights, 1.5, bad, 0.1, 2, true, 0.0)).contains("max_rate"));
        }
        assert!(message(form_units(&sights, -1.0, 0.01, 0.1, 2, true, 0.0)).contains("max_gap"));
        assert!(
            message(form_units(&sights, 1.5, 0.01, 0.1, 2, true, f64::NAN))
                .contains("parallax_margin")
        );
        assert!(
            message(form_units(&sights, f64::NAN, 0.01, 0.1, 2, true, 0.0)).contains("max_gap")
        );
        let mut exact = sights.clone();
        exact[2].sigma = 0.0;
        assert!(message(form_units(&exact, 1.5, 0.01, 0.1, 2, true, 0.0)).contains("sight 2"));
        for bad in [0, 1] {
            let result = form_units(&sights, 1.5, 0.01, 0.1, bad, true, 0.0);
            assert!(message(result).contains("min_tracklet"));
        }
    }
}
