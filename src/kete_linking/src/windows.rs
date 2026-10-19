// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! A search over a whole survey, one tile of the sky at a time, optionally
//! in windows of time.

use kete_core::prelude::{Error, KeteResult};
use kete_fitting::AstrometricObservation;

use crate::calibration::Scaled;
use crate::merge::merge;
use crate::search::{check_settings, in_passes, optical_sights};
use crate::{Confirmed, SearchSettings, Sight, tiles};

/// Most windows [`search_windows`] searches. A survey of decades in windows
/// of a few days needs some thousands; more than this is a window far
/// shorter than any cadence calls for, and would only loop over empty
/// windows.
const MAX_WINDOWS: f64 = 1e5;

/// [`crate::search()`] over a whole survey, one tile of the sky at a time, in
/// windows of time.
///
/// # Tiles
///
/// Each pass of the settings runs over the optical observations no earlier pass
/// confirmed, cut into [`crate::tiles()`] with the `max_rate` of that pass,
/// the `parallax_margin` of the settings and `nside`, about the middle of the time those
/// observations span. Each tile with at least as many observations as a
/// confirmation keeps is searched in turn with that pass alone. Memory so holds
/// the lines of sight of the observations, made once, the search of one tile,
/// which uses every thread, and the distinct objects the pass has confirmed so
/// far. Of the objects of all tiles of the pass which share an observation, one
/// is kept by the overlap rule of [`crate::confirm()`], with ties going to the
/// earlier tile. An object confirmed in several tiles is so kept once, and
/// pieces of it confirmed in tiles which held only part of it give way to the
/// whole. Their observations are removed before the next pass; the observations
/// of an object which gave way are not, so the later passes see them. Radar
/// observations are passed over.
///
/// The hypotheses of every tile of a pass aim at that middle time: each tile
/// takes as origin the observer of the sight of one of its units nearest
/// that time, moved to the Sun, at the time of that sight. That sight lies
/// within the span of the units of the tile, as [`crate::link()`] needs. A
/// tile with no units is skipped.
///
/// Every object whose direction at that time lies in a core and whose
/// apparent rate is at most the `max_rate` of a pass has all of its
/// remaining observations in one tile of that pass, so the search of that
/// tile can find it. Tiles find what a search of their own observations
/// finds, which differs a little from what a search of all observations
/// finds: the units, origin and grid of a tile are its own. The halo grows
/// with the rate limit, so a first pass with a small one confirms the slow
/// objects, most of a survey, in small tiles, and leaves a later pass with
/// the full rate limit, and large tiles, only what it did not confirm.
/// Tiling costs more than a search of all the observations at once, as a
/// sight in the halos of several tiles is searched again in each of them;
/// the test `space_and_ground_survey` prints the time of both. It is for
/// surveys whose search does not fit in memory.
///
/// With `stations`, the factors on the uncertainties of the stations are
/// estimated as by [`crate::search()`], from the objects the first pass
/// confirms in all of its tiles, as merged. The candidates of every tile of
/// the first pass are kept until then, which memory then also holds, and
/// are confirmed again with the scaled uncertainties. The later passes and
/// the merge of the pieces confirm with the scaled uncertainties, and every
/// pass links with the stated ones.
///
/// # Windows
///
/// The halo of a tile is the rate limit times the time from its reference
/// time, so the tiles of a long span overlap much, and the largest of them
/// hold more observations than fit in memory. Windows a few weeks long or
/// less keep the halos, and the tiles, small whatever the span of the
/// survey. Without `window` the whole span is one window.
///
/// The windows are `2 * window` days long, and start at the first optical
/// observation and every `window` days after it, so that each overlaps the
/// next by half. The last starts within `2 * window` of the last optical
/// observation and holds everything from its start on.
///
/// Any stretch of `window` days within the span of the observations then
/// lies whole in at least one window, wherever it falls. Take the last
/// window which starts at or before the start of the stretch. If it is the
/// last window, it has no end. Otherwise the next window starts after the
/// stretch starts, so this one starts less than `window` days before the
/// stretch, and as it is `2 * window` days long it ends after the stretch
/// does.
///
/// The windows are searched in turn, in tiles as above, each with the halos
/// and hypotheses of the middle of its own observations, over the
/// observations no earlier window confirmed, as the passes of a search each
/// run over what the earlier ones left. The lines of sight of
/// the observations are made once for all windows, and the tiles of every
/// pass are checked for the longest window before the first is searched.
///
/// An object whose observations, those it needs to be confirmed, all lie
/// within `window` days therefore has them all in some window. It is found
/// there whole, or an earlier window which held part of it confirmed that
/// part, and the window holds the rest, which is confirmed as another piece
/// if it can be. `window` is at least the longest time a unit spans, the
/// larger of the `max_gap` of the confirmation rules and `tracklet_span`, so
/// every unit of the search also lies whole in some window. Nothing is
/// promised for an object whose observations spread over more than
/// `window` days: it is found only where a window holds enough of it to
/// confirm, such as an object seen on three nights a week apart, which a
/// search of the whole span finds and windows of less than two weeks may
/// not. The window is therefore chosen for the cadence of the survey, at
/// least the time over which most of its objects collect a confirmation.
///
/// Pieces of one object confirmed in different windows share no
/// observation. Versions which shared observations would have to be
/// resolved by the overlap rule of [`crate::confirm()`], which keeps one whole
/// and drops the rest: the version of a middle window would win, and the
/// parts of the object in the windows on either side, too few to confirm
/// alone, would be lost. The pieces of all windows and passes are merged
/// once, at the end, where several are one object, by
/// [`crate::merge_pieces`], whose prediction test carries the uncertainty of
/// each piece's orbit to the times of the other, so that it reaches across
/// windows, passes and tiles alike.
///
/// With `stations`, each window estimates its own factors from the objects
/// its own first pass confirms: the scatter of an observer can change over a
/// long survey, and a window holds the observations its factors are applied
/// to. The final merge sees each observation of a piece with the factor of
/// the window which confirmed that piece.
///
/// # Arguments
///
/// * `observations` - The observations to search.
/// * `stations` - The label of the station of each observation, see
///   [`crate::search()`]. `None` keeps the stated uncertainties.
/// * `settings` - See [`SearchSettings`].
/// * `window` - Time between the starts of consecutive windows, in days, half
///   the length of each. Finite, and at least the larger of the `max_gap` of
///   the confirmation rules and `tracklet_span`. `None` searches the whole span
///   as one window.
/// * `nside` - See [`crate::tiles()`], the same for every pass and window.
///   `None` takes it for each pass from the halo of that pass.
///
/// # Returns
///
/// The confirmed objects of every window, those of earlier windows first,
/// and within a window those of its first pass first, with indices into
/// `observations`, and merged objects in the place of their first piece. No
/// two share an observation.
///
/// # Errors
///
/// [`Error::ValueError`] if `window` is not finite and positive, is less
/// than the larger of the `max_gap` of the confirmation rules and
/// `tracklet_span`, or makes more than 100000 windows, and the errors of
/// [`crate::tiles()`], for the tiles of the longest window, and of
/// [`crate::search()`]. As there, fails if the loaded kernels do not cover
/// the Sun at the epoch of an optical observation, or do not cover the Sun,
/// the planets and the Moon, which the orbit fit pulls with, from a day
/// before the first to the last of the epochs of the candidates of a tile and of their
/// observations, see [`crate::confirm()`].
pub fn search_windows(
    observations: &[AstrometricObservation],
    stations: Option<&[usize]>,
    settings: &SearchSettings,
    window: Option<f64>,
    nside: Option<u32>,
) -> KeteResult<Vec<Confirmed>> {
    if let Some(window) = window
        && !(window.is_finite() && window > 0.0)
    {
        return Err(Error::ValueError(format!(
            "Search windows: window must be finite and positive, got {window}"
        )));
    }
    check_settings(observations, stations, settings)?;
    let unit = settings.confirm.max_gap.max(settings.tracklet_span);
    if let Some(window) = window
        && window < unit
    {
        return Err(Error::ValueError(format!(
            "Search windows: window must be at least the larger of max_gap and tracklet_span, \
             {unit}, got {window}"
        )));
    }
    let sights = optical_sights(observations, "Search windows")?;
    let time = |i: usize| sights[i].map_or(f64::NAN, |sight| sight.time.jd());
    let mut order: Vec<usize> = (0..observations.len())
        .filter(|&i| sights[i].is_some())
        .collect();
    order.sort_by(|&a, &b| time(a).total_cmp(&time(b)).then(a.cmp(&b)));
    let (Some(&first), Some(&last)) = (order.first(), order.last()) else {
        return Ok(Vec::new());
    };
    let (first, last) = (time(first), time(last));
    // The start of each window, the last of which has no end.
    let starts: Vec<f64> = match window {
        None => vec![first],
        Some(window) => {
            // The fewest windows whose last starts within two windows of the
            // end.
            let count = ((last - first) / window - 1.0).ceil().max(1.0);
            if count > MAX_WINDOWS {
                return Err(Error::ValueError(format!(
                    "Search windows: a window of {window} days makes {count} windows, more \
                     than {MAX_WINDOWS}, take a longer window"
                )));
            }
            #[allow(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "a whole number from 1 to MAX_WINDOWS"
            )]
            let count = count as usize;
            (0..count)
                .map(|k| (k as f64).mul_add(window, first))
                .collect()
        }
    };
    // The positions in `order` each window spans.
    let spans: Vec<(usize, usize)> = starts
        .iter()
        .enumerate()
        .map(|(k, &start)| {
            let from = order.partition_point(|&i| time(i) < start);
            let to = match window {
                Some(window) if k + 1 < starts.len() => {
                    order.partition_point(|&i| time(i) < 2.0_f64.mul_add(window, start))
                }
                _ => order.len(),
            };
            (from, to)
        })
        .collect();
    let longest = spans
        .iter()
        .filter(|(from, to)| from < to)
        .map(|&(from, to)| time(order[to - 1]) - time(order[from]))
        .fold(0.0, f64::max);
    if let Some(start) = sights[order[0]] {
        let end = Sight {
            time: start.time + longest,
            ..start
        };
        for pass in &settings.passes {
            let margin = settings.parallax_margin;
            let _ = tiles(&[start, end], pass.max_rate, margin, nside, None)?;
        }
    }
    let mut taken = vec![false; observations.len()];
    let mut found = Vec::new();
    // The factor of each observation of a confirmed object, from the window
    // which confirmed it.
    let mut factors = vec![1.0; observations.len()];
    for (from, to) in spans {
        let mut inside: Vec<usize> = order[from..to]
            .iter()
            .copied()
            .filter(|&i| !taken[i])
            .collect();
        if inside.len() < settings.confirm.min_sights {
            continue;
        }
        inside.sort_unstable();
        let (confirmed, own) = in_passes(
            observations,
            &sights,
            stations,
            inside,
            settings,
            true,
            nside,
        )?;
        for &i in confirmed.iter().flat_map(|object| &object.sights) {
            taken[i] = true;
            factors[i] = own[i];
        }
        found.extend(confirmed);
    }
    let scaled = Scaled {
        sights: &sights,
        factors: &factors,
    };
    merge(observations, scaled, found, &settings.confirm)
}
