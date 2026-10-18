// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Searches in passes: each pass forms units, builds a grid of hypotheses,
//! links and confirms, over the observations no earlier pass confirmed.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use kete_core::constants::GMS;
use kete_core::desigs::Desig;
use kete_core::frames::{SunCenter, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_fitting::AstrometricObservation;
use rayon::prelude::*;

use crate::calibration::{Scaled, station_factors};
use crate::confirm::{accept_disjoint, confirm_with, rank, sight_from_observation, sun_centered};
use crate::grid::grid_layers;
use crate::link::check_unit_count;
use crate::merge::merge;
use crate::sight::first_and_last;
use crate::tiles::tiles;
use crate::units::check_limits;
use crate::{
    Candidate, ConfirmRules, Confirmed, DEPTH_STEP, LinkSettings, RATE_STEP, Sight, form_units,
    hypothesis_grid, link,
};

/// One pass of [`search()`]: its rate limit and the steps of its grid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchPass {
    /// Factor which multiplies both [`DEPTH_STEP`] and [`RATE_STEP`] for the
    /// grid of the pass.
    pub grid_factor: f64,

    /// Largest apparent rate of motion within a tracklet and of a pair, in
    /// radians per day, finite and positive, see [`form_units`].
    ///
    /// Pairing does not compare observers, so for the pairs of sights from
    /// observers at different places to form, the rate has to cover the
    /// parallax between those observers over the time between the sights of
    /// a pair.
    ///
    /// In [`crate::search_windows`] the rate is also how fast the halo of a
    /// tile grows. The parallax between observers is not added to the rate
    /// for the tiles: it goes in the `parallax_margin` of
    /// [`crate::search_windows`].
    pub max_rate: f64,
}

/// Settings of [`search()`].
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSettings {
    /// Smallest time between the sights of a pair, in days, see
    /// [`form_units`]. The largest is the `max_gap` of [`Self::confirm`].
    pub min_gap: f64,

    /// Longest tracklet, in days, see [`form_units`]. A little longer than
    /// the time between the revisits of a field within a night, so that the
    /// detections of one object in a night form a tracklet, see
    /// [`SearchSettings::default`].
    pub tracklet_span: f64,

    /// Fewest sights of a tracklet, at least 2, see [`form_units`]. Shorter
    /// tracklets are left out, and their sights are lone. Two, the default,
    /// keeps every tracklet.
    ///
    /// Two false detections within `tracklet_span` and the rate limit of each
    /// other form a tracklet of two as readily as an object does, and a chain
    /// of them breaks into such short pieces. Three or more leaves those out,
    /// for a survey which sees each object three or more times in a visit,
    /// such as a spacecraft which revisits a field every orbit, in data with
    /// many false detections. An object seen fewer times than this in a visit
    /// has no tracklet there, and is found only through the pairs of those
    /// sights, see [`Self::lone_pairs`].
    pub min_tracklet: usize,

    /// Whether two lone sights, those in no tracklet, may be paired, see
    /// [`form_units`]. False, the default, does not pair them.
    ///
    /// A lone sight pairs with every sight within the rate limit, not only
    /// along the track of a tracklet, so where most sights are false
    /// detections, which seldom form tracklets, those pairs grow as the square
    /// of their density and outnumber every other unit, see
    /// [`crate::MAX_UNITS`]. False leaves them out: every pair then holds the
    /// end of a tracklet and lies on its track, which suits any survey that
    /// sees each object at least [`Self::min_tracklet`] times in a visit,
    /// such as one on the ground which takes its visits in pairs, or a
    /// spacecraft which revisits a field every orbit. An object seen fewer
    /// times than that in a visit is then found only through the pairs of
    /// those sights with the ends of its tracklets in other visits, and an
    /// object with no tracklet is not found. True suits a survey which sees
    /// most objects once per visit, in data with few false detections.
    pub lone_pairs: bool,

    /// Smallest depth of the hypotheses, in AU, see [`hypothesis_grid`].
    pub min_depth: f64,

    /// Largest depth of the hypotheses, in AU.
    pub max_depth: f64,

    /// The passes, in the order they run, at least one, see [`SearchPass`].
    pub passes: Vec<SearchPass>,

    /// Settings of [`link()`], the same in every pass.
    pub link: LinkSettings,

    /// Rules of [`crate::confirm()`], the same in every pass. Their
    /// `min_sights` is also the fewest sights of a candidate of [`link()`], and
    /// their `max_gap` the largest time between the sights of a pair, see
    /// [`form_units`].
    pub confirm: ConfirmRules,
}

impl Default for SearchSettings {
    /// Tracklets up to an eighth of a day long, time groups parted at gaps of
    /// more than a tenth of a day, pairs half a day to a day and a half apart,
    /// motion up to 0.6 degrees a day in the first pass and a degree a day in
    /// the second, and depths from 0.5 to 4.5 AU. These suit main belt objects
    /// seen by surveys on the ground in tracklets within a night; other
    /// cadences and populations need their own.
    ///
    /// A degree a day is about twice the rate of the fastest main belt
    /// objects, the Hungarias, near opposition. A larger one finds no more
    /// main belt objects; it costs little, as the ends of tracklets are paired
    /// only along their tracks, but the pairs of lone sights grow as its
    /// square.
    ///
    /// A tracklet of an eighth of a day holds the detections of one object
    /// from one site in a night as one unit, for surveys which revisit a
    /// field minutes apart and for those which revisit it two to three hours
    /// apart, and its first and last detection are also paired across nights
    /// along its track. A sight whose revisit falls just past the tracklet
    /// span is lone and pairs with every sight in reach, so a span shorter
    /// than a survey's revisits costs time and objects in that survey's
    /// fields. A longer span also chains the visits of two sites a few hours
    /// apart in one night, which part into two tracklets where their parallax
    /// is larger than the track allows, and otherwise join into one, which
    /// leaves the link fewer units of those objects. So the span is chosen for
    /// the cadence: a little longer than the longest time between revisits of
    /// a field within a night. It is not taken from the sights themselves: in
    /// observations of several surveys the times between revisits follow the
    /// mix of surveys, which changes from field to field, and their tail is
    /// set by chance neighbors, which grow with the density.
    ///
    /// An object close to the observer can move too much within a tracklet
    /// for one orbit to fit it, and is then found through the pairs of its
    /// ends alone; a search aimed at objects within a few hundredths of an AU
    /// takes tracklets and time groups of a hundredth of a day or less, see
    /// the tests `units_close_observers` and `matrix_close_objects`.
    ///
    /// Detections no more than a tenth of a day after the one before are in one
    /// time group of confirmation, a span set apart from that of tracklets:
    /// tracklets only have to join the revisits of one object, while time
    /// groups decide how many independent measurements of the motion a
    /// confirmation has. Longer groups merge visits a few hours apart into one,
    /// and turn down objects seen in those visits and once more beyond
    /// `max_gap`.
    ///
    /// The first pass is at one and a half times [`DEPTH_STEP`] and
    /// [`RATE_STEP`]. The number of hypotheses of a grid, and with it the cost
    /// of a pass, goes about as the inverse of the product of the two steps, so
    /// this coarse grid is the cheap one. It still finds most main belt and
    /// more distant objects: a tracklet averages the noise of its sights, so
    /// its orbit under a hypothesis a step off still matches the other units of
    /// its object. The second pass is at three tenths of the steps, over the
    /// observations the first did not confirm, and recovers faster objects,
    /// near-Earth objects and objects the coarse grid split. Its grid has many
    /// times the hypotheses of the first, so it can take the larger part of the
    /// time to link even over fewer observations, and a finer second pass
    /// multiplies that time again. The test `search_in_passes` shows the second
    /// pass recovering what the first misses.
    ///
    /// The rate limit of the first pass is what keeps the halos of the tiles
    /// of [`crate::search_windows`] small in the pass which confirms most
    /// objects. It is above the rates of nearly all main belt objects, so
    /// that few cross it within a window: an object whose rate crosses the
    /// limit of a pass can be confirmed by that pass in a piece, and the rest
    /// of its observations by a later pass as another. Neither grid alone
    /// finds everything the other finds, so a first pass whose limit leaves
    /// many objects to the fine pass alone also finds fewer of them.
    ///
    /// Four observations, two on each side of their largest gap, and a
    /// residual of twice the uncertainties confirm, with the defaults of
    /// [`LinkSettings`]; a candidate needs as many observations as a
    /// confirmation keeps.
    fn default() -> Self {
        Self {
            min_gap: 0.5,
            tracklet_span: 0.125,
            min_tracklet: 2,
            lone_pairs: false,
            min_depth: 0.5,
            max_depth: 4.5,
            passes: vec![
                SearchPass {
                    grid_factor: 1.5,
                    max_rate: 0.6_f64.to_radians(),
                },
                SearchPass {
                    grid_factor: 0.3,
                    max_rate: 1.0_f64.to_radians(),
                },
            ],
            link: LinkSettings::default(),
            confirm: ConfirmRules::default(),
        }
    }
}

/// Link and confirm the observations in passes, each over the observations
/// no earlier pass confirmed.
///
/// The lines of sight of the optical observations are made once. Each pass
/// forms the units of those remaining with the rate limit of the pass,
/// builds the grid of hypotheses with the steps of the pass, links, and
/// confirms. The observations it confirms are then removed. Only confirmed
/// ones are: before confirmation most candidates are false or mix objects,
/// so a later pass sees every observation which was merely in a candidate.
///
/// Each pass takes as the origin of its hypotheses the observer of the
/// sight of a unit nearest the middle of the time the sights of its units
/// span, moved to the Sun, and the time of that sight as the reference time.
/// That holds for any observer and any geometry: depth is measured from a
/// place the objects were seen from, so it is positive for nearly every
/// object, as the logarithmic grid needs, and the reference time lies within
/// the span of the sights of the units, as [`link()`] needs. Taking it from the
/// units of each pass keeps that true when a later pass has fewer sights. A
/// pass with no units is skipped.
///
/// Every setting is checked before the first pass, so that a bad setting of
/// a later pass fails before the work of the earlier ones. The size of the
/// grid of each pass is checked about the observer of the first optical
/// observation, or about a point a unit from the Sun if there is none; the
/// grids of the passes are about other observers, whose grids hold about as
/// many hypotheses.
///
/// The confirmed objects of all passes are then merged where several are
/// pieces of one object, see [`crate::merge_pieces`]: a pass can confirm
/// part of an object's observations, and a later pass the part the first
/// left out.
///
/// Radar observations are passed over.
///
/// # Stations
///
/// Every rule of confirmation is in units of the stated uncertainties of the
/// observations, so the objects of an observer whose stated uncertainties
/// are smaller than the scatter of its observations fail `max_rms`, or lose
/// observations to the outlier rule. `stations`, the label of the station of
/// each observation, such as its observatory code, lets the search scale the
/// uncertainties of each station to the scatter its observations show.
///
/// The factor of each station is estimated once, from the objects the first
/// pass confirms: each optical observation their fits keep adds the squares
/// of its residuals over its stated uncertainties on the two axes of the
/// sky, and `2 (1 - 3 / n)` to what is expected, for a fit which keeps `n`,
/// as the six parameters of the orbit absorb six of its `2 n` residuals. The
/// factor is the square root of their ratio, for a station with at least 100
/// observations kept by the fits of at least 20 objects, and at least one,
/// so that uncertainties only grow. Each other station keeps its stated
/// uncertainties. The residuals are those of the fits over the stated
/// uncertainties of the observations. The
/// candidates of the first pass are then confirmed again with both axes of
/// the sky of every observation of each station scaled by its factor,
/// without linking again, and those confirmations replace the first. The
/// later passes confirm with the scaled uncertainties, and so does the merge
/// of the pieces.
///
/// Linking always uses the stated uncertainties: the reach of a unit follows
/// from them, and scaling them pushes units over the `max_uncertainty` of
/// [`LinkSettings`] and changes the groups, which loses objects. The first
/// pass confirms only objects whose fits pass `max_rms` with the stated
/// uncertainties, so where those are far smaller than the scatter the factor
/// is an underestimate. The fits of the confirmed objects are made with the
/// scaled uncertainties, which their observations carry, so their residuals
/// and `rms` are in those units. Without `stations` every uncertainty is as
/// stated.
///
/// # Arguments
///
/// * `observations` - The observations to search.
/// * `stations` - The label of the station of each observation, such as its
///   observatory, one per observation, see above. `None` keeps the stated
///   uncertainties.
/// * `settings` - See [`SearchSettings`].
///
/// # Returns
///
/// The confirmed objects of every pass, those of the first pass first, as by
/// [`crate::confirm()`], with indices into `observations`, and merged objects
/// in the place of their first piece. No two share an observation.
///
/// # Errors
///
/// [`Error::ValueError`] for settings which [`form_units`],
/// [`hypothesis_grid`], [`link()`] or [`crate::confirm()`] refuse, if there are
/// no passes, if a pass forms more than [`crate::MAX_UNITS`] units, if `stations` does not hold one label per observation, or
/// naming the first optical observation whose line of sight is not
/// [`Sight::is_usable`]. Fails if the loaded kernels do not cover the Sun at
/// the epoch of an optical observation, or do not cover the Sun, the planets
/// and the Moon, which the orbit fit pulls with, at the first and the last of
/// the epochs of the candidates of a pass and of their observations, see
/// [`crate::confirm()`].
pub fn search(
    observations: &[AstrometricObservation],
    stations: Option<&[usize]>,
    settings: &SearchSettings,
) -> KeteResult<Vec<Confirmed>> {
    check_settings(observations, stations, settings)?;
    let sights = optical_sights(observations, "Search")?;
    let members = (0..observations.len())
        .filter(|&i| sights[i].is_some())
        .collect();
    let (found, factors) = in_passes(observations, &sights, stations, members, settings, None)?;
    let scaled = Scaled {
        sights: &sights,
        factors: &factors,
    };
    merge(observations, scaled, found, &settings.confirm)
}

/// The errors of the settings of a search, checked before its first pass so
/// that a bad setting of a later pass, or of a pass with no units, fails at
/// once: [`Error::ValueError`] if there are no passes, which would search
/// nothing, or if `stations` is given and not one per observation, those of
/// the rules of confirmation and of the settings of [`link()`], and of the
/// limits of [`form_units`] and the inputs of [`hypothesis_grid`] for each
/// pass. The size of the grid of each pass is checked about the observer of
/// the first optical observation moved to the Sun, see [`search()`], or about a
/// point on a circular orbit a unit from the Sun if there is none.
pub(crate) fn check_settings(
    observations: &[AstrometricObservation],
    stations: Option<&[usize]>,
    settings: &SearchSettings,
) -> KeteResult<()> {
    if settings.passes.is_empty() {
        return Err(Error::ValueError(
            "Search: there must be at least one pass".into(),
        ));
    }
    if let Some(stations) = stations
        && stations.len() != observations.len()
    {
        return Err(Error::ValueError(format!(
            "Search: stations must hold one label per observation, got {} for {} observations",
            stations.len(),
            observations.len()
        )));
    }
    settings.confirm.check()?;
    settings.link.check()?;
    let first = observations
        .iter()
        .find(|observation| !observation.is_radar());
    let origin = match first {
        Some(observation) => sun_centered(&observation.observer()?)?,
        None => State::new(
            Desig::Empty,
            2_451_545.0,
            Vector::new([1.0, 0.0, 0.0]),
            Vector::new([0.0, GMS.sqrt(), 0.0]),
            SunCenter,
        ),
    };
    for pass in &settings.passes {
        check_limits(
            settings.min_gap,
            settings.confirm.max_gap,
            pass.max_rate,
            settings.tracklet_span,
            settings.min_tracklet,
        )?;
        let (depth_step, rate_step) = (DEPTH_STEP * pass.grid_factor, RATE_STEP * pass.grid_factor);
        let (min_depth, max_depth) = (settings.min_depth, settings.max_depth);
        let _ = grid_layers(&origin, min_depth, max_depth, depth_step, rate_step)?;
    }
    Ok(())
}

/// The confirmed objects of the passes of a search over the observations at
/// `members`, each pass over those no earlier pass confirmed, not yet merged,
/// and the factor on the uncertainties of each observation, see [`search()`].
/// `sights` holds the line of sight of every optical observation, and every
/// member is optical.
///
/// Each pass is searched in tiles of the sky: with `tiling`, its parallax
/// margin and `nside`, the [`tiles()`] of the members with the rate limit of
/// the pass, whose hypotheses all aim at the middle of the time the members
/// span, see [`crate::search_windows`]; without it, one tile of every member.
/// A tile with fewer members than a confirmation keeps is skipped. The
/// objects confirmed in the tiles of a pass are gathered, and of objects
/// which share an observation one is kept by the overlap rule of
/// [`crate::confirm()`], ties going to the earlier tile, see
/// [`Gathered::merge`].
///
/// With `stations`, the first pass estimates the factors from its confirmed
/// objects, and its candidates are confirmed again with them if any is above
/// one. Every later pass confirms with them.
pub(crate) fn in_passes(
    observations: &[AstrometricObservation],
    sights: &[Option<Sight>],
    stations: Option<&[usize]>,
    mut members: Vec<usize>,
    settings: &SearchSettings,
    tiling: Option<(f64, Option<u32>)>,
) -> KeteResult<(Vec<Confirmed>, Vec<f64>)> {
    let rules = &settings.confirm;
    let mut factors = vec![1.0; observations.len()];
    let mut found = Vec::new();
    for (n, pass) in settings.passes.iter().enumerate() {
        let calibrating = if n == 0 { stations } else { None };
        let (tiles, target) = match tiling {
            None => (vec![members.clone()], None),
            Some((parallax_margin, nside)) => {
                let own: Vec<Sight> = members.iter().filter_map(|&i| sights[i]).collect();
                let (first, last) = first_and_last(own.iter().map(|sight| sight.time.jd()));
                let tiles = tiles(&own, pass.max_rate, parallax_margin, nside, None)?
                    .into_iter()
                    .map(|tile| tile.into_iter().map(|k| members[k]).collect())
                    .collect();
                (tiles, Some(f64::midpoint(first, last)))
            }
        };
        // The candidates of the tiles are kept only to confirm them again
        // with the factors the pass estimates.
        let mut kept = Vec::new();
        let mut gathered = Gathered::default();
        for tile in tiles {
            if tile.len() < rules.min_sights {
                continue;
            }
            let candidates = pass_candidates(observations, sights, &tile, pass, settings, target)?;
            let scaled = Scaled {
                sights,
                factors: &factors,
            };
            gathered.add(confirm_with(observations, scaled, &candidates, rules)?);
            if calibrating.is_some() {
                kept.push(candidates);
            }
        }
        let mut confirmed = gathered.merge();
        if let Some(stations) = calibrating {
            factors = station_factors(observations, stations, &confirmed);
            if factors.iter().any(|&factor| factor > 1.0) {
                let scaled = Scaled {
                    sights,
                    factors: &factors,
                };
                let mut gathered = Gathered::default();
                for candidates in &kept {
                    gathered.add(confirm_with(observations, scaled, candidates, rules)?);
                }
                confirmed = gathered.merge();
            }
        }
        let mut taken = vec![false; observations.len()];
        for &index in confirmed.iter().flat_map(|c| &c.sights) {
            taken[index] = true;
        }
        members.retain(|&index| !taken[index]);
        found.extend(confirmed);
    }
    Ok((found, factors))
}

/// The candidates of one pass of a search over the observations at
/// `members`, all optical, with `sights` holding the line of sight of every
/// optical observation: their units with the rate limit of the pass, the
/// grid of hypotheses with its steps, and the link, with the stated
/// uncertainties, see [`search()`]. The candidates index `observations`.
///
/// The hypotheses are about the observer of the sight of a unit nearest
/// `target`, a Julian date, or nearest the middle of the span of the sights
/// of the units if `target` is `None`, moved to the Sun. A pass with no
/// units has no candidates, and one with more than [`crate::MAX_UNITS`] is
/// refused before its grid is built.
fn pass_candidates(
    observations: &[AstrometricObservation],
    sights: &[Option<Sight>],
    members: &[usize],
    pass: &SearchPass,
    settings: &SearchSettings,
    target: Option<f64>,
) -> KeteResult<Vec<Candidate>> {
    let own: Vec<Sight> = members.iter().filter_map(|&i| sights[i]).collect();
    debug_assert_eq!(own.len(), members.len(), "every member is optical");
    let units = form_units(
        &own,
        settings.min_gap,
        settings.confirm.max_gap,
        pass.max_rate,
        settings.tracklet_span,
        settings.min_tracklet,
        settings.lone_pairs,
    )?;
    check_unit_count("Search", units.len())?;
    let target = target.unwrap_or_else(|| {
        let (first, last) = first_and_last(units.iter().flatten().map(|&i| own[i].time.jd()));
        f64::midpoint(first, last)
    });
    let Some(nearest) = nearest(&own, &units, target) else {
        return Ok(Vec::new());
    };
    let origin = sun_centered(&observations[members[nearest]].observer()?)?;
    let grid = hypothesis_grid(
        &origin,
        settings.min_depth,
        settings.max_depth,
        DEPTH_STEP * pass.grid_factor,
        RATE_STEP * pass.grid_factor,
    )?;
    let min_sights = settings.confirm.min_sights;
    let mut candidates = link(&own, &units, &grid, min_sights, &settings.link)?;
    for index in candidates.iter_mut().flat_map(|c| &mut c.sights) {
        *index = members[*index];
    }
    Ok(candidates)
}

/// The confirmed objects of the tiles of a pass of a search, with sights
/// indexing the observations, sorted, see [`Gathered::merge`].
///
/// Of objects with the same observations only the first in the order of the
/// merge is held. That changes nothing: the merge drops every other one,
/// either for the first or for what drops the first, which shares an
/// observation with all of them. An object found whole in several tiles is so
/// held once, which keeps memory to the distinct objects of the tiles.
#[derive(Debug, Default)]
pub(crate) struct Gathered {
    /// Each distinct set of observations, with the order it arrived in and
    /// its confirmed object.
    held: HashMap<Vec<usize>, (usize, Confirmed)>,

    /// Confirmed objects added so far.
    added: usize,
}

impl Gathered {
    /// Add confirmed objects, in the order given.
    pub(crate) fn add(&mut self, objects: impl IntoIterator<Item = Confirmed>) {
        for object in objects {
            let order = self.added;
            self.added += 1;
            match self.held.entry(object.sights.clone()) {
                Entry::Occupied(mut held) => {
                    if rank(&object, &held.get().1).is_lt() {
                        let _ = held.insert((order, object));
                    }
                }
                Entry::Vacant(free) => {
                    let _ = free.insert((order, object));
                }
            }
        }
    }

    /// The objects held without overlaps, by the overlap rule of
    /// [`crate::confirm()`], with ties going to the object added earlier:
    /// in that order, each one which shares an observation with one before
    /// it is dropped. An object confirmed in several tiles is kept once,
    /// and pieces of it confirmed in tiles which held only part of it give
    /// way to the whole. An object which gives way is dropped whole, not cut
    /// to the observations it does not share: those are in no object of the
    /// merge, and remain for the later passes of the search.
    pub(crate) fn merge(self) -> Vec<Confirmed> {
        let mut all: Vec<(usize, Confirmed)> = self.held.into_values().collect();
        all.sort_by(|(i, a), (j, b)| rank(a, b).then(i.cmp(j)));
        let size = all
            .iter()
            .flat_map(|(_, object)| &object.sights)
            .max()
            .map_or(0, |&i| i + 1);
        let mut taken = vec![false; size];
        let mut accepted = Vec::new();
        accept_disjoint(
            all.into_iter().map(|(_, object)| object),
            &mut taken,
            &mut accepted,
        );
        accepted
    }
}

/// The line of sight of each optical observation, and `None` for each radar
/// observation.
///
/// # Errors
///
/// [`Error::ValueError`] naming, after `caller`, the first optical
/// observation whose line of sight is not [`Sight::is_usable`]. Fails if the
/// loaded kernels do not cover the Sun at the epoch of one.
pub(crate) fn optical_sights(
    observations: &[AstrometricObservation],
    caller: &str,
) -> KeteResult<Vec<Option<Sight>>> {
    observations
        .par_iter()
        .enumerate()
        .map(|(index, observation)| {
            if observation.is_radar() {
                return Ok(None);
            }
            let sight = sight_from_observation(observation).map_err(|err| {
                if let Error::ValueError(message) = err {
                    Error::ValueError(format!("{caller}: observation {index}: {message}"))
                } else {
                    err
                }
            })?;
            Ok(Some(sight))
        })
        .collect()
}

/// The sight of a unit nearest the Julian date `target`, of the lowest index
/// among equals. `None` if there are no units.
fn nearest(sights: &[Sight], units: &[Vec<usize>], target: f64) -> Option<usize> {
    let off = |index: usize| (sights[index].time.jd() - target).abs();
    let paired = units.iter().flatten().copied();
    paired.min_by(|&a, &b| off(a).total_cmp(&off(b)).then(a.cmp(&b)))
}
