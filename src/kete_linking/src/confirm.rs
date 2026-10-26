// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The boundary with the rest of kete: sights from observations, and
//! candidates confirmed by an orbit fit, with the rules of [`ConfirmRules`].

use kete_core::ephemeris::EphemerisSource;
use kete_core::forces::GravParams;
use kete_core::frames::{CenterBody, Equatorial, SSB, SunCenter, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_fitting::{AstrometricObservation, OrbitFit, fit_orbit};
use kete_spice::prelude::SpiceEphemeris;
use rayon::prelude::*;

use crate::calibration::Scaled;
use crate::sight::{UNUSABLE, first_and_last};
use crate::two_body_fit::{fewest_kept, fit_two_body};
use crate::{Candidate, Sight};

/// Most iterations of each pass of the orbit fit.
const FIT_ITERATIONS: usize = 50;

/// Outlier threshold of the orbit fit, as a multiple of the expected
/// chi-square of one observation. Nine rejects an observation off by about
/// three sigma on both axes at once. The two-body fit that starts the orbit
/// fit drops sights by the same rule.
pub(crate) const FIT_REJECT_THRESHOLD: f64 = 9.0;

/// Most outlier rejection passes of the orbit fit.
const FIT_REJECT_PASSES: usize = 3;

/// Least motion of the observations of a confirmed candidate, as the
/// largest difference in direction between two of them over their combined
/// uncertainty. A source fixed on the sky passes five with probability about
/// `exp(-12.5)`, four in a million, for each two of its observations.
const MIN_MOTION: f64 = 5.0;

/// Time before the first epoch over which the planets must be known, in
/// days, as the orbit fit propagates to the time the light left the object.
/// A day is the light time from about 170 AU.
const LIGHT_TIME: f64 = 1.0;

/// A state moved from the Solar System Barycenter to the Sun, with the
/// loaded ephemeris. An observer's state this way is an origin for
/// hypotheses.
///
/// # Arguments
///
/// * `state` - State with respect to the Solar System Barycenter.
///
/// # Errors
///
/// Fails if the loaded kernels do not cover the Sun at the epoch.
pub(crate) fn sun_centered(
    state: &State<Equatorial, SSB>,
) -> KeteResult<State<Equatorial, SunCenter>> {
    SpiceEphemeris::loaded()?.try_to_sun(state.clone().into())
}

/// The line of sight of an optical observation.
///
/// The direction is the observed right ascension and declination. The
/// observer moves from the Solar System Barycenter to the Sun with the
/// loaded ephemeris. The uncertainty is the root mean square of the errors
/// on the two axes of the sky, so the total variance is kept, and a
/// correlation between them is ignored.
///
/// # Arguments
///
/// * `observation` - An optical observation.
///
/// # Errors
///
/// [`Error::ValueError`] for a radar observation, or for a line of sight
/// which is not [`Sight::is_usable`], such as one of an observation without
/// a positive uncertainty. Fails if the loaded kernels do not cover the Sun
/// at the epoch.
pub fn sight_from_observation(observation: &AstrometricObservation) -> KeteResult<Sight> {
    let AstrometricObservation::Optical {
        observer,
        ra,
        dec,
        sigma_ra,
        sigma_dec,
        ..
    } = observation
    else {
        return Err(Error::ValueError(
            "Sight from observation: only optical observations have a line of sight".into(),
        ));
    };
    let across = sigma_ra * dec.cos();
    let sight = Sight {
        time: observer.epoch,
        observer: sun_centered(observer)?.pos,
        direction: Vector::from_ra_dec(*ra, *dec),
        sigma: f64::midpoint(across * across, sigma_dec * sigma_dec).sqrt(),
    };
    if !sight.is_usable() {
        return Err(Error::ValueError(format!(
            "Sight from observation: the line of sight has {UNUSABLE}"
        )));
    }
    Ok(sight)
}

/// A candidate which an orbit fit confirmed.
#[derive(Debug)]
pub struct Confirmed {
    /// Indices of the observations the fit kept, sorted.
    pub sights: Vec<usize>,

    /// The orbit fit, at the middle of the time its optical observations span
    /// unless its two-body fit failed, see [`confirm()`]. It is over the
    /// observations of the candidate its two-body fit kept, and its radar
    /// observations, less those dropped for being too few on one side of the
    /// largest time gap, not over all of the observations of the candidate.
    pub fit: OrbitFit,
}

/// Fewest observations [`ConfirmRules::check`] accepts as `min_sights`.
/// Three optical observations give six measurements for the six parameters
/// of an orbit, leaving no degrees of freedom for the reduced residual.
const MIN_FIT_SIGHTS: usize = 4;

/// The rules by which [`confirm()`] confirms a candidate, also those of
/// [`crate::merge_pieces`] and of the confirmation of
/// [`crate::SearchSettings`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfirmRules {
    /// Fewest observations a confirmed fit keeps, at least 4.
    pub min_sights: usize,

    /// Largest reduced root mean square residual of a confirmed fit, in
    /// units of the observation uncertainties, finite and positive.
    pub max_rms: f64,

    /// Fewest observations a confirmed fit keeps on each side of its largest
    /// time gap, at least 1, and the count of the observations a fit of three
    /// or more time groups needs, see [`confirm()`]. 2 favors purity, as a
    /// search for unknown objects needs; 1 turns both rules off and favors
    /// completeness. Above half of `min_sights` it also raises the fewest
    /// observations of a confirmation to twice its value.
    pub min_sights_per_side: usize,

    /// Largest time between consecutive observations of one time group of a
    /// confirmation, in days, finite and not negative: detections closer in
    /// time than this are taken to measure only a direction and a motion on
    /// the sky. See [`confirm()`] for the time groups.
    ///
    /// It must be below the time between the revisits of a field by the
    /// survey, such as the orbit of a spacecraft which revisits a field every
    /// orbit: each revisit then starts a new time group. A span longer than
    /// that chains the revisits of a whole visit into one time group, and an
    /// object seen in one visit has one group however many times it was seen.
    pub group_span: f64,

    /// Largest time between the two time groups of a confirmation with only
    /// two, in days, not negative, see [`confirm()`]. Infinite turns that
    /// rule off. In a search
    /// it is also the largest time between the sights of a pair, see
    /// [`crate::form_units`]: the largest gap the search links across without a
    /// third group to check it.
    pub max_gap: f64,
}

impl Default for ConfirmRules {
    /// Four observations, two on each side of their largest gap, a residual
    /// of twice the uncertainties, time groups parted at gaps of more than a
    /// tenth of a day, and two time groups at most a day and a half apart,
    /// the rules of
    /// [`crate::SearchSettings::default`].
    fn default() -> Self {
        Self {
            min_sights: 4,
            max_rms: 2.0,
            min_sights_per_side: 2,
            group_span: 0.1,
            max_gap: 1.5,
        }
    }
}

impl ConfirmRules {
    /// Check that the rules are valid.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `min_sights` is below 4,
    /// `min_sights_per_side` is 0, `max_rms` is not finite and positive,
    /// `group_span` is negative or not finite, or `max_gap` is negative or
    /// not a number.
    pub(crate) fn check(&self) -> KeteResult<()> {
        let Self {
            min_sights,
            max_rms,
            min_sights_per_side,
            group_span,
            max_gap,
        } = *self;
        if !(max_rms.is_finite() && max_rms > 0.0) {
            return Err(Error::ValueError(format!(
                "Confirm: max_rms must be finite and positive, got {max_rms}"
            )));
        }
        if !(group_span.is_finite() && group_span >= 0.0 && max_gap >= 0.0) {
            return Err(Error::ValueError(format!(
                "Confirm: group_span must be finite and not negative, and max_gap not negative, \
                 got {group_span} and {max_gap}"
            )));
        }
        if min_sights < MIN_FIT_SIGHTS {
            return Err(Error::ValueError(format!(
                "Confirm: min_sights must be at least {MIN_FIT_SIGHTS}, got {min_sights}"
            )));
        }
        if min_sights_per_side == 0 {
            return Err(Error::ValueError(
                "Confirm: min_sights_per_side must be at least 1, got 0".into(),
            ));
        }
        Ok(())
    }
}

/// Factor over `max_rms` past which the reduced residual of the two-body fit
/// of a candidate turns it down without the orbit fit.
///
/// The orbit fit starts from the two-body orbit and adds the planets, which
/// over the short arcs of a search move the residual of an orbit little. A
/// two-body fit this far past `max_rms` has found no orbit near the sights,
/// and the orbit fit rarely finds one from there. The orbit fit integrates
/// the pull of the planets, which the two-body fit does not, and most
/// candidates of a search are false, see [`crate::search()`], so turning
/// these down first spares [`confirm()`] the costlier fit of each.
const HOPELESS: f64 = 10.0;

/// Candidates fit at a time by [`confirm()`], between the steps which accept
/// confirmed candidates and skip those which overlap them.
///
/// A candidate is skipped only for an overlap with one accepted in an
/// earlier batch, so the batch size is part of the result, and it is fixed
/// rather than taken from the count of threads, so that the result is the
/// same on every machine. Smaller batches skip a few more candidates without
/// a fit; larger ones keep the threads of a large machine busy past the few
/// slow fits of each batch.
const BATCH: usize = 512;

/// The candidates an orbit fit confirms, without overlaps.
///
/// The state a candidate carries is that of one unit under one hypothesis. On a
/// short arc, hypotheses far from the truth group the same sights, so the state
/// can be far off along the line of sight, often outside the reach of the orbit
/// fit. Each candidate is therefore first fit with a damped least-squares
/// two-body orbit about the Sun to its own sights, kept bound once bound, at
/// the middle of the time its sights span, and that orbit moved to the Solar
/// System Barycenter there starts the fit. The reference time of a candidate's
/// hypothesis can be days from its sights, and a start there would carry the
/// two-body orbit's neglect of the planets over that time into the residuals of
/// a start which the fit only scores, see below. The orbit of a confirmation is
/// at the epoch of its start. The two-body orbit leaves out the planets, so for
/// an object passing close to a planet it can start the fit too far from the
/// solution for the fit to reach it.
///
/// The two-body fit drops sights which do not fit, one at a time and the
/// worst first, by the rule the orbit fit uses to reject observations and
/// with the same threshold. It drops a sight only if at least four sights,
/// and at least two thirds of the optical observations of the candidate,
/// would remain. Sights it drops are not part of the confirmation: the
/// orbit fit sees only the rest.
///
/// Some candidates are turned down without the orbit fit:
///
/// * one whose two-body fit keeps fewer than `min_sights` observations, or
///   leaves a reduced root mean square residual more than ten times
///   `max_rms`, which has found no orbit near the sights for the orbit fit
///   to start from;
/// * one whose kept observations cannot pass the rules on motion and time
///   groups below, whichever of them the orbit fit keeps. That is so when
///   all of the kept sights together show no motion, or when no subset of
///   them can form passing time groups: no two of them are more than
///   `group_span` and at most `max_gap` days apart, which alone would form
///   two groups, and either no three are each more than `group_span` days
///   after the one before, which alone would form three, or there are no
///   more than twice `min_sights_per_side` of them, fewer than three or more
///   groups need.
///
/// The second test looks at subsets rather than at all of the kept sights,
/// as the orbit fit keeps only some of them, and dropping one which bridges
/// a time group can split that group in two. So it turns down no candidate
/// the orbit fit would confirm.
///
/// The orbit fit is `kete_fitting::fit_orbit`, with outlier rejection. When
/// it fails, as it does when the observations of a short arc do not
/// constrain every orbit parameter, or when its orbit is not bound to the
/// Sun, the start is scored as it is: `fit_orbit` with no iterations gives
/// its residuals under the same force model and outlier rejection, and
/// reports that it did not converge; its covariance may not be a number.
/// The question confirmation answers is whether the observations fit one
/// orbit, which the residual answers even when some parameters of that orbit
/// are poorly known.
///
/// A candidate is confirmed if its fit:
///
/// * keeps at least `min_sights` observations, and at least two thirds of
///   those of the candidate, so that the fit cannot pick a mixture of two
///   objects on similar orbits out of a larger candidate;
/// * has a reduced root mean square residual of at most `max_rms`, with a
///   finite residual and state. The fit need not report that it converged:
///   on a short arc it can stop short of its convergence test with residuals
///   as small as the noise;
/// * has an orbit bound to the Sun, the scope of the crate;
/// * keeps observations which move: two of them differ in direction by more
///   than 5 times their combined uncertainty. A source fixed on the sky,
///   seen by observers at different places, fits an orbit far away, and
///   this turns it down. A real object moves more than that over a few
///   observations, a distant one through the motion of the observer;
/// * keeps at least `min_sights_per_side` observations on each side of the
///   largest time between two consecutive ones it keeps;
/// * keeps observations in at least two time groups: in exactly two only
///   when they are at most `max_gap` days apart, and in three or more only
///   when two of the groups hold at least `min_sights_per_side` observations
///   each, or all of them together at least three times that. The time
///   groups are the clusters of the kept observations in time, parted
///   wherever two consecutive ones are more than `group_span` days apart. The
///   gap between two groups is from the last observation of the first to the
///   first of the second.
///
/// The last two rules are against links of two objects. Observations close in
/// time fix the direction and the motion on the sky of an orbit, four of its
/// six numbers, and leave its distance and radial speed free. Those two can
/// always be chosen to pass the orbit through one more direction at another
/// time, so a single observation across a gap is fit whatever object it shows,
/// and the fit tests nothing there. Two observations across the gap give four
/// measurements for the two free numbers, which a mixture of two objects does
/// not fit. The largest gap is where the kept observations are least tied
/// together. Two per side favors purity, as a search for unknown objects needs,
/// and turns down a real object seen once across its largest gap; one per side
/// turns the rule off, and favors completeness, as the recovery of objects
/// expected in the observations allows.
///
/// A fit with too few observations on one side of its largest gap is not
/// turned down at once. Those observations are dropped, and the rest are fit
/// again from the same start, under all the rules above, the gap rule on
/// their own largest gap included. A candidate of one object with a stray
/// observation across a gap is so confirmed without the stray. The two thirds
/// are of the observations of the candidate as given, as for observations
/// the fit rejects: the rule bounds how much of a candidate the confirmation
/// may leave out, whatever left it out. Each fit again has fewer
/// observations, so after at most a third of those of the candidate are
/// dropped the two thirds end it.
///
/// The time groups carry the same argument from observations to visits. One
/// group, observations so close in time that they measure no change of the
/// motion, fixes only the direction and motion on the sky, and any orbit
/// through them fits: it confirms nothing. Two groups give four measurements
/// for the two free numbers, two degrees of freedom, which is a weak test: the
/// motion of each group is known only to its own noise, and the farther apart
/// the groups, the more room that noise leaves for a tracklet of another object
/// to line up by chance, so chance links of two tracklets grow with the time
/// between them. A confirmation of two groups is therefore held to the time
/// between them that a pair may span, `max_gap`, the largest gap the search
/// links across without a third group to check it. Three or more groups test
/// the orbit at a third time, and have no gap limit, but only when they hold as
/// many observations as two groups across a gap are asked for: two groups of
/// `min_sights_per_side` among them, or three times that in all. Where the
/// cadence is close against `group_span`, as for continuous coverage, every
/// observation can be its own group, and among many false detections a few
/// single ones line up with a false tracklet by chance; the count turns those
/// down whatever the cadence. A fit which fails these rules is turned down. A
/// candidate whose observations all lie within `group_span` days is one time
/// group whatever the fit keeps, and is turned down without a fit.
///
/// Of confirmed candidates which share an observation, the one which keeps
/// more observations wins, then the one with the lower residual, then the
/// one listed first.
///
/// Candidates are fit in batches, those with the most observations first. A
/// confirmed candidate is accepted, or turned down for an overlap, once no
/// candidate still to be fit has as many observations as it keeps, so that
/// none could win over it. A candidate which shares an observation with one
/// already accepted is not fit. That one keeps more observations than the
/// candidate has, so the candidate would lose to it unless its fit dropped
/// every shared observation. That case alone separates the result from
/// fitting every candidate and resolving the overlaps after.
///
/// # Arguments
///
/// * `observations` - The observations the sights of the candidates index.
/// * `candidates` - Candidates, such as from [`crate::link()`].
/// * `rules` - The rules above, see [`ConfirmRules`].
///
/// # Errors
///
/// [`Error::ValueError`] if the rules are not valid, see [`ConfirmRules`], if
/// a candidate indexes past the observations, or if an optical observation of
/// a candidate has a line of sight which is not [`Sight::is_usable`]. Fails if
/// the loaded kernels do not cover the Sun at the epoch of a candidate or of
/// one of its optical observations, or do not cover the Sun, the planets and
/// the Moon, which the orbit fit pulls with, from a day before the first to
/// the last of the epochs of the candidates and of their observations, or if
/// an orbit fit fails for want of data the kernels do not hold. A candidate whose fit
/// otherwise fails is not confirmed, and is not an error.
pub fn confirm(
    observations: &[AstrometricObservation],
    candidates: &[Candidate],
    rules: &ConfirmRules,
) -> KeteResult<Vec<Confirmed>> {
    rules.check()?;
    let too_far = |candidate: &Candidate| candidate.sights.iter().any(|&i| i >= observations.len());
    if candidates.iter().any(too_far) {
        return Err(Error::ValueError(
            "Confirm: a candidate indexes past the observations".into(),
        ));
    }
    let indices = candidates.iter().flat_map(|candidate| &candidate.sights);
    let sights = sights_of(observations, indices.copied())?;
    let factors = vec![1.0; observations.len()];
    let scaled = Scaled {
        sights: &sights,
        factors: &factors,
    };
    confirm_with(observations, scaled, candidates, rules)
}

/// The line of sight of each optical observation at `indices`, with its
/// stated uncertainty, and `None` for every other observation. The indices
/// must be within `observations`.
///
/// # Errors
///
/// [`Error::ValueError`] if one of those lines of sight is not
/// [`Sight::is_usable`]. Fails if the loaded kernels do not cover the Sun at
/// the epoch of one.
pub(crate) fn sights_of(
    observations: &[AstrometricObservation],
    indices: impl IntoIterator<Item = usize>,
) -> KeteResult<Vec<Option<Sight>>> {
    let mut sights: Vec<Option<Sight>> = vec![None; observations.len()];
    for index in indices {
        if sights[index].is_none() && !observations[index].is_radar() {
            sights[index] = Some(sight_from_observation(&observations[index])?);
        }
    }
    Ok(sights)
}

/// [`confirm()`] without the checks of the rules and of the indices of the
/// candidates, with `sights` holding the line of sight of every optical
/// observation of the candidates. The two-body fits, the orbit fits and every
/// rule see the uncertainties of the observations as `sights` scales them.
///
/// The lines of sight, for the two-body fits, and the starts below are made,
/// and the kernels the orbit fits need are checked, before the fits, so that
/// missing kernels are an error rather than a candidate which is not
/// confirmed.
pub(crate) fn confirm_with(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    candidates: &[Candidate],
    rules: &ConfirmRules,
) -> KeteResult<Vec<Confirmed>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    // The lock is released before the fits take it again.
    let starts = {
        let ephemeris = SpiceEphemeris::loaded()?;
        let times = candidates.iter().flat_map(|candidate| {
            let own = candidate.sights.iter().map(|&i| observations[i].epoch());
            own.chain([candidate.state.epoch]).map(|time| time.jd())
        });
        check_force_model(&ephemeris, times)?;
        candidates
            .iter()
            .map(|candidate| ephemeris.try_to_ssb(candidate.state.clone().into()))
            .collect::<KeteResult<Vec<_>>>()?
    };
    let confirms = |index: usize| -> KeteResult<Option<(usize, Confirmed)>> {
        let found = confirm_one(
            observations,
            sights,
            &candidates[index],
            &starts[index],
            rules,
        )?;
        Ok(found.map(|found| (index, found)))
    };
    // Most observations first, and of equal counts in the order given. A
    // candidate within one time group stays in one whatever the fit keeps.
    let spans_groups = |candidate: &Candidate| {
        let times = candidate
            .sights
            .iter()
            .map(|&i| observations[i].epoch().jd());
        let (first, last) = first_and_last(times);
        last - first > rules.group_span
    };
    let mut order: Vec<usize> = (0..candidates.len())
        .filter(|&index| {
            let candidate = &candidates[index];
            candidate.sights.len() >= rules.min_sights && spans_groups(candidate)
        })
        .collect();
    order.sort_by_key(|&index| std::cmp::Reverse(candidates[index].sights.len()));
    let mut taken = vec![false; observations.len()];
    let mut waiting: Vec<(usize, Confirmed)> = Vec::new();
    let mut accepted = Vec::new();
    for (n, chunk) in order.chunks(BATCH).enumerate() {
        let open = |&&index: &&usize| !candidates[index].sights.iter().any(|&i| taken[i]);
        let chosen: Vec<usize> = chunk.iter().filter(open).copied().collect();
        let found: Vec<Option<(usize, Confirmed)>> = chosen
            .into_par_iter()
            .map(confirms)
            // The first error in the order of the candidates, on any threads.
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<KeteResult<_>>()?;
        waiting.extend(found.into_iter().flatten());
        waiting.sort_by(|(i, a), (j, b)| rank(a, b).then(i.cmp(j)));
        // No candidate still to be fit keeps more observations than the
        // next one has.
        let rest = order
            .get((n + 1) * BATCH)
            .map_or(0, |&index| candidates[index].sights.len());
        let settled = waiting.partition_point(|(_, found)| found.sights.len() > rest);
        let settled = waiting.drain(..settled).map(|(_, found)| found);
        accept_disjoint(settled, &mut taken, &mut accepted);
    }
    Ok(accepted)
}

/// Fails unless the loaded kernels hold every body the orbit fit pulls with,
/// with respect to the Solar System Barycenter, from [`LIGHT_TIME`] before
/// the first of `times`, Julian dates, to the last. The orbit fit propagates
/// within that span, and a body the kernels do not hold makes it fail to
/// converge, which [`confirm()`] would take for a candidate which is not
/// confirmed. The ephemerides of the planets are continuous over their span,
/// so its ends stand for the time between. No `times` need nothing.
pub(crate) fn check_force_model(
    ephemeris: &SpiceEphemeris,
    times: impl IntoIterator<Item = f64>,
) -> KeteResult<()> {
    let (first, last) = first_and_last(times);
    if first > last {
        return Ok(());
    }
    for time in [first - LIGHT_TIME, last] {
        for body in GravParams::planets() {
            let _ = ephemeris.try_get_state_with_center(body.naif_id, time.into(), 0)?;
        }
    }
    Ok(())
}

/// The fit of one candidate if it is confirmed by the rules of [`confirm()`],
/// from `start`, its state moved to the Solar System Barycenter. `sights`
/// holds the lines of sight of its optical observations. The fit keeps at
/// least `min_sights` observations, see [`fit`].
///
/// # Errors
///
/// A fit which fails for want of data, see [`is_missing_data`], is an
/// error, not a candidate which is not confirmed. Every other failure of the
/// fit turns the candidate down.
pub(crate) fn confirm_one(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    candidate: &Candidate,
    start: &State<Equatorial, SSB>,
    rules: &ConfirmRules,
) -> KeteResult<Option<Confirmed>> {
    let found = match fit(observations, sights, candidate, start, rules) {
        Ok(found) => found,
        Err(error) if is_missing_data(&error) => return Err(error),
        Err(_) => return Ok(None),
    };
    let finite = found
        .fit
        .uncertain_state
        .state::<Equatorial>()
        .is_ok_and(|state| state.is_finite());
    let good = finite
        && is_bound(&found.fit)
        && moves(sights, &found.sights)
        && time_groups_hold(observations, &found.sights, rules)
        && found.fit.rms <= rules.max_rms;
    Ok(good.then_some(found))
}

/// Whether an error is a failure to read data, such as a body or a time the
/// loaded kernels do not hold, rather than a fit which does not succeed.
fn is_missing_data(error: &Error) -> bool {
    matches!(
        error,
        Error::Bounds(_) | Error::IOError(_) | Error::LockFailed
    )
}

/// The order of the overlap rule of [`confirm()`], without its last tie
/// break, the order the candidates are listed in.
pub(crate) fn rank(a: &Confirmed, b: &Confirmed) -> std::cmp::Ordering {
    b.sights
        .len()
        .cmp(&a.sights.len())
        .then(a.fit.rms.total_cmp(&b.fit.rms))
}

/// Accept, in the order given, each confirmed candidate which shares no
/// observation with one accepted before it. `taken` flags the observations
/// of those accepted, and is updated.
pub(crate) fn accept_disjoint(
    found: impl IntoIterator<Item = Confirmed>,
    taken: &mut [bool],
    accepted: &mut Vec<Confirmed>,
) {
    for found in found {
        if found.sights.iter().any(|&i| taken[i]) {
            continue;
        }
        for &i in &found.sights {
            taken[i] = true;
        }
        accepted.push(found);
    }
}

/// Whether the orbit of a fit is bound to the Sun.
fn is_bound(fit: &OrbitFit) -> bool {
    let elements = &fit.uncertain_state.elements;
    elements.center_id == SunCenter::NAIF_ID && elements.eccentricity() < 1.0
}

/// Whether the lines of sight of these observations show motion: some two of
/// them differ in direction by more than [`MIN_MOTION`] times their combined
/// uncertainty. Observations without a line of sight are passed over.
fn moves(sights: Scaled<'_>, indices: &[usize]) -> bool {
    let own: Vec<Sight> = indices.iter().filter_map(|&i| sights.sight(i)).collect();
    own.iter().enumerate().any(|(k, a)| {
        own[k + 1..]
            .iter()
            .any(|b| (a.direction - b.direction).norm() > MIN_MOTION * a.sigma.hypot(b.sigma))
    })
}

/// Whether some of these observations may pass the rules [`confirm()`] puts
/// on the observations an orbit fit keeps, which are some of them, so that
/// a candidate turned down here is one the orbit fit could not confirm.
///
/// Some of them show motion when all of them do, see [`moves`]. Some of
/// them fall in time groups which pass [`time_groups_hold`] exactly when two
/// of them are more than `group_span` and at most `max_gap` days apart, which
/// alone form two groups, or three of them are each more than `group_span`
/// days after the one before, which alone form three, and there are at least
/// as many as two groups of `min_sights_per_side` and one more, the fewest
/// either rule for three or more groups accepts. [`time_groups_hold`] on all
/// of them is not such a test: dropping an observation which bridges a group
/// can split it in two.
fn may_hold(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    indices: &[usize],
    rules: &ConfirmRules,
) -> bool {
    let (group_span, max_gap) = (rules.group_span, rules.max_gap);
    let mut times: Vec<f64> = indices
        .iter()
        .map(|&i| observations[i].epoch().jd())
        .collect();
    times.sort_by(f64::total_cmp);
    let two = times.iter().enumerate().any(|(k, a)| {
        times[k + 1..]
            .iter()
            .any(|b| b - a > group_span && b - a <= max_gap)
    });
    // Each group starts with the first time more than `group_span` after the
    // start of the last, which finds three if any three are so spaced.
    let mut groups = 0;
    let mut start = f64::NEG_INFINITY;
    for &time in &times {
        if time - start > group_span {
            groups += 1;
            start = time;
        }
    }
    let three = groups >= 3 && times.len() > 2 * rules.min_sights_per_side;
    (two || three) && moves(sights, indices)
}

/// Whether these observations fall in time groups which pass the rules of
/// [`confirm()`]: two groups of at least `min_sights_per_side` each at most
/// `max_gap` days apart, or three or more groups of which two hold at least
/// `min_sights_per_side` each or which hold three times that in all.
fn time_groups_hold(
    observations: &[AstrometricObservation],
    indices: &[usize],
    rules: &ConfirmRules,
) -> bool {
    let (group_span, max_gap) = (rules.group_span, rules.max_gap);
    let per_side = rules.min_sights_per_side;
    let mut times: Vec<f64> = indices
        .iter()
        .map(|&i| observations[i].epoch().jd())
        .collect();
    times.sort_by(f64::total_cmp);
    // Each group as its first and last time and its count.
    let mut groups: Vec<(f64, f64, usize)> = Vec::new();
    for &time in &times {
        match groups.last_mut() {
            Some((_, last, count)) if time - *last <= group_span => {
                *last = time;
                *count += 1;
            }
            _ => groups.push((time, time, 1)),
        }
    }
    let rich = groups.iter().filter(|group| group.2 >= per_side).count();
    match groups.as_slice() {
        [] | [_] => false,
        [first, second] => rich == 2 && second.0 - first.1 <= max_gap,
        _ => rich >= 2 || times.len() >= 3 * per_side,
    }
}

/// The observations left once those on a side of the largest time gap
/// between consecutive ones with fewer than `per_side` are dropped, or `None`
/// if neither side has too few. Of equal largest gaps, one with enough on
/// each side is enough, and otherwise the first decides. See [`confirm()`].
fn short_side(
    observations: &[AstrometricObservation],
    indices: &[usize],
    per_side: usize,
) -> Option<Vec<usize>> {
    let mut sorted = indices.to_vec();
    sorted.sort_by(|&a, &b| {
        observations[a]
            .epoch()
            .jd()
            .total_cmp(&observations[b].epoch().jd())
    });
    let gaps: Vec<f64> = sorted
        .windows(2)
        .map(|pair| observations[pair[1]].epoch().jd() - observations[pair[0]].epoch().jd())
        .collect();
    let largest = gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Gap `k` has `k + 1` observations before it and the rest after.
    let widest = (0..gaps.len()).filter(|&k| gaps[k] == largest);
    let enough = |k: usize| {
        let (before, after) = (k + 1, sorted.len() - k - 1);
        before >= per_side && after >= per_side
    };
    if widest.clone().any(enough) {
        return None;
    }
    let k = widest.min()?;
    let (before, after) = sorted.split_at(k + 1);
    Some(
        if before.len() < per_side {
            after
        } else {
            before
        }
        .to_vec(),
    )
}

/// The orbit fit of one candidate, with the observations it kept. `start` is
/// the state of the candidate moved to the Solar System Barycenter, which
/// starts the orbit fit only if the two-body fit fails, and `sights` the lines
/// of sight of the observations. An error is a candidate which is not
/// confirmed, unless it is for want of data, see [`confirm_one`], such as one
/// whose two-body fit is hopeless under `min_sights` and `max_rms` of the
/// rules, see [`HOPELESS`]. The observations it keeps are at least
/// `min_sights` and two thirds of those of the candidate, with at least
/// `min_sights_per_side` on each side of their largest gap, see [`confirm()`].
fn fit(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    candidate: &Candidate,
    start: &State<Equatorial, SSB>,
    rules: &ConfirmRules,
) -> KeteResult<Confirmed> {
    let &ConfirmRules {
        min_sights,
        max_rms,
        min_sights_per_side: per_side,
        ..
    } = rules;
    // The two-body fit is about the Sun, at the middle of the optical
    // observations. It is moved to the barycenter there to start the orbit
    // fit, so that the planets it leaves out act only over the arc of the
    // observations, not also over the time from the reference time of the
    // candidate to them.
    let (optical, others): (Vec<usize>, Vec<usize>) = candidate
        .sights
        .iter()
        .partition(|&&i| sights.is_optical(i));
    let own: Vec<Sight> = optical.iter().filter_map(|&i| sights.sight(i)).collect();
    let (start, mut indices) = match fit_two_body(&candidate.state, &own, FIT_REJECT_THRESHOLD) {
        Some((_, kept, rms))
            if rms > HOPELESS * max_rms || kept.len() + others.len() < min_sights =>
        {
            return Err(Error::ValueError(
                "Confirm: the two-body fit leaves too few observations or too large a residual"
                    .into(),
            ));
        }
        Some((refined, kept, _)) => (
            SpiceEphemeris::loaded()?.try_to_ssb(refined.into())?,
            kept.iter().map(|&k| optical[k]).chain(others).collect(),
        ),
        None => (start.clone(), candidate.sights.clone()),
    };
    if !may_hold(observations, sights, &indices, rules) {
        return Err(Error::ValueError(
            "Confirm: the observations the two-body fit keeps cannot be confirmed".into(),
        ));
    }

    // Observations on a side of the largest gap with too few are dropped,
    // and the rest fit again. Each fit has fewer observations than the last,
    // and the floor ends the loop, without a fit of fewer observations than
    // it.
    let floor = min_sights.max(fewest_kept(candidate.sights.len()));
    loop {
        let (fit, mut kept) = fit_kept(observations, sights, &start, &mut indices)?;
        let rest = short_side(observations, &kept, per_side);
        if rest.as_ref().map_or(kept.len(), Vec::len) < floor {
            return Err(Error::ValueError(
                "Confirm: the fit keeps fewer than min_sights observations, or than two thirds of \
                 the candidate, on enough sides of its largest gap"
                    .into(),
            ));
        }
        match rest {
            None => {
                kept.sort_unstable();
                return Ok(Confirmed { sights: kept, fit });
            }
            Some(rest) => indices = rest,
        }
    }
}

/// The orbit fit from `start` of the observations at `indices`, with their
/// uncertainties as `sights` scales them, and the indices of those it kept,
/// in time order. The indices are sorted first, in time and then by index.
fn fit_kept(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    start: &State<Equatorial, SSB>,
    indices: &mut [usize],
) -> KeteResult<(OrbitFit, Vec<usize>)> {
    // In time order, as the fit stable-sorts them, so that its inclusion
    // flags line up with the indices. Each one is checked by its epoch. Of
    // equal times the lower index is first, so that the order of the fit
    // follows from the indices it kept, see `kept_chi_squares`.
    indices.sort_by(|&a, &b| {
        let epoch = |i: usize| observations[i].epoch().jd();
        epoch(a).total_cmp(&epoch(b)).then(a.cmp(&b))
    });
    let chosen: Vec<_> = indices
        .iter()
        .map(|&i| sights.observation(observations, i))
        .collect();
    let run = |iterations| {
        fit_orbit(
            start,
            &chosen,
            false,
            None,
            iterations,
            FIT_REJECT_THRESHOLD,
            FIT_REJECT_PASSES,
            None,
        )
    };
    let fit = match run(FIT_ITERATIONS) {
        Ok(fit) if is_bound(&fit) => fit,
        Err(error) if is_missing_data(&error) => return Err(error),
        _ => run(0)?,
    };
    if fit.included.len() != indices.len() || fit.observations.len() != indices.len() {
        return Err(Error::ValueError(
            "Confirm: the fit left out observations, such as ones with an observer state which \
             is not finite"
                .into(),
        ));
    }
    for (k, (fitted, &index)) in fit.observations.iter().zip(indices.iter()).enumerate() {
        if fitted.epoch() != observations[index].epoch() {
            return Err(Error::ValueError(format!(
                "Confirm: observation {k} of the fit is not at the epoch of observation {index}, \
                 so its inclusion flags do not line up"
            )));
        }
    }
    let kept = indices
        .iter()
        .zip(&fit.included)
        .filter_map(|(&index, &kept)| kept.then_some(index))
        .collect();
    Ok((fit, kept))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::Band;
    use kete_core::desigs::Desig;
    use kete_core::time::{TDB, Time};

    #[test]
    fn missing_data_is_an_error() {
        // A fit which cannot read the kernels fails the search; one which
        // does not converge, or is refused, only turns the candidate down.
        for error in [
            Error::Bounds("no Jupiter".into()),
            Error::IOError("unreadable".into()),
            Error::LockFailed,
        ] {
            assert!(is_missing_data(&error), "{error}");
        }
        for error in [
            Error::Convergence("no".into()),
            Error::ValueError("no".into()),
            Error::Impact(399, Time::new(2460000.5)),
        ] {
            assert!(!is_missing_data(&error), "{error}");
        }
    }

    #[test]
    fn bridged_groups_are_left_to_the_fit() {
        // Two sights 0.16 days apart, bridged by one between them, and one
        // two days later: one time group too far from the other. Without the
        // bridge there are three groups, which an orbit fit that drops the
        // bridge as an outlier may confirm, so the candidate is not turned
        // down before the fit. Without the first sight, no sights of them
        // form passing groups, and it is.
        // One per side, so that three single sights may be three groups.
        let rules = ConfirmRules {
            min_sights_per_side: 1,
            group_span: 0.1,
            max_gap: 1.5,
            ..ConfirmRules::default()
        };
        let observations: Vec<AstrometricObservation> = [0.0, 0.08, 0.16, 2.0]
            .iter()
            .enumerate()
            .map(|(k, &days)| AstrometricObservation::Optical {
                observer: State::new(
                    Desig::Empty,
                    Time::<TDB>::new(2460000.5 + days),
                    Vector::new([1.0, 0.0, 0.0]),
                    Vector::new([0.0, 0.017, 0.0]),
                    SSB,
                ),
                ra: 0.01 * k as f64,
                dec: 0.0,
                sigma_ra: 1e-6,
                sigma_dec: 1e-6,
                sigma_corr: 0.0,
                time_sigma: 0.0,
                is_occultation: false,
                band: Band::Unknown([0; 8]),
                mag: f64::NAN,
            })
            .collect();
        let sights: Vec<Option<Sight>> = observations
            .iter()
            .map(|observation| {
                let AstrometricObservation::Optical {
                    observer, ra, dec, ..
                } = observation
                else {
                    unreachable!()
                };
                Some(Sight {
                    time: observer.epoch,
                    observer: observer.pos,
                    direction: Vector::from_ra_dec(*ra, *dec),
                    sigma: 1e-6,
                })
            })
            .collect();
        let (all, unbridged) = ([0, 1, 2, 3], [0, 2, 3]);
        assert!(!time_groups_hold(&observations, &all, &rules));
        assert!(time_groups_hold(&observations, &unbridged, &rules));
        assert!(may_hold(
            &observations,
            Scaled {
                sights: &sights,
                factors: &[1.0; 4],
            },
            &all,
            &rules
        ));
        assert!(!time_groups_hold(&observations, &[1, 2, 3], &rules));
        assert!(!may_hold(
            &observations,
            Scaled {
                sights: &sights,
                factors: &[1.0; 4],
            },
            &[1, 2, 3],
            &rules
        ));
    }

    #[test]
    fn groups_need_their_observations() {
        // Sights at the given days, each a group of its own unless less than
        // `group_span` 0.03 days apart, moving steadily.
        let rules = ConfirmRules {
            group_span: 0.03,
            ..ConfirmRules::default()
        };
        let observations = |days: &[f64]| -> Vec<AstrometricObservation> {
            days.iter()
                .enumerate()
                .map(|(k, &day)| AstrometricObservation::Optical {
                    observer: State::new(
                        Desig::Empty,
                        Time::<TDB>::new(2460000.5 + day),
                        Vector::new([1.0, 0.0, 0.0]),
                        Vector::new([0.0, 0.017, 0.0]),
                        SSB,
                    ),
                    ra: 1e-3 * k as f64,
                    dec: 0.0,
                    sigma_ra: 1e-6,
                    sigma_dec: 1e-6,
                    sigma_corr: 0.0,
                    time_sigma: 0.0,
                    is_occultation: false,
                    band: Band::Unknown([0; 8]),
                    mag: f64::NAN,
                })
                .collect()
        };
        let holds = |days: &[f64], rules: &ConfirmRules| {
            let all: Vec<usize> = (0..days.len()).collect();
            time_groups_hold(&observations(days), &all, rules)
        };
        let pair = |day: f64| [day, day + 0.02];
        // A sight every 95 minutes: five single groups are fewer than three
        // groups of two, six are enough, and with one per side three are.
        let orbits: Vec<f64> = (0..6).map(|k| f64::from(k) * 95.0 / 1440.0).collect();
        assert!(!holds(&orbits[..5], &rules));
        assert!(holds(&orbits, &rules));
        let one = ConfirmRules {
            min_sights_per_side: 1,
            ..rules
        };
        assert!(holds(&orbits[..3], &one));
        // Two pairs and a single sight between them or after them: two
        // groups of two among three or more, at any time apart.
        assert!(holds(
            &[&pair(0.0)[..], &[1.0], &pair(2.0)].concat(),
            &rules
        ));
        assert!(holds(
            &[&pair(0.0)[..], &[2.0], &pair(4.0)].concat(),
            &rules
        ));
        assert!(holds(
            &[&pair(0.0)[..], &pair(7.0), &[7.1]].concat(),
            &rules
        ));
        // One pair and two single sights are not enough.
        assert!(!holds(&[&pair(0.0)[..], &[1.0, 2.0]].concat(), &rules));
        // Two pairs alone are held to `max_gap`.
        assert!(holds(&[pair(0.0), pair(1.0)].concat(), &rules));
        assert!(!holds(&[pair(0.0), pair(2.0)].concat(), &rules));

        // Before a fit, sights which no subset of could pass are turned
        // down: three spaced single sights and one more are too few for
        // either rule of three groups, five are enough.
        let sights = |days: &[f64]| -> Vec<Option<Sight>> {
            observations(days)
                .iter()
                .map(|observation| {
                    let AstrometricObservation::Optical {
                        observer, ra, dec, ..
                    } = observation
                    else {
                        unreachable!()
                    };
                    Some(Sight {
                        time: observer.epoch,
                        observer: observer.pos,
                        direction: Vector::from_ra_dec(*ra, *dec),
                        sigma: 1e-6,
                    })
                })
                .collect()
        };
        let may = |days: &[f64]| {
            let all: Vec<usize> = (0..days.len()).collect();
            let lines = sights(days);
            let factors = vec![1.0; days.len()];
            let scaled = Scaled {
                sights: &lines,
                factors: &factors,
            };
            may_hold(&observations(days), scaled, &all, &rules)
        };
        assert!(!may(&[0.0, 2.0, 4.0, 4.02]));
        assert!(may(&[0.0, 0.02, 2.0, 4.0, 4.02]));
    }
}
