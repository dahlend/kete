// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Merging confirmed objects which are pieces of one object.

use std::collections::{BTreeMap, HashMap};

use kete_core::ephemeris::Ephemeris;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::time::{TDB, Time};
use kete_fitting::AstrometricObservation;
use kete_spice::prelude::SpiceEphemeris;
use kete_stats::healpix::{ang_to_nested, max_pixel_radius, query_disc_nested};
use rayon::prelude::*;

use crate::calibration::Scaled;
use crate::confirm::{FIT_REJECT_THRESHOLD, check_force_model, confirm_one, rank, sights_of};
use crate::linalg::{inverse, scaled_solve};
use crate::observe::{direction_at, predicted};
use crate::sight::angle;
use crate::two_body_fit::{fit_two_body, normal_at};
use crate::{Candidate, ConfirmRules, Confirmed, Sight};

/// Width of the time bins of the index of [`merge_pieces`], in days. An orbit
/// is predicted once for each bin, so narrower bins cost more predictions,
/// and wider ones larger reaches: the motion of the object and of the
/// observers within a bin widen the reach.
const BIN_DAYS: f64 = 0.25;

/// Width of the cubic cells of observer positions which split each time bin
/// of the index of [`merge_pieces`], in AU, about 23 radii of the Earth.
/// Observers in one cell are at most its diagonal apart, which keeps the
/// parallax within a bin to about a degree for an object a tenth of an AU
/// away, whatever the distance between the observers of a survey. The cells
/// are fixed about the Sun, and the Earth moves across several of them in a
/// bin, so the sights from the Earth in one bin fall in a few cells one
/// after another, which splits the bin in time as well. That costs a few
/// more predictions, and changes no reach: each cell's spread and mean
/// observer are its own, and the reach covers the motion over the whole bin.
const OBSERVER_CELL: f64 = 1e-3;

/// `HEALPix` resolution of the cells of the index of [`merge_pieces`], whose
/// cells are about a quarter of a degree across, about the reach of a
/// prediction within a bin for objects of the main belt seen from the
/// ground.
const NSIDE: u32 = 256;

/// Largest reach of a prediction into the index of [`merge_pieces`], in
/// radians. A piece whose orbit predicts its direction in a bin no better
/// than this does not look for pieces there; a piece seen there with a
/// better known orbit finds it instead.
const MAX_REACH: f64 = 0.05;

/// The two-body orbit of a piece, fit to its own lines of sight.
#[derive(Debug)]
struct Orbit {
    /// State about the Sun.
    state: State<Equatorial, SunCenter>,

    /// Normal matrix of the residuals of its sights at that state, in units
    /// of their uncertainties, over the square of its reduced residual where
    /// that is above one: the inverse of the covariance of the state.
    normal: [[f64; 6]; 6],

    /// Covariance of the state, `None` if the normal matrix is singular.
    covariance: Option<[[f64; 6]; 6]>,
}

impl Orbit {
    /// The orbit of a confirmed object, from its orbit fit, with `own` the
    /// lines of sight of its optical observations. `None` if the fit is not
    /// about the Sun or the two-body fit fails.
    fn new(piece: &Confirmed, own: &[Sight]) -> Option<Self> {
        let start: State<Equatorial> = piece.fit.uncertain_state.state().ok()?;
        let start = State::<Equatorial, SunCenter>::try_from(start).ok()?;
        let (state, _, rms) = fit_two_body(&start, own, f64::INFINITY)?;
        let (_, normal, _) = normal_at(&state, own)?;
        let scale = rms.powi(2).max(1.0);
        let normal = normal.map(|row| row.map(|x| x / scale));
        Some(Self {
            state,
            normal,
            covariance: inverse(&normal),
        })
    }

    /// Whether this orbit predicts `sights`: the chi-square of their
    /// residuals under it, with its own uncertainty, is at most twice
    /// [`FIT_REJECT_THRESHOLD`] for each sight, see [`merge_pieces`].
    fn predicts(&self, sights: &[Sight]) -> bool {
        let Some((chi2, normal, gradient)) = normal_at(&self.state, sights) else {
            return false;
        };
        let joint: [[f64; 6]; 6] =
            std::array::from_fn(|i| std::array::from_fn(|j| self.normal[i][j] + normal[i][j]));
        let Some(step) = scaled_solve(&joint, gradient, 0.0) else {
            return false;
        };
        let explained: f64 = (0..6).map(|i| step[i] * gradient[i]).sum();
        chi2 - explained <= 2.0 * FIT_REJECT_THRESHOLD * sights.len() as f64
    }
}

/// A piece in the merge: a confirmed object, the lines of sight of its
/// optical observations, and its orbit, `None` if that could not be fit.
#[derive(Debug)]
struct Piece {
    /// The confirmed object.
    object: Confirmed,

    /// The lines of sight of its optical observations.
    sights: Vec<Sight>,

    /// Its two-body orbit, see [`Orbit::new`].
    orbit: Option<Orbit>,
}

/// A time bin and a cell of observer positions of the index, by the number
/// of the bin and the cell's integer coordinates.
type BinKey = (i64, [i64; 3]);

/// The sights of one time bin seen from one cell of observer positions.
#[derive(Debug)]
struct Bin {
    /// Middle of the time bin.
    time: Time<TDB>,

    /// Mean position of the observers of the sights in the bin.
    observer: Vector<Equatorial>,

    /// Largest distance of an observer of the bin from that mean, in AU.
    spread: f64,

    /// Largest uncertainty of a sight of the bin, in radians.
    sigma: f64,
}

/// The lines of sight of the pieces, by time bin and cell of the sky.
#[derive(Debug)]
struct Index {
    /// Each bin with sights.
    bins: BTreeMap<BinKey, Bin>,

    /// The pieces with a sight in each bin and cell of the sky, sorted and
    /// distinct.
    cells: HashMap<(BinKey, u64), Vec<usize>>,
}

impl Index {
    /// The index of the sights of each piece, or only of the pieces `only`
    /// flags.
    fn new(pieces: &[Option<Piece>], only: Option<&[bool]>) -> Self {
        let all = || {
            pieces
                .iter()
                .enumerate()
                .filter(|&(k, _)| only.is_none_or(|only| only[k]))
                .filter_map(|(k, piece)| Some((k, &piece.as_ref()?.sights)))
                .flat_map(|(k, own)| own.iter().map(move |sight| (k, sight)))
        };
        let first = all()
            .map(|(_, sight)| sight.time.jd())
            .fold(f64::INFINITY, f64::min);
        let key = |sight: &Sight| -> BinKey {
            let number = ((sight.time.jd() - first) / BIN_DAYS).floor() as i64;
            let place = <[f64; 3]>::from(sight.observer);
            (number, place.map(|x| (x / OBSERVER_CELL).floor() as i64))
        };
        let mut sums: BTreeMap<BinKey, (Vector<Equatorial>, usize, f64)> = BTreeMap::new();
        let mut cells: HashMap<(BinKey, u64), Vec<usize>> = HashMap::new();
        for (k, sight) in all() {
            let bin = key(sight);
            let sum = sums.entry(bin).or_insert((Vector::new([0.0; 3]), 0, 0.0));
            sum.0 += sight.observer;
            sum.1 += 1;
            sum.2 = sum.2.max(sight.sigma);
            let (ra, dec) = sight.direction.to_ra_dec();
            let cell = cells
                .entry((bin, ang_to_nested(NSIDE, ra, dec)))
                .or_default();
            if cell.last() != Some(&k) {
                cell.push(k);
            }
        }
        let mut bins: BTreeMap<BinKey, Bin> = sums
            .into_iter()
            .map(|(bin, (sum, count, sigma))| {
                let time = Time::new(first + (bin.0 as f64 + 0.5) * BIN_DAYS);
                let observer = sum / count as f64;
                (
                    bin,
                    Bin {
                        time,
                        observer,
                        spread: 0.0,
                        sigma,
                    },
                )
            })
            .collect();
        for (_, sight) in all() {
            if let Some(bin) = bins.get_mut(&key(sight)) {
                bin.spread = bin.spread.max((sight.observer - bin.observer).norm());
            }
        }
        Self { bins, cells }
    }

    /// The pieces with a sight in a cell within the reach of the prediction
    /// of `orbit` in its time bin, sorted and distinct, see [`merge_pieces`].
    fn near(&self, orbit: &Orbit) -> Vec<usize> {
        let Some(covariance) = &orbit.covariance else {
            return Vec::new();
        };
        let mut found = Vec::new();
        for (&key, bin) in &self.bins {
            let Some((direction, range, jacobian)) =
                predicted(&orbit.state, bin.time, bin.observer)
            else {
                continue;
            };
            let half = 0.5 * BIN_DAYS;
            let mut motion = 0.0_f64;
            for edge in [bin.time.jd() - half, bin.time.jd() + half] {
                let Some(there) = direction_at(&orbit.state, Time::new(edge), bin.observer) else {
                    motion = f64::INFINITY;
                    break;
                };
                motion = motion.max(angle(&direction, &there));
            }
            let mut variance = 0.0;
            for row in &jacobian {
                for i in 0..6 {
                    for j in 0..6 {
                        variance += row[i] * covariance[i][j] * row[j];
                    }
                }
            }
            let spread = motion + (bin.spread / range).min(1.0).asin();
            let miss = (2.0 * FIT_REJECT_THRESHOLD * bin.sigma.mul_add(bin.sigma, variance)).sqrt();
            let reach = spread + miss;
            if !reach.is_finite() || reach > MAX_REACH {
                continue;
            }
            let (ra, dec) = direction.to_ra_dec();
            for pixel in query_disc_nested(NSIDE, ra, dec, reach + max_pixel_radius(NSIDE)) {
                if let Some(pieces) = self.cells.get(&(key, pixel)) {
                    found.extend_from_slice(pieces);
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }
}

/// Confirmed objects with those which are pieces of one object merged.
///
/// A search can confirm one object in several pieces, which share no
/// observation: a pass confirms part of its observations and a later pass
/// the part the first left out, the tiles of a pass each hold part of its
/// arc, or the windows of time of [`crate::search_windows`] each hold part
/// of it. Each two pieces whose orbits agree are fit again as one candidate,
/// and replaced by that fit if it is confirmed.
///
/// Each piece's orbit is a two-body orbit about the Sun fit to its own
/// optical observations, from the state of its orbit fit, with the
/// covariance of that fit's normal matrix, scaled by the square of the
/// reduced residual of the fit where that is above one: a piece can be
/// confirmed with residuals up to `max_rms`, and the two-body orbit leaves
/// residuals of its own over a long arc.
///
/// The observations of the pieces are indexed in bins, each a quarter of a
/// day of time and a cube of observer positions 0.001 AU wide, and by cell
/// of the sky within each bin. The orbit of each piece predicts its
/// direction in each bin. Two pieces are compared if, in some bin, the
/// other piece has an observation in a cell of the sky within the reach of
/// that prediction. Directions are predicted from the mean observer of the
/// bin, and the reach
/// adds the motion of the object over half a bin, the parallax of the
/// farthest observer of the bin from that mean, and the miss the prediction
/// test below accepts for one observation: the square root of twice the
/// outlier threshold of the orbit fit of [`crate::confirm()`] times the sum of
/// the largest variance of an observation of the bin and the variance of the
/// prediction. The test bounds the sum over all of a piece's observations, so
/// a piece of several observations whose miss is shared unevenly can pass it
/// with one observation beyond this reach; the lookup from the other piece's
/// side, and the other bins of a piece seen in several, find most of those,
/// at far fewer pairs than a reach for the whole sum. A prediction whose
/// reach is more than 0.05 radians is not used: the orbit is too poorly
/// known there, and the other piece looks from its side.
///
/// The orbit of piece A predicts piece B if the chi-square of B's
/// residuals under A's orbit, with A's uncertainty, is at most twice the
/// outlier threshold of the orbit fit for each of B's optical observations,
/// two measurements each: B's observations fit A's orbit, moved within its
/// uncertainty, no worse on average than an observation the orbit fit keeps.
/// The test only picks the pairs to fit, and the fit of the two pieces as
/// one decides. It is as loose as the outlier rule of the orbit fit rather
/// than held to `max_rms`, since the observations a later pass confirms as
/// a second piece can be ones the fit of the first rejected, and the fit of
/// the union is what judges them. That chi-square, over the residuals `r`
/// of B and their Jacobian `J` over A's state, is
/// `r^T (I + J N^-1 J^T)^-1 r` for the normal matrix `N` of A's own
/// residuals, which is `r^T r - g^T (N + J^T J)^-1 g` for `g = J^T r`. It is
/// also, to first order, how much a joint fit raises the chi-square of A's
/// fit, so it needs no inverse of `N`. Two pieces are fit as one if either
/// predicts the other, from the orbit of the one that does, or of the one
/// with more observations if both do.
///
/// A merge is accepted if the fit of the observations of both pieces is
/// confirmed by the rules of [`crate::confirm()`], keeps at least two thirds
/// of the observations of each piece, and keeps more observations than either
/// piece alone. The two-thirds rule of [`crate::confirm()`], applied to each
/// piece here, stops a fit of a large piece and a small piece of another
/// object from dropping the small one.
///
/// Merges go in rounds. In the first round every piece is new, and in each
/// later round the merged objects of the round before are. In each round,
/// every two pieces of which at least one is new are compared, and fit if
/// one predicts the other. The confirmed fits are accepted in the order of
/// the overlap rule of [`crate::confirm()`], with ties going to the earlier
/// pieces. A fit is accepted only if neither of its pieces was merged
/// before it in the round. Merged objects are compared again in the next
/// round, as new pieces, so that an object in three pieces is merged whole
/// and every merge is tested on the merged orbit. Rounds end when none
/// merges.
///
/// # Arguments
///
/// * `observations` - The observations the sights of the pieces index.
/// * `pieces` - Confirmed objects sharing no observation, such as from the
///   passes of a search.
/// * `rules` - The rules of [`crate::confirm()`].
///
/// # Returns
///
/// The pieces, each merged object in the place of its first piece and the
/// other piece removed. Observations of the pieces the fit of a merge
/// dropped are in no object.
///
/// # Errors
///
/// [`Error::ValueError`] if the rules are not valid, see [`ConfirmRules`], if
/// a piece indexes past the observations, an observation is in two pieces or
/// twice in one, or an optical observation of a piece has a line of sight
/// which is not [`Sight::is_usable`]. Fails if the loaded kernels do not
/// cover the Sun at the epoch of an optical observation of a piece, or the
/// planets and the Moon the orbit fit pulls with at the first and last epoch
/// of the observations of the pieces, or if an orbit fit fails for want of
/// data the kernels do not hold.
pub fn merge_pieces(
    observations: &[AstrometricObservation],
    pieces: Vec<Confirmed>,
    rules: &ConfirmRules,
) -> KeteResult<Vec<Confirmed>> {
    rules.check()?;
    let mut used = vec![false; observations.len()];
    for &index in pieces.iter().flat_map(|piece| &piece.sights) {
        if index >= observations.len() {
            return Err(Error::ValueError(
                "Merge pieces: a piece indexes past the observations".into(),
            ));
        }
        if used[index] {
            return Err(Error::ValueError(format!(
                "Merge pieces: observation {index} is in more than one piece, or twice in one"
            )));
        }
        used[index] = true;
    }
    let sights = sights_of(
        observations,
        pieces.iter().flat_map(|p| p.sights.iter().copied()),
    )?;
    {
        let ephemeris = SpiceEphemeris::loaded()?;
        let times = pieces
            .iter()
            .flat_map(|piece| &piece.sights)
            .map(|&i| observations[i].epoch().jd());
        check_force_model(&ephemeris, times)?;
    }
    let factors = vec![1.0; observations.len()];
    let scaled = Scaled {
        sights: &sights,
        factors: &factors,
    };
    merge(observations, scaled, pieces, rules)
}

/// [`merge_pieces`] without the checks of the rules and of the pieces, with
/// `sights` holding the line of sight of every optical observation of the
/// pieces. The orbits of the pieces, the prediction test and the fits of the
/// merges see the uncertainties of the observations as `sights` scales them.
pub(crate) fn merge(
    observations: &[AstrometricObservation],
    sights: Scaled<'_>,
    pieces: Vec<Confirmed>,
    rules: &ConfirmRules,
) -> KeteResult<Vec<Confirmed>> {
    let piece = |object: Confirmed| -> Piece {
        let own: Vec<Sight> = object
            .sights
            .iter()
            .filter_map(|&i| sights.sight(i))
            .collect();
        let orbit = Orbit::new(&object, &own);
        Piece {
            object,
            sights: own,
            orbit,
        }
    };
    let mut pieces: Vec<Option<Piece>> = pieces.into_par_iter().map(|p| Some(piece(p))).collect();
    let mut new = vec![true; pieces.len()];
    while new.iter().any(|&n| n) {
        // Pieces which are not new look only for new ones, in an index of
        // those alone, after the first round.
        let index = Index::new(&pieces, None);
        let fresh = new
            .contains(&false)
            .then(|| Index::new(&pieces, Some(&new)));
        let orbit = |k: usize| pieces[k].as_ref()?.orbit.as_ref();
        let mut pairs: Vec<(usize, usize)> = (0..pieces.len())
            .into_par_iter()
            .filter_map(|a| Some((a, orbit(a)?)))
            .flat_map_iter(|(a, orbit)| {
                let near = match &fresh {
                    Some(fresh) if !new[a] => fresh.near(orbit),
                    _ => index.near(orbit),
                };
                let new = &new;
                near.into_iter()
                    .filter(move |&b| b != a && (new[a] || new[b]))
                    .map(move |b| (a.min(b), a.max(b)))
            })
            .collect();
        pairs.sort_unstable();
        pairs.dedup();
        // The piece whose orbit starts the fit of each pair, and the other.
        let size = |k: usize| pieces[k].as_ref().map_or(0, |p| p.object.sights.len());
        let predicts = |a: usize, b: usize| {
            let (Some(orbit), Some(other)) = (orbit(a), &pieces[b]) else {
                return false;
            };
            orbit.predicts(&other.sights)
        };
        let tested: Vec<(usize, usize, State<Equatorial, SunCenter>)> = pairs
            .into_par_iter()
            .filter_map(|(a, b)| {
                let (a, b) = if size(b) > size(a) { (b, a) } else { (a, b) };
                let (a, b) = if predicts(a, b) {
                    (a, b)
                } else if predicts(b, a) {
                    (b, a)
                } else {
                    return None;
                };
                Some((a, b, orbit(a)?.state.clone()))
            })
            .collect();
        // The lock is released before the fits take it again.
        let starts = {
            let ephemeris = SpiceEphemeris::loaded()?;
            tested
                .iter()
                .map(|(_, _, state)| ephemeris.try_to_ssb(state.clone().into()))
                .collect::<KeteResult<Vec<_>>>()?
        };
        let merged: Vec<Option<(usize, usize, Confirmed)>> = tested
            .into_par_iter()
            .zip(starts)
            .map(|((a, b, state), start)| -> KeteResult<_> {
                let (Some(first), Some(second)) = (&pieces[a], &pieces[b]) else {
                    return Ok(None);
                };
                let (first, second) = (&first.object, &second.object);
                let mut union = [first.sights.as_slice(), second.sights.as_slice()].concat();
                union.sort_unstable();
                let candidate = Candidate {
                    sights: union,
                    state,
                };
                let Some(found) = confirm_one(observations, sights, &candidate, &start, rules)?
                else {
                    return Ok(None);
                };
                let kept = |piece: &Confirmed| {
                    let count = piece
                        .sights
                        .iter()
                        .filter(|i| found.sights.binary_search(i).is_ok())
                        .count();
                    count >= (2 * piece.sights.len()).div_ceil(3)
                };
                let gains = found.sights.len() > first.sights.len().max(second.sights.len());
                Ok((gains && kept(first) && kept(second)).then_some((a.min(b), a.max(b), found)))
            })
            .collect::<KeteResult<_>>()?;
        let mut merged: Vec<(usize, usize, Confirmed)> = merged.into_iter().flatten().collect();
        merged.sort_by(|(a, b, x), (c, d, y)| rank(x, y).then((a, b).cmp(&(c, d))));
        new.fill(false);
        for (a, b, found) in merged {
            if new[a] || new[b] || pieces[a].is_none() || pieces[b].is_none() {
                continue;
            }
            pieces[a] = Some(piece(found));
            pieces[b] = None;
            new[a] = true;
        }
    }
    Ok(pieces.into_iter().flatten().map(|p| p.object).collect())
}
