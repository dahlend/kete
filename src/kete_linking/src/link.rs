// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Grouping the orbits of units into candidate objects.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::{Mutex, PoisonError};

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, SunCenter, Vector};
use kete_core::kepler::propagate_two_body;
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::time::{TDB, Time};
use rayon::prelude::*;

use crate::hypothesis::check_hypothesis;
use crate::linalg::dot;
use crate::sight::{UNUSABLE, first_and_last};
use crate::unit_fit::{Fit, plane_axes, solve_unit};
use crate::{Hypothesis, MIN_ALIGNMENT, Sight};

/// Width of the finest hash cells, in AU. The solve itself is no more exact
/// than about a ten-billionth of the range, so no tolerance is finer than
/// this.
const MIN_CELL: f64 = 1e-9;

/// Ratio of the cell widths of neighboring levels of the hash.
const LEVEL_RATIO: f64 = 4.0;

/// A hasher for the integer keys of grouping, such as hash cells and the
/// indices of sights, cheaper than the default one, which is built to resist
/// keys chosen by an adversary, which these are not. Matching looks up many
/// cells for each unit, so the hasher is on its inner loop.
/// Each word is combined with the hash by an exclusive or, and the result
/// mixed by the finalizer of `SplitMix64` (Steele, Lea and Flood 2014), a
/// bijection in which every bit of the input moves about half the bits of
/// the output. A rotate and multiply alone, as in the hasher of the Rust
/// compiler, carries no high bits of a key into the low bits of the hash
/// that pick its bucket, and keys mirrored about zero collide on it.
#[derive(Debug, Default)]
struct CellHasher(u64);

impl CellHasher {
    /// Mix one word into the hash.
    fn add(&mut self, word: u64) {
        let mut z = (self.0 ^ word).wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        self.0 = z ^ (z >> 31);
    }
}

impl Hasher for CellHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_i32(&mut self, value: i32) {
        self.add(u64::from(value.cast_unsigned()));
    }

    fn write_i64(&mut self, value: i64) {
        self.add(value.cast_unsigned());
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

/// Settings of [`link()`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinkSettings {
    /// How far two units may miss each other, in units of the error that
    /// astrometric noise gives the miss on one axis of the sky. For a miss
    /// with two axes, noise alone takes one comparison past this with
    /// probability about `exp(-threshold^2 / 2)`. A match of two pairs takes
    /// four comparisons, each pair against both sights of the other, so noise
    /// loses a true match with probability at most about four times that.
    /// A run whose own sights miss its orbit by more than this times their
    /// uncertainty, plus the grid allowance, is left out.
    pub threshold: f64,

    /// Miss allowed for the grid, as an angle in radians at the sight being
    /// matched. It is the same for every match. A grid step leaves the
    /// hypothesis wrong by up to half a step, which spreads the units of one
    /// object more the farther apart they are in time, so the allowance has
    /// to suit the longest time between units in a search.
    pub grid_allowance: f64,

    /// Largest position uncertainty of a unit at the reference time, as a
    /// fraction of its range, for the unit to take part in grouping. Positive.
    /// The uncertainty is the error of the position at the time it is best
    /// known plus the velocity error times the time from there to the
    /// reference time, each on its worst axis, the sum the reach of a unit
    /// is built on, see [`link()`]. It should be no more than the step of the
    /// hypothesis grid in the natural log of the depth, since neighboring
    /// depths are that fraction of the range apart. A unit less certain than
    /// that cannot tell them apart: it matches under each of them, adding false
    /// matches and no information. A unit at this limit reaches `threshold`
    /// times it from its position at the reference time, plus the grid
    /// allowance, so a value of the depth step over `threshold` or less keeps
    /// the reach that noise gives every grouped unit within one depth step.
    pub max_uncertainty: f64,
}

impl Default for LinkSettings {
    /// A threshold of four loses a true match to noise at most about once in
    /// eight hundred, by the relation of [`LinkSettings::threshold`], which
    /// the test `match_noise_calibration` checks at lower thresholds where
    /// losses can be counted. The grid allowance of about twenty arcseconds
    /// keeps groups tight at the steps [`crate::DEPTH_STEP`] and
    /// [`crate::RATE_STEP`]: a main belt or more distant object which a step
    /// leaves outside it is split into several candidates rather than lost.
    /// The test `matrix_grid_allowance` prints how whole it keeps main belt
    /// objects seen nightly for a month. A larger allowance splits fewer
    /// objects but lets more units of different objects match, which costs
    /// purity where a search forms many of them, see `matrix_cadence`. The
    /// largest uncertainty of 0.05 is under [`crate::DEPTH_STEP`] over the
    /// default threshold, see [`LinkSettings::max_uncertainty`].
    fn default() -> Self {
        Self {
            threshold: 4.0,
            grid_allowance: 1e-4,
            max_uncertainty: 0.05,
        }
    }
}

/// A group of sights which could be one object.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// Indices of the sights, sorted and distinct.
    pub sights: Vec<usize>,

    /// State of the object at the reference time of the hypothesis it was
    /// found under, from the unit it grew from.
    pub state: State<Equatorial, SunCenter>,
}

/// Candidate objects among the sights.
///
/// A unit is two or more sights which could be one object, such as from
/// [`crate::form_units`]: a run of sights close in time, or a pair across a
/// gap. Under each hypothesis, every unit whose sights are all aligned with
/// its direction is solved, see below. Units whose position at the reference
/// time is too uncertain to group are left out, and so are runs whose own
/// sights miss their orbit by more than `threshold` times their uncertainty
/// plus `grid_allowance`, such as a run joined across two objects.
///
/// Each solved unit then seeds a group, grown from the units which match the
/// seed, taken best match first. The best one joins. Each later one joins
/// only if it also matches a unit which has already joined, other than the
/// seed. Every member matches the seed itself: a unit which matches only a
/// member, and not the seed, does not join, so there is no chaining from one
/// match to the next. In a crowded field some unit of another object matches
/// nearly any seed by chance, and without the rule on later units every
/// group of an object would hold one, more than confirmation can drop; such
/// a unit seldom matches the seed's other units too. Asking more of it, such
/// as matching half of the group, parts objects whose units far apart in
/// time predict only their neighbors.
///
/// A unit of two sights has one orbit under a hypothesis, which passes
/// through both. A unit of more is fit by least squares within the
/// hypothesis, from the orbit through its first and last sight, with exact
/// two-body motion, light time, and the observer of each sight. The fit gives
/// the covariance of the four numbers the hypothesis leaves free: the
/// position on its plane and the velocity along it at the reference time.
/// The position is best known at some time near the sights. At any other
/// time, the error of a unit is the position error at the time it is best
/// known and the velocity error times the time from there, added in
/// quadrature, each on its worst axis. The plane of the hypothesis
/// stretches the error of a sight along one axis by the inverse of the
/// cosine between sight and direction, which the covariance holds.
///
/// Two units match when the orbit of each, carried to the time of each sight
/// of the other, passes within the allowed miss of the other's position
/// there. The allowed miss is `threshold` times the error that noise gives
/// the miss on one axis of the sky, plus `grid_allowance` times the range of
/// the sight. The error adds, in quadrature, the error of the carried unit at
/// the time it is carried to and the position error of the other's sight,
/// which is its range times the largest uncertainty over cosine among the
/// sights of its unit. The tolerance so follows from the data, and a short
/// unit is not held to the standard of a long one.
///
/// Units to compare are found by hashing their positions at the reference
/// time. Each unit has a reach there: `threshold` times the sum of its
/// position error and its velocity error carried to the reference time, plus
/// the grid allowance at its range. Every two units whose positions at the
/// reference time are within the sum of their reaches are compared, and no
/// others. Reaches differ by orders of magnitude between units minutes long
/// and units days long, so the hash has levels of cell width, each four
/// times as wide as the one below. Each unit is hashed at the first level
/// whose cells are at least as wide as its reach, and each two units are
/// found from the side with the smaller reach, which looks through the
/// cells near it at its own level and every level above. Two units which
/// would match but are farther apart than that are not compared.
/// That happens when the reference time is outside the span of the sights:
/// carried past their sights, the orbits of a wrong hypothesis drift apart.
/// The test `hashed_matches_equal_brute_force` measures it. A reference time
/// farther outside the span of the sights of the units than that span is
/// refused.
///
/// A group is the sights of its units. Two sights at the same time from the
/// same place cannot both be one object, and the one of the better matching
/// unit is kept. A group with at least `min_sights` sights is a candidate.
/// Of candidates with the same sights, the one whose units matched best on
/// average is kept; a unit which matched none is the worst match. Candidates
/// whose sights are all in another candidate are removed, across all
/// hypotheses. The candidates are returned with the most sights first, then
/// the best matching first.
///
/// Ties between equal scores go to the lower index: among the members of a
/// group to the unit of lower index, among equal groups under one hypothesis
/// to the seed of lower index, and among equal groups under different
/// hypotheses to the hypothesis of lower index. The result does not depend
/// on how the work is spread over threads.
///
/// Grouping is generous. Purity comes from confirming the candidates with an
/// orbit fit.
///
/// # Arguments
///
/// * `sights` - All lines of sight.
/// * `units` - Groups of two or more indices into `sights`, each in time
///   order, such as from [`crate::form_units`].
/// * `hypotheses` - Hypotheses to search under, such as from
///   [`crate::hypothesis_grid`]. They need not share a reference time.
/// * `min_sights` - Fewest distinct sights in a candidate.
/// * `settings` - See [`LinkSettings`].
///
/// # Errors
///
/// [`Error::ValueError`] if a sight is not [`Sight::is_usable`], a unit is
/// not two or more sights in time order spanning some time, a unit holds an
/// index past the end of `sights`, a hypothesis is not finite or its
/// direction not a unit vector, a reference time is more than 9e15 days from
/// JD 0 or farther outside the span of the sights of the units than that
/// span, a setting is negative or not finite, `max_uncertainty` is zero, or
/// `threshold` and `grid_allowance` are both zero.
pub fn link(
    sights: &[Sight],
    units: &[Vec<usize>],
    hypotheses: &[Hypothesis],
    min_sights: usize,
    settings: &LinkSettings,
) -> KeteResult<Vec<Candidate>> {
    settings.check()?;
    check_inputs(sights, units, hypotheses)?;
    let merged = Mutex::new(Merged::default());
    let direction =
        |index: &usize| <[f64; 3]>::from(hypotheses[*index].direction).map(f64::to_bits);
    let mut order: Vec<usize> = (0..hypotheses.len()).collect();
    order.sort_by_key(direction);
    // Hypotheses of one direction, with the units aligned with it, so that
    // each hypothesis looks only at units it can solve.
    for same in order.chunk_by(|i, j| direction(i) == direction(j)) {
        let aligned = aligned_units(sights, units, &hypotheses[same[0]].direction);
        same.par_iter().for_each(|&index| {
            let hypothesis = &hypotheses[index];
            let solved = solve_units(sights, units, &aligned, hypothesis, settings);
            let mut matches = matches(&solved, settings, hypothesis);
            // The groups of one hypothesis are reduced before they are
            // merged with the rest, which is one thread at a time.
            let mut found = Merged::default();
            groups(
                sights,
                units,
                &solved,
                &mut matches,
                min_sights,
                index,
                &mut found,
            );
            drop((solved, matches));
            found.drop_subsets();
            merged
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .absorb(found);
        });
    }
    let merged = merged.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(merged.candidates())
}

/// The indices of the units whose sights are all aligned with `direction`
/// by at least [`MIN_ALIGNMENT`], in order. No other unit can be solved
/// under a hypothesis along it.
fn aligned_units(
    sights: &[Sight],
    units: &[Vec<usize>],
    direction: &Vector<Equatorial>,
) -> Vec<usize> {
    let aligned = |index: &usize| direction.dot(&sights[*index].direction) >= MIN_ALIGNMENT;
    (0..units.len())
        .into_par_iter()
        .filter(|&u| units[u].iter().all(aligned))
        .collect()
}

/// The largest uncertainty of the sights of a unit on one axis of the sky,
/// stretched by the plane of the hypothesis, in radians.
fn stretched(sights: &[Sight], unit: &[usize], hypothesis: &Hypothesis) -> f64 {
    let direction = hypothesis.direction;
    unit.iter()
        .map(|&i| sights[i].sigma / direction.dot(&sights[i].direction))
        .fold(0.0, f64::max)
}

impl LinkSettings {
    /// Check that the settings are valid.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if a setting is negative or not finite,
    /// `max_uncertainty` is zero, or `threshold` and `grid_allowance` are
    /// both zero.
    pub(crate) fn check(&self) -> KeteResult<()> {
        let Self {
            threshold,
            grid_allowance,
            max_uncertainty,
        } = *self;
        let valid = [threshold, grid_allowance, max_uncertainty]
            .iter()
            .all(|value| *value >= 0.0 && value.is_finite());
        if !valid {
            return Err(Error::ValueError(
                "Link: settings must be finite and not negative".into(),
            ));
        }
        if max_uncertainty == 0.0 {
            return Err(Error::ValueError(
                "Link: max_uncertainty is zero, so every unit is left out".into(),
            ));
        }
        if threshold == 0.0 && grid_allowance == 0.0 {
            return Err(Error::ValueError(
                "Link: threshold and grid_allowance are both zero, so no two units can match"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// The errors of [`link()`] for its inputs other than the settings.
fn check_inputs(
    sights: &[Sight],
    units: &[Vec<usize>],
    hypotheses: &[Hypothesis],
) -> KeteResult<()> {
    if let Some(bad) = sights.iter().position(|sight| !sight.is_usable()) {
        return Err(Error::ValueError(format!(
            "Link: sight {bad} has {UNUSABLE}"
        )));
    }
    let jd = |i: usize| sights[i].time.jd();
    let bad_unit = |unit: &Vec<usize>| {
        unit.len() < 2
            || unit.iter().any(|&i| i >= sights.len())
            || unit.windows(2).any(|w| jd(w[0]) > jd(w[1]))
            || jd(unit[0]) >= jd(unit[unit.len() - 1])
    };
    if let Some(bad) = units.iter().position(bad_unit) {
        return Err(Error::ValueError(format!(
            "Link: unit {bad} is not two or more sights in time order over a span of time"
        )));
    }
    // Matches are complete only when the reference time is within the span
    // of the sights, see `hashed_matches_equal_brute_force`.
    let (first, last) = first_and_last(units.iter().flatten().map(|&i| jd(i)));
    for (index, hypothesis) in hypotheses.iter().enumerate() {
        check_hypothesis(hypothesis)
            .map_err(|err| Error::ValueError(format!("Link: hypothesis {index}: {err}")))?;
        let epoch = hypothesis.origin.epoch.jd();
        if let Some(outside) = refused_outside(epoch, (first, last)) {
            return Err(Error::ValueError(format!(
                "Link: the reference time of hypothesis {index} is {outside} days outside the \
                 span of the sights of the units, more than the span itself"
            )));
        }
    }
    Ok(())
}

/// Days a reference time `epoch` is outside the span `first` to `last` of
/// the sights of the units, if that is more than the span, which [`link()`]
/// refuses. `None` also when there are no units, an empty span.
fn refused_outside(epoch: f64, (first, last): (f64, f64)) -> Option<f64> {
    let outside = (first - epoch).max(epoch - last);
    (first <= last && outside > last - first).then_some(outside)
}

/// One sight of a solved unit, at its range under the orbit.
#[derive(Debug)]
struct Point {
    /// Time light left the object.
    time: Time<TDB>,

    /// Position along the sight at the range.
    pos: Vector<Equatorial>,

    /// Range along the sight, in AU.
    range: f64,
}

/// A unit solved under one hypothesis.
#[derive(Debug)]
struct Solved {
    /// Index of the unit.
    unit: usize,

    /// State at the reference time.
    state: State<Equatorial, SunCenter>,

    /// The sights, as solved.
    points: Vec<Point>,

    /// Largest uncertainty of the direction of a sight on one axis of the
    /// sky, stretched by the plane of the hypothesis, in radians.
    sigma: f64,

    /// Days from the reference time to the time the position is best known,
    /// from the covariance of the fit.
    ahead: f64,

    /// Error of the position on its worst axis at the time it is best known,
    /// in AU.
    pos_error: f64,

    /// Error of the velocity on its worst axis, in AU per day.
    vel_error: f64,

    /// Acceleration toward the Sun at the reference time, in AU per day
    /// squared.
    accel: Vector<Equatorial>,

    /// Largest rate of change of the acceleration anywhere on the orbit, in
    /// AU per day cubed.
    jerk: f64,
}

impl Solved {
    /// Largest miss of this unit, carried to each sight of `other`, as a
    /// fraction of the allowed miss there. Infinite if it cannot be carried.
    ///
    /// Once one sight is missed by more than allowed, the rest are not
    /// looked at, and the value is only known to be over one.
    ///
    /// The second order Taylor series of the orbit about the reference time
    /// is off by at most a sixth of the largest jerk on the orbit times the
    /// cube of the time from the reference. A sight farther from the series
    /// than that plus the allowed miss is missed, and the orbit is not
    /// propagated to it.
    fn miss(&self, other: &Self, settings: &LinkSettings) -> f64 {
        let mut worst = 0.0_f64;
        for point in &other.points {
            let dt = (point.time - self.state.epoch).elapsed;
            // A velocity error carried from the time the position is best
            // known, and the position errors of both units, on one axis.
            let error = (self.vel_error * (dt - self.ahead).abs())
                .hypot(self.pos_error)
                .hypot(other.sigma * point.range);
            let allowed = settings.threshold * error + settings.grid_allowance * point.range;
            if self.least(dt, &point.pos) > allowed {
                return f64::INFINITY;
            }
            let Ok(carried) = propagate_two_body(&self.state, point.time) else {
                return f64::INFINITY;
            };
            worst = worst.max((carried.pos - point.pos).norm() / allowed);
            if worst > 1.0 {
                break;
            }
        }
        worst
    }

    /// Least distance from the orbit `dt` days after the reference time to
    /// `pos`: the distance from the second order Taylor series there, less
    /// the bound on its remainder. Not more than the distance from the
    /// propagated orbit, see `taylor_bound_is_a_lower_bound`.
    fn least(&self, dt: f64, pos: &Vector<Equatorial>) -> f64 {
        let series = self.state.pos + self.state.vel * dt + self.accel * (0.5 * dt * dt);
        (series - *pos).norm() - self.jerk * dt.abs().powi(3) / 6.0
    }

    /// Distance from the position at the reference time within which this
    /// unit's matches are expected there.
    fn reach(&self, settings: &LinkSettings) -> f64 {
        settings.threshold * self.uncertainty() + settings.grid_allowance * self.points[0].range
    }

    /// Error of the position at the reference time: the error at the time it
    /// is best known plus the velocity error over the time between.
    fn uncertainty(&self) -> f64 {
        self.vel_error * self.ahead.abs() + self.pos_error
    }
}

/// Acceleration toward the Sun of `state`, in AU per day squared, and the
/// largest rate of change of the acceleration anywhere on its orbit, in AU
/// per day cubed.
///
/// The jerk is `GM (3 r (r . v) / |r|^2 - v) / |r|^3`, whose size is at most
/// `2 GM |v| / |r|^3`. Speed is largest and distance least at perihelion,
/// where the speed is the angular momentum over the distance, so that bounds
/// it over the whole orbit, bound or not.
fn accel_and_jerk(state: &State<Equatorial, SunCenter>) -> (Vector<Equatorial>, f64) {
    let (pos, vel) = (state.pos, state.vel);
    let momentum = pos.cross(&vel).norm();
    let eccentricity = (vel.cross(&pos.cross(&vel)) / GMS - pos / pos.norm()).norm();
    let perihelion = momentum * momentum / GMS / (1.0 + eccentricity);
    (
        -pos * (GMS / pos.norm().powi(3)),
        2.0 * GMS * momentum / perihelion.powi(4),
    )
}

/// Every unit of `aligned`, which are aligned with the hypothesis, solved,
/// without those too uncertain at the reference time to group, or whose own
/// sights miss their orbit by more than allowed.
///
/// Room for every unit is taken at once. Grown step by step instead, the
/// list is copied by the system allocator each time it outgrows its room,
/// and many threads copying large lists at once can stall each other in the
/// kernel for far longer than the solves take.
fn solve_units(
    sights: &[Sight],
    units: &[Vec<usize>],
    aligned: &[usize],
    hypothesis: &Hypothesis,
    settings: &LinkSettings,
) -> Vec<Solved> {
    let mut out = Vec::with_capacity(aligned.len());
    out.extend(aligned.iter().filter_map(|&u| {
        let fit = solve_unit(sights, &units[u], hypothesis).ok()?;
        solved(sights, units, u, fit, hypothesis, settings)
    }));
    out
}

/// Largest eigenvalue of a symmetric two by two matrix.
fn largest(m: [[f64; 2]; 2]) -> f64 {
    let half = f64::midpoint(m[0][0], m[1][1]);
    let diff = 0.5 * (m[0][0] - m[1][1]);
    half + diff.hypot(m[0][1])
}

/// A unit solved under a hypothesis, unless its position at the reference
/// time is too uncertain to group, or one of its sights misses its orbit by
/// more than `threshold` times its uncertainty plus the grid allowance.
fn solved(
    sights: &[Sight],
    units: &[Vec<usize>],
    u: usize,
    fit: Fit,
    hypothesis: &Hypothesis,
    settings: &LinkSettings,
) -> Option<Solved> {
    let unit = &units[u];
    // The residuals are over the sigma of each sight, so this is the miss as
    // an angle against `threshold` sigma plus the allowance.
    let consistent = unit.iter().zip(&fit.rows).all(|(&i, row)| {
        let miss = dot(&row.residual, &row.residual).sqrt();
        miss * sights[i].sigma <= settings.threshold * sights[i].sigma + settings.grid_allowance
    });
    if !consistent {
        return None;
    }
    // Collected from the rows they replace, whose room the list reuses.
    let points: Vec<Point> = fit
        .rows
        .into_iter()
        .zip(unit)
        .map(|(row, &i)| Point {
            time: row.time,
            pos: sights[i].position(row.range),
            range: row.range,
        })
        .collect();
    // The covariance at the reference time, carried in a straight line to
    // the time the position is best known. A covariance with a variance that
    // is not positive, as from a normal matrix too near singular to invert,
    // would make the unit look certain, and the unit is left out. So is one
    // whose position error at that time is not real.
    let c = &fit.covariance;
    let block = |r: usize, k: usize| [[c[r][k], c[r][k + 1]], [c[r + 1][k], c[r + 1][k + 1]]];
    let (pp, pv, vv) = (block(0, 0), block(0, 2), block(2, 2));
    let positive = [pp[0][0], pp[1][1], vv[0][0], vv[1][1]]
        .iter()
        .all(|&v| v > 0.0);
    if !positive {
        return None;
    }
    let ahead = -(pv[0][0] + pv[1][1]) / (vv[0][0] + vv[1][1]);
    let mut at = pp;
    for i in 0..2 {
        for j in 0..2 {
            at[i][j] += ahead * (pv[i][j] + pv[j][i]) + ahead * ahead * vv[i][j];
        }
    }
    let (accel, jerk) = accel_and_jerk(&fit.state);
    let solved = Solved {
        unit: u,
        accel,
        jerk,
        state: fit.state,
        sigma: stretched(sights, unit, hypothesis),
        ahead,
        pos_error: largest(at).sqrt(),
        vel_error: largest(vv).sqrt(),
        points,
    };
    let uncertainty = solved.uncertainty();
    let certain =
        uncertainty.is_finite() && uncertainty <= settings.max_uncertainty * solved.points[0].range;
    certain.then_some(solved)
}

/// For each solved unit, the units it matches and how well, as the larger of
/// the two misses as a fraction of the allowed miss.
///
/// A unit is hashed at the first level whose cells, `MIN_CELL` times a power
/// of `LEVEL_RATIO` wide, are at least as wide as its reach. It then looks
/// for the units of larger reach, which are at its level or above: at each
/// such level, in the cells within its own reach plus the width of a cell
/// there. Every unit of larger reach within the sum of the two reaches lies
/// in those cells, and a unit looks through a few cells per level however
/// wide its reach is.
///
/// Every solved unit is on the plane of the hypothesis at the reference time,
/// so the cells are squares on that plane. Two units are no farther apart on
/// the plane than in space, so no match is lost by leaving out the depth.
fn matches(
    solved: &[Solved],
    settings: &LinkSettings,
    hypothesis: &Hypothesis,
) -> Vec<Vec<(f64, usize)>> {
    let reach: Vec<f64> = solved.iter().map(|s| s.reach(settings)).collect();
    let [across, along] = plane_axes(&hypothesis.direction);
    let plane = |s: &Solved| [across.dot(&s.state.pos), along.dot(&s.state.pos)];

    let width = |level: i32| MIN_CELL * LEVEL_RATIO.powi(level);
    let cell = |x: f64, level: i32| (x / width(level)).floor() as i64;
    let level: Vec<i32> = reach
        .iter()
        .map(|&r| {
            let k = (r / MIN_CELL).max(1.0).log(LEVEL_RATIO).ceil() as i32;
            // The logarithm can round down past a whole power.
            if width(k) < r { k + 1 } else { k }
        })
        .collect();
    let mut cells: HashMap<(i32, [i64; 2]), Vec<usize>, BuildHasherDefault<CellHasher>> =
        HashMap::default();
    for (index, s) in solved.iter().enumerate() {
        let key = plane(s).map(|x| cell(x, level[index]));
        cells.entry((level[index], key)).or_default().push(index);
    }
    let mut levels = level.clone();
    levels.sort_unstable();
    levels.dedup();

    let mut found = vec![Vec::new(); solved.len()];
    for (i, unit) in solved.iter().enumerate() {
        // Compare with the units of larger reach, so that each two units are
        // compared once. When this unit misses the other, the other need not
        // be carried back.
        let mut compare = |j: usize| {
            if (reach[j], j) <= (reach[i], i) {
                return;
            }
            let apart = (unit.state.pos - solved[j].state.pos).norm();
            if apart > reach[i] + reach[j] {
                return;
            }
            let there = unit.miss(&solved[j], settings);
            if there > 1.0 {
                return;
            }
            let score = there.max(solved[j].miss(unit, settings));
            if score <= 1.0 {
                found[i].push((score, j));
                found[j].push((score, i));
            }
        };
        let pos = plane(unit);
        for &k in levels.iter().filter(|&&k| k >= level[i]) {
            let near = reach[i] + width(k);
            let [lo, hi] = [-near, near].map(|d| pos.map(|x| cell(x + d, k)));
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    if let Some(list) = cells.get(&(k, [x, y])) {
                        list.iter().copied().for_each(&mut compare);
                    }
                }
            }
        }
    }
    found
}

/// A candidate without its sights, and how well its units matched on
/// average.
#[derive(Debug)]
struct Found {
    /// State of the object at the reference time, from the seed.
    state: State<Equatorial, SunCenter>,

    /// Mean score of the units matched to the seed, infinite if there are
    /// none, so that a lone seed is the worst match.
    score: f64,

    /// Index of the hypothesis it was found under, and of its seed among the
    /// units solved there.
    order: (usize, usize),
}

/// Of the units matching a seed, best first, the first and those which each
/// match one of the ones kept before them, in `kept`, which is cleared first.
fn agreeing(sorted: &[(f64, usize)], neighbors: &[Vec<usize>], kept: &mut Vec<(f64, usize)>) {
    kept.clear();
    for &member in sorted {
        let agrees = kept.is_empty()
            || kept
                .iter()
                .any(|other| neighbors[other.1].binary_search(&member.1).is_ok());
        if agrees {
            kept.push(member);
        }
    }
}

/// What makes two sights clash in a group: the parts of their time, and
/// their observer. Two usable sights clash when these are equal, and the
/// bits are equal when the values are, once a negative zero is made
/// positive by adding zero.
fn clash_key(sight: &Sight) -> [u64; 5] {
    let (day, frac) = sight.time.jd_parts();
    let [x, y, z] = <[f64; 3]>::from(sight.observer);
    [day, frac, x, y, z].map(|value| (value + 0.0).to_bits())
}

/// The candidates grown from each solved unit under hypothesis `index`,
/// added to `found` as they are grown, so that of equal candidates grown
/// from different seeds only the best is held. The matches of each unit are
/// sorted, best first.
fn groups(
    sights: &[Sight],
    units: &[Vec<usize>],
    solved: &[Solved],
    matches: &mut [Vec<(f64, usize)>],
    min_sights: usize,
    index: usize,
    found: &mut Merged,
) {
    // The clash keys of the sights of the group being grown. A sight already
    // in it has the key of its own time and place.
    let mut taken: HashSet<[u64; 5], BuildHasherDefault<CellHasher>> = HashSet::default();
    // The units each unit matches, sorted, to look up whether two match.
    let neighbors: Vec<Vec<usize>> = matches
        .iter()
        .map(|matched| {
            let mut near: Vec<usize> = matched.iter().map(|m| m.1).collect();
            near.sort_unstable();
            near
        })
        .collect();
    // The members and the sights of the group being grown, held from one
    // seed to the next for their room.
    let (mut members, mut group) = (Vec::new(), Vec::new());
    for (seed, matched) in matches.iter_mut().enumerate() {
        // The seed, then the units which match it, best first, and of equal
        // scores the lowest index first, each kept if it agrees with the
        // units kept before it.
        matched.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        agreeing(matched, &neighbors, &mut members);
        let score = if members.is_empty() {
            f64::INFINITY
        } else {
            members.iter().map(|m| m.0).sum::<f64>() / members.len() as f64
        };
        group.clear();
        taken.clear();
        for member in std::iter::once(seed).chain(members.iter().map(|m| m.1)) {
            for &index in &units[solved[member].unit] {
                if taken.insert(clash_key(&sights[index])) {
                    group.push(index);
                }
            }
        }
        if group.len() < min_sights {
            continue;
        }
        group.sort_unstable();
        let state = solved[seed].state.clone();
        let order = (index, seed);
        found.insert(
            group.clone(),
            Found {
                state,
                score,
                order,
            },
        );
    }
}

/// The distinct groups found so far, each with the best score it had, keyed
/// by its sights.
///
/// A group within a larger one is never a candidate. Such groups are dropped
/// each time the count of groups has doubled since the last time, so the
/// groups held are never more than twice those which survived that.
#[derive(Debug, Default)]
struct Merged {
    /// The groups, with their sights taken out into the key.
    best: HashMap<Vec<usize>, Found, BuildHasherDefault<CellHasher>>,

    /// Count of groups after those within others were last dropped.
    after_drop: usize,
}

impl Merged {
    /// Add the groups of another, as [`Merged::insert`] does. Groups dropped
    /// from either, as within a larger group, are never candidates, so the
    /// outcome is that of adding all their groups here.
    fn absorb(&mut self, other: Self) {
        for (sights, found) in other.best {
            self.insert(sights, found);
        }
        self.drop_when_doubled();
    }

    /// Add one group with its sights taken out. Of groups with the same
    /// sights, the one with the best score is kept, and of equal scores the
    /// one of lowest order, so the outcome does not depend on the order
    /// groups are added in.
    fn insert(&mut self, sights: Vec<usize>, found: Found) {
        match self.best.entry(sights) {
            Entry::Occupied(mut entry) => {
                let kept = entry.get();
                let better = found.score.total_cmp(&kept.score);
                if better.then(found.order.cmp(&kept.order)).is_lt() {
                    let _ = entry.insert(found);
                }
            }
            Entry::Vacant(entry) => {
                let _ = entry.insert(found);
            }
        }
    }

    /// Drop the groups within others when their count has doubled since
    /// the last time.
    fn drop_when_doubled(&mut self) {
        if self.best.len() > 2 * self.after_drop {
            self.drop_subsets();
            self.after_drop = self.best.len();
        }
    }

    /// Remove the groups whose sights are all in another, larger group.
    ///
    /// A group within a larger one which is itself removed is also within
    /// the group that one is in, so each group is tested against all the
    /// others.
    fn drop_subsets(&mut self) {
        let sets: Vec<&Vec<usize>> = self.best.keys().collect();
        // The groups which have each sight, those of sight `s` at
        // `having[starts[s]..starts[s + 1]]`.
        let size = sets
            .iter()
            .filter_map(|set| set.last())
            .max()
            .map_or(0, |&s| s + 1);
        let mut starts = vec![0; size + 1];
        for &sight in sets.iter().copied().flatten() {
            starts[sight + 1] += 1;
        }
        for s in 0..size {
            starts[s + 1] += starts[s];
        }
        let mut next = starts.clone();
        let mut having = vec![0; starts[size]];
        for (k, set) in sets.iter().enumerate() {
            for &sight in *set {
                having[next[sight]] = k;
                next[sight] += 1;
            }
        }
        let within = |set: &[usize]| {
            // A group which holds every sight of this one is on the list of
            // each of them, so the shortest list is enough to look through.
            let shortest = set
                .iter()
                .map(|&sight| &having[starts[sight]..starts[sight + 1]])
                .min_by_key(|list| list.len())
                .unwrap_or_default();
            shortest
                .iter()
                .any(|&k| sets[k].len() > set.len() && is_subset(set, sets[k]))
        };
        let dropped: Vec<Vec<usize>> = sets
            .iter()
            .filter(|set| within(set))
            .map(|&set| set.clone())
            .collect();
        for set in dropped {
            let _ = self.best.remove(&set);
        }
    }

    /// The candidates, without those within others, with the most sights
    /// first, then the lowest score, then the lowest order.
    fn candidates(mut self) -> Vec<Candidate> {
        self.drop_subsets();
        let mut found: Vec<(Vec<usize>, Found)> = self.best.into_iter().collect();
        found.sort_by(|(a_sights, a), (b_sights, b)| {
            b_sights
                .len()
                .cmp(&a_sights.len())
                .then(a.score.total_cmp(&b.score))
                .then(a.order.cmp(&b.order))
        });
        found
            .into_iter()
            .map(|(sights, found)| Candidate {
                sights,
                state: found.state,
            })
            .collect()
    }
}

/// Whether every value of the sorted `small` is in the sorted `large`.
fn is_subset(small: &[usize], large: &[usize]) -> bool {
    let mut rest = large.iter();
    small.iter().all(|value| rest.any(|other| other == value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;

    use crate::tests::synthetic::Rng;
    use crate::{DEPTH_STEP, RATE_STEP, form_units, hypothesis_grid};

    #[test]
    fn cell_keys_spread() {
        // Grids of cell keys about zero, far from it, and spaced by large
        // powers of two, at several levels: no two keys share a hash, and
        // the low sixteen bits, which pick buckets, take most of the values
        // a random hash would.
        use std::hash::BuildHasher;
        let build = BuildHasherDefault::<CellHasher>::default();
        let hash = |key: &(i32, [i64; 2])| build.hash_one(key);
        for (origin, step) in [(-128, 1), (1 << 30, 1), (0, 1 << 16), (0, 1 << 40)] {
            let keys: Vec<(i32, [i64; 2])> = (-2..3)
                .flat_map(|level| {
                    (0..128_i64).flat_map(move |x| {
                        (0..128_i64).map(move |y| (level, [origin + x * step, origin + y * step]))
                    })
                })
                .collect();
            let mut hashes: Vec<u64> = keys.iter().map(hash).collect();
            let mut low: Vec<u64> = hashes.iter().map(|h| h & 0xffff).collect();
            hashes.sort_unstable();
            hashes.dedup();
            assert_eq!(hashes.len(), keys.len(), "{origin} {step}");
            low.sort_unstable();
            low.dedup();
            // A random hash of 81920 keys takes about 46000 of the 65536.
            assert!(low.len() > 40_000, "{origin} {step}: {}", low.len());
        }
    }

    /// A group of these sights, whose state holds `id` in its epoch.
    fn found(sights: &[usize], score: f64, id: usize) -> (Vec<usize>, Found) {
        let zero = Vector::new([0.0; 3]);
        let epoch = 2460000.5 + id as f64;
        let found = Found {
            state: State::new(Desig::Empty, epoch, zero, zero, SunCenter),
            score,
            order: (0, id),
        };
        (sights.to_vec(), found)
    }

    /// The groups found under hypothesis `index`, reduced as [`link()`]
    /// reduces those of one hypothesis.
    fn under(index: usize, found: Vec<(Vec<usize>, Found)>) -> Merged {
        let mut merged = Merged::default();
        for (sights, mut found) in found {
            found.order.0 = index;
            merged.insert(sights, found);
        }
        merged.drop_subsets();
        merged
    }

    #[test]
    fn subsets_are_removed() {
        let mut merged = Merged::default();
        merged.absorb(under(
            0,
            vec![
                found(&[1, 2], 0.0, 0),
                found(&[0, 1, 2, 3], 0.5, 1),
                found(&[2, 3, 4], 0.0, 2),
            ],
        ));
        merged.absorb(under(
            1,
            vec![
                found(&[0, 1, 2, 3], 0.2, 3),
                found(&[3, 4], 0.0, 4),
                found(&[5, 6, 7], 0.0, 5),
                found(&[0, 3], 0.0, 6),
            ],
        ));
        let kept = merged.candidates();
        let sets: Vec<_> = kept.iter().map(|c| c.sights.clone()).collect();
        assert_eq!(sets, vec![vec![0, 1, 2, 3], vec![2, 3, 4], vec![5, 6, 7]]);
        // Of the two equal candidates, the better matching one is kept.
        assert_eq!(kept[0].state.epoch.jd(), 2460003.5);

        assert!(is_subset(&[1, 3], &[0, 1, 2, 3]));
        assert!(is_subset(&[], &[0]));
        assert!(!is_subset(&[1, 4], &[0, 1, 2, 3]));
        assert!(!is_subset(&[0, 5], &[0, 1]));
    }

    /// The candidates of the groups of every hypothesis in turn, by sorting
    /// them all on sights and score and keeping each which is not within one
    /// kept before it.
    fn sort_and_sweep(found: Vec<(Vec<usize>, Found)>) -> Vec<Candidate> {
        let mut found = found;
        found.sort_by(|(a_sights, a), (b_sights, b)| {
            b_sights
                .len()
                .cmp(&a_sights.len())
                .then(a.score.total_cmp(&b.score))
        });
        let mut kept: Vec<Candidate> = Vec::new();
        for (sights, Found { state, .. }) in found {
            if !kept.iter().any(|k| is_subset(&sights, &k.sights)) {
                kept.push(Candidate { sights, state });
            }
        }
        kept
    }

    #[test]
    fn merge_equals_sort_and_sweep() {
        // Random groups of a few sights with few distinct scores, so that
        // equal groups, ties and subsets are common, merged with hypotheses
        // added in a shuffled order. The merge must give the candidates, in
        // the order and with the states, of sorting every group at once.
        let mut rng = Rng(3);
        let mut next = |count: usize| rng.below(count);
        for trial in 0..200 {
            let universe = 4 + trial % 9;
            let hypotheses = 1 + next(12);
            let mut lists = Vec::new();
            let mut id = 0;
            for _ in 0..hypotheses {
                let mut list = Vec::new();
                for _ in 0..next(10) {
                    let mut sights: Vec<usize> = (0..2 + next(universe - 1))
                        .map(|_| next(universe))
                        .collect();
                    sights.sort_unstable();
                    sights.dedup();
                    list.push(found(&sights, next(4) as f64 / 4.0, id));
                    id += 1;
                }
                lists.push(list);
            }
            let all: Vec<(Vec<usize>, Found)> = lists
                .iter()
                .flatten()
                .map(|(sights, f)| found(sights, f.score, f.order.1))
                .collect();
            let expected = sort_and_sweep(all);

            let mut order: Vec<usize> = (0..hypotheses).collect();
            for k in (1..order.len()).rev() {
                order.swap(k, next(k + 1));
            }
            let mut merged = Merged::default();
            let mut lists: Vec<Option<Vec<(Vec<usize>, Found)>>> =
                lists.into_iter().map(Some).collect();
            for index in order {
                merged.absorb(under(index, lists[index].take().unwrap()));
            }
            assert_eq!(merged.candidates(), expected, "trial {trial}");
        }
    }

    #[test]
    fn taylor_bound_is_a_lower_bound() {
        // Random orbits, bound and slightly unbound, every other one falling
        // nearly straight at the Sun so that its perihelion is small, carried
        // up to thirty days either way, and sights near where each one then
        // is. Wherever the series and its bound put a sight farther than an
        // allowed miss, so that `Solved::miss` gives up on it without
        // propagating, the propagated orbit misses it by more than that. The
        // series is off from the propagated orbit by no more than its bound,
        // past a ten-billionth of the distance from the Sun for the precision
        // of the propagation, which no tolerance of a match is finer than,
        // see `MIN_CELL`.
        let epoch = 2460000.5;
        let mut rng = Rng(17);
        let mut uniform = || rng.uniform();
        let direction = |uniform: &mut dyn FnMut() -> f64| {
            Vector::<Equatorial>::new([0; 3].map(|_| 2.0 * uniform() - 1.0)).normalize()
        };
        let (mut pruned, mut checked, mut worst) = (0, 0, 0.0_f64);
        for trial in 0..2000 {
            let radius = 0.1 * 400.0_f64.powf(uniform());
            let pos = direction(&mut uniform) * radius;
            let wander = direction(&mut uniform);
            let heading = if trial % 2 == 0 {
                wander
            } else {
                (wander * 0.05 - pos / radius).normalize()
            };
            let speed = (2.0 * GMS / radius).sqrt() * (0.2 + 0.85 * uniform());
            let state = State::new(Desig::Empty, epoch, pos, heading * speed, SunCenter);
            let (accel, jerk) = accel_and_jerk(&state);
            let solved = Solved {
                unit: 0,
                state,
                points: Vec::new(),
                sigma: 0.0,
                ahead: 0.0,
                pos_error: 0.0,
                vel_error: 0.0,
                accel,
                jerk,
            };
            for days in [0.01, 0.3, 1.0, 3.0, 10.0, 30.0] {
                for dt in [-days, days] {
                    let Ok(carried) = propagate_two_body(&solved.state, (epoch + dt).into()) else {
                        continue;
                    };
                    checked += 1;
                    let off = solved.least(dt, &carried.pos) + jerk * dt.abs().powi(3) / 6.0;
                    let bound = jerk * dt.abs().powi(3) / 6.0 + 1e-10 * radius;
                    worst = worst.max(off / bound);
                    for _ in 0..4 {
                        let shift = direction(&mut uniform) * (1e-10 * 1e9_f64.powf(uniform()));
                        let sight = carried.pos + shift;
                        let allowed = 1e-9 * 1e8_f64.powf(uniform());
                        if solved.least(dt, &sight) > allowed {
                            pruned += 1;
                            let missed = (carried.pos - sight).norm();
                            assert!(missed > allowed, "trial {trial}, {dt} days");
                        }
                    }
                }
            }
        }
        println!(
            "orbits carried {checked}, sights left out {pruned}, largest share of the bound {worst:.3}"
        );
        assert!(worst <= 1.0, "the series is off by {worst} of its bound");
        assert!(pruned > 1000, "only {pruned} sights left out");
    }

    /// State at `epoch` on a circular orbit of `radius` AU near the plane of
    /// the equator, at `phase` radians.
    fn near_equator(epoch: f64, radius: f64, phase: f64) -> State<Equatorial, SunCenter> {
        let speed = (GMS / radius).sqrt();
        let (sin, cos) = phase.sin_cos();
        let pos = Vector::new([cos * radius, sin * radius, 0.05 * radius]);
        let vel = Vector::new([-sin * speed, cos * speed, 0.0]);
        State::new(Desig::Empty, epoch, pos, vel, SunCenter)
    }

    /// Matches found by the hash and by comparing every two units, for
    /// objects near opposition and clutter seen from 1 AU in runs of four
    /// sights over half an hour on the given nights, with the reference time
    /// at night zero. The units are the runs and the pairs across nights of
    /// their ends and of the clutter. Within the reaches the two must agree.
    ///
    /// Returns the matches between units of one object, how many of them
    /// are beyond the reaches, and how many other matches are.
    fn hashed_against_brute_force(nights: std::ops::RangeInclusive<i32>) -> [usize; 3] {
        let epoch = 2460000.5;
        let observer = near_equator(epoch, 1.0, 0.0);
        let mut rng = Rng(9);
        let mut uniform = || rng.uniform();
        let mut objects = Vec::new();
        for _ in 0..4 {
            let radius = 2.0 + uniform();
            objects.push(near_equator(epoch, radius, 0.06 * (uniform() - 0.5)));
        }
        let mut times = Vec::new();
        for night in nights {
            let night = epoch + f64::from(night);
            times.extend([
                night,
                night + 1.0 / 48.0,
                night + 1.0 / 1440.0,
                night + 2.0 / 1440.0,
            ]);
        }
        let mut sights = Vec::new();
        for &time in &times {
            let seen_from = propagate_two_body(&observer, time.into()).unwrap().pos;
            for object in &objects {
                let at = propagate_two_body(object, time.into()).unwrap().pos;
                for k in 0..2 {
                    // The object, then a clutter sight near it.
                    let off = Vector::new([uniform() - 0.5, uniform() - 0.5, uniform() - 0.5]);
                    let shift = if k == 0 { 2e-7 } else { 5e-3 };
                    sights.push(Sight {
                        time: time.into(),
                        observer: seen_from,
                        direction: ((at - seen_from).normalize() + off * shift).normalize(),
                        sigma: 1e-6,
                    });
                }
            }
        }
        let units = form_units(&sights, 0.5, 1.1, 0.01, 0.05).unwrap();
        assert!(units.iter().any(|unit| unit.len() == 4), "no runs");
        assert!(units.iter().any(|unit| unit.len() == 2), "no pairs");
        let every: Vec<usize> = (0..units.len()).collect();
        let origin = observer.clone();
        let grid = hypothesis_grid(&origin, 1.0, 2.2, DEPTH_STEP, RATE_STEP).unwrap();
        let settings = LinkSettings::default();
        let near_truth = grid
            .iter()
            .filter(|h| h.direction[0] > 0.5 && (h.depth_rate / h.depth).abs() < 2e-3)
            .step_by(60);
        // Sight `index` shows object `index / 2 % count` when it is even, and
        // is clutter when it is odd.
        let count = objects.len();
        let object = |index: usize| index.is_multiple_of(2).then_some(index / 2 % count);
        let one_object = |s: &Solved| {
            let unit = &units[s.unit];
            object(unit[0]).filter(|&o| unit.iter().all(|&i| object(i) == Some(o)))
        };
        let (mut true_matches, mut true_outside, mut false_outside) = (0, 0, 0);
        for hypothesis in near_truth {
            let solved = solve_units(&sights, &units, &every, hypothesis, &settings);
            let hashed = matches(&solved, &settings, hypothesis);
            let reach: Vec<f64> = solved.iter().map(|s| s.reach(&settings)).collect();
            for i in 0..solved.len() {
                let near = |j: usize| {
                    let apart = (solved[i].state.pos - solved[j].state.pos).norm();
                    apart <= reach[i] + reach[j]
                };
                let mut brute = Vec::new();
                for j in (0..solved.len()).filter(|&j| j != i) {
                    let score = solved[i]
                        .miss(&solved[j], &settings)
                        .max(solved[j].miss(&solved[i], &settings));
                    if score > 1.0 {
                        continue;
                    }
                    let real = one_object(&solved[i]).is_some()
                        && one_object(&solved[i]) == one_object(&solved[j]);
                    true_matches += usize::from(real);
                    if near(j) {
                        brute.push(j);
                    } else if real {
                        true_outside += 1;
                    } else {
                        false_outside += 1;
                    }
                }
                // Within the reaches, the hash finds exactly what comparing
                // every two units finds.
                let mut got: Vec<usize> = hashed[i].iter().map(|m| m.1).collect();
                got.sort_unstable();
                assert_eq!(got, brute, "unit {i}");
            }
        }
        [true_matches, true_outside, false_outside]
    }

    #[test]
    fn hashed_matches_equal_brute_force() {
        // Nights about the reference time, as a search would choose.
        let [found, beyond, others] = hashed_against_brute_force(-1..=1);
        println!(
            "centered: matches of one object {found}, beyond the reaches {beyond}, others beyond {others}"
        );
        assert!(found > 100, "only {found} matches of one object");
        assert_eq!(beyond, 0, "matches of one object beyond the reaches");

        // Nights seven to nine days after the reference time. Orbits are carried
        // far past the sights to it, and drift apart there.
        let [found, beyond, others] = hashed_against_brute_force(7..=9);
        println!(
            "off center: matches of one object {found}, beyond the reaches {beyond}, others beyond {others}"
        );
        assert!(found > 100, "only {found} matches of one object");
    }

    #[test]
    fn groups_keep_units_that_agree() {
        // The seed is 0. Units 1-4 are one object; unit 4 is far in time and
        // matches only its neighbor 3 besides the seed, and joins. Unit 5
        // matches the seed alone and is left out. Unit 1, the best match,
        // joins with nothing kept before it.
        let neighbors: Vec<Vec<usize>> = vec![
            vec![1, 2, 3, 4, 5],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2, 4],
            vec![0, 3],
            vec![0],
        ];
        let sorted: Vec<(f64, usize)> = (1..6).map(|k| (k as f64, k)).collect();
        let mut kept = Vec::new();
        agreeing(&sorted, &neighbors, &mut kept);
        let kept: Vec<usize> = kept.iter().map(|m| m.1).collect();
        assert_eq!(kept, vec![1, 2, 3, 4]);
    }
}
