// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Python bindings for observation linking.
//!
//! Wraps `kete_linking` for use from Python. Detections are held in Rust as a
//! [`PySights`] collection, units as a [`PyUnits`] collection, and hypotheses
//! as a [`PyHypotheses`] collection, so that millions of them do not become
//! Python objects.

use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::*;
use kete_core::time::Time;
use kete_fitting::AstrometricObservation;
use kete_linking::{
    Candidate, ConfirmRules, Confirmed, DEPTH_STEP, Hypothesis, LinkSettings, RATE_STEP,
    SearchPass, SearchSettings, Sight, confirm, form_units, hypothesis_grid, link, merge_pieces,
    parallax_margin, search, search_windows, sight_from_observation, tiles,
};
use kete_spice::prelude::LOADED_SPK;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySlice};
use rayon::prelude::*;

use crate::fitting::{PyObservation, PyOrbitFit, RAD_TO_ARCSEC};
use crate::frame::PyFrames;
use crate::state::PyState;
use crate::time::PyTime;
use crate::vector::{PyVector, VectorLike};

/// The state of an origin with respect to the Sun, from a state about any
/// center.
fn sun_centered(state: &PyState) -> PyResult<State<Equatorial, SunCenter>> {
    let spk = LOADED_SPK.try_read().map_err(Error::from)?;
    Ok(spk.try_to_sun(state.raw.clone())?)
}

/// An angle, or a rate of angle when `rate` is set, given in degrees and
/// checked in those units, as radians. `what` names it in the error. It must
/// be finite and positive, or also zero when `zero` is set.
fn radians(what: &str, degrees: f64, rate: bool, zero: bool) -> PyResult<f64> {
    if !(degrees.is_finite() && (degrees > 0.0 || (zero && degrees == 0.0))) {
        let bound = if zero { "not negative" } else { "positive" };
        let unit = if rate { "degrees per day" } else { "degrees" };
        Err(Error::ValueError(format!(
            "{what} must be finite and {bound}, in {unit}, got {degrees}"
        )))?;
    }
    Ok(degrees.to_radians())
}

/// A count given as a Python integer, which must not be negative. `what`
/// names it in the error.
fn count(what: &str, value: i64) -> PyResult<usize> {
    usize::try_from(value)
        .map_err(|_| Error::ValueError(format!("{what} must not be negative, got {value}")).into())
}

/// A HEALPix resolution given as a Python integer, as the linking crate
/// takes it, which checks that it is a power of two.
fn checked_nside(nside: Option<i64>) -> PyResult<Option<u32>> {
    nside
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                Error::ValueError(format!(
                    "Tiles: nside must be a power of two up to 2^29, got {value}"
                ))
                .into()
            })
        })
        .transpose()
}

/// The station of each observation as an index, from an iterable of one
/// hashable label per observation, numbered in the order the labels first
/// appear. The numbers only tell stations apart: what a search does with each
/// station does not depend on them.
fn station_indices(
    py: Python<'_>,
    stations: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<usize>>> {
    let Some(labels) = stations else {
        return Ok(None);
    };
    let numbers = PyDict::new(py);
    let mut indices = Vec::new();
    for label in labels.try_iter()? {
        let label = label?;
        let index = match numbers.get_item(&label)? {
            Some(index) => index.extract()?,
            None => {
                let index = numbers.len();
                numbers.set_item(&label, index)?;
                index
            }
        };
        indices.push(index);
    }
    Ok(Some(indices))
}

/// The rules of [`confirm_py`] from its keyword arguments.
fn confirm_rules(
    min_sights: i64,
    max_rms: f64,
    min_sights_per_side: i64,
    group_span: f64,
    max_gap: f64,
) -> PyResult<ConfirmRules> {
    Ok(ConfirmRules {
        min_sights: count("Confirm: min_sights", min_sights)?,
        max_rms,
        min_sights_per_side: count("Confirm: min_sights_per_side", min_sights_per_side)?,
        group_span,
        max_gap,
    })
}

/// The settings of [`link_py`] from its keyword arguments, with
/// `grid_allowance` in arcseconds.
fn link_settings(threshold: f64, grid_allowance: f64, max_uncertainty: f64) -> LinkSettings {
    LinkSettings {
        threshold,
        grid_allowance: grid_allowance / RAD_TO_ARCSEC,
        max_uncertainty,
    }
}

/// A default count of the linking crate as a Python integer.
#[allow(
    clippy::cast_possible_wrap,
    reason = "the default counts of the crate are small"
)]
const fn default_count(value: usize) -> i64 {
    value as i64
}

/// An error naming the first sight which the linking crate does not accept.
fn check_usable(sights: &[Sight]) -> PyResult<()> {
    if let Some(bad) = sights.iter().position(|sight| !sight.is_usable()) {
        Err(Error::ValueError(format!(
            "Sights: detection {bad} has a time, observer, direction or sigma which is not \
             finite, a time more than 9e15 days from JD 0, a direction of zero length, or a \
             sigma which is not positive"
        )))?;
    }
    Ok(())
}

/// Detections as lines of sight, the input of linking.
///
/// Each detection is a time, the position of the observer with respect to the
/// Sun, a direction from the observer, and an astrometric uncertainty. Nothing
/// else about the observer is used, so detections from any observers can be
/// mixed.
///
/// The arrays hold one row per detection, and other functions in this module
/// refer to detections by their row index. Every detection is checked when the
/// collection is built, so the functions of this module accept any
/// :class:`Sights`.
///
/// Parameters
/// ----------
/// time : list[float or Time]
///     Time each detection was received, JD in the TDB scale.
/// observer : list[list[float]]
///     Position of the observer with respect to the Sun at that time, in AU,
///     as rows of x, y, z in ``frame``.
/// direction : list[list[float]]
///     Direction from the observer to the detection, as rows of x, y, z in
///     ``frame``. Each row is scaled to unit length. This is the astrometric
///     direction: corrected for light travel time only, not for aberration.
/// sigma : list[float]
///     One-sigma uncertainty of the direction on each axis of the sky, in
///     arcseconds, positive.
/// frame : Frames
///     Frame of ``observer`` and ``direction``, Ecliptic by default.
///
/// Raises
/// ------
/// ValueError
///     If the arrays differ in length, or for the first detection with a value
///     which is not finite, a time more than 9e15 days from JD 0, a direction
///     of zero length, or a sigma which is not positive.
#[pyclass(frozen, module = "kete.linking", name = "Sights")]
#[derive(Debug)]
pub struct PySights {
    /// The lines of sight.
    sights: Vec<Sight>,

    /// Frame the observer and direction arrays are given in.
    frame: PyFrames,
}

impl PySights {
    /// One component of every sight, in [`PySights::frame`].
    fn rows(&self, pick: impl Fn(&Sight) -> Vector<Equatorial>) -> Vec<[f64; 3]> {
        self.sights
            .iter()
            .map(|sight| PyVector::new(pick(sight), self.frame).raw())
            .collect()
    }
}

#[pymethods]
impl PySights {
    #[new]
    #[pyo3(signature = (time, observer, direction, sigma, frame=PyFrames::Ecliptic))]
    fn new(
        time: Vec<PyTime>,
        observer: Vec<[f64; 3]>,
        direction: Vec<[f64; 3]>,
        sigma: Vec<f64>,
        frame: PyFrames,
    ) -> PyResult<Self> {
        let n = time.len();
        if observer.len() != n || direction.len() != n || sigma.len() != n {
            Err(Error::ValueError(format!(
                "Sights: time, observer, direction and sigma need the same length, got {n}, {}, \
                 {} and {}",
                observer.len(),
                direction.len(),
                sigma.len()
            )))?;
        }
        let sights: Vec<Sight> = time
            .iter()
            .zip(&observer)
            .zip(&direction)
            .zip(&sigma)
            .map(|(((time, &pos), &dir), &sig)| {
                let dir = VectorLike::Arr(dir).into_vector(frame);
                Sight {
                    time: time.0,
                    observer: VectorLike::Arr(pos).into_vector(frame),
                    direction: dir / dir.norm(),
                    sigma: sig / RAD_TO_ARCSEC,
                }
            })
            .collect();
        check_usable(&sights)?;
        Ok(Self { sights, frame })
    }

    /// The lines of sight of optical observations.
    ///
    /// The direction is the observed right ascension and declination. The
    /// observer is moved to the Sun with the loaded SPICE kernels. The
    /// uncertainty is the root mean square of the uncertainties on the two axes
    /// of the sky, and a correlation between them is ignored.
    ///
    /// Row ``i`` of the result is ``observations[i]``, so the indices of
    /// candidates found from these sights are also indices into
    /// ``observations``.
    ///
    /// Parameters
    /// ----------
    /// observations : list[Observation]
    ///     Optical :class:`~kete.orbit_fitting.Observation` objects, such as from
    ///     :func:`~kete.orbit_fitting.mpc_obs_to_observations` with
    ///     ``apply_over_obs_reweight=False``. Its default reweighting is for
    ///     fitting the observations of one object, and inflates the
    ///     uncertainties of a survey's detections many times over.
    /// frame : Frames
    ///     Frame the ``observer`` and ``direction`` arrays are returned in,
    ///     Ecliptic by default.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Naming the first observation which fails, if it is a radar
    ///     observation, the loaded kernels do not cover the Sun at its time, or
    ///     its line of sight has a value which is not finite, a time more than
    ///     9e15 days from JD 0, or an uncertainty which is not positive.
    #[staticmethod]
    #[pyo3(signature = (observations, frame=PyFrames::Ecliptic))]
    fn from_observations(
        py: Python<'_>,
        observations: Vec<PyObservation>,
        frame: PyFrames,
    ) -> PyResult<Self> {
        let sights = py.detach(|| {
            observations
                .par_iter()
                .enumerate()
                .map(|(index, observation)| {
                    sight_from_observation(&observation.obs).map_err(|err| {
                        Error::ValueError(format!("Sights: observation {index}: {err}"))
                    })
                })
                .collect::<KeteResult<Vec<_>>>()
        })?;
        Ok(Self { sights, frame })
    }

    /// Time each detection was received, JD in the TDB scale.
    #[getter]
    fn time(&self) -> Vec<f64> {
        self.sights.iter().map(|sight| sight.time.jd()).collect()
    }

    /// Position of the observer with respect to the Sun, in AU, as rows of x,
    /// y, z in :attr:`frame`.
    #[getter]
    fn observer(&self) -> Vec<[f64; 3]> {
        self.rows(|sight| sight.observer)
    }

    /// Unit direction from the observer to the detection, as rows of x, y, z in
    /// :attr:`frame`.
    #[getter]
    fn direction(&self) -> Vec<[f64; 3]> {
        self.rows(|sight| sight.direction)
    }

    /// One-sigma uncertainty of the direction on each axis of the sky, in
    /// arcseconds.
    #[getter]
    fn sigma(&self) -> Vec<f64> {
        self.sights
            .iter()
            .map(|sight| sight.sigma * RAD_TO_ARCSEC)
            .collect()
    }

    /// Frame of the :attr:`observer` and :attr:`direction` arrays.
    #[getter]
    fn frame(&self) -> PyFrames {
        self.frame
    }

    fn __len__(&self) -> usize {
        self.sights.len()
    }

    fn __repr__(&self) -> String {
        format!("Sights(<{} detections>)", self.sights.len())
    }
}

/// Groups of detections which could be one object, the units of linking.
///
/// Returned by :func:`form_units`. Each unit is a list of two or more row
/// indices into the :class:`Sights` it was formed from, in time order.
///
/// Parameters
/// ----------
/// indices : list[list[int]]
///     Units, to link units chosen by other means. Each is two or more row
///     indices in time order, spanning some time.
///
/// Raises
/// ------
/// ValueError
///     If an index is negative. The other conditions on a unit are checked by
///     :func:`link`.
#[pyclass(frozen, module = "kete.linking", name = "Units")]
#[derive(Debug)]
pub struct PyUnits(Vec<Vec<usize>>);

#[pymethods]
impl PyUnits {
    #[new]
    fn new(indices: Vec<Vec<i64>>) -> PyResult<Self> {
        let units = indices
            .into_iter()
            .enumerate()
            .map(|(k, unit)| {
                unit.into_iter()
                    .map(|index| {
                        usize::try_from(index).map_err(|_| {
                            Error::ValueError(format!(
                                "Units: unit {k} has the negative index {index}"
                            ))
                            .into()
                        })
                    })
                    .collect()
            })
            .collect::<PyResult<_>>()?;
        Ok(Self(units))
    }

    /// The units, as lists of row indices into the :class:`Sights`, each in
    /// time order.
    #[getter]
    fn indices(&self) -> Vec<Vec<usize>> {
        self.0.clone()
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __repr__(&self) -> String {
        format!("Units(<{} units>)", self.0.len())
    }
}

/// The units of linking: tracklets of detections close in time, and pairs
/// across gaps, which could be one object.
///
/// A tracklet is two or more detections, within ``tracklet_span`` days, of one
/// smooth motion on the sky that one observer could have seen. Two detections
/// are neighbors when the later is more than zero and at most
/// ``tracklet_span`` days after the earlier, their directions are within
/// ``max_rate`` times the time between them, and their observers are no
/// farther apart than one observer could move in that time.
///
/// Every two neighbors at most ``tracklet_span / (min_tracklet - 1)`` days
/// apart seed a tracklet: a tracklet of ``min_tracklet`` detections within
/// ``tracklet_span`` has two in a row at most that far apart, so a larger
/// ``min_tracklet`` means fewer seeds. A tracklet grows from its seed along
/// its track, defined below, from its first to its last detection as it
/// grows: of the neighbors of its last detection which are later and within
/// ``tracklet_span`` of its first, it takes the earliest on its track, of
/// those at one time the one which misses it least for what the track
/// allows, then the lowest index, until none is on the track. It then grows
/// back in time the same way from its first detection. Detections off the
/// track between those it takes are passed over. Within a tracklet the
/// track allows the noise of the detections and of the rate the tracklet
/// measures, with no allowance for a change of rate, so false detections
/// seldom line up by chance. Every seed grows a tracklet; those of at least
/// ``min_tracklet`` detections are taken most detections first, then least
/// summed miss, then lowest first index, and each keeps its detections no
/// earlier tracklet holds, when at least ``min_tracklet`` of them remain.
///
/// A false detection seldom lies on the track of an object, so it neither
/// takes the place of the object's own detection nor ends its tracklet, even
/// when nearer in direction, and a missed detection does not end a tracklet
/// either. Detections of one object from two places whose parallax is larger
/// than the track allows form a tracklet for each place. The costs: an object
/// whose motion on the sky curves beyond the track within ``tracklet_span``,
/// such as a very close object seen from the ground, is cut into several
/// shorter tracklets, which still link; two objects whose tracks cross can
/// swap detections where they meet, and of two tracklets which share a
/// detection the rule of most detections first keeps one; and a few
/// detections of an object seen over longer than ``tracklet_span`` can be
/// left lone between its tracklets.
///
/// Pairs are formed among the lone detections, those in no tracklet, and the
/// first and last detection of each tracklet, except two of the same
/// tracklet: detections ``i`` and ``j`` are paired when ``j`` is later than
/// ``i`` by at most ``max_gap`` days, and the angle between their directions
/// is at most ``max_rate`` times that gap plus ``parallax_margin``. Pairs have
/// no least gap, so a detection is paired however soon after a tracklet it
/// falls. Pairing the ends
/// of tracklets keeps objects which move too much within a tracklet for one
/// orbit to fit it, such as objects close to the observer. With
/// ``tracklet_span`` zero there are no tracklets, and the units are the pairs
/// of all the detections with ``lone_pairs``, and none without. The order of
/// the units is not specified.
///
/// The track of a tracklet is its motion along the great circle from its
/// first to its last detection, carried on at the same rate. The end of a
/// tracklet is paired only with detections on its track. The miss allowed is
/// four times the uncertainties of the two detections added in quadrature,
/// plus the time between the two times an allowed error of rate: four times
/// ``sqrt(2)`` times the larger uncertainty of the first and last detection of
/// the tracklet over its span, plus a twentieth of ``max_rate``, and no more
/// than ``max_rate``, plus ``parallax_margin``. Tracklets of two are held to
/// their track too. Of the pairs on its track, each end of a tracklet keeps the
/// two nearest it on each side in time, by the miss over what the track
/// allows, and a pair of two tracklet ends has to be kept by both: among many
/// false detections the track admits many detections of nothing, and the
/// detection of the object is nearly always the nearest.
///
/// The observers are not compared for pairs. Two observers at different places
/// see one object in different directions at the same time, by up to
/// ``parallax_margin``, see :func:`parallax_margin`; a parallax larger than
/// the margin and the track allow keeps a detection off the track of a
/// tracklet seen from another place.
///
/// Where most detections are false, few of them form tracklets, and the pairs
/// of two lone detections, which are not held to any track, grow as the
/// square of their density and outnumber every other unit. Two false
/// detections close in time also form a tracklet of two, and since every seed
/// is grown, chance alignments of three or more grow about as the cube of the
/// density of false detections where each has many neighbors. ``min_tracklet``
/// and ``lone_pairs`` trade the objects found only through such units for
/// fewer units. ``lone_pairs`` changes neither the tracklets nor the pairs of
/// the ends of two tracklets.
///
/// Parameters
/// ----------
/// sights : Sights
///     The detections.
/// max_gap : float
///     Largest time between the detections of a pair, in days.
/// max_rate : float
///     Largest apparent rate of motion, in degrees per day.
/// tracklet_span : float
///     Longest tracklet, in days. Zero forms no tracklets.
/// min_tracklet : int
///     Fewest detections of a tracklet, at least 2. Shorter tracklets are left
///     out, and their detections are lone. Default is 2, which keeps every
///     tracklet. Three or more suits a survey which sees each object three or
///     more times in a visit, such as a spacecraft which revisits a field every
///     orbit, and leaves out the tracklets of two that false detections form,
///     and seeds only detections closer in time. An object seen fewer times
///     in a visit has no tracklet there.
/// lone_pairs : bool
///     Whether two lone detections may be paired. Default is False: every
///     pair holds the end of a tracklet and lies on its track, so the pairs
///     grow with the density of the detections rather than its square. That
///     suits any survey which sees each object at least ``min_tracklet``
///     times in a visit, such as one on the ground which takes its visits in
///     pairs, or a spacecraft which revisits a field every orbit. An object
///     seen fewer times than that in a visit is then found only through the
///     pairs of those detections with the ends of its tracklets in other
///     visits, and an object with no tracklet is not found; with
///     ``tracklet_span`` zero there are then no units. True also pairs lone
///     detections, which a survey that sees most objects once per visit
///     needs, at a cost that grows as the square of the density of false
///     detections.
/// parallax_margin : float
///     Largest angle between the directions of one object seen at one time by
///     two of the observers, in degrees, see :func:`parallax_margin`. Default
///     is 0.
///
/// Returns
/// -------
/// Units
///
/// Raises
/// ------
/// ValueError
///     Unless ``max_gap`` is not negative, ``max_rate`` is finite and
///     positive, ``tracklet_span`` and ``parallax_margin`` are finite and not
///     negative, and ``min_tracklet`` is at least 2.
#[pyfunction]
#[pyo3(
    name = "form_units",
    signature = (
        sights,
        max_gap,
        max_rate,
        tracklet_span,
        min_tracklet=default_count(SearchSettings::default().min_tracklet),
        lone_pairs=SearchSettings::default().lone_pairs,
        parallax_margin=0.0,
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of form_units are keyword arguments"
)]
pub fn form_units_py(
    py: Python<'_>,
    sights: PyRef<'_, PySights>,
    max_gap: f64,
    max_rate: f64,
    tracklet_span: f64,
    min_tracklet: i64,
    lone_pairs: bool,
    parallax_margin: f64,
) -> PyResult<PyUnits> {
    let max_rate = radians("Form units: max_rate", max_rate, true, false)?;
    let margin = radians("Form units: parallax_margin", parallax_margin, false, true)?;
    let min_tracklet = count("Form units: min_tracklet", min_tracklet)?;
    let sights = &sights.sights;
    let units = py.detach(|| {
        form_units(
            sights,
            max_gap,
            max_rate,
            tracklet_span,
            min_tracklet,
            lone_pairs,
            margin,
        )
    })?;
    Ok(PyUnits(units))
}

/// An assertion of the depth of an object along a fixed direction, and of its
/// rate, at a reference time.
///
/// At the epoch of :attr:`origin`, which is the reference time, the object is
/// on the plane ``direction . (position - origin position) = depth``, and moves
/// through it at ``direction . (velocity - origin velocity) = depth_rate``.
/// Under one hypothesis a pair of detections has one orbit, and a tracklet the
/// one that fits it best.
///
/// Taken from :class:`Hypotheses`, such as from :func:`hypothesis_grid`.
#[pyclass(frozen, module = "kete.linking", name = "Hypothesis", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyHypothesis(Hypothesis);

#[pymethods]
impl PyHypothesis {
    /// Point the depth is measured from, with its velocity, as a Sun-centered
    /// state. Its epoch is the reference time.
    #[getter]
    fn origin(&self) -> PyState {
        State::<Equatorial>::from(self.0.origin.clone()).into()
    }

    /// Unit vector the depth is measured along.
    #[getter]
    fn direction(&self) -> PyVector {
        self.0.direction.into()
    }

    /// Distance of the object from the origin along the direction, in AU.
    #[getter]
    fn depth(&self) -> f64 {
        self.0.depth
    }

    /// Rate of change of the depth, in AU per day.
    #[getter]
    fn depth_rate(&self) -> f64 {
        self.0.depth_rate
    }

    fn __repr__(&self) -> String {
        let [x, y, z]: [f64; 3] = self.0.direction.into();
        format!(
            "Hypothesis(epoch={:.6}, direction=[{x}, {y}, {z}] (Equatorial), depth={}, \
             depth_rate={})",
            self.0.origin.epoch.jd(),
            self.0.depth,
            self.0.depth_rate
        )
    }
}

/// An index into a [`PyHypotheses`], or a slice of it.
#[derive(FromPyObject)]
enum HypothesesIndex<'py> {
    /// One position, negative from the end.
    At(isize),

    /// A slice.
    Slice(Bound<'py, PySlice>),
}

/// One hypothesis, or a collection of them.
#[derive(IntoPyObject)]
enum HypothesesItem {
    /// One hypothesis.
    One(PyHypothesis),

    /// A collection.
    Many(PyHypotheses),
}

/// Hypotheses to search under, held without making each a Python object.
///
/// Returned by :func:`hypothesis_grid`. Indexing gives one
/// :class:`Hypothesis`, and a slice gives a :class:`Hypotheses`.
///
/// :func:`link` removes duplicate candidates, and candidates within others,
/// only among the hypotheses of one call, so a search split into parts linked
/// separately can return candidates which one call over all the hypotheses
/// would not.
#[pyclass(frozen, module = "kete.linking", name = "Hypotheses")]
#[derive(Debug)]
pub struct PyHypotheses(Vec<Hypothesis>);

#[pymethods]
impl PyHypotheses {
    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __getitem__(&self, index: HypothesesIndex<'_>) -> PyResult<HypothesesItem> {
        let len = self.0.len() as isize;
        match index {
            HypothesesIndex::At(at) => {
                let at = if at < 0 { at + len } else { at };
                if !(0..len).contains(&at) {
                    return Err(PyIndexError::new_err("Hypotheses index out of range"));
                }
                Ok(HypothesesItem::One(PyHypothesis(
                    self.0[at as usize].clone(),
                )))
            }
            HypothesesIndex::Slice(slice) => {
                let range = slice.indices(len)?;
                let picked = (0..range.slicelength as isize)
                    .map(|k| self.0[(range.start + k * range.step) as usize].clone())
                    .collect();
                Ok(HypothesesItem::Many(Self(picked)))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("Hypotheses(<{} hypotheses>)", self.0.len())
    }
}

/// Every hypothesis of a search about one origin.
///
/// The directions are the six Equatorial coordinate axes, both ways. Depths run
/// from ``min_depth`` to ``max_depth`` in steps of ``depth_step`` in their natural
/// log. At each depth the ratio of depth rate to depth runs in steps of
/// ``rate_step`` over what a bound orbit about the Sun allows, with a finer
/// step where that range is narrow.
///
/// On an axis across the line from the Sun to the origin, the plane of a small
/// depth passes close to the Sun, so the size of a grid grows as the smallest
/// depth to the power -1.5, and each population is best searched with its own
/// depth range. A grid of more than :data:`MAX_HYPOTHESES` hypotheses is
/// refused.
///
/// The default steps are coarse, for speed. They find main belt and more
/// distant objects seen on several nights, sometimes split into several
/// candidates, and lose more near-Earth objects and close objects, which
/// :func:`search` recovers with a second, finer pass over the detections the
/// first did not confirm. Objects seen once a night are found only with
/// ``lone_pairs=True``, and more often split.
///
/// Parameters
/// ----------
/// origin : State
///     Point depths are measured from, with its velocity, such as an observer
///     near the middle of the detections. Its epoch is the reference time,
///     which should lie within the span of the detections: :func:`link`
///     refuses one farther outside the span of the detections of its units
///     than that span. It may be about any center; it is moved to the Sun
///     with the loaded SPICE kernels.
/// min_depth : float
///     Smallest depth, in AU.
/// max_depth : float
///     Largest depth, in AU.
/// depth_step : float
///     Step in the natural log of the depth. Default is :data:`DEPTH_STEP`.
/// rate_step : float
///     Step in depth rate over depth, per day. Default is :data:`RATE_STEP`.
///
/// Returns
/// -------
/// Hypotheses
///
/// Raises
/// ------
/// ValueError
///     Unless ``0 < min_depth <= max_depth``, both steps are positive, and all
///     values are finite, or if the grid would be too large, or if the loaded
///     kernels do not cover the Sun at the epoch of ``origin``.
#[pyfunction]
#[pyo3(name = "hypothesis_grid", signature = (origin, min_depth, max_depth, depth_step=DEPTH_STEP, rate_step=RATE_STEP))]
pub fn hypothesis_grid_py(
    py: Python<'_>,
    origin: PyState,
    min_depth: f64,
    max_depth: f64,
    depth_step: f64,
    rate_step: f64,
) -> PyResult<PyHypotheses> {
    let origin = sun_centered(&origin)?;
    let grid =
        py.detach(|| hypothesis_grid(&origin, min_depth, max_depth, depth_step, rate_step))?;
    Ok(PyHypotheses(grid))
}

/// The hypotheses argument of [`link_py`].
#[derive(Debug, FromPyObject)]
pub enum HypothesesLike<'py> {
    /// A collection held in Rust, used without a copy.
    Held(PyRef<'py, PyHypotheses>),

    /// A list of single hypotheses.
    List(Vec<PyHypothesis>),
}

/// A group of detections which could be one object.
///
/// Returned by :func:`link`.
#[pyclass(frozen, module = "kete.linking", name = "Candidate", from_py_object)]
#[derive(Debug, Clone)]
pub struct PyCandidate(Candidate);

#[pymethods]
impl PyCandidate {
    /// Row indices of the detections into the :class:`Sights` given to
    /// :func:`link`, sorted and distinct. For sights built with
    /// :meth:`Sights.from_observations` these are also indices into the
    /// observations.
    #[getter]
    fn sights(&self) -> Vec<usize> {
        self.0.sights.clone()
    }

    /// Sun-centered state of the object at the reference time of the
    /// hypothesis it was found under, from two-body motion.
    #[getter]
    fn state(&self) -> PyState {
        State::<Equatorial>::from(self.0.state.clone()).into()
    }

    fn __len__(&self) -> usize {
        self.0.sights.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Candidate(sights={}, epoch={:.6})",
            self.0.sights.len(),
            self.0.state.epoch.jd()
        )
    }
}

/// Candidate objects among the detections.
///
/// Under each hypothesis, every unit whose detections all point within about
/// 63 degrees of its direction, a cosine of at least :data:`MIN_ALIGNMENT`,
/// is turned into an orbit, which is a two-body orbit about the Sun: the one
/// through both detections of a pair, and the least-squares fit, with light
/// time and each detection's own observer, of a tracklet. A tracklet whose own
/// detections miss its orbit by more than ``threshold`` times their
/// uncertainty plus ``grid_allowance`` is left out, and so are units whose
/// position at the reference time is more uncertain than ``max_uncertainty``
/// of their range. Each remaining unit then seeds a group: the units whose
/// orbits agree with it, with no chaining from one match to the next. Two
/// units agree when the orbit of each passes close enough to every detection
/// of the other, at their own times. How close is ``threshold`` times the
/// error that the astrometric uncertainty gives that miss, from the unit's own
/// covariance, plus ``grid_allowance`` times the distance to the detection.
///
/// A group with at least ``min_sights`` detections is a candidate. Of two
/// detections at the same time from the same place, only one is kept in a
/// group. Candidates whose detections are all in another candidate are removed,
/// across all hypotheses of the call. The candidates are returned with the most
/// detections first.
///
/// Grouping is generous. Purity comes from confirming the candidates with an
/// orbit fit, see :func:`confirm`.
///
/// Parameters
/// ----------
/// sights : Sights
///     The detections.
/// units : Units
///     Units of detections, such as from :func:`form_units`.
/// hypotheses : Hypotheses or list[Hypothesis]
///     Hypotheses to search under, such as from :func:`hypothesis_grid`. They
///     need not share a reference time.
/// threshold : float
///     Largest miss between two units, in units of the error astrometric noise
///     gives it on one axis of the sky. Default is 4.
/// grid_allowance : float
///     Miss allowed for the hypothesis grid, as an angle in arcseconds at the
///     detection being matched, which has to suit the longest time between
///     units in a search. The default is 1e-4 radians, about 20.6 arcseconds.
/// min_sights : int
///     Fewest distinct detections in a candidate. Default is 4.
/// max_uncertainty : float
///     Largest position uncertainty of a unit at the reference time, as a
///     fraction of its range, for the unit to take part in grouping, best no
///     more than the ``depth_step`` of the grid over ``threshold``. Positive.
///     Default is 0.05.
///
/// Returns
/// -------
/// list[Candidate]
///
/// Raises
/// ------
/// ValueError
///     If a unit indexes past the sights or is not two or more detections in
///     time order spanning some time, the reference time of a hypothesis is
///     more than 9e15 days from JD 0 or farther outside the span of the
///     detections of the units than that span, a setting is negative or not
///     finite, ``min_sights`` is negative, ``max_uncertainty`` is zero,
///     ``threshold`` and ``grid_allowance`` are both zero, or there are more
///     than :data:`MAX_UNITS` units.
#[pyfunction]
#[pyo3(
    name = "link",
    signature = (
        sights,
        units,
        hypotheses,
        threshold=LinkSettings::default().threshold,
        grid_allowance=LinkSettings::default().grid_allowance * RAD_TO_ARCSEC,
        min_sights=default_count(ConfirmRules::default().min_sights),
        max_uncertainty=LinkSettings::default().max_uncertainty,
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of link are keyword arguments"
)]
pub fn link_py(
    py: Python<'_>,
    sights: PyRef<'_, PySights>,
    units: PyRef<'_, PyUnits>,
    hypotheses: HypothesesLike<'_>,
    threshold: f64,
    grid_allowance: f64,
    min_sights: i64,
    max_uncertainty: f64,
) -> PyResult<Vec<PyCandidate>> {
    let settings = link_settings(threshold, grid_allowance, max_uncertainty);
    let min_sights = count("Link: min_sights", min_sights)?;
    let listed: Vec<Hypothesis>;
    let hypotheses: &[Hypothesis] = match &hypotheses {
        HypothesesLike::Held(held) => &held.0,
        HypothesesLike::List(list) => {
            listed = list.iter().map(|h| h.0.clone()).collect();
            &listed
        }
    };
    let (sights, units) = (&sights.sights, &units.0);
    let found = py.detach(|| link(sights, units, hypotheses, min_sights, &settings))?;
    Ok(found.into_iter().map(PyCandidate).collect())
}

/// A candidate which an orbit fit confirmed.
///
/// Returned by :func:`confirm`.
#[pyclass(frozen, module = "kete.linking", name = "Confirmed")]
#[derive(Debug)]
pub struct PyConfirmed(Confirmed);

impl From<Confirmed> for PyConfirmed {
    fn from(value: Confirmed) -> Self {
        Self(value)
    }
}

impl From<&PyConfirmed> for Confirmed {
    fn from(value: &PyConfirmed) -> Self {
        Self {
            sights: value.0.sights.clone(),
            fit: value.0.fit.clone(),
        }
    }
}

#[pymethods]
impl PyConfirmed {
    /// Indices into the observations given to :func:`confirm` of those the
    /// fit kept, sorted. They are among the :attr:`Candidate.sights` of the
    /// candidate.
    #[getter]
    fn sights(&self) -> Vec<usize> {
        self.0.sights.clone()
    }

    /// The orbit fit of the confirmation.
    ///
    /// The fit is over only the observations the confirmation kept, plus any
    /// its outlier rejection left out, and holds them in time order: its
    /// :attr:`~kete.orbit_fitting.OrbitFit.observations` and
    /// :attr:`~kete.orbit_fitting.OrbitFit.residuals` are those of
    /// :attr:`sights` in time order, and its
    /// :attr:`~kete.orbit_fitting.OrbitFit.all_observations` and
    /// :attr:`~kete.orbit_fitting.OrbitFit.included` also hold the rejected
    /// ones. So ``fit.residuals[k]`` and ``fit.included[k]`` do not line up
    /// with ``sights[k]``, which is in index order. Match them by epoch, such
    /// as ``fit.observations[k].epoch.jd`` against
    /// ``observations[i].epoch.jd`` for ``i`` in :attr:`sights`; observations
    /// at the same time may be in either order.
    ///
    /// Its epoch is the middle of the time the optical observations of the
    /// candidate span, or, if the two-body fit which starts it fails, the
    /// reference time of the hypothesis the candidate was found under.
    #[getter]
    fn fit(&self) -> PyOrbitFit {
        PyOrbitFit {
            inner: self.0.fit.clone(),
        }
    }

    fn __len__(&self) -> usize {
        self.0.sights.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Confirmed(sights={}, rms={:.4})",
            self.0.sights.len(),
            self.0.fit.rms
        )
    }
}

/// Confirmed objects as Python objects.
fn confirmed_py(confirmed: Vec<Confirmed>) -> Vec<PyConfirmed> {
    confirmed.into_iter().map(PyConfirmed::from).collect()
}

/// The candidates an orbit fit confirms, without overlaps.
///
/// Each candidate is first fit with a two-body orbit about the Sun to its own
/// detections, starting from the candidate's state, at the middle of the time
/// they span, dropping detections that do not fit while more than two thirds
/// of them remain. A candidate whose two-body fit keeps fewer than
/// ``min_sights`` detections, or leaves a residual more than ten times
/// ``max_rms``, is turned down there. Otherwise the two-body orbit starts
/// :func:`~kete.orbit_fitting.fit_orbit`, with outlier rejection and the planets of
/// the loaded SPICE kernels, over the detections it kept, at the same epoch,
/// see :attr:`Confirmed.fit`. When that fit fails or its orbit is not bound,
/// the start is scored by the same fit with no iterations.
///
/// A candidate is confirmed if its fit:
///
/// - keeps at least ``min_sights`` observations, and at least two thirds of
///   those of the candidate;
/// - has a reduced root mean square residual of at most ``max_rms``, with a
///   finite residual and state, whether or not the fit reports convergence;
/// - has an orbit bound to the Sun;
/// - keeps observations which move: two of them differ in direction by more
///   than five times their combined uncertainty;
/// - keeps at least ``min_sights_per_side`` observations on each side of the
///   largest time between two consecutive ones it keeps;
/// - keeps observations in at least two time groups: in exactly two only when
///   they are at most ``max_gap`` days apart, and in three or more only when
///   two of the groups hold at least ``min_sights_per_side`` observations
///   each, or all of them together at least three times that, so that a few
///   single detections, each its own group under a close cadence, do not
///   confirm a link. The time groups are the clusters of the kept
///   observations in time, parted wherever two consecutive ones are more
///   than ``group_span`` days apart.
///
/// When a side of the largest gap has too few observations, those are
/// dropped and the rest fit again from the same start under all the rules,
/// until the rule holds or too few observations remain; the two thirds are of
/// the candidate as given. A candidate whose observations all lie within
/// ``group_span`` days is turned down without a fit.
///
/// Of confirmed candidates which share an observation, the one which keeps
/// more observations wins, then the one with the lower residual, then the one
/// listed first. Candidates are fit in batches, those with the most
/// observations first, and one which shares an observation with a candidate
/// already accepted is not fit. A candidate whose fit fails is not confirmed,
/// and is not an error.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     The observations the candidates index, in the order the
///     :class:`Sights` were built from, such as with
///     :meth:`Sights.from_observations`.
/// candidates : list[Candidate]
///     Candidates, such as from :func:`link`.
/// min_sights : int
///     Fewest observations a confirmed fit keeps, at least 4. Default is 4,
///     that of :func:`search`.
/// max_rms : float
///     Largest reduced root mean square residual, in units of the observation
///     uncertainties. Finite and positive. Default is 2, that of
///     :func:`search`.
/// min_sights_per_side : int
///     Fewest observations a confirmed fit keeps on each side of its largest
///     time gap, and the count of observations a fit of three or more time
///     groups needs, see above; at least 1. Default is 2, that of
///     :func:`search`; 1 turns both rules off.
/// group_span : float
///     Largest time between consecutive detections of one time group, in
///     days. Default is 0.1, that of :func:`search`. It must be below the time
///     between the revisits of a field by the survey, such as the orbit of a
///     spacecraft which revisits a field every orbit, else the revisits of a
///     whole visit are one time group.
/// max_gap : float
///     Largest time between the two time groups of a confirmation with only
///     two, in days, the ``max_gap`` the units were formed with. Default is 1.5.
///     Infinite turns the rule off.
///
/// Returns
/// -------
/// list[Confirmed]
///
/// Raises
/// ------
/// ValueError
///     If ``min_sights`` is below 4, ``min_sights_per_side`` is below 1,
///     ``max_rms`` is not finite and positive, ``group_span`` is negative or
///     not finite, ``max_gap`` is negative or not a number, a candidate
///     indexes past the observations, an optical observation of a candidate has a line of sight
///     which :class:`Sights` would refuse, the loaded kernels do not cover the
///     Sun at the epoch of a candidate or of one of its optical observations,
///     or they do not cover the Sun, the planets and the Moon, which the orbit
///     fit pulls with, from a day before the first to the last of the epochs
///     of the candidates and of their observations, or if an orbit fit fails for
///     want of data the kernels do not hold.
#[pyfunction]
#[pyo3(
    name = "confirm",
    signature = (
        observations,
        candidates,
        min_sights=default_count(ConfirmRules::default().min_sights),
        max_rms=ConfirmRules::default().max_rms,
        min_sights_per_side=default_count(ConfirmRules::default().min_sights_per_side),
        group_span=ConfirmRules::default().group_span,
        max_gap=ConfirmRules::default().max_gap,
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of confirm are keyword arguments"
)]
pub fn confirm_py(
    py: Python<'_>,
    observations: Vec<PyObservation>,
    candidates: Vec<PyCandidate>,
    min_sights: i64,
    max_rms: f64,
    min_sights_per_side: i64,
    group_span: f64,
    max_gap: f64,
) -> PyResult<Vec<PyConfirmed>> {
    let rules = confirm_rules(
        min_sights,
        max_rms,
        min_sights_per_side,
        group_span,
        max_gap,
    )?;
    let observations: Vec<AstrometricObservation> =
        observations.into_iter().map(|o| o.obs).collect();
    let candidates: Vec<Candidate> = candidates.into_iter().map(|c| c.0).collect();
    let confirmed = py.detach(|| confirm(&observations, &candidates, &rules))?;
    Ok(confirmed_py(confirmed))
}

/// Link and confirm observations in passes, each over the observations no
/// earlier pass confirmed.
///
/// The lines of sight of the optical observations are made once. Each pass
/// forms the units of those remaining with :func:`form_units`, builds a grid
/// with :func:`hypothesis_grid` at the steps of the pass, links with
/// :func:`link`, and confirms with :func:`confirm`. The observations it
/// confirms are then removed; those which were only in a candidate stay for
/// the later passes. Every setting is checked before the first pass. The
/// confirmed objects of all passes are then merged where several are pieces
/// of one object, see :func:`merge_pieces`.
///
/// Each pass takes as the origin of its hypotheses the observer of the
/// detection of a unit nearest the middle of the time the detections of its
/// units span, and the time of that detection as the reference time. A pass
/// with no units is skipped. Radar observations are passed over.
///
/// The defaults suit main belt objects seen by surveys on the ground in
/// tracklets within a night: a first pass at one and a half times
/// :data:`DEPTH_STEP` and :data:`RATE_STEP` and up to 0.6 degrees a day, then a
/// second at three tenths of them and up to a degree a day. ``tracklet_span``
/// suits a survey when it is a little longer than the longest time between
/// revisits of a field within a night. A search aimed at objects within a few
/// hundredths of an AU takes ``tracklet_span`` and ``group_span`` of a
/// hundredth of a day or less.
///
/// With ``stations``, confirmation scales the stated uncertainties of each
/// station to the scatter of its detections. The factor of each station is
/// estimated once, from the objects the first pass confirms: the square root
/// of the sum of the chi-squares of the residuals over the stated
/// uncertainties and their correlation of the detections their fits keep,
/// over the sum expected, ``2 (1 - 3 / n)``
/// for each detection of a fit which keeps ``n``. It applies to a station
/// with at least 100 detections kept by the fits of at least 20 objects, and
/// is at least 1; every other station keeps its stated uncertainties. The
/// candidates of the first pass are then confirmed again with the scaled
/// uncertainties, and the later passes and :func:`merge_pieces` confirm with
/// them too. Linking always uses the stated uncertainties. The fits of the
/// confirmed objects carry the scaled uncertainties, so their residuals and
/// ``rms`` are in those units.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     The observations to search, such as from
///     :func:`~kete.orbit_fitting.mpc_obs_to_observations` with
///     ``apply_over_obs_reweight=False``. Its default reweighting is for
///     fitting the observations of one object: it inflates the uncertainties
///     of an observatory's observations on a night by ``sqrt(n / 4)`` when it
///     made ``n > 4`` of them, counted over the whole list, which for a
///     survey's detections is many times over, and loosens linking,
///     confirmation and the scaling of ``stations``.
/// stations : iterable, optional
///     One hashable label for each observation, such as its MPC observatory
///     code: the observations of one station have their uncertainties scaled
///     together, see above. Default is None, which keeps the stated
///     uncertainties. :func:`~kete.orbit_fitting.mpc_obs_to_observations`
///     skips records it cannot convert, so build the labels from the
///     observations it returns rather than from the records, such as
///     ``[o.observer.desig for o in observations]``, which is the
///     observatory code of each.
/// parallax_margin : float
///     Largest angle between the directions of one object seen at one time by
///     two of the observers, in degrees, see :func:`parallax_margin`: it widens
///     the pairs, see :func:`form_units`, and in :func:`search_windows` the
///     halos of the tiles. Default is 0, which suits observers no farther apart
///     than the sites of the Earth, for objects past a few tenths of an AU.
///     Observers farther apart, such as one on the ground and one in space,
///     need the margin of their separation and ``min_depth``.
/// max_gap : float
///     Largest time between the detections of a pair, in days, and between the
///     two time groups of a confirmation with only two. Default is 1.5.
/// tracklet_span : float
///     Longest tracklet, in days. Default is 0.125.
/// min_tracklet : int
///     Fewest detections of a tracklet, at least 2, see :func:`form_units`.
///     Default is 2, which keeps every tracklet.
/// lone_pairs : bool
///     Whether two lone detections, those in no tracklet, may be paired, see
///     :func:`form_units`. Default is False, which suits any survey that sees
///     each object several times in a visit, such as a spacecraft which
///     revisits a field every orbit, or one on the ground which takes its
///     visits in pairs: the pairs, and so the memory and time of a pass, then
///     grow with the density of false detections rather than its square, see
///     :data:`MAX_UNITS`. Objects seen fewer than ``min_tracklet`` times in a
///     visit are found only through the tracklets of their other visits. True
///     suits a survey which sees most objects once per visit, in data with few
///     false detections.
/// group_span : float
///     Largest time between consecutive detections of one time group of a
///     confirmation, in days. Default is 0.1. It must be below the time
///     between the revisits of a field by the survey, such as the orbit of a
///     spacecraft which revisits a field every orbit, else the revisits of a
///     whole visit are one time group.
/// min_depth : float
///     Smallest depth of the hypotheses, in AU. Default is 0.5. The size of
///     the grid grows as this to the power -1.5.
/// max_depth : float
///     Largest depth of the hypotheses, in AU. Default is 4.5.
/// passes : list[tuple[float, float]]
///     One ``(grid_factor, max_rate)`` for each pass, in the order they run,
///     at least one: the factor on both :data:`DEPTH_STEP` and
///     :data:`RATE_STEP` for the grid of the pass, and its largest apparent
///     rate of motion within a tracklet and of a pair, in degrees per day, see
///     :func:`form_units`; both finite and positive. For pairs of detections
///     from observers at different places the rate also has to cover the
///     parallax between them over the time of the pair. Default is
///     ``[(1.5, 0.6), (0.3, 1.0)]``.
/// threshold : float
///     ``threshold`` of :func:`link`. Default is 4.
/// grid_allowance : float
///     ``grid_allowance`` of :func:`link`, in arcseconds. Default is about
///     20.6.
/// max_uncertainty : float
///     ``max_uncertainty`` of :func:`link`. Default is 0.05.
/// min_sights : int
///     ``min_sights`` of :func:`confirm`, the fewest observations a confirmed
///     fit keeps, at least 4, and of :func:`link`, the fewest detections in a
///     candidate. Default is 4.
/// max_rms : float
///     ``max_rms`` of :func:`confirm`, in units of the observation
///     uncertainties. Default is 2.
/// min_sights_per_side : int
///     ``min_sights_per_side`` of :func:`confirm`. Default is 2; 1 turns its
///     rules off.
///
/// Returns
/// -------
/// list[Confirmed]
///     The confirmed objects of every pass, those of the first pass first,
///     with :attr:`Confirmed.sights` indexing ``observations``, and merged
///     objects in the place of their first piece. No two share an
///     observation.
///
/// Raises
/// ------
/// ValueError
///     If ``passes`` is empty or a ``grid_factor`` is not finite and positive,
///     if ``stations`` does not hold one label per observation, if a pass forms
///     more than :data:`MAX_UNITS` units,
///     for settings that :func:`form_units`, :func:`hypothesis_grid`,
///     :func:`link` or :func:`confirm` refuse, for the first optical
///     observation whose line of sight :class:`Sights` would refuse, or if the
///     loaded kernels do not cover the Sun at the epoch of an optical
///     observation.
#[pyfunction]
#[pyo3(
    name = "search",
    signature = (
        observations,
        stations=None,
        parallax_margin=SearchSettings::default().parallax_margin.to_degrees(),
        max_gap=ConfirmRules::default().max_gap,
        tracklet_span=SearchSettings::default().tracklet_span,
        min_tracklet=default_count(SearchSettings::default().min_tracklet),
        lone_pairs=SearchSettings::default().lone_pairs,
        group_span=ConfirmRules::default().group_span,
        min_depth=SearchSettings::default().min_depth,
        max_depth=SearchSettings::default().max_depth,
        passes=default_passes(),
        threshold=LinkSettings::default().threshold,
        grid_allowance=LinkSettings::default().grid_allowance * RAD_TO_ARCSEC,
        max_uncertainty=LinkSettings::default().max_uncertainty,
        min_sights=default_count(ConfirmRules::default().min_sights),
        max_rms=ConfirmRules::default().max_rms,
        min_sights_per_side=default_count(ConfirmRules::default().min_sights_per_side),
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of search are keyword arguments"
)]
pub fn search_py(
    py: Python<'_>,
    observations: Vec<PyObservation>,
    stations: Option<Bound<'_, PyAny>>,
    parallax_margin: f64,
    max_gap: f64,
    tracklet_span: f64,
    min_tracklet: i64,
    lone_pairs: bool,
    group_span: f64,
    min_depth: f64,
    max_depth: f64,
    passes: Vec<(f64, f64)>,
    threshold: f64,
    grid_allowance: f64,
    max_uncertainty: f64,
    min_sights: i64,
    max_rms: f64,
    min_sights_per_side: i64,
) -> PyResult<Vec<PyConfirmed>> {
    let settings = search_settings(
        (parallax_margin, max_gap),
        (tracklet_span, group_span),
        (min_tracklet, lone_pairs),
        (min_depth, max_depth, passes),
        (threshold, grid_allowance, max_uncertainty),
        (min_sights, max_rms, min_sights_per_side),
    )?;
    let observations: Vec<AstrometricObservation> =
        observations.into_iter().map(|o| o.obs).collect();
    let stations = station_indices(py, stations)?;
    let confirmed = py.detach(|| search(&observations, stations.as_deref(), &settings))?;
    Ok(confirmed_py(confirmed))
}

/// The default passes of :func:`search`, as ``(grid_factor, max_rate)`` with
/// the rate in degrees per day.
fn default_passes() -> Vec<(f64, f64)> {
    SearchSettings::default()
        .passes
        .iter()
        .map(|pass| (pass.grid_factor, pass.max_rate.to_degrees()))
        .collect()
}

/// The settings of a search from the keyword arguments of :func:`search`, in
/// their Python units, grouped as pairs, spans, tracklets, grid, link and
/// confirm settings.
fn search_settings(
    (parallax_margin, max_gap): (f64, f64),
    (tracklet_span, group_span): (f64, f64),
    (min_tracklet, lone_pairs): (i64, bool),
    (min_depth, max_depth, passes): (f64, f64, Vec<(f64, f64)>),
    (threshold, grid_allowance, max_uncertainty): (f64, f64, f64),
    (min_sights, max_rms, min_sights_per_side): (i64, f64, i64),
) -> PyResult<SearchSettings> {
    let passes = passes
        .into_iter()
        .enumerate()
        .map(|(k, (grid_factor, max_rate))| {
            if !(grid_factor.is_finite() && grid_factor > 0.0) {
                Err(Error::ValueError(format!(
                    "Search: the grid_factor of pass {k} must be finite and positive, got \
                     {grid_factor}"
                )))?;
            }
            let what = format!("Search: the max_rate of pass {k}");
            Ok(SearchPass {
                grid_factor,
                max_rate: radians(&what, max_rate, true, false)?,
            })
        })
        .collect::<PyResult<_>>()?;
    Ok(SearchSettings {
        parallax_margin: radians("Search: parallax_margin", parallax_margin, false, true)?,
        tracklet_span,
        min_tracklet: count("Search: min_tracklet", min_tracklet)?,
        lone_pairs,
        min_depth,
        max_depth,
        passes,
        link: link_settings(threshold, grid_allowance, max_uncertainty),
        confirm: confirm_rules(
            min_sights,
            max_rms,
            min_sights_per_side,
            group_span,
            max_gap,
        )?,
    })
}

/// Largest angle between the directions of an object seen at one time from two
/// observers, in degrees.
///
/// For observers ``separation`` AU apart and an object at least
/// ``min_distance`` AU from them it is ``asin(separation / min_distance)``, or
/// 180 when the separation is as large as the distance. This is the
/// ``parallax_margin`` of :func:`search`, :func:`search_windows`,
/// :func:`form_units` and :func:`tiles` for observers at most that far apart
/// at one time, searched down to that distance, such as ``min_depth`` of the
/// search. For sites on the Earth the separation is at most the Earth's
/// diameter; for a spacecraft and a site on the Earth it is their largest
/// distance apart during the observations. The pairs of :func:`form_units`
/// allow for the parallax of sites on the Earth for objects past a few tenths
/// of an AU without a margin. The tiles of :func:`tiles` hold close objects
/// whole only with a margin even for one observer: the turning of the Earth,
/// or the orbit of a spacecraft about it, moves the direction of a close
/// object back and forth by more than its rate shows over a window, so one
/// observer there needs the margin of the diameter of the Earth, or of that
/// orbit.
///
/// Parameters
/// ----------
/// separation : float
///     Largest distance between two observers at one time, in AU, not negative.
/// min_distance : float
///     Smallest distance of the objects from the observers, in AU, positive.
///
/// Returns
/// -------
/// float
///     The margin, in degrees.
///
/// Raises
/// ------
/// ValueError
///     If ``separation`` is negative or ``min_distance`` is not positive, or
///     either is not finite.
#[pyfunction]
#[pyo3(name = "parallax_margin")]
pub fn parallax_margin_py(separation: f64, min_distance: f64) -> PyResult<f64> {
    Ok(parallax_margin(separation, min_distance)?.to_degrees())
}

/// Tiles of the sky, each the indices of the detections in it.
///
/// The core of a tile is one HEALPix pixel at ``nside``, in the nested order,
/// of right ascension and declination in the Equatorial frame. A
/// detection at time ``t`` joins every tile whose core is within its halo,
/// ``max_rate * |t - reference| + parallax_margin``: the angle from the center
/// of the core to the detection is at most the halo plus the largest angle
/// from that center to a point of the core.
///
/// Every object whose direction at the reference time lies in a core then has
/// all of its detections in that tile, if its apparent rate of motion is never
/// more than ``max_rate`` and ``parallax_margin`` bounds the angle between its
/// directions from two observers at one time. Each direction lies in one core,
/// so each such object is whole in at least one tile, and a search of that
/// tile can find it. :func:`search_windows` cuts and searches the tiles of each
/// pass with that pass's rate limit, over the observations the earlier passes
/// left.
///
/// The halo grows with the time from the reference time, so the tiles of a
/// long window overlap more. :func:`search_windows` searches a long survey in
/// windows of time.
///
/// Unless ``nside`` is given, it is the largest power of two whose cores, as
/// the square root of their solid angle, are at least as large as the largest
/// halo of a detection, or 1 if none is, and at most 1024. A detection joins
/// about as many tiles as cores fit in a disc of its halo plus the radius of a
/// core, so the overlap is least for cores large against the halo, while the
/// memory of searching a tile grows with its detections. Give a larger
/// ``nside`` for smaller tiles.
///
/// Parameters
/// ----------
/// sights : Sights
///     The detections.
/// max_rate : float
///     Largest apparent rate of motion of the objects, in degrees per day, the
///     rate limit of a pass of the search.
/// parallax_margin : float
///     Largest angle between the directions of an object at one time from two
///     observers of the detections, in degrees: :func:`parallax_margin` of
///     the largest separation of two observers at one time and the smallest
///     depth searched. One observer needs that of its own excursion about a
///     smooth path, see :func:`parallax_margin`; 0 suits one on a smooth
///     path, such as a spacecraft far from the Earth. Default is 0.
/// nside : int, optional
///     HEALPix resolution of the cores, a power of two. Default is taken from
///     the halo as above.
/// reference : float or Time, optional
///     Time the cores hold the directions of the objects at, JD in the TDB
///     scale. Default is the middle of the time the detections span.
///
/// Returns
/// -------
/// list[list[int]]
///     The row indices of the detections in each tile, sorted, in the order of
///     the pixel of the tile. Tiles with no detections are left out.
///
/// Raises
/// ------
/// ValueError
///     Unless ``max_rate`` is finite and positive and ``parallax_margin``
///     finite and not negative, if ``reference`` is not finite or is more than
///     9e15 days from JD 0, ``nside`` is not a power of two up to 2**29, or the
///     tiles would be so small against the halo that a detection would be
///     tested against more than a thousand of them.
#[pyfunction]
#[pyo3(name = "tiles", signature = (sights, max_rate, parallax_margin=0.0, nside=None, reference=None))]
pub fn tiles_py(
    py: Python<'_>,
    sights: &PySights,
    max_rate: f64,
    parallax_margin: f64,
    nside: Option<i64>,
    reference: Option<PyTime>,
) -> PyResult<Vec<Vec<usize>>> {
    let max_rate = radians("Tiles: max_rate", max_rate, true, false)?;
    let margin = radians("Tiles: parallax_margin", parallax_margin, false, true)?;
    let (nside, reference) = (checked_nside(nside)?, reference.map(Time::from));
    let sights = &sights.sights;
    Ok(py.detach(|| tiles(sights, max_rate, margin, nside, reference))?)
}

/// :func:`search` over a whole survey, one tile of the sky at a time, in
/// windows of time.
///
/// Each pass runs over the optical observations no earlier pass confirmed,
/// cut into :func:`tiles` with the rate limit of that pass,
/// ``parallax_margin`` and ``nside``, about the middle of the time those
/// observations span. Each tile with at least ``min_sights`` observations is
/// searched in turn by that pass of :func:`search`, with every thread, so
/// that memory holds one tile's search at a time. A tile takes its own
/// origin and grid, so it finds what a search of its own observations finds,
/// which can differ a little from what a search of all of them finds. The
/// objects of all tiles of a pass are resolved by the overlap rule of
/// :func:`confirm`, so an object confirmed in several tiles is kept once,
/// and their observations are removed before the next pass. Radar
/// observations are passed over.
///
/// The windows are ``2 * window`` days long, and start at the first optical
/// observation and every ``window`` days after it, so any stretch of
/// ``window`` days lies whole in at least one window; the last holds
/// everything from its start on. Without ``window`` the whole span is one
/// window. The windows are searched in turn, each over the observations no
/// earlier window confirmed, and the pieces of all windows and passes are
/// merged by :func:`merge_pieces`. Every setting, and the tiles of every pass
/// for the longest window, are checked before the first window is searched.
/// An object whose observations spread over more than ``window`` days is
/// found only where a window holds enough of it to confirm, so ``window`` is
/// chosen for the cadence of the survey.
///
/// With ``stations``, each window estimates its own factors as :func:`search`
/// does, from the objects its first pass confirms in all of its tiles; the
/// candidates of every tile of that pass are held in memory until then.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     The observations to search, see :func:`search`, such as from
///     :func:`~kete.orbit_fitting.mpc_obs_to_observations` with
///     ``apply_over_obs_reweight=False``.
/// stations : iterable, optional
///     One hashable label for each observation, such as its MPC observatory
///     code, built from the observations rather than from the records they
///     came from, see :func:`search`. Default is None, which keeps the stated
///     uncertainties.
/// window : float, optional
///     Time between the starts of consecutive windows, in days, half the length
///     of each. Finite, and at least the larger of ``max_gap`` and
///     ``tracklet_span``. Default is None, one window over the whole span.
/// nside : int, optional
///     See :func:`tiles`, the same for every pass and window. Default is taken
///     for each pass from its halo.
/// parallax_margin : float
///     See :func:`search`, in degrees. Default is 0.
/// max_gap : float
///     See :func:`search`, in days. Default is 1.5.
/// tracklet_span : float
///     See :func:`search`, in days. Default is 0.125.
/// min_tracklet : int
///     See :func:`search`. Default is 2.
/// lone_pairs : bool
///     See :func:`search`. Default is False.
/// group_span : float
///     See :func:`search`, in days. Default is 0.1.
/// min_depth : float
///     See :func:`search`, in AU. Default is 0.5.
/// max_depth : float
///     See :func:`search`, in AU. Default is 4.5.
/// passes : list[tuple[float, float]]
///     See :func:`search`, each rate limit in degrees per day; it also sets the
///     halo of the tiles of the pass. Default is ``[(1.5, 0.6), (0.3, 1.0)]``.
/// threshold : float
///     See :func:`search`. Default is 4.
/// grid_allowance : float
///     See :func:`search`, in arcseconds. Default is about 20.6.
/// max_uncertainty : float
///     See :func:`search`. Default is 0.05.
/// min_sights : int
///     See :func:`search`; also the fewest observations of a tile searched.
///     Default is 4.
/// max_rms : float
///     See :func:`search`. Default is 2.
/// min_sights_per_side : int
///     See :func:`search`. Default is 2.
///
/// Returns
/// -------
/// list[Confirmed]
///     The confirmed objects of every window, those of earlier windows first,
///     and within a window those of its first pass first, with
///     :attr:`Confirmed.sights` indexing ``observations``, and merged objects
///     in the place of their first piece, as by :func:`merge_pieces`. No two
///     share an observation.
///
/// Raises
/// ------
/// ValueError
///     If ``window`` is not finite and positive, is less than the larger of
///     ``max_gap`` and ``tracklet_span``, or makes more than a hundred thousand
///     windows, for settings that :func:`tiles` or :func:`search` refuse, for
///     the first optical observation whose line of sight :class:`Sights` would
///     refuse, or if the loaded kernels do not cover the Sun at the epoch of an
///     optical observation.
#[pyfunction]
#[pyo3(
    name = "search_windows",
    signature = (
        observations,
        stations=None,
        window=None,
        nside=None,
        parallax_margin=SearchSettings::default().parallax_margin.to_degrees(),
        max_gap=ConfirmRules::default().max_gap,
        tracklet_span=SearchSettings::default().tracklet_span,
        min_tracklet=default_count(SearchSettings::default().min_tracklet),
        lone_pairs=SearchSettings::default().lone_pairs,
        group_span=ConfirmRules::default().group_span,
        min_depth=SearchSettings::default().min_depth,
        max_depth=SearchSettings::default().max_depth,
        passes=default_passes(),
        threshold=LinkSettings::default().threshold,
        grid_allowance=LinkSettings::default().grid_allowance * RAD_TO_ARCSEC,
        max_uncertainty=LinkSettings::default().max_uncertainty,
        min_sights=default_count(ConfirmRules::default().min_sights),
        max_rms=ConfirmRules::default().max_rms,
        min_sights_per_side=default_count(ConfirmRules::default().min_sights_per_side),
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of search are keyword arguments"
)]
pub fn search_windows_py(
    py: Python<'_>,
    observations: Vec<PyObservation>,
    stations: Option<Bound<'_, PyAny>>,
    window: Option<f64>,
    nside: Option<i64>,
    parallax_margin: f64,
    max_gap: f64,
    tracklet_span: f64,
    min_tracklet: i64,
    lone_pairs: bool,
    group_span: f64,
    min_depth: f64,
    max_depth: f64,
    passes: Vec<(f64, f64)>,
    threshold: f64,
    grid_allowance: f64,
    max_uncertainty: f64,
    min_sights: i64,
    max_rms: f64,
    min_sights_per_side: i64,
) -> PyResult<Vec<PyConfirmed>> {
    let settings = search_settings(
        (parallax_margin, max_gap),
        (tracklet_span, group_span),
        (min_tracklet, lone_pairs),
        (min_depth, max_depth, passes),
        (threshold, grid_allowance, max_uncertainty),
        (min_sights, max_rms, min_sights_per_side),
    )?;
    let observations: Vec<AstrometricObservation> =
        observations.into_iter().map(|o| o.obs).collect();
    let nside = checked_nside(nside)?;
    let stations = station_indices(py, stations)?;
    let confirmed =
        py.detach(|| search_windows(&observations, stations.as_deref(), &settings, window, nside))?;
    Ok(confirmed_py(confirmed))
}

/// Confirmed objects with those which are pieces of one object merged.
///
/// A search can confirm one object in several pieces, which share no
/// detection: a pass confirms part of its detections and a later pass the part
/// the first left out, or the tiles of a pass each hold part of its arc.
/// :func:`search` and :func:`search_windows` merge them last; this is that step
/// on its own, for confirmed objects from anywhere.
///
/// Each piece gets a two-body orbit about the Sun fit to its own detections,
/// with the covariance of that fit. Pieces are compared when the orbit of one,
/// predicted in bins of a quarter of a day and of observer positions in a cube
/// a thousandth of an AU wide, from the mean observer of the bin, reaches a
/// cell of the sky where the other has a detection in that bin. Each bin holds
/// observers close together, so observers far apart, such as a spacecraft and
/// a site on the Earth, add no parallax to the reach.
/// The orbit of piece A predicts piece B if the chi-square of B's residuals
/// under A's orbit, with A's uncertainty, is at most 18 per detection of B,
/// twice the outlier threshold of 9 of the orbit fit of :func:`confirm`.
/// Pieces either of which predicts the other are fit as one candidate by the
/// rules of :func:`confirm`, and merged if that fit is confirmed, keeps two
/// thirds of the observations of each piece, and keeps more observations than
/// either piece alone. Merges go in rounds, the best fits first, and merged
/// objects are compared again in the next round, until none merges.
///
/// Parameters
/// ----------
/// observations : list[Observation]
///     The observations the pieces index.
/// pieces : list[Confirmed]
///     Confirmed objects sharing no observation, such as from :func:`confirm`.
/// min_sights : int
///     See :func:`confirm`. Default is 4.
/// max_rms : float
///     See :func:`confirm`. Default is 2.
/// min_sights_per_side : int
///     See :func:`confirm`. Default is 2.
/// group_span : float
///     See :func:`confirm`, in days. Default is 0.1.
/// max_gap : float
///     See :func:`confirm`, in days. Default is 1.5.
///
/// Returns
/// -------
/// list[Confirmed]
///     The pieces, each merged object in the place of its first piece and the
///     other piece removed.
///
/// Raises
/// ------
/// ValueError
///     For settings that :func:`confirm` refuses, if a piece indexes past the
///     observations, if an observation is in two pieces or twice in one, for
///     an optical observation of a piece whose line of
///     sight :class:`Sights` would refuse, or if the loaded kernels do not
///     cover the Sun at the epoch of one, or the planets and the Moon, which
///     the orbit fit pulls with, from a day before the first to the last
///     epoch of the observations of the pieces.
#[pyfunction]
#[pyo3(
    name = "merge_pieces",
    signature = (
        observations,
        pieces,
        min_sights=default_count(ConfirmRules::default().min_sights),
        max_rms=ConfirmRules::default().max_rms,
        min_sights_per_side=default_count(ConfirmRules::default().min_sights_per_side),
        group_span=ConfirmRules::default().group_span,
        max_gap=ConfirmRules::default().max_gap,
    )
)]
#[allow(
    clippy::too_many_arguments,
    reason = "the settings of confirm are keyword arguments"
)]
pub fn merge_pieces_py(
    py: Python<'_>,
    observations: Vec<PyObservation>,
    pieces: Vec<PyRef<'_, PyConfirmed>>,
    min_sights: i64,
    max_rms: f64,
    min_sights_per_side: i64,
    group_span: f64,
    max_gap: f64,
) -> PyResult<Vec<PyConfirmed>> {
    let observations: Vec<AstrometricObservation> =
        observations.into_iter().map(|o| o.obs).collect();
    let rules = confirm_rules(
        min_sights,
        max_rms,
        min_sights_per_side,
        group_span,
        max_gap,
    )?;
    let pieces: Vec<Confirmed> = pieces
        .iter()
        .map(|object| Confirmed::from(&**object))
        .collect();
    let merged = py.detach(|| merge_pieces(&observations, pieces, &rules))?;
    Ok(confirmed_py(merged))
}
