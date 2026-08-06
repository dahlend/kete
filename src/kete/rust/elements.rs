//! Python support for orbital elements
use kete_core::elements;
use kete_core::frames::Ecliptic;
use kete_core::prelude;
use kete_core::{constants::GMS_SQRT, forces::GravParams};
use nalgebra::Vector6;
use pyo3::{PyResult, pyclass, pymethods};

use crate::{state::PyState, time::PyTime};

/// Resolve the square root of a central body's gravitational parameter from its
/// NAIF ID, falling back to the Sun's when the body is unrecognized.
///
/// Shared by [`PyCometElements::new`] and [`PyEquinoctialElements::new`], which both
/// build an element set directly from user-supplied floats rather than from a `State`
/// and so need this looked up rather than carried along.
fn gm_sqrt_for(center_id: i32) -> f64 {
    let known = GravParams::known_masses();
    known
        .iter()
        .find(|p| p.naif_id == center_id)
        .map_or(GMS_SQRT, |p| p.mass.sqrt())
}

/// Cometary Elements class made accessible to python.
///
/// Angles must be in degrees, distances in AU.
///
/// Parameters
/// ----------
/// desig:
///     The designations of the object.
/// epoch:
///     The epoch time for the orbital elements.
/// eccentricity:
///     The eccentricity of the orbit.
/// inclination:
///     The inclination, must be in degrees.
/// peri_dist:
///     The perihelion distance in AU.
/// peri_arg:
///     The argument of perihelion, must be in degrees.
/// peri_time:
///     The JD time of perihelion.
/// lon_of_ascending:
///     The longitude of ascending node, in degrees.
/// center_id:
///     NAIF ID of the central body, defaults to 10 (Sun).
#[pyclass(module = "kete", frozen, name = "CometElements", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyCometElements(pub elements::CometElements);

#[pymethods]
impl PyCometElements {
    /// Construct a new CometElements object.
    ///
    /// Cometary elements are in the Ecliptic frame.
    ///
    /// Parameters
    /// ----------
    /// desig: str
    ///     Designation of the object.
    /// epoch: float
    ///     Epoch of the orbit fit in JD.
    /// eccentricity: float
    ///     Eccentricity of the orbit.
    /// inclination: float
    ///     Inclination of the orbit in degrees.
    /// peri_dist: float
    ///     Perihelion Distance in au.
    /// peri_arg: float
    ///     Argument of perihelion in degrees.
    /// peri_time: float
    ///     Time of perihelion passage in JD.
    /// lon_of_ascending: float
    ///     Longitude of ascending node in degrees.
    /// center_id: int
    ///     NAIF ID of the central body (default 10 = Sun).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(desig, epoch, eccentricity, inclination, peri_dist, peri_arg, peri_time, lon_of_ascending, center_id=10))]
    pub fn new(
        desig: String,
        epoch: PyTime,
        eccentricity: f64,
        inclination: f64,
        peri_dist: f64,
        peri_arg: f64,
        peri_time: PyTime,
        lon_of_ascending: f64,
        center_id: i32,
    ) -> Self {
        let gm_sqrt = gm_sqrt_for(center_id);
        Self(elements::CometElements {
            desig: prelude::Desig::Name(desig),
            epoch: epoch.into(),
            eccentricity,
            inclination: inclination.to_radians(),
            lon_of_ascending: lon_of_ascending.to_radians(),
            peri_time: peri_time.into(),
            peri_arg: peri_arg.to_radians(),
            peri_dist,
            center_id,
            gm_sqrt,
        })
    }

    /// Construct a new CometElements object from a `State`.
    ///
    /// Parameters
    /// ----------
    /// State :
    ///     State Object.
    #[staticmethod]
    pub fn from_state(state: PyState) -> PyResult<Self> {
        Ok(Self(elements::CometElements::from_state(
            &state.raw.into_frame(),
        )?))
    }

    /// Epoch of the elements in JD.
    #[getter]
    pub fn epoch(&self) -> PyTime {
        self.0.epoch.into()
    }

    /// Designation of the object.
    #[getter]
    pub fn desig(&self) -> String {
        match &self.0.desig {
            prelude::Desig::Naif(s) => {
                kete_core::desigs::try_name_from_id(*s).unwrap_or(s.to_string())
            }
            _ => self.0.desig.to_string(),
        }
    }

    /// Eccentricity of the orbit.
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.0.eccentricity
    }

    /// Inclination of the orbit in degrees.
    #[getter]
    pub fn inclination(&self) -> f64 {
        self.0.inclination.to_degrees()
    }

    /// Longitude of the ascending node of the orbit in degrees.
    #[getter]
    pub fn lon_of_ascending(&self) -> f64 {
        self.0.lon_of_ascending.to_degrees()
    }

    /// Perihelion time of the orbit in JD.
    #[getter]
    pub fn peri_time(&self) -> PyTime {
        self.0.peri_time.into()
    }

    /// NAIF ID of the central body.
    #[getter]
    pub fn center_id(&self) -> i32 {
        self.0.center_id
    }

    /// Argument of Perihelion of the orbit in degrees.
    #[getter]
    pub fn peri_arg(&self) -> f64 {
        self.0.peri_arg.to_degrees()
    }

    /// Distance of Perihelion of the orbit in au.
    #[getter]
    pub fn peri_dist(&self) -> f64 {
        self.0.peri_dist
    }

    /// Semi Major Axis of the orbit in au.
    #[getter]
    pub fn semi_major(&self) -> f64 {
        self.0.semi_major()
    }

    /// Mean Motion of the orbit in degrees per day.
    ///
    /// A parabolic orbit has no angular mean motion, and this is not an angular rate
    /// for one; see :attr:`mean_anomaly`.
    #[getter]
    pub fn mean_motion(&self) -> f64 {
        self.0.mean_motion().to_degrees()
    }

    /// Orbital Period in days, infinite if the orbit is not bound.
    #[getter]
    pub fn orbital_period(&self) -> f64 {
        self.0.orbital_period()
    }

    /// Eccentric Anomaly in degrees.
    #[getter]
    pub fn eccentric_anomaly(&self) -> PyResult<f64> {
        Ok(self.0.eccentric_anomaly().map(|x| x.to_degrees())?)
    }

    /// Mean Anomaly in degrees.
    ///
    /// Near-parabolic orbits, those within 1e-4 of unit eccentricity, are solved through
    /// Barker's equation, whose independent variable is not an angle. This value is not
    /// meaningful in degrees there; :attr:`true_anomaly` and :attr:`eccentric_anomaly`
    /// are.
    #[getter]
    pub fn mean_anomaly(&self) -> f64 {
        self.0.mean_anomaly().to_degrees()
    }

    /// Aphelion distance in au, infinite if the orbit is not bound.
    #[getter]
    pub fn aphelion(&self) -> f64 {
        self.0.aphelion()
    }

    /// True Anomaly in degrees.
    #[getter]
    pub fn true_anomaly(&self) -> PyResult<f64> {
        Ok(self.0.true_anomaly().map(|x| x.to_degrees())?)
    }

    /// Convert the orbital elements into a cartesian State.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        Ok(self.0.try_to_state()?.into_frame::<Ecliptic>().into())
    }

    fn __repr__(&self) -> String {
        format!(
            "CometElements(desig={:?}, epoch={}, eccentricity={}, inclination={}, lon_of_ascending={}, peri_time={}, peri_arg={}, peri_dist={}, center_id={})",
            self.desig(),
            self.epoch().jd(),
            self.eccentricity(),
            self.inclination(),
            self.lon_of_ascending(),
            self.peri_time().jd(),
            self.peri_arg(),
            self.peri_dist(),
            self.center_id()
        )
    }
}

/// Modified equinoctial orbital elements, in the formulation of Walker, Ireland and
/// Owens (1985).
///
/// Six floats, no constraints between them, describing one conic about a gravitating
/// body. This is the representation :attr:`~kete.UncertainState.cov_matrix` stores a
/// covariance over, and :attr:`~kete.UncertainState.elements` returns one of; unlike
/// :class:`CometElements` there is no Kepler solve or eccentricity branch anywhere in
/// its conversions, which is why it is what a fitted covariance is carried in.
///
/// ================ ======= ==================================================
/// attribute        symbol  meaning
/// ================ ======= ==================================================
/// ``semi_latus``    ``p``  semi-latus rectum in AU, positive on every conic
/// ``ecc_f``         ``f``  ``e * cos(peri_arg + lon_of_ascending)``
/// ``ecc_g``         ``g``  ``e * sin(peri_arg + lon_of_ascending)``
/// ``pole_h``        ``h``  ``tan(i / 2) * cos(lon_of_ascending)``
/// ``pole_k``        ``k``  ``tan(i / 2) * sin(lon_of_ascending)``
/// ``true_lon``      ``L``  true longitude at the epoch, in RADIANS
/// ================ ======= ==================================================
///
/// ``true_lon`` is in radians rather than degrees, unlike every angle
/// :class:`CometElements` reports -- these six are the stored representation itself,
/// matching :attr:`~kete.UncertainState.cov_matrix`, not a classical angle derived
/// from it. The derived quantities below (:attr:`inclination`, :attr:`true_anomaly`,
/// etc.) follow :class:`CometElements` and are in degrees.
///
/// Zero eccentricity and zero inclination are ordinary points, with no perihelion or
/// ascending node needed to measure an angle from. The one singularity is the
/// retrograde-equatorial pole, ``inclination = 180`` degrees exactly, where ``(h, k)``
/// diverges; :meth:`from_state` rejects states there.
///
/// Parameters
/// ----------
/// desig:
///     The designation of the object.
/// epoch:
///     The epoch time for the elements.
/// semi_latus:
///     Semi-latus rectum in AU.
/// ecc_f:
///     Eccentricity component ``f``.
/// ecc_g:
///     Eccentricity component ``g``.
/// pole_h:
///     Pole component ``h``.
/// pole_k:
///     Pole component ``k``.
/// true_lon:
///     True longitude at the epoch, in radians.
/// center_id:
///     NAIF ID of the central body, defaults to 10 (Sun).
#[pyclass(module = "kete", frozen, name = "EquinoctialElements", from_py_object)]
#[derive(Clone, Debug)]
pub struct PyEquinoctialElements(pub elements::EquinoctialElements);

#[pymethods]
impl PyEquinoctialElements {
    /// Construct a new EquinoctialElements object.
    ///
    /// Equinoctial elements are in the Ecliptic frame.
    ///
    /// Parameters
    /// ----------
    /// desig: str
    ///     Designation of the object.
    /// epoch: float
    ///     Epoch of the elements in JD.
    /// semi_latus: float
    ///     Semi-latus rectum in AU.
    /// ecc_f: float
    ///     Eccentricity component ``f``.
    /// ecc_g: float
    ///     Eccentricity component ``g``.
    /// pole_h: float
    ///     Pole component ``h``.
    /// pole_k: float
    ///     Pole component ``k``.
    /// true_lon: float
    ///     True longitude at the epoch, in radians.
    /// center_id: int
    ///     NAIF ID of the central body (default 10 = Sun).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature=(desig, epoch, semi_latus, ecc_f, ecc_g, pole_h, pole_k, true_lon, center_id=10))]
    pub fn new(
        desig: String,
        epoch: PyTime,
        semi_latus: f64,
        ecc_f: f64,
        ecc_g: f64,
        pole_h: f64,
        pole_k: f64,
        true_lon: f64,
        center_id: i32,
    ) -> Self {
        Self(elements::EquinoctialElements {
            desig: prelude::Desig::Name(desig),
            epoch: epoch.into(),
            semi_latus,
            ecc_f,
            ecc_g,
            pole_h,
            pole_k,
            true_lon,
            center_id,
            gm_sqrt: gm_sqrt_for(center_id),
        })
    }

    /// Construct a new EquinoctialElements object from a `State`.
    ///
    /// Parameters
    /// ----------
    /// State :
    ///     State Object.
    #[staticmethod]
    pub fn from_state(state: PyState) -> PyResult<Self> {
        Ok(Self(elements::EquinoctialElements::from_state(
            &state.raw.into_frame(),
        )?))
    }

    /// Epoch of the elements in JD.
    #[getter]
    pub fn epoch(&self) -> PyTime {
        self.0.epoch.into()
    }

    /// Designation of the object.
    #[getter]
    pub fn desig(&self) -> String {
        match &self.0.desig {
            prelude::Desig::Naif(s) => {
                kete_core::desigs::try_name_from_id(*s).unwrap_or(s.to_string())
            }
            _ => self.0.desig.to_string(),
        }
    }

    /// NAIF ID of the central body.
    #[getter]
    pub fn center_id(&self) -> i32 {
        self.0.center_id
    }

    /// Semi-latus rectum in AU, ``p``. Finite and positive on every conic.
    #[getter]
    pub fn semi_latus(&self) -> f64 {
        self.0.semi_latus
    }

    /// Eccentricity component ``f = e cos(peri_arg + lon_of_ascending)``.
    #[getter]
    pub fn ecc_f(&self) -> f64 {
        self.0.ecc_f
    }

    /// Eccentricity component ``g = e sin(peri_arg + lon_of_ascending)``.
    #[getter]
    pub fn ecc_g(&self) -> f64 {
        self.0.ecc_g
    }

    /// Pole component ``h = tan(i / 2) cos(lon_of_ascending)``.
    #[getter]
    pub fn pole_h(&self) -> f64 {
        self.0.pole_h
    }

    /// Pole component ``k = tan(i / 2) sin(lon_of_ascending)``.
    #[getter]
    pub fn pole_k(&self) -> f64 {
        self.0.pole_k
    }

    /// True longitude at the epoch, in RADIANS.
    ///
    /// Unlike every other angle on this class and on :class:`CometElements`, this is
    /// in radians -- it is the stored coordinate itself, matching row/column 5 of
    /// :attr:`~kete.UncertainState.cov_matrix`. It wraps; use :meth:`offset_to` rather
    /// than subtracting two values directly.
    #[getter]
    pub fn true_lon(&self) -> f64 {
        self.0.true_lon
    }

    /// Eccentricity.
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.0.eccentricity()
    }

    /// Semi major axis in AU. Infinite if the orbit is parabolic.
    #[getter]
    pub fn semi_major(&self) -> f64 {
        self.0.semi_major()
    }

    /// Inverse of the semi major axis, in 1/AU.
    ///
    /// Passes smoothly through zero at unit eccentricity, where the semi major axis
    /// itself diverges; prefer this over ``1 / semi_major`` near there.
    #[getter]
    pub fn inverse_semi_major(&self) -> f64 {
        self.0.inverse_semi_major()
    }

    /// Perihelion distance in AU. Finite and positive for every conic.
    #[getter]
    pub fn peri_dist(&self) -> f64 {
        self.0.peri_dist()
    }

    /// Aphelion distance in AU. Infinite if the orbit is not bound.
    #[getter]
    pub fn aphelion(&self) -> f64 {
        self.0.aphelion()
    }

    /// Orbital period in days. Infinite if the orbit is not bound.
    #[getter]
    pub fn orbital_period(&self) -> f64 {
        self.0.orbital_period()
    }

    /// Inclination in degrees, between 0 and 180.
    #[getter]
    pub fn inclination(&self) -> f64 {
        self.0.inclination().to_degrees()
    }

    /// Longitude of ascending node in degrees, between 0 and 360.
    ///
    /// Undefined for an uninclined orbit; zero is returned there rather than an error.
    #[getter]
    pub fn lon_of_ascending(&self) -> f64 {
        self.0.lon_of_ascending().to_degrees()
    }

    /// Longitude of perihelion in degrees, between 0 and 360.
    ///
    /// Unlike the argument of perihelion this is defined for an uninclined orbit,
    /// since it is measured from the reference direction rather than from the node.
    #[getter]
    pub fn lon_of_peri(&self) -> f64 {
        self.0.lon_of_peri().to_degrees()
    }

    /// Argument of perihelion in degrees, between 0 and 360.
    ///
    /// Meaningful only when both the eccentricity and the inclination are non-zero.
    #[getter]
    pub fn peri_arg(&self) -> f64 {
        self.0.peri_arg().to_degrees()
    }

    /// True anomaly at the epoch in degrees, between 0 and 360.
    ///
    /// Closed form and defined at every eccentricity; only meaningful when the
    /// eccentricity is non-zero.
    #[getter]
    pub fn true_anomaly(&self) -> f64 {
        self.0.true_anomaly().to_degrees()
    }

    /// Distance from the central body at the epoch, in AU.
    ///
    /// Only physical while the orbit equation is positive, see :attr:`state`.
    #[getter]
    pub fn epoch_distance(&self) -> f64 {
        self.0.epoch_distance()
    }

    /// Convert the elements into a cartesian State.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the true longitude is outside the physical domain of an open orbit.
    #[getter]
    pub fn state(&self) -> PyResult<PyState> {
        Ok(self.0.try_to_state()?.into_frame::<Ecliptic>().into())
    }

    /// The same orbit displaced by an offset of the six stored floats.
    ///
    /// Parameters
    /// ----------
    /// delta : list[float]
    ///     Length-6 offset, in the order ``[p, f, g, h, k, L]`` -- the same order as
    ///     :attr:`~kete.UncertainState.cov_matrix`. ``true_lon`` is not reduced by
    ///     this; see :meth:`offset_to`.
    pub fn displaced_by(&self, delta: Vec<f64>) -> PyResult<Self> {
        if delta.len() != 6 {
            return Err(kete_core::errors::Error::ValueError(format!(
                "delta must have length 6, got {}",
                delta.len()
            ))
            .into());
        }
        let delta = Vector6::from_row_slice(delta.as_slice());
        Ok(Self(self.0.displaced_by(&delta)))
    }

    /// The offset carrying this orbit to ``other``, the inverse of :meth:`displaced_by`.
    ///
    /// Five coordinates are a plain difference; ``true_lon`` is reduced to the
    /// shortest signed angle, so two orbits a whole turn apart read as coincident.
    ///
    /// Parameters
    /// ----------
    /// other : EquinoctialElements
    pub fn offset_to(&self, other: &Self) -> Vec<f64> {
        self.0.offset_to(&other.0).as_slice().to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "EquinoctialElements(desig={:?}, epoch={}, semi_latus={}, ecc_f={}, ecc_g={}, pole_h={}, pole_k={}, true_lon={}, center_id={})",
            self.desig(),
            self.epoch().jd(),
            self.semi_latus(),
            self.ecc_f(),
            self.ecc_g(),
            self.pole_h(),
            self.pole_k(),
            self.true_lon(),
            self.center_id()
        )
    }
}
