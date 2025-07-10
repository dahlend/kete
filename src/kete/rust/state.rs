//! Python support for State vectors
use crate::desigs::NaifIDLike;
use crate::elements::PyCometElements;
use crate::frame::*;
use crate::time::PyTime;
use crate::vector::*;
use kete_core::frames::InertialFrame;
use kete_core::prelude::*;
use pyo3::prelude::*;

/// Representation of the state of an object at a specific moment in time.
///
/// Parameters
/// ----------
/// desig : str
///     Name of the object, optional.
/// jd :
///     The time of the state in TDB jd time, see :py:class:`kete.Time`.
/// pos :
///     Position of the object with respect to the center ID in au.
/// vel :
///     Velocity of the object with respect to the center ID in au / day.
/// frame :
///     The frame of reference defining the position and velocity vectors.
/// center_id :
///     The SPICE kernel ID which defines the central reference point, defaults to the
///     Sun (10).
#[pyclass(module = "kete", name = "State")]
#[derive(Clone, Debug)]
pub struct PyState {
    /// The raw state object, always in the Equatorial frame.
    pub raw: State<Equatorial>,

    /// Frame of reference used to define the coordinate system.
    pub frame: PyFrames,

    /// Cometary orbital elements of the state, computes on first use.
    pub elements: Option<Box<PyCometElements>>,
}

impl<T: InertialFrame> From<State<T>> for PyState {
    fn from(value: State<T>) -> Self {
        Self {
            raw: value.into_frame(),
            // note that the raw value is always equatorial, this frame
            // only specifies how the equatorial vector is displayed in python
            frame: PyFrames::Ecliptic,
            elements: None,
        }
    }
}

#[pymethods]
impl PyState {
    /// Construct a new State
    #[new]
    #[pyo3(signature = (desig, jd, pos, vel, frame=None, center_id=NaifIDLike::Int(10)))]
    pub fn new(
        desig: Option<String>,
        jd: PyTime,
        pos: VectorLike,
        vel: VectorLike,
        frame: Option<PyFrames>,
        center_id: Option<NaifIDLike>,
    ) -> Self {
        let desig = match desig {
            Some(name) => Desig::Name(name),
            None => Desig::Empty,
        };
        let center_id = center_id.map(|id| id.try_into().map(|(_, x)| x).unwrap_or(10));

        // if no frame is provided, but pos or vel have a frame, use that one.
        let frame = frame.unwrap_or({
            if let VectorLike::Vec(v) = &pos {
                v.frame()
            } else if let VectorLike::Vec(v) = &vel {
                v.frame()
            } else {
                PyFrames::Ecliptic
            }
        });

        // change all vectors into equatorial.
        let pos = pos.into_vector(frame);
        let vel = vel.into_vector(frame);

        let center_id = center_id.unwrap_or(10);
        let state = State::new(desig, jd.jd(), pos, vel, center_id);
        Self {
            raw: state,
            frame,
            elements: None,
        }
    }

    /// Change the center ID of the state from the current state to the target state.
    ///
    /// If the desired state is not a known NAIF id this will raise an exception.
    pub fn change_center(&self, naif_id: NaifIDLike) -> PyResult<Self> {
        let naif_id: (String, i32) = naif_id.try_into()?;
        let mut state = self.raw.clone();
        let spk = LOADED_SPK.try_read().unwrap();
        spk.try_change_center(&mut state, naif_id.1)?;
        Ok(Self {
            raw: state,
            frame: self.frame,
            elements: None,
        })
    }

    /// Change the frame of the state to the target frame.
    pub fn change_frame(&self, frame: PyFrames) -> Self {
        let raw = self.raw.clone();
        Self {
            raw,
            frame,
            elements: None,
        }
    }

    /// Convert state to the Ecliptic Frame.
    #[getter]
    pub fn as_ecliptic(&self) -> Self {
        self.change_frame(PyFrames::Ecliptic)
    }

    /// Convert state to the Equatorial Frame.
    #[getter]
    pub fn as_equatorial(&self) -> Self {
        self.change_frame(PyFrames::Equatorial)
    }

    /// Convert state to the Galactic Frame.
    #[getter]
    pub fn as_galactic(&self) -> Self {
        self.change_frame(PyFrames::Galactic)
    }

    /// Convert state to the FK4 Frame.
    #[getter]
    pub fn as_fk4(&self) -> Self {
        self.change_frame(PyFrames::FK4)
    }

    /// JD of the object's state in TDB scaled time.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.raw.jd
    }

    /// Position of the object in AU with respect to the central object.
    #[getter]
    pub fn pos(&self) -> PyVector {
        PyVector::new(self.raw.pos, self.frame)
    }

    /// Velocity of the object in AU/Day.
    #[getter]
    pub fn vel(&self) -> PyVector {
        PyVector::new(self.raw.vel, self.frame)
    }

    /// Frame of reference used to define the coordinate system.
    #[getter]
    pub fn frame(&self) -> PyFrames {
        self.frame
    }

    /// Is the state finite?
    #[getter]
    pub fn is_finite(&self) -> bool {
        self.raw.pos.norm().is_finite()
            && self.raw.vel.norm().is_finite()
            && self.raw.jd.is_finite()
    }

    /// Central ID of the object used as reference for the coordinate frame.
    #[getter]
    pub fn center_id(&self) -> i32 {
        self.raw.center_id
    }

    /// Cometary orbital elements of the state.
    #[getter]
    pub fn elements(&mut self) -> PyCometElements {
        if self.elements.is_none() {
            self.elements = Some(Box::new(PyCometElements::from_state(self.clone())));
        }
        *self.elements.clone().unwrap()
    }

    /// Eccentricity of the orbit.
    #[getter]
    pub fn eccentricity(&mut self) -> f64 {
        self.elements().eccentricity()
    }

    /// Inclination of the orbit in degrees.
    #[getter]
    pub fn inclination(&mut self) -> f64 {
        self.elements().inclination()
    }

    /// Longitude of the ascending node of the orbit in degrees.
    #[getter]
    pub fn lon_of_ascending(&mut self) -> f64 {
        self.elements().lon_of_ascending()
    }

    /// Perihelion time of the orbit in JD.
    #[getter]
    pub fn peri_time(&mut self) -> f64 {
        self.elements().peri_time()
    }

    /// Argument of Perihelion of the orbit in degrees.
    #[getter]
    pub fn peri_arg(&mut self) -> f64 {
        self.elements().peri_arg()
    }

    /// Distance of Perihelion of the orbit in au.
    #[getter]
    pub fn peri_dist(&mut self) -> f64 {
        self.elements().peri_dist()
    }

    /// Distance of Aphelion of the orbit in au.
    #[getter]
    pub fn aphelion(&mut self) -> f64 {
        self.elements().aphelion()
    }

    /// Semi Major Axis of the orbit in au.
    #[getter]
    pub fn semi_major(&mut self) -> f64 {
        self.elements().semi_major()
    }

    /// Mean Motion of the orbit in degrees.
    #[getter]
    pub fn mean_motion(&mut self) -> f64 {
        self.elements().mean_motion()
    }

    /// Orbital Period in days, nan if non-elliptical.
    #[getter]
    pub fn orbital_period(&mut self) -> f64 {
        self.elements().orbital_period()
    }

    /// Eccentric Anomaly in degrees.
    #[getter]
    pub fn eccentric_anomaly(&mut self) -> PyResult<f64> {
        self.elements().eccentric_anomaly()
    }

    /// Mean Anomaly in degrees.
    #[getter]
    pub fn mean_anomaly(&mut self) -> f64 {
        self.elements().mean_anomaly()
    }

    /// True Anomaly in degrees.
    #[getter]
    pub fn true_anomaly(&mut self) -> PyResult<f64> {
        self.elements().true_anomaly()
    }

    /// Designation of the object if defined.
    #[getter]
    pub fn desig(&self) -> String {
        self.raw.desig.clone().try_naif_id_to_name().to_string()
    }

    /// Text representation of the state.
    pub fn __repr__(&self) -> String {
        let center = Desig::Naif(self.raw.center_id)
            .try_naif_id_to_name()
            .to_string();

        format!(
            "State(desig={:?}, jd={:?}, pos={:?}, vel={:?}, frame={:?}, center={:?})",
            self.desig(),
            self.jd(),
            self.pos().raw(),
            self.vel().raw(),
            self.frame(),
            center
        )
    }
}
