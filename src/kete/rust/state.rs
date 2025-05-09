//! Python support for State vectors
use crate::elements::PyCometElements;
use crate::frame::*;
use crate::time::PyTime;
use crate::vector::*;
use kete_core::frames::Equatorial;
use kete_core::prelude;
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
#[pyclass(frozen, module = "kete", name = "State")]
#[derive(Clone, Debug)]
pub struct PyState {
    /// The raw state object, always in the Equatorial frame.
    pub raw: prelude::State<Equatorial>,

    /// Frame of reference used to define the coordinate system.
    pub frame: PyFrames,
}

impl From<prelude::State<Equatorial>> for PyState {
    fn from(value: prelude::State<Equatorial>) -> Self {
        Self {
            raw: value,
            frame: PyFrames::Equatorial,
        }
    }
}

#[pymethods]
impl PyState {
    /// Construct a new State
    #[new]
    #[pyo3(signature = (desig, jd, pos, vel, frame=None, center_id=10))]
    pub fn new(
        desig: Option<String>,
        jd: PyTime,
        pos: VectorLike,
        vel: VectorLike,
        frame: Option<PyFrames>,
        center_id: Option<i32>,
    ) -> Self {
        let desig = match desig {
            Some(name) => prelude::Desig::Name(name),
            None => prelude::Desig::Empty,
        };

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
        let state = prelude::State::new(desig, jd.jd(), pos, vel, center_id);
        Self { raw: state, frame }
    }

    /// Change the center ID of the state from the current state to the target state.
    ///
    /// If the desired state is not a known NAIF id this will raise an exception.
    pub fn change_center(&self, naif_id: i32) -> PyResult<Self> {
        let mut state = self.raw.clone();
        let spk = prelude::LOADED_SPK.try_read().unwrap();
        spk.try_change_center(&mut state, naif_id)?;
        Ok(Self {
            raw: state,
            frame: self.frame,
        })
    }

    /// Change the frame of the state to the target frame.
    pub fn change_frame(&self, frame: PyFrames) -> Self {
        let raw = self.raw.clone();
        Self { raw, frame }
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

    /// Central ID of the object used as reference for the coordinate frame.
    #[getter]
    pub fn center_id(&self) -> i32 {
        self.raw.center_id
    }

    /// Cometary orbital elements of the state.
    #[getter]
    pub fn elements(&self) -> PyCometElements {
        PyCometElements::from_state(self.clone())
    }

    /// Designation of the object if defined.
    #[getter]
    pub fn desig(&self) -> String {
        match &self.raw.desig {
            prelude::Desig::Name(s) => s.clone(),
            prelude::Desig::Naif(s) => {
                kete_core::spice::try_name_from_id(*s).unwrap_or(s.to_string())
            }
            prelude::Desig::Perm(s) => format!("{:?}", s),
            prelude::Desig::Prov(s) => s.clone(),
            prelude::Desig::Empty => "None".into(),
        }
    }

    /// Text representation of the state.
    pub fn __repr__(&self) -> String {
        format!(
            "State(desig={:?}, jd={:?}, pos={:?}, vel={:?}, frame={:?}, center_id={:?})",
            self.desig(),
            self.jd(),
            self.pos().raw(),
            self.vel().raw(),
            self.frame(),
            self.center_id()
        )
    }
}
