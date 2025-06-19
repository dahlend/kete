//! Python vector support with frame information.
use kete_core::frames::Vector;
use kete_core::util::Degrees;
use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError};
use std::f64::consts::FRAC_PI_2;

use crate::frame::*;
use kete_core::prelude::*;
use pyo3::prelude::*;

/// Vector class which is a vector along with a reference frame.
///
/// Vectors are always 3 dimensional cartesian coordinates, where the coordinate system
/// is defined by the frame. For example, the Ecliptic frame is the coordinate frame
/// of the solar system as used by JPL Horizons and SPICE.
///
/// Only inertial frames are supported in the Python wrapper, these are available in
/// the :py:class:`kete.Frames` enum.
///
/// Parameters
/// ----------
/// raw : list
///     3 floats which define the direction of the vector.
/// frame :
///     The frame of reference defining the coordinate frame of the vector, defaults
///     to ecliptic.
#[pyclass(sequence, frozen, module = "kete", name = "Vector")]
#[derive(Clone, Debug)]
pub struct PyVector {
    /// X/Y/Z numbers of the vector
    raw: Vector<Equatorial>,

    frame: PyFrames,
}
impl PyVector {
    /// Create a new vector
    pub fn new(raw: Vector<Equatorial>, frame: PyFrames) -> Self {
        Self { raw, frame }
    }
}

impl From<Vector<Equatorial>> for PyVector {
    fn from(value: Vector<Equatorial>) -> Self {
        Self {
            raw: value,
            // Note that the raw value is always equatorial, the frame here
            // only specifies how python displays it.
            frame: PyFrames::Ecliptic,
        }
    }
}

/// Polymorphic support
#[derive(Debug, FromPyObject, IntoPyObject)]
pub enum VectorLike {
    /// Vector directly
    Vec(PyVector),

    /// Vector from x/y/z
    Arr([f64; 3]),
}

impl VectorLike {
    /// Cast VectorLike into a Vector
    pub fn into_vector(self, frame: PyFrames) -> Vector<Equatorial> {
        match self {
            VectorLike::Arr(arr) => match frame {
                PyFrames::Equatorial => Vector::<Equatorial>::new(arr),
                PyFrames::Ecliptic => Vector::<Ecliptic>::new(arr).into_frame(),
                PyFrames::Galactic => Vector::<Galactic>::new(arr).into_frame(),
                PyFrames::FK4 => Vector::<FK4>::new(arr).into_frame(),
            },
            VectorLike::Vec(vec) => vec.raw,
        }
    }

    /// Cast VectorLike into a python Vector
    pub fn into_pyvector(self, frame: PyFrames) -> PyVector {
        let vec = self.into_vector(frame);
        PyVector::new(vec, frame)
    }
}

impl From<PyVector> for Vector<Equatorial> {
    fn from(value: PyVector) -> Self {
        value.raw
    }
}

#[pymethods]
impl PyVector {
    /// Create new vector
    #[new]
    #[pyo3(signature = (raw, frame=PyFrames::Ecliptic))]
    pub fn py_new(raw: VectorLike, frame: PyFrames) -> Self {
        PyVector::new(raw.into_vector(frame), frame)
    }

    /// Create a new Vector from the elevation and azimuthal angle in degrees.
    ///
    /// Parameters
    /// ----------
    /// el : float
    ///     Elevation above the X-Y plane of the frame. (Degrees)
    /// az : float
    ///     Azimuthal angle on the X-Y plane for the frame. (Degrees)
    /// frame : Frames
    ///     Frame of reference which define the coordinate axis.
    /// r :
    ///     Optional length of the vector, defaults to 1.
    #[staticmethod]
    pub fn from_el_az(el: f64, az: f64, r: f64, frame: PyFrames) -> Self {
        let (el_sin, el_cos) = (FRAC_PI_2 - el.to_radians()).sin_cos();
        let (az_sin, az_cos) = az.to_radians().sin_cos();
        let x = r * el_sin * az_cos;
        let y = r * el_sin * az_sin;
        let z = r * el_cos;
        VectorLike::Arr([x, y, z]).into_pyvector(frame)
    }

    /// Create a new Ecliptic Vector with the specified latitude/longitude.
    ///
    /// Parameters
    /// ----------
    /// lat : float
    ///     Latitude in the ecliptic frame. (Degrees)
    /// lon : float
    ///     Longitude in the ecliptic frame. (Degrees)
    /// r :
    ///     Optional length of the vector, defaults to 1.
    #[staticmethod]
    #[pyo3(signature = (lat, lon, r=1.0))]
    pub fn from_lat_lon(lat: f64, lon: f64, r: f64) -> Self {
        Self::from_el_az(lat, lon, r, PyFrames::Ecliptic)
    }

    /// Create a new Equatorial Vector with the specified RA/DEC.
    ///
    /// Parameters
    /// ----------
    /// ra : float
    ///     Right Ascension in the equatorial frame. (Degrees)
    /// dec : float
    ///     Declination in the equatorial frame. (Degrees)
    /// r :
    ///     Optional length of the vector, defaults to 1.
    #[staticmethod]
    #[pyo3(signature = (ra, dec, r=1.0))]
    pub fn from_ra_dec(ra: f64, dec: f64, r: f64) -> Self {
        Self::from_el_az(dec, ra, r, PyFrames::Equatorial)
    }

    /// The raw vector without the Frame.
    #[getter]
    pub fn raw(&self) -> [f64; 3] {
        match self.frame {
            PyFrames::Equatorial => self.raw.into(),
            PyFrames::Ecliptic => self.raw.into_frame::<Ecliptic>().into(),
            PyFrames::Galactic => self.raw.into_frame::<Galactic>().into(),
            PyFrames::FK4 => self.raw.into_frame::<FK4>().into(),
        }
    }

    /// The Frame of reference.
    #[getter]
    pub fn frame(&self) -> PyFrames {
        self.frame
    }

    /// Length of the Vector
    #[getter]
    pub fn r(&self) -> f64 {
        self.raw.norm()
    }

    /// Azimuth in degrees from the X axis in the X-Y plane of the coordinate frame.
    #[getter]
    pub fn az(&self) -> f64 {
        let data = self.raw();
        let r = self.r();
        if r < 1e-8 {
            return 0.0;
        }
        f64::atan2(data[1], data[0]).to_degrees().rem_euclid(360.0)
    }

    /// Elevation in degrees from the X-Y plane of the coordinate frame.
    /// Values will be between -180 and 180
    #[getter]
    pub fn el(&self) -> f64 {
        let data = self.raw();
        let r = self.r();
        if r < 1e-8 {
            return 0.0;
        }
        ((FRAC_PI_2 - (data[2] / r).clamp(-1.0, 1.0).acos()).to_degrees() + 180.0).rem_euclid(360.0)
            - 180.0
    }

    /// Right Ascension in degrees in the Equatorial Frame.
    #[getter]
    pub fn ra(&self) -> f64 {
        self.raw.to_ra_dec().0.to_degrees()
    }

    /// The right ascension, in hours-minutes-seconds string format.
    #[getter]
    pub fn ra_hms(&self) -> String {
        let deg = Degrees::from_radians(self.raw.to_ra_dec().0);
        deg.to_hms_str()
    }

    /// Declination in degrees in the Equatorial Frame.
    #[getter]
    pub fn dec(&self) -> f64 {
        self.raw.to_ra_dec().1.to_degrees()
    }

    /// The declination, in degrees-arcminutes-arcseconds string format.
    #[getter]
    pub fn dec_dms(&self) -> String {
        let deg = Degrees::from_radians(self.raw.to_ra_dec().1);
        deg.to_dms_str()
    }

    /// Latitude in degrees in the Ecliptic Frame.
    #[getter]
    pub fn lat(&self) -> f64 {
        let v = self.raw.into_frame::<Ecliptic>();
        v.to_lat_lon().0.to_degrees()
    }

    /// Longitude in degrees in the Ecliptic Frame.
    #[getter]
    pub fn lon(&self) -> f64 {
        let v = self.raw.into_frame::<Ecliptic>();
        v.to_lat_lon().1.to_degrees()
    }

    /// Compute the angle in degrees between two vectors in degrees.
    /// This will automatically make a frame change if necessary.
    pub fn angle_between(&self, other: VectorLike) -> f64 {
        let other_vec = other.into_vector(self.frame());
        self.raw.angle(&other_vec).to_degrees()
    }

    /// Return the vector in the ecliptic frame, regardless of starting frame.
    #[getter]
    pub fn as_ecliptic(&self) -> Self {
        self.change_frame(PyFrames::Ecliptic)
    }

    /// Return the vector in the equatorial frame, regardless of starting frame.
    #[getter]
    pub fn as_equatorial(&self) -> Self {
        self.change_frame(PyFrames::Equatorial)
    }

    /// Return the vector in the galactic frame, regardless of starting frame.
    #[getter]
    pub fn as_galactic(&self) -> Self {
        self.change_frame(PyFrames::Galactic)
    }

    /// Return the vector in the fk4 frame, regardless of starting frame.
    #[getter]
    pub fn as_fk4(&self) -> Self {
        self.change_frame(PyFrames::FK4)
    }

    /// Return the vector in the target frame, regardless of starting frame.
    pub fn change_frame(&self, target_frame: PyFrames) -> Self {
        let raw = self.raw;
        PyVector {
            raw,
            frame: target_frame,
        }
    }

    /// X coordinate in au.
    #[getter]
    pub fn x(&self) -> f64 {
        self.raw()[0]
    }

    /// Y coordinate in au.
    #[getter]
    pub fn y(&self) -> f64 {
        self.raw()[1]
    }

    /// Z coordinate in au.
    #[getter]
    pub fn z(&self) -> f64 {
        self.raw()[2]
    }

    /// Rotate this vector around another vector by the provided angle.
    ///
    ///
    /// Parameters
    /// ----------
    /// other : Vector
    ///     The other vector to rotate around.
    /// angle :
    ///     The angle in degrees of the rotation.
    pub fn rotate_around(&self, other: VectorLike, angle: f64) -> Self {
        let self_vec = self.raw;
        let other_vec = other.into_vector(self.frame());
        let new_vec = self_vec.rotate_around(other_vec, angle.to_radians());
        Self::new(new_vec, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __repr__(&self) -> String {
        // 1e-12 AU is about 15cm, this seems like a reasonable printing resolution
        let raw = self.raw();
        // adding 0.0 will flip the sign of -0.0 to 0.0
        let x = (raw[0] * 1e12).round() / 1e12 + 0.0;
        let y = (raw[1] * 1e12).round() / 1e12 + 0.0;
        let z = (raw[2] * 1e12).round() / 1e12 + 0.0;
        format!("Vector([{:?}, {:?}, {:?}], {:?})", x, y, z, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __sub__(&self, other: VectorLike) -> Self {
        let self_vec = self.raw;
        let other_vec = other.into_vector(self.frame());
        let diff = self_vec - other_vec;
        Self::new(diff, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __add__(&self, other: VectorLike) -> Self {
        let self_vec = self.raw;
        let other_vec = other.into_vector(self.frame());
        let diff = self_vec + other_vec;
        Self::new(diff, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __mul__(&self, other: f64) -> Self {
        let self_vec = self.raw;
        Self::new(self_vec * other, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __truediv__(&self, other: f64) -> Self {
        let self_vec = self.raw;
        Self::new(self_vec / other, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __neg__(&self) -> Self {
        Self::new(-self.raw, self.frame)
    }

    #[allow(missing_docs)]
    pub fn __len__(&self) -> usize {
        3
    }

    #[allow(missing_docs)]
    pub fn __getitem__(&self, idx: usize) -> PyResult<f64> {
        if idx >= 3 {
            return Err(PyErr::new::<PyIndexError, _>("Index out of bounds"));
        }
        Ok(self.raw()[idx])
    }

    fn __richcmp__(&self, other: VectorLike, op: CompareOp, _py: Python<'_>) -> PyResult<bool> {
        let self_vec = self.raw;
        let other_vec = other.into_vector(self.frame());
        match op {
            CompareOp::Eq => Ok((self_vec - other_vec).norm() < 1e-12),
            CompareOp::Ne => Ok((self_vec - other_vec).norm() >= 1e-12),
            _ => Err(PyNotImplementedError::new_err(
                "Vectors can only be checked for equality.",
            )),
        }
    }
}
