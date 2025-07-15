use kete_core::fov::{self};
use kete_core::fov::{FovLike, SkyPatch};
use kete_core::frames::Vector;
use pyo3::{exceptions, prelude::*};

use crate::vector::VectorLike;
use crate::{state::PyState, vector::PyVector};

/// Field of view of a WISE CMOS chip.
/// Since all WISE CMOS see the same patch of sky, there is no differentiation
/// of the individual wavelength bands.
#[pyclass(module = "kete", frozen, name = "WiseCmos")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyWiseCmos(pub fov::WiseCmos);

/// Field of view of a NEOS CMOS chip.
///
/// Parameters
/// ----------
/// pointing :
///     Vector defining the center of the FOV.
/// rotation :
///     Rotation of the FOV in degrees.
/// observer :
///     State of the observer.
/// side_id :
///     Side ID indicating where we are in the survey.
/// stack_id :
///     Stack ID indicating where we are in the survey.
/// quad_id :
///     Quad ID indicating where we are in the survey.
/// loop_id :
///     Loop ID indicating where we are in the survey.
/// subloop_id :
///     Subloop ID indicating where we are in the survey.
/// exposure_id :
///     Exposure number indicating where we are in the survey.
/// cmos_id :
///     Which chip of the target band this represents.
/// band :
///     Band, can be either 1 or 2 to represent NC1/NC2.
#[pyclass(module = "kete", frozen, name = "NeosCmos")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyNeosCmos(pub fov::NeosCmos);

/// Field of view of a NEOS Visit.
///
/// This is a collection of 4 :py:class:`NeosCmos` chips at the same moment in time.
///
/// .. plot::
///     :context: close-figs
///     :include-source: false
///
///         import kete
///         import matplotlib.pyplot as plt
///
///         obs = kete.spice.get_state("earth", kete.Time.j2000())
///         vec = [0, 1, 0]
///         fov = kete.fov.NeosVisit(7.2, 1.6, 0.1, vec, 0, obs, 0, 0, 0, 0, 0, 0, 1)
///         plt.figure(figsize=(6, 2))             
///         for chip in fov:
///             corners = chip.corners
///             corners.append(corners[0])
///             plt.plot([x.as_ecliptic.lon for x in corners],
///                      [y.as_ecliptic.lat for y in corners],
///                      c="k")
///         plt.gca().set_aspect("equal")
///         plt.gca().set_axis_off()
///         plt.annotate("0", [0.07, 0.8], xycoords="axes fraction")
///         plt.annotate("1", [0.3, 0.8], xycoords="axes fraction")
///         plt.annotate("2", [0.53, 0.8], xycoords="axes fraction")
///         plt.annotate("3", [0.75, 0.8], xycoords="axes fraction")
///         arrow = dict(facecolor="black", shrink=0.01, width=1, headwidth=5)
///         plt.annotate(
///             "X",
///             [0.8, -0.07],
///             xytext=(0.2, -0.07),
///             xycoords="axes fraction",
///             arrowprops=arrow,
///             horizontalalignment="right",
///             verticalalignment="center",
///         )
///         plt.annotate(
///             "Y",
///             [0.98, 0.9],
///             xytext=(0.98, 0.1),
///             xycoords="axes fraction",
///             arrowprops=arrow,
///             horizontalalignment="center",
///             verticalalignment="center",
///         )
///         plt.annotate(
///             "Gap",
///             [0.265, 0.2],
///             xytext=(0.18, 0.2),
///             xycoords="axes fraction",
///             arrowprops=arrow,
///             horizontalalignment="center",
///             verticalalignment="center",
///         )
///         plt.annotate(
///             "",
///             [0.28, 0.2],
///             xytext=(0.34, 0.2),
///             xycoords="axes fraction",
///             arrowprops=arrow,
///             horizontalalignment="center",
///             verticalalignment="center",
///         )
///         plt.tight_layout()
//
///
/// Where the bottom is the sun shield.
///
/// Parameters
/// ----------
/// x_width :
///     Width of the long axis of the Visit in degrees.
/// y_width :
///     Width of the short axis of the Visit in degrees.
/// gap_angle :
///     Width of the gap between chips in degrees.
/// pointing :
///     Vector defining the center of the FOV.
/// rotation :
///     Rotation of the FOV in degrees.
/// observer :
///     State of the observer.
/// side_id :
///     Side ID indicating where we are in the survey.
/// stack_id :
///     Stack ID indicating where we are in the survey.
/// quad_id :
///     Quad ID indicating where we are in the survey.
/// loop_id :
///     Loop ID indicating where we are in the survey.
/// subloop_id :
///     Subloop ID indicating where we are in the survey.
/// exposure_id :
///     Exposure number indicating where we are in the survey.
/// band :
///     Band, can be either 1 or 2 to represent NC1/NC2.
#[pyclass(module = "kete", frozen, name = "NeosVisit")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyNeosVisit(pub fov::NeosVisit);

/// Field of view of a Single ZTF chips/quad combination.
#[pyclass(module = "kete", frozen, name = "ZtfCcdQuad")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyZtfCcdQuad(pub fov::ZtfCcdQuad);

/// Field of view of all 64 ZTF chips/quad combinations.
/// This is a meta collection of individual ZTF CCD Quad FOVs.
#[pyclass(module = "kete", frozen, name = "ZtfField", sequence)]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyZtfField(pub fov::ZtfField);

/// Field of view of all PTF ccds.
/// This is a meta collection of individual PTF CCD Quad FOVs.
#[pyclass(module = "kete", frozen, name = "PtfField", sequence)]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyPtfField(pub fov::PtfField);

/// Field of view of a Single PTF ccd.
#[pyclass(module = "kete", frozen, name = "PtfCcd")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyPtfCcd(pub fov::PtfCcd);

/// Field of view of multiple Spherex CMOS at one time.
/// This is a meta collection of individual Spherex CMOS FOVs.
#[pyclass(module = "kete", frozen, name = "SpherexField", sequence)]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PySpherexField(pub fov::SpherexField);

/// Field of view of a Single Spherex cmos.
#[pyclass(module = "kete", frozen, name = "SpherexCmos")]
#[derive(Clone, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub struct PySpherexCmos(pub fov::SpherexCmos);

/// Generic Rectangular Field of view.
///
/// There are other constructors for this, for example the
/// :py:meth:`~RectangleFOV.from_corners` which allows construction from the 4 corners
/// of the field.
///
/// Parameters
/// ----------
/// pointing :
///     Vector defining the center of the field of cone.
/// rotation :
///     The rotation of the field of view in degrees.
/// observer :
///     The state of the observer.
/// lon_width :
///     The longitudinal width of the rectangle in degrees.
/// lat_width:
///     The latitudinal width of the rectangle in degrees.
#[pyclass(module = "kete", frozen, name = "RectangleFOV")]
#[derive(Clone, Debug)]
pub struct PyGenericRectangle(pub fov::GenericRectangle);

/// Generic Cone field of view.
///
/// A cone directly out from the observer's location to a point on the sky.
///
/// Parameters
/// ----------
/// pointing :
///     Vector defining the center of the field of cone.
/// angle :
///     The radius of the cone in degrees, from the center to the edge of the cone.
/// observer :
///     The state of the observer.
#[pyclass(module = "kete", frozen, name = "ConeFOV")]
#[derive(Clone, Debug)]
pub struct PyGenericCone(pub fov::GenericCone);

/// Omni Directional field of view.
///
/// This is a good choice when the exact field is not important.
///
/// Parameters
/// ----------
/// observer :
///     State of the omniscient observer.
#[pyclass(module = "kete", frozen, name = "OmniDirectionalFOV")]
#[derive(Clone, Debug)]
pub struct PyOmniDirectional(pub fov::OmniDirectional);

/// Field of views supported by the python interface
#[derive(Debug, Clone, FromPyObject, IntoPyObject)]
#[allow(clippy::upper_case_acronyms, missing_docs)]
pub enum AllowedFOV {
    WISE(PyWiseCmos),
    NEOS(PyNeosCmos),
    ZTF(PyZtfCcdQuad),
    ZTFField(PyZtfField),
    NEOSVisit(PyNeosVisit),
    Rectangle(PyGenericRectangle),
    Cone(PyGenericCone),
    OmniDirectional(PyOmniDirectional),
    PTF(PyPtfCcd),
    PTFField(PyPtfField),
    SPHEREx(PySpherexCmos),
    SPHERExField(PySpherexField),
}

impl AllowedFOV {
    #[allow(missing_docs)]
    pub fn jd(&self) -> f64 {
        match self {
            AllowedFOV::NEOS(fov) => fov.0.observer().jd,
            AllowedFOV::WISE(fov) => fov.0.observer().jd,
            AllowedFOV::Rectangle(fov) => fov.0.observer().jd,
            AllowedFOV::ZTF(fov) => fov.0.observer().jd,
            AllowedFOV::ZTFField(fov) => fov.0.observer().jd,
            AllowedFOV::NEOSVisit(fov) => fov.0.observer().jd,
            AllowedFOV::Cone(fov) => fov.0.observer().jd,
            AllowedFOV::OmniDirectional(fov) => fov.0.observer().jd,
            AllowedFOV::PTF(fov) => fov.0.observer().jd,
            AllowedFOV::PTFField(fov) => fov.0.observer().jd,
            AllowedFOV::SPHEREx(fov) => fov.0.observer().jd,
            AllowedFOV::SPHERExField(fov) => fov.0.observer().jd,
        }
    }

    #[allow(missing_docs)]
    pub fn get_fov(self, idx: Option<usize>) -> fov::FOV {
        let idx = idx.unwrap_or_default();
        match self {
            AllowedFOV::WISE(fov) => fov.0.get_fov(idx),
            AllowedFOV::Rectangle(fov) => fov.0.get_fov(idx),
            AllowedFOV::NEOS(fov) => fov.0.get_fov(idx),
            AllowedFOV::ZTF(fov) => fov.0.get_fov(idx),
            AllowedFOV::ZTFField(fov) => fov.0.get_fov(idx),
            AllowedFOV::NEOSVisit(fov) => fov.0.get_fov(idx),
            AllowedFOV::Cone(fov) => fov.0.get_fov(idx),
            AllowedFOV::OmniDirectional(fov) => fov.0.get_fov(idx),
            AllowedFOV::PTF(fov) => fov.0.get_fov(idx),
            AllowedFOV::PTFField(fov) => fov.0.get_fov(idx),
            AllowedFOV::SPHEREx(fov) => fov.0.get_fov(idx),
            AllowedFOV::SPHERExField(fov) => fov.0.get_fov(idx),
        }
    }

    #[allow(missing_docs)]
    pub fn unwrap(self) -> fov::FOV {
        match self {
            AllowedFOV::WISE(fov) => fov::FOV::Wise(fov.0),
            AllowedFOV::Rectangle(fov) => fov::FOV::GenericRectangle(fov.0),
            AllowedFOV::NEOS(fov) => fov::FOV::NeosCmos(fov.0),
            AllowedFOV::ZTF(fov) => fov::FOV::ZtfCcdQuad(fov.0),
            AllowedFOV::ZTFField(fov) => fov::FOV::ZtfField(fov.0),
            AllowedFOV::NEOSVisit(fov) => fov::FOV::NeosVisit(fov.0),
            AllowedFOV::Cone(fov) => fov::FOV::GenericCone(fov.0),
            AllowedFOV::OmniDirectional(fov) => fov::FOV::OmniDirectional(fov.0),
            AllowedFOV::PTF(fov) => fov::FOV::PtfCcd(fov.0),
            AllowedFOV::PTFField(fov) => fov::FOV::PtfField(fov.0),
            AllowedFOV::SPHEREx(fov) => fov::FOV::SpherexCmos(fov.0),
            AllowedFOV::SPHERExField(fov) => fov::FOV::SpherexField(fov.0),
        }
    }

    #[allow(missing_docs)]
    pub fn __repr__(self) -> String {
        match self {
            AllowedFOV::WISE(fov) => fov.__repr__(),
            AllowedFOV::Rectangle(fov) => fov.__repr__(),
            AllowedFOV::NEOS(fov) => fov.__repr__(),
            AllowedFOV::ZTF(fov) => fov.__repr__(),
            AllowedFOV::ZTFField(fov) => fov.__repr__(),
            AllowedFOV::NEOSVisit(fov) => fov.__repr__(),
            AllowedFOV::Cone(fov) => fov.__repr__(),
            AllowedFOV::OmniDirectional(fov) => fov.__repr__(),
            AllowedFOV::PTF(fov) => fov.__repr__(),
            AllowedFOV::PTFField(fov) => fov.__repr__(),
            AllowedFOV::SPHEREx(fov) => fov.__repr__(),
            AllowedFOV::SPHERExField(fov) => fov.__repr__(),
        }
    }
}

impl From<fov::FOV> for AllowedFOV {
    fn from(value: fov::FOV) -> Self {
        match value {
            fov::FOV::Wise(fov) => AllowedFOV::WISE(PyWiseCmos(fov)),
            fov::FOV::ZtfCcdQuad(fov) => AllowedFOV::ZTF(PyZtfCcdQuad(fov)),
            fov::FOV::NeosCmos(fov) => AllowedFOV::NEOS(PyNeosCmos(fov)),
            fov::FOV::GenericRectangle(fov) => AllowedFOV::Rectangle(PyGenericRectangle(fov)),
            fov::FOV::ZtfField(fov) => AllowedFOV::ZTFField(PyZtfField(fov)),
            fov::FOV::NeosVisit(fov) => AllowedFOV::NEOSVisit(PyNeosVisit(fov)),
            fov::FOV::GenericCone(fov) => AllowedFOV::Cone(PyGenericCone(fov)),
            fov::FOV::OmniDirectional(fov) => AllowedFOV::OmniDirectional(PyOmniDirectional(fov)),
            fov::FOV::PtfCcd(fov) => AllowedFOV::PTF(PyPtfCcd(fov)),
            fov::FOV::PtfField(fov) => AllowedFOV::PTFField(PyPtfField(fov)),
            fov::FOV::SpherexCmos(fov) => AllowedFOV::SPHEREx(PySpherexCmos(fov)),
            fov::FOV::SpherexField(fov) => AllowedFOV::SPHERExField(PySpherexField(fov)),
        }
    }
}

#[pymethods]
impl PyWiseCmos {
    /// Construct a WISE CMOS fov from a pointing vector, rotation and observer state.
    #[staticmethod]
    pub fn from_pointing(
        pointing: VectorLike,
        rotation: f64,
        observer: PyState,
        frame_num: u64,
        scan_id: String,
    ) -> Self {
        let pointing = pointing.into_vector(observer.frame());
        let scan_id = scan_id.into();
        PyWiseCmos(fov::WiseCmos::new(
            pointing,
            rotation.to_radians(),
            observer.raw,
            frame_num,
            scan_id,
        ))
    }

    /// Construct a WISE CMOS fov the corners of the FOV and observer state.
    #[new]
    pub fn new(
        corners: [VectorLike; 4],
        observer: PyState,
        frame_num: u64,
        scan_id: String,
    ) -> Self {
        let corners: [Vector<_>; 4] = corners.map(|x| x.into_vector(observer.frame()));
        let scan_id = scan_id.into();
        PyWiseCmos(fov::WiseCmos::from_corners(
            corners,
            observer.raw,
            frame_num,
            scan_id,
        ))
    }

    /// Position of the observer in this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// WISE Frame number.
    #[getter]
    pub fn frame_num(&self) -> u64 {
        self.0.frame_num
    }

    /// WISE Scan ID.
    #[getter]
    pub fn scan_id(&self) -> String {
        self.0.scan_id.to_string()
    }

    /// Corners of this FOV.
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "WiseCmos(pointing={}, observer={}, frame_num={}, scan_id={:?})",
            self.pointing().__repr__(),
            self.observer().__repr__(),
            self.frame_num(),
            self.scan_id()
        )
    }
}

#[pymethods]
impl PyGenericRectangle {
    #[new]
    /// Construct a Generic Rectangular FOV from a central vector, rotation, and observer state.
    pub fn new(
        pointing: VectorLike,
        rotation: f64,
        observer: PyState,
        lon_width: f64,
        lat_width: f64,
    ) -> Self {
        let pointing = pointing.into_vector(observer.frame());
        PyGenericRectangle(fov::GenericRectangle::new(
            pointing,
            rotation.to_radians(),
            lon_width.to_radians(),
            lat_width.to_radians(),
            observer.raw,
        ))
    }

    /// Construct a new Rectangle FOV from the corners.
    /// The corners must be provided in order, either clockwise or counter-clockwise.
    ///
    /// Parameters
    /// ----------
    /// corners :
    ///     4 Vectors which represent the corners of the FOV, these must be provided in
    ///     order.
    /// observer :
    ///     The observer as a State, the time and position of the observer.
    /// expand_angle :
    ///     Expand the specified corners by this angle (degrees). This will move the
    ///     corners away from the middle of the FoV by the specified angle.
    #[staticmethod]
    #[pyo3(signature=(corners, observer, expand_angle=0.0))]
    pub fn from_corners(corners: [VectorLike; 4], observer: PyState, expand_angle: f64) -> Self {
        let corners: [Vector<_>; 4] =
            corners.map(|x| x.into_vector(crate::frame::PyFrames::Equatorial));
        PyGenericRectangle(fov::GenericRectangle::from_corners(
            corners,
            observer.raw,
            expand_angle.to_radians(),
        ))
    }

    /// The observer State.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Direction that the observer is looking.
    /// Average center of the FOV.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// The longitudinal width of the FOV.
    #[getter]
    pub fn lon_width(&self) -> f64 {
        self.0.lon_width().to_degrees()
    }

    /// The Latitudinal width of the FOV.
    #[getter]
    pub fn lat_width(&self) -> f64 {
        self.0.lat_width().to_degrees()
    }

    /// Corners of this FOV.
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "GenericRectangle(pointing={}, observer={}, lon_width={}, lat_width={})",
            self.pointing().__repr__(),
            self.observer().__repr__(),
            self.lon_width(),
            self.lat_width(),
        )
    }
}

#[pymethods]
impl PyGenericCone {
    /// Construct a Generic Cone FOV from a central vector, angle, and observer state.
    #[new]
    pub fn new(pointing: VectorLike, angle: f64, observer: PyState) -> Self {
        let pointing = pointing.into_vector(observer.frame());
        PyGenericCone(fov::GenericCone::new(
            pointing,
            angle.to_radians(),
            observer.raw,
        ))
    }

    /// The observer State.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.patch.pointing().into()
    }

    /// The longitudinal width of the FOV.
    #[getter]
    pub fn angle(&self) -> f64 {
        self.0.angle().to_degrees()
    }

    fn __repr__(&self) -> String {
        format!(
            "GenericCone(pointing={}, angle={}, observer={})",
            self.pointing().__repr__(),
            self.angle(),
            self.observer().__repr__(),
        )
    }
}

#[pymethods]
impl PyOmniDirectional {
    #[new]
    #[allow(missing_docs)]
    pub fn new(observer: PyState) -> Self {
        PyOmniDirectional(fov::OmniDirectional::new(observer.raw))
    }

    /// The observer State.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    fn __repr__(&self) -> String {
        format!("OmniDirectional(observer={})", self.observer().__repr__(),)
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyNeosCmos {
    #[new]
    #[allow(missing_docs)]
    pub fn new(
        pointing: VectorLike,
        rotation: f64,
        observer: PyState,
        side_id: u16,
        stack_id: u8,
        quad_id: u8,
        loop_id: u8,
        subloop_id: u8,
        exposure_id: u8,
        cmos_id: u8,
        band: u8,
    ) -> Self {
        let pointing = pointing.into_vector(observer.frame());
        PyNeosCmos(fov::NeosCmos::new(
            pointing,
            rotation.to_radians(),
            observer.raw,
            side_id,
            stack_id,
            quad_id,
            loop_id,
            subloop_id,
            exposure_id,
            cmos_id,
            band,
        ))
    }

    /// The observer State.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn side_id(&self) -> u16 {
        self.0.side_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn stack_id(&self) -> u8 {
        self.0.stack_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn quad_id(&self) -> u8 {
        self.0.quad_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn loop_id(&self) -> u8 {
        self.0.loop_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn subloop_id(&self) -> u8 {
        self.0.subloop_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn exposure_id(&self) -> u8 {
        self.0.exposure_id
    }

    /// Chip ID number
    #[getter]
    pub fn cmos_id(&self) -> u8 {
        self.0.cmos_id
    }

    /// Band Number
    #[getter]
    pub fn band(&self) -> u8 {
        self.0.band
    }

    /// Rotation angle of the FOV in degrees.
    #[getter]
    pub fn rotation(&self) -> f64 {
        self.0.rotation.to_degrees()
    }

    /// Corners of this FOV
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "NeosCmos(pointing={}, rotation={}, observer={}, side_id={}, stack_id={}, quad_id={}, loop_id={}, subloop_id={}, exposure_id={}, cmos_id={}, band={})",
            self.pointing().__repr__(),
            self.rotation(),
            self.observer().__repr__(),
            self.side_id(),
            self.stack_id(),
            self.quad_id(),
            self.loop_id(),
            self.subloop_id(),
            self.exposure_id(),
            self.cmos_id(),
            self.band()
        )
    }
}
#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyNeosVisit {
    #[allow(missing_docs)]
    #[new]
    pub fn new(
        x_width: f64,
        y_width: f64,
        gap_angle: f64,
        pointing: VectorLike,
        rotation: f64,
        observer: PyState,
        side_id: u16,
        stack_id: u8,
        quad_id: u8,
        loop_id: u8,
        subloop_id: u8,
        exposure_id: u8,
        band: u8,
    ) -> Self {
        let pointing = pointing.into_vector(crate::frame::PyFrames::Equatorial);
        let observer = observer.raw;
        PyNeosVisit(fov::NeosVisit::from_pointing(
            x_width.to_radians(),
            y_width.to_radians(),
            gap_angle.to_radians(),
            pointing,
            rotation.to_radians(),
            observer,
            side_id,
            stack_id,
            quad_id,
            loop_id,
            subloop_id,
            exposure_id,
            band,
        ))
    }

    /// Observer State.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn side_id(&self) -> u16 {
        self.0.side_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn stack_id(&self) -> u8 {
        self.0.stack_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn quad_id(&self) -> u8 {
        self.0.quad_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn loop_id(&self) -> u8 {
        self.0.loop_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn subloop_id(&self) -> u8 {
        self.0.subloop_id
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn exposure_id(&self) -> u8 {
        self.0.exposure_id
    }

    /// Rotation angle of the FOV in degrees.
    #[getter]
    pub fn rotation(&self) -> f64 {
        self.0.rotation
    }

    #[allow(missing_docs)]
    pub fn __len__(&self) -> usize {
        4
    }

    /// Retrieve a specific CMOS FOV..
    pub fn __getitem__(&self, idx: usize) -> PyResult<PyNeosCmos> {
        if idx >= self.__len__() {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(""));
        }

        Ok(PyNeosCmos(match self.0.get_fov(idx) {
            fov::FOV::NeosCmos(fov) => fov,
            _ => unreachable!(),
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "NEOSVisit(pointing={}, rotation={}, observer={}, side_id={}, stack_id={}, quad_id={}, loop_id={}, subloop_id={}, exposure_id={})",
            self.pointing().__repr__(),
            self.rotation(),
            self.observer().__repr__(),
            self.side_id(),
            self.stack_id(),
            self.quad_id(),
            self.loop_id(),
            self.subloop_id(),
            self.exposure_id()
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyZtfCcdQuad {
    /// Construct a new ZTF CCD FOV from the corners.
    /// The corners must be provided in clockwise order.
    ///
    /// Parameters
    /// ----------
    /// corners :
    ///     4 Vectors which represent the corners of the FOV, these must be provided in clockwise order.
    /// observer :
    ///     Position of the observer as a State.
    /// field :
    ///     Field number of the survey.
    /// filefracday :
    ///     Which fraction of a day was this FOV captured.
    /// ccdid :
    ///     CCD ID describes which of the 16 CCDs this represents.
    /// filtercode :
    ///     Which filter was used for this exposure.
    /// imgtypecode :
    ///     Type code describing the data product of this field.
    /// qid :
    ///     Which quadrant of the CCD does this FOV represent.
    /// maglimit :
    ///     Effective magnitude limit of this exposure.
    /// fid :
    ///     The FID of this exposure.
    #[new]
    pub fn new(
        corners: [VectorLike; 4],
        observer: PyState,
        field: u32,
        filefracday: u64,
        ccdid: u8,
        filtercode: String,
        imgtypecode: String,
        qid: u8,
        maglimit: f64,
        fid: u64,
    ) -> Self {
        let corners = corners
            .into_iter()
            .map(|v| v.into_vector(observer.frame()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PyZtfCcdQuad(fov::ZtfCcdQuad::new(
            corners,
            observer.raw,
            field,
            filefracday,
            ccdid,
            filtercode.into(),
            imgtypecode.into(),
            qid,
            maglimit,
            fid,
        ))
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn field(&self) -> u32 {
        self.0.field
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn filefracday(&self) -> u64 {
        self.0.filefracday
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn ccdid(&self) -> u8 {
        self.0.ccdid
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn filtercode(&self) -> String {
        self.0.filtercode.to_string()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn imgtypecode(&self) -> String {
        self.0.imgtypecode.to_string()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn qid(&self) -> u8 {
        self.0.qid
    }

    /// Magnitude limit of this exposure.
    #[getter]
    pub fn maglimit(&self) -> f64 {
        self.0.maglimit
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn fid(&self) -> u64 {
        self.0.fid
    }

    /// Corners of this FOV
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ZtfCcdQuad(observer={}, filefracday={}, ccdid={}, filtercode={:?}, imgtypecode={:?}, qid={}, maglimit={}, fid={})",
            self.observer().__repr__(),
            self.filefracday(),
            self.ccdid(),
            self.filtercode(),
            self.imgtypecode(),
            self.qid(),
            self.maglimit(),
            self.fid()
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyZtfField {
    /// Representation of an entire ZTF Field, made up of up to 64 ZTF CCD FOVs.
    ///
    /// Parameters
    /// ----------
    /// ztf_ccd_fields :
    ///     List containing all of the CCD FOVs.
    ///     These must have matching metadata.
    #[new]
    pub fn new(ztf_ccd_fields: Vec<PyZtfCcdQuad>) -> Self {
        PyZtfField(fov::ZtfField::new(ztf_ccd_fields.into_iter().map(|x| x.0).collect()).unwrap())
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn field(&self) -> u32 {
        self.0.field
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn filtercode(&self) -> String {
        self.0.filtercode.to_string()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn imgtypecode(&self) -> String {
        self.0.imgtypecode.to_string()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn fid(&self) -> u64 {
        self.0.fid
    }

    /// Return all of the individual CCD quads present in this field.
    #[getter]
    pub fn ccd_quads(&self) -> Vec<PyZtfCcdQuad> {
        (0..self.0.n_patches())
            .map(|idx| {
                PyZtfCcdQuad(match self.0.get_fov(idx) {
                    fov::FOV::ZtfCcdQuad(fov) => fov,
                    _ => unreachable!(),
                })
            })
            .collect()
    }

    #[allow(missing_docs)]
    pub fn __len__(&self) -> usize {
        self.0.n_patches()
    }

    /// Retrieve a specific CCD FOV from the contained list of FOVs.
    pub fn __getitem__(&self, idx: usize) -> PyResult<PyZtfCcdQuad> {
        if idx >= self.__len__() {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(""));
        }

        Ok(PyZtfCcdQuad(match self.0.get_fov(idx) {
            fov::FOV::ZtfCcdQuad(fov) => fov,
            _ => unreachable!(),
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "ZtfField(ccd_quads=<{} frames>, observer={}, filtercode={:?}, imgtypecode={:?}, fid={})",
            self.0.n_patches(),
            self.observer().__repr__(),
            self.filtercode(),
            self.imgtypecode(),
            self.fid()
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyPtfCcd {
    /// Construct a new PTF CCD FOV from the corners.
    /// The corners must be provided in clockwise order.
    ///
    /// Parameters
    /// ----------
    /// corners :
    ///     4 Vectors which represent the corners of the FOV, these must be provided in clockwise order.
    /// observer :
    ///     Position of the observer as a State.
    /// field :
    ///     Field number of the survey.
    /// ccdid :
    ///     CCD ID describes which of the 16 CCDs this represents.
    /// filter :
    ///     Which filter was used for this exposure, 'G', 'R', 'HA656', 'HA663'.
    /// filename:
    ///     Filename of the associated image.
    /// info_bits:
    ///     Info bits of the frame.
    /// seeing :
    ///     Effective limit of this exposure.
    #[new]
    pub fn new(
        corners: [VectorLike; 4],
        observer: PyState,
        field: u32,
        ccdid: u8,
        filter: String,
        filename: String,
        info_bits: u32,
        seeing: f32,
    ) -> Self {
        let corners = corners
            .into_iter()
            .map(|v| v.into_vector(observer.frame()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PyPtfCcd(fov::PtfCcd::new(
            corners,
            observer.raw,
            field,
            ccdid,
            filter.parse().unwrap(),
            filename.into(),
            info_bits,
            seeing,
        ))
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn field(&self) -> u32 {
        self.0.field
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn ccdid(&self) -> u8 {
        self.0.ccdid
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn filter(&self) -> String {
        self.0.filter.to_string()
    }

    /// Magnitude limit of this exposure.
    #[getter]
    pub fn seeing(&self) -> f32 {
        self.0.seeing
    }

    /// Info Bits
    #[getter]
    pub fn info_bits(&self) -> u32 {
        self.0.info_bits
    }

    /// Filename of the frame
    #[getter]
    pub fn filename(&self) -> String {
        self.0.filename.clone().into_string()
    }

    /// Corners of this FOV
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PtfCcd(observer={}, ccdid={}, filter={:?}, field={:?}, ccdid={}, filename={}, info_bits={}, seeing={})",
            self.observer().__repr__(),
            self.ccdid(),
            self.filter(),
            self.field(),
            self.ccdid(),
            self.filename(),
            self.info_bits(),
            self.seeing(),
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PyPtfField {
    /// Representation of an entire PTF Field, made up of multiple CCDs.
    ///
    /// Parameters
    /// ----------
    /// ptf_ccd_fields :
    ///     List containing all of the CCD FOVs.
    ///     These must have matching metadata.
    #[new]
    pub fn new(ptf_ccd_fields: Vec<PyPtfCcd>) -> Self {
        PyPtfField(fov::PtfField::new(ptf_ccd_fields.into_iter().map(|x| x.0).collect()).unwrap())
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn field(&self) -> u32 {
        self.0.field
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn filter(&self) -> String {
        self.0.filter.to_string()
    }

    /// Return all of the individual CCD quads present in this field.
    #[getter]
    pub fn ccds(&self) -> Vec<PyPtfCcd> {
        (0..self.0.n_patches())
            .map(|idx| {
                PyPtfCcd(match self.0.get_fov(idx) {
                    fov::FOV::PtfCcd(fov) => fov,
                    _ => unreachable!(),
                })
            })
            .collect()
    }

    #[allow(missing_docs)]
    pub fn __len__(&self) -> usize {
        self.0.n_patches()
    }

    /// Retrieve a specific CCD FOV from the contained list of FOVs.
    pub fn __getitem__(&self, idx: usize) -> PyResult<PyPtfCcd> {
        if idx >= self.__len__() {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(""));
        }

        Ok(PyPtfCcd(match self.0.get_fov(idx) {
            fov::FOV::PtfCcd(fov) => fov,
            _ => unreachable!(),
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "PtfField(ccd_quads=<{} frames>, observer={}, filter={:?}, field={})",
            self.0.n_patches(),
            self.observer().__repr__(),
            self.filter(),
            self.field(),
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PySpherexCmos {
    /// Construct a new PTF CCD FOV from the corners.
    /// The corners must be provided in clockwise order.
    ///
    /// Parameters
    /// ----------
    /// corners :
    ///     4 Vectors which represent the corners of the FOV, these must be provided in clockwise order.
    /// observer :
    ///     Position of the observer as a State.
    /// uri:
    ///     Location of the frame in IRSA
    /// plane_id:
    ///     planeid in the spherex.plane database table in IRSA.
    #[new]
    pub fn new(corners: [VectorLike; 4], observer: PyState, uri: String, plane_id: String) -> Self {
        let corners = corners
            .into_iter()
            .map(|v| v.into_vector(observer.frame()))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        PySpherexCmos(fov::SpherexCmos::new(
            corners,
            observer.raw,
            uri.into_boxed_str(),
            plane_id.into_boxed_str(),
        ))
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn plane_id(&self) -> String {
        self.0.plane_id.clone().into()
    }

    /// Direction that the observer is looking.
    #[getter]
    pub fn pointing(&self) -> PyVector {
        self.0.pointing().unwrap().into()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn uri(&self) -> String {
        self.0.uri.clone().into()
    }

    /// Corners of this FOV
    #[getter]
    pub fn corners(&self) -> Vec<PyVector> {
        self.0
            .corners()
            .unwrap()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpherexCmos(observer={}, plane_id={}, uri={:?})",
            self.observer().__repr__(),
            self.plane_id(),
            self.uri(),
        )
    }
}

#[pymethods]
#[allow(clippy::too_many_arguments)]
impl PySpherexField {
    /// Representation of an entire PTF Field, made up of multiple CCDs.
    ///
    /// Parameters
    /// ----------
    /// ptf_ccd_fields :
    ///     List containing all of the CCD FOVs.
    ///     These must have matching metadata.
    /// obsid:
    ///     Observation GUID used in the IRSA tables.
    /// observation_id:
    ///     Also called `obs_id`, this is not the same as obsid. This is another
    ///     identified in the IRSA tables, there is a 1-to-1 matching between the two.
    #[new]
    pub fn new(ptf_ccd_fields: Vec<PySpherexCmos>, obsid: String, observation_id: String) -> Self {
        PySpherexField(
            fov::SpherexField::new(
                ptf_ccd_fields.into_iter().map(|x| x.0).collect(),
                obsid.into_boxed_str(),
                observation_id.into_boxed_str(),
            )
            .unwrap(),
        )
    }

    /// State of the observer for this FOV.
    #[getter]
    pub fn observer(&self) -> PyState {
        self.0.observer().clone().into()
    }

    /// JD of the observer location.
    #[getter]
    pub fn jd(&self) -> f64 {
        self.0.observer().jd
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn obsid(&self) -> String {
        self.0.obsid.to_string()
    }

    /// Metadata about where this FOV is in the Survey.
    #[getter]
    pub fn observation_id(&self) -> String {
        self.0.observationid.to_string()
    }

    /// Return all of the individual CMOS frames present in this field.
    #[getter]
    pub fn cmos(&self) -> Vec<PySpherexCmos> {
        (0..self.0.n_patches())
            .map(|idx| {
                PySpherexCmos(match self.0.get_fov(idx) {
                    fov::FOV::SpherexCmos(fov) => fov,
                    _ => unreachable!(),
                })
            })
            .collect()
    }

    #[allow(missing_docs)]
    pub fn __len__(&self) -> usize {
        self.0.n_patches()
    }

    /// Retrieve a specific CMOS FOV from the contained list of FOVs.
    pub fn __getitem__(&self, idx: usize) -> PyResult<PySpherexCmos> {
        if idx >= self.__len__() {
            return Err(PyErr::new::<exceptions::PyIndexError, _>(""));
        }

        Ok(PySpherexCmos(match self.0.get_fov(idx) {
            fov::FOV::SpherexCmos(fov) => fov,
            _ => unreachable!(),
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "PtfField(ccd_quads=<{} frames>, observer={}, obs_id={:?}, observation_id={})",
            self.0.n_patches(),
            self.observer().__repr__(),
            self.obsid(),
            self.observation_id(),
        )
    }
}
