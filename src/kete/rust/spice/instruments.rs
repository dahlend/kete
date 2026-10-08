// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

use kete_core::errors::Error;
use kete_spice::instruments::instrument_fov_at;
use kete_spice::prelude::LOADED_TEXT_KERNELS;
use kete_spice::text::TextKernelValue;
use kete_spice::text::ik::FovShape;
use pyo3::{FromPyObject, IntoPyObject, PyResult, pyfunction};

use crate::fovs::AllowedFOV;
use crate::time::PyTime;

/// An instrument given from Python as a NAIF ID or a name.
#[derive(Debug, Clone, FromPyObject)]
pub enum InstrumentLike {
    /// NAIF ID of the instrument.
    Id(i32),

    /// Name of the instrument, from `NAIF_BODY_NAME` in a loaded text kernel.
    Name(String),
}

impl InstrumentLike {
    /// The NAIF ID of the instrument.
    ///
    /// # Errors
    /// `ValueError` if no loaded text kernel names the instrument, or the text
    /// kernels cannot be read.
    fn id(&self) -> PyResult<i32> {
        match self {
            Self::Id(id) => Ok(*id),
            Self::Name(name) => LOADED_TEXT_KERNELS
                .try_read()
                .map_err(|_| Error::LockFailed)?
                .body_id(name)
                .ok_or_else(|| {
                    Error::ValueError(format!("No loaded text kernel names {name}.")).into()
                }),
        }
    }
}

/// The value of a text kernel variable.
#[derive(Debug, IntoPyObject)]
pub enum KernelValue {
    /// Numbers.
    Numbers(Vec<f64>),

    /// Strings, and dates as text that starts with ``@``.
    Strings(Vec<String>),
}

/// The field of view of an instrument at a time, from the loaded kernels.
///
/// The instrument kernel gives the field of view in the frame of the
/// instrument. The frame is resolved through its chain of reference frames to
/// the equatorial frame. The observer is the spacecraft of the instrument, from
/// the loaded SPK files. It is the spacecraft of the CK frame the instrument
/// frame is fixed to. That is ``CK_<id>_SPK`` if it is set, else the CK ID
/// divided by 1000 for a CK ID of -1000 or less. Without such a CK frame, it is
/// the instrument ID divided by 1000. The pointing has no light time or
/// aberration correction.
///
/// Parameters
/// ----------
/// instrument : int or str
///   NAIF ID or name of the instrument, such as ``-226111`` or
///   ``"ROS_OSIRIS_NAC"``.
/// jd : :class:`~kete.Time` or float
///   Time of the field of view. A float is a Julian date in the TDB scale.
///
/// Returns
/// -------
/// :class:`~kete.fov.RectangleFOV`, :class:`~kete.fov.ConeFOV` or :class:`~kete.fov.PolygonFOV`
///   The field of view.
///
/// Raises
/// ------
/// ValueError
///   If the instrument or its field of view is not defined, or the field of
///   view is an ellipse. Also if a frame in the chain has no data at ``jd``, if
///   these rules give no spacecraft ID, or if the SPK files have no state of
///   the spacecraft at ``jd``.
#[pyfunction]
#[pyo3(name = "instrument_fov")]
pub fn instrument_fov_py(instrument: InstrumentLike, jd: PyTime) -> PyResult<AllowedFOV> {
    Ok(instrument_fov_at(instrument.id()?, jd.0)?.into())
}

/// The field of view definition of an instrument, in its own frame: the shape,
/// the frame name, the boresight, and the boundary vectors.
///
/// `kete.spice.instrument_fov_definition` wraps it.
#[pyfunction]
#[pyo3(name = "_instrument_fov_definition")]
// The tuple is the Python return value, so a named type would not simplify it.
#[allow(
    clippy::type_complexity,
    reason = "the tuple is the Python return value"
)]
pub fn instrument_fov_definition_py(
    instrument: InstrumentLike,
) -> PyResult<(String, String, [f64; 3], Vec<[f64; 3]>)> {
    let id = instrument.id()?;
    let fov = LOADED_TEXT_KERNELS
        .try_read()
        .map_err(|_| Error::LockFailed)?
        .instrument_fov(id)?;
    let shape = match fov.shape {
        FovShape::Circle => "CIRCLE",
        FovShape::Ellipse => "ELLIPSE",
        FovShape::Rectangle => "RECTANGLE",
        FovShape::Polygon => "POLYGON",
    };
    Ok((
        shape.into(),
        fov.frame_name,
        fov.boresight.into(),
        fov.bounds.into_iter().map(Into::into).collect(),
    ))
}

/// The value of a variable of the loaded text kernels.
///
/// Parameters
/// ----------
/// name : str
///   Name of the variable, such as ``"INS-226111_PIXEL_SIZE"``. The name is
///   matched exactly.
///
/// Returns
/// -------
/// list of float, list of str, or None
///   The numbers or strings of the variable, or None if no loaded text kernel
///   defines it. A date is a string that starts with ``@``.
#[pyfunction]
#[pyo3(name = "kernel_variable")]
pub fn kernel_variable_py(name: &str) -> PyResult<Option<KernelValue>> {
    let text = LOADED_TEXT_KERNELS
        .try_read()
        .map_err(|_| Error::LockFailed)?;
    Ok(text.vars().get(name).map(|value| match value {
        TextKernelValue::Numbers(v) => KernelValue::Numbers(v.clone()),
        TextKernelValue::Strings(v) => KernelValue::Strings(v.clone()),
        TextKernelValue::Dates(v) => {
            KernelValue::Strings(v.iter().map(|d| format!("@{d}")).collect())
        }
    }))
}
