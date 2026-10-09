// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

use kete_spice::frames::try_frame_at;
use kete_spice::prelude::LOADED_CK;
use pyo3::{PyResult, pyfunction};

use crate::{
    frame::PyFrames,
    spice::FrameLike,
    time::PyTime,
    vector::{PyVector, VectorLike},
};

/// Reset the contents of the CK shared memory to the default set of CK kernels.
#[pyfunction]
#[pyo3(name = "ck_reset")]
pub fn ck_reset_py() {
    LOADED_CK.write().unwrap().reset()
}

/// Convert a vector in an instrument frame to the equatorial frame.
///
/// The frame resolves through its chain of reference frames, such as a camera
/// on a spacecraft. Where loaded CK kernels overlap, the kernel loaded last
/// takes precedence. Pointing is not extrapolated. A time inside a gap between
/// the intervals of a kernel has no pointing, unless another loaded kernel
/// covers that time.
///
/// Parameters
/// ----------
/// instrument_id : int or str
///   SPICE frame ID or frame name of the instrument frame. A frame that no
///   loaded frames kernel defines is a CK frame with that ID.
/// jd : :class:`~kete.Time` or float
///   Time of the request. A float is a Julian date in the TDB scale.
/// vec : list of float
///   Vector in the instrument frame, with 3 components.
///
/// Returns
/// -------
/// tuple of (:class:`~kete.Time`, :class:`~kete.Vector`)
///   Time of the frame, and the vector in the equatorial frame. The time is
///   the time of the CK pointing for a CK frame, and ``jd`` otherwise.
///
/// Raises
/// ------
/// ValueError
///   If the frame or a frame in its chain of reference frames is unknown, or
///   of an unsupported class. Also if a CK frame in the chain has no pointing
///   at ``jd``, or its spacecraft clock is not loaded.
#[pyfunction]
#[pyo3(name = "instrument_frame_to_equatorial")]
pub fn ck_sc_frame_to_equatorial(
    instrument_id: FrameLike,
    jd: PyTime,
    vec: [f64; 3],
) -> PyResult<(PyTime, PyVector)> {
    // The frame comes back resolved through its chain of reference frames, such as a
    // camera mounted on its spacecraft.
    let frame = try_frame_at(instrument_id.frame_id()?, jd.0)?;
    let time = frame.time;
    let rot = frame.rotation_to_equatorial()?;
    let pos = rot.transform_vector(&vec.into());

    let vec = PyVector::new(pos.into(), PyFrames::Equatorial);

    Ok((time.into(), vec))
}

/// Convert a vector from the equatorial frame to an instrument frame.
///
/// The frame resolves through its chain of reference frames, such as a camera
/// on a spacecraft. Where loaded CK kernels overlap, the kernel loaded last
/// takes precedence. Pointing is not extrapolated. A time inside a gap between
/// the intervals of a kernel has no pointing, unless another loaded kernel
/// covers that time.
///
/// Parameters
/// ----------
/// instrument_id : int or str
///   SPICE frame ID or frame name of the instrument frame. A frame that no
///   loaded frames kernel defines is a CK frame with that ID.
/// jd : :class:`~kete.Time` or float
///   Time of the request. A float is a Julian date in the TDB scale.
/// vec : :class:`~kete.Vector` or list of float
///   Vector to convert. A list is taken to be in the equatorial frame. A
///   :class:`~kete.Vector` is converted from its own frame.
///
/// Returns
/// -------
/// tuple of (:class:`~kete.Time`, list of float)
///   Time of the frame, and the 3 components of the vector in the
///   instrument frame. The time is the time of the CK pointing for a CK frame,
///   and ``jd`` otherwise.
///
/// Raises
/// ------
/// ValueError
///   If the frame or a frame in its chain of reference frames is unknown, or
///   of an unsupported class. Also if a CK frame in the chain has no pointing
///   at ``jd``, or its spacecraft clock is not loaded.
#[pyfunction]
#[pyo3(name = "instrument_equatorial_to_frame")]
pub fn ck_sc_equatorial_to_frame(
    instrument_id: FrameLike,
    jd: PyTime,
    vec: VectorLike,
) -> PyResult<(PyTime, [f64; 3])> {
    let vec = vec.into_vector(PyFrames::Equatorial);
    // The frame comes back resolved through its chain of reference frames, such as a
    // camera mounted on its spacecraft.
    let frame = try_frame_at(instrument_id.frame_id()?, jd.0)?;
    let time = frame.time;
    let rot = frame.rotation_to_equatorial()?;
    let pos = rot.inverse_transform_vector(&vec.into());

    Ok((time.into(), pos.into()))
}
