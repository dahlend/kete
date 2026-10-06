// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! State Transition matrix computation
use kete_core::prelude::*;
use kete_core::propagation::compute_state_transition;
use kete_spice::ephemeris::SpiceEphemeris;
use kete_spice::spk::LOADED_SPK;
use nalgebra::DMatrix;
use pyo3::{PyResult, Python, pyfunction};

use crate::frame::PyFrames;
use crate::nongrav::PyNonGravModel;
use crate::state::PyState;
use crate::time::PyTime;

/// Compute the state transition and parameter sensitivity matrix of a state.
///
/// The propagation uses the Radau 15th-order integrator with N-body gravity.
/// The input state may use any center. The function moves the state to the
/// solar system barycenter for the integration. The final state has the center
/// and the frame of the input state.
///
/// Parameters
/// ----------
/// state : :class:`~kete.State`
///   State of a single object.
/// jd_end : :class:`~kete.Time` or float
///   Time of the final state. A float is a Julian Date in TDB.
/// include_asteroids : bool, optional
///   If ``True``, the force model includes the selected massive asteroids.
///   Default is ``False``.
/// non_grav : :class:`~kete.NonGravModel`, optional
///   Non-gravitational force model. Default is ``None``.
///
/// Returns
/// -------
/// tuple of (:class:`~kete.State`, list of list of float)
///   The final state and the sensitivity matrix. The matrix has 6 rows and
///   ``6 + N`` columns. ``N`` is the number of fittable parameters of
///   ``non_grav``, and is 0 without a model. Columns 0 to 5 are the state
///   transition matrix. Column ``6 + k`` is the derivative of the final state
///   with respect to parameter ``k``, in the parameter order of the model. The
///   function evaluates every parameter at its value in ``non_grav``, and a
///   NaN value as 0. The matrix is in the frame of the input state.
///
/// Raises
/// ------
/// ValueError
///   If an SPK query fails, if the integration does not converge, or if the
///   object impacts a massive body.
#[pyfunction]
#[pyo3(name = "compute_stm", signature = (state, jd_end, include_asteroids=false, non_grav=None))]
pub fn compute_stm_py(
    py: Python<'_>,
    state: PyState,
    jd_end: PyTime,
    include_asteroids: bool,
    non_grav: Option<PyNonGravModel>,
) -> PyResult<(PyState, Vec<Vec<f64>>)> {
    let center = state.center_id();
    let frame = state.frame;
    let raw_state = state.raw;

    // Re-center to SSB (center_id = 0) as required by the Radau integrator.
    // The input state may use any center; we convert, integrate, then convert back.
    let ssb_state = {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_to_ssb(raw_state)?
    };

    // Every parameter of the model gets a sensitivity column, evaluated at the
    // value stored in the model.
    let non_grav = non_grav.map(|ng| ng.to_fixed());
    let non_grav = non_grav
        .as_ref()
        .map(|ng| ng.fixed_values().map(|values| (ng.inner(), values)))
        .transpose()?;
    let jd = jd_end.into();

    let (final_state_ssb, sens) = py.detach(|| {
        let eph = SpiceEphemeris::loaded()?;
        compute_state_transition(&eph, &ssb_state, jd, include_asteroids, non_grav)
    })?;

    // Re-center back to original center
    let mut final_state: State<Equatorial> = final_state_ssb.into();
    {
        let spk = &LOADED_SPK.try_read().map_err(Error::from)?;
        spk.try_change_center(&mut final_state, center)?;
    }

    // The integration is in the Equatorial frame, and the caller expects the
    // frame of the input state. With R the block rotation from Equatorial, the
    // state columns become R Phi R^T and the parameter columns become R dx/dp.
    let rot = match frame {
        PyFrames::Equatorial => Equatorial::rotation_to_frame::<Equatorial>(),
        PyFrames::Ecliptic => Equatorial::rotation_to_frame::<Ecliptic>(),
        PyFrames::Galactic => Equatorial::rotation_to_frame::<Galactic>(),
        PyFrames::FK4 => Equatorial::rotation_to_frame::<FK4>(),
    };
    let mut block = DMatrix::<f64>::identity(6, 6);
    block.view_mut((0, 0), (3, 3)).copy_from(rot.matrix());
    block.view_mut((3, 3), (3, 3)).copy_from(rot.matrix());
    let mut sens = &block * sens;
    let state_cols = sens.columns(0, 6) * block.transpose();
    sens.columns_mut(0, 6).copy_from(&state_cols);

    // Convert DMatrix to Vec<Vec<f64>> for Python
    let nrows = sens.nrows();
    let ncols = sens.ncols();
    let mat: Vec<Vec<f64>> = (0..nrows)
        .map(|r| (0..ncols).map(|c| sens[(r, c)]).collect())
        .collect();

    let py_state: PyState = final_state.into();
    let py_state = py_state.change_frame(frame);
    Ok((py_state, mat))
}
