//! Python wrapper for [`kete_core::state::DiffuseState`].

use super::{PyState, PyUncertainState};
use crate::nongrav::PyNonGravModel;
use crate::time::PyTime;
use kete_core::forces::NonGravMask;
use kete_core::forces::ParameterizedForce;
use kete_core::frames::Equatorial;
use kete_core::prelude::*;
use kete_spice::propagation::SpkNonGravs;
use kete_spice::propagation::{
    SplitConfig, mixture_sigma_point_divergence, propagate_diffuse_state_adaptive,
};
use kete_spice::spk::LOADED_SPK;
use nalgebra::Vector3;
use pyo3::prelude::*;

/// A weighted mixture of :class:`~kete.UncertainState` components,
/// representing a diffuse cloud of states.
///
/// All components share an epoch, a center, and a covariance dimension
/// ``(6 + Np)``. Components may carry different free-parameter values:
/// a dust cloud with K different beta values is K components with the
/// same `Dust` template but different `free_params[0]`.
#[pyclass(frozen, module = "kete", name = "DiffuseState", from_py_object)]
#[derive(Clone)]
pub struct PyDiffuseState {
    /// Underlying weighted mixture of states.
    pub mixture: DiffuseState,
    /// All-`None` parameter mask over the non-grav ParameterizedForce template.
    /// Free-parameter values are stored per-component in each component's
    /// `free_params`; the mask itself holds no frozen values.
    pub non_grav: Option<NonGravMask>,
}

impl std::fmt::Debug for PyDiffuseState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyDiffuseState")
            .field("mixture", &self.mixture)
            .field("non_grav_present", &self.non_grav.is_some())
            .finish()
    }
}

impl PyDiffuseState {
    fn build_forces<'a>(
        &self,
        spk: &'a kete_spice::spk::SpkCollection,
        include_extended: bool,
    ) -> SpkNonGravs<'a> {
        if let Some(ref ng) = self.non_grav {
            SpkNonGravs::with_non_grav_mask(spk, include_extended, ng.clone())
        } else {
            SpkNonGravs::gravity(spk, include_extended)
        }
    }

    /// Resolves the Sun against the barycenter for the propagation paths below.
    ///
    /// Elements are referred to the Sun while the force models are barycentric, so a state
    /// crosses between them at every epoch the propagation touches; the adaptive path
    /// reaches many intermediate epochs while splitting, so it needs something it can ask.
    fn sun_resolver(
        spk: &kete_spice::spk::SpkCollection,
    ) -> impl Fn(Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> + Sync + '_ {
        move |time| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        }
    }
}

#[pymethods]
impl PyDiffuseState {
    /// Wrap a single :class:`~kete.UncertainState` as a one-component
    /// mixture with weight ``1.0``.
    #[staticmethod]
    fn from_uncertain(state: PyUncertainState) -> Self {
        Self {
            mixture: DiffuseState::from_uncertain(state.state),
            non_grav: state.non_grav,
        }
    }

    /// Construct a mixture from explicit weights and components.
    ///
    /// All components must share the same NonGravModel template
    /// (variant + fixed coefficients); only their free-parameter values
    /// may differ.
    #[staticmethod]
    fn new(weights: Vec<f64>, components: Vec<PyUncertainState>) -> PyResult<Self> {
        if components.is_empty() {
            return Err(
                Error::ValueError("DiffuseState must have at least one component".into()).into(),
            );
        }
        // Verify all components share the same non_grav template.
        let first_ng = components[0].non_grav.clone();
        for (i, c) in components.iter().enumerate().skip(1) {
            let same = match (&first_ng, &c.non_grav) {
                (None, None) => true,
                (Some(a), Some(b)) => a.free_param_names() == b.free_param_names(),
                _ => false,
            };
            if !same {
                return Err(Error::ValueError(format!(
                    "component {i} non_grav variant does not match component 0"
                ))
                .into());
            }
        }
        let raw: Vec<UncertainState> = components.into_iter().map(|c| c.state).collect();
        Ok(Self {
            mixture: DiffuseState::new(weights, raw)?,
            non_grav: first_ng,
        })
    }

    /// Mixture weights (a copy of the underlying vector).
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.mixture.weights.clone()
    }

    /// Mixture components as a list of :class:`~kete.UncertainState`.
    #[getter]
    fn components(&self) -> Vec<PyUncertainState> {
        let template = self.non_grav.clone();
        (0..self.mixture.n_components())
            .filter_map(|i| self.mixture.component(i).ok())
            .map(|us| PyUncertainState {
                state: us,
                non_grav: template.clone(),
            })
            .collect()
    }

    /// Common epoch shared by all components.
    #[getter]
    fn epoch(&self) -> PyTime {
        self.mixture.epoch().jd.into()
    }

    /// Number of mixture components.
    #[getter]
    fn n_components(&self) -> usize {
        self.mixture.n_components()
    }

    /// Maximum ``max_unresolved_divergence`` across all components.
    ///
    /// This is the peak STM-linearization error recorded anywhere in
    /// the mixture's history.  See
    /// :attr:`~kete.UncertainState.max_unresolved_divergence` for the
    /// per-component metric definition.  Values above the adaptive
    /// ``split_threshold`` used during propagation indicate at least
    /// one component is under-resolved -- typically because the
    /// ``max_components`` budget was full or the chaos has driven the
    /// covariance past the linear regime.
    #[getter]
    fn max_unresolved_divergence(&self) -> f64 {
        self.mixture.max_unresolved_divergence()
    }

    /// Total weight of components whose ``max_unresolved_divergence``
    /// exceeds ``threshold``.
    ///
    /// Useful to ask "how much of the distribution is under-resolved
    /// past my tolerance?"  Pass the same ``split_threshold`` used at
    /// propagation time to get a probability-mass measure of
    /// under-resolution.
    fn unresolved_weight(&self, threshold: f64) -> f64 {
        self.mixture.unresolved_weight(threshold)
    }

    /// Number of free parameters per component.
    #[getter]
    fn n_params(&self) -> usize {
        self.mixture.n_params()
    }

    /// Total covariance dimension, ``6 + n_params``.
    #[getter]
    fn cov_dim(&self) -> usize {
        self.mixture.cov_dim()
    }

    /// Names of all parameters in the per-component covariance matrix, in
    /// row/column order.
    ///
    /// Always starts with ``["p", "f", "g", "h", "k", "L"]``, the modified equinoctial
    /// elements described on :attr:`kete.UncertainState.cov_matrix`, followed by any
    /// non-gravitational parameter names.  Identical for every component (all components
    /// share the same covariance layout).
    #[getter]
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["p", "f", "g", "h", "k", "L"]
            .iter()
            .map(|s| String::from(*s))
            .collect();
        if let Some(ref ng) = self.non_grav {
            names.extend(ng.free_param_names().into_iter().map(String::from));
        }
        names
    }

    /// Non-gravitational model template, or None.
    ///
    /// Parameter values are taken from the first component's `free_params`.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        let mask = self.non_grav.as_ref()?;
        let values = self.mixture.free_params();
        let full = mask.merge(values).ok()?;
        PyNonGravModel::from_force(&mask.inner, &full)
    }

    /// Draw random samples from the mixture distribution.
    #[pyo3(signature = (n_samples, seed=None))]
    fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> PyResult<(Vec<PyState>, Vec<Option<PyNonGravModel>>)> {
        let samples = self.mixture.sample::<Equatorial>(n_samples, seed)?;
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let mut states = Vec::with_capacity(n_samples);
        let mut non_gravs = Vec::with_capacity(n_samples);
        for (mut st, sampled_params) in samples {
            if st.center_id() != 10 {
                spk.try_change_center(&mut st, 10)?;
            }
            states.push(st.into());
            let ng = self.non_grav.as_ref().and_then(|mask| {
                let raw = if sampled_params.is_empty() {
                    self.mixture.free_params()
                } else {
                    sampled_params.as_slice()
                };
                let full = mask.merge(raw).ok()?;
                PyNonGravModel::from_force(&mask.inner, &full)
            });
            non_gravs.push(ng);
        }
        Ok((states, non_gravs))
    }

    /// Recursively K=3 split every component along its dominant covariance
    /// eigenvector, ``depth`` times.
    ///
    /// Each level multiplies the component count by 3 and recomputes the
    /// dominant eigenvector on each child's updated covariance, so
    /// successive levels target the next-most-uncertain direction.  Use
    /// before propagation as a static pre-split when the initial mixture
    /// is too coarse to capture later nonlinearity.
    ///
    /// Parameters
    /// ----------
    /// depth : int
    ///     Number of recursive split levels.  ``0`` returns a copy.
    #[pyo3(signature = (depth=1))]
    fn split_all(&self, depth: u32) -> PyResult<Self> {
        let mixture = self.mixture.split_all(depth)?;
        Ok(Self {
            mixture,
            non_grav: self.non_grav.clone(),
        })
    }

    /// Adaptively split nonlinear components, then propagate.
    ///
    /// ``split_threshold`` is a Mahalanobis-distance threshold in the
    /// propagated covariance (element coordinates, floored against
    /// near-null directions); see
    /// :attr:`~kete.UncertainState.max_unresolved_divergence` for the
    /// metric definition.  The default ``0.1`` is the density
    /// calibration: it keeps the represented probability density
    /// faithful, since probes off by a few tenths of a sigma already
    /// distort the distribution's shape.  Pass ``3.0``-``4.0`` when only
    /// the mean and covariance matter (the state-estimation calibration:
    /// 3.0 is ~90% containment for samples drawn from the predicted
    /// Gaussian), at far fewer components.
    #[pyo3(signature = (
        jd,
        split_threshold=0.1,
        max_components=1024,
        max_split_depth=10,
        n_axes=3,
        sigma_factor=1.0,
        position_spacing_au=Some(0.001),
        min_split_improvement=0.1,
        include_asteroids=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn propagate(
        &self,
        py: Python<'_>,
        jd: PyTime,
        split_threshold: f64,
        max_components: usize,
        max_split_depth: u32,
        n_axes: usize,
        sigma_factor: f64,
        position_spacing_au: Option<f64>,
        min_split_improvement: f64,
        include_asteroids: bool,
    ) -> PyResult<Self> {
        let cfg = SplitConfig {
            split_threshold,
            max_components,
            max_split_depth,
            n_axes,
            sigma_factor,
            position_spacing_au,
            min_split_improvement,
        };
        let target: Time<TDB> = jd.into();
        py.detach(|| {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            let forces = self.build_forces(&spk, include_asteroids);
            let propagated = propagate_diffuse_state_adaptive(
                &self.mixture,
                &forces,
                target,
                &cfg,
                &Self::sun_resolver(&spk),
            )?;
            Ok(Self {
                mixture: propagated,
                non_grav: self.non_grav.clone(),
            })
        })
    }

    /// Per-component sigma-point divergence between linear and nonlinear
    /// propagation to ``jd``.
    #[pyo3(signature = (
        jd,
        n_axes=3,
        sigma_factor=1.0,
        position_spacing_au=Some(0.001),
        include_asteroids=false,
    ))]
    fn sigma_point_divergence(
        &self,
        py: Python<'_>,
        jd: PyTime,
        n_axes: usize,
        sigma_factor: f64,
        position_spacing_au: Option<f64>,
        include_asteroids: bool,
    ) -> PyResult<Vec<f64>> {
        let target: Time<TDB> = jd.into();
        py.detach(|| {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            let forces = self.build_forces(&spk, include_asteroids);
            Ok(mixture_sigma_point_divergence(
                &self.mixture,
                &forces,
                target,
                n_axes,
                sigma_factor,
                position_spacing_au,
                &Self::sun_resolver(&spk),
            )?)
        })
    }

    /// Number of mixture components.
    fn __len__(&self) -> usize {
        self.mixture.n_components()
    }

    /// Indexed access: returns the ``(weight, UncertainState)`` pair at
    /// position ``idx``.  Supports negative indexing.
    ///
    /// Combined with ``__len__``, this also enables direct iteration::
    ///
    ///     for weight, component in diffuse_state:
    ///         ...
    fn __getitem__(&self, mut idx: isize) -> PyResult<(f64, PyUncertainState)> {
        let n = self.mixture.n_components() as isize;
        if idx < 0 {
            idx += n;
        }
        if idx < 0 || idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "DiffuseState index out of range",
            ));
        }
        let i = idx as usize;
        let component = PyUncertainState {
            state: self.mixture.component(i)?,
            non_grav: self.non_grav.clone(),
        };
        Ok((self.mixture.weights[i], component))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let max_div = self.max_unresolved_divergence();
        format!(
            "DiffuseState(n_components={}, cov_dim={}, epoch={:.6}, max_unresolved_divergence={:.4})",
            self.mixture.n_components(),
            self.mixture.cov_dim(),
            self.mixture.epoch().jd,
            max_div,
        )
    }
}
