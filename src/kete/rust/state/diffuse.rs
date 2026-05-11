//! Python wrapper for [`kete_core::state::DiffuseState`].
//!
//! Bridges between the Rust shape (per-component `free_params`, no
//! force template stored on the state) and the Python user-facing
//! shape (a single `non_grav` model template that applies to every
//! component, with per-sample parameters preserved through sampling
//! and propagation).

use super::{PyState, PyUncertainState};
use crate::nongrav::PyNonGravModel;
use crate::time::PyTime;
use kete_core::forces::{ForceSet, GravParams, NonGravMask, ParameterizedForce};
use kete_core::frames::{Equatorial, SSB};
use kete_core::prelude::*;
use kete_spice::propagation::Recenter;
use kete_spice::propagation::SpkNBody;
use kete_spice::propagation::{
    SplitConfig, mixture_sigma_point_divergence, propagate_diffuse_state_adaptive,
};
use kete_spice::spk::LOADED_SPK;
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
        planets: &'a [GravParams],
    ) -> ForceSet<'a, Equatorial, SSB> {
        let mut force_set: ForceSet<'_, Equatorial, SSB> =
            ForceSet::new().with(Box::new(SpkNBody::new(spk, planets)));
        if let Some(ref ng) = self.non_grav {
            force_set = force_set.with(Box::new(Recenter::<SSB, _>::new(spk, ng.clone())));
        }
        force_set
    }

    /// Convert all components from DynCenter to SSB-typed UncertainStates.
    fn components_ssb(
        &self,
        spk: &kete_spice::spk::SpkCollection,
    ) -> KeteResult<Vec<UncertainState<Equatorial, SSB>>> {
        self.mixture
            .components
            .iter()
            .map(|c| {
                let ssb_state = spk.try_to_ssb(c.state.clone())?;
                UncertainState::<Equatorial, SSB>::new(
                    ssb_state,
                    c.cov_matrix.clone(),
                    c.free_params.clone(),
                )
            })
            .collect()
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
        self.mixture
            .components
            .iter()
            .cloned()
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
    /// Always starts with ``["x", "y", "z", "vx", "vy", "vz"]``, followed
    /// by any non-gravitational parameter names.  Identical for every
    /// component (all components share the same covariance layout).
    #[getter]
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["x", "y", "z", "vx", "vy", "vz"]
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
        let values = self
            .mixture
            .components
            .first()
            .map(|c| c.free_params.as_slice())
            .unwrap_or(&[]);
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
        let samples = self.mixture.sample(n_samples, seed)?;
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
                    self.mixture
                        .components
                        .first()
                        .map(|c| c.free_params.as_slice())
                        .unwrap_or(&[])
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

    /// Drop components below ``min_weight`` and renormalize.
    ///
    /// After many adaptive propagation steps, outer sub-components
    /// accumulate at small weights and can be discarded without
    /// meaningfully changing the mixture.
    fn prune(&self, min_weight: f64) -> PyResult<Self> {
        let mut next = self.mixture.clone();
        next.prune(min_weight)?;
        Ok(Self {
            mixture: next,
            non_grav: self.non_grav.clone(),
        })
    }

    /// Propagate the mixture linearly to a target epoch.
    ///
    /// Each component is propagated via the variational integrator;
    /// the propagated mean and covariance update share a single STM
    /// per component.
    #[pyo3(signature = (jd, include_asteroids=false))]
    fn propagate(&self, jd: PyTime, include_asteroids: bool) -> PyResult<Self> {
        use kete_core::state::StateLike;

        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let target: Time<TDB> = jd.into();
        let components_ssb = self.components_ssb(&spk)?;
        let propagated_ssb: KeteResult<Vec<UncertainState<Equatorial, SSB>>> = if include_asteroids
        {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            components_ssb
                .into_iter()
                .map(|c| c.propagate_with(&forces, target))
                .collect()
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            components_ssb
                .into_iter()
                .map(|c| c.propagate_with(&forces, target))
                .collect()
        };
        // Convert back to DynCenter for the Python wrapper.
        let propagated_dyn: Vec<UncertainState> = propagated_ssb?
            .into_iter()
            .map(|c| {
                UncertainState::new(c.state.into(), c.cov_matrix, c.free_params)
                    .expect("dimension preserved")
            })
            .collect();
        let mixture = DiffuseState::new(self.mixture.weights.clone(), propagated_dyn)?;
        Ok(Self {
            mixture,
            non_grav: self.non_grav.clone(),
        })
    }

    /// Adaptively split nonlinear components, then propagate.
    #[pyo3(signature = (
        jd,
        split_threshold=0.05,
        max_components=64,
        max_split_depth=4,
        n_axes=3,
        sigma_factor=1.0,
        prune_threshold=0.0,
        include_asteroids=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn propagate_adaptive(
        &self,
        jd: PyTime,
        split_threshold: f64,
        max_components: usize,
        max_split_depth: u32,
        n_axes: usize,
        sigma_factor: f64,
        prune_threshold: f64,
        include_asteroids: bool,
    ) -> PyResult<Self> {
        let cfg = SplitConfig {
            split_threshold,
            max_components,
            max_split_depth,
            n_axes,
            sigma_factor,
            prune_threshold,
        };
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let target: Time<TDB> = jd.into();
        let components_ssb = self.components_ssb(&spk)?;
        let mixture_ssb =
            DiffuseState::<Equatorial, SSB>::new(self.mixture.weights.clone(), components_ssb)?;
        let propagated = if include_asteroids {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            propagate_diffuse_state_adaptive(&mixture_ssb, &forces, target, &cfg)?
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            propagate_diffuse_state_adaptive(&mixture_ssb, &forces, target, &cfg)?
        };
        let propagated_dyn: Vec<UncertainState> = propagated
            .components
            .into_iter()
            .map(|c| {
                UncertainState::new(c.state.into(), c.cov_matrix, c.free_params)
                    .expect("dimension preserved")
            })
            .collect();
        let mixture = DiffuseState::new(propagated.weights, propagated_dyn)?;
        Ok(Self {
            mixture,
            non_grav: self.non_grav.clone(),
        })
    }

    /// Per-component sigma-point divergence between linear and nonlinear
    /// propagation to ``jd``.
    #[pyo3(signature = (jd, n_axes=3, sigma_factor=1.0, include_asteroids=false))]
    fn sigma_point_divergence(
        &self,
        jd: PyTime,
        n_axes: usize,
        sigma_factor: f64,
        include_asteroids: bool,
    ) -> PyResult<Vec<f64>> {
        let spk = LOADED_SPK.try_read().map_err(Error::from)?;
        let target: Time<TDB> = jd.into();
        let components_ssb = self.components_ssb(&spk)?;
        let mixture_ssb =
            DiffuseState::<Equatorial, SSB>::new(self.mixture.weights.clone(), components_ssb)?;
        let result = if include_asteroids {
            let extended = GravParams::selected_masses();
            let forces = self.build_forces(&spk, &extended);
            mixture_sigma_point_divergence(&mixture_ssb, &forces, target, n_axes, sigma_factor)?
        } else {
            let planets = GravParams::planets();
            let forces = self.build_forces(&spk, &planets);
            mixture_sigma_point_divergence(&mixture_ssb, &forces, target, n_axes, sigma_factor)?
        };
        Ok(result)
    }

    /// Number of mixture components.
    fn __len__(&self) -> usize {
        self.mixture.n_components()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "DiffuseState(n_components={}, cov_dim={}, epoch={:.6})",
            self.mixture.n_components(),
            self.mixture.cov_dim(),
            self.mixture.epoch().jd,
        )
    }
}
