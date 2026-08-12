//! Python wrapper for [`kete_core::state::DiffuseState`].

use super::{PyState, PyUncertainState};
use crate::nongrav::PyNonGravModel;
use crate::time::PyTime;
use kete_core::forces::NonGravMask;
use kete_core::forces::ParameterizedForce;
use kete_core::frames::Equatorial;
use kete_core::prelude::*;
use kete_core::state::{DEFAULT_STEP_DAYS, StepReport, Termination};
use kete_spice::propagation::SpkNonGravs;
use kete_spice::propagation::{SplitConfig, propagate_diffuse_state, step_diffuse_state};
use kete_spice::spk::LOADED_SPK;
use nalgebra::Vector3;
use pyo3::prelude::*;

/// What one leg of an adaptive march decided.
///
/// Returned alongside the mixture by :meth:`kete.DiffuseState.step` and
/// :meth:`kete.DiffuseState.propagate`.  Everything about the *state* a leg produced -
/// per-component ``eta``, the residual behind it - is on the mixture itself; this holds
/// only what the mixture cannot say.
#[pyclass(
    frozen,
    module = "kete",
    name = "StepReport",
    get_all,
    skip_from_py_object
)]
#[derive(Clone, Copy, Debug)]
pub struct PyStepReport {
    /// Why splitting stopped: ``"converged"``, ``"component_cap"`` or
    /// ``"no_split_direction"``.
    ///
    /// ``"converged"`` means every component finished the leg under ``split_threshold``.
    /// ``"component_cap"`` means a component was still over it and ``max_components``
    /// refused the split, so raising the budget would change the answer.
    /// ``"no_split_direction"`` means a component was still over it but its covariance
    /// carried no direction to split along, so no budget would have helped.
    pub termination: &'static str,

    /// How many components were given fresh probes on this leg.
    ///
    /// Seeding restarts a component's measurement from zero.  That is correct for a new
    /// mixture and for the children of a split; for a component rebuilt mid-march it is a
    /// silent loss of accumulated history, which is why it is counted rather than left to
    /// be inferred.
    pub seeded: usize,
}

impl From<StepReport> for PyStepReport {
    fn from(report: StepReport) -> Self {
        Self {
            termination: match report.termination {
                Termination::Converged => "converged",
                Termination::ComponentCap => "component_cap",
                Termination::NoSplitDirection => "no_split_direction",
            },
            seeded: report.seeded,
        }
    }
}

#[pymethods]
impl PyStepReport {
    fn __repr__(&self) -> String {
        format!(
            "StepReport(termination={:?}, seeded={})",
            self.termination, self.seeded
        )
    }
}

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
    /// Underlying weighted mixture, carrying its components, the model their
    /// free parameters belong to, and its perturber set.
    pub mixture: DiffuseState,
}

impl std::fmt::Debug for PyDiffuseState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyDiffuseState")
            .field("mixture", &self.mixture)
            .field(
                "non_grav_present",
                &self.mixture.components[0].non_grav.is_some(),
            )
            .field("include_asteroids", &self.mixture.include_asteroids)
            .finish()
    }
}

impl PyDiffuseState {
    /// The model this mixture's free parameters belong to.
    ///
    /// Every component carries it and `DiffuseState::new` checks they agree, so
    /// reading the first is reading all of them.
    fn mask(&self) -> Option<&NonGravMask> {
        self.mixture.components.first()?.non_grav.as_ref()
    }

    fn build_forces<'a>(&self, spk: &'a kete_spice::spk::SpkCollection) -> SpkNonGravs<'a> {
        if let Some(ng) = self.mask() {
            SpkNonGravs::with_non_grav_mask(spk, self.mixture.include_asteroids, ng.clone())
        } else {
            SpkNonGravs::gravity(spk, self.mixture.include_asteroids)
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
    ///
    /// Parameters
    /// ----------
    /// state : :class:`~kete.UncertainState`
    ///     The component to wrap.
    /// include_asteroids : bool
    ///     Whether the largest asteroids perturb this mixture.  Fixed here rather than
    ///     supplied per call, so that every leg of a march runs under one force model -
    ///     see :attr:`include_asteroids`.
    #[staticmethod]
    #[pyo3(signature = (state, include_asteroids=false))]
    fn from_uncertain(state: PyUncertainState, include_asteroids: bool) -> Self {
        let mut component = state.state;
        component.clear_probes();
        let mut mixture = DiffuseState::from_uncertain(component);
        mixture.include_asteroids = include_asteroids;
        Self { mixture }
    }

    /// Construct a mixture from explicit weights and components.
    ///
    /// All components must share the same NonGravModel template
    /// (variant + fixed coefficients); only their free-parameter values
    /// may differ.
    ///
    /// Parameters
    /// ----------
    /// weights : list[float]
    ///     Mixture weights, non-negative and summing to one.
    /// components : list[:class:`~kete.UncertainState`]
    ///     The components, sharing an epoch, center and covariance dimension.
    /// include_asteroids : bool
    ///     Whether the largest asteroids perturb this mixture - see
    ///     :attr:`include_asteroids`.
    #[staticmethod]
    #[pyo3(signature = (weights, components, include_asteroids=false))]
    fn new(
        weights: Vec<f64>,
        components: Vec<PyUncertainState>,
        include_asteroids: bool,
    ) -> PyResult<Self> {
        // The components must agree on a force model, which `DiffuseState::new`
        // checks, along with the weights and the shared epoch.
        //
        // Building a mixture is not continuing a march. The components can come from
        // anywhere - another mixture, another force model, a different arc - and the
        // probes they carry were integrated under whatever model produced them, which
        // this call is free to contradict. Dropping them costs a reseed on the next step,
        // reported through ``StepReport.seeded``; keeping them would measure the new
        // mixture's flow against the old one's, and say nothing about it.
        let raw: Vec<UncertainState> = components
            .into_iter()
            .map(|c| {
                let mut state = c.state;
                state.clear_probes();
                state
            })
            .collect();
        let mut mixture = DiffuseState::new(weights, raw)?;
        mixture.include_asteroids = include_asteroids;
        Ok(Self { mixture })
    }

    /// Whether the largest asteroids perturb this mixture.
    ///
    /// With :attr:`non_grav` this is the whole force model, fixed when the mixture is
    /// built.  Pinning it here rather than taking it per call is what lets a march be
    /// driven one leg at a time: the components carry probes integrated under this model,
    /// and a model that changed between legs would leave them measuring a flow that is no
    /// longer the one being propagated.
    #[getter]
    fn include_asteroids(&self) -> bool {
        self.mixture.include_asteroids
    }

    /// Mixture weights (a copy of the underlying vector).
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.mixture.weights.clone()
    }

    /// Mixture components as a list of :class:`~kete.UncertainState`.
    #[getter]
    fn components(&self) -> Vec<PyUncertainState> {
        (0..self.mixture.n_components())
            .filter_map(|i| self.mixture.component(i).ok())
            .map(|state| PyUncertainState { state })
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

    /// Largest nonlinearity any component is carrying, in sigma of the propagated position
    /// distribution that component is measured against.
    ///
    /// ``0.1`` means the linear model landed a tenth of a cloud width away from where the
    /// probes went.  The width is the component's own until it splits; a split child keeps
    /// measuring against the width its parent had, so that a split which removes curvature
    /// lowers the number rather than having the gain cancelled by the narrower covariance
    /// the split produced.  The value therefore means the same thing at every split depth,
    /// and a deep component reports how much its error matters to the mixture rather than
    /// how well it describes its own local density -
    /// :attr:`component_residual_meters` answers the second question at any depth.
    ///
    /// This is a statement about what the splitter did, not an
    /// error bound on the represented density: a value above the ``split_threshold`` used
    /// at propagation time means at least one component was still failing the test when
    /// the march stopped, and the :class:`StepReport` returned by the last step says why
    /// it stopped splitting.
    ///
    /// Read off the components themselves, so it describes the mixture in hand rather
    /// than the call that produced it.  ``None`` if any component has never been marched,
    /// which is a different statement from zero.
    #[getter]
    fn max_eta(&self) -> Option<f64> {
        self.mixture.max_eta()
    }

    /// Per-component nonlinearity at this epoch, in mixture order, or ``None`` if any
    /// component has never been marched.
    ///
    /// Each is measured against probes carried since that component last split, so it says
    /// how far the component is from linear now rather than what the last leg added.
    ///
    /// Publishing these is what lets a starved tail be seen.  Splitting yields weights
    /// ``w/6, 2w/3, w/6`` and the splitter serves components in order of
    /// ``weight * eta``, so a low-weight component can be left badly represented while
    /// the mixture as a whole looks resolved.  Work that cares about exactly that tail -
    /// impact probability is the motivating case - should read these, and build a
    /// :class:`DiffuseState` over the tail region and propagate it directly rather than
    /// expecting the ranking to reach it.
    #[getter]
    fn component_eta(&self) -> Option<Vec<f64>> {
        self.mixture.components.iter().map(|c| c.eta).collect()
    }

    /// Worst residual behind :attr:`max_eta`, as a cartesian position offset in
    /// meters, or ``None`` if any component has never been marched.
    ///
    /// This is what makes the metric's own resolution legible.  The propagator has an
    /// absolute position resolution of roughly a meter, so ``max_eta`` of ``0.003``
    /// standing on a residual of ``1.2`` m says the number is numerical noise rather than
    /// a curved flow - a covariance direction narrower than the propagator resolves -
    /// with no second run and no estimator in the library.
    #[getter]
    fn residual_meters(&self) -> Option<f64> {
        self.mixture.residual_meters()
    }

    /// Per-component residual in meters, in mixture order, or ``None`` if any component
    /// has never been marched.
    ///
    /// The partner of :attr:`component_eta`: the two are read together, since a large
    /// ``eta`` standing on a residual at the propagator's resolution is numerics rather
    /// than curvature.
    #[getter]
    fn component_residual_meters(&self) -> Option<Vec<f64>> {
        self.mixture
            .components
            .iter()
            .map(|c| c.residual_meters)
            .collect()
    }

    /// Total weight of components whose nonlinearity exceeds ``threshold``.
    ///
    /// Pass the ``split_threshold`` used at propagation time to read how much of the
    /// distribution finished under-resolved.  The component count says nothing about
    /// where the weight sits, which is what this answers.
    fn weight_above_eta(&self, threshold: f64) -> Option<f64> {
        self.mixture.weight_above_eta(threshold)
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
    /// elements described on :attr:`kete.UncertainState.cov_matrix`, with ``L`` in
    /// degrees, followed by any non-gravitational parameter names.  Identical for every component (all components
    /// share the same covariance layout).
    #[getter]
    fn param_names(&self) -> Vec<String> {
        let mut names: Vec<String> = ["p", "f", "g", "h", "k", "L"]
            .iter()
            .map(|s| String::from(*s))
            .collect();
        if let Some(ng) = self.mask() {
            names.extend(ng.free_param_names().into_iter().map(String::from));
        }
        names
    }

    /// Non-gravitational model template, or None.
    ///
    /// Parameter values are taken from the first component's `free_params`.
    #[getter]
    fn non_grav(&self) -> Option<PyNonGravModel> {
        let mask = self.mask()?;
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
            let ng = self.mask().and_then(|mask| {
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

    /// Propagate to ``jd``, splitting components where the flow stops being linear over
    /// their own covariance.
    ///
    /// Every component is probed along every direction its covariance carries, at twice
    /// that direction's own width.  The probes are integrated along with the component and
    /// never re-placed, and their departure from the linear prediction is read off in sigma
    /// of the propagated cloud, so what is measured is how far the component has drifted
    /// from linear since it last split rather than what one leg added.  Re-placing the
    /// probes each leg would hide nonlinearity that arrives gradually.
    ///
    /// The arc is cut into legs of ``step_days`` and marched.  A component that exceeds
    /// ``split_threshold`` is rolled back to the start of the leg, split three ways along
    /// its worst probe's direction, and the leg is redone with the children on fresh
    /// probes.  Checking at every leg boundary is what places a split near the time the
    /// flow actually stops being linear.
    ///
    /// This is :meth:`step` folded over that grid, and nothing more: the probes ride on
    /// the components, so driving the same legs by hand measures the same thing.
    ///
    /// Returns ``(mixture, report)``.  The mixture reports where the test was still
    /// failing through :attr:`max_eta`, :attr:`component_eta` and
    /// :attr:`residual_meters`; the :class:`StepReport` says why the final leg stopped
    /// splitting.  Those describe what the splitter did; none of them is an error bound on
    /// the represented density.
    ///
    /// Parameters
    /// ----------
    /// jd :
    ///     Target epoch.  May be earlier than the mixture's own epoch.
    /// split_threshold :
    ///     The accuracy/cost dial.  A component is split when a probe at twice its own
    ///     width lands more than this many sigma of a predicted position distribution
    ///     away from the linear model - see :attr:`max_eta` for which distribution, which
    ///     is what makes the setting mean the same thing at every split depth.
    ///     Tightening it splits earlier and
    ///     more often: the mixture tracks the true density more faithfully and the run
    ///     costs more components and more time.  That trade is the setting's entire
    ///     meaning.
    /// max_components :
    ///     Hard cap on the returned component count - the brake on what a tight
    ///     threshold may spend.  Splits are three-way, so powers of three avoid
    ///     truncating a cascade mid-generation.  When it binds, the returned report
    ///     says ``"component_cap"``.
    /// step_days :
    ///     Length of one leg of the time grid, in days.  Equal steps of time, so every
    ///     component reaches the same leg boundaries whatever its orbit.  This sets the
    ///     time resolution a split is placed at rather than the accuracy of the result -
    ///     composing the propagation across legs is exact, so subdividing an arc changes
    ///     neither the propagated covariance nor the reported nonlinearity, which means
    ///     ``split_threshold`` is the same demand at any step.  Deep encounters do not
    ///     need a shorter step: a split placed anywhere in the pre-encounter linear
    ///     window is equivalent, since the children are narrow enough to propagate
    ///     linearly to the encounter from any lead.  Lengthen it on arcs of many
    ///     millennia, where the cost is ``arc / step_days`` legs and nothing caps that
    ///     on your behalf.  The arc's remainder is taken as a shorter first leg, so
    ///     every leg after it is the step you asked for rather than the arc divided
    ///     into equal parts near that length.
    #[pyo3(signature = (
        jd,
        split_threshold=0.15,
        max_components=729,
        step_days=DEFAULT_STEP_DAYS,
    ))]
    fn propagate(
        &self,
        py: Python<'_>,
        jd: PyTime,
        split_threshold: f64,
        max_components: usize,
        step_days: f64,
    ) -> PyResult<(Self, PyStepReport)> {
        let cfg = SplitConfig {
            split_threshold,
            max_components,
        };
        let target: Time<TDB> = jd.into();
        py.detach(|| {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            let forces = self.build_forces(&spk);
            let (propagated, report) = propagate_diffuse_state(
                &self.mixture,
                &forces,
                target,
                &cfg,
                step_days,
                &Self::sun_resolver(&spk),
            )?;
            Ok((
                Self {
                    mixture: propagated,
                },
                report.into(),
            ))
        })
    }

    /// Advance one leg to ``jd``, splitting components where the flow stops being linear.
    ///
    /// The unit :meth:`propagate` is built from, exposed so a march can be driven by the
    /// caller::
    ///
    ///     for t in times:
    ///         mixture, report = mixture.step(t)
    ///
    /// Stepping by hand and propagating in one call are the same measurement.  Each
    /// component carries its own probes, so ``eta`` keeps accumulating across calls
    /// exactly as it does across the legs of a single call - it is the departure from
    /// linearity since that component last split either way.
    ///
    /// A component with no probes yet - a new mixture, the children of a split, or one
    /// rebuilt from its parts - is seeded here, which restarts its measurement.  The
    /// returned report counts those, so a march that lost its history says so.
    ///
    /// Returns ``(mixture, report)``.
    ///
    /// Parameters
    /// ----------
    /// jd :
    ///     Epoch to advance to.  May be earlier than the mixture's own epoch.
    /// split_threshold :
    ///     As :meth:`propagate`.  Passed per call rather than held, so changing it partway
    ///     through a march is visible where it happens.
    /// max_components :
    ///     As :meth:`propagate`.
    #[pyo3(signature = (jd, split_threshold=0.15, max_components=729))]
    fn step(
        &self,
        py: Python<'_>,
        jd: PyTime,
        split_threshold: f64,
        max_components: usize,
    ) -> PyResult<(Self, PyStepReport)> {
        let cfg = SplitConfig {
            split_threshold,
            max_components,
        };
        let target: Time<TDB> = jd.into();
        py.detach(|| {
            let spk = LOADED_SPK.try_read().map_err(Error::from)?;
            let forces = self.build_forces(&spk);
            let (stepped, report) = step_diffuse_state(
                &self.mixture,
                &forces,
                target,
                &cfg,
                &Self::sun_resolver(&spk),
            )?;
            Ok((Self { mixture: stepped }, report.into()))
        })
    }

    /// Save this mixture to a file.
    ///
    /// The file keeps the weights, every component with its covariance and free
    /// parameters, the non-gravitational model those parameters belong to,
    /// :attr:`include_asteroids`, and any probes the components were carrying. A
    /// mixture loaded back is the mixture that was saved, so a march can continue
    /// from it without restarting its ``eta``.
    ///
    /// Use :meth:`save_list` when saving more than one. A directory of
    /// single-mixture files costs a file and a header for each.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to write. The format is the gzipped kete binary format.
    fn save(&self, filename: String) -> PyResult<()> {
        self.mixture.save(filename)?;
        Ok(())
    }

    /// Load a single mixture from a file.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to read.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the file holds several mixtures, or a different type. Use
    ///     :meth:`load_list` for a file holding several.
    #[staticmethod]
    fn load(filename: String) -> PyResult<Self> {
        Ok(Self {
            mixture: DiffuseState::load(filename)?,
        })
    }

    /// Save many mixtures to one file.
    ///
    /// Parameters
    /// ----------
    /// mixtures :
    ///     Mixtures to save. They do not have to share an epoch or a model.
    /// filename :
    ///     Path to write.
    #[staticmethod]
    fn save_list(mixtures: Vec<Self>, filename: String) -> PyResult<()> {
        let mixtures: Vec<DiffuseState> = mixtures.into_iter().map(|m| m.mixture).collect();
        DiffuseState::save_vec(&mixtures, filename)?;
        Ok(())
    }

    /// Load many mixtures from a file.
    ///
    /// A file holding a single mixture reads back as a list of one.
    ///
    /// Parameters
    /// ----------
    /// filename :
    ///     Path to read.
    #[staticmethod]
    fn load_list(filename: String) -> PyResult<Vec<Self>> {
        Ok(DiffuseState::load_vec(filename)?
            .into_iter()
            .map(|mixture| Self { mixture })
            .collect())
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
        };
        Ok((self.mixture.weights[i], component))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match self.max_eta() {
            Some(eta) => format!(
                "DiffuseState(n_components={}, cov_dim={}, epoch={:.6}, max_eta={eta:.4})",
                self.mixture.n_components(),
                self.mixture.cov_dim(),
                self.mixture.epoch().jd,
            ),
            None => format!(
                "DiffuseState(n_components={}, cov_dim={}, epoch={:.6})",
                self.mixture.n_components(),
                self.mixture.cov_dim(),
                self.mixture.epoch().jd,
            ),
        }
    }
}
