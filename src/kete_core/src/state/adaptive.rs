//! Adaptive variational propagation of [`DiffuseState`] mixtures with
//! sigma-point linearity diagnostics.
//!
//! Each component is propagated by the variational Radau-15 integrator,
//! producing both a propagated mean and a `(6 + Np) x (6 + Np)` augmented
//! state-transition matrix. The augmented STM updates the component
//! covariance:
//!
//! ```text
//! Phi_aug = [ Phi_xx (6x6)   Phi_xp (6xNp) ]
//!           [ 0    (Npx6)    I    (NpxNp)  ]
//! P_f = Phi_aug * P_0 * Phi_aug^T
//! ```
//!
//! The bottom block is `[0 | I]` because free parameters are inputs to
//! the dynamics, not propagated quantities.
//!
//! The sigma-point diagnostic measures how far the linear (STM-based)
//! propagation deviates from a fully nonlinear propagation along the
//! dominant covariance eigenvectors; large divergence flags components
//! that benefit from a Gaussian-mixture split before propagation.
//!
//! All entry points take a generic [`ParameterizedForce<Frame = Equatorial, Center = SSB>`]
//! and an SSB-centered state. Callers compose their own gravity +
//! perturbation `ForceSet` and convert any `DynCenter` states to SSB
//! before calling.

use crate::errors::Error;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SSB};
use crate::prelude::{KeteResult, State, UncertainState};
use crate::state::{
    DiffuseState, covariance_update, propagate_state, propagate_with_stm, split_for_propagation,
};
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rayon::prelude::*;

struct SigmaPoint {
    delta_initial: DVector<f64>,
    lin_pred: DVector<f64>,
}

struct PropagationStep {
    mean_final: State<Equatorial, SSB>,
    /// `6 x (6 + Np)` sensitivity matrix from variational integration.
    sens: DMatrix<f64>,
    /// Augmented `(6 + Np) x (6 + Np)` STM; bottom Np rows are `[0 | I_Np]`.
    augmented_stm: DMatrix<f64>,
    propagated: UncertainState<Equatorial, SSB>,
}

fn propagate_step<F>(
    component: &UncertainState<Equatorial, SSB>,
    forces: &F,
    jd: Time<TDB>,
) -> KeteResult<PropagationStep>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    let np = component.free_params.len();
    if forces.n_free_params() != np {
        return Err(Error::ValueError(format!(
            "forces.n_free_params() = {} does not match component.free_params.len() = {np}",
            forces.n_free_params()
        )));
    }
    let n = 6 + np;

    let (pos_f, vel_f, sens) = propagate_with_stm(
        forces,
        component.state.pos.into(),
        component.state.vel.into(),
        &component.free_params,
        component.state.epoch,
        jd,
    )?;

    let mean_final = State::<Equatorial, SSB> {
        desig: component.state.desig.clone(),
        epoch: jd,
        pos: pos_f.into(),
        vel: vel_f.into(),
        center: SSB,
    };

    let mut phi_aug = DMatrix::<f64>::zeros(n, n);
    phi_aug.view_mut((0, 0), (6, n)).copy_from(&sens);
    for i in 0..np {
        phi_aug[(6 + i, 6 + i)] = 1.0;
    }
    let new_cov = covariance_update(&sens, &component.cov_matrix);
    let propagated = UncertainState::<Equatorial, SSB>::new(
        mean_final.clone(),
        new_cov,
        component.free_params.clone(),
    )?;

    Ok(PropagationStep {
        mean_final,
        sens,
        augmented_stm: phi_aug,
        propagated,
    })
}

/// Result of [`propagate_with_diagnosis`]: the linearly-propagated
/// component plus its sigma-point divergence at the same target epoch.
///
/// Sharing one variational integration between the propagation and the
/// diagnosis is the entire point of this struct -- callers that need
/// both should never compute them separately.
#[derive(Debug, Clone)]
pub struct LinearityDiagnosis {
    /// Linearly propagated [`UncertainState`].
    pub propagated: UncertainState<Equatorial, SSB>,
    /// Maximum sigma-point divergence across the tested axes.
    pub divergence: f64,
    /// Augmented `(6 + Np) x (6 + Np)` state transition matrix from
    /// initial to final epoch. Top 6 rows are the sensitivity matrix
    /// returned by the variational integrator; bottom Np rows are
    /// `[0 | I_Np]`.
    pub augmented_stm: DMatrix<f64>,
}

/// Propagate a single [`UncertainState`] linearly *and* compute the
/// sigma-point divergence between the linear and nonlinear results,
/// sharing a single variational integration between the two.
///
/// `n_axes` is capped at the covariance dimension `(6 + Np)`.
/// Eigenvectors with non-positive eigenvalues are skipped; if every
/// selected axis is degenerate, `divergence` is reported as `0.0`.
///
/// # Errors
/// Returns an error if `n_axes == 0`, if `sigma_factor` is non-finite
/// or non-positive, or if integration fails.
pub fn propagate_with_diagnosis<F>(
    component: &UncertainState<Equatorial, SSB>,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
) -> KeteResult<LinearityDiagnosis>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    if n_axes == 0 {
        return Err(Error::ValueError("n_axes must be at least 1".into()));
    }
    if !sigma_factor.is_finite() || sigma_factor <= 0.0 {
        return Err(Error::ValueError(
            "sigma_factor must be finite and positive".into(),
        ));
    }

    let step = propagate_step(component, forces, jd)?;

    let np = component.free_params.len();
    let n_dim = 6 + np;
    let n_axes = n_axes.min(n_dim);

    let sym = SymmetricEigen::new(component.cov_matrix.clone());
    let mut order: Vec<usize> = (0..n_dim).collect();
    order.sort_by(|&a, &b| {
        sym.eigenvalues[b]
            .partial_cmp(&sym.eigenvalues[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let axes: Vec<usize> = order.into_iter().take(n_axes).collect();

    let mut points: Vec<SigmaPoint> = Vec::with_capacity(2 * axes.len());
    for &axis in &axes {
        let lambda = sym.eigenvalues[axis];
        if !lambda.is_finite() || lambda <= 0.0 {
            continue;
        }
        let scale = sigma_factor * lambda.sqrt();
        let delta = sym.eigenvectors.column(axis) * scale;
        let lin = &step.sens * &delta;
        for &sign in &[1.0_f64, -1.0] {
            points.push(SigmaPoint {
                delta_initial: sign * &delta,
                lin_pred: sign * &lin,
            });
        }
    }

    let divergence = if points.is_empty() {
        0.0
    } else {
        let divergences: KeteResult<Vec<f64>> = points
            .into_par_iter()
            .map(|p| {
                sigma_point_divergence_one(
                    &component.state,
                    &step.mean_final,
                    &component.free_params,
                    forces,
                    &p.delta_initial,
                    &p.lin_pred,
                    jd,
                )
            })
            .collect();
        divergences?.into_iter().fold(0.0_f64, f64::max)
    };

    Ok(LinearityDiagnosis {
        propagated: step.propagated,
        divergence,
        augmented_stm: step.augmented_stm,
    })
}

/// Sigma-point divergence: a relative measure of how far the linear
/// (STM-based) propagation deviates from the full nonlinear N-body
/// propagation along the dominant eigenvectors of the component's
/// covariance.
///
/// For each of the top `n_axes` eigenvectors of the covariance, the
/// mean state is perturbed by `+/- sigma_factor * sqrt(lambda) * v`,
/// propagated nonlinearly to `jd`, and compared against the linear
/// prediction `Phi * delta_x_0`. The returned value is the maximum
/// across all sampled points of
///
/// ```text
/// ||delta_x_full - delta_x_lin|| / ||delta_x_lin||
/// ```
///
/// A value below `~0.05` is generally taken as "linearity holds".
/// Values much larger than that suggest the component's covariance
/// has grown beyond the linear regime and would benefit from a
/// Gaussian-mixture split.
///
/// Thin wrapper around [`propagate_with_diagnosis`] that discards the
/// propagated state. Callers that also want the propagated state
/// should use [`propagate_with_diagnosis`] directly to avoid a second
/// STM call.
///
/// # Errors
/// Returns an error if `n_axes == 0`, if `sigma_factor` is non-finite
/// or non-positive, or if integration fails.
pub fn sigma_point_divergence<F>(
    component: &UncertainState<Equatorial, SSB>,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
) -> KeteResult<f64>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    propagate_with_diagnosis(component, forces, jd, n_axes, sigma_factor).map(|d| d.divergence)
}

#[allow(
    clippy::too_many_arguments,
    reason = "All inputs are needed at the perturbation site"
)]
fn sigma_point_divergence_one<F>(
    mean_initial: &State<Equatorial, SSB>,
    mean_final: &State<Equatorial, SSB>,
    base_params: &[f64],
    forces: &F,
    delta_initial: &DVector<f64>,
    lin_pred: &DVector<f64>,
    jd: Time<TDB>,
) -> KeteResult<f64>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    let np = base_params.len();
    let perturbed_pos = nalgebra::Vector3::new(
        mean_initial.pos[0] + delta_initial[0],
        mean_initial.pos[1] + delta_initial[1],
        mean_initial.pos[2] + delta_initial[2],
    );
    let perturbed_vel = nalgebra::Vector3::new(
        mean_initial.vel[0] + delta_initial[3],
        mean_initial.vel[1] + delta_initial[4],
        mean_initial.vel[2] + delta_initial[5],
    );
    let perturbed_params: Vec<f64> = (0..np)
        .map(|i| base_params[i] + delta_initial[6 + i])
        .collect();

    let (pos_f, vel_f) = propagate_state(
        forces,
        perturbed_pos,
        perturbed_vel,
        &perturbed_params,
        mean_initial.epoch,
        jd,
    )?;

    let mut nonlin_dev = DVector::<f64>::zeros(6);
    for i in 0..3 {
        nonlin_dev[i] = pos_f[i] - mean_final.pos[i];
        nonlin_dev[3 + i] = vel_f[i] - mean_final.vel[i];
    }

    let diff = &nonlin_dev - lin_pred;
    let scale = lin_pred.norm();
    if scale <= 0.0 {
        return Ok(0.0);
    }
    Ok(diff.norm() / scale)
}

/// Configuration for [`propagate_diffuse_state_adaptive`].
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Sigma-point divergence above which a component is split.
    /// `0.05` is a common starting value.
    pub split_threshold: f64,
    /// Hard cap on the number of components in the propagated mixture.
    /// Splitting stops once any further split would exceed this count.
    pub max_components: usize,
    /// Maximum recursive split depth applied to a single original
    /// component. Prevents pathological cases where a component
    /// remains nonlinear no matter how often it is split.
    pub max_split_depth: u32,
    /// Number of dominant covariance eigenvectors to test in the
    /// sigma-point divergence diagnostic.
    pub n_axes: usize,
    /// Sigma-factor at which the divergence diagnostic samples sigma
    /// points (`1.0` = 1-sigma surface).
    pub sigma_factor: f64,
    /// Components with weight below this threshold are pruned from the
    /// settled mixture before it is returned.  After deep splitting,
    /// outer sub-components can reach weights as small as
    /// `HUBER_K3_WEIGHTS[0]^max_split_depth`; pruning removes them
    /// without meaningfully changing the mixture.  Set to `0.0` to
    /// disable pruning.
    pub prune_threshold: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            split_threshold: 0.05,
            max_components: 1024,
            max_split_depth: 10,
            n_axes: 3,
            sigma_factor: 1.0,
            prune_threshold: 0.0,
        }
    }
}

/// Adaptively split nonlinear components, propagating the resulting
/// (now finer) mixture to `jd` in a single BFS pass.
///
/// For each component pulled from the work queue, one of three things
/// happens:
///
/// 1. If a hypothetical split would breach `max_components`, or the
///    component is already at `max_split_depth`, it is propagated
///    directly via the variational integrator. No diagnosis is run --
///    there is no point asking whether to split when we cannot.
/// 2. Otherwise the component is run through [`propagate_with_diagnosis`].
///    If the divergence is below `split_threshold`, the propagated
///    state is settled directly -- no second STM call required.
/// 3. If the divergence is above `split_threshold`, the propagated
///    state is discarded, the component is K=3 split, and the
///    sub-components are enqueued at `depth + 1`.
///
/// Total mixture weight is preserved by the split itself; per-component
/// linear approximation error is bounded by the threshold (subject to
/// the caps).
///
/// # Errors
/// Returns an error if any propagation, diagnosis, or split fails, or
/// if the final mixture fails its [`DiffuseState::new`] invariant check.
pub fn propagate_diffuse_state_adaptive<F>(
    diffuse: &DiffuseState<Equatorial, SSB>,
    forces: &F,
    jd: Time<TDB>,
    config: &SplitConfig,
) -> KeteResult<DiffuseState<Equatorial, SSB>>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    if !config.split_threshold.is_finite() || config.split_threshold < 0.0 {
        return Err(Error::ValueError(
            "split_threshold must be finite and non-negative".into(),
        ));
    }
    if config.max_components < diffuse.n_components() {
        return Err(Error::ValueError(format!(
            "max_components ({}) must be at least the input component count ({})",
            config.max_components,
            diffuse.n_components()
        )));
    }

    let mut queue: std::collections::VecDeque<(f64, UncertainState<Equatorial, SSB>, u32)> =
        diffuse
            .weights
            .iter()
            .zip(diffuse.components.iter())
            .map(|(w, c)| (*w, c.clone(), 0_u32))
            .collect();

    let mut settled: Vec<(f64, UncertainState<Equatorial, SSB>)> =
        Vec::with_capacity(diffuse.n_components());

    while let Some((w, c, depth)) = queue.pop_front() {
        let post_split_total = settled.len() + queue.len() + 3;
        let cap_blocks_split = post_split_total > config.max_components;
        let depth_blocks_split = depth >= config.max_split_depth;

        if cap_blocks_split || depth_blocks_split {
            let prop = propagate_step(&c, forces, jd)?.propagated;
            settled.push((w, prop));
            continue;
        }

        let diag = propagate_with_diagnosis(&c, forces, jd, config.n_axes, config.sigma_factor)?;
        if diag.divergence <= config.split_threshold {
            settled.push((w, diag.propagated));
            continue;
        }

        let parts = split_for_propagation(&c, &diag.propagated.cov_matrix, &diag.augmented_stm)?;
        for (w_sub, c_sub) in parts {
            queue.push_back((w * w_sub, c_sub, depth + 1));
        }
    }

    let (weights, components): (Vec<f64>, Vec<UncertainState<Equatorial, SSB>>) =
        settled.into_iter().unzip();
    let mut result = DiffuseState::new(weights, components)?;
    if config.prune_threshold > 0.0 {
        result.prune(config.prune_threshold)?;
    }
    Ok(result)
}

/// Per-component sigma-point divergence for every component of a
/// [`DiffuseState`].
///
/// Components are evaluated in parallel and the returned vector has
/// the same length and ordering as `mixture.components`. See
/// [`sigma_point_divergence`] for the metric definition.
///
/// # Errors
/// Returns the first error encountered across components.
pub fn mixture_sigma_point_divergence<F>(
    mixture: &DiffuseState<Equatorial, SSB>,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
) -> KeteResult<Vec<f64>>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB> + Sync,
{
    mixture
        .components
        .par_iter()
        .with_min_len(2)
        .map(|c| sigma_point_divergence(c, forces, jd, n_axes, sigma_factor))
        .collect()
}
