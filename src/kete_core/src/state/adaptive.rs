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
//! dominant covariance eigenvectors.  The deviation is reported as a
//! Mahalanobis distance in the propagated 6-D position+velocity
//! covariance -- the prediction error in sigma-equivalent units of the
//! predicted Gaussian.  Large divergence flags components that benefit
//! from a Gaussian-mixture split before propagation; see
//! [`SplitConfig::split_threshold`] for typical thresholds.
//!
//! All entry points take a generic [`ParameterizedForce<Frame = Equatorial, Center = SSB>`]
//! and an SSB-centered state. Callers compose their own gravity +
//! perturbation force model (typically via [`Sum`](crate::forces::Sum))
//! and convert any `DynCenter` states to SSB before calling.

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

/// Push a +/- pair of sigma points along `unit_eigvec` at `scale`.
fn push_sigma_pair(
    points: &mut Vec<SigmaPoint>,
    unit_eigvec: &DVector<f64>,
    scale: f64,
    sens: &DMatrix<f64>,
) {
    let delta = unit_eigvec * scale;
    let lin = sens * &delta;
    for &sign in &[1.0_f64, -1.0] {
        points.push(SigmaPoint {
            delta_initial: sign * &delta,
            lin_pred: sign * &lin,
        });
    }
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
    let mut propagated = UncertainState::<Equatorial, SSB>::new(
        mean_final.clone(),
        new_cov,
        component.free_params.clone(),
    )?;
    propagated.max_unresolved_divergence = component.max_unresolved_divergence;

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
    /// Maximum sigma-point Mahalanobis divergence across the tested
    /// axes.  See [`sigma_point_divergence`] for the metric definition.
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
    position_spacing_au: Option<f64>,
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
    if let Some(s) = position_spacing_au
        && (!s.is_finite() || s <= 0.0)
    {
        return Err(Error::ValueError(
            "position_spacing_au must be finite and positive".into(),
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

    // Each axis may contribute up to two probes (edge + interior), each
    // with two signs.
    //
    // - Edge probe: perturbs along the full eigenvector at sigma-shell
    //   scale.  Tests STM accuracy at the natural covariance extent.
    //
    // - Interior probe: perturbs *position only* by `position_spacing_au`
    //   along the position-projection of the eigenvector.  Tests whether
    //   a small spatial shift gives a linear response -- catches
    //   position-localized nonlinearity (planets near the mean,
    //   resonances) that the sigma-shell test misses when the covariance
    //   has grown large.  Velocity and parameters are left unchanged so
    //   the perturbed state stays physically meaningful.
    //
    //   The interior probe is only added when the edge probe's position
    //   perturbation along this axis already exceeds `position_spacing_au`
    //   -- otherwise the edge probe is naturally finer and the interior
    //   probe would only test at a coarser scale.
    let mut points: Vec<SigmaPoint> = Vec::with_capacity(4 * axes.len());
    for &axis in &axes {
        let lambda = sym.eigenvalues[axis];
        if !lambda.is_finite() || lambda <= 0.0 {
            continue;
        }
        let eigenvec = sym.eigenvectors.column(axis).clone_owned();
        let edge_scale = sigma_factor * lambda.sqrt();

        // Edge probe (sigma-shell, mixed pos/vel/params).
        push_sigma_pair(&mut points, &eigenvec, edge_scale, &step.sens);

        // Interior probe (pure position perturbation).
        if let Some(pos_cap_au) = position_spacing_au {
            let pos_part_norm = eigenvec.rows(0, 3).norm();
            let edge_position_extent = edge_scale * pos_part_norm;
            if pos_part_norm > 0.0 && edge_position_extent > pos_cap_au * 0.99 {
                // Build a pure-position unit vector in the (6+Np)-dim space.
                let mut pos_dir = DVector::<f64>::zeros(n_dim);
                pos_dir[0] = eigenvec[0] / pos_part_norm;
                pos_dir[1] = eigenvec[1] / pos_part_norm;
                pos_dir[2] = eigenvec[2] / pos_part_norm;
                push_sigma_pair(&mut points, &pos_dir, pos_cap_au, &step.sens);
            }
        }
    }

    // Compute regularized inverse of the 6x6 position+velocity block of
    // P_f once, before the parallel sigma-point loop.  Mahalanobis
    // divergence at each sigma point uses this single inverse.  The
    // +eps*I regularization handles near-singular covariances (e.g.
    // freshly-initialized states with tiny variance or anisotropic
    // covariances with one near-zero eigenvalue).
    let p_f_pv = step.propagated.cov_matrix.view((0, 0), (6, 6)).into_owned();
    let trace = (0..6).map(|i| p_f_pv[(i, i)]).sum::<f64>();
    let eps = (trace * 1e-12).max(1e-30);
    let mut p_f_reg = p_f_pv;
    for i in 0..6 {
        p_f_reg[(i, i)] += eps;
    }
    let inv_p_f_6 = p_f_reg.try_inverse().ok_or_else(|| {
        Error::ValueError(
            "propagate_with_diagnosis: regularized P_f position+velocity block is not invertible"
                .into(),
        )
    })?;

    // Per-sigma-point Mahalanobis divergences.  Integration failures on
    // a single sigma point (typically a perturbed state landing at a
    // gravitational singularity, producing NaN) are treated as
    // `f64::INFINITY` rather than propagating up -- the failure indicates
    // the component is genuinely unrepresentable at this scale and the
    // adaptive loop should split it (or, at depth cap, record it as
    // fully unresolved).  This preserves forward progress instead of
    // aborting the whole mixture's propagation.
    let divergence = if points.is_empty() {
        0.0
    } else {
        let divergences: Vec<f64> = points
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
                    &inv_p_f_6,
                )
                .unwrap_or(f64::INFINITY)
            })
            .collect();
        divergences.into_iter().fold(0.0_f64, f64::max)
    };

    Ok(LinearityDiagnosis {
        propagated: step.propagated,
        divergence,
        augmented_stm: step.augmented_stm,
    })
}

/// Sigma-point Mahalanobis divergence: how far the linear (STM-based)
/// propagation deviates from full nonlinear N-body propagation,
/// measured as a sigma-equivalent distance in the propagated
/// covariance.
///
/// For each of the top `n_axes` eigenvectors of the component
/// covariance, the mean state is perturbed by
/// `+/- sigma_factor * sqrt(lambda) * v`, propagated nonlinearly to
/// `jd`, and compared against the linear prediction `Phi * delta_x_0`.
/// The returned value is the maximum across all sampled points of
///
/// ```text
/// d = sqrt( (delta_full - delta_lin)^T * P_f^-1 * (delta_full - delta_lin) )
/// ```
///
/// where `P_f` is the 6x6 position+velocity block of the propagated
/// covariance.  `d` is the Mahalanobis distance of the prediction
/// error within the predicted Gaussian -- "how many sigma off is the
/// linear answer, relative to its own predicted uncertainty?"
///
/// For samples drawn from the predicted distribution, `d` follows a
/// chi distribution in 6 dimensions: `E[d] ~ 2.4`, 90% containment
/// near `d ~ 3.0`, 95% near `3.55`, 99% near `4.1`.  Typical splitting
/// thresholds sit in the `3.0 - 5.0` range -- "split when the linear
/// prediction puts the answer outside the predicted ellipsoid."
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
    position_spacing_au: Option<f64>,
) -> KeteResult<f64>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    propagate_with_diagnosis(
        component,
        forces,
        jd,
        n_axes,
        sigma_factor,
        position_spacing_au,
    )
    .map(|d| d.divergence)
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
    inv_p_f_6: &DMatrix<f64>,
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

    // Mahalanobis divergence: error normalized by the propagated
    // covariance, measured as a sigma-equivalent distance.
    let diff = &nonlin_dev - lin_pred;
    let mahal_sq = (diff.transpose() * inv_p_f_6 * &diff)[(0, 0)];
    Ok(mahal_sq.max(0.0).sqrt())
}

/// Configuration for [`propagate_diffuse_state_adaptive`].
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Mahalanobis-distance threshold above which a component is split.
    ///
    /// The diagnostic measures the linear (STM-based) prediction error
    /// as a sigma-equivalent distance in the propagated 6-D
    /// position+velocity covariance.  Values are roughly comparable to
    /// a 6-D chi distribution: E[d] ~ 2.4, 90% containment ~ 3.0, 95% ~
    /// 3.55.  A threshold of `3.0` splits whenever the linear prediction
    /// puts the answer outside the 90% containment ellipsoid of the
    /// predicted distribution.
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
    /// Spatial scale (AU) for an additional "interior" sigma-point
    /// probe.  In addition to the sigma-shell test, each dominant
    /// eigenvector with a non-trivial position component contributes a
    /// pure-position perturbation of this magnitude along the
    /// position-projection of the eigenvector.  Velocity and parameters
    /// are left unchanged.  This catches position-localized nonlinearity
    /// (planets near the mean, resonances) that the sigma-shell test
    /// misses when the covariance becomes large (e.g. a 10 AU spread
    /// containing Jupiter near the mean).  The probe is skipped on axes
    /// where the sigma-shell already produces a smaller position
    /// perturbation than this scale.  `None` disables the interior probe
    /// entirely.  Typical value: `0.001` AU (~150,000 km).
    pub position_spacing_au: Option<f64>,
    /// Target maximum arc length (in days) for a single state-space
    /// adaptive propagation step.
    ///
    /// Propagation requests with `|jd - t_initial| > target_arc_days`
    /// are recursively bisected in time: the mixture is propagated to
    /// the arc midpoint via a nested adaptive call, and the result is
    /// then propagated from the midpoint to the target.  Bisection
    /// terminates once each leaf arc fits within `target_arc_days`.
    ///
    /// Why: state-space adaptivity (component splitting) handles
    /// non-Gaussian shape at a fixed time, but its splits cannot
    /// reduce nonlinearity that accumulates over a long arc -- the
    /// linear approximation simply fails over enough time.  Time
    /// bisection caps the arc so each state-space adaptive call works
    /// in a regime where splitting can converge.
    ///
    /// Set to `f64::INFINITY` to disable time bisection (full arc in
    /// one shot, the pre-bisection behavior).  Typical value: `60.0`
    /// days (~2 months), short enough that state-space adaptivity
    /// reliably converges in a single round for typical solar-system
    /// dynamics.  Long propagations incur many leaf calls; each leaf
    /// is cheap because Radau cost scales superlinearly with arc.
    pub target_arc_days: f64,
    /// Minimum fractional reduction in divergence required from parent to
    /// child before a split is allowed to continue.  After a K=3 split,
    /// the child covariance is reduced by ~30% along the split axis; for
    /// a well-behaved orbit this produces a similar fractional reduction
    /// in divergence.  For chaotic orbits the divergence does not drop --
    /// the split does not help -- and further cascading wastes compute.
    ///
    /// A value of `0.1` means "force-settle if divergence did not drop by
    /// at least 10% from the parent that was split."  `0.0` disables the
    /// check (cascade continues until depth or budget cap).
    pub min_split_improvement: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            split_threshold: 3.0,
            max_components: 1024,
            max_split_depth: 10,
            n_axes: 3,
            sigma_factor: 1.0,
            position_spacing_au: Some(0.001),
            target_arc_days: f64::INFINITY,
            min_split_improvement: 0.1,
        }
    }
}

/// Adaptively propagate a [`DiffuseState`] mixture to `jd`, splitting
/// in both state space (mixture components) and time (recursive arc
/// bisection) as needed to keep the linear approximation accurate.
///
/// **Time bisection (outer loop).**  If the requested arc
/// `|jd - mixture.epoch()|` exceeds `config.target_arc_days`, the call
/// recursively bisects: propagate to the arc midpoint, then from the
/// midpoint to `jd`.  Each leaf arc fits within `target_arc_days`, where
/// state-space adaptivity below can converge.
///
/// **State-space adaptivity (inner BFS loop).**  Within each leaf arc,
/// for each component pulled from the work queue one of three things
/// happens:
///
/// 1. If a hypothetical split would breach `max_components`, or the
///    component is already at `max_split_depth`, it is settled with
///    its propagated state. `max_unresolved_divergence` is updated
///    with the divergence at the abort point.
/// 2. Otherwise the component is run through [`propagate_with_diagnosis`].
///    If the divergence is below `split_threshold`, the propagated
///    state is settled directly -- no second STM call required.
/// 3. If the divergence is above `split_threshold`, the propagated
///    state is discarded, the component is K=3 split at the leaf-arc
///    initial epoch, and the sub-components are enqueued at
///    `depth + 1`.
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
    if config.target_arc_days.is_nan() || config.target_arc_days <= 0.0 {
        return Err(Error::ValueError(
            "target_arc_days must be positive (use f64::INFINITY to disable bisection)".into(),
        ));
    }
    // The round loop below only budget-checks newly created split children;
    // components that settle directly bypass it.  An input mixture already
    // over budget must fail fast here or the cap is silently violated.
    if config.max_components < diffuse.n_components() {
        return Err(Error::ValueError(format!(
            "max_components ({}) must be at least the input component count ({})",
            config.max_components,
            diffuse.n_components()
        )));
    }

    // Time bisection: if the requested arc is longer than `target_arc_days`,
    // recursively propagate to the arc midpoint, then from the midpoint to
    // the target.  State-space adaptive splitting (the rest of this
    // function) handles non-Gaussian shape at a fixed time; time bisection
    // handles nonlinearity that accumulates over long arcs, where
    // state-space splits alone cannot help.
    let arc_days = (jd.jd - diffuse.epoch().jd).abs();
    if arc_days > config.target_arc_days && config.target_arc_days.is_finite() {
        let t_mid = Time::<TDB>::new(0.5 * (diffuse.epoch().jd + jd.jd));
        let mid = propagate_diffuse_state_adaptive(diffuse, forces, t_mid, config)?;
        return propagate_diffuse_state_adaptive(&mid, forces, jd, config);
    }

    // Round-based parallel propagation.
    //
    // Each round (generation) holds all current candidates.  Each item
    // carries (weight, component, split_depth, parent_divergence) where
    // parent_divergence is the divergence that triggered the split
    // creating this component: `None` for the initial generation (no parent
    // split yet), `Some(d)` for a split child.  Using `Option` rather than an
    // `f64::INFINITY` sentinel keeps "no parent" distinct from "infinite
    // divergence" (a sigma point that could not be propagated); conflating the
    // two previously let an infinite child divergence satisfy the
    // diminishing-returns test and settle a first-generation component with
    // zero splits.  This lets us detect diminishing-returns splits: if a
    // child's divergence is close to (or exceeds) the parent's, further
    // splitting won't help.
    let mut generation: Vec<(f64, UncertainState<Equatorial, SSB>, u32, Option<f64>)> = diffuse
        .weights
        .iter()
        .zip(diffuse.components.iter())
        .map(|(w, c)| (*w, c.clone(), 0_u32, None))
        .collect();

    let mut settled: Vec<(f64, UncertainState<Equatorial, SSB>)> =
        Vec::with_capacity(diffuse.n_components());

    while !generation.is_empty() {
        enum GenOutcome {
            Settled(f64, UncertainState<Equatorial, SSB>),
            WantsSplit {
                weight: f64,
                /// Already-propagated parent at epoch `jd`.  Used to settle
                /// the component when the budget cannot accommodate the split.
                propagated: UncertainState<Equatorial, SSB>,
                parts: Vec<(f64, UncertainState<Equatorial, SSB>, u32, Option<f64>)>,
            },
        }

        let outcomes: KeteResult<Vec<GenOutcome>> = generation
            .into_par_iter()
            .with_min_len(4)
            .map(
                |(w, c, depth, parent_divergence)| -> KeteResult<GenOutcome> {
                    let diag = propagate_with_diagnosis(
                        &c,
                        forces,
                        jd,
                        config.n_axes,
                        config.sigma_factor,
                        config.position_spacing_au,
                    )?;
                    let mut prop = diag.propagated;
                    if diag.divergence > prop.max_unresolved_divergence {
                        prop.max_unresolved_divergence = diag.divergence;
                    }

                    // A non-finite divergence means a sigma point could not be
                    // propagated -- typically a perturbed state hitting a
                    // gravitational singularity during a deep encounter.
                    // Splitting cannot fix this: the children inherit the same
                    // singular geometry.  Settle and let max_unresolved_divergence
                    // (now infinite) flag the component as untrustworthy.  The
                    // Hill-sphere boundary check is meant to route these to a
                    // fallback before they reach here; this is the honest last
                    // resort if one slips through.
                    let unresolvable = !diag.divergence.is_finite();

                    // Diminishing-returns check: if this split produced less
                    // than `min_split_improvement` fractional reduction in
                    // divergence, further splitting won't help (chaos is the
                    // floor, not covariance size).  Force-settle immediately
                    // rather than cascading.  Only meaningful against a real,
                    // finite parent divergence; `None` marks the initial
                    // generation, which is always allowed its first split.
                    let no_improvement = match parent_divergence {
                        Some(pd) if config.min_split_improvement > 0.0 => {
                            diag.divergence >= pd * (1.0 - config.min_split_improvement)
                        }
                        _ => false,
                    };

                    if diag.divergence <= config.split_threshold
                        || depth >= config.max_split_depth
                        || unresolvable
                        || no_improvement
                    {
                        return Ok(GenOutcome::Settled(w, prop));
                    }

                    // `unresolvable` is false here, so `diag.divergence` is
                    // finite -- children carry a finite `Some(parent)`.
                    let parts = split_for_propagation(&c, &prop.cov_matrix, &diag.augmented_stm)?;
                    let child_divergence = diag.divergence;
                    let sub: Vec<_> = parts
                        .into_iter()
                        .map(|(ws, cs)| (w * ws, cs, depth + 1, Some(child_divergence)))
                        .collect();
                    Ok(GenOutcome::WantsSplit {
                        weight: w,
                        propagated: prop,
                        parts: sub,
                    })
                },
            )
            .collect();

        let mut next_gen: Vec<(f64, UncertainState<Equatorial, SSB>, u32, Option<f64>)> =
            Vec::new();
        let mut pending_splits: Vec<(
            f64,
            UncertainState<Equatorial, SSB>,
            Vec<(f64, UncertainState<Equatorial, SSB>, u32, Option<f64>)>,
        )> = Vec::new();
        for outcome in outcomes? {
            match outcome {
                GenOutcome::Settled(w, c) => settled.push((w, c)),
                GenOutcome::WantsSplit {
                    weight,
                    propagated,
                    parts,
                } => pending_splits.push((weight, propagated, parts)),
            }
        }

        for (weight, propagated, parts) in pending_splits {
            let added = parts.len().saturating_sub(1);
            let projected = settled.len() + next_gen.len() + added;
            if projected <= config.max_components {
                next_gen.extend(parts);
            } else {
                settled.push((weight, propagated));
            }
        }

        generation = next_gen;
    }

    let (weights, components): (Vec<f64>, Vec<UncertainState<Equatorial, SSB>>) =
        settled.into_iter().unzip();
    DiffuseState::new(weights, components)
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
    position_spacing_au: Option<f64>,
) -> KeteResult<Vec<f64>>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB> + Sync,
{
    mixture
        .components
        .par_iter()
        .with_min_len(2)
        .map(|c| sigma_point_divergence(c, forces, jd, n_axes, sigma_factor, position_spacing_au))
        .collect()
}
