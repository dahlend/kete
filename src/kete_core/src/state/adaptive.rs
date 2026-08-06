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
//! Mahalanobis distance in the propagated element covariance -- the
//! prediction error in sigma-equivalent units of the predicted
//! Gaussian, floored so near-null directions do not dominate.  Large divergence flags components that benefit
//! from a Gaussian-mixture split before propagation; see
//! [`SplitConfig::split_threshold`] for typical thresholds.
//!
//! All entry points take a generic [`ParameterizedForce<Frame = Equatorial, Center = SSB>`]
//! and an SSB-centered state. Callers compose their own gravity +
//! perturbation force model (typically via [`Sum`](crate::forces::Sum))
//! and convert any `DynCenter` states to SSB before calling.

use crate::elements::EquinoctialElements;
use crate::errors::Error;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SSB};
use crate::prelude::{KeteResult, State, UncertainState};
use crate::state::{
    DiffuseState, covariance_update, propagate_state, propagate_with_stm, split_for_propagation,
};
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, DVector, SymmetricEigen, Vector3, Vector6};
use rayon::prelude::*;

struct SigmaPoint {
    delta_initial: DVector<f64>,
    lin_pred: DVector<f64>,
    /// `true` for the fixed-scale interior (pure position) probes, `false` for the
    /// sigma-shell edge probes.  The two families scale differently under splitting -
    /// see [`LinearityDiagnosis::edge_divergence`] - so their maxima are tracked apart.
    interior: bool,
}

/// Resolves the elements' central body relative to the force model's center, at a time.
///
/// Element sets are referred to a gravitating body - the Sun for a minor planet - while the
/// force models here are barycentric, so a state has to cross between the two at every
/// epoch touched. The adaptive path propagates to many intermediate times while splitting,
/// so a pair of endpoint states will not do; it needs something it can ask.
///
/// This exists because `kete_core` has no ephemeris of its own. When the two centers
/// coincide, the resolver returns zeros.
pub type CenterResolver<'a> =
    &'a (dyn Fn(Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> + Sync);

/// Push a +/- pair of sigma points along `unit_eigvec` at `scale`.
fn push_sigma_pair(
    points: &mut Vec<SigmaPoint>,
    unit_eigvec: &DVector<f64>,
    scale: f64,
    sens: &DMatrix<f64>,
    interior: bool,
) {
    let delta = unit_eigvec * scale;
    let lin = sens * &delta;
    for &sign in &[1.0_f64, -1.0] {
        points.push(SigmaPoint {
            delta_initial: sign * &delta,
            lin_pred: sign * &lin,
            interior,
        });
    }
}

struct PropagationStep {
    /// Augmented `(6 + Np) x (6 + Np)` STM; bottom Np rows are `[0 | I_Np]`.
    augmented_stm: DMatrix<f64>,
    propagated: UncertainState,
}

fn propagate_step<F>(
    component: &UncertainState,
    forces: &F,
    center_at: CenterResolver<'_>,
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

    // Elements are referred to their central body; the force model is barycentric. Cross
    // once on the way in and once on the way out, so the integration runs in the frame the
    // force model expects and the result comes back as elements.
    let epoch = component.elements.epoch;
    let start = component.state::<Equatorial>()?;
    let (offset_pos, offset_vel) = center_at(epoch)?;

    let (pos_f, vel_f, sens) = propagate_with_stm(
        forces,
        Vector3::from(start.pos) + offset_pos,
        Vector3::from(start.vel) + offset_vel,
        &component.free_params,
        epoch,
        jd,
    )?;

    let (final_offset_pos, final_offset_vel) = center_at(jd)?;
    let final_centered = State::<Equatorial>::new(
        component.elements.desig.clone(),
        jd,
        pos_f - final_offset_pos,
        vel_f - final_offset_vel,
        component.elements.center_id,
    );

    // The covariance lives in element coordinates, so the sensitivity has to be expressed
    // element-to-element rather than cartesian-to-cartesian:
    //
    //     Phi_E = J(t_f) . Phi_cart . K(t_0)
    //
    // with `K` the element-to-state Jacobian at the start and `J` its inverse at the end,
    // both requested in the frame the force model integrates in. The parameter columns are
    // untouched by both, since the force model's free parameters are not elements.
    let final_elements = EquinoctialElements::from_state(&final_centered.into_frame())?;
    let start_jacobian = component.elements.state_jacobian::<Equatorial>()?;
    let final_inverse = final_elements.state_jacobian_inverse::<Equatorial>()?;

    let mut sens_local = sens.clone();
    let state_block = sens.view((0, 0), (6, 6)) * start_jacobian;
    sens_local.view_mut((0, 0), (6, 6)).copy_from(&state_block);
    let sens_local = DMatrix::from_iterator(6, n, (final_inverse * sens_local).iter().copied());

    let mut phi_aug = DMatrix::<f64>::zeros(n, n);
    phi_aug.view_mut((0, 0), (6, n)).copy_from(&sens_local);
    for i in 0..np {
        phi_aug[(6 + i, 6 + i)] = 1.0;
    }
    let new_cov = covariance_update(&sens_local, &component.cov_matrix);
    let mut propagated =
        UncertainState::new(final_elements, new_cov, component.free_params.clone())?;
    propagated.max_unresolved_divergence = component.max_unresolved_divergence;

    Ok(PropagationStep {
        augmented_stm: phi_aug,
        propagated,
    })
}

/// Propagate an [`UncertainState`] and its covariance, without the sigma-point diagnosis.
///
/// The mean is propagated nonlinearly and the covariance through the element-coordinate state
/// transition matrix, `J(t_f) . Phi . K(t_0)`. Use [`propagate_with_diagnosis`] instead
/// when the linearity of that step also matters; this is the cheap path for callers that
/// only want the propagated state.
///
/// `center_at` resolves the elements' central body against the force model's center - see
/// [`CenterResolver`].
///
/// # Errors
/// Fails if the force model's free-parameter count disagrees with the state's, if the
/// elements leave their physical domain, or if the integration fails.
pub fn propagate_uncertain<F>(
    component: &UncertainState,
    forces: &F,
    jd: Time<TDB>,
    center_at: CenterResolver<'_>,
) -> KeteResult<UncertainState>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    Ok(propagate_step(component, forces, center_at, jd)?.propagated)
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
    pub propagated: UncertainState,
    /// Maximum sigma-point Mahalanobis divergence across the tested
    /// axes, edge and interior probes together.  See
    /// [`sigma_point_divergence`] for the metric definition.
    pub divergence: f64,
    /// Maximum divergence across the sigma-shell (edge) probes alone,
    /// excluding the fixed-scale interior probes.
    ///
    /// The two families scale differently under splitting.  An edge
    /// probe rides the covariance: splitting narrows the component and
    /// the probe's residual falls with it.  An interior probe sits at a
    /// fixed spatial scale while the normalizing covariance narrows, so
    /// its divergence *grows* under splitting until the component is
    /// narrower than the probe scale and the probe switches off.  A
    /// diminishing-returns comparison across generations is therefore
    /// only meaningful within the edge family; this field is what the
    /// adaptive loop's `min_split_improvement` check reads.
    pub edge_divergence: f64,
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
    component: &UncertainState,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
    position_spacing_au: Option<f64>,
    center_at: CenterResolver<'_>,
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

    let step = propagate_step(component, forces, center_at, jd)?;

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
    // - Interior probe: a pure spatial displacement of `position_spacing_au`
    //   along the eigenvector's position image.  Tests whether a small
    //   spatial shift gives a linear response -- catches position-localized
    //   nonlinearity (planets near the mean, resonances) that the
    //   sigma-shell test misses when the covariance has grown large.
    //
    //   The eigenvector lives in element coordinates, where no subset of
    //   rows is a position, so the probe crosses through the element-state
    //   Jacobian: the eigenvector's cartesian image selects the spatial
    //   direction, a `position_spacing_au` displacement along it (velocity
    //   untouched) maps back through the inverse Jacobian to the element
    //   displacement actually applied.
    //
    //   The interior probe is only added when the edge probe's spatial
    //   extent already exceeds `position_spacing_au` -- otherwise the edge
    //   probe is naturally finer and the interior probe would only test at
    //   a coarser scale.
    let elem_to_state = component.elements.state_jacobian::<Equatorial>()?;
    let state_to_elem = component.elements.state_jacobian_inverse::<Equatorial>()?;
    let mut points: Vec<SigmaPoint> = Vec::with_capacity(4 * axes.len());
    for &axis in &axes {
        let lambda = sym.eigenvalues[axis];
        if !lambda.is_finite() || lambda <= 0.0 {
            continue;
        }
        let eigenvec = sym.eigenvectors.column(axis).clone_owned();
        let edge_scale = sigma_factor * lambda.sqrt();

        // Edge probe (sigma-shell, mixed element/params).
        push_sigma_pair(
            &mut points,
            &eigenvec,
            edge_scale,
            &step.augmented_stm,
            false,
        );

        // Interior probe (pure position perturbation).
        if let Some(pos_cap_au) = position_spacing_au {
            let elem_part = Vector6::from_iterator(eigenvec.iter().take(6).copied());
            let cart_image = elem_to_state * elem_part;
            let pos_norm = cart_image.rows(0, 3).norm();
            let edge_position_extent_au = edge_scale * pos_norm;
            if pos_norm > 0.0 && edge_position_extent_au > pos_cap_au * 0.99 {
                let mut cart_probe = Vector6::<f64>::zeros();
                for i in 0..3 {
                    cart_probe[i] = cart_image[i] / pos_norm * pos_cap_au;
                }
                let elem_probe = state_to_elem * cart_probe;
                let mut dir = DVector::<f64>::zeros(n_dim);
                for i in 0..6 {
                    dir[i] = elem_probe[i];
                }
                push_sigma_pair(&mut points, &dir, 1.0, &step.augmented_stm, true);
            }
        }
    }

    // Per-sigma-point Mahalanobis divergences.  Integration failures on
    // a single sigma point (typically a perturbed state landing at a
    // gravitational singularity, producing NaN) are treated as
    // `f64::INFINITY` rather than propagating up -- the failure indicates
    // the component is genuinely unrepresentable at this scale and the
    // adaptive loop should split it (or, at depth cap, record it as
    // fully unresolved).  This preserves forward progress instead of
    // aborting the whole mixture's propagation.
    let (divergence, edge_divergence) = if points.is_empty() {
        (0.0, 0.0)
    } else {
        // One regularized inverse of the full propagated element
        // covariance, shared by every sigma point.  See
        // `regularized_inverse` for the per-coordinate floor.
        let inv_p_f = regularized_inverse(&step.propagated.cov_matrix)?;
        let divergences: Vec<(f64, bool)> = points
            .into_par_iter()
            .map(|p| {
                let d = sigma_point_divergence_one(
                    &component.elements,
                    &step.propagated.elements,
                    &component.free_params,
                    forces,
                    center_at,
                    &p.delta_initial,
                    &p.lin_pred,
                    jd,
                    &inv_p_f,
                )
                .unwrap_or(f64::INFINITY);
                (d, p.interior)
            })
            .collect();
        let mut overall = 0.0_f64;
        let mut edge = 0.0_f64;
        for (d, interior) in divergences {
            overall = overall.max(d);
            if !interior {
                edge = edge.max(d);
            }
        }
        (overall, edge)
    };

    Ok(LinearityDiagnosis {
        propagated: step.propagated,
        divergence,
        edge_divergence,
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
/// where `P_f` is the full propagated covariance in element
/// coordinates, regularized so that directions narrower than about
/// 1e-3 of their own coordinates' marginal spread do not dominate.
/// `d` is the Mahalanobis distance of the prediction error within the
/// predicted Gaussian -- "how many sigma off is the linear answer,
/// relative to its own predicted uncertainty?"  The floor matters for
/// near-singular covariances (a delta-function release position, one
/// exquisitely determined element): without it the metric divides a
/// constant curvature remainder by a near-null width and reports
/// arbitrarily large values that splitting cannot reduce and that
/// correspond to no density error a wider direction could see.  The
/// floor is per coordinate because the element covariance mixes units;
/// see `regularized_inverse` for the construction.
///
/// For samples drawn from the predicted distribution, `d` follows a
/// chi distribution in 6 dimensions: `E[d] ~ 2.4`, 90% containment
/// near `d ~ 3.0`, 95% near `3.55`, 99% near `4.1`.  Two threshold
/// calibrations follow ([`SplitConfig::split_threshold`]): `0.1` keeps
/// the represented density faithful (probes off by tenths of a sigma
/// already distort the distribution's shape), while `3.0 - 5.0` only
/// keeps the answer inside the predicted ellipsoid, for uses where mean
/// and covariance are all that matter.
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
    component: &UncertainState,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
    position_spacing_au: Option<f64>,
    center_at: CenterResolver<'_>,
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
        center_at,
    )
    .map(|d| d.divergence)
}

#[allow(
    clippy::too_many_arguments,
    reason = "All inputs are needed at the perturbation site"
)]
fn sigma_point_divergence_one<F>(
    base: &EquinoctialElements,
    propagated_base: &EquinoctialElements,
    base_params: &[f64],
    forces: &F,
    center_at: CenterResolver<'_>,
    delta_initial: &DVector<f64>,
    lin_pred: &DVector<f64>,
    jd: Time<TDB>,
    inv_p_f: &DMatrix<f64>,
) -> KeteResult<f64>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    let np = base_params.len();

    // Everything here is in element coordinates, deliberately and throughout. The
    // perturbation displaces the stored elements rather than being added to a cartesian
    // state; the nonlinear answer is read back as an element offset from the propagated
    // base, with the true longitude reduced to the shortest signed angle; and the
    // Mahalanobis norm uses the element covariance. Mixing the two - an element offset
    // added to a cartesian state, or an element covariance normalizing a cartesian
    // residual - produces a divergence that means nothing, and drove the adaptive splitter
    // straight through its component cap on a short arc.
    let step = Vector6::from_iterator(delta_initial.iter().take(6).copied());
    let perturbed = base.displaced_by(&step);
    let start: State<Equatorial> = perturbed.try_to_state()?.into_frame();
    let (offset_pos, offset_vel) = center_at(base.epoch)?;

    let perturbed_params: Vec<f64> = (0..np)
        .map(|i| base_params[i] + delta_initial[6 + i])
        .collect();

    let (pos_f, vel_f) = propagate_state(
        forces,
        Vector3::from(start.pos) + offset_pos,
        Vector3::from(start.vel) + offset_vel,
        &perturbed_params,
        base.epoch,
        jd,
    )?;

    let (final_offset_pos, final_offset_vel) = center_at(jd)?;
    let final_state = State::<Equatorial>::new(
        base.desig.clone(),
        jd,
        pos_f - final_offset_pos,
        vel_f - final_offset_vel,
        base.center_id,
    );
    let final_elements = EquinoctialElements::from_state(&final_state.into_frame())?;

    let local = propagated_base.offset_to(&final_elements);
    let mut nonlin_dev = DVector::<f64>::zeros(6 + np);
    for i in 0..6 {
        nonlin_dev[i] = local[i];
    }
    for i in 0..np {
        nonlin_dev[6 + i] = perturbed_params[i] - base_params[i];
    }

    let diff = &nonlin_dev - lin_pred;
    let mahal_sq = (diff.transpose() * inv_p_f * &diff)[(0, 0)];
    Ok(mahal_sq.max(0.0).sqrt())
}

/// Regularized inverse of a covariance, the whitening metric of the sigma-point
/// divergence.
///
/// The floor exists because the metric divides by widths: without one, a near-null
/// direction - a delta-function release position, or one exquisitely determined
/// combination of elements - turns the probe's constant curvature remainder into an
/// arbitrarily large "sigma" count that splitting cannot reduce and that corresponds
/// to no density error a carrying direction could see.
///
/// The floor is per coordinate rather than a scalar ridge.  The element covariance
/// mixes units - AU, dimensionless shape and pole components, radians, and free
/// parameters - and its diagonal spans several decades for a fitted orbit, with the
/// along-track phase widest.  A scalar ridge scaled by the trace floors every
/// direction at a fraction of that widest spread, swallowing the well-determined
/// directions entirely.  Scaling to correlation form first puts the ridge on each
/// coordinate's own marginal scale:
///
/// ```text
/// D = diag(sqrt(P_ii))    C = D^-1 P D^-1
/// inv = D^-1 (C + eps I)^-1 D^-1        eps = 1e-6
/// ```
///
/// so directions narrower than about `1e-3` of their own coordinates' marginal spread
/// are floored, and everything wider is measured unchanged.  A near-null direction
/// formed by cancellation between strongly correlated coordinates is floored at that
/// same relative scale.  A coordinate with no extent at all (a zero diagonal entry)
/// is given a scale of `1e-6 * sqrt(trace)`, the only scale available for it.
///
/// # Errors
/// Fails if the trace is not positive and finite, or if the ridged correlation matrix
/// is not invertible.
fn regularized_inverse(cov: &DMatrix<f64>) -> KeteResult<DMatrix<f64>> {
    let n = cov.nrows();
    let trace = cov.trace();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(Error::ValueError(
            "regularized_inverse: covariance trace must be positive and finite".into(),
        ));
    }
    // Marginal scales.  The max() floors coordinates with no extent (and clips any
    // slightly negative diagonal rounding) so the whitening stays finite.
    let scales: Vec<f64> = cov
        .diagonal()
        .iter()
        .map(|&v| v.max(trace * 1e-12).sqrt())
        .collect();

    let mut corr = cov.clone();
    for r in 0..n {
        for c in 0..n {
            corr[(r, c)] /= scales[r] * scales[c];
        }
    }
    for i in 0..n {
        corr[(i, i)] += 1e-6;
    }

    let mut inv = corr.try_inverse().ok_or_else(|| {
        Error::ValueError(
            "regularized_inverse: the ridged correlation matrix is not invertible".into(),
        )
    })?;
    for r in 0..n {
        for c in 0..n {
            inv[(r, c)] /= scales[r] * scales[c];
        }
    }
    Ok(inv)
}

/// `log2(3)`, the exponent relating a required sigma reduction to the
/// component count it costs.  See [`minimum_components_for_divergence`].
const LOG2_3: f64 = 1.584_962_500_721_156;

/// Lower bound on the number of mixture components a K=3 splitting cascade
/// needs in order to bring `divergence` down to `threshold`.
///
/// Each split level replaces one component with three, each narrower than its
/// parent by `s = sqrt(1/2)` along the split axis.  If divergence scales with
/// component spread as `sigma^q`, one level reduces it by `r = s^-q = 2^(q/2)`,
/// so reaching the threshold costs `log(divergence / threshold) / log(r)`
/// levels and `3^levels` components.
///
/// A larger `q` means a faster reduction and therefore a *smaller* count, so
/// bounding the count from below requires bounding `q` from above.  This uses
/// `q = 2`, which assumes the dominant nonlinearity is no worse than cubic in
/// the perturbation.  Real trajectories reduce more slowly than that, on quiet
/// arcs and through planetary encounters alike, so the true requirement is
/// larger than this bound rather than smaller.  With `q = 2` the expression
/// reduces to `(divergence / threshold)^log2(3)`.
///
/// `splitting_tractability_boundary` in `kete_spice` exercises the reduction
/// this bound rests on.
///
/// The intended use is the give-up decision.  If the bound already exceeds
/// [`SplitConfig::max_components`] then no cascade within that budget can
/// converge, and the mixture should be abandoned in favor of sampling the
/// distribution directly.  The converse does not hold: a bound below the
/// budget is not a guarantee that splitting will converge.
///
/// Returns `1.0` when the divergence is already at or below the threshold,
/// and infinity if either input is NaN or `threshold` is not positive.  The
/// return is `f64` rather than a count because the bound is unbounded above
/// and its magnitude is the useful part of the answer.
#[must_use]
pub fn minimum_components_for_divergence(divergence: f64, threshold: f64) -> f64 {
    if divergence.is_nan() || threshold.is_nan() || threshold <= 0.0 {
        return f64::INFINITY;
    }
    if divergence <= threshold {
        return 1.0;
    }
    (divergence / threshold).powf(LOG2_3)
}

/// Configuration for [`propagate_diffuse_state_adaptive`].
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Mahalanobis-distance threshold above which a component is split.
    ///
    /// The diagnostic measures the linear (STM-based) prediction error
    /// as a sigma-equivalent distance in the propagated element
    /// covariance.  The threshold selects which question the split
    /// answers:
    ///
    /// - Density calibration, `0.1` (the default): keep the represented
    ///   probability density faithful.  A probe off by a few tenths of a
    ///   sigma already distorts the distribution's shape even though the
    ///   state estimate is fine, so density work needs the tight value.
    /// - State-estimation calibration, `3.0`-`4.0`: keep the propagated
    ///   state inside its predicted ellipsoid.  Values compare to a 6-D
    ///   chi distribution: `E[d] ~ 2.4`, 90% containment ~ 3.0, 95% ~
    ///   3.55.  Use when only the mean and covariance matter, at far
    ///   fewer components.
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
    /// Minimum fractional reduction in edge-probe divergence required
    /// from parent to child before a split is allowed to continue.  After
    /// a K=3 split, the child covariance is reduced by ~30% along the
    /// split axis; for a well-behaved orbit this produces a similar
    /// fractional reduction in divergence.  For chaotic orbits the
    /// divergence does not drop -- the split does not help -- and further
    /// cascading wastes compute.
    ///
    /// The comparison reads [`LinearityDiagnosis::edge_divergence`], not
    /// the overall maximum: interior probes sit at a fixed spatial scale
    /// and their divergence grows under splitting by construction, so
    /// including them would read a resolving component as a stalled one.
    /// The check is also skipped when the parent's edge divergence was
    /// itself below `split_threshold` (an interior-driven split), where
    /// the cascade is bounded by the interior probe's own gate, the depth
    /// cap and the budget instead.
    ///
    /// A value of `0.1` means "force-settle if the edge divergence did
    /// not drop by at least 10% from the parent that was split."  `0.0`
    /// disables the check (cascade continues until depth or budget cap).
    pub min_split_improvement: f64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            split_threshold: 0.1,
            max_components: 1024,
            max_split_depth: 10,
            n_axes: 3,
            sigma_factor: 1.0,
            position_spacing_au: Some(0.001),
            min_split_improvement: 0.1,
        }
    }
}

/// Adaptively propagate a [`DiffuseState`] mixture to `jd`, splitting
/// components in state space as needed to keep the linear approximation
/// accurate.
///
/// The full arc runs in one adaptive pass; for each component pulled
/// from the work queue one of three things happens:
///
/// 1. If a hypothetical split would breach `max_components`, or the
///    component is already at `max_split_depth`, it is settled with
///    its propagated state. `max_unresolved_divergence` is updated
///    with the divergence at the abort point.
/// 2. Otherwise the component is run through [`propagate_with_diagnosis`].
///    If the divergence is below `split_threshold`, the propagated
///    state is settled directly -- no second STM call required.
/// 3. If the divergence is above `split_threshold`, the propagated
///    state is discarded, the component is K=3 split at the initial
///    epoch, and the sub-components are enqueued at `depth + 1`.
///
/// Total mixture weight is preserved by the split itself; per-component
/// linear approximation error is bounded by the threshold (subject to
/// the caps).
///
/// # Errors
/// Returns an error if any propagation, diagnosis, or split fails, or
/// if the final mixture fails its [`DiffuseState::new`] invariant check.
pub fn propagate_diffuse_state_adaptive<F>(
    diffuse: &DiffuseState,
    forces: &F,
    jd: Time<TDB>,
    config: &SplitConfig,
    center_at: CenterResolver<'_>,
) -> KeteResult<DiffuseState>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    if !config.split_threshold.is_finite() || config.split_threshold < 0.0 {
        return Err(Error::ValueError(
            "split_threshold must be finite and non-negative".into(),
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

    // Round-based parallel propagation.
    //
    // Each round (generation) holds all current candidates.  Each item
    // carries (weight, component, split_depth, parent_edge_divergence)
    // where parent_edge_divergence is the *edge-probe* divergence of the
    // parent whose split created this component: `None` for the initial
    // generation (no parent split yet), `Some(d)` for a split child.
    // Using `Option` rather than an
    // `f64::INFINITY` sentinel keeps "no parent" distinct from "infinite
    // divergence" (a sigma point that could not be propagated).  Conflating the
    // two would let an infinite child divergence satisfy the diminishing-returns
    // test and settle a first-generation component with zero splits.  The
    // distinction is what detects diminishing-returns splits: if a
    // child's divergence is close to (or exceeds) the parent's, further
    // splitting won't help.
    //
    // The comparison is restricted to the edge family because interior
    // probes grow under splitting by construction (fixed spatial scale,
    // shrinking normalizer -- see [`LinearityDiagnosis::edge_divergence`]),
    // so including them would read resolving components as stalled ones.
    let components = (0..diffuse.n_components())
        .map(|i| diffuse.component(i))
        .collect::<KeteResult<Vec<_>>>()?;
    let mut generation: Vec<(f64, UncertainState, u32, Option<f64>)> = diffuse
        .weights
        .iter()
        .zip(components)
        .map(|(w, c)| (*w, c, 0_u32, None::<f64>))
        .collect();

    let mut settled: Vec<(f64, UncertainState)> = Vec::with_capacity(diffuse.n_components());

    while !generation.is_empty() {
        enum GenOutcome {
            Settled(f64, UncertainState),
            WantsSplit {
                weight: f64,
                /// Already-propagated parent at epoch `jd`.  Used to settle
                /// the component when the budget cannot accommodate the split.
                propagated: UncertainState,
                parts: Vec<(f64, UncertainState, u32, Option<f64>)>,
            },
        }

        let outcomes: KeteResult<Vec<GenOutcome>> = generation
            .into_par_iter()
            .with_min_len(4)
            .map(
                |(w, c, depth, parent_edge_divergence)| -> KeteResult<GenOutcome> {
                    let diag = propagate_with_diagnosis(
                        &c,
                        forces,
                        jd,
                        config.n_axes,
                        config.sigma_factor,
                        config.position_spacing_au,
                        center_at,
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
                    // edge divergence, further splitting won't help (chaos is
                    // the floor, not covariance size).  Force-settle
                    // immediately rather than cascading.  Only meaningful
                    // against a real, finite parent divergence; `None` marks
                    // the initial generation, which is always allowed its
                    // first split.
                    //
                    // Edge probes only, and only when the parent's own edge
                    // divergence was above the split threshold.  A parent
                    // split by its interior probe alone has an edge divergence
                    // with nothing to improve on, and its cascade is instead
                    // bounded by the interior probe's gate (the probe switches
                    // off once the component is narrower than the probe
                    // scale), the depth cap and the budget.
                    let no_improvement = match parent_edge_divergence {
                        Some(pd)
                            if config.min_split_improvement > 0.0
                                && pd > config.split_threshold =>
                        {
                            diag.edge_divergence >= pd * (1.0 - config.min_split_improvement)
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
                    // finite, and the edge maximum it bounds is too --
                    // children carry a finite `Some(parent_edge)`.
                    let parts = split_for_propagation(&c, &prop.cov_matrix, &diag.augmented_stm)?;
                    let child_edge_divergence = diag.edge_divergence;
                    let sub: Vec<_> = parts
                        .into_iter()
                        .map(|(ws, cs)| (w * ws, cs, depth + 1, Some(child_edge_divergence)))
                        .collect();
                    Ok(GenOutcome::WantsSplit {
                        weight: w,
                        propagated: prop,
                        parts: sub,
                    })
                },
            )
            .collect();

        let mut next_gen: Vec<(f64, UncertainState, u32, Option<f64>)> = Vec::new();
        let mut pending_splits: Vec<(
            f64,
            UncertainState,
            Vec<(f64, UncertainState, u32, Option<f64>)>,
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

        // Budget check. Two things have to be counted that the obvious version misses.
        //
        // The parent is already out of `generation` and lands in neither `settled` nor
        // `next_gen`, so a split adds all of `parts`, not `parts.len() - 1`.
        //
        // And the components still queued behind this one each contribute at least one
        // component whether they split or settle. Ignoring them let early splits in a round
        // consume the whole budget and the later ones overshoot it: with a cap of four, one
        // accepted three-way split plus two components that then had to settle gives five.
        // Unbounded, that overshoot compounds over the rounds.
        let queued = pending_splits.len();
        for (index, (weight, propagated, parts)) in pending_splits.into_iter().enumerate() {
            let still_queued = queued - index - 1;
            let projected = settled.len() + next_gen.len() + parts.len() + still_queued;
            if projected <= config.max_components {
                next_gen.extend(parts);
            } else {
                settled.push((weight, propagated));
            }
        }

        generation = next_gen;
    }

    let (weights, components): (Vec<f64>, Vec<UncertainState>) = settled.into_iter().unzip();
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
    mixture: &DiffuseState,
    forces: &F,
    jd: Time<TDB>,
    n_axes: usize,
    sigma_factor: f64,
    position_spacing_au: Option<f64>,
    center_at: CenterResolver<'_>,
) -> KeteResult<Vec<f64>>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB> + Sync,
{
    let components = (0..mixture.n_components())
        .map(|i| mixture.component(i))
        .collect::<KeteResult<Vec<_>>>()?;
    components
        .par_iter()
        .with_min_len(2)
        .map(|c| {
            sigma_point_divergence(
                c,
                forces,
                jd,
                n_axes,
                sigma_factor,
                position_spacing_au,
                center_at,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{minimum_components_for_divergence, regularized_inverse};
    use nalgebra::{DMatrix, DVector};

    /// A residual along a well-determined direction is measured against that
    /// direction's own width, not against a floor set by the widest coordinate.
    ///
    /// The diagonal here spans four decades, the range a fitted orbit covariance
    /// actually carries between its shape components and its along-track phase.  A
    /// scalar ridge scaled by the trace floors every direction at a fraction of the
    /// along-track spread and scores the tight-direction residual several times low.
    #[test]
    fn regularized_inverse_keeps_well_determined_directions() {
        let sigmas = [1e-8_f64, 1e-8, 1e-8, 1e-8, 1e-8, 1e-4];
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigmas.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let inv = regularized_inverse(&cov).unwrap();

        for (i, s) in sigmas.iter().enumerate() {
            let mut residual = DVector::<f64>::zeros(6);
            residual[i] = 2.0 * s;
            let mahal = (residual.transpose() * &inv * &residual)[(0, 0)].sqrt();
            assert!(
                (mahal - 2.0).abs() < 1e-3,
                "a 2-sigma residual along coordinate {i} scored {mahal}"
            );
        }
    }

    /// A direction with essentially no width - formed here by near-perfect
    /// correlation between two coordinates - is floored at about 1e-3 of the
    /// marginal scales rather than diverging.
    #[test]
    fn regularized_inverse_floors_near_null_directions() {
        let (s0, s1) = (1e-6_f64, 2e-6_f64);
        let rho = 1.0 - 1e-12;
        let mut cov = DMatrix::<f64>::zeros(2, 2);
        cov[(0, 0)] = s0 * s0;
        cov[(1, 1)] = s1 * s1;
        cov[(0, 1)] = rho * s0 * s1;
        cov[(1, 0)] = cov[(0, 1)];
        let inv = regularized_inverse(&cov).unwrap();

        // The anti-correlated direction carries a variance of ~1e-12 in correlation
        // form; unfloored, this residual would score ~1.4e3.  The floor holds it at
        // the residual's size relative to the 1e-3 marginal floor instead.
        let residual = DVector::from_column_slice(&[s0 * 1e-3, -s1 * 1e-3]);
        let mahal = (residual.transpose() * &inv * &residual)[(0, 0)].sqrt();
        assert!(
            mahal < 2.0,
            "the near-null direction was not floored: scored {mahal}"
        );
        assert!(
            mahal > 1.0,
            "the floor is wider than its own 1e-3 scale: scored {mahal}"
        );
    }

    /// For a well-conditioned covariance the regularization is a perturbation at the
    /// ridge's own 1e-6 level and the result is the plain inverse.
    #[test]
    fn regularized_inverse_matches_plain_inverse_when_well_conditioned() {
        let mut cov = DMatrix::<f64>::identity(3, 3) * 4.0;
        cov[(0, 1)] = 1.0;
        cov[(1, 0)] = 1.0;
        let inv = regularized_inverse(&cov).unwrap();
        let plain = cov.try_inverse().unwrap();
        let relative = (&inv - &plain).norm() / plain.norm();
        assert!(
            relative < 1e-5,
            "diverged from the plain inverse: {relative:e}"
        );
    }

    /// A covariance with no extent at all cannot define the metric.
    #[test]
    fn regularized_inverse_rejects_zero_covariance() {
        assert!(regularized_inverse(&DMatrix::<f64>::zeros(3, 3)).is_err());
    }

    #[test]
    fn bound_is_one_below_the_threshold() {
        assert!((minimum_components_for_divergence(1.0, 3.0) - 1.0).abs() < 1e-12);
        assert!((minimum_components_for_divergence(3.0, 3.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bound_grows_with_divergence() {
        // One K=3 level at the assumed q = 2 halves the divergence, so a
        // ratio of 2 must cost exactly one level, i.e. 3 components.
        assert!((minimum_components_for_divergence(6.0, 3.0) - 3.0).abs() < 1e-9);
        assert!((minimum_components_for_divergence(12.0, 3.0) - 9.0).abs() < 1e-9);
        assert!((minimum_components_for_divergence(24.0, 3.0) - 27.0).abs() < 1e-9);
    }

    #[test]
    fn bound_rejects_degenerate_input() {
        assert!(minimum_components_for_divergence(10.0, 0.0).is_infinite());
        assert!(minimum_components_for_divergence(10.0, -1.0).is_infinite());
        assert!(minimum_components_for_divergence(f64::NAN, 3.0).is_infinite());
        assert!(minimum_components_for_divergence(10.0, f64::NAN).is_infinite());
    }

    #[test]
    fn bound_is_below_the_measured_requirement() {
        // Divergence / component pairs measured by
        // `small_covariance_survives_the_encounter` in kete_spice at a
        // 0.003 AU encounter.  The bound must sit at or below every one.
        for (divergence, measured) in [
            (82.06, 729.0),
            (24.62, 81.0),
            (8.21, 9.0),
            (2.46, 1.0),
            (1.10, 1.0),
        ] {
            let bound = minimum_components_for_divergence(divergence, 3.0);
            assert!(
                bound <= measured,
                "bound {bound} exceeded the measured {measured} components"
            );
        }
    }
}
