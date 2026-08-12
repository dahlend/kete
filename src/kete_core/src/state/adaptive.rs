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
//! Nonlinearity is measured with probes under integration. Every direction the
//! component's covariance carries is probed at two times that direction's
//! width, the probes are carried forward leg by leg rather than re-placed, and their
//! departure from the linear prediction is read off in sigma of the position distribution
//! the component's whitening reference describes - its own until it splits, its parent's
//! afterwards, see [`UncertainState::whitening_cov`]. That miss decides whether the
//! component is split before a
//! leg is accepted. See [`step_diffuse_state`] for one leg of the controller, metric
//! included, and [`propagate_diffuse_state`] for the fold that marches it across a whole
//! arc.
//!
//! The probes ride on the component they measure (`UncertainState::probes`), so a caller
//! can take one leg at a time and get the same answer as a single call: what the probes
//! report is the departure accumulated since a component last split, whoever drove the
//! legs, and the reported number does not depend on how the arc was cut.
//!
//! All entry points take a generic [`ParameterizedForce<Frame = Equatorial, Center = SSB>`]
//! and an SSB-centered state. Callers compose their own gravity +
//! perturbation force model (typically via [`Sum`](crate::forces::Sum))
//! and convert any `DynCenter` states to SSB before calling.

use crate::elements::EquinoctialElements;
use crate::errors::Error;
use crate::forces::ParameterizedForce;
use crate::frames::{Ecliptic, Equatorial, InertialFrame, SSB};
use crate::prelude::{Desig, KeteResult, UncertainState};
use crate::state::{
    DiffuseState, ProbeSet, covariance_update, propagate_state, propagate_with_stm,
    split_axial_k3_along,
};
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, DVector, Matrix6, SymmetricEigen, Vector3, Vector6};
use rayon::prelude::*;

/// Absolute position resolution of the propagator, in AU (about one meter).
///
/// Two trajectories integrated to the same epoch by different step sequences differ by
/// roughly this much, so a residual at or below it carries no dynamical information.
/// It enters the metric only through [`POSITION_NOISE_FLOOR_AU`], the floor on the
/// whitening covariance, so an error in it changes which directions are treated as
/// resolved rather than any reported number directly.
const PROPAGATOR_RESOLUTION_AU: f64 = 1.0 / 1.495_978_707e11;

/// Meters per AU, for reporting a residual in physical units.
const M_PER_AU: f64 = 1.495_978_707e11;

/// Floor on the whitening position covariance used to scale the probe residual, in AU.
///
/// The residual is measured in sigma of a propagated position distribution. A direction of
/// that distribution narrower than the propagation can place two states
/// carries no measurable information, so the floor keeps such directions from dominating
/// the answer. Ten propagator resolutions covers the spread measured between propagation
/// paths taking different step sequences over multi-year arcs.
///
/// It bites less often on a component that has split, whose whitening reference is the
/// wider pre-split one - which is the intended direction: the floor exists to keep an
/// unmeasurable width out of the denominator, and a width the density actually had is not
/// one of those.
const POSITION_NOISE_FLOOR_AU: f64 = 10.0 * PROPAGATOR_RESOLUTION_AU;

/// Default leg length on the common time grid, in days.
///
/// A quarter of a year is short against the orbital periods kete is used on and long
/// against the cost of restarting the integrator, and it places splits to within a
/// season of where the flow stops being linear. It is a default rather than a constant
/// because an arc of many millennia wants a longer step or it does nothing but pay for
/// legs. Encounters do not need a shorter one: a split placed anywhere in the
/// pre-encounter linear window is equivalent, since the children are narrow enough to
/// propagate linearly to the encounter from any lead.
pub const DEFAULT_STEP_DAYS: f64 = 90.0;

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

/// Why a leg stopped splitting.
///
/// A caller has to be able to tell a resolved leg from one that ran out of budget, or from
/// one holding a component nothing could split, without re-running. Recorded by the step
/// that made the decision rather than inferred afterwards from the component `eta` values,
/// which cannot distinguish the last two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Every component finished the leg under `split_threshold`.
    Converged,

    /// A component was still over threshold and `max_components` refused the split.
    ComponentCap,

    /// A component was still over threshold and its covariance carried no direction to
    /// split along, so no split could have helped. Raising `max_components` will not
    /// change this leg.
    NoSplitDirection,
}

/// What one leg of the march decided.
///
/// Everything about the *state* the leg produced - per-component `eta`, the residual
/// behind it - lives on the components themselves. This holds only what the returned
/// mixture cannot say: the decisions the step made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepReport {
    /// Why splitting stopped on this leg.
    ///
    /// [`Termination::ComponentCap`] takes precedence over
    /// [`Termination::NoSplitDirection`] when both fired, since the budget is the one a
    /// caller can act on.
    pub termination: Termination,

    /// How many components were given fresh probes on this leg.
    ///
    /// A component with no carried probes is seeded, which restarts its measurement from
    /// zero. That is correct for a new mixture and for the children of a split, and it is
    /// a silent loss of history for a component rebuilt mid-march, so it is counted rather
    /// than left to be inferred.
    pub seeded: usize,
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

    // Back to the elements' own center and storage frame, encoded directly rather than
    // through a named intermediate `State`.
    let (final_offset_pos, final_offset_vel) = center_at(jd)?;
    let to_ecliptic = Equatorial::rotation_to_frame::<Ecliptic>();
    let final_elements = EquinoctialElements::from_pos_vel(
        component.elements.desig.clone(),
        jd,
        &(to_ecliptic * (pos_f - final_offset_pos)),
        &(to_ecliptic * (vel_f - final_offset_vel)),
        component.elements.center_id,
        component.elements.gm_sqrt,
    )?;

    // The covariance lives in element coordinates, so the sensitivity has to be expressed
    // element-to-element rather than cartesian-to-cartesian:
    //
    //     Phi_E = J(t_f) . Phi_cart . K(t_0)
    //
    // with `K` the element-to-state Jacobian at the start and `J` its inverse at the end,
    // both requested in the frame the force model integrates in. The parameter columns are
    // untouched by both, since the force model's free parameters are not elements.
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
    propagated.non_grav.clone_from(&component.non_grav);

    Ok(PropagationStep {
        augmented_stm: phi_aug,
        propagated,
    })
}

/// Propagate an [`UncertainState`] and its covariance, without the sigma-point diagnosis.
///
/// The mean is propagated nonlinearly and the covariance through the element-coordinate state
/// transition matrix, `J(t_f) . Phi . K(t_0)`. Use [`step_diffuse_state`] on a
/// one-component mixture when the linearity of that step also matters; this is the cheap
/// path for callers that only want the propagated state.
///
/// Any probes the input carried are dropped rather than advanced, since this path does not
/// measure them; the result reports no `eta`.
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

/// Nonlinearity carried at the end of a propagation leg, and the propagated component it
/// belongs to.
///
/// Sharing one variational integration between the propagation and the probes is the
/// point of this struct: callers that need both should never compute them separately.
///
/// The measurement spans everything since the probes were seeded, which is everything
/// since the component last split under [`step_diffuse_state`].
///
/// Internal: `eta` and the residual reach a caller on the propagated component, and the
/// split direction is consumed by the controller that asked for it.
#[derive(Debug, Clone)]
struct LegDiagnosis {
    /// Linearly propagated component at the end of the leg, carrying the advanced probes,
    /// `eta` and the residual behind it.
    propagated: UncertainState,

    /// Worst whitened position miss of any probe against the linear model: the
    /// two-sigma probe landed off by this many sigma
    /// of the position distribution the component's whitening reference describes, floored
    /// at ten propagator resolutions.
    ///
    /// Also stored on `propagated`, which is where a caller reads it; kept here because
    /// the controller ranks and thresholds on it while the leg is still provisional.
    ///
    /// The residual behind it - the same miss as a cartesian position offset in meters,
    /// before whitening - is only on `propagated`, since nothing in the controller
    /// decides on it.
    eta: f64,

    /// Direction carrying the miss that `eta` measured, and the direction a split
    /// removes it along.  Expressed against the component this leg started from, so it
    /// applies to the state a caller would roll back to rather than to the anchor the probe
    /// was placed at.  Not normalized.  `None` when no direction carried width, or when the
    /// propagation left the probed direction with no extent to split.
    split_direction: Option<DVector<f64>>,

    /// Whether this leg placed fresh probes rather than advancing carried ones.
    seeded: bool,
}

/// What one `+/-` pair contributes: where both probes now are, the worse whitened
/// position miss of the two, and that miss in meters.  A pair that could not be
/// propagated has no states and infinite nonlinearity.
type PairMeasurement = (Option<[(Vector3<f64>, Vector3<f64>); 2]>, f64, f64);

/// Advance a component and its probes across one leg, and measure how far the flow has
/// departed from its linear model on every direction the covariance carries.
///
/// For each eigen-direction `v_i` of the anchor covariance with `lambda_i > 0`, a `+/-`
/// probe pair at two widths:
///
/// ```text
/// r_+/- = offset(phi(m +/- d v_i), phi(m)) -/+ Phi d v_i
/// eta   = max over probes of ||W (J r)_pos||
/// ```
///
/// with `Phi` the accumulated state transition matrix from the anchor, `J` the
/// element-to-cartesian Jacobian at the end of the leg, and `W` the whitening by the
/// *position* image of [`UncertainState::whitening_cov`], floored at
/// [`POSITION_NOISE_FLOOR_AU`].  `eta` answers
/// "the probe's position landed off by this many sigma of the position distribution the
/// component was predicting", where the distribution referred to is the component's own
/// until its first split and its parent's after one.  The model and the residual live in
/// element coordinates,
/// where the flow is nearly linear; the norm is taken on the residual's position image,
/// which is where a metric and a statable noise floor exist.  A covariance direction
/// narrower than the floor cannot dominate the answer, so nonlinearity confined to
/// sub-resolution directions reads as no error rather than as amplified noise.  The pair
/// reports its worse side, so one-sided bending is not averaged away.
///
/// The probes are integrated from wherever the previous leg left them, so nothing here is
/// re-placed and nothing is re-linearized: the residual is against the accumulated
/// transition matrix from the anchor, not against this leg's.
///
/// The returned diagnosis carries the advanced probes on its propagated component.  A
/// caller that accepts the leg keeps that component; a caller that splits it discards it
/// and works from the component the leg started at, whose children carry no probes.
///
/// # Errors
/// Fails if the force model's free-parameter count disagrees with the component's, or if
/// the elements leave their physical domain.  A probe that cannot be propagated reports
/// infinite `eta` rather than failing the call, since the component is genuinely
/// unrepresentable at that scale and the controller should act on it rather than abandon
/// the whole mixture.
fn advance_leg<F>(
    component: &UncertainState,
    carried: &ProbeSet,
    seeded: bool,
    forces: &F,
    center_at: CenterResolver<'_>,
    jd: Time<TDB>,
) -> KeteResult<LegDiagnosis>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    let step = propagate_step(component, forces, center_at, jd)?;
    let np = component.free_params.len();
    let n = 6 + np;

    let mut advanced = carried.clone();
    advanced.stm = &step.augmented_stm * &carried.stm;
    advanced.epoch = jd;

    // The covariance `eta` is whitened against, carried across this leg by the same
    // transition matrix as the component's own.  A component that has never split carries
    // its own covariance here, so the two stay equal; a split child carries the parent's,
    // which is what keeps the threshold meaning one thing at every split depth.  See
    // `UncertainState::whitening_cov`.
    let reference = match &component.whitening_cov {
        Some(existing) => {
            if existing.nrows() != n || existing.ncols() != n {
                return Err(Error::ValueError(format!(
                    "component with {n} covariance dimensions carries a whitening \
                     reference of {}x{}",
                    existing.nrows(),
                    existing.ncols()
                )));
            }
            existing.clone()
        }
        None => component.cov_matrix.clone(),
    };
    let advanced_reference = &step.augmented_stm * reference * step.augmented_stm.transpose();

    let mut propagated = step.propagated;
    if carried.directions.is_empty() {
        propagated.probes = Some(advanced);
        propagated.whitening_cov = Some(advanced_reference);
        propagated.eta = Some(0.0);
        propagated.residual_meters = Some(0.0);
        return Ok(LegDiagnosis {
            propagated,
            eta: 0.0,
            split_direction: None,
            seeded,
        });
    }

    // The significance norm is taken on the residual's cartesian position image.  The
    // linear model, the covariance and the probes stay in element coordinates - that is
    // where the flow is nearly linear and the model stays valid - but element coordinates
    // carry no canonical metric to weigh a residual in, and a covariance pressed into that
    // role is degenerate on exactly the thin directions fitted covariances always have.
    // Position space has a metric, and it is the space the propagator's own resolution is
    // stated in, so the whitening is a propagated position covariance floored at that
    // resolution: a direction narrower than the floor cannot dominate the answer, and a
    // direction the cloud is wide in de-weights a residual the density already covers.
    //
    // The covariance whitened against is the reference above, not this component's own.
    // They are the same matrix until the component's first split.
    //
    // Converting the small residual at a single epoch is a local linearization; it says
    // nothing about the shape of the propagated cloud and does not try to.
    let final_jacobian = propagated.elements.state_jacobian::<Equatorial>()?;
    let element_cov =
        Matrix6::from_iterator(advanced_reference.view((0, 0), (6, 6)).iter().copied());
    let cart_cov = final_jacobian * element_cov * final_jacobian.transpose();
    let pos_cov = cart_cov.fixed_view::<3, 3>(0, 0).into_owned();
    let pos_eigen = SymmetricEigen::new((pos_cov + pos_cov.transpose()) * 0.5);
    let floor_sq = POSITION_NOISE_FLOOR_AU * POSITION_NOISE_FLOOR_AU;
    let inv_vars = [
        1.0 / (pos_eigen.eigenvalues[0].max(0.0) + floor_sq),
        1.0 / (pos_eigen.eigenvalues[1].max(0.0) + floor_sq),
        1.0 / (pos_eigen.eigenvalues[2].max(0.0) + floor_sq),
    ];
    let whiten_pos = |r: &Vector3<f64>| -> f64 {
        (0..3)
            .map(|axis| {
                let projection = pos_eigen.eigenvectors.column(axis).dot(r);
                projection * projection * inv_vars[axis]
            })
            .sum::<f64>()
            .sqrt()
    };
    let (final_offset_pos, final_offset_vel) = center_at(jd)?;
    let to_ecliptic = Equatorial::rotation_to_frame::<Ecliptic>();

    let dead: PairMeasurement = (None, f64::INFINITY, f64::INFINITY);

    let measured: Vec<PairMeasurement> = (0..carried.directions.len())
        .into_par_iter()
        .map(|i| {
            let Some(states) = carried.states[i] else {
                return dead;
            };
            let direction = &carried.directions[i];
            let displacement = carried.displacements[i];

            let mut moved = [(Vector3::zeros(), Vector3::zeros()); 2];
            let mut residual = [DVector::<f64>::zeros(n), DVector::<f64>::zeros(n)];
            for (slot, sign) in [1.0_f64, -1.0].into_iter().enumerate() {
                let Ok((pos, vel)) = propagate_state(
                    forces,
                    states[slot].0,
                    states[slot].1,
                    &carried.params[i][slot],
                    carried.epoch,
                    jd,
                ) else {
                    return dead;
                };
                moved[slot] = (pos, vel);

                // Back to the elements' own center, and read off as an element offset from
                // the propagated base with the true longitude reduced to the shortest
                // signed angle.  Mixing the two descriptions - an element offset added to a
                // cartesian state, or an element covariance normalizing a cartesian
                // residual - produces a number that means nothing, and drove the splitter
                // straight through its component cap on a short arc.
                let Ok(elements) = EquinoctialElements::from_pos_vel(
                    Desig::Empty,
                    jd,
                    &(to_ecliptic * (pos - final_offset_pos)),
                    &(to_ecliptic * (vel - final_offset_vel)),
                    component.elements.center_id,
                    component.elements.gm_sqrt,
                ) else {
                    return dead;
                };

                // The free-parameter rows cancel exactly: parameters are carried through as
                // the identity, so the linear prediction along them is exact.
                let delta = direction * (sign * displacement);
                let local = propagated.elements.offset_to(&elements);
                let mut nonlinear = DVector::<f64>::zeros(n);
                for row in 0..6 {
                    nonlinear[row] = local[row];
                }
                for p in 0..np {
                    nonlinear[6 + p] = delta[6 + p];
                }
                residual[slot] = nonlinear - &advanced.stm * delta;
            }

            // The residual's cartesian position image at the epoch.  The free-parameter
            // rows cancel exactly and carry no position content, so only the element rows
            // convert.  The pair reports its worse side, so one-sided bending is not
            // averaged away.
            let position_of = |v: &DVector<f64>| {
                (final_jacobian * Vector6::from_iterator(v.iter().take(6).copied()))
                    .fixed_rows::<3>(0)
                    .into_owned()
            };
            let plus = position_of(&residual[0]);
            let minus = position_of(&residual[1]);
            let (eta_plus, eta_minus) = (whiten_pos(&plus), whiten_pos(&minus));
            if eta_plus >= eta_minus {
                (Some(moved), eta_plus, plus.norm() * M_PER_AU)
            } else {
                (Some(moved), eta_minus, minus.norm() * M_PER_AU)
            }
        })
        .collect();

    let mut eta = 0.0_f64;
    let mut residual_meters = 0.0_f64;
    let mut worst = 0_usize;
    for (index, &(moved, miss, meters)) in measured.iter().enumerate() {
        advanced.states[index] = moved;
        if miss > eta {
            eta = miss;
            residual_meters = meters;
            worst = index;
        }
    }

    // The probe direction is expressed at the anchor, while the split acts on the component
    // as it stood at the start of this leg, so it has to be carried forward: the states
    // displaced along `v` at the anchor lie along `Phi v` by now, and splitting along `v`
    // itself would narrow an unrelated direction.  For a freshly seeded set the accumulated
    // matrix is the identity and this is `v` unchanged.
    let mapped = &carried.stm * &carried.directions[worst];
    let extent = mapped.norm();
    let split_direction = (extent.is_finite() && extent > 0.0).then_some(mapped);

    propagated.probes = Some(advanced);
    propagated.whitening_cov = Some(advanced_reference);
    propagated.eta = Some(eta);
    propagated.residual_meters = Some(residual_meters);

    Ok(LegDiagnosis {
        propagated,
        eta,
        split_direction,
        seeded,
    })
}

/// Configuration for [`step_diffuse_state`] and [`propagate_diffuse_state`].
///
/// Two settings: how nonlinear a component is allowed to become, and what resolving that
/// is allowed to cost.  Everything else the controller needs it derives from the mixture.
/// How finely an arc is cut is not here, because it is a property of a multi-leg
/// propagation rather than of a leg - [`propagate_diffuse_state`] takes it directly, and a
/// caller driving legs by hand has already chosen it by choosing the epochs.
///
/// Passed to every call rather than stored on anything, so changing the threshold partway
/// through a hand-driven march is visible at the call site.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Nonlinearity above which a component is split: the accuracy/cost dial.
    ///
    /// The controller measures `eta` over every leg - the worst whitened position miss of
    /// a probe at two widths against the linear
    /// model - and splits a component that exceeds this.  Tightening the threshold splits
    /// earlier and more often: the mixture tracks the true density more faithfully and
    /// the run costs more components and more time.  That trade is the setting's entire
    /// meaning; `max_components` bounds what a tight setting may spend.
    ///
    /// The value is in sigma of a predicted position distribution, so it means the same
    /// thing at every covariance size.  It also means the same thing at every split depth:
    /// the distribution referred to is the one the component had before it last split, so
    /// a split that removes curvature lowers `eta` instead of being cancelled by the
    /// narrower covariance it produced.  See [`UncertainState::whitening_cov`].
    pub split_threshold: f64,

    /// Hard cap on the number of components in the propagated mixture.
    ///
    /// Splitting stops once a further split would exceed this, and the leg reports
    /// [`Termination::ComponentCap`], so a caller can tell a saturated budget from a
    /// converged one without re-running.
    pub max_components: usize,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            split_threshold: 0.15,
            // A power of three: splits are three-way, so any other cap truncates a
            // cascade mid-generation and the returned count reports where the budget ran
            // out rather than where the splitter converged.
            max_components: 729,
        }
    }
}

/// Advance a [`DiffuseState`] over one leg, splitting components where the flow stops
/// being linear over their own covariance.
///
/// This is the whole controller for a single leg, and the unit a caller marches with: step
/// to `jd`, look at what came back, step again.  Every component carries its own probes,
/// so a march driven one leg at a time is the same measurement as a single call over the
/// whole arc - `eta` is the departure accumulated since that component last split either
/// way.  [`propagate_diffuse_state`] is exactly this folded over a time grid.
///
/// A component whose `eta` clears `split_threshold` is rolled back to the start of the leg,
/// split three ways along the worst probe's direction, and the leg is redone with the
/// children on fresh probes.  Components are served in order of `weight * eta`, so a heavy
/// badly-represented component is resolved before a light one.
///
/// A component carrying no probes - a new mixture, a split child, or one a caller
/// rebuilt - is seeded here, which restarts its measurement.  [`StepReport::seeded`] counts
/// them, so a march that quietly lost its history says so.
///
/// Splitting stops when every component falls under the threshold, when `max_components`
/// refuses the next split, or when the only components left over threshold carry no
/// direction to split along - the threshold is the accuracy/cost dial and the cap is the
/// brake, and there is deliberately nothing else.  Splitting yields weights `w/6, 2w/3,
/// w/6`, so an outer-of-outer component carries a thirty-sixth of its grandparent's weight
/// and the ranking starves a deep tail on its own.  Work that cares about exactly that
/// tail - impact probability is the motivating case - should read the per-component `eta`
/// on the returned components, and build a [`DiffuseState`] over the tail region and
/// propagate that directly rather than expecting the ranking to reach it.
///
/// What the returned mixture reports is where this test was still failing when the leg
/// ended.  That is not an error bound on the represented density: certifying a mixture
/// against the true pushforward would require knowing that density.
///
/// # Errors
/// Fails if `split_threshold` is not finite and non-negative, if `max_components` is below
/// the input component count, if the weights do not describe the components, if `jd` is not
/// finite, if a component's carried probes do not match it, or if any propagation, split or
/// diagnosis fails.
pub fn step_diffuse_state<F>(
    mixture: &DiffuseState,
    forces: &F,
    jd: Time<TDB>,
    config: &SplitConfig,
    center_at: CenterResolver<'_>,
) -> KeteResult<(DiffuseState, StepReport)>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    validate(mixture, config, jd)?;

    let mut weights = mixture.weights.clone();
    let mut components = mixture.components.clone();
    let mut measured = measure(&components, forces, jd, center_at)?;
    let mut seeded = measured.iter().filter(|d| d.seeded).count();

    // A component whose covariance is too degenerate to yield a split direction is set
    // aside for this leg rather than re-selected forever; the next leg asks again.
    let mut skip = vec![false; components.len()];
    let mut degenerate = false;
    let mut capped = false;

    loop {
        // Priority queue keyed on `weight * eta`: a heavy badly-represented component
        // outranks a light one.  A ranking heuristic; it is not a norm of anything.
        // A component whose probes died reads infinite `eta` and is split like any
        // other; its children re-seed fresh probes, which is the retry, and the cap
        // bounds it.
        let next = (0..components.len())
            .filter(|&i| !skip[i] && measured[i].eta > config.split_threshold)
            .max_by(|&a, &b| {
                (weights[a] * measured[a].eta)
                    .partial_cmp(&(weights[b] * measured[b].eta))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some(index) = next else { break };

        // A K=3 split replaces one component with three.
        if components.len() + 2 > config.max_components {
            capped = true;
            break;
        }

        // Rolling back is free: nothing has been advanced yet, so `components` still
        // holds every component as it stood at the start of this leg.
        let Some(direction) = measured[index].split_direction.clone() else {
            skip[index] = true;
            degenerate = true;
            continue;
        };
        let parts = split_axial_k3_along(&components[index], &direction)?;
        // The one thing the children do inherit. Their probes are fresh, so their `eta`
        // measures the children rather than continuing the parent's history - but the
        // covariance that number is whitened against is the parent's, so a split that
        // helps shows up as a smaller `eta` instead of being cancelled by the narrower
        // denominator it created. See `UncertainState::whitening_cov`.
        let inherited = components[index]
            .whitening_cov
            .clone()
            .unwrap_or_else(|| components[index].cov_matrix.clone());
        let offspring: Vec<UncertainState> = parts
            .iter()
            .map(|(_, c)| {
                let mut child = c.clone();
                child.whitening_cov = Some(inherited.clone());
                child
            })
            .collect();
        let children = measure(&offspring, forces, jd, center_at)?;
        seeded += children.iter().filter(|d| d.seeded).count();

        let parent = weights[index];
        let _ = weights.remove(index);
        let _ = components.remove(index);
        let _ = measured.remove(index);
        let _ = skip.remove(index);
        for ((share, child), outcome) in parts.into_iter().zip(children) {
            weights.push(parent * share);
            components.push(child);
            measured.push(outcome);
            skip.push(false);
        }
    }

    // Accept the leg.  Every component advances to the state its own diagnosis already
    // propagated, carrying the probes and the numbers that diagnosis produced, whether it
    // split on this leg or not.
    let advanced: Vec<UncertainState> = measured.into_iter().map(|d| d.propagated).collect();

    // The cap is what a caller can act on, so it wins when both fired.
    let termination = if capped {
        Termination::ComponentCap
    } else if degenerate {
        Termination::NoSplitDirection
    } else {
        Termination::Converged
    };

    let mut stepped = DiffuseState::new(weights, advanced)?;
    stepped.include_asteroids = mixture.include_asteroids;

    Ok((
        stepped,
        StepReport {
            termination,
            seeded,
        },
    ))
}

/// Adaptively propagate a [`DiffuseState`] to `jd`, splitting components where the flow
/// stops being linear over their own covariance.
///
/// The arc is cut into legs of `step_days` on a common time grid, and
/// [`step_diffuse_state`] is folded over it.  The returned report describes the final leg.
/// Driving the same grid by hand gives the same answer, since the probes ride on the
/// components rather than on this call.
///
/// `step_days` is the time resolution a split is placed at, not an accuracy setting:
/// composing state transition matrices across legs is exact, so subdividing an arc does not
/// change the propagated covariance, and the probes are carried rather than re-placed, so
/// it does not change the measured nonlinearity either.  What a shorter step buys is a
/// split landing closer to where the flow actually stopped being linear, and a component
/// that is not yet nonlinear not being split as though it were.  `split_threshold` is
/// therefore the same demand at any value of it.
///
/// [`DEFAULT_STEP_DAYS`] suits the orbital periods kete is used on.  Deep planetary
/// encounters do not require a shorter step: split placement anywhere in the pre-encounter
/// linear window is equivalent, because the children are narrow enough to propagate
/// linearly to the encounter from any lead.  Lengthen it on arcs of many millennia, where
/// the cost of the default is `arc / step_days` legs and nothing caps that on the caller's
/// behalf.
///
/// An arc of no length is no work: the mixture comes back as it went in, reporting
/// [`Termination::Converged`] and nothing seeded.
///
/// # Errors
/// As [`step_diffuse_state`], and additionally if `step_days` is not finite and positive.
pub fn propagate_diffuse_state<F>(
    diffuse: &DiffuseState,
    forces: &F,
    jd: Time<TDB>,
    config: &SplitConfig,
    step_days: f64,
    center_at: CenterResolver<'_>,
) -> KeteResult<(DiffuseState, StepReport)>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    validate(diffuse, config, jd)?;
    if !step_days.is_finite() || step_days <= 0.0 {
        return Err(Error::ValueError(
            "step_days must be finite and positive".into(),
        ));
    }

    let mut mixture = diffuse.clone();
    let mut report = StepReport {
        termination: Termination::Converged,
        seeded: 0,
    };
    for target in leg_grid(step_days, diffuse.epoch(), jd) {
        let stepped = step_diffuse_state(&mixture, forces, target, config, center_at)?;
        mixture = stepped.0;
        report = stepped.1;
    }
    Ok((mixture, report))
}

/// Shared entry checks for the two public marching paths.
///
/// The mixture's fields are public, so the weights are checked against the components here
/// rather than assumed - a mixture can be taken apart and rebuilt between legs.
fn validate(mixture: &DiffuseState, config: &SplitConfig, jd: Time<TDB>) -> KeteResult<()> {
    if !config.split_threshold.is_finite() || config.split_threshold < 0.0 {
        return Err(Error::ValueError(
            "split_threshold must be finite and non-negative".into(),
        ));
    }
    if config.max_components < mixture.n_components() {
        return Err(Error::ValueError(format!(
            "max_components ({}) must be at least the input component count ({})",
            config.max_components,
            mixture.n_components()
        )));
    }
    if !jd.jd.is_finite() {
        return Err(Error::ValueError(format!(
            "target epoch must be finite, got {}",
            jd.jd
        )));
    }
    if mixture.weights.len() != mixture.components.len() {
        return Err(Error::ValueError(format!(
            "weights ({}) and components ({}) must have equal length",
            mixture.weights.len(),
            mixture.components.len()
        )));
    }
    let sum: f64 = mixture.weights.iter().sum();
    if (sum - 1.0).abs() > crate::state::WEIGHT_SUM_TOL {
        return Err(Error::ValueError(format!(
            "weights must sum to 1.0 within {}, got {sum}",
            crate::state::WEIGHT_SUM_TOL
        )));
    }
    Ok(())
}

/// Cut `start -> end` into legs of `step_days`, with whatever is left over as a shorter
/// final leg.
///
/// Equal steps of time, not of orbital phase: every component reaches the same leg
/// boundaries, so the epochs a split can be placed at do not depend on which component
/// asked for one. The step is a wall-clock duration rather than a fraction of a period, so
/// it does not depend on which component the period is read from, and it stays defined for
/// the hyperbolic and near-parabolic orbits whose period is infinite or far longer than any
/// arc anyone propagates over.
///
/// The grid does not make the reported `eta` comparable across components and is not
/// trying to: probes are carried, so each component's `eta` spans its own history since it
/// last split. That is the intended reading - it is a statement about how far a component
/// is from linear now, not about what the last leg did to it - and the queue ranks current
/// states against each other.
///
/// The step is the one the caller asked for. Dividing the arc into equal parts near that
/// length instead would silently run a different step - a 200-day arc at the 90-day default
/// would step 66.7 days - changing both the cost and the epochs a split can land on without
/// saying so. An arc shorter than one step is a single leg.
///
/// The remainder is the *first* leg, not the last, so every leg after it is a full step and
/// the grid a caller gets is the one they named apart from a single short leg at the start.
///
/// Endpoints are counted back from `end` in multiples of the step rather than accumulated
/// forward, so they do not drift, and the last one is `end` itself.
///
/// The cost is `arc / step_days` legs and nothing bounds it, because the caller can see
/// both numbers and a bound could only be enforced by silently running a step other than
/// the one asked for.
fn leg_grid(step_days: f64, start: Time<TDB>, end: Time<TDB>) -> Vec<Time<TDB>> {
    let arc = end.jd - start.jd;
    if arc == 0.0 || !arc.is_finite() {
        return Vec::new();
    }
    let step = step_days.copysign(arc);

    let mut offsets = Vec::new();
    let mut index = 1.0_f64;
    loop {
        let back = step * index;
        // Strictly inside the arc, since `end` is appended below. The slack keeps an arc
        // that is a whole number of steps from starting with a leg a rounding error long,
        // which would cost a full diagnosis to measure nothing.
        if back.abs() >= arc.abs() * (1.0 - 1e-9) {
            break;
        }
        offsets.push(arc - back);
        index += 1.0;
    }
    offsets.reverse();

    let mut grid: Vec<Time<TDB>> = offsets
        .into_iter()
        .map(|offset| Time::new(start.jd + offset))
        .collect();
    grid.push(end);
    grid
}

/// Advance every component and its own probes over the same leg, seeding any component
/// that carries none.
fn measure<F>(
    components: &[UncertainState],
    forces: &F,
    jd: Time<TDB>,
    center_at: CenterResolver<'_>,
) -> KeteResult<Vec<LegDiagnosis>>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SSB>,
{
    components
        .par_iter()
        .map(|component| {
            let (carried, seeded) = match &component.probes {
                Some(existing) => {
                    check_probes(component, existing)?;
                    (existing.clone(), false)
                }
                None => (ProbeSet::seed(component, center_at)?, true),
            };
            advance_leg(component, &carried, seeded, forces, center_at, jd)
        })
        .collect()
}

/// Reject probes that cannot belong to the component carrying them.
///
/// The probes were placed against a covariance at an epoch, and both are public fields, so
/// a caller can hand back a component whose mean or dimension has moved out from under
/// them.  A mismatched set would still produce a number, measured against an anchor that no
/// longer exists, so it is an error rather than a silent reseed.
fn check_probes(component: &UncertainState, probes: &ProbeSet) -> KeteResult<()> {
    if probes.epoch() != component.elements.epoch {
        return Err(Error::ValueError(format!(
            "component at epoch {} carries probes at epoch {}; probes belong to the \
             component they were seeded from",
            component.elements.epoch.jd,
            probes.epoch().jd
        )));
    }
    let expected = 6 + component.free_params.len();
    if probes.cov_dim() != expected {
        return Err(Error::ValueError(format!(
            "component with {expected} covariance dimensions carries probes of dimension {}",
            probes.cov_dim()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_STEP_DAYS, leg_grid};
    use crate::time::{TDB, Time};

    /// Leg endpoints as offsets in days from the start of the arc.
    fn grid(step: f64, span: f64) -> Vec<f64> {
        let start = Time::<TDB>::new(2_460_000.5);
        leg_grid(step, start, Time::new(start.jd + span))
            .into_iter()
            .map(|t| t.jd - start.jd)
            .collect()
    }

    /// Legs are equal steps of time, so twice the arc is twice the legs at the same step,
    /// and the step does not depend on the orbit being propagated.
    #[test]
    fn grid_steps_equal_intervals_of_time() {
        let short = grid(DEFAULT_STEP_DAYS, DEFAULT_STEP_DAYS * 8.0);
        let long = grid(DEFAULT_STEP_DAYS, DEFAULT_STEP_DAYS * 16.0);
        assert_eq!(short.len(), 8);
        assert_eq!(long.len(), 16);
        for (k, offset) in long.iter().enumerate() {
            #[allow(clippy::cast_precision_loss, reason = "sixteen legs converts exactly")]
            let expected = (k + 1) as f64 * DEFAULT_STEP_DAYS;
            assert!((offset - expected).abs() < 1e-9, "leg {k} at {offset}");
        }
    }

    /// An arc shorter than one step is a single leg landing on the target, not a step
    /// past it.
    #[test]
    fn grid_of_a_short_arc_is_one_leg() {
        let legs = grid(DEFAULT_STEP_DAYS, 10.0);
        assert_eq!(legs.len(), 1);
        assert!((legs[0] - 10.0).abs() < 1e-12);
    }

    /// A partial final leg lands exactly on the requested epoch rather than overshooting
    /// it or accumulating a step at a time towards it.
    #[test]
    fn grid_lands_on_the_target() {
        let span = DEFAULT_STEP_DAYS * 3.5;
        let legs = grid(DEFAULT_STEP_DAYS, span);
        assert_eq!(legs.len(), 4);
        assert!((legs[3] - span).abs() < 1e-9);
    }

    /// The legs are the step the caller asked for. Dividing the arc into equal parts near
    /// that length instead would run a different step - here 66.7 days rather than 90 -
    /// against which the reported `eta` would mean something the caller never asked for.
    ///
    /// The remainder is the first leg. `eta` is reported off the last leg, so a remainder
    /// left there would make the headline number a function of `arc mod step_days` rather
    /// than of the dynamics.
    #[test]
    fn grid_puts_the_remainder_in_the_first_leg() {
        let legs = grid(DEFAULT_STEP_DAYS, 200.0);
        assert_eq!(legs, vec![20.0, 110.0, 200.0]);

        // Shifting the target by a day moves the short leg, not the last one.
        let shifted = grid(DEFAULT_STEP_DAYS, 201.0);
        assert_eq!(shifted, vec![21.0, 111.0, 201.0]);
    }

    /// Backward propagation is the same grid with the sign carried through.
    #[test]
    fn grid_runs_backwards() {
        let span = -DEFAULT_STEP_DAYS * 4.0;
        let back = grid(DEFAULT_STEP_DAYS, span);
        assert_eq!(back.len(), 4);
        assert!(back.iter().all(|&offset| offset < 0.0));
        assert!((back[3] - span).abs() < 1e-9);
    }

    /// An arc of no length has no legs; the mixture comes back as it went in.
    #[test]
    fn grid_of_a_zero_arc_is_empty() {
        assert!(grid(DEFAULT_STEP_DAYS, 0.0).is_empty());
    }
}
