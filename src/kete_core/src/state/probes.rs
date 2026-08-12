//! Probes carried alongside a component to measure its departure from linearity.
//!
//! A [`ProbeSet`] is seeded from one [`UncertainState`]'s covariance, integrated forward
//! with that component, and discarded when the component splits. It is owned by the
//! component it measures, so the two cannot be paired incorrectly.
//!
//! This is the splitting controller's own machinery and is not part of the public API.
//! A caller sees what the probes reported - [`UncertainState::eta`] and
//! [`UncertainState::residual_meters`] - and whether a component still carries any, but
//! never the probes themselves. What can be done with them from outside is exactly two
//! things: ask whether they are there, and drop them.
//!
//! See [`crate::state::step_diffuse_state`] for the controller that advances them.

use crate::frames::Equatorial;
use crate::prelude::{KeteResult, State, UncertainState};
use crate::state::CenterResolver;
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, DVector, SymmetricEigen, Vector3, Vector6};

/// Probe displacement, in 1-sigma widths of the direction probed.
///
/// Two sigma is where the density pays for representation error: the mean log-density
/// damage of a distortion is the displacement squared weighted by the density,
/// `exp(-k^2 / 2) k^4`, which peaks near `k = 2`, and standard sigma-point spacings for
/// these dimensions sit at `sqrt(3)` to `sqrt(n)` widths. A probe at one sigma measures
/// the error where the density barely accumulates it; much beyond two, probes on wide
/// clouds start leaving the element domain.
pub(crate) const PROBE_SIGMA: f64 = 2.0;

/// Probes under integration, and the covariance they were placed against.
///
/// The probes are integrated forward leg by leg and never re-placed, so what they report
/// is the departure from linearity accumulated since they were seeded rather than the
/// departure the last leg added.  Re-placing them each leg would put them back on the
/// ellipsoid the linear model currently predicts, which discards exactly the disagreement
/// the measurement is looking for: nonlinearity that arrives slowly would never register
/// at any step, since each leg individually would look linear.
///
/// Carrying also takes the leg length out of the reported number.  The composed flow and
/// the composed transition matrix do not depend on how the arc was cut, so neither does
/// the residual between them, and `split_threshold` therefore means the same thing at
/// every `step_days`, and across a march taken one leg at a time by the caller.
///
/// The anchor moves only when a component splits.  The children are narrower than the
/// parent along the split direction, so the parent's probes no longer sit at their widths;
/// a split therefore produces children carrying no probes, and the next step seeds them.
///
/// A probe set is only meaningful against the force model it was integrated under and the
/// component it was seeded from. It travels with that component, and
/// [`crate::state::step_diffuse_state`] checks the epoch and dimension still agree before
/// using it.
#[derive(Clone)]
pub(crate) struct ProbeSet {
    /// Eigen-directions of the anchor covariance that carried width, unit length and
    /// `6 + Np` long.
    pub(crate) directions: Vec<DVector<f64>>,

    /// Displacement each pair was placed at: [`PROBE_SIGMA`] times the direction's
    /// 1-sigma width at the anchor.
    pub(crate) displacements: Vec<f64>,

    /// Barycentric cartesian state of the `+` and `-` probe of each pair, at `epoch`.
    /// `None` once either has failed to propagate, which the pair then reports as infinite
    /// nonlinearity for the rest of the run.
    pub(crate) states: Vec<Option<[(Vector3<f64>, Vector3<f64>); 2]>>,

    /// Free parameters each probe of the pair was displaced to.  Constant in time, since
    /// parameters are inputs to the dynamics rather than propagated quantities.
    pub(crate) params: Vec<[Vec<f64>; 2]>,

    /// Accumulated augmented transition matrix from the anchor to `epoch`.
    pub(crate) stm: DMatrix<f64>,

    /// Epoch the carried states are at.
    pub(crate) epoch: Time<TDB>,
}

/// Compact, because a mixture holds one of these per component and the carried states are
/// not something a caller reads: a mixture at the component cap would otherwise print
/// hundreds of thousands of coordinates.
#[allow(
    clippy::missing_fields_in_debug,
    reason = "the carried states and transition matrix are bulk, and summarized by count"
)]
impl std::fmt::Debug for ProbeSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbeSet")
            .field("pairs", &self.directions.len())
            .field("alive", &self.states.iter().filter(|s| s.is_some()).count())
            .field("epoch", &self.epoch.jd)
            .finish()
    }
}

impl ProbeSet {
    /// Place a `+/-` pair on every direction the covariance carries width along.
    ///
    /// Directions with no width are not probed.  A covariance can carry them - Horizons
    /// supplies non-gravitational rows it did not fit, giving eigenvalues of exactly zero -
    /// and a direction with no extent contributes no representation error, which no split
    /// could change either way.
    ///
    /// # Errors
    /// Fails if the central body cannot be resolved at the epoch.  A pair whose displaced
    /// elements leave their physical domain is marked dead instead of failing the call.
    pub(crate) fn seed(
        component: &UncertainState,
        center_at: CenterResolver<'_>,
    ) -> KeteResult<Self> {
        let np = component.free_params.len();
        let n = 6 + np;
        let epoch = component.elements.epoch;
        let sym = SymmetricEigen::new(component.cov_matrix.clone());
        let (offset_pos, offset_vel) = center_at(epoch)?;

        let mut set = Self {
            directions: Vec::with_capacity(n),
            displacements: Vec::with_capacity(n),
            states: Vec::with_capacity(n),
            params: Vec::with_capacity(n),
            stm: DMatrix::identity(n, n),
            epoch,
        };

        for i in 0..n {
            let lambda = sym.eigenvalues[i];
            if !lambda.is_finite() || lambda <= 0.0 {
                continue;
            }
            let direction = sym.eigenvectors.column(i).clone_owned();
            let width = lambda.sqrt();
            let displacement = PROBE_SIGMA * width;
            let mut placed: [Option<(Vector3<f64>, Vector3<f64>)>; 2] = [None, None];
            let mut params: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            for (slot, sign) in [1.0_f64, -1.0].into_iter().enumerate() {
                let delta = &direction * (sign * displacement);
                let shifted = component
                    .elements
                    .displaced_by(&Vector6::from_iterator(delta.iter().take(6).copied()));
                if let Ok(state) = shifted.try_to_state() {
                    let state: State<Equatorial> = state.into_frame();
                    placed[slot] = Some((
                        Vector3::from(state.pos) + offset_pos,
                        Vector3::from(state.vel) + offset_vel,
                    ));
                }
                params[slot] = (0..np)
                    .map(|p| component.free_params[p] + delta[6 + p])
                    .collect();
            }

            set.directions.push(direction);
            set.displacements.push(displacement);
            set.states.push(match (placed[0], placed[1]) {
                (Some(plus), Some(minus)) => Some([plus, minus]),
                _ => None,
            });
            set.params.push(params);
        }

        Ok(set)
    }

    /// Epoch the carried probe states are at.
    pub(crate) fn epoch(&self) -> Time<TDB> {
        self.epoch
    }

    /// Covariance dimension the set was seeded against, `6 + Np`.
    pub(crate) fn cov_dim(&self) -> usize {
        self.stm.nrows()
    }
}
