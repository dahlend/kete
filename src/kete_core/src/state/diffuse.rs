//! Weighted mixture of [`UncertainState`] components.
//!
//! See [`DiffuseState`] for a full description of the model and when to use it.
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use core::f64;

use crate::frames::{CenterBody, DynCenter, Equatorial, InertialFrame};
#[cfg(test)]
use crate::prelude::Desig;
use crate::prelude::{Error, KeteResult, State, TDB, Time, UncertainState};
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardUniform};

/// Tolerance on the sum of mixture weights when validating a new
/// [`DiffuseState`].  The sum is checked against `1.0` with absolute
/// tolerance equal to this value.
pub const WEIGHT_SUM_TOL: f64 = 1e-10;

// K=3 moment-preserving Gaussian-mixture split constants.
//
// The splitting model replaces a univariate Gaussian N(0, 1) with a
// three-component mixture:
//
//   N(0,1) ~ w_o * N(-d, s^2)  +  w_c * N(0, s^2)  +  w_o * N(d, s^2)
//
// where w_o = HUBER_K3_WEIGHTS[0] = HUBER_K3_WEIGHTS[2],
//       w_c = HUBER_K3_WEIGHTS[1],
//       d   = HUBER_K3_MEANS[2] = sqrt(3/2)  (in units of sqrt(lambda)),
//       s   = HUBER_K3_SIGMA.
//
// Derivation of constants (Huber 2008, DeMars et al. 2013)
// ---------------------------------------------------------
// Three moment-preservation constraints fix the free parameters:
//
//   (1) Weights sum to one:   2*w_o + w_c = 1
//   (2) Mean is zero:         symmetric by construction
//   (3) Variance is one:      2*w_o*(d^2 + s^2) + w_c*s^2 = 1
//
// Combined with (1), constraint (3) gives:
//
//   s^2 = 1 - 2*w_o * d^2
//
// The L^2-optimal solution (minimizing the integrated squared difference
// between the mixture and the original Gaussian) gives:
//
//   w_o = 1/6,  w_c = 2/3,  d = sqrt(3/2),  s^2 = 1/2
//
// Substituting: s^2 = 1 - 2*(1/6)*(3/2) = 1 - 1/2 = 1/2.  These are the
// values used here.  They place the outer components farther into the tails
// (d ~ 1.225 sigma vs the naive d = 1) where the dynamics are most
// nonlinear, and produce sub-components that are narrower (s ~ 0.707 vs
// 0.742), both of which are advantages for orbit uncertainty propagation.
//
// Multivariate extension
// ----------------------
// For a multivariate component N(m, P), splitting along unit direction v
// (typically the dominant eigenvector of P, scaled by sqrt(lambda)) gives:
//
//   mean shift:  delta_k = HUBER_K3_MEANS[k] * sqrt(lambda) * v
//   new cov:     P_new = P - (1 - s^2) * lambda * v v^T   (rank-1 reduction)
//
// The rank-1 reduction exactly cancels the between-component variance
// contributed by the shifted means, so the total mixture mean and covariance
// equal the original -- verifiable via the law of total covariance.

/// Mixture weights for the K=3 univariate split: `[w_outer, w_center, w_outer]`.
/// L^2-optimal values: `[1/6, 2/3, 1/6]`.
pub const HUBER_K3_WEIGHTS: [f64; 3] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];

/// Component mean offsets for the K=3 univariate split, in units of
/// `sqrt(lambda)` along the chosen axis: `[-sqrt(3/2), 0, +sqrt(3/2)]`.
pub const HUBER_K3_MEANS: [f64; 3] = [-1.224744871391589, 0.0, 1.224744871391589];

/// Per-component standard-deviation scale for the K=3 split.
/// Derived from `s^2 = 1 - 2 * w_outer * d^2 = 1 - 2*(1/6)*(3/2) = 1/2`,
/// so `s = sqrt(1/2) = 1/sqrt(2)`.
pub const HUBER_K3_SIGMA: f64 = f64::consts::FRAC_1_SQRT_2;

/// A weighted mixture of [`UncertainState`] components.
///
/// # What this represents
///
/// An [`UncertainState`] describes a single best-fit orbit surrounded by a
/// covariance ellipsoid -- the familiar "1-sigma uncertainty region" in
/// position-velocity space.  A `DiffuseState` extends that to a weighted
/// sum of such ellipsoids:
///
/// ```text
/// p(x) = sum_k  w_k * N(x | mu_k, P_k)
/// ```
///
/// where each component k has weight `w_k` (non-negative, summing to 1),
/// mean state `mu_k`, and `N(x | mu_k, P_k)` is the multivariate normal
/// (Gaussian) distribution with mean `mu_k` and covariance matrix `P_k`,
/// evaluated at point x.
/// All components share an epoch, center body, and covariance dimension (6 + Np).
///
/// # When to use it
///
/// Use a `DiffuseState` when a single Gaussian is not an adequate description
/// of your uncertainty.  Common cases:
///
/// - A physical cloud -- debris field, dust trail, or cometary coma -- where
///   each piece has slightly different non-gravitational parameters (e.g. beta
///   values for radiation pressure).  Store one component per sampled beta.
/// - A single orbit whose uncertainty has become large enough that propagating
///   it as one Gaussian introduces significant linear approximation error (the
///   "banana" problem).  Use adaptive splitting to replace that one wide
///   Gaussian with several narrower ones before propagating.
///
/// # The splitting model
///
/// When a covariance ellipsoid is wide along one axis, propagating it
/// forward in time through nonlinear N-body dynamics distorts it into a
/// curved, banana-shaped region that a Gaussian approximates poorly.
/// Splitting the component along its dominant uncertainty axis before
/// propagation keeps each sub-component narrow enough that the Gaussian
/// approximation remains valid.
///
/// The K=3 split (see [`HUBER_K3_WEIGHTS`], [`HUBER_K3_MEANS`],
/// [`HUBER_K3_SIGMA`]) replaces one component N(mu, P) with three:
///
///   - two outer components at mu +/- sqrt(3/2) * sqrt(lambda) * v, weight 1/6 each
///   - one central component at mu, weight 2/3
///
/// where v is the unit eigenvector of P along its dominant axis and lambda
/// is the corresponding eigenvalue (the largest variance direction).  Each
/// sub-component gets a narrower covariance: the variance along v shrinks by
/// a factor of `HUBER_K3_SIGMA`^2 = 1/2, while all other directions are
/// unchanged.  The construction is exact: the weighted mean and total
/// covariance of the three sub-components equal those of the original
/// component (verified by the law of total covariance).
///
/// The split constants are derived from three constraints -- weights sum to 1,
/// the mixture mean equals the original mean, the mixture variance equals the
/// original variance -- with the outer means fixed at +/-1 sigma.  This gives
/// `s^2 = 1 - 2*(1/6)*(3/2) = 1/2`.  See the constant definitions for details.
///
/// # When a split is triggered
///
/// Splitting is decided by the sigma-point divergence diagnostic.  Before
/// propagating a component, the integrator tests whether the component's
/// uncertainty region is small enough that linear propagation (via the
/// state transition matrix, STM) is a reliable approximation.
///
/// The test works as follows.  Take the dominant eigenvectors of the
/// covariance (the axes of largest uncertainty), and for each one perturb
/// the mean orbit by +/- `sigma_factor` * sqrt(lambda) along that axis
/// (i.e. place a "sigma point" one standard deviation away from the mean
/// in that direction).  Propagate each perturbed orbit via the full N-body
/// integrator to the target epoch.  The STM predicts where those points
/// should have moved under a linearized model; compare the two predictions.
/// The divergence score is:
///
/// ```text
/// divergence = max_k  ||delta_full_k - delta_lin_k|| / ||delta_lin_k||
/// ```
///
/// where delta is the displacement from the propagated mean.  A value
/// near zero means the dynamics are nearly linear over the component's
/// uncertainty region and propagating it as a single Gaussian is safe.
/// A value above the split threshold (default 0.05, i.e. 5% relative
/// error) means the banana distortion is significant and the component
/// should be split into narrower sub-components first.
///
/// In practice a well-observed main-belt asteroid rarely needs splitting
/// even over months of propagation, while a poorly-constrained near-Earth
/// object on a close-approach trajectory may need several splits to keep
/// the mixture accurate.
///
/// Splitting stops when either a hard cap on the total number of
/// components is reached (`max_components`, default 64) or when a component
/// has been split `max_split_depth` times (default 4) without falling below
/// the threshold.  In both cases the component is propagated linearly
/// as-is rather than split further.  The caps exist to bound runtime; if
/// they are frequently hit, lower the threshold or increase the caps.
///
/// # Usage
///
/// Use [`DiffuseState::from_uncertain`] to wrap a single [`UncertainState`],
/// [`DiffuseState::new`] for explicit multi-component construction, and
/// [`DiffuseState::split_component`] to split one component in place.
/// Adaptive propagation (repeated split-then-propagate) is handled by
/// `kete_spice::propagation::propagate_diffuse_state_adaptive`.
///
/// Generic over the per-component frame `F` and center `C`. The
/// defaults `<Equatorial, DynCenter>` match the historical concrete
/// shape; existing callers writing `DiffuseState` (no generic args)
/// get the same type as before.
#[derive(Debug, Clone)]
pub struct DiffuseState<F = Equatorial, C = DynCenter>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Mixture weights.  Same length as `components`, non-negative,
    /// summing to `1.0` within `WEIGHT_SUM_TOL`.
    pub weights: Vec<f64>,

    /// Component states.  All share epoch, center, and covariance
    /// dimension; per-component `free_params` may differ in value.
    pub components: Vec<UncertainState<F, C>>,
}

impl<F, C> DiffuseState<F, C>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    /// Construct a [`DiffuseState`] from explicit weights and components.
    ///
    /// # Errors
    /// Returns an error if any of the structural invariants is
    /// violated: mismatched lengths, empty input, negative or
    /// non-finite weights, weights not summing to `1.0` within
    /// [`WEIGHT_SUM_TOL`], or components disagreeing on epoch, center,
    /// or covariance dimension.
    pub fn new(weights: Vec<f64>, components: Vec<UncertainState<F, C>>) -> KeteResult<Self> {
        if components.is_empty() {
            return Err(Error::ValueError(
                "DiffuseState must have at least one component".into(),
            ));
        }
        if weights.len() != components.len() {
            return Err(Error::ValueError(format!(
                "weights length {} does not match components length {}",
                weights.len(),
                components.len()
            )));
        }
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(Error::ValueError(
                "weights must be finite and non-negative".into(),
            ));
        }
        let sum: f64 = weights.iter().sum();
        if (sum - 1.0).abs() > WEIGHT_SUM_TOL {
            return Err(Error::ValueError(format!(
                "weights must sum to 1.0 within {WEIGHT_SUM_TOL}, got {sum}"
            )));
        }

        let first = &components[0];
        let epoch = first.state.epoch;
        let center = first.state.center_id();
        let n_dim = first.cov_matrix.nrows();

        for (i, c) in components.iter().enumerate().skip(1) {
            if c.state.epoch != epoch {
                return Err(Error::ValueError(format!(
                    "component {i} epoch {} does not match component 0 epoch {}",
                    c.state.epoch.jd, epoch.jd
                )));
            }
            if c.state.center_id() != center {
                return Err(Error::ValueError(format!(
                    "component {i} center {} does not match component 0 center {center}",
                    c.state.center_id()
                )));
            }
            if c.cov_matrix.nrows() != n_dim {
                return Err(Error::ValueError(format!(
                    "component {i} covariance dimension {} does not match component 0 \
                     dimension {n_dim}",
                    c.cov_matrix.nrows()
                )));
            }
        }

        Ok(Self {
            weights,
            components,
        })
    }

    /// Wrap a single [`UncertainState`] as a one-component mixture.
    #[must_use]
    pub fn from_uncertain(state: UncertainState<F, C>) -> Self {
        Self {
            weights: vec![1.0],
            components: vec![state],
        }
    }

    /// Common epoch shared by all components.
    pub fn epoch(&self) -> Time<TDB> {
        self.components[0].state.epoch
    }

    /// Number of mixture components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Number of free parameters per component (`0` if all components
    /// have empty `free_params`).
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.components[0].free_params.len()
    }

    /// Total covariance dimension, `6 + n_params()`.
    #[must_use]
    pub fn cov_dim(&self) -> usize {
        6 + self.n_params()
    }

    /// Drop components whose weight is below `min_weight` and renormalize
    /// the remaining weights to sum to `1.0`.
    ///
    /// After adaptive splitting, outer sub-components accumulate at
    /// weights as small as `HUBER_K3_WEIGHTS[0]^depth`.  Pruning removes
    /// these negligible components before the next propagation step,
    /// keeping the component count bounded without meaningfully changing
    /// the mixture.
    ///
    /// # Errors
    /// Returns an error if every component would be pruned or if the
    /// surviving weights sum to zero.
    pub fn prune(&mut self, min_weight: f64) -> KeteResult<()> {
        let mut new_weights = Vec::with_capacity(self.weights.len());
        let mut new_components = Vec::with_capacity(self.components.len());
        for (w, c) in self.weights.iter().zip(self.components.iter()) {
            if *w >= min_weight {
                new_weights.push(*w);
                new_components.push(c.clone());
            }
        }
        if new_weights.is_empty() {
            return Err(Error::ValueError(format!(
                "prune({min_weight}) would remove every component"
            )));
        }
        let sum: f64 = new_weights.iter().sum();
        if sum <= 0.0 {
            return Err(Error::ValueError(
                "surviving weights sum to zero after prune".into(),
            ));
        }
        for w in &mut new_weights {
            *w /= sum;
        }
        self.weights = new_weights;
        self.components = new_components;
        Ok(())
    }

    /// Draw random samples from the mixture distribution.
    ///
    /// Each sample is drawn by selecting a component with probability
    /// proportional to its weight, then sampling that component's
    /// underlying [`UncertainState`].  Returns `(state, free_params)`
    /// pairs in the same shape as [`UncertainState::sample`].
    ///
    /// # Errors
    /// Returns an error if any component's sampling fails.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<F, C>, Vec<f64>)>> {
        // Build a CDF over the component weights for inverse-CDF sampling.
        let mut cdf = Vec::with_capacity(self.weights.len());
        let mut acc = 0.0;
        for w in &self.weights {
            acc += w;
            cdf.push(acc);
        }

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_seed(rand::random()),
        };

        // Count how many samples each component owes, drawing the
        // selection up front so the per-component sample counts are
        // deterministic with respect to the seed.
        let mut counts = vec![0_usize; self.components.len()];
        for _ in 0..n_samples {
            let u: f64 = StandardUniform.sample(&mut rng);
            let idx = cdf
                .iter()
                .position(|&c| u <= c)
                .unwrap_or(self.components.len() - 1);
            counts[idx] += 1;
        }

        // Per-component sampling.  Salt the seed so different components
        // do not share a draw sequence, while still being reproducible.
        let mut results = Vec::with_capacity(n_samples);
        for (idx, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let comp_seed = seed.map(|s| s.wrapping_add(idx as u64).wrapping_add(1));
            let mut comp = self.components[idx].sample(count, comp_seed)?;
            results.append(&mut comp);
        }

        Ok(results)
    }
}

/// Split a component for adaptive propagation.
///
/// Prefers a dynamics-aware direction: finds the dominant eigenvector of
/// `prop_cov` (the propagated covariance) and maps it back to initial state
/// space via `augmented_stm^{-1}`, then calls [`split_axial_k3_along`].
/// Falls back to splitting along the dominant eigenvector of the component's
/// own covariance if the STM is singular or the mapped direction fails the
/// positive-definiteness check.
///
/// # Errors
/// Returns an error only if both the dynamics-aware split and the fallback
/// fail (e.g. the component has zero covariance).
pub fn split_for_propagation<F, C>(
    component: &UncertainState<F, C>,
    prop_cov: &DMatrix<f64>,
    augmented_stm: &DMatrix<f64>,
) -> KeteResult<Vec<(f64, UncertainState<F, C>)>>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    // Prefer dynamics-aware direction; fall back to dominant initial-covariance axis.
    if let Ok(v_f) = dominant_eigenvector(prop_cov)
        && let Some(phi_inv) = augmented_stm.clone().try_inverse()
        && let Ok(parts) = split_axial_k3_along(component, &(phi_inv * v_f))
    {
        return Ok(parts);
    }

    let fallback_dir = dominant_eigenvector(&component.cov_matrix)?;
    split_axial_k3_along(component, &fallback_dir)
}

/// Dominant eigenvector of a symmetric positive-semi-definite matrix.
fn dominant_eigenvector(cov: &DMatrix<f64>) -> KeteResult<DVector<f64>> {
    let sym = SymmetricEigen::new(cov.clone());
    let (max_idx, &max_lambda) = sym
        .eigenvalues
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
        .expect("eigenvalues is non-empty");
    if !max_lambda.is_finite() || max_lambda <= 0.0 {
        return Err(Error::ValueError(
            "covariance has no positive eigenvalue".into(),
        ));
    }
    Ok(sym.eigenvectors.column(max_idx).into_owned())
}

/// Split a single [`UncertainState`] into a K=3 sub-mixture along an
/// explicitly supplied direction in initial state space.
///
/// `direction` does not need to be a unit vector (it is normalized
/// internally) and does not need to be an eigenvector of the
/// component's covariance.  The K=3 sub-mixture preserves the
/// component's mean and total covariance exactly for any direction;
/// the only failure mode is when the rank-1 covariance reduction
/// produces a non-positive-definite result, which can happen when
/// `direction` aligns with a small-eigenvalue subspace of a strongly
/// anisotropic covariance.
///
/// The motivating use case is dynamics-aware splitting for
/// adaptive cloud propagation: the calling layer computes the
/// dominant eigenvector of the *propagated* covariance and maps it
/// back through the augmented STM, picking out the initial-state
/// direction that most amplifies under the dynamics.  For an
/// isotropic initial covariance the eigenvalue decomposition is
/// degenerate and naive dominant-eigenvector splitting picks an
/// arbitrary axis; this entry point lets callers pick a meaningful
/// one instead.
///
/// # Errors
/// Returns an error if `direction` has the wrong length, is zero
/// (or non-finite), if the variance of the component along
/// `direction` is non-positive, or if the post-split covariance
/// fails a positive-definiteness sanity check.
fn split_axial_k3_along<F, C>(
    component: &UncertainState<F, C>,
    direction: &DVector<f64>,
) -> KeteResult<Vec<(f64, UncertainState<F, C>)>>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    let dim = component.cov_matrix.nrows();
    if direction.len() != dim {
        return Err(Error::ValueError(format!(
            "split_axial_k3_along: direction length {} does not match \
             component covariance dimension {dim}",
            direction.len()
        )));
    }
    let norm = direction.norm();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(Error::ValueError(
            "split_axial_k3_along: direction must have finite, nonzero norm".into(),
        ));
    }
    let d = direction / norm;

    // Variance of the input covariance along d.
    let sigma_sq_mat = d.transpose() * &component.cov_matrix * &d;
    let sigma_sq = sigma_sq_mat[(0, 0)];
    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return Err(Error::ValueError(
            "split_axial_k3_along: variance along the requested direction is non-positive".into(),
        ));
    }
    let sigma = sigma_sq.sqrt();

    // Rank-1 reduction:
    //   P_new = P - (1 - HUBER_K3_SIGMA^2) * sigma^2 * d d^T
    //
    // For any direction d, this preserves the total mean (Sigma w_i m_i = 0)
    // and total covariance (the rank-1 reduction is exactly the
    // between-component variance contributed by the K=3 means).
    let alpha = (1.0 - HUBER_K3_SIGMA.powi(2)) * sigma_sq;
    let outer = &d * d.transpose();
    let mut cov_new = component.cov_matrix.clone();
    cov_new -= outer * alpha;
    // Force exact symmetry to eliminate floating-point roundoff drift.
    let cov_new = (&cov_new + cov_new.transpose()) * 0.5;

    // Sanity check: the rank-1 reduction is guaranteed PD when d lies
    // in (or near) the dominant eigenspace of P.  For pathological
    // alignments with a small-eigenvalue direction it can fail; surface
    // that as an error so callers can fall back to a safer split.
    let sym = SymmetricEigen::new(cov_new.clone());
    let min_eig = sym
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if min_eig < -sigma_sq.abs() * 1e-10 {
        return Err(Error::ValueError(format!(
            "split_axial_k3_along: post-split covariance not positive-definite \
             (min eigenvalue {min_eig:.3e}, projected variance {sigma_sq:.3e})"
        )));
    }

    let mut result = Vec::with_capacity(3);
    for k in 0..3 {
        let offset = HUBER_K3_MEANS[k] * sigma;
        let delta = &d * offset;
        let new_uncertain = build_split_component(component, &delta, cov_new.clone())?;
        result.push((HUBER_K3_WEIGHTS[k], new_uncertain));
    }
    Ok(result)
}

/// Build a new [`UncertainState`] by shifting `base`'s mean by `delta`
/// in the augmented `(6 + Np)` space and replacing its covariance.
fn build_split_component<F, C>(
    base: &UncertainState<F, C>,
    delta: &DVector<f64>,
    new_cov: DMatrix<f64>,
) -> KeteResult<UncertainState<F, C>>
where
    F: InertialFrame,
    C: CenterBody,
    DynCenter: From<C>,
{
    let np = base.free_params.len();

    let new_state = State::<F, C> {
        desig: base.state.desig.clone(),
        epoch: base.state.epoch,
        pos: crate::frames::Vector::<F>::new([
            base.state.pos[0] + delta[0],
            base.state.pos[1] + delta[1],
            base.state.pos[2] + delta[2],
        ]),
        vel: crate::frames::Vector::<F>::new([
            base.state.vel[0] + delta[3],
            base.state.vel[1] + delta[4],
            base.state.vel[2] + delta[5],
        ]),
        center: base.state.center,
    };

    let new_params: Vec<f64> = (0..np)
        .map(|i| base.free_params[i] + delta[6 + i])
        .collect();

    UncertainState::new(new_state, new_cov, new_params)
}

#[cfg(test)]
mod tests {
    impl<F, C> DiffuseState<F, C>
    where
        F: InertialFrame,
        C: CenterBody,
        DynCenter: From<C>,
    {
        /// Weighted mean of the Cartesian state across all components.
        /// Test-only helper for verifying moment preservation after splits.
        fn mean_state(&self) -> State<F, C> {
            let first = &self.components[0].state;
            let mut pos = [0.0; 3];
            let mut vel = [0.0; 3];
            for (w, c) in self.weights.iter().zip(self.components.iter()) {
                for i in 0..3 {
                    pos[i] += w * c.state.pos[i];
                    vel[i] += w * c.state.vel[i];
                }
            }
            State {
                desig: Desig::Empty,
                epoch: first.epoch,
                pos: crate::frames::Vector::<F>::new(pos),
                vel: crate::frames::Vector::<F>::new(vel),
                center: first.center,
            }
        }

        /// Weighted mean of the free parameters across all components.
        /// Test-only helper for verifying moment preservation after splits.
        fn mean_params(&self) -> Vec<f64> {
            let np = self.n_params();
            if np == 0 {
                return Vec::new();
            }
            let mut params = vec![0.0; np];
            for (w, c) in self.weights.iter().zip(self.components.iter()) {
                for (i, p) in c.free_params.iter().enumerate() {
                    params[i] += w * p;
                }
            }
            params
        }

        /// Total covariance of the mixture via the law of total covariance:
        /// `P = sum_i w_i * (P_i + (m_i - m)(m_i - m)^T)`.
        ///
        /// Test-only helper for verifying that splitting preserves total
        /// moments.  Not part of the public API -- in Cartesian space this
        /// quantity is of limited interpretive value since components spread
        /// along a curved orbit track produce large off-diagonal terms that
        /// obscure the actual per-orbit uncertainty.
        fn covariance(&self) -> DMatrix<f64> {
            let n = self.cov_dim();
            let mean_state = self.mean_state();
            let mean_params = self.mean_params();
            let mut mean = DVector::<f64>::zeros(n);
            for i in 0..3 {
                mean[i] = mean_state.pos[i];
                mean[3 + i] = mean_state.vel[i];
            }
            for (i, p) in mean_params.iter().enumerate() {
                mean[6 + i] = *p;
            }
            let mut total = DMatrix::<f64>::zeros(n, n);
            let mut comp_vec = DVector::<f64>::zeros(n);
            for (w, c) in self.weights.iter().zip(self.components.iter()) {
                total += &c.cov_matrix * *w;
                for i in 0..3 {
                    comp_vec[i] = c.state.pos[i];
                    comp_vec[3 + i] = c.state.vel[i];
                }
                for (i, p) in c.free_params.iter().enumerate() {
                    comp_vec[6 + i] = *p;
                }
                let dev = &comp_vec - &mean;
                total += &dev * dev.transpose() * *w;
            }
            total
        }

        /// Split component `idx` along its dominant covariance eigenvector
        /// into a K=3 sub-mixture.  Test-only helper used to verify moment
        /// preservation; splitting is not part of the public API.
        fn split_component(&self, idx: usize) -> KeteResult<Self> {
            if idx >= self.components.len() {
                return Err(Error::ValueError(format!(
                    "split_component: idx {idx} out of range (n_components={})",
                    self.components.len()
                )));
            }
            let dir = dominant_eigenvector(&self.components[idx].cov_matrix)?;
            let parts = split_axial_k3_along(&self.components[idx], &dir)?;
            let original_w = self.weights[idx];
            let mut new_weights = Vec::with_capacity(self.weights.len() + 2);
            let mut new_components = Vec::with_capacity(self.components.len() + 2);
            for (i, (w, c)) in self.weights.iter().zip(self.components.iter()).enumerate() {
                if i == idx {
                    continue;
                }
                new_weights.push(*w);
                new_components.push(c.clone());
            }
            for (w_split, c_split) in parts {
                new_weights.push(original_w * w_split);
                new_components.push(c_split);
            }
            Self::new(new_weights, new_components)
        }
    }

    use super::*;

    fn test_state(desig: &str) -> State<Equatorial> {
        State::new(
            Desig::Name(desig.into()),
            // J2000.0
            2451545.0,
            [1.0, 0.0, 0.0],
            [0.0, 0.01720209895, 0.0],
            10,
        )
    }

    fn small_uncertain(desig: &str) -> UncertainState {
        let cov = DMatrix::identity(6, 6) * 1e-12;
        UncertainState::new(test_state(desig), cov, vec![]).unwrap()
    }

    #[test]
    fn test_new_validates_weights_sum() {
        let comps = vec![small_uncertain("A"), small_uncertain("B")];
        // Sum != 1.
        assert!(DiffuseState::new(vec![0.4, 0.4], comps.clone()).is_err());
        // Negative weight.
        assert!(DiffuseState::new(vec![1.2, -0.2], comps.clone()).is_err());
        // Length mismatch.
        assert!(DiffuseState::new(vec![1.0], comps.clone()).is_err());
        // Empty components.
        assert!(DiffuseState::<Equatorial, DynCenter>::new(vec![], vec![]).is_err());
        // Valid.
        assert!(DiffuseState::new(vec![0.5, 0.5], comps).is_ok());
    }

    #[test]
    fn test_new_validates_epoch_and_center() {
        let mut a = small_uncertain("A");
        let b = small_uncertain("B");

        // Different epoch.
        a.state.epoch = 2451600.0.into();
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![a.clone(), b.clone()]).is_err());

        // Different center.
        let mut a = small_uncertain("A");
        let mut b = small_uncertain("B");
        b.state = State::new(
            b.state.desig.clone(),
            b.state.epoch,
            [b.state.pos[0], b.state.pos[1], b.state.pos[2]],
            [b.state.vel[0], b.state.vel[1], b.state.vel[2]],
            // SSB instead of Sun.
            0,
        );
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![a.clone(), b]).is_err());

        // Restore matching center, expect ok.
        a = small_uncertain("A");
        let b = small_uncertain("B");
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![a, b]).is_ok());
    }

    #[test]
    fn test_new_validates_covariance_dim() {
        let st = test_state("A");
        let cov_6 = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let cov_7 = DMatrix::<f64>::identity(7, 7) * 1e-12;

        // Component without free params.
        let plain = UncertainState::new(st.clone(), cov_6.clone(), vec![]).unwrap();
        // Component with one free param -> 7x7 cov.
        let with_param = UncertainState::new(st.clone(), cov_7, vec![0.01]).unwrap();

        // Mixing 6x6 and 7x7 cov dims is invalid.
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![plain, with_param.clone()]).is_err());

        // Two components both with one free param (different values) are valid.
        let with_param2 =
            UncertainState::new(st, DMatrix::<f64>::identity(7, 7) * 1e-12, vec![0.05]).unwrap();
        let mix = DiffuseState::new(vec![0.5, 0.5], vec![with_param, with_param2]).unwrap();
        assert_eq!(mix.n_components(), 2);
        assert_eq!(mix.n_params(), 1);
        assert_eq!(mix.cov_dim(), 7);
    }

    #[test]
    fn test_from_uncertain_single_component() {
        let u = small_uncertain("solo");
        let d = DiffuseState::from_uncertain(u);
        assert_eq!(d.n_components(), 1);
        assert_eq!(d.weights, vec![1.0]);
        assert_eq!(d.cov_dim(), 6);
    }

    #[test]
    fn test_mean_state_collapses_for_single_component() {
        let u = small_uncertain("A");
        let want_pos = u.state.pos;
        let want_vel = u.state.vel;
        let d = DiffuseState::from_uncertain(u);
        let m = d.mean_state();
        for i in 0..3 {
            assert_eq!(m.pos[i], want_pos[i]);
            assert_eq!(m.vel[i], want_vel[i]);
        }
    }

    #[test]
    fn test_mean_state_two_component() {
        let mut a = small_uncertain("A");
        let mut b = small_uncertain("B");
        a.state = State::new(
            a.state.desig.clone(),
            a.state.epoch,
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            10,
        );
        b.state = State::new(
            b.state.desig.clone(),
            b.state.epoch,
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            10,
        );
        let d = DiffuseState::new(vec![0.25, 0.75], vec![a, b]).unwrap();
        let m = d.mean_state();
        // 0.25 * 0 + 0.75 * 2 = 1.5
        assert!((m.pos[0] - 1.5).abs() < 1e-15);
        // 0.25 * 0 + 0.75 * 1 = 0.75
        assert!((m.vel[1] - 0.75).abs() < 1e-15);
    }

    #[test]
    fn test_covariance_law_of_total_variance() {
        // Two components offset along x with identical small spherical
        // covariance.  Total covariance should equal individual P plus
        // the between-component spread along x.
        let st = test_state("A");
        let p = DMatrix::<f64>::identity(6, 6) * 1e-6;

        let mut a_state = st.clone();
        a_state.pos = [-1.0, 0.0, 0.0].into();
        let mut b_state = st;
        b_state.pos = [1.0, 0.0, 0.0].into();

        let a = UncertainState::new(a_state, p.clone(), vec![]).unwrap();
        let b = UncertainState::new(b_state, p.clone(), vec![]).unwrap();
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();

        let cov = d.covariance();
        // Within: 1e-6.  Between along x: 0.5 * (-1)^2 + 0.5 * 1^2 = 1.0.
        assert!((cov[(0, 0)] - (1e-6 + 1.0)).abs() < 1e-12);
        // Other diagonal entries get only the within term.
        assert!((cov[(1, 1)] - 1e-6).abs() < 1e-12);
        assert!((cov[(2, 2)] - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_mean_params_with_free_params() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(7, 7) * 1e-12;
        let a = UncertainState::new(st.clone(), cov.clone(), vec![0.01]).unwrap();
        let b = UncertainState::new(st, cov, vec![0.05]).unwrap();
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();
        let p = d.mean_params();
        assert_eq!(p.len(), 1);
        assert!((p[0] - 0.03).abs() < 1e-15);
    }

    #[test]
    fn test_sample_count_and_distribution() {
        // A 90/10 mixture should yield ~90% samples from the first
        // component.  We validate via the desig label (each component
        // carries a different desig) so we know which it came from.
        let mut a = small_uncertain("A");
        let mut b = small_uncertain("B");
        // Make the components separable in position so samples
        // unambiguously belong to one or the other given the tiny
        // covariance.
        a.state.pos = [0.0, 0.0, 0.0].into();
        b.state.pos = [100.0, 0.0, 0.0].into();
        let d = DiffuseState::new(vec![0.9, 0.1], vec![a, b]).unwrap();

        let samples = d.sample(1000, Some(7)).unwrap();
        assert_eq!(samples.len(), 1000);
        let n_a = samples.iter().filter(|(s, _)| s.pos[0] < 50.0).count();
        // Expect ~900 from component A; allow generous slack.
        assert!(n_a > 850 && n_a < 950, "got n_a = {n_a}");
    }

    #[test]
    fn test_sample_seed_is_deterministic() {
        let a = small_uncertain("A");
        let b = small_uncertain("B");
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();
        let s1 = d.sample(20, Some(42)).unwrap();
        let s2 = d.sample(20, Some(42)).unwrap();
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            for i in 0..3 {
                assert_eq!(a.0.pos[i], b.0.pos[i]);
                assert_eq!(a.0.vel[i], b.0.vel[i]);
            }
        }
    }

    #[test]
    fn test_prune_drops_low_weight_components() {
        let a = small_uncertain("A");
        let b = small_uncertain("B");
        let c = small_uncertain("C");
        let mut d = DiffuseState::new(vec![0.6, 0.3, 0.1], vec![a, b, c]).unwrap();
        d.prune(0.2).unwrap();
        assert_eq!(d.n_components(), 2);
        // Remaining weights renormalize to sum to 1.
        assert!((d.weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        // Original ratio 0.6 : 0.3 = 2 : 1 -> 2/3, 1/3.
        assert!((d.weights[0] - 2.0 / 3.0).abs() < 1e-15);
        assert!((d.weights[1] - 1.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn test_prune_rejects_total_pruning() {
        let a = small_uncertain("A");
        let mut d = DiffuseState::from_uncertain(a);
        assert!(d.prune(2.0).is_err());
    }

    /// The K=3 split tables must satisfy the moment-preservation
    /// constraints for `N(0,1) -> sum_i w_i N(m_i, sigma^2)`.
    #[test]
    fn test_huber_k3_constants_preserve_moments() {
        let sum: f64 = HUBER_K3_WEIGHTS.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-15,
            "weights must sum to 1: got {sum}"
        );
        let mean: f64 = HUBER_K3_WEIGHTS
            .iter()
            .zip(HUBER_K3_MEANS.iter())
            .map(|(w, m)| w * m)
            .sum();
        assert!(mean.abs() < 1e-15, "mean must be 0: got {mean}");
        let variance: f64 = HUBER_K3_WEIGHTS
            .iter()
            .zip(HUBER_K3_MEANS.iter())
            .map(|(w, m)| w * (m * m + HUBER_K3_SIGMA * HUBER_K3_SIGMA))
            .sum();
        assert!(
            (variance - 1.0).abs() < 1e-15,
            "variance must be 1: got {variance}"
        );
    }

    /// Splitting a component preserves the total weight (with K=3
    /// sub-weights summing to the original component's weight) and
    /// produces 2 additional components (1 -> 3).
    #[test]
    fn test_split_component_basic() {
        let mut a = small_uncertain("A");
        // Inflate covariance so the dominant eigenvalue is well-defined
        // and large enough that split delta is non-trivial.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 1.0;
        cov[(1, 1)] = 1e-6;
        cov[(2, 2)] = 1e-6;
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        a.cov_matrix = cov;

        let d = DiffuseState::from_uncertain(a);
        let split = d.split_component(0).unwrap();
        assert_eq!(split.n_components(), 3);
        let total_w: f64 = split.weights.iter().sum();
        assert!((total_w - 1.0).abs() < 1e-15);

        // Each sub-weight should equal HUBER_K3_WEIGHTS[k] (since
        // original weight was 1.0).
        for (got, &want) in split.weights.iter().zip(HUBER_K3_WEIGHTS.iter()) {
            assert!((got - want).abs() < 1e-15);
        }
    }

    /// Splitting must preserve the mixture's mean and total covariance
    /// (law of total covariance).  The dominant axis variance shrinks
    /// per-component but the BETWEEN-component spread compensates.
    #[test]
    fn test_split_component_preserves_moments() {
        let st = test_state("A");
        // Anisotropic covariance -- dominant axis is x.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 4.0;
        cov[(1, 1)] = 1.0;
        cov[(2, 2)] = 1.0;
        for i in 3..6 {
            cov[(i, i)] = 0.01;
        }
        let a = UncertainState::new(st.clone(), cov.clone(), vec![]).unwrap();
        let original_mean = a.state.pos;
        let d = DiffuseState::from_uncertain(a);

        let split = d.split_component(0).unwrap();
        let split_mean = split.mean_state();
        let split_cov = split.covariance();

        // Mean is preserved.
        for i in 0..3 {
            assert!(
                (split_mean.pos[i] - original_mean[i]).abs() < 1e-12,
                "mean[{i}] changed: {} vs {}",
                split_mean.pos[i],
                original_mean[i]
            );
        }
        // Total covariance is preserved (law of total variance).
        for r in 0..6 {
            for c in 0..6 {
                let diff = (split_cov[(r, c)] - cov[(r, c)]).abs();
                let scale = cov[(r, c)].abs().max(1e-12);
                assert!(
                    diff / scale < 1e-12,
                    "cov[{r},{c}] changed: {} vs {}, rel_err={}",
                    split_cov[(r, c)],
                    cov[(r, c)],
                    diff / scale
                );
            }
        }
    }

    /// Splitting a parameter-dispersed component shifts the
    /// free-parameter values of the sub-components (since the dominant
    /// eigenvector is along the parameter axis when the position
    /// covariance is small relative to the parameter variance).
    #[test]
    fn test_split_component_param_axis() {
        let st = test_state("A");
        // 7x7 cov: tiny in (r,v), large in the free param.
        let mut cov = DMatrix::<f64>::zeros(7, 7);
        for i in 0..3 {
            cov[(i, i)] = 1e-12;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        cov[(6, 6)] = 1e-4;

        let a = UncertainState::new(st, cov, vec![0.01]).unwrap();
        let d = DiffuseState::from_uncertain(a);

        let split = d.split_component(0).unwrap();
        assert_eq!(split.n_components(), 3);

        // Each sub-component should have a different free-param value
        // (sub means shifted by HUBER_K3_MEANS[k] * sqrt(1e-4) from the original 0.01).
        let offset = HUBER_K3_MEANS[2] * 1e-4_f64.sqrt(); // sqrt(3/2) * 0.01
        let mut params: Vec<f64> = split.components.iter().map(|c| c.free_params[0]).collect();
        params.sort_by(f64::total_cmp);
        assert!((params[0] - (0.01 - offset)).abs() < 1e-10);
        assert!((params[1] - 0.01).abs() < 1e-10);
        assert!((params[2] - (0.01 + offset)).abs() < 1e-10);
    }

    #[test]
    fn test_split_component_rejects_zero_covariance() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::zeros(6, 6);
        let a = UncertainState::new(st, cov, vec![]).unwrap();
        let d = DiffuseState::from_uncertain(a);
        assert!(d.split_component(0).is_err());
    }

    #[test]
    fn test_split_component_rejects_out_of_range() {
        let a = small_uncertain("A");
        let d = DiffuseState::from_uncertain(a);
        assert!(d.split_component(7).is_err());
    }

    /// `split_axial_k3_along` must preserve the mixture's total mean
    /// and total covariance exactly for any direction (including
    /// non-eigenvector directions), as long as the resulting
    /// covariance stays positive-definite.
    #[test]
    fn test_split_along_arbitrary_direction_preserves_moments() {
        let st = test_state("A");
        // Mildly anisotropic covariance -- not isotropic, but
        // condition number well below the rank-1-reduction limit so
        // off-axis splits stay PD.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 1.5;
        cov[(1, 1)] = 1.0;
        cov[(2, 2)] = 0.8;
        for i in 3..6 {
            cov[(i, i)] = 1.2;
        }
        let component = UncertainState::new(st.clone(), cov.clone(), vec![]).unwrap();

        // A direction with components in both position and velocity --
        // not aligned with any single eigenvector.
        let mut direction = DVector::<f64>::zeros(6);
        direction[0] = 1.0;
        direction[1] = 0.5;
        direction[3] = 0.7;
        direction[5] = -0.3;

        let parts = split_axial_k3_along(&component, &direction).unwrap();
        let weights: Vec<f64> = parts.iter().map(|(w, _)| *w).collect();
        let comps: Vec<UncertainState> = parts.into_iter().map(|(_, c)| c).collect();
        let mixture = DiffuseState::new(weights, comps).unwrap();

        let m = mixture.mean_state();
        for i in 0..3 {
            assert!(
                (m.pos[i] - st.pos[i]).abs() < 1e-12,
                "pos[{i}] mismatch under arbitrary-direction split"
            );
            assert!(
                (m.vel[i] - st.vel[i]).abs() < 1e-12,
                "vel[{i}] mismatch under arbitrary-direction split"
            );
        }
        let cov_total = mixture.covariance();
        for r in 0..6 {
            for c in 0..6 {
                let diff = (cov_total[(r, c)] - cov[(r, c)]).abs();
                // Absolute + relative tolerance: zero entries are
                // dominated by float roundoff from the rank-1 update,
                // not by relative error.
                let tol = 1e-12 + cov[(r, c)].abs() * 1e-12;
                assert!(
                    diff < tol,
                    "cov[{r},{c}] mismatch under arbitrary-direction split: {} vs {}",
                    cov_total[(r, c)],
                    cov[(r, c)],
                );
            }
        }
    }

    /// A direction with zero norm or wrong dimension is rejected.
    #[test]
    fn test_split_along_validates_direction() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(6, 6);
        let component = UncertainState::new(st, cov, vec![]).unwrap();
        // Zero direction.
        let zero = DVector::<f64>::zeros(6);
        assert!(split_axial_k3_along(&component, &zero).is_err());
        // Wrong length.
        let wrong = DVector::<f64>::from_element(7, 1.0);
        assert!(split_axial_k3_along(&component, &wrong).is_err());
    }

    /// For an isotropic covariance, splitting along any unit
    /// direction produces a positive-definite result (the
    /// generalization that motivated `split_axial_k3_along` in the
    /// first place).
    #[test]
    fn test_split_along_isotropic_is_psd_for_any_direction() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(6, 6) * 0.25;
        let component = UncertainState::new(st, cov, vec![]).unwrap();
        // A handful of arbitrary unit-ish directions, none aligned
        // with the canonical basis.
        let directions = [
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 1.0],
            [0.4, 0.5, -0.3, 0.6, 0.0, -0.4],
            [1e-6, 1e-6, 1e-6, 1.0, 1.0, 1.0],
        ];
        for entries in directions {
            let d = DVector::<f64>::from_iterator(6, entries.iter().copied());
            let parts = split_axial_k3_along(&component, &d)
                .unwrap_or_else(|e| panic!("split failed for {entries:?}: {e}"));
            assert_eq!(parts.len(), 3);
        }
    }
}
