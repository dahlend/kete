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

use crate::frames::InertialFrame;
#[cfg(test)]
use crate::prelude::Desig;
use crate::prelude::{Error, KeteResult, State, TDB, Time, UncertainState};
use nalgebra::{DMatrix, DVector, SymmetricEigen, Vector6};
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
// where w_o = K3_SPLIT_WEIGHTS[0] = K3_SPLIT_WEIGHTS[2],
//       w_c = K3_SPLIT_WEIGHTS[1],
//       d   = K3_SPLIT_MEANS[2] = sqrt(3/2)  (in units of sqrt(lambda)),
//       s   = K3_SPLIT_SIGMA.
//
// Derivation of constants
// -----------------------
// Symmetry handles the odd moments; four constraints then fix the three
// free parameters (w_o, d, s) uniquely:
//
//   (1) Weights sum to one:   2*w_o + w_c = 1
//   (2) Mean is zero:         symmetric by construction
//   (3) Variance is one:      2*w_o*(d^2 + s^2) + w_c*s^2 = 1
//   (4) Fourth moment is 3:   2*w_o*(d^4 + 6*d^2*s^2 + 3*s^4)
//                               + w_c*3*s^4 = 3
//
// Combined with (1), constraint (3) gives s^2 = 1 - 2*w_o*d^2, and (4)
// then selects
//
//   w_o = 1/6,  w_c = 2/3,  d = sqrt(3/2),  s^2 = 1/2
//
// so the mixture reproduces the moments of N(0, 1) through 5th order
// exactly (the 6th is 14.25 against 15).  Splitting is applied
// recursively, so the property that compounds across levels is the
// moments, which is why the 4th-moment constraint is chosen over
// alternatives.  The splitting framework follows the Gaussian-mixture
// splitting literature (e.g. DeMars, Bishop and Jah 2013); note their
// K=3 library minimizes the L^2 distance to the parent instead, which
// gives different values (weights ~ [0.2252, 0.5496, 0.2252], means
// +/- 1.0575, s ~ 0.6716) and does not preserve the 4th moment.
//
// Multivariate extension
// ----------------------
// For a multivariate component N(m, P), a unit direction u selects the
// univariate marginal a = u^T (x - m), with variance sigma^2 = u^T P u.
// Replacing that marginal by the K=3 mixture and keeping the exact
// conditional p(x | a) intact gives, with the regression vector
//
//   r = P u / sigma,
//
//   mean shift:  delta_k = K3_SPLIT_MEANS[k] * r
//   new cov:     P_new = P - (1 - s^2) * r r^T   (rank-1 reduction)
//
// The children slide along the conditional-expectation line E[x | a],
// which is the ridge of the parent density, so the mixture approximates
// the parent with the one-dimensional Huber fidelity for ANY direction u.
// Displacing along u itself instead (delta_k = K3_SPLIT_MEANS[k] * sigma * u,
// reduction on sigma^2 u u^T) agrees only when u is an eigenvector of P;
// for any other direction it pushes the children off the ridge, and the
// gap between siblings grows without bound as u rotates away from the
// eigenframe, even while the total moments stay exact.
//
// When u is an eigenvector, r = sqrt(lambda) u and the two forms coincide.
// The rank-1 reduction exactly cancels the between-component variance
// contributed by the shifted means, so the total mixture mean and covariance
// equal the original -- verifiable via the law of total covariance. P_new is
// positive semi-definite for every u: P - r r^T is the conditional
// covariance of x given a (a Schur complement), and P_new exceeds it by
// s^2 r r^T.

/// Mixture weights for the K=3 univariate split: `[w_outer, w_center, w_outer]`.
/// Moment-matched values: `[1/6, 2/3, 1/6]`.
pub const K3_SPLIT_WEIGHTS: [f64; 3] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];

/// Component mean offsets for the K=3 univariate split, in units of
/// `sqrt(lambda)` along the chosen axis: `[-sqrt(3/2), 0, +sqrt(3/2)]`.
pub const K3_SPLIT_MEANS: [f64; 3] = [-1.224744871391589, 0.0, 1.224744871391589];

/// Per-component standard-deviation scale for the K=3 split.
/// Derived from `s^2 = 1 - 2 * w_outer * d^2 = 1 - 2*(1/6)*(3/2) = 1/2`,
/// so `s = sqrt(1/2) = 1/sqrt(2)`.
pub const K3_SPLIT_SIGMA: f64 = f64::consts::FRAC_1_SQRT_2;

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
/// The K=3 split (see [`K3_SPLIT_WEIGHTS`], [`K3_SPLIT_MEANS`],
/// [`K3_SPLIT_SIGMA`]) replaces one component N(mu, P) with three:
///
///   - two outer components at mu +/- sqrt(3/2) * sqrt(lambda) * v, weight 1/6 each
///   - one central component at mu, weight 2/3
///
/// where v is the unit eigenvector of P along its dominant axis and lambda
/// is the corresponding eigenvalue (the largest variance direction).  Each
/// sub-component gets a narrower covariance: the variance along v shrinks by
/// a factor of `K3_SPLIT_SIGMA`^2 = 1/2, while all other directions are
/// unchanged.  The construction is exact: the weighted mean and total
/// covariance of the three sub-components equal those of the original
/// component (verified by the law of total covariance).
///
/// The split constants are derived from four constraints -- weights sum to 1,
/// the mixture mean equals the original mean, the mixture variance equals the
/// original variance, and the mixture's fourth moment equals the original's --
/// which place the outer means at `+/- sqrt(3/2)` sigma and give
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
/// The divergence score is a Mahalanobis distance -- the linear
/// prediction error normalized by the propagated element covariance:
///
/// ```text
/// d = max_k  sqrt( (delta_full_k - delta_lin_k)^T P_f^-1 (delta_full_k - delta_lin_k) )
/// ```
///
/// where `delta_full_k` is the displacement of the nonlinearly propagated
/// sigma point from the propagated mean, `delta_lin_k` is the STM (linear)
/// prediction of that displacement, and `P_f` is the propagated element
/// covariance, floored per coordinate so near-null directions do not
/// dominate.  `d` answers "how many sigma off is
/// the linear answer, relative to its own predicted uncertainty?"  For
/// samples drawn from the predicted distribution it follows a 6-D chi
/// distribution, whose mean and high-containment quantiles are both of order a
/// few.  A value below the
/// split threshold (`SplitConfig::split_threshold`) means the
/// component may be propagated as a single Gaussian; above it, the banana
/// distortion is significant and the component is split into narrower
/// sub-components first.  In addition to these sigma-shell probes, each
/// dominant axis contributes a fixed-scale pure-position probe
/// (`SplitConfig::position_spacing_au`) that catches nonlinearity
/// localized near the mean, such as a nearby perturbing planet.
///
/// In practice a well-observed main-belt asteroid rarely needs splitting
/// even over months of propagation, while a poorly-constrained near-Earth
/// object on a close-approach trajectory may need several splits to keep
/// the mixture accurate.
///
/// Splitting stops when the total number of components reaches
/// `SplitConfig::max_components` (default 1024), when a component has been
/// split `SplitConfig::max_split_depth` times (default 10), or when a split
/// stops meaningfully reducing the divergence
/// (`SplitConfig::min_split_improvement`).  In each case the component is
/// propagated linearly as-is and its `max_unresolved_divergence` records the
/// divergence at which it settled.  The caps exist to bound runtime; if they
/// are frequently hit, raise the threshold or the caps.
///
/// # Usage
///
/// Use [`DiffuseState::from_uncertain`] to wrap a single [`UncertainState`] and
/// [`DiffuseState::new`] for explicit multi-component construction. Adaptive propagation
/// (repeated split-then-propagate) is handled by
/// `kete_spice::propagation::propagate_diffuse_state_adaptive`.
///
/// # Component independence
///
/// Every component carries its own mean orbit and its own covariance over that orbit's
/// element coordinates. The element storage is a single global coordinate system with no
/// basis rebuilt per point, so components do not need to share one - which is what makes
/// the mixture a plain list rather than a base point and a set of offsets.
///
/// The one thing that does not commute with independence is the **true longitude**, which
/// wraps. Averaging it directly is wrong whenever components straddle the branch cut, and
/// wrong silently. [`Self::mean_and_covariance`] reduces every component to the shortest
/// signed offset from a reference before averaging, which is correct wherever the mixture
/// spans less than half a turn.
#[derive(Debug, Clone)]
pub struct DiffuseState {
    /// Mixture weights. Same length as `components`, non-negative, summing to `1.0`
    /// within [`WEIGHT_SUM_TOL`].
    pub weights: Vec<f64>,

    /// The mixture components, each with its own mean orbit, covariance and free
    /// parameter values. All share an epoch, a central body and a parameter count.
    pub components: Vec<UncertainState>,
}

impl DiffuseState {
    /// Construct from explicit weights and components.
    ///
    /// # Errors
    /// Returns an error if the lengths disagree, the input is empty, the weights are
    /// negative, non-finite or do not sum to `1.0` within [`WEIGHT_SUM_TOL`], or the
    /// components disagree on epoch, central body or parameter count.
    pub fn new(weights: Vec<f64>, components: Vec<UncertainState>) -> KeteResult<Self> {
        let first = components.first().ok_or_else(|| {
            Error::ValueError("DiffuseState must have at least one component".into())
        })?;
        if weights.len() != components.len() {
            return Err(Error::ValueError(format!(
                "weights ({}) and components ({}) must have equal length",
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

        let (epoch, center_id, n_params) = (
            first.elements.epoch,
            first.elements.center_id,
            first.free_params.len(),
        );
        for (i, c) in components.iter().enumerate().skip(1) {
            if c.elements.epoch != epoch {
                return Err(Error::ValueError(format!(
                    "component {i} epoch {} does not match component 0 at {}",
                    c.elements.epoch.jd, epoch.jd
                )));
            }
            if c.elements.center_id != center_id {
                return Err(Error::ValueError(format!(
                    "component {i} center {} does not match component 0 at {center_id}",
                    c.elements.center_id
                )));
            }
            if c.free_params.len() != n_params {
                return Err(Error::ValueError(format!(
                    "component {i} has {} free parameters, expected {n_params}",
                    c.free_params.len()
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
    pub fn from_uncertain(state: UncertainState) -> Self {
        Self {
            weights: vec![1.0],
            components: vec![state],
        }
    }

    /// Component `index`.
    ///
    /// # Errors
    /// Fails if `index` is out of range.
    pub fn component(&self, index: usize) -> KeteResult<UncertainState> {
        self.components.get(index).cloned().ok_or_else(|| {
            Error::ValueError(format!(
                "component {index} out of range, mixture has {}",
                self.components.len()
            ))
        })
    }

    /// Weighted mean of the mixture and its total covariance, both in the element
    /// coordinates of the returned mean.
    ///
    /// The covariance is the law of total covariance,
    /// `P = sum_i w_i (P_i + (m_i - m)(m_i - m)^T)`, with the deviations taken in the
    /// mean's own coordinates. Rows and columns beyond the sixth are the free parameters.
    ///
    /// **The true longitude wraps and is handled explicitly.** Every component is reduced
    /// to its shortest signed offset from component zero through
    /// [`EquinoctialElements::offset_to`](crate::elements::EquinoctialElements::offset_to)
    /// before anything is averaged, and the result is
    /// carried back onto that reference. A plain arithmetic mean of true longitudes is
    /// wrong whenever the components straddle the branch cut - two components at
    /// `L = pi - eps` and `L = -pi + eps` are adjacent on the orbit but average to zero,
    /// half a turn away from both - and it fails without any symptom. Splitting keeps
    /// components close, so the failure would essentially never be triggered by ordinary
    /// use and essentially never be noticed when it was.
    ///
    /// The reduction is correct for any mixture spanning less than half a turn in true
    /// longitude, which every mixture produced by splitting does by a wide margin.
    ///
    /// # Errors
    /// Fails if the mean leaves the elements' physical domain.
    pub fn mean_and_covariance(&self) -> KeteResult<(UncertainState, DMatrix<f64>)> {
        let reference = &self.components[0];
        let n_params = reference.free_params.len();
        let n_dim = 6 + n_params;

        // Offsets from the reference, with the true longitude reduced to the shortest
        // signed angle. Everything downstream is linear in these.
        let offsets: Vec<DVector<f64>> = self
            .components
            .iter()
            .map(|c| {
                let element_offset = reference.elements.offset_to(&c.elements);
                DVector::from_iterator(
                    n_dim,
                    (0..6)
                        .map(|i| element_offset[i])
                        .chain((0..n_params).map(|i| c.free_params[i] - reference.free_params[i])),
                )
            })
            .collect();

        let mut mean_offset = DVector::<f64>::zeros(n_dim);
        for (w, offset) in self.weights.iter().zip(offsets.iter()) {
            mean_offset += offset * *w;
        }

        let mut total = DMatrix::<f64>::zeros(n_dim, n_dim);
        for ((w, offset), component) in self
            .weights
            .iter()
            .zip(offsets.iter())
            .zip(self.components.iter())
        {
            total += &component.cov_matrix * *w;
            let dev = offset - &mean_offset;
            total += &dev * dev.transpose() * *w;
        }

        let step = Vector6::from_iterator(mean_offset.iter().take(6).copied());
        let elements = reference.elements.displaced_by(&step);
        let free_params: Vec<f64> = (0..n_params)
            .map(|i| reference.free_params[i] + mean_offset[6 + i])
            .collect();
        let mean = UncertainState::new(elements, total.clone(), free_params)?;
        Ok((mean, total))
    }

    /// Total weight of components whose peak divergence exceeds `threshold`.
    ///
    /// A component that hit the `max_components` budget before it could be split keeps the
    /// divergence that would have triggered the split, so this answers whether a component
    /// count is convergence or saturation: a large value means the mixture stopped
    /// splitting because it ran out of budget, not because it had resolved the
    /// distribution.
    #[must_use]
    pub fn unresolved_weight(&self, threshold: f64) -> f64 {
        self.components
            .iter()
            .zip(self.weights.iter())
            .filter(|(component, _)| component.max_unresolved_divergence > threshold)
            .map(|(_, weight)| *weight)
            .sum()
    }

    /// Peak sigma-point divergence recorded anywhere in the mixture.
    #[must_use]
    pub fn max_unresolved_divergence(&self) -> f64 {
        self.components
            .iter()
            .map(|c| c.max_unresolved_divergence)
            .fold(0.0_f64, f64::max)
    }

    /// Common epoch of the mixture.
    pub fn epoch(&self) -> Time<TDB> {
        self.components[0].elements.epoch
    }

    /// Free parameter values of the first component, the mixture's nominal set.
    #[must_use]
    pub fn free_params(&self) -> &[f64] {
        &self.components[0].free_params
    }

    /// Number of mixture components.
    #[must_use]
    pub fn n_components(&self) -> usize {
        self.components.len()
    }

    /// Number of free parameters.
    #[must_use]
    pub fn n_params(&self) -> usize {
        self.components[0].free_params.len()
    }

    /// Total covariance dimension, `6 + n_params()`.
    #[must_use]
    pub fn cov_dim(&self) -> usize {
        6 + self.n_params()
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
    pub fn sample<F: InertialFrame>(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<F>, Vec<f64>)>> {
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
        let mut counts = vec![0_usize; self.n_components()];
        for _ in 0..n_samples {
            let u: f64 = StandardUniform.sample(&mut rng);
            let idx = cdf
                .iter()
                .position(|&c| u <= c)
                .unwrap_or(self.n_components() - 1);
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

    /// Recursively K=3 split every component along its dominant covariance
    /// eigenvector, `depth` times.
    ///
    /// Each level multiplies the component count by 3.  Between levels the
    /// dominant eigenvector is recomputed on each child's updated
    /// covariance, so subsequent splits target the next-most-uncertain
    /// direction.  Useful as a static pre-split before propagation when
    /// the initial mixture is too coarse to capture later nonlinearity.
    ///
    /// `depth == 0` returns a clone.
    ///
    /// # Errors
    /// Returns the first error from the underlying axial split -- typically
    /// a degenerate covariance or non-positive variance along the chosen
    /// direction.
    pub fn split_all(&self, depth: u32) -> KeteResult<Self> {
        let mut current_weights = self.weights.clone();
        let mut current_components = self.components.clone();
        for _ in 0..depth {
            let mut next_weights = Vec::with_capacity(current_weights.len() * 3);
            let mut next_components = Vec::with_capacity(current_components.len() * 3);
            for (w, c) in current_weights.iter().zip(current_components.iter()) {
                let dir = dominant_eigenvector(&c.cov_matrix)?;
                let parts = split_axial_k3_along(c, &dir)?;
                for (w_split, c_split) in parts {
                    next_weights.push(w * w_split);
                    next_components.push(c_split);
                }
            }
            current_weights = next_weights;
            current_components = next_components;
        }
        Self::new(current_weights, current_components)
    }

    /// Split component `idx` along its dominant covariance eigenvector into a K=3
    /// sub-mixture, leaving the other components untouched.
    ///
    /// # Errors
    /// Fails if `idx` is out of range, or if the split fails - typically a degenerate
    /// covariance or a non-positive variance along the chosen direction.
    pub fn split_component(&self, idx: usize) -> KeteResult<Self> {
        let target = self.component(idx)?;
        let parts = split_axial_k3_along(&target, &dominant_eigenvector(&target.cov_matrix)?)?;

        let mut weights = Vec::with_capacity(self.n_components() + 2);
        let mut components = Vec::with_capacity(self.n_components() + 2);
        for (i, (w, c)) in self.weights.iter().zip(self.components.iter()).enumerate() {
            if i != idx {
                weights.push(*w);
                components.push(c.clone());
            }
        }
        for (w_split, c_split) in parts {
            weights.push(self.weights[idx] * w_split);
            components.push(c_split);
        }
        Self::new(weights, components)
    }
}

/// Split a component for adaptive propagation.
///
/// Prefers a dynamics-aware direction: finds the dominant eigenvector
/// `v_f` of `prop_cov` (the propagated covariance) and splits the marginal
/// of the functional `a = v_f^T x_f`, whose pullback to the initial epoch
/// is `u = augmented_stm^T v_f` (a covector pulls back through the
/// transpose, not the inverse).  With the regression-form split the
/// children's initial-space displacement is then `P u / sigma`, which the
/// STM carries onto `sqrt(lambda_f) * v_f` at the final epoch -- the
/// propagated mixture is exactly the dominant-eigenvector split of the
/// propagated covariance.
///
/// Falls back to splitting along the dominant eigenvector of the
/// component's own covariance if `prop_cov` has no positive eigenvalue or
/// the mapped direction carries no variance.
///
/// # Errors
/// Returns an error only if both the dynamics-aware split and the fallback
/// fail (e.g. the component has zero covariance).
pub fn split_for_propagation(
    component: &UncertainState,
    prop_cov: &DMatrix<f64>,
    augmented_stm: &DMatrix<f64>,
) -> KeteResult<Vec<(f64, UncertainState)>> {
    if let Ok(v_f) = dominant_eigenvector(prop_cov)
        && let Ok(parts) = split_axial_k3_along(component, &(augmented_stm.transpose() * v_f))
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

/// Split a single [`UncertainState`] into a K=3 sub-mixture of the
/// marginal along an explicitly supplied direction in initial state space.
///
/// `direction` does not need to be a unit vector (it is normalized
/// internally) and does not need to be an eigenvector of the component's
/// covariance: it selects the univariate functional `a = u^T x` whose
/// marginal is replaced by the K=3 mixture, while the conditional of the
/// remaining coordinates given `a` is kept exact.  The children are
/// therefore displaced along the regression vector `P u / sigma` -- the
/// ridge of the parent density -- not along `u` itself, and the mixture
/// approximates the parent with the one-dimensional Huber fidelity for
/// any direction.  Mean and total covariance are preserved exactly, and
/// the reduced covariance is positive semi-definite by construction.
///
/// The motivating use case is dynamics-aware splitting for adaptive
/// cloud propagation: the calling layer computes the dominant
/// eigenvector of the *propagated* covariance and pulls it back through
/// the transpose of the augmented STM, picking out the initial-state
/// functional that most amplifies under the dynamics.  For an isotropic
/// initial covariance the eigenvalue decomposition is degenerate and
/// naive dominant-eigenvector splitting picks an arbitrary axis; this
/// entry point lets callers pick a meaningful one instead.
///
/// # Errors
/// Returns an error if `direction` has the wrong length, is zero
/// (or non-finite), or if the variance of the component along
/// `direction` is non-positive.
fn split_axial_k3_along(
    component: &UncertainState,
    direction: &DVector<f64>,
) -> KeteResult<Vec<(f64, UncertainState)>> {
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

    // Variance of the marginal along d.
    let sigma_sq_mat = d.transpose() * &component.cov_matrix * &d;
    let sigma_sq = sigma_sq_mat[(0, 0)];
    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return Err(Error::ValueError(
            "split_axial_k3_along: variance along the requested direction is non-positive".into(),
        ));
    }
    let sigma = sigma_sq.sqrt();

    // Regression vector r = P d / sigma: the conditional-expectation line of
    // the parent, along which the children slide.  Rank-1 reduction:
    //
    //   P_new = P - (1 - K3_SPLIT_SIGMA^2) * r r^T
    //
    // For any direction d this preserves the total mean and covariance (the
    // reduction is exactly the between-component variance contributed by the
    // K=3 means), and P_new >= P - r r^T, the conditional covariance of the
    // parent given the marginal -- a Schur complement, so PSD always.
    let r = (&component.cov_matrix * &d) / sigma;
    let alpha = 1.0 - K3_SPLIT_SIGMA.powi(2);
    let outer = &r * r.transpose();
    let mut cov_new = component.cov_matrix.clone();
    cov_new -= outer * alpha;
    // Force exact symmetry, and clip the floating-point noise that the
    // subtraction can leave at the ~1e-16 relative level on eigenvalues
    // that are exactly zero in real arithmetic.
    let cov_new = (&cov_new + cov_new.transpose()) * 0.5;
    let sym = SymmetricEigen::new(cov_new.clone());
    let min_eig = sym
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let cov_new = if min_eig < 0.0 {
        let lambda_clipped: Vec<f64> = sym.eigenvalues.iter().map(|&e| e.max(0.0)).collect();
        let lambda_diag = DMatrix::from_diagonal(&DVector::from_vec(lambda_clipped));
        let rebuilt = &sym.eigenvectors * lambda_diag * sym.eigenvectors.transpose();
        (&rebuilt + rebuilt.transpose()) * 0.5
    } else {
        cov_new
    };

    let mut result = Vec::with_capacity(3);
    for k in 0..3 {
        let delta = &r * K3_SPLIT_MEANS[k];
        let new_uncertain = build_split_component(component, &delta, cov_new.clone())?;
        result.push((K3_SPLIT_WEIGHTS[k], new_uncertain));
    }
    Ok(result)
}

/// Build a new [`UncertainState`] by shifting `base`'s mean by `delta`
/// in the augmented `(6 + Np)` space and replacing its covariance.
fn build_split_component(
    base: &UncertainState,
    delta: &DVector<f64>,
    new_cov: DMatrix<f64>,
) -> KeteResult<UncertainState> {
    let np = base.free_params.len();

    // The first six entries are element coordinates, so the child's mean is placed by
    // displacing the stored floats. The child is then an exact orbit, which matters here
    // because splitting deliberately places components far enough apart that a linear
    // cartesian offset would not describe one.
    let step = Vector6::from_iterator(delta.iter().take(6).copied());
    let new_elements = base.elements.displaced_by(&step);

    let new_params: Vec<f64> = (0..np)
        .map(|i| base.free_params[i] + delta[6 + i])
        .collect();

    let mut new_uncertain = UncertainState::new(new_elements, new_cov, new_params)?;
    // Children inherit the parent's accumulated linear-approximation
    // history.  Their own future diagnoses will update the field
    // independently as they evolve.
    new_uncertain.max_unresolved_divergence = base.max_unresolved_divergence;
    Ok(new_uncertain)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::EquinoctialElements;
    use crate::frames::Equatorial;
    use nalgebra::Vector3;

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
        UncertainState::from_state(&test_state(desig), &cov, vec![]).unwrap()
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
        assert!(DiffuseState::new(vec![], vec![]).is_err());
        // Valid.
        assert!(DiffuseState::new(vec![0.5, 0.5], comps.clone()).is_ok());
    }

    #[test]
    fn test_new_validates_epoch_and_center() {
        let mut a = small_uncertain("A");
        let b = small_uncertain("B");

        // Different epoch.
        a.elements.epoch = 2451600.0.into();
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![a.clone(), b.clone()]).is_err());

        // Different center.
        let mut a = small_uncertain("A");
        let mut b = small_uncertain("B");
        // The center now lives on the elements rather than on a cartesian state, so the
        // mismatch is made by relabeling the element center directly.
        b.elements.center_id = 0;
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
        let plain = UncertainState::from_state(&st.clone(), &cov_6.clone(), vec![]).unwrap();
        // Component with one free param -> 7x7 cov.
        let with_param = UncertainState::from_state(&st.clone(), &cov_7, vec![0.01]).unwrap();

        // Mixing 6x6 and 7x7 cov dims is invalid.
        assert!(DiffuseState::new(vec![0.5, 0.5], vec![plain, with_param.clone()]).is_err());

        // Two components both with one free param (different values) are valid.
        let with_param2 =
            UncertainState::from_state(&st, &(DMatrix::<f64>::identity(7, 7) * 1e-12), vec![0.05])
                .unwrap();
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
    fn test_mean_collapses_for_single_component() {
        let u = small_uncertain("A");
        let d = DiffuseState::from_uncertain(u.clone());
        let (mean, cov) = d.mean_and_covariance().unwrap();
        // One component is its own mean, so nothing moves and the covariance is its own.
        let offset = u.elements.offset_to(&mean.elements);
        for i in 0..6 {
            assert!(offset[i].abs() < 1e-14, "mean moved in coordinate {i}");
        }
        assert!((cov - u.cov_matrix).norm() < 1e-30);
    }

    #[test]
    fn test_mean_state_two_component() {
        // Both components must be real orbits: the mean is taken over states reconstructed
        // from elements, and the zero state the cartesian version of this test used has no
        // element representation at all.
        let cov = DMatrix::identity(6, 6) * 1e-12;
        let state_a = State::<Equatorial>::new(
            Desig::Name("A".into()),
            2451545.0,
            [1.0, 0.0, 0.0],
            [0.0, 0.01720209895, 0.0],
            10,
        );
        let state_b = State::<Equatorial>::new(
            Desig::Name("B".into()),
            2451545.0,
            [2.0, 0.0, 0.0],
            [0.0, 0.01216622, 0.0],
            10,
        );
        let a = UncertainState::from_state(&state_a, &cov.clone(), vec![]).unwrap();
        let b = UncertainState::from_state(&state_b, &cov, vec![]).unwrap();
        let reference = a.elements.clone();
        let offset_b = reference.offset_to(&b.elements);
        let d = DiffuseState::new(vec![0.25, 0.75], vec![a, b]).unwrap();
        let (mean, _) = d.mean_and_covariance().unwrap();

        // The weighted mean is taken in element coordinates, where it is linear by
        // construction, and is carried back onto the reference component.
        let got = reference.offset_to(&mean.elements);
        for i in 0..6 {
            let expect = 0.75 * offset_b[i];
            assert!((got[i] - expect).abs() < 1e-12, "mean coordinate {i}");
        }
    }

    /// The true longitude wraps, and the mixture moments must reduce it before averaging.
    ///
    /// Two components placed either side of the branch cut are a fifth of a radian apart
    /// on the orbit. A plain arithmetic mean of their stored true longitudes puts the
    /// answer half a turn away from both, and nothing about the result announces that it
    /// is wrong. This is the one silent failure mode independent component means
    /// introduce, and splitting keeps components close enough that ordinary use would
    /// essentially never trigger it.
    ///
    /// Checked against the states rather than against the coordinates: the mean orbit's
    /// position must sit between the two components' positions, which is a fact about the
    /// geometry and not about the arithmetic being tested.
    #[test]
    fn mixture_mean_reduces_the_wrapped_true_longitude() {
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-14;
        let base = UncertainState::from_state(&test_state("A"), &cov, vec![]).unwrap();

        // Straddle the branch cut: one component just below `pi`, one just above, which
        // `from_state` would store as just above `-pi`.
        let mut low = base.clone();
        low.elements.true_lon = std::f64::consts::PI - 0.1;
        let mut high = base.clone();
        high.elements.true_lon = -std::f64::consts::PI + 0.1;
        assert!(
            (low.elements.true_lon - high.elements.true_lon).abs() > 6.0,
            "the stored longitudes must actually straddle the cut"
        );

        let mixture = DiffuseState::new(vec![0.5, 0.5], vec![low.clone(), high.clone()]).unwrap();
        let (mean, _) = mixture.mean_and_covariance().unwrap();

        // The two components are 0.2 rad apart on the orbit, so the mean sits at `pi`
        // exactly, halfway between them.
        let offset = low.elements.offset_to(&mean.elements);
        assert!(
            (offset[5] - 0.1).abs() < 1e-14,
            "the mean sits {} rad from the low component, expected 0.1",
            offset[5]
        );

        // The geometric statement, independent of the element arithmetic: the mean orbit's
        // position lies between the two components' positions, so it is closer to each of
        // them than they are to each other.
        let pos = |u: &UncertainState| -> Vector3<f64> {
            Vector3::from(u.state::<Equatorial>().unwrap().pos)
        };
        let (low_pos, high_pos, mean_pos) = (pos(&low), pos(&high), pos(&mean));
        let separation = (low_pos - high_pos).norm();
        assert!((mean_pos - low_pos).norm() < separation);
        assert!((mean_pos - high_pos).norm() < separation);

        // What a plain arithmetic mean of the stored floats would have produced: half a
        // turn away, on the far side of the orbit. Stated as a number so the failure this
        // guards against is on the record rather than described.
        let naive = 0.5 * (low.elements.true_lon + high.elements.true_lon);
        let mut naive_elements = base.elements.clone();
        naive_elements.true_lon = naive;
        let naive_pos = Vector3::from(
            naive_elements
                .try_to_state()
                .unwrap()
                .into_frame::<Equatorial>()
                .pos,
        );
        println!(
            "wrapped mean {:.6} rad, naive mean {naive:.6} rad, naive position error {:.4} AU",
            mean.elements.true_lon,
            (naive_pos - mean_pos).norm()
        );
        assert!(
            (naive_pos - mean_pos).norm() > 1.0,
            "the naive mean must be visibly wrong, or this test proves nothing"
        );
    }

    #[test]
    fn test_covariance_law_of_total_variance() {
        // Two components offset along x with identical small spherical
        // covariance.  Total covariance should equal individual P plus
        // the between-component spread along x.
        let st = test_state("A");
        let p = DMatrix::<f64>::identity(6, 6) * 1e-6;

        // Separated along track rather than by a whole AU: the components must stay
        // within half a revolution of each other for the true longitude offset to place them
        // unambiguously, and the cartesian version of this test put them on opposite
        // sides of the orbit, which is exactly the ambiguous case.
        let mut a_state = st.clone();
        a_state.pos = [1.0, -0.02, 0.0].into();
        let mut b_state = st;
        b_state.pos = [1.0, 0.02, 0.0].into();

        let a = UncertainState::from_state(&a_state, &p.clone(), vec![]).unwrap();
        let b = UncertainState::from_state(&b_state, &p.clone(), vec![]).unwrap();
        let reference = a.elements.clone();
        // Each component carries its own covariance; the within-term of the law of total
        // covariance is the weighted sum of those, not one of them twice.
        let within_a = a.cov_matrix.clone();
        let within_b = b.cov_matrix.clone();
        let offset_a = reference.offset_to(&a.elements);
        let offset_b = reference.offset_to(&b.elements);
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();

        let (_, cov) = d.mean_and_covariance().unwrap();

        // The identity itself, not a hard-coded number: total = within + between, with
        // the between term formed from where the components actually sit in element coordinates.
        let mean: Vec<f64> = (0..6).map(|i| 0.5 * (offset_a[i] + offset_b[i])).collect();
        for r in 0..6 {
            for c in 0..6 {
                let between = 0.5 * (offset_a[r] - mean[r]) * (offset_a[c] - mean[c])
                    + 0.5 * (offset_b[r] - mean[r]) * (offset_b[c] - mean[c]);
                let within = 0.5 * within_a[(r, c)] + 0.5 * within_b[(r, c)];
                let expect = within + between;
                let scale = (0.5 * (within_a[(r, r)] + within_b[(r, r)])).sqrt()
                    * (0.5 * (within_a[(c, c)] + within_b[(c, c)])).sqrt()
                    + between.abs();
                assert!(
                    (cov[(r, c)] - expect).abs() / scale.max(1e-30) < 1e-9,
                    "cov[{r},{c}]: {} vs {expect}",
                    cov[(r, c)]
                );
            }
        }
    }

    #[test]
    fn test_mean_params_with_free_params() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(7, 7) * 1e-12;
        let a = UncertainState::from_state(&st.clone(), &cov.clone(), vec![0.01]).unwrap();
        let b = UncertainState::from_state(&st, &cov, vec![0.05]).unwrap();
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();
        let (mean, _) = d.mean_and_covariance().unwrap();
        assert_eq!(mean.free_params.len(), 1);
        assert!((mean.free_params[0] - 0.03).abs() < 1e-15);
    }

    #[test]
    fn test_sample_count_and_distribution() {
        // A 90/10 mixture should yield ~90% of samples from the first component. The two
        // are separated along track by a timing offset, which is far enough apart to tell
        // them apart from a sampled position.
        let a = small_uncertain("A");
        let base = a.elements.clone();
        let mut step = Vector6::zeros();
        step[5] = 0.35;
        let mut shifted = a.clone();
        shifted.elements = base.displaced_by(&step);

        let d = DiffuseState::new(vec![0.9, 0.1], vec![a.clone(), shifted]).unwrap();

        let far = base.displaced_by(&step).try_to_state().unwrap();
        let near = base.try_to_state().unwrap();
        let midpoint = 0.5 * (near.pos[1] + far.pos[1]);

        let samples: Vec<(State<Equatorial>, Vec<f64>)> = d.sample(1000, Some(7)).unwrap();
        assert_eq!(samples.len(), 1000);
        let n_a = samples
            .iter()
            .filter(|(s, _)| (s.pos[1] - near.pos[1]).abs() < (midpoint - near.pos[1]).abs())
            .count();
        assert!(n_a > 850 && n_a < 950, "got n_a = {n_a}");
    }

    #[test]
    fn test_sample_seed_is_deterministic() {
        let a = small_uncertain("A");
        let b = small_uncertain("B");
        let d = DiffuseState::new(vec![0.5, 0.5], vec![a, b]).unwrap();
        let s1: Vec<(State<Equatorial>, Vec<f64>)> = d.sample(20, Some(42)).unwrap();
        let s2: Vec<(State<Equatorial>, Vec<f64>)> = d.sample(20, Some(42)).unwrap();
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            for i in 0..3 {
                assert_eq!(a.0.pos[i], b.0.pos[i]);
                assert_eq!(a.0.vel[i], b.0.vel[i]);
            }
        }
    }

    /// The K=3 split tables must satisfy the moment-preservation
    /// constraints for `N(0,1) -> sum_i w_i N(m_i, sigma^2)`, through the
    /// fourth moment -- the constraint that selects these constants over
    /// the L^2-optimal library.
    #[test]
    fn test_k3_split_constants_preserve_moments() {
        let sum: f64 = K3_SPLIT_WEIGHTS.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-15,
            "weights must sum to 1: got {sum}"
        );
        let mean: f64 = K3_SPLIT_WEIGHTS
            .iter()
            .zip(K3_SPLIT_MEANS.iter())
            .map(|(w, m)| w * m)
            .sum();
        assert!(mean.abs() < 1e-15, "mean must be 0: got {mean}");
        let variance: f64 = K3_SPLIT_WEIGHTS
            .iter()
            .zip(K3_SPLIT_MEANS.iter())
            .map(|(w, m)| w * (m * m + K3_SPLIT_SIGMA * K3_SPLIT_SIGMA))
            .sum();
        assert!(
            (variance - 1.0).abs() < 1e-15,
            "variance must be 1: got {variance}"
        );
        // E[a^4] of a mixture of N(m, s^2) components is
        // sum_i w_i (m^4 + 6 m^2 s^2 + 3 s^4); N(0,1) has 3.
        let s_sq = K3_SPLIT_SIGMA * K3_SPLIT_SIGMA;
        let fourth: f64 = K3_SPLIT_WEIGHTS
            .iter()
            .zip(K3_SPLIT_MEANS.iter())
            .map(|(w, m)| {
                let m_sq = m * m;
                w * (m_sq * m_sq + 6.0 * m_sq * s_sq + 3.0 * s_sq * s_sq)
            })
            .sum();
        assert!(
            (fourth - 3.0).abs() < 1e-15,
            "fourth moment must be 3: got {fourth}"
        );
    }

    /// Splitting a component preserves the total weight (with K=3
    /// sub-weights summing to the original component's weight) and
    /// produces 2 additional components (1 -> 3).
    #[test]
    fn test_split_component_basic() {
        let mut a = small_uncertain("A");
        // Inflate covariance so the dominant eigenvalue is well-defined and large enough
        // that the split delta is non-trivial - but keep it physical. The cartesian
        // version of this test used a 1 AU standard deviation on a 1 AU orbit, which is a
        // 100 percent positional uncertainty; harmless as arithmetic, but a one sigma step
        // in element coordinates drives the orbit outside its own domain and is rejected.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 1e-6;
        cov[(1, 1)] = 1e-10;
        cov[(2, 2)] = 1e-10;
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        a.cov_matrix = cov;

        let d = DiffuseState::from_uncertain(a);
        let split = d.split_component(0).unwrap();
        assert_eq!(split.n_components(), 3);
        let total_w: f64 = split.weights.iter().sum();
        assert!((total_w - 1.0).abs() < 1e-15);

        // Each sub-weight should equal K3_SPLIT_WEIGHTS[k] (since
        // original weight was 1.0).
        for (got, &want) in split.weights.iter().zip(K3_SPLIT_WEIGHTS.iter()) {
            assert!((got - want).abs() < 1e-15);
        }
    }

    /// Splitting must preserve the mixture's mean and total covariance
    /// (law of total covariance).  The dominant axis variance shrinks
    /// per-component but the BETWEEN-component spread compensates.
    #[test]
    fn test_split_component_preserves_moments() {
        let st = test_state("A");
        // Scaled to a physical orbit uncertainty. The cartesian version of this test used
        // AU-scale position sigmas and velocity sigmas several times the orbital speed;
        // harmless as arithmetic, but not a distribution any orbit representation
        // describes.
        // Anisotropic covariance -- dominant axis is x.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 4e-8;
        cov[(1, 1)] = 1e-8;
        cov[(2, 2)] = 1e-8;
        for i in 3..6 {
            cov[(i, i)] = 1e-14;
        }
        let a = UncertainState::from_state(&st.clone(), &cov.clone(), vec![]).unwrap();
        let element_cov = a.cov_matrix.clone();
        let parent_elements = a.elements.clone();
        let d = DiffuseState::from_uncertain(a);

        let split = d.split_component(0).unwrap();
        let (split_mean, split_cov) = split.mean_and_covariance().unwrap();

        // The moments are asserted **in element coordinates**, which is where the K3
        // construction does its arithmetic. The cartesian weighted mean of the children is
        // deliberately not preserved: the map from elements to a state is nonlinear, and
        // that curvature is the whole reason the element representation exists. Requiring
        // it here would be requiring the mixture to be a worse description than it is.
        //
        // Displacement and differencing are exact vector arithmetic on the stored floats,
        // so the mean returns to the parent at rounding rather than at any solver floor.
        let moved = parent_elements.offset_to(&split_mean.elements);
        for i in 0..6 {
            assert!(
                moved[i].abs() < 1e-14 * element_cov[(i, i)].sqrt().max(1e-12),
                "element mean {i} moved to {}",
                moved[i]
            );
        }
        for r in 0..6 {
            for c in 0..6 {
                let diff = (split_cov[(r, c)] - element_cov[(r, c)]).abs();
                let scale = element_cov[(r, r)].sqrt() * element_cov[(c, c)].sqrt();
                assert!(
                    diff / scale < 1e-8,
                    "cov[{r},{c}] changed: {} vs {}, rel_err={}",
                    split_cov[(r, c)],
                    element_cov[(r, c)],
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

        let a = UncertainState::from_state(&st, &cov, vec![0.01]).unwrap();
        let d = DiffuseState::from_uncertain(a);

        let split = d.split_component(0).unwrap();
        assert_eq!(split.n_components(), 3);

        // Each sub-component should have a different free-param value
        // (sub means shifted by K3_SPLIT_MEANS[k] * sqrt(1e-4) from the original 0.01).
        let offset = K3_SPLIT_MEANS[2] * 1e-4_f64.sqrt(); // sqrt(3/2) * 0.01
        let mut params: Vec<f64> = (0..split.n_components())
            .map(|i| split.component(i).unwrap().free_params[0])
            .collect();
        params.sort_by(f64::total_cmp);
        assert!((params[0] - (0.01 - offset)).abs() < 1e-10);
        assert!((params[1] - 0.01).abs() < 1e-10);
        assert!((params[2] - (0.01 + offset)).abs() < 1e-10);
    }

    #[test]
    fn test_split_component_rejects_zero_covariance() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::zeros(6, 6);
        let a = UncertainState::from_state(&st, &cov, vec![]).unwrap();
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
        // Built in element coordinates, where the split arithmetic lives.
        let elements = EquinoctialElements::from_state(&st.into_frame()).unwrap();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 1.5e-8;
        cov[(1, 1)] = 1.0e-8;
        cov[(2, 2)] = 0.8e-8;
        for i in 3..6 {
            cov[(i, i)] = 1.2e-8;
        }
        let component = UncertainState::new(elements, cov.clone(), vec![]).unwrap();

        // A direction with components in both position and velocity --
        // not aligned with any single eigenvector.
        let mut direction = DVector::<f64>::zeros(6);
        direction[0] = 1.0;
        direction[1] = 0.5;
        direction[3] = 0.7;
        direction[5] = -0.3;

        let base = component.elements.clone();
        let parts = split_axial_k3_along(&component, &direction).unwrap();
        let weights: Vec<f64> = parts.iter().map(|(w, _)| *w).collect();
        let comps: Vec<UncertainState> = parts.into_iter().map(|(_, c)| c).collect();
        let mixture = DiffuseState::new(weights, comps).unwrap();

        // Asserted in element coordinates, where the split arithmetic lives. See
        // `test_split_component_preserves_moments` for why the cartesian mean is not the
        // invariant here.
        let element_cov = component.cov_matrix.clone();
        let (mean, cov_total) = mixture.mean_and_covariance().unwrap();
        let m = base.offset_to(&mean.elements);
        for i in 0..6 {
            assert!(
                m[i].abs() < 1e-14 * element_cov[(i, i)].sqrt().max(1e-12),
                "element mean {i} moved under arbitrary-direction split"
            );
        }
        for r in 0..6 {
            for c in 0..6 {
                let diff = (cov_total[(r, c)] - element_cov[(r, c)]).abs();
                let scale = element_cov[(r, r)].sqrt() * element_cov[(c, c)].sqrt();
                // The offsets are plain differences of the stored floats, so the only
                // error here is the rounding of the rank-1 reduction and the outer
                // products that undo it.
                assert!(
                    diff / scale.max(1e-30) < 1e-10,
                    "cov[{r},{c}] mismatch under arbitrary-direction split: {} vs {}",
                    cov_total[(r, c)],
                    element_cov[(r, c)],
                );
            }
        }
    }

    /// Children of a split must sit on the parent's ridge, whatever
    /// direction is asked for: the Mahalanobis distance between adjacent
    /// siblings, measured in the child covariance, equals the univariate
    /// Huber spacing `d / s = sqrt(3)` for every direction.  The
    /// direction-along displacement form failed this off the eigenframe --
    /// moments stayed exact while siblings drifted arbitrarily far apart
    /// in probability, rendering as separate blobs.
    #[test]
    fn test_split_children_stay_on_the_ridge_for_any_direction() {
        let st = test_state("A");
        let elements = EquinoctialElements::from_state(&st.into_frame()).unwrap();
        // Strongly anisotropic and correlated, like a sheared cloud.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        let scales = [1.0e-6, 1.0e-8, 3.0e-9, 1.0e-9, 4.0e-10, 2.0e-10];
        for (i, s) in scales.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        cov[(0, 1)] = 0.9 * scales[0] * scales[1];
        cov[(1, 0)] = cov[(0, 1)];
        cov[(2, 3)] = -0.5 * scales[2] * scales[3];
        cov[(3, 2)] = cov[(2, 3)];
        let component = UncertainState::new(elements, cov, vec![]).unwrap();

        let directions: [[f64; 6]; 4] = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], // far off the dominant eigenvector
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.1, -0.4, 1.0, 0.0, -1.0, 0.3],
        ];
        for dir in directions {
            let d = DVector::from_row_slice(&dir);
            let parts = split_axial_k3_along(&component, &d).unwrap();
            let child_cov = parts[0].1.cov_matrix.clone();
            let base = parts[1].1.elements.clone();
            let delta_vec = base.offset_to(&parts[2].1.elements);
            let delta = DVector::from_row_slice(delta_vec.as_slice());
            let solved = child_cov
                .clone()
                .svd(true, true)
                .solve(&delta, 1e-30)
                .unwrap();
            let mahal = (delta.transpose() * solved)[(0, 0)].sqrt();
            assert!(
                (mahal - 3.0_f64.sqrt()).abs() < 1e-6,
                "sibling separation {mahal} != sqrt(3) for direction {dir:?}"
            );
        }
    }

    /// A direction with zero norm or wrong dimension is rejected.
    #[test]
    fn test_split_along_validates_direction() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(6, 6);
        let component = UncertainState::from_state(&st, &cov, vec![]).unwrap();
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
        // Isotropy is basis dependent, and the split operates in element coordinates, so the
        // covariance is built there directly. Converting an isotropic *cartesian*
        // covariance would hand the splitter something with a condition number of order
        // cond(K)^2, and testing direction independence against that measures the change
        // of basis rather than the splitter.
        let elements = EquinoctialElements::from_state(&st.into_frame()).unwrap();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let component = UncertainState::new(elements, cov, vec![]).unwrap();
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
