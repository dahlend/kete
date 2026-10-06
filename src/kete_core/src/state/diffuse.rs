// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Weighted mixture of [`UncertainState`] components.
//!
//! See [`DiffuseState`] for a full description of the model and when to use it.

use core::f64;

use crate::forces::{NonGravMask, ParameterizedForce};
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

/// Univariate splitting libraries, one per component count `k`.
///
/// Each entry holds the weights and the child width `s`, in units of the parent's
/// width along the split axis.  The means are not stored: they are `spacing * j` on
/// the symmetric integer grid `j`, with `spacing` set by the variance constraint
/// `sum_j w_j m_j^2 = 1 - s^2`, so that constraint is satisfied in arithmetic rather
/// than to the precision a stored decimal would carry.
///
/// Every entry preserves the mean, the variance and the fourth moment, and every
/// entry has the same integral square difference to the parent as the `k = 3` one,
/// so `k` buys narrowing rather than accuracy: `1/s` runs from 1.41 at `k = 3` to
/// 8.24 at `k = 21`.
///
/// That equality is in the integral, and in the worst absolute density error, which
/// stays near five percent of the parent's peak at every entry.  It does not hold
/// relative to the local density out in the wings, where a larger library is worse:
/// the fractional error within two sigma stays under about 0.14 at every `k`, while
/// within three sigma it grows from 0.11 at `k = 3` to 1.7 at `k = 21`.  Work that
/// reads a low-probability tail directly, rather than the bulk, should keep that in
/// mind and prefer the approach in [`DiffuseState`]'s own note on tails - build a
/// mixture over the region of interest and propagate that.
///
/// That is the reason the table exists.  Narrowing an axis by a factor `F` through
/// repeated `k = 3` splits takes `2 log2(F)` generations and `F^(log2 9)` - about
/// `F^3.17` - components, because each generation triples the count and narrows by
/// only `sqrt(2)`.  One split from this table costs about `2.4 F` components for the
/// same `F`, so refinement deep enough to resolve an encounter is affordable in one
/// step and not in a cascade.
///
/// `k = 2` and `k = 4` are absent, and not by omission: the fourth moment cannot be
/// matched on a uniform symmetric grid of either size.
///
/// Generated by `analysis/adaptive_splitting/split_libraries.py`, which also states
/// the construction and checks it against the `k = 3` entry.
const SPLIT_LIBRARIES: [(&[f64], f64); 18] = [
    // K = 3, narrowing 1.41
    (&K3_SPLIT_WEIGHTS, K3_SPLIT_SIGMA),
    // K = 5, narrowing 1.87
    (
        &[
            0.03469857999053662,
            0.2208388585132312,
            0.4889251229924643,
            0.2208388585132312,
            0.03469857999053662,
        ],
        0.5345052337646484,
    ),
    // K = 6, narrowing 2.30
    (
        &[
            0.026714147026147803,
            0.12229142208860469,
            0.3509944308852475,
            0.3509944308852475,
            0.12229142208860469,
            0.026714147026147803,
        ],
        0.43508239746093746,
    ),
    // K = 7, narrowing 2.71
    (
        &[
            0.022469049360244426,
            0.07365830606043001,
            0.23754674799297953,
            0.33265179317269206,
            0.23754674799297953,
            0.07365830606043001,
            0.022469049360244426,
        ],
        0.3685425567626954,
    ),
    // K = 8, narrowing 3.12
    (
        &[
            0.019411963108674695,
            0.048162800483565686,
            0.16064669300617684,
            0.27177854340158275,
            0.27177854340158275,
            0.16064669300617684,
            0.048162800483565686,
            0.019411963108674695,
        ],
        0.32066856384277354,
    ),
    // K = 9, narrowing 3.52
    (
        &[
            0.017198287703702596,
            0.03353084641060711,
            0.1116646136187615,
            0.2088875431193186,
            0.2574374182952204,
            0.2088875431193186,
            0.1116646136187615,
            0.03353084641060711,
            0.017198287703702596,
        ],
        0.28407333374023447,
    ),
    // K = 10, narrowing 3.92
    (
        &[
            0.015492850006378643,
            0.024603670648968456,
            0.08006089392120487,
            0.15817843946722657,
            0.2216641459562215,
            0.2216641459562215,
            0.15817843946722657,
            0.08006089392120487,
            0.024603670648968456,
            0.015492850006378643,
        ],
        0.2551380920410157,
    ),
    // K = 11, narrowing 4.32
    (
        &[
            0.014134768685381879,
            0.018855098925113056,
            0.059200647443918876,
            0.12012726288063653,
            0.18281594166042642,
            0.20973256080904631,
            0.18281594166042642,
            0.12012726288063653,
            0.059200647443918876,
            0.018855098925113056,
            0.014134768685381879,
        ],
        0.23164596557617187,
    ),
    // K = 12, narrowing 4.71
    (
        &[
            0.013019823391561245,
            0.014989477453524625,
            0.04503163650304095,
            0.0922592173075276,
            0.1480486692376637,
            0.18665117610668197,
            0.18665117610668197,
            0.1480486692376637,
            0.0922592173075276,
            0.04503163650304095,
            0.014989477453524625,
            0.013019823391561245,
        ],
        0.21218383789062503,
    ),
    // K = 13, narrowing 5.11
    (
        &[
            0.012085636831084259,
            0.012291717043814186,
            0.03514249243815987,
            0.0718784591978765,
            0.11933334017676553,
            0.16063199030845954,
            0.17727272800768026,
            0.16063199030845954,
            0.11933334017676553,
            0.0718784591978765,
            0.03514249243815987,
            0.012291717043814186,
            0.012085636831084259,
        ],
        0.19577972412109373,
    ),
    // K = 14, narrowing 5.50
    (
        &[
            0.011288584127044678,
            0.010347978890120721,
            0.02806192721325536,
            0.05686084885495462,
            0.09641874677125384,
            0.13590254776194968,
            0.16111936638142105,
            0.16111936638142105,
            0.13590254776194968,
            0.09641874677125384,
            0.05686084885495462,
            0.02806192721325536,
            0.010347978890120721,
            0.011288584127044678,
        ],
        0.1817607116699219,
    ),
    // K = 15, narrowing 5.89
    (
        &[
            0.01059887652945876,
            0.008907936188004936,
            0.022873811285433792,
            0.04566442963370251,
            0.07838705138467529,
            0.11412703507945109,
            0.14264639727990663,
            0.15358892523873402,
            0.14264639727990663,
            0.11412703507945109,
            0.07838705138467529,
            0.04566442963370251,
            0.022873811285433792,
            0.008907936188004936,
            0.01059887652945876,
        ],
        0.16963706970214848,
    ),
    // K = 16, narrowing 6.29
    (
        &[
            0.009994143733945892,
            0.007814153370966248,
            0.018991752809043787,
            0.03720541584953333,
            0.06424341833864375,
            0.095682622075124,
            0.12438648082614415,
            0.14168201299659883,
            0.14168201299659883,
            0.12438648082614415,
            0.095682622075124,
            0.06424341833864375,
            0.03720541584953333,
            0.018991752809043787,
            0.007814153370966248,
            0.009994143733945892,
        ],
        0.15904991149902348,
    ),
    // K = 17, narrowing 6.68
    (
        &[
            0.009460151930618924,
            0.006965442396766034,
            0.01603232918476875,
            0.030728198216529726,
            0.053123187378057726,
            0.08036487693722205,
            0.10758708257030046,
            0.12796308142687188,
            0.13555129991772893,
            0.12796308142687188,
            0.10758708257030046,
            0.08036487693722205,
            0.053123187378057726,
            0.030728198216529726,
            0.01603232918476875,
            0.006965442396766034,
            0.009460151930618924,
        ],
        0.14971511840820315,
    ),
    // K = 18, narrowing 7.07
    (
        &[
            0.00898416195387232,
            0.00629364908565827,
            0.01373740768513301,
            0.02570253230864754,
            0.044332174248119874,
            0.06776237397956215,
            0.09272915699818218,
            0.11404677599221673,
            0.12641176774860796,
            0.12641176774860796,
            0.11404677599221673,
            0.09272915699818218,
            0.06776237397956215,
            0.044332174248119874,
            0.02570253230864754,
            0.01373740768513301,
            0.00629364908565827,
            0.00898416195387232,
        ],
        0.14142333984375,
    ),
    // K = 19, narrowing 7.46
    (
        &[
            0.00855555688832494,
            0.005751803139369667,
            0.011929651198431404,
            0.021753258489383986,
            0.03733284290549829,
            0.057427136680484304,
            0.07988094757518541,
            0.10083781377538231,
            0.11586292051092385,
            0.12133613767403165,
            0.11586292051092385,
            0.10083781377538231,
            0.07988094757518541,
            0.057427136680484304,
            0.03733284290549829,
            0.021753258489383986,
            0.011929651198431404,
            0.005751803139369667,
            0.00855555688832494,
        ],
        0.13401382446289065,
    ),
    // K = 20, narrowing 7.85
    (
        &[
            0.008168778415109894,
            0.0053082225744307855,
            0.010486074459076882,
            0.01861358899869073,
            0.031717016673574974,
            0.04894804164533922,
            0.06891020783333011,
            0.08877342028439454,
            0.10496987348443193,
            0.11410477563162097,
            0.11410477563162097,
            0.10496987348443193,
            0.08877342028439454,
            0.06891020783333011,
            0.04894804164533922,
            0.031717016673574974,
            0.01861358899869073,
            0.010486074459076882,
            0.0053082225744307855,
            0.008168778415109894,
        ],
        0.12734451293945312,
    ),
    // K = 21, narrowing 8.24
    (
        &[
            0.00781742216007927,
            0.00493957255797647,
            0.00931830970905999,
            0.016089670921338905,
            0.027174690655451306,
            0.0419734975689527,
            0.05960533346731668,
            0.07801105242393701,
            0.09438960265400907,
            0.10576138485669305,
            0.109838926050371,
            0.10576138485669305,
            0.09438960265400907,
            0.07801105242393701,
            0.05960533346731668,
            0.0419734975689527,
            0.027174690655451306,
            0.016089670921338905,
            0.00931830970905999,
            0.00493957255797647,
            0.00781742216007927,
        ],
        0.12131072998046877,
    ),
];

/// Largest `k` the table holds.
pub const MAX_SPLIT_COUNT: usize = 21;

/// Split sizes the table holds, ascending.
///
/// Not a contiguous range: `k = 2` and `k = 4` cannot match the fourth moment on a
/// uniform symmetric grid, so there is no entry for either.
pub const SPLIT_SIZES: [usize; 18] = [
    3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
];

/// Weights and child width for a `k`-component split, or an error if `k` has no entry.
fn split_library(k: usize) -> KeteResult<(&'static [f64], f64)> {
    let index = match k {
        3 => 0,
        5..=MAX_SPLIT_COUNT => k - 4,
        _ => {
            return Err(Error::ValueError(format!(
                "no splitting library for k = {k}; k must be 3 or 5 to {MAX_SPLIT_COUNT}"
            )));
        }
    };
    Ok(SPLIT_LIBRARIES[index])
}

/// Component means for a library, in units of the parent's width along the split axis.
fn split_means(weights: &[f64], sigma: f64) -> Vec<f64> {
    let centre = (weights.len() as f64 - 1.0) * 0.5;
    let grid: Vec<f64> = (0..weights.len()).map(|j| j as f64 - centre).collect();
    let spread: f64 = weights
        .iter()
        .zip(grid.iter())
        .map(|(w, j)| w * j * j)
        .sum();
    let spacing = ((1.0 - sigma * sigma) / spread).sqrt();
    grid.into_iter().map(|j| j * spacing).collect()
}

/// How much a `k`-component split narrows the axis it cuts, as a factor.
///
/// The inverse of [`split_count_for_narrowing`], and what a caller needs to predict
/// what a split will buy: the second-order residual along the cut axis falls by the
/// square of this.
///
/// # Errors
/// Returns an error if `k` has no library entry.
pub fn split_narrowing(k: usize) -> KeteResult<f64> {
    Ok(1.0 / split_library(k)?.1)
}

/// The smallest `k` whose split narrows the cut axis by at least `narrowing`.
///
/// The controller asks this: a component reading `eta` against a threshold needs its
/// second-order residual reduced by `eta / threshold`, and that residual scales with
/// the square of the width, so the narrowing it needs is the square root of that
/// ratio.  Returns [`MAX_SPLIT_COUNT`] when the table cannot reach the request, which
/// is a deliberate cap on how much one split may cost rather than a failure - the next
/// leg measures what is left and asks again.
#[must_use]
pub fn split_count_for_narrowing(narrowing: f64) -> usize {
    if narrowing.is_nan() || narrowing <= 1.0 / K3_SPLIT_SIGMA {
        return 3;
    }
    for k in 5..=MAX_SPLIT_COUNT {
        if let Ok((_, sigma)) = split_library(k)
            && 1.0 / sigma >= narrowing
        {
            return k;
        }
    }
    MAX_SPLIT_COUNT
}

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
/// Splitting is decided one propagation leg at a time.  Over each leg the flow is
/// probed along every direction the component's covariance carries, and the probe's
/// departure from the linear (state transition matrix) prediction is read off in
/// sigma of a propagated position distribution - the component's own until it splits, and
/// its parent's from then on, so the threshold means one thing at every split depth, see
/// [`UncertainState::whitening_cov`].  A component whose worst
/// probe misses by more than `SplitConfig::split_threshold` is rolled back to the
/// start of its leg, split along that probe's direction, and the leg is redone with
/// the children.  Deciding this locally, leg by leg, is what places a split at the
/// time the flow actually stopped being linear rather than smearing it across the
/// whole arc.
///
/// The threshold is an accuracy/cost dial: tightening it splits earlier and more
/// often, tracking the true density more faithfully at a higher component count,
/// and `SplitConfig::max_components` bounds what a tight setting may spend.  In
/// practice a well-observed main-belt asteroid rarely needs splitting even over
/// years of propagation, while a dispersing dust cloud may consume whatever budget
/// it is given.
///
/// Splitting a component stops when it falls under `SplitConfig::split_threshold`, when
/// `SplitConfig::max_components` refuses the next split, or when nothing is left over
/// threshold that carries a direction to split along.  Which of these fired is reported by
/// the step that made the decision, in
/// [`StepReport`](crate::state::StepReport); the nonlinearity each component is still
/// holding stays on the component, in [`UncertainState::eta`].
///
/// # Usage
///
/// Use [`DiffuseState::from_uncertain`] to wrap a single [`UncertainState`] and
/// [`DiffuseState::new`] for explicit multi-component construction.
/// [`step_diffuse_state`](crate::state::step_diffuse_state) advances the mixture one leg,
/// and [`propagate_diffuse_state`](crate::state::propagate_diffuse_state)
/// folds that over a whole arc.  Marching by hand and marching in one call measure the
/// same thing, because each component carries its own probes.
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
    ///
    /// Each also carries its own probes and the nonlinearity they last reported, which is
    /// what lets a caller march the mixture one leg at a time and get the same measurement
    /// as a single call. See [`UncertainState::probes`].
    pub components: Vec<UncertainState>,

    /// Whether the largest asteroids perturb this mixture.
    ///
    /// Held on the mixture rather than passed per call, so every leg of a march runs
    /// under one force model. The components carry probes integrated under that model,
    /// and a leg taken under a different one would measure them against a flow they
    /// never saw.
    pub include_asteroids: bool,
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
        // Same check the Python layer used to make: the components have to be
        // describing one force model, or their free parameters do not mean the
        // same thing and the mixture is not a single density.
        let model = first.non_grav.as_ref().map(NonGravMask::free_param_names);
        if components
            .iter()
            .any(|c| c.non_grav.as_ref().map(NonGravMask::free_param_names) != model)
        {
            return Err(Error::ValueError(
                "All components must share the same non-gravitational model".into(),
            ));
        }
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
            if !c.elements.epoch.same_instant(&epoch) {
                return Err(Error::ValueError(format!(
                    "component {i} epoch {} does not match component 0 at {}",
                    c.elements.epoch.jd(),
                    epoch.jd()
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
            include_asteroids: false,
        })
    }

    /// Wrap a single [`UncertainState`] as a one-component mixture.
    #[must_use]
    pub fn from_uncertain(state: UncertainState) -> Self {
        Self {
            weights: vec![1.0],
            components: vec![state],
            include_asteroids: false,
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
    /// longitude. Splitting keeps *adjacent* components close, but that does not bound
    /// the span of the mixture: a phase-spread cloud propagated for an orbit reaches
    /// component separations at the half-turn limit while every neighboring pair is
    /// still tightly packed. Nothing here detects the crossing, so past it the moments
    /// are formed about a longitude no component is near and are returned anyway.
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
        let mut mean = UncertainState::new(elements, total.clone(), free_params)?;
        mean.non_grav.clone_from(&self.components[0].non_grav);
        Ok((mean, total))
    }

    /// Largest nonlinearity any component is carrying, in sigma of the propagated position
    /// distribution that component is whitened against - its own until it splits, its
    /// parent's afterwards, see [`UncertainState::whitening_cov`].
    ///
    /// Read off the components themselves ([`UncertainState::eta`]), so it describes the
    /// mixture in hand rather than the call that produced it. This is a statement about
    /// what the controller did, not an error bound on the represented density. A value
    /// above the `split_threshold` used at propagation time means at least one component
    /// was still failing the test when the march stopped;
    /// [`StepReport::termination`](crate::state::StepReport::termination) says why the last
    /// leg stopped splitting.
    ///
    /// Returns `None` if any component has never been marched, which is a different
    /// statement from zero: the largest value over a mixture is unknown when one member is
    /// unmeasured.
    #[must_use]
    pub fn max_eta(&self) -> Option<f64> {
        self.components
            .iter()
            .try_fold(0.0_f64, |worst, c| Some(worst.max(c.eta?)))
    }

    /// Worst residual behind those numbers, as a cartesian position offset in meters.
    ///
    /// Read with [`Self::max_eta`]: the propagator places a position to roughly a meter,
    /// so `eta 0.003` on a residual of `1.2` m is numerical noise rather than curvature,
    /// with no estimator in the library and no second run.
    ///
    /// Returns `None` if any component has never been marched.
    #[must_use]
    pub fn residual_meters(&self) -> Option<f64> {
        self.components
            .iter()
            .try_fold(0.0_f64, |worst, c| Some(worst.max(c.residual_meters?)))
    }

    /// Weight-weighted mean of the component `eta`.
    ///
    /// The mixture-level reading of how far from linear the distribution is, against
    /// [`Self::max_eta`], which is an extreme over the components and answers a different
    /// question: the worst single component, however little weight it carries.  Those
    /// diverge as a mixture grows, because splitting adds components faster than it
    /// improves the worst one, so a march that is resolving the bulk can report a rising
    /// maximum throughout.  Read this to see whether the budget bought anything.
    ///
    /// Returns `None` if any component has never been marched.
    #[must_use]
    pub fn mean_eta(&self) -> Option<f64> {
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            return None;
        }
        let weighted = self
            .components
            .iter()
            .zip(self.weights.iter())
            .try_fold(0.0_f64, |sum, (component, weight)| {
                Some(sum + weight * component.eta?)
            })?;
        Some(weighted / total)
    }

    /// Total weight of components whose nonlinearity exceeds `threshold`.
    ///
    /// Pass the `split_threshold` used at propagation time to read how much of the
    /// distribution finished under-resolved. This is the measure a starved low-weight tail
    /// shows up in: the count of components says nothing about where the weight sits.
    ///
    /// Returns `None` if any component has never been marched.
    #[must_use]
    pub fn weight_above_eta(&self, threshold: f64) -> Option<f64> {
        self.components.iter().zip(self.weights.iter()).try_fold(
            0.0_f64,
            |total, (component, weight)| {
                Some(
                    total
                        + if component.eta? > threshold {
                            *weight
                        } else {
                            0.0
                        },
                )
            },
        )
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
}

#[cfg(test)]
mod summary_tests {
    use super::DiffuseState;
    use crate::frames::Ecliptic;
    use crate::prelude::{Desig, State, UncertainState};
    use nalgebra::DMatrix;

    /// A mixture of components with known `eta`, to check what the summaries say about it.
    fn mixture(weights: &[f64], etas: &[f64]) -> DiffuseState {
        let components: Vec<UncertainState> = etas
            .iter()
            .map(|&eta| {
                let state = State::<Ecliptic>::new(
                    Desig::Name("test".into()),
                    2_451_545.0,
                    [1.0, 0.0, 0.0],
                    [0.0, 0.017_202_098_95, 0.0],
                    10,
                );
                let mut component =
                    UncertainState::from_state(&state, &(DMatrix::identity(6, 6) * 1e-12), vec![])
                        .expect("component");
                component.eta = Some(eta);
                component
            })
            .collect();
        DiffuseState::new(weights.to_vec(), components).expect("mixture")
    }

    /// The maximum is one component's number; the mean is the mixture's. A starved tail
    /// moves the first and barely touches the second, which is the whole reason both are
    /// published.
    #[test]
    fn the_mean_reads_the_mixture_and_the_max_reads_one_component() {
        let mix = mixture(&[0.98, 0.02], &[0.1, 50.0]);
        assert!((mix.max_eta().unwrap() - 50.0).abs() < 1e-12);
        let mean = mix.mean_eta().unwrap();
        assert!((mean - (0.98 * 0.1 + 0.02 * 50.0)).abs() < 1e-12, "{mean}");
        assert!(
            mean < 1.1,
            "a 2 percent tail must not set the mixture's number: {mean}"
        );
    }

    /// The weight above a threshold is what the tolerance is stated in.
    #[test]
    fn weight_above_a_threshold_counts_weight_not_components() {
        let mix = mixture(&[0.5, 0.25, 0.25], &[0.1, 0.2, 9.0]);
        assert!((mix.weight_above_eta(1.0).unwrap() - 0.25).abs() < 1e-12);
        assert!((mix.weight_above_eta(10.0).unwrap()).abs() < 1e-12);
    }

    /// An unmarched component leaves every summary undefined rather than counting as zero.
    #[test]
    fn an_unmarched_component_has_no_summary() {
        let mut mix = mixture(&[0.5, 0.5], &[0.1, 0.2]);
        mix.components[1].eta = None;
        assert!(mix.mean_eta().is_none());
        assert!(mix.max_eta().is_none());
        assert!(mix.weight_above_eta(1.0).is_none());
    }
}

/// Split a single [`UncertainState`] into a `k`-component sub-mixture of the
/// marginal along an explicitly supplied direction in initial state space.
///
/// `direction` does not need to be a unit vector (it is normalized
/// internally) and does not need to be an eigenvector of the component's
/// covariance: it selects the univariate functional `a = u^T x` whose
/// marginal is replaced by the K=3 mixture, while the conditional of the
/// remaining coordinates given `a` is kept exact.  The children are
/// therefore displaced along the regression vector `P u / sigma` -- the
/// ridge of the parent density - not along `u` itself, and the mixture
/// approximates the parent with the one-dimensional Huber fidelity for
/// any direction.  Mean and total covariance are preserved exactly, and
/// the reduced covariance is positive semi-definite by construction.
///
/// The motivating use case is adaptive cloud propagation, where the
/// calling layer supplies the direction whose curvature it measured over
/// the leg about to be redone - the direction the split is meant to
/// remove.  Taking a direction rather than deriving one also handles the
/// isotropic covariance, where the eigenvalue decomposition is degenerate
/// and picking a dominant eigenvector picks an arbitrary axis.
///
/// # Errors
/// Returns an error if `k` has no library entry, if `direction` has the wrong
/// length, is zero (or non-finite), or if the variance of the component along
/// `direction` is non-positive.
pub fn split_axial_along(
    component: &UncertainState,
    direction: &DVector<f64>,
    k: usize,
) -> KeteResult<Vec<(f64, UncertainState)>> {
    let (weights, child_sigma) = split_library(k)?;
    let means = split_means(weights, child_sigma);
    let dim = component.cov_matrix.nrows();
    if direction.len() != dim {
        return Err(Error::ValueError(format!(
            "split_axial_along: direction length {} does not match \
             component covariance dimension {dim}",
            direction.len()
        )));
    }
    let norm = direction.norm();
    if !norm.is_finite() || norm <= 0.0 {
        return Err(Error::ValueError(
            "split_axial_along: direction must have finite, nonzero norm".into(),
        ));
    }
    let d = direction / norm;

    // Variance of the marginal along d.
    let sigma_sq_mat = d.transpose() * &component.cov_matrix * &d;
    let sigma_sq = sigma_sq_mat[(0, 0)];
    if !sigma_sq.is_finite() || sigma_sq <= 0.0 {
        return Err(Error::ValueError(
            "split_axial_along: variance along the requested direction is non-positive".into(),
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
    let alpha = 1.0 - child_sigma * child_sigma;
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

    weights
        .iter()
        .zip(means.iter())
        .map(|(&weight, &mean)| {
            let delta = &r * mean;
            Ok((
                weight,
                build_split_component(component, &delta, cov_new.clone())?,
            ))
        })
        .collect()
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

    let mut child = UncertainState::new(new_elements, new_cov, new_params)?;
    // The model interprets the free parameters the child inherited, so it has to
    // come with them.
    child.non_grav.clone_from(&base.non_grav);
    Ok(child)
}

impl DiffuseState {
    /// Save into a binary file.
    ///
    /// # Errors
    /// Saving is fallible due to filesystem calls.
    pub fn save(&self, filename: String) -> KeteResult<()> {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::fs::File;
        use std::io::BufWriter;
        let f = BufWriter::new(File::create(filename)?);
        let mut gz = GzEncoder::new(f, Compression::default());
        crate::io::binary::write_diffuse_kete_file(self, &mut gz)?;
        let _ = gz.finish()?;
        Ok(())
    }

    /// Load from a binary file.
    ///
    /// # Errors
    /// Loading is fallible due to filesystem calls, and the file must hold a
    /// single mixture rather than a collection or another type.
    pub fn load(filename: String) -> KeteResult<Self> {
        match Self::read(filename)? {
            crate::io::binary::KeteFileType::Diffuse(mixture) => Ok(*mixture),
            crate::io::binary::KeteFileType::DiffuseVec(v) => Err(Error::ValueError(format!(
                "Expected a single DiffuseState, but found a vector of length {}.",
                v.len()
            ))),
            crate::io::binary::KeteFileType::Single(_)
            | crate::io::binary::KeteFileType::Vec(_) => Err(Error::ValueError(
                "Expected a DiffuseState, but the file holds SimultaneousStates.".into(),
            )),
            crate::io::binary::KeteFileType::Uncertain(_)
            | crate::io::binary::KeteFileType::UncertainVec(_) => Err(Error::ValueError(
                "Expected a DiffuseState, but the file holds UncertainStates.".into(),
            )),
        }
    }

    /// Save a vector of `DiffuseState` into a binary file.
    ///
    /// # Errors
    /// Saving is fallible due to filesystem calls.
    pub fn save_vec(vec: &[Self], filename: String) -> KeteResult<()> {
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::fs::File;
        use std::io::BufWriter;
        let f = BufWriter::new(File::create(filename)?);
        let mut gz = GzEncoder::new(f, Compression::default());
        crate::io::binary::write_diffuse_vec_kete_file(vec, &mut gz)?;
        let _ = gz.finish()?;
        Ok(())
    }

    /// Load a vector of `DiffuseState` from a binary file.
    ///
    /// A file holding a single mixture reads back as a collection of one.
    ///
    /// # Errors
    /// Loading is fallible due to filesystem calls, and the file must hold
    /// mixtures rather than another type.
    pub fn load_vec(filename: String) -> KeteResult<Vec<Self>> {
        match Self::read(filename)? {
            crate::io::binary::KeteFileType::DiffuseVec(v) => Ok(v),
            crate::io::binary::KeteFileType::Diffuse(mixture) => Ok(vec![*mixture]),
            crate::io::binary::KeteFileType::Single(_)
            | crate::io::binary::KeteFileType::Vec(_) => Err(Error::ValueError(
                "Expected DiffuseStates, but the file holds SimultaneousStates.".into(),
            )),
            crate::io::binary::KeteFileType::Uncertain(_)
            | crate::io::binary::KeteFileType::UncertainVec(_) => Err(Error::ValueError(
                "Expected DiffuseStates, but the file holds UncertainStates.".into(),
            )),
        }
    }

    fn read(filename: String) -> KeteResult<crate::io::binary::KeteFileType> {
        use flate2::read::GzDecoder;
        use std::fs::File;
        use std::io::BufReader;
        let mut f = BufReader::new(GzDecoder::new(File::open(filename)?));
        crate::io::binary::read_kete_file(&mut f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::EquinoctialElements;
    use crate::frames::Equatorial;
    use core::f64::consts::TAU;
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
        let naive = f64::midpoint(low.elements.true_lon, high.elements.true_lon);
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
        let mean: Vec<f64> = (0..6)
            .map(|i| f64::midpoint(offset_a[i], offset_b[i]))
            .collect();
        for r in 0..6 {
            for c in 0..6 {
                let between = 0.5 * (offset_a[r] - mean[r]) * (offset_a[c] - mean[c])
                    + 0.5 * (offset_b[r] - mean[r]) * (offset_b[c] - mean[c]);
                let within = 0.5 * within_a[(r, c)] + 0.5 * within_b[(r, c)];
                let expect = within + between;
                let scale = f64::midpoint(within_a[(r, r)], within_b[(r, r)]).sqrt()
                    * f64::midpoint(within_a[(c, c)], within_b[(c, c)]).sqrt()
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
        let midpoint = f64::midpoint(near.pos[1], far.pos[1]);

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

    /// A split produces three components carrying the K=3 weights, which sum to the
    /// parent's.
    #[test]
    fn test_split_weights_are_the_k3_table() {
        let mut a = small_uncertain("A");
        // Physical, and anisotropic enough that the split delta is non-trivial. A one
        // sigma step in element coordinates at AU scale drives the orbit outside its own
        // domain, so the covariance stays small.
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        cov[(0, 0)] = 1e-6;
        cov[(1, 1)] = 1e-10;
        cov[(2, 2)] = 1e-10;
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        a.cov_matrix = cov;

        let mut direction = DVector::<f64>::zeros(6);
        direction[0] = 1.0;
        let parts = split_axial_along(&a, &direction, 3).unwrap();
        let total: f64 = parts.iter().map(|(w, _)| w).sum();
        assert!((total - 1.0).abs() < 1e-15);
        for ((got, _), &want) in parts.iter().zip(K3_SPLIT_WEIGHTS.iter()) {
            assert!((got - want).abs() < 1e-15);
        }
    }

    /// Splitting along a free-parameter direction disperses that parameter across the
    /// children, which is what makes a beta-spread cloud expressible as a split mixture.
    #[test]
    fn test_split_along_a_parameter_axis() {
        let st = test_state("A");
        // 7x7 cov: tiny in the elements, large in the free param.
        let mut cov = DMatrix::<f64>::zeros(7, 7);
        for i in 0..3 {
            cov[(i, i)] = 1e-12;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        cov[(6, 6)] = 1e-4;

        let a = UncertainState::from_state(&st, &cov, vec![0.01]).unwrap();
        let mut direction = DVector::<f64>::zeros(7);
        direction[6] = 1.0;
        let parts = split_axial_along(&a, &direction, 3).unwrap();

        // Means shifted by K3_SPLIT_MEANS[k] * sqrt(1e-4) from the original 0.01.
        let offset = K3_SPLIT_MEANS[2] * 1e-4_f64.sqrt(); // sqrt(3/2) * 0.01
        let mut params: Vec<f64> = parts.iter().map(|(_, c)| c.free_params[0]).collect();
        params.sort_by(f64::total_cmp);
        assert!((params[0] - (0.01 - offset)).abs() < 1e-10);
        assert!((params[1] - 0.01).abs() < 1e-10);
        assert!((params[2] - (0.01 + offset)).abs() < 1e-10);
    }

    /// A covariance with no extent has no marginal to replace, so the split is refused
    /// rather than producing three copies of the parent.
    #[test]
    fn test_split_rejects_zero_covariance() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::zeros(6, 6);
        let a = UncertainState::from_state(&st, &cov, vec![]).unwrap();
        let mut direction = DVector::<f64>::zeros(6);
        direction[0] = 1.0;
        assert!(split_axial_along(&a, &direction, 3).is_err());
    }

    /// `split_axial_along` must preserve the mixture's total mean
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
        let parts = split_axial_along(&component, &direction, 3).unwrap();
        let weights: Vec<f64> = parts.iter().map(|(w, _)| *w).collect();
        let comps: Vec<UncertainState> = parts.into_iter().map(|(_, c)| c).collect();
        let mixture = DiffuseState::new(weights, comps).unwrap();

        // Asserted in element coordinates, where the split arithmetic lives. The
        // cartesian weighted mean of the children is deliberately not preserved: the map
        // from elements to a state is nonlinear, and that curvature is the whole reason
        // the element representation exists. Requiring it here would be requiring the
        // mixture to be a worse description than it is.
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
            let parts = split_axial_along(&component, &d, 3).unwrap();
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

    /// Every library in the table is a moment-matched replacement for the standard
    /// normal: weights that are positive and sum to one, and the first four moments of
    /// the mixture equal to `0, 1, 0, 3`.  The first two are what the multivariate
    /// rank-1 reduction depends on; the fourth is what keeps repeated splits from
    /// drifting in shape.
    #[test]
    fn every_library_matches_the_first_four_moments() {
        for k in SPLIT_SIZES {
            let (w, sigma) = split_library(k).unwrap();
            let m = split_means(w, sigma);
            assert_eq!(w.len(), k, "k = {k}");
            assert!(
                w.iter().all(|&x| x > 0.0),
                "k = {k} has a non-positive weight"
            );
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "k = {k} weights sum to {sum}");

            let v = sigma * sigma;
            let moment = |order: i32| -> f64 {
                w.iter().zip(m.iter()).map(|(w, m)| w * m.powi(order)).sum()
            };
            assert!(moment(1).abs() < 1e-12, "k = {k} mean");
            assert!((moment(2) + v - 1.0).abs() < 1e-12, "k = {k} variance");
            assert!(
                (moment(3) + 3.0 * v * moment(1)).abs() < 1e-12,
                "k = {k} third moment"
            );
            let fourth = moment(4) + 6.0 * v * moment(2) + 3.0 * v * v;
            assert!(
                (fourth - 3.0).abs() < 1e-9,
                "k = {k} fourth moment is {fourth}"
            );
        }
    }

    /// The `k = 3` entry is the library that shipped before the table existed, so a
    /// three-way split is the same arithmetic it always was and a run that never asks
    /// for a larger one is unchanged.
    #[test]
    fn the_three_way_entry_is_the_one_that_shipped() {
        let (w, sigma) = split_library(3).unwrap();
        assert_eq!(w, K3_SPLIT_WEIGHTS);
        assert_eq!(sigma, K3_SPLIT_SIGMA);
        for (got, want) in split_means(w, sigma).iter().zip(K3_SPLIT_MEANS.iter()) {
            assert!(
                (got - want).abs() <= f64::EPSILON * want.abs().max(1.0),
                "{got} vs {want}"
            );
        }
    }

    /// Integral square difference between a symmetric homoscedastic mixture and the
    /// standard normal, in closed form.
    fn library_isd(weights: &[f64], means: &[f64], sigma: f64) -> f64 {
        let normal = |x: f64, var: f64| (-0.5 * x * x / var).exp() / (TAU * var).sqrt();
        let v = sigma * sigma;
        let mut cross = 0.0;
        for (wi, mi) in weights.iter().zip(means.iter()) {
            for (wj, mj) in weights.iter().zip(means.iter()) {
                cross += wi * wj * normal(mi - mj, 2.0 * v);
            }
        }
        let overlap: f64 = weights
            .iter()
            .zip(means.iter())
            .map(|(w, m)| w * normal(*m, 1.0 + v))
            .sum();
        cross - 2.0 * overlap + 1.0 / (2.0 * f64::consts::PI.sqrt())
    }

    /// Fidelity is held constant across the table and `k` buys narrowing alone.  Both
    /// halves are asserted: no library is a worse approximation of its parent than the
    /// three-way one, and the narrowing rises with every entry.
    #[test]
    fn a_bigger_library_narrows_more_at_the_same_fidelity() {
        let (w3, s3) = split_library(3).unwrap();
        let bar = library_isd(w3, &split_means(w3, s3), s3);
        let mut previous = 1.0 / s3;
        for k in SPLIT_SIZES.iter().skip(1).copied() {
            let (w, sigma) = split_library(k).unwrap();
            let isd = library_isd(w, &split_means(w, sigma), sigma);
            assert!(
                isd <= bar * 1.001,
                "k = {k} is less faithful than k = 3: {isd} vs {bar}"
            );
            let narrowing = 1.0 / sigma;
            assert!(
                narrowing > previous,
                "k = {k} narrows {narrowing}, no more than {previous}"
            );
            previous = narrowing;
        }
        // The table's reach, which is what bounds one split's cost.
        assert!(
            previous > 8.0,
            "the largest library only narrows {previous}"
        );
    }

    /// The controller asks for a narrowing and gets the cheapest library that delivers
    /// it.  Below what a three-way split already does the answer is three, and a request
    /// past the table is answered with the largest entry rather than an error.
    #[test]
    fn the_chooser_returns_the_cheapest_library_that_reaches_the_request() {
        assert_eq!(split_count_for_narrowing(0.5), 3);
        assert_eq!(split_count_for_narrowing(1.0), 3);
        assert_eq!(split_count_for_narrowing(1.0 / K3_SPLIT_SIGMA), 3);
        assert_eq!(split_count_for_narrowing(f64::NAN), 3);
        assert_eq!(split_count_for_narrowing(f64::INFINITY), MAX_SPLIT_COUNT);
        assert_eq!(split_count_for_narrowing(1e9), MAX_SPLIT_COUNT);

        for k in SPLIT_SIZES.iter().skip(1).copied() {
            let (_, sigma) = split_library(k).unwrap();
            // Asking for exactly what this library delivers must return this library,
            // and asking for a hair more must step up.
            assert_eq!(
                split_count_for_narrowing(1.0 / sigma - 1e-12),
                k,
                "at k = {k}"
            );
            if k < MAX_SPLIT_COUNT {
                let bigger = split_count_for_narrowing(1.0 / sigma + 1e-9);
                assert!(bigger > k, "a larger request at k = {k} returned {bigger}");
            }
        }
    }

    /// A `k` with no entry is refused rather than silently rounded to one that exists.
    #[test]
    fn a_split_size_the_table_does_not_hold_is_refused() {
        let st = test_state("A");
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let component = UncertainState::from_state(&st, &cov, vec![]).unwrap();
        let mut direction = DVector::<f64>::zeros(6);
        direction[0] = 1.0;
        for k in [0_usize, 1, 2, 4, MAX_SPLIT_COUNT + 1] {
            assert!(
                split_axial_along(&component, &direction, k).is_err(),
                "k = {k} was accepted"
            );
        }
    }

    /// A split of any size preserves the parent's mean and covariance exactly, for a
    /// direction that is not an eigenvector.  This is the property the whole mixture
    /// rests on: the children are a re-description of the parent, not a new
    /// distribution, whatever `k` the controller chose.
    #[test]
    fn a_split_of_any_size_preserves_the_parent_moments() {
        let st = test_state("A");
        let elements = EquinoctialElements::from_state(&st.into_frame()).unwrap();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        let scales = [1.0e-6, 3.0e-7, 2.0e-7, 1.5e-7, 1.0e-7, 8.0e-8];
        for (i, scale) in scales.iter().enumerate() {
            cov[(i, i)] = scale * scale;
        }
        cov[(0, 1)] = 0.6 * scales[0] * scales[1];
        cov[(1, 0)] = cov[(0, 1)];
        let component = UncertainState::new(elements, cov.clone(), vec![]).unwrap();
        let base = component.elements.clone();

        let direction = DVector::from_row_slice(&[1.0, 0.5, 0.0, 0.7, 0.0, -0.3]);
        for k in SPLIT_SIZES {
            let parts = split_axial_along(&component, &direction, k).unwrap();
            assert_eq!(parts.len(), k);
            let weights: Vec<f64> = parts.iter().map(|(w, _)| *w).collect();
            let comps: Vec<UncertainState> = parts.into_iter().map(|(_, c)| c).collect();
            let mixture = DiffuseState::new(weights, comps).unwrap();
            let (mean, total) = mixture.mean_and_covariance().unwrap();

            let moved = base.offset_to(&mean.elements);
            for i in 0..6 {
                assert!(
                    moved[i].abs() < 1e-12 * cov[(i, i)].sqrt(),
                    "k = {k}: element mean {i} moved"
                );
            }
            for r in 0..6 {
                for c in 0..6 {
                    let scale = (cov[(r, r)] * cov[(c, c)]).sqrt();
                    assert!(
                        (total[(r, c)] - cov[(r, c)]).abs() / scale < 1e-9,
                        "k = {k}: cov[{r},{c}] is {} against {}",
                        total[(r, c)],
                        cov[(r, c)]
                    );
                }
            }
        }
    }

    /// A split narrows the axis it cuts by the library's `s` and leaves every direction
    /// conjugate to it alone.  That is what makes the split size a dial on nonlinearity:
    /// the second-order residual along the cut axis scales with the square of the width,
    /// so `s` of 1/9 is an eighty-fold reduction from one split.
    #[test]
    fn a_split_narrows_the_axis_it_cuts_and_no_other() {
        let st = test_state("A");
        let elements = EquinoctialElements::from_state(&st.into_frame()).unwrap();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        let scales = [2.0e-6, 5.0e-7, 4.0e-7, 3.0e-7, 2.0e-7, 1.0e-7];
        for (i, scale) in scales.iter().enumerate() {
            cov[(i, i)] = scale * scale;
        }
        cov[(2, 4)] = 0.4 * scales[2] * scales[4];
        cov[(4, 2)] = cov[(2, 4)];
        let component = UncertainState::new(elements, cov.clone(), vec![]).unwrap();

        let direction = DVector::from_row_slice(&[0.3, 1.0, 0.0, -0.5, 0.2, 0.0]);
        let unit: DVector<f64> = &direction / direction.norm();
        let variance = (unit.transpose() * &cov * &unit)[(0, 0)];
        // The regression vector: the ridge the children slide along, and the direction
        // the reduction acts on.
        let ridge = (&cov * &unit) / variance.sqrt();

        for k in SPLIT_SIZES {
            let (_, sigma) = split_library(k).unwrap();
            let child = split_axial_along(&component, &direction, k).unwrap()[0]
                .1
                .cov_matrix
                .clone();
            let cut = (unit.transpose() * &child * &unit)[(0, 0)];
            assert!(
                (cut / (variance * sigma * sigma) - 1.0).abs() < 1e-9,
                "k = {k}: cut axis variance {cut} against {}",
                variance * sigma * sigma
            );

            // A direction conjugate to the ridge keeps the width it had.
            let mut other = DVector::<f64>::zeros(6);
            other[5] = 1.0;
            let other = &other - &ridge * (ridge.dot(&other) / ridge.dot(&ridge));
            let before = (other.transpose() * &cov * &other)[(0, 0)];
            let after = (other.transpose() * &child * &other)[(0, 0)];
            assert!(
                (after / before - 1.0).abs() < 1e-6,
                "k = {k}: an uncut direction moved from {before} to {after}"
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
        assert!(split_axial_along(&component, &zero, 3).is_err());
        // Wrong length.
        let wrong = DVector::<f64>::from_element(7, 1.0);
        assert!(split_axial_along(&component, &wrong, 3).is_err());
    }

    /// For an isotropic covariance, splitting along any unit
    /// direction produces a positive-definite result (the
    /// generalization that motivated `split_axial_along` in the
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
            let parts = split_axial_along(&component, &d, 3)
                .unwrap_or_else(|e| panic!("split failed for {entries:?}: {e}"));
            assert_eq!(parts.len(), 3);
        }
    }
}
