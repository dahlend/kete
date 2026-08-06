//! Uncertain state representation: a best-fit orbit as [`EquinoctialElements`], plus a
//! covariance over that orbit's six stored floats and any fitted force parameters.
//!
//! The covariance is **not** cartesian. It is a Gaussian over the elements themselves,
//! which carry no constraint between them and so need no chart.
//! [`covariance_to_equinoctial`] and [`covariance_from_equinoctial`] convert at the
//! boundaries; see [`UncertainState`] for why.
//!
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

use crate::elements::{CometElements, EquinoctialElements};
use crate::frames::{CenterBody, DynCenter, Ecliptic, InertialFrame};
use crate::prelude::{Desig, Error, KeteResult, State};
use crate::time::{TDB, Time};
use nalgebra::{DMatrix, Matrix6, Vector3, Vector6};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

/// A best-fit orbit together with a covariance matrix and zero or more fitted free
/// parameters.
///
/// The covariance is `(6 + Np) x (6 + Np)` where `Np = free_params.len()`. Rows and
/// columns 6 onward are the free parameters in the same order as `free_params`; their
/// semantic labels live with the [`Force`](crate::forces::Force) impls that produce them,
/// and the state itself stores values only.
///
/// **Rows and columns 0 to 5 are equinoctial orbital elements, not cartesian position and
/// velocity.** They are the six stored floats of [`EquinoctialElements`], in field order:
/// semi-latus rectum in AU, the two eccentricity components, the two pole components, and
/// the true longitude at the epoch in radians.
///
/// The reason is not stylistic. A linear map describes the propagated distribution far more
/// accurately in these coordinates than in cartesian ones, because a cartesian covariance
/// shears into a curved distribution that no linear map can represent. The gain is
/// reported by `element_vs_cartesian_linearity_horizon` in `kete_spice`, which scores both
/// bases against a clone ensemble under the real N-body force model.
///
/// Use [`Self::cartesian_covariance`] for reporting, interchange, or comparison against an
/// externally supplied covariance, and [`Self::from_state`] to enter from a cartesian one.
/// Both conversions are exact as linear maps.
#[derive(Debug, Clone)]
pub struct UncertainState {
    /// Best-fit orbit, and the point the covariance is referred to. Carries the
    /// designation, epoch, central body and its gravitational parameter.
    pub elements: EquinoctialElements,

    /// Covariance matrix, `(6 + Np) x (6 + Np)`, over the stored floats of
    /// [`EquinoctialElements`]. See the type documentation: rows 0 to 5 are orbital
    /// elements, **not** cartesian position and velocity.
    pub cov_matrix: DMatrix<f64>,

    /// Current best estimates of the fitted parameters, length `Np`.
    /// `Np` may be zero (no fitted parameters), the typical case for a
    /// pure orbit-determination state.
    pub free_params: Vec<f64>,

    /// Peak sigma-point Mahalanobis divergence ever recorded for this
    /// trajectory during adaptive propagation, never reset.  The metric
    /// measures the linear (STM-based) prediction error as a
    /// sigma-equivalent distance in the propagated covariance -- see
    /// [`sigma_point_divergence`](crate::state::sigma_point_divergence)
    /// for the full definition.
    ///
    /// `0.0` means the state has never been propagated through adaptive
    /// diagnosis, or every diagnosis returned a clean linear result.
    /// A value above the adaptive `split_threshold` indicates the
    /// linear representation of this component lost accuracy during its
    /// history and could not be split further (e.g. due to a
    /// `max_components` budget cap).  Inherited by split children.
    pub max_unresolved_divergence: f64,
}

impl UncertainState {
    /// Construct from a best-fit orbit and a covariance already in element coordinates.
    ///
    /// Use [`Self::from_state`] when the covariance is cartesian.
    ///
    /// # Errors
    /// Returns an error if `cov_matrix` is not
    /// `(6 + free_params.len()) x (6 + free_params.len())`.
    pub fn new(
        elements: EquinoctialElements,
        cov_matrix: DMatrix<f64>,
        free_params: Vec<f64>,
    ) -> KeteResult<Self> {
        let expected = 6 + free_params.len();
        if cov_matrix.nrows() != expected || cov_matrix.ncols() != expected {
            return Err(Error::ValueError(format!(
                "Covariance matrix must be {expected}x{expected}, \
                 got {}x{}",
                cov_matrix.nrows(),
                cov_matrix.ncols()
            )));
        }
        Ok(Self {
            elements,
            cov_matrix,
            free_params,
            max_unresolved_divergence: 0.0,
        })
    }

    /// Construct from a cartesian state and a **cartesian** covariance, converting the
    /// covariance into element coordinates.
    ///
    /// This is the entry point for anything which solved in cartesian coordinates, the
    /// orbit fitter among them. The covariance is taken to be in the same frame as the
    /// state.
    ///
    /// # Errors
    /// Fails if the covariance is the wrong shape, if the state has no valid orbit about
    /// its center, or if the orbit is too close to the equinoctial seam to carry a
    /// covariance.
    pub fn from_state<F: InertialFrame, C: CenterBody>(
        state: &State<F, C>,
        cov_cartesian: &DMatrix<f64>,
        free_params: Vec<f64>,
    ) -> KeteResult<Self>
    where
        DynCenter: From<C>,
    {
        let ecliptic: State<Ecliptic, C> = state.clone().into_frame();
        let elements = EquinoctialElements::from_state(&ecliptic)?;
        let cov_matrix = covariance_to_equinoctial::<F>(&elements, cov_cartesian)?;
        Self::new(elements, cov_matrix, free_params)
    }

    /// The best-fit cartesian state, relative to the elements' central body.
    ///
    /// # Errors
    /// Fails if the elements are outside their physical domain.
    pub fn state<F: InertialFrame>(&self) -> KeteResult<State<F>> {
        Ok(self.elements.try_to_state()?.into_frame())
    }

    /// The covariance re-expressed in cartesian position and velocity, in frame `F`.
    ///
    /// The transform is exact as a linear map, but a cartesian covariance is a much worse
    /// description of the same distribution over a long arc - see the type documentation.
    /// Use this for reporting and interchange, not as a working representation.
    ///
    /// # Errors
    /// Fails if the elements are outside their physical domain, or too close to the seam.
    pub fn cartesian_covariance<F: InertialFrame>(&self) -> KeteResult<DMatrix<f64>> {
        covariance_from_equinoctial::<F>(&self.elements, &self.cov_matrix)
    }

    /// Epoch of the best-fit orbit.
    pub fn epoch(&self) -> Time<TDB> {
        self.elements.epoch
    }

    /// NAIF id of the central body the elements are referred to.
    #[must_use]
    pub fn center_id(&self) -> i32 {
        self.elements.center_id
    }

    /// Number of fitted free parameters (`free_params.len()`).
    #[must_use]
    pub fn n_free_params(&self) -> usize {
        self.free_params.len()
    }

    /// Construct an `UncertainState` from cometary orbital elements and
    /// a covariance expressed in cometary element space.
    ///
    /// The cometary-element covariance is transformed to a cartesian
    /// covariance via the numerically evaluated Jacobian
    /// `J = d(x,y,z,vx,vy,vz) / d(e,q,tp,node,w,i)`, then into equinoctial coordinates.
    /// Both element sets are defined by ecliptic angles, so no frame rotation enters.
    ///
    /// When the covariance is larger than 6x6 (i.e. includes free
    /// parameters), the off-diagonal cross-terms are transformed by `J`
    /// and the parameter block is left unchanged.
    ///
    /// # Arguments
    /// * `elements` -- Cometary orbital elements with desig and epoch.
    /// * `cov_elem` -- Covariance in element space, `(6+Np) x (6+Np)`.
    ///   Row/column ordering:
    ///   0. eccentricity (dimensionless)
    ///   1. `peri_dist` (AU)
    ///   2. `peri_time` (JD, TDB)
    ///   3. `lon_of_ascending` (**radians**)
    ///   4. `peri_arg` (**radians**)
    ///   5. inclination (**radians**)
    ///   6. free parameters (if any)
    ///
    ///   Angular elements must be in radians, matching the units stored
    ///   in [`CometElements`].  If your source covariance is in degrees
    ///   (e.g. JPL Horizons), scale angular rows/columns by `pi/180`
    ///   before calling this function.
    /// * `free_params` -- Initial free-parameter values, length `Np`.
    ///   `Np = 0` is the no-parameter case.
    ///
    /// # Errors
    /// Returns an error if element-to-state conversion fails or if the
    /// covariance dimensions are inconsistent.
    pub fn from_cometary(
        elements: &CometElements,
        cov_elem: &DMatrix<f64>,
        free_params: Vec<f64>,
    ) -> KeteResult<Self> {
        let np = free_params.len();
        let expected = 6 + np;
        if cov_elem.nrows() != expected || cov_elem.ncols() != expected {
            return Err(Error::ValueError(format!(
                "Element covariance must be {expected}x{expected}, \
                 got {}x{}",
                cov_elem.nrows(),
                cov_elem.ncols()
            )));
        }

        let state = elements.try_to_state()?;
        let jac = cometary_to_cartesian_jacobian(elements)?;

        // Transform the orbital-element covariance block.
        let c_elem_6x6 = cov_elem.view((0, 0), (6, 6));
        let c_cart = &jac * c_elem_6x6 * jac.transpose();

        if np == 0 {
            return Self::from_state(&state, &c_cart, free_params);
        }

        // Full (6+Np)x(6+Np) covariance with transformed blocks.
        let mut cov_cart = DMatrix::zeros(expected, expected);
        cov_cart.view_mut((0, 0), (6, 6)).copy_from(&c_cart);

        // Off-diagonal: J * C_cross_elem  (6xNp block).
        let cross_elem = cov_elem.view((0, 6), (6, np));
        let cross_cart = &jac * cross_elem;
        cov_cart.view_mut((0, 6), (6, np)).copy_from(&cross_cart);
        cov_cart
            .view_mut((6, 0), (np, 6))
            .copy_from(&cross_cart.transpose());

        // Lower-right: parameter block unchanged.
        cov_cart
            .view_mut((6, 6), (np, np))
            .copy_from(&cov_elem.view((6, 6), (np, np)));

        Self::from_state(&state, &cov_cart, free_params)
    }

    /// Draw random samples from the covariance distribution.
    ///
    /// Returns a vector of `(State, Vec<f64>)` pairs, where the second
    /// element is a perturbed copy of `free_params` with the same
    /// length. Perturbations are drawn from the multivariate normal
    /// defined by `cov_matrix`.
    ///
    /// # Arguments
    /// * `n_samples` -- Number of samples to draw.
    /// * `seed` -- Optional RNG seed for reproducibility.
    ///
    /// # Errors
    /// Returns an error if a drawn sample leaves the elements' physical domain.
    pub fn sample<F: InertialFrame>(
        &self,
        n_samples: usize,
        seed: Option<u64>,
    ) -> KeteResult<Vec<(State<F>, Vec<f64>)>> {
        let n = self.cov_matrix.nrows();

        // Decompose using eigenvalues to handle positive semi-definite
        // matrices (e.g. when some parameters have zero variance).
        // C = V * diag(d) * V^T  ->  L = V * diag(sqrt(max(d,0)))
        // so that L * z produces samples in the non-null subspace.
        let sym = nalgebra::SymmetricEigen::new(self.cov_matrix.clone());
        let l = {
            let sqrt_diag = DMatrix::from_diagonal(
                &sym.eigenvalues
                    .map(|v| if v > 0.0 { v.sqrt() } else { 0.0 }),
            );
            &sym.eigenvectors * sqrt_diag
        };

        // Build RNG.
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_seed(rand::random()),
        };

        let np = self.free_params.len();
        let mut results = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            // Draw z ~ N(0, I) and compute delta = L * z.
            let z = nalgebra::DVector::from_fn(n, |_, _| StandardNormal.sample(&mut rng));
            let delta = &l * z;

            // The leading six are element coordinates, so each draw displaces the stored
            // floats rather than adding a cartesian offset. Every sample is therefore an
            // exact orbit rather than a state which only nearly satisfies the two-body
            // equations, and the placement stays correct for wide covariances where a
            // linear cartesian offset would not.
            let step = Vector6::from_iterator(delta.iter().take(6).copied());
            let sampled = self
                .elements
                .displaced_by(&step)
                .try_to_state()?
                .into_frame();
            let sampled_params: Vec<f64> = (0..np)
                .map(|i| self.free_params[i] + delta[6 + i])
                .collect();

            results.push((sampled, sampled_params));
        }

        Ok(results)
    }

    /// How faithfully this state's element covariance and its cartesian image describe
    /// the same distribution, as a sigma-point divergence.
    ///
    /// The conversion between the two bases is exact as a linear map, but a Gaussian in
    /// one basis is not a Gaussian in the other: the change of chart is nonlinear off the
    /// mean, and the linear map keeps only its value and first derivative there. Called
    /// immediately after construction from a cartesian covariance, this measures what the
    /// entry conversion discarded - the fidelity of the *initial* Gaussian, before any
    /// propagation adds loss of its own.
    ///
    /// See [`equinoctial_conversion_divergence`] for the definition, the scale, and the
    /// probe placement. The result is frame invariant, so no frame is exposed here.
    ///
    /// # Errors
    /// Fails if `sigma_factor` is not positive and finite, if the covariance is zero, or
    /// if the orbit is outside the covariance domain.
    pub fn conversion_divergence(&self, sigma_factor: f64) -> KeteResult<f64> {
        let cartesian = self.cartesian_covariance::<Ecliptic>()?;
        equinoctial_conversion_divergence::<Ecliptic>(&self.elements, &cartesian, sigma_factor)
    }
}

/// Re-express an augmented covariance in equinoctial element coordinates.
///
/// The input is `(6 + Np) x (6 + Np)` with rows and columns 0 to 5 the cartesian position
/// and velocity in frame `F`, about the same center as `elements`. The output has those
/// six replaced by the stored floats of [`EquinoctialElements`], in field order.
///
/// The storage is unconstrained, so the result is a full-rank covariance over the six
/// numbers themselves. See [`covariance_from_equinoctial`] for the inverse.
///
/// # Errors
/// Fails if the covariance is not `(6 + Np) x (6 + Np)`, if the elements are outside
/// their physical domain, or if the orbit is close enough to the equinoctial seam that
/// the result would not be representable, see [`equinoctial_covariance_domain`].
pub fn covariance_to_equinoctial<F: InertialFrame>(
    elements: &EquinoctialElements,
    cov: &DMatrix<f64>,
) -> KeteResult<DMatrix<f64>> {
    equinoctial_covariance_domain(elements)?;
    congruence(cov, &elements.state_jacobian_inverse::<F>()?)
}

/// Re-express an augmented covariance in element coordinates back into cartesian position
/// and velocity in frame `F`, the inverse of [`covariance_to_equinoctial`].
///
/// The pair exists because the two bases are useful for different things. A covariance
/// stored in element coordinates stays an honest description of the distribution far
/// longer, since the element flow map is nearly linear where the cartesian one shears; a
/// covariance reported in cartesian is what external formats and observation residuals
/// expect. The transform between them is exact as a linear map - it is what the *linear
/// map itself* means over a long arc that differs.
///
/// # Errors
/// Fails if the covariance is not `(6 + Np) x (6 + Np)`, if the elements are outside
/// their physical domain, or if the orbit is too close to the equinoctial seam.
pub fn covariance_from_equinoctial<F: InertialFrame>(
    elements: &EquinoctialElements,
    cov: &DMatrix<f64>,
) -> KeteResult<DMatrix<f64>> {
    equinoctial_covariance_domain(elements)?;
    congruence(cov, &elements.state_jacobian::<F>()?)
}

/// Sigma-point divergence of converting a cartesian covariance into equinoctial element
/// coordinates: how far the exact nonlinear change of coordinates moves the input
/// Gaussian's sigma points from where the linear conversion puts them, as a Mahalanobis
/// distance inside the converted covariance.
///
/// [`covariance_to_equinoctial`] is exact as a linear map, but a Gaussian in cartesian
/// coordinates is not a Gaussian in element coordinates: the change of chart is nonlinear
/// off the mean, and the linear conversion keeps only its value and first derivative at
/// the mean. This measures what it discards, which grows as the square of the input's
/// width and is invisible to any comparison of the matrices alone.
///
/// The returned value is on the same scale as
/// [`sigma_point_divergence`](crate::state::sigma_point_divergence): how many sigma the
/// exact answer sits from the linear one, inside the converted Gaussian. A value well
/// below a splitting threshold (about 3) means the element Gaussian is a faithful
/// description of the input and any structure that develops later comes from the
/// dynamics, not from the entry conversion. A larger value means the input is already too
/// wide for a single Gaussian in element coordinates and should be split or sampled
/// rather than converted whole. Infinity is returned when a probe point has no element
/// representation at all.
///
/// The probes are placed at `+/- sigma_factor` standard deviations along each eigenvector
/// of the leading `6 x 6` block of `cov`, in frame `F`. Rows and columns beyond the sixth
/// are ignored: force-model parameters convert exactly, so they carry no loss.
///
/// # Errors
/// Fails if `sigma_factor` is not positive and finite, if the covariance is not square
/// and at least `6 x 6` or is zero, or if the orbit is outside the covariance domain, see
/// [`equinoctial_covariance_domain`].
pub fn equinoctial_conversion_divergence<F: InertialFrame>(
    elements: &EquinoctialElements,
    cov: &DMatrix<f64>,
    sigma_factor: f64,
) -> KeteResult<f64> {
    if !sigma_factor.is_finite() || sigma_factor <= 0.0 {
        return Err(Error::ValueError(
            "sigma_factor must be positive and finite".into(),
        ));
    }
    if cov.nrows() < 6 || cov.nrows() != cov.ncols() {
        return Err(Error::ValueError(format!(
            "Covariance must be square and at least 6x6, got {}x{}",
            cov.nrows(),
            cov.ncols()
        )));
    }
    equinoctial_covariance_domain(elements)?;

    let linear = elements.state_jacobian_inverse::<F>()?;
    let cart = Matrix6::from_iterator(cov.view((0, 0), (6, 6)).iter().copied());
    let converted = linear * cart * linear.transpose();

    // The whitening factor for the Mahalanobis norm. The trace-scaled ridge keeps an
    // input that is singular along some directions whitenable; directions the input
    // actually has no extent in contribute nothing, since their probes are skipped.
    let trace = converted.trace();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(Error::ValueError(
            "Covariance must be non-zero and finite.".into(),
        ));
    }
    let ridge = Matrix6::identity() * (trace / 6.0 * 1e-12);
    let whitener = (converted + ridge)
        .cholesky()
        .ok_or_else(|| Error::ValueError("Converted covariance is not positive definite.".into()))?
        .l();

    let nominal: State<F> = elements.try_to_state()?.into_frame();
    let nominal_pos = Vector3::from(nominal.pos);
    let nominal_vel = Vector3::from(nominal.vel);
    // The exact image of a cartesian displacement, as an offset of the element
    // coordinates. A probe whose image does not exist reads as infinite divergence: the
    // input Gaussian reaches configurations the element chart cannot represent, which no
    // finite number understates.
    let exact = |delta: &Vector6<f64>| -> KeteResult<Vector6<f64>> {
        let moved: State<Ecliptic> = State::<F>::new(
            Desig::Empty,
            elements.epoch,
            nominal_pos + Vector3::new(delta[0], delta[1], delta[2]),
            nominal_vel + Vector3::new(delta[3], delta[4], delta[5]),
            elements.center_id,
        )
        .into_frame();
        Ok(elements.offset_to(&EquinoctialElements::from_state(&moved)?))
    };

    let eig = nalgebra::SymmetricEigen::new(cart);
    let mut worst = 0.0_f64;
    for axis in 0..6 {
        let lambda = eig.eigenvalues[axis];
        if lambda <= 0.0 {
            continue;
        }
        let direction: Vector6<f64> = eig.eigenvectors.column(axis).into();
        let probe = direction * (sigma_factor * lambda.sqrt());
        let prediction = linear * probe;
        for sign in [1.0_f64, -1.0] {
            let Ok(offset) = exact(&(probe * sign)) else {
                return Ok(f64::INFINITY);
            };
            let residual = offset - prediction * sign;
            let Some(whitened) = whitener.solve_lower_triangular(&residual) else {
                return Ok(f64::INFINITY);
            };
            worst = worst.max(whitened.norm());
        }
    }
    Ok(worst)
}

/// Reject orbits too close to the equinoctial seam to carry a covariance.
///
/// The seam is the ecliptic retrograde pole, where the pole components grow as
/// `tan(i/2)`. The point representation survives far closer to it than a covariance does:
/// the equilibrated conditioning of the state Jacobian stays flat right up to the seam, but
/// a covariance picks up the square of the inverse Jacobian on top of the dynamic range a
/// fitted orbit covariance already carries. Once `cond(P)` passes `1/eps` the small
/// eigenvalues are not determined at all and no equilibration recovers range that has left
/// `f64`.
///
/// The limit below is `tan(i/2) > 1000`, an inclination of `179.8854` degrees, which is
/// `0.115` degrees from the seam. `equinoctial_seam_covariance_range` sweeps a physically
/// fixed covariance toward the seam and reports where the range is actually lost, so the
/// margin this limit leaves can be checked rather than taken on faith. The solid angle
/// within `psi` of a pole is `(1 - cos psi) / 2` of the sphere, so an isotropic population
/// puts very little inside this limit, and real populations are more concentrated away from
/// retrograde than isotropic.
///
/// Constructing [`EquinoctialElements`] does **not** apply this; only the covariance path
/// does.
///
/// # Errors
/// Fails when the inclination is within `0.115` degrees of `180`.
pub fn equinoctial_covariance_domain(elements: &EquinoctialElements) -> KeteResult<()> {
    /// `tan(i/2)` squared at the limit.
    const SEAM_LIMIT: f64 = 1e6;
    if elements
        .pole_h
        .mul_add(elements.pole_h, elements.pole_k.powi(2))
        > SEAM_LIMIT
    {
        return Err(Error::ValueError(format!(
            "The orbit's inclination is {:.6} degrees, within {:.4} degrees of the \
             ecliptic retrograde pole. A covariance in equinoctial coordinates loses its \
             dynamic range there and the small eigenvalues are not determined.",
            elements.inclination().to_degrees(),
            180.0 - elements.inclination().to_degrees()
        )));
    }
    Ok(())
}

/// `T P T^T` for `T = diag(block, I)`, the shared half of the two conversions above.
///
/// The parameter block is left alone rather than multiplied by an identity, so it comes
/// out bit identical: the force model's free parameters are not elements and no change of
/// state coordinates reaches them. The cross-covariance between the state and the
/// parameters is transformed on one side only, which is what the block structure of
/// `T P T^T` works out to and what [`UncertainState::from_cometary`] already does with the
/// cometary Jacobian.
fn congruence(cov: &DMatrix<f64>, block: &Matrix6<f64>) -> KeteResult<DMatrix<f64>> {
    if cov.nrows() < 6 || cov.nrows() != cov.ncols() {
        return Err(Error::ValueError(format!(
            "Covariance must be square and at least 6x6, got {}x{}",
            cov.nrows(),
            cov.ncols()
        )));
    }
    let np = cov.nrows() - 6;
    let mut out = DMatrix::zeros(cov.nrows(), cov.ncols());

    let state_block = block * cov.view((0, 0), (6, 6)) * block.transpose();
    out.view_mut((0, 0), (6, 6)).copy_from(&state_block);

    if np > 0 {
        let cross = block * cov.view((0, 6), (6, np));
        out.view_mut((0, 6), (6, np)).copy_from(&cross);
        out.view_mut((6, 0), (np, 6)).copy_from(&cross.transpose());
        out.view_mut((6, 6), (np, np))
            .copy_from(&cov.view((6, 6), (np, np)));
    }
    Ok(out)
}

/// Compute the 6x6 Jacobian `d(x,y,z,vx,vy,vz) / d(e,q,tp,node,w,i)`
/// by central finite differences on `CometElements::try_to_state()`.
///
/// The element ordering is: eccentricity, `peri_dist`, `peri_time`,
/// `lon_of_ascending`, `peri_arg`, inclination. Both sides are Ecliptic, which is the
/// frame `CometElements` is defined in.
fn cometary_to_cartesian_jacobian(elements: &CometElements) -> KeteResult<DMatrix<f64>> {
    let mut jac = DMatrix::zeros(6, 6);

    // Central differences are optimal at h ~ eps^(1/3) * scale.
    // Most elements use their own magnitude (floored at 1.0 for near-zero
    // angles).  peri_time is special: its JD value is ~2.5e6, but orbit
    // sensitivity is per-day, so we use an absolute step of eps^(1/3) days.
    // ~6.06e-6
    let eps3 = f64::EPSILON.cbrt();
    let rel = |v: f64| eps3 * v.abs().max(1.0);
    let steps = [
        // eccentricity (dimensionless)
        rel(elements.eccentricity),
        // peri_dist (AU)
        rel(elements.peri_dist),
        // peri_time (days, absolute)
        eps3,
        // lon_of_ascending (rad)
        rel(elements.lon_of_ascending),
        // peri_arg (rad)
        rel(elements.peri_arg),
        // inclination (rad)
        rel(elements.inclination),
    ];

    for col in 0..6 {
        let h = steps[col];

        // For eccentricity near zero, a central difference would perturb
        // to negative e (which is unphysical).  Fall back to a forward
        // difference in that case (O(h) instead of O(h^2), but still
        // adequate for covariance transformation).
        let forward_only = col == 0 && elements.eccentricity < 2.0 * h;

        if forward_only {
            let state_plus = perturb_element(elements, col, h).try_to_state()?;
            let state_nom = elements.try_to_state()?;

            let inv_h = 1.0 / h;
            for row in 0..3 {
                jac[(row, col)] = (state_plus.pos[row] - state_nom.pos[row]) * inv_h;
                jac[(row + 3, col)] = (state_plus.vel[row] - state_nom.vel[row]) * inv_h;
            }
        } else {
            let state_plus = perturb_element(elements, col, h).try_to_state()?;
            let state_minus = perturb_element(elements, col, -h).try_to_state()?;

            let inv_2h = 1.0 / (2.0 * h);
            for row in 0..3 {
                jac[(row, col)] = (state_plus.pos[row] - state_minus.pos[row]) * inv_2h;
                jac[(row + 3, col)] = (state_plus.vel[row] - state_minus.vel[row]) * inv_2h;
            }
        }
    }

    Ok(jac)
}

/// Return a copy of `elements` with the `col`-th element perturbed by `delta`.
///
/// Column mapping: 0=eccentricity, 1=`peri_dist`, 2=`peri_time`,
/// 3=`lon_of_ascending`, 4=`peri_arg`, 5=inclination.
fn perturb_element(elements: &CometElements, col: usize, delta: f64) -> CometElements {
    let mut e = elements.clone();
    match col {
        0 => e.eccentricity += delta,
        1 => e.peri_dist += delta,
        2 => e.peri_time = (e.peri_time.jd + delta).into(),
        3 => e.lon_of_ascending += delta,
        4 => e.peri_arg += delta,
        5 => e.inclination += delta,
        _ => unreachable!("column index must be 0..6"),
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS_SQRT;
    use crate::frames::{Ecliptic, Equatorial, InertialFrame};
    use crate::prelude::Desig;
    use crate::time::Time;

    /// Helper: build a simple Earth-like state for testing.
    fn test_state() -> State<Equatorial> {
        State::new(
            Desig::Name("Test".into()),
            // J2000.0
            2451545.0,
            [1.0, 0.0, 0.0],
            // ~1 AU circular
            [0.0, 0.01720209895, 0.0],
            10,
        )
    }

    use nalgebra::{DVector, Rotation3, Vector3, Vector6};

    /// The same realistic covariance shape, in equinoctial coordinates about the
    /// Ecliptic. The elements' storage frame is Ecliptic, so the cartesian side of the
    /// conversion is too.
    fn equinoctial_case() -> (EquinoctialElements, DMatrix<f64>) {
        let state: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&state).unwrap();
        // Along-track dominated, with two fitted parameters and cross terms. The phase
        // sigma is in radians here rather than days, so it is scaled by the mean motion.
        let sigma = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1.7e-4, 3e-10, 7e-11];
        let mut cov = DMatrix::<f64>::zeros(8, 8);
        for (index, value) in sigma.iter().enumerate() {
            cov[(index, index)] = value * value;
        }
        for (row, col, value) in [(0, 5, 4e-13), (2, 5, -9e-13), (1, 6, 2e-18), (5, 7, 4e-15)] {
            cov[(row, col)] = value;
            cov[(col, row)] = value;
        }
        (elements, cov)
    }

    /// The two equinoctial conversions are inverses, the parameter block is untouched,
    /// and symmetry and positive semi-definiteness survive.
    #[test]
    fn equinoctial_covariance_conversions_round_trip() {
        let (elements, equinoctial) = equinoctial_case();
        let cartesian = covariance_from_equinoctial::<Ecliptic>(&elements, &equinoctial).unwrap();
        let recovered = covariance_to_equinoctial::<Ecliptic>(&elements, &cartesian).unwrap();
        let jacobian = elements.state_jacobian::<Ecliptic>().unwrap();
        let inverse = elements.state_jacobian_inverse::<Ecliptic>().unwrap();

        let relative = (&recovered - &equinoctial).norm() / equinoctial.norm();
        println!("equinoctial covariance round trip: {relative:e} relative");
        assert!(relative < 1e-12, "round trip {relative:e} exceeded 1e-12");

        // The parameter block is copied rather than multiplied by an identity, so it must
        // come back bit for bit rather than merely close.
        for row in 6..8 {
            for col in 6..8 {
                assert_eq!(
                    cartesian[(row, col)],
                    equinoctial[(row, col)],
                    "parameter block moved at ({row}, {col})"
                );
            }
        }

        // Positive semi-definiteness is asserted against the rounding a congruence
        // transform actually carries, `eps ||T||^2 ||P_in||`, which is set by the input
        // covariance's scale rather than the output's. At this conditioning the sign of
        // the smallest eigenvalue is not determined in f64 at all; the check is for a
        // sign flip or a broken transform, which would be orders above this floor.
        let transform_scale = jacobian.norm().max(inverse.norm()).powi(2) * equinoctial.norm();
        let floor = -100.0 * f64::EPSILON * transform_scale;
        for (label, matrix) in [("cartesian", &cartesian), ("recovered", &recovered)] {
            let asymmetry = (matrix - matrix.transpose()).norm() / matrix.norm();
            assert!(asymmetry < 1e-14, "{label} lost symmetry: {asymmetry:e}");

            let eigenvalues = matrix.clone().symmetric_eigenvalues();
            let smallest = eigenvalues.iter().fold(f64::INFINITY, |acc, v| acc.min(*v));
            println!("{label}: smallest eigenvalue {smallest:e}, zero floor {floor:e}");
            assert!(
                smallest > floor,
                "{label} lost positive semi-definiteness: {smallest:e} below {floor:e}"
            );
        }
    }

    /// The cross-covariance between the state and the fitted parameters is transformed on
    /// one side only. Getting this wrong is invisible in the diagonal and in the round
    /// trip, since a wrong-but-consistent convention still inverts itself.
    #[test]
    fn equinoctial_covariance_transforms_cross_terms() {
        let (elements, equinoctial) = equinoctial_case();
        let cartesian = covariance_from_equinoctial::<Ecliptic>(&elements, &equinoctial).unwrap();
        let jac = elements.state_jacobian::<Ecliptic>().unwrap();

        let expected = jac * equinoctial.view((0, 6), (6, 2));
        let actual = cartesian.view((0, 6), (6, 2));
        let relative = (actual - &expected).norm() / expected.norm();
        assert!(relative < 1e-14, "cross block {relative:e} exceeded 1e-14");
        assert!(
            expected.norm() > 0.0,
            "the test case has no cross terms to check"
        );
    }

    /// The semantic check. Everything above is algebra on a matrix and would pass for any
    /// self-consistent but wrong convention.
    ///
    /// The equinoctial covariance is decomposed, its sigma points displaced along the
    /// element coordinates and converted to states, and the sample covariance formed from
    /// those. For a linear map that reproduces `covariance_from_equinoctial` exactly, so
    /// the difference measures the elements' curvature over a spread of one sigma rather
    /// than any error in the transform.
    #[test]
    fn equinoctial_covariance_matches_displaced_sigma_points() {
        let (elements, equinoctial) = equinoctial_case();
        let cartesian = covariance_from_equinoctial::<Ecliptic>(&elements, &equinoctial).unwrap();

        let factor = equinoctial
            .view((0, 0), (6, 6))
            .into_owned()
            .cholesky()
            .expect("the equinoctial covariance is positive definite")
            .l();
        let nominal = elements.try_to_state().unwrap();
        let nominal_pos = Vector3::from(nominal.pos);
        let nominal_vel = Vector3::from(nominal.vel);

        let mut sampled = DMatrix::<f64>::zeros(6, 6);
        for column in 0..6 {
            for sign in [1.0, -1.0] {
                let step = factor.column(column) * sign;
                let moved = elements.displaced_by(&Vector6::from_iterator(step.iter().copied()));
                let state = moved.try_to_state().unwrap();
                let pos = Vector3::from(state.pos) - nominal_pos;
                let vel = Vector3::from(state.vel) - nominal_vel;
                let offset =
                    DVector::from_iterator(6, (0..3).map(|i| pos[i]).chain((0..3).map(|i| vel[i])));
                sampled += (&offset * offset.transpose()) * 0.5;
            }
        }

        let reference = cartesian.view((0, 0), (6, 6));
        let relative = (&sampled - reference).norm() / reference.norm();
        println!("displaced sigma point covariance vs transform: {relative:e} relative");
        assert!(
            relative < 1e-6,
            "sigma point covariance {relative:e} exceeded 1e-6"
        );
    }

    /// The frame parameter on the conversions must carry the covariance's frame, and
    /// [`UncertainState::from_state`] must apply it.
    ///
    /// Everything else about the conversions is checked in the storage frame on both
    /// sides and would pass unchanged if the rotation were transposed, inverted or
    /// missing. Here the same physical distribution is entered twice, once described in
    /// the Ecliptic and once in the Equatorial, and the two must produce the same element
    /// covariance - which they only do if each is rotated by the right amount in the right
    /// direction.
    #[test]
    fn equinoctial_covariance_carries_its_frame() {
        let ecliptic: State<Ecliptic> = test_state().into_frame();
        let equatorial: State<Equatorial> = test_state();

        // One physical covariance, written in each frame. A congruence by the block
        // diagonal rotation is what "the same distribution in another frame" means.
        let mut cov_ecl = DMatrix::<f64>::zeros(6, 6);
        for (i, sigma) in [3e-9, 1e-9, 2e-9, 4e-11, 7e-11, 5e-11].iter().enumerate() {
            cov_ecl[(i, i)] = sigma * sigma;
        }
        cov_ecl[(0, 4)] = 1e-20;
        cov_ecl[(4, 0)] = 1e-20;
        let rotation = Ecliptic::rotation_to_frame::<Equatorial>();
        let mut rot6 = DMatrix::<f64>::zeros(6, 6);
        rot6.view_mut((0, 0), (3, 3)).copy_from(rotation.matrix());
        rot6.view_mut((3, 3), (3, 3)).copy_from(rotation.matrix());
        let cov_eq = &rot6 * &cov_ecl * rot6.transpose();

        // Non-vacuous: the two descriptions must be different matrices.
        let separation = (&cov_eq - &cov_ecl).norm() / cov_ecl.norm();
        println!("ecliptic vs equatorial covariance: {separation:e} relative");
        assert!(separation > 0.1, "the frames are indistinguishable");

        let from_ecliptic = UncertainState::from_state(&ecliptic, &cov_ecl, vec![]).unwrap();
        let from_equatorial = UncertainState::from_state(&equatorial, &cov_eq, vec![]).unwrap();

        let relative = (&from_ecliptic.cov_matrix - &from_equatorial.cov_matrix).norm()
            / from_ecliptic.cov_matrix.norm();
        println!("same distribution entered through two frames: {relative:e} relative");
        assert!(relative < 1e-12, "frame handling differs by {relative:e}");

        // And back out through either frame reproduces what went in.
        let out_ecl = from_ecliptic.cartesian_covariance::<Ecliptic>().unwrap();
        let out_eq = from_ecliptic.cartesian_covariance::<Equatorial>().unwrap();
        assert!((&out_ecl - &cov_ecl).norm() / cov_ecl.norm() < 1e-12);
        assert!((&out_eq - &cov_eq).norm() / cov_eq.norm() < 1e-12);
    }

    /// A state at inclination `180 - psi` degrees, `psi` degrees from the seam.
    ///
    /// `CometElements` is defined by ecliptic angles, so this is deliberately left in the
    /// Ecliptic: converting it to another frame tilts the pole by the obliquity and the
    /// sweep never reaches the seam at all.
    fn seam_state(psi_deg: f64) -> State<Ecliptic> {
        CometElements {
            desig: Desig::Empty,
            epoch: Time::new(2460000.5),
            eccentricity: 0.3,
            peri_dist: 1.0 / 1.3,
            peri_time: Time::new(2459950.5),
            lon_of_ascending: std::f64::consts::FRAC_PI_4,
            peri_arg: std::f64::consts::FRAC_PI_3,
            inclination: (180.0 - psi_deg).to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        }
        .try_to_state()
        .unwrap()
    }

    /// The seam guard fires where the covariance stops being representable and nowhere
    /// sooner, and its message names the inclination rather than the pole components.
    #[test]
    fn equinoctial_covariance_rejects_the_seam() {
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-16;

        // The stated limit is an inclination of 179.8854 degrees.
        for (psi, expected) in [(1.0, true), (0.2, true), (0.1146, true), (0.05, false)] {
            let elements = EquinoctialElements::from_state(&seam_state(psi)).unwrap();
            let result = covariance_to_equinoctial::<Ecliptic>(&elements, &cov);
            assert_eq!(
                result.is_ok(),
                expected,
                "psi = {psi} deg, i = {:.6} deg, tan(i/2) = {:.4e}",
                elements.inclination().to_degrees(),
                elements.pole_h.hypot(elements.pole_k)
            );
            if let Err(err) = result {
                let message = err.to_string();
                assert!(
                    message.contains("inclination"),
                    "the message must name the inclination: {message}"
                );
                println!("psi = {psi} deg rejected: {message}");
            }
        }

        // Both directions of the conversion are guarded.
        let elements = EquinoctialElements::from_state(&seam_state(0.05)).unwrap();
        assert!(covariance_from_equinoctial::<Ecliptic>(&elements, &cov).is_err());
    }

    /// Where a realistic covariance stops surviving the seam, and how much margin the
    /// guard leaves.
    ///
    /// Conditioning and round trip of the *state* Jacobian measure the point
    /// representation, which survives essentially to the seam. A covariance fails
    /// differently: its entries pick up the square of the inverse Jacobian on top of the
    /// dynamic range a fitted orbit covariance already carries, and no equilibration
    /// recovers range that has left `f64`.
    ///
    /// The covariance is held **physically fixed** across the sweep rather than fixed in
    /// element coordinates. One shape is built at 90 degrees inclination and rigidly
    /// rotated onto each target orbit, which is exact because two orbits differing only in
    /// inclination are related by a rotation about the node line. Fixing it in element
    /// coordinates instead makes the measurement circular: the pole components are
    /// magnified near the seam, so a fixed variance in them is a physical uncertainty that
    /// shrinks to nothing as the seam is approached, and the conditioning then does not
    /// move at all.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn equinoctial_seam_covariance_range() {
        // One physical covariance, built at 90 degrees where the coordinates are well
        // conditioned, then carried onto each orbit by the rotation between them.
        const REFERENCE_PSI: f64 = 90.0;

        // Shape and orientation to parts in 1e8, along-track phase 1.7e-4 rad.
        let sigmas = [1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1.7e-4];
        let mut reference = DMatrix::<f64>::zeros(6, 6);
        for (index, value) in sigmas.iter().enumerate() {
            reference[(index, index)] = value * value;
        }

        let base = EquinoctialElements::from_state(&seam_state(REFERENCE_PSI)).unwrap();
        let base_jac = base.state_jacobian::<Ecliptic>().unwrap();
        let base_cov = base_jac * &reference * base_jac.transpose();

        println!(
            "{:>10}  {:>12}  {:>11}  {:>11}  {:>11}  {:>11}",
            "psi (deg)", "i (deg)", "tan(i/2)", "cond(P)", "isotropic", "verdict"
        );
        for &psi in &[
            90.0_f64, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1146, 0.1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4,
        ] {
            let elements = EquinoctialElements::from_state(&seam_state(psi)).unwrap();
            // The rotation carrying the reference orbit onto this one: about the node
            // line by the difference in inclination.
            let node = std::f64::consts::FRAC_PI_4;
            let rot = Rotation3::from_axis_angle(&Vector3::z_axis(), node)
                * Rotation3::from_axis_angle(
                    &Vector3::x_axis(),
                    (REFERENCE_PSI - psi).to_radians(),
                )
                * Rotation3::from_axis_angle(&Vector3::z_axis(), -node);
            let mut rot6 = DMatrix::<f64>::zeros(6, 6);
            rot6.view_mut((0, 0), (3, 3)).copy_from(rot.matrix());
            rot6.view_mut((3, 3), (3, 3)).copy_from(rot.matrix());
            let cartesian = &rot6 * base_cov * rot6.transpose();

            let Some(inverse) = elements.state_jacobian::<Ecliptic>().unwrap().try_inverse() else {
                println!("{psi:>10.1e}   the Jacobian is singular to working precision");
                continue;
            };
            let image = inverse * &cartesian * inverse.transpose();
            if !image.iter().all(|v| v.is_finite()) {
                println!("{psi:>10.1e}   the covariance is not finite");
                continue;
            }

            let eig = image.symmetric_eigenvalues();
            let cond = eig.max() / eig.min().abs().max(f64::MIN_POSITIVE);
            // The failure is not loss of positive semi-definiteness - a negative
            // eigenvalue within `eps` of the matrix's own scale says nothing. It is the
            // dynamic range leaving f64: once `cond(P) > 1/eps` the small eigenvalues are
            // not determined at all.
            let usable = cond < 1.0 / f64::EPSILON;
            let isotropic = (1.0 - psi.to_radians().cos()) / 2.0;
            println!(
                "{psi:>10.1e}  {:>12.6}  {:>11.3e}  {cond:>11.3e}  {isotropic:>11.3e}  {:>11}",
                180.0 - psi,
                elements.pole_h.hypot(elements.pole_k),
                if usable { "usable" } else { "RANGE LOST" }
            );
        }
        println!(
            "  f64 range limit 1/eps = {:.3e}; the guard rejects below psi = 0.1146 deg",
            1.0 / f64::EPSILON
        );
    }

    #[test]
    fn test_new_validates_dimensions() {
        let state = test_state();
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::from_state(&state.clone(), &cov_6x6, vec![]);
        assert!(result.is_ok());

        // Wrong size should fail.
        let cov_7x7 = DMatrix::identity(7, 7) * 1e-8;
        let result = UncertainState::from_state(&state, &cov_7x7, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_with_free_params_validates_dimensions() {
        let state = test_state();
        // 3 free params -> need 9x9.
        let cov_9x9 = DMatrix::identity(9, 9) * 1e-8;
        let result = UncertainState::from_state(&state.clone(), &cov_9x9, vec![1e-8, 2e-8, 3e-8]);
        assert!(result.is_ok());

        // 6x6 with 3 free params should fail.
        let cov_6x6 = DMatrix::identity(6, 6) * 1e-8;
        let result = UncertainState::from_state(&state, &cov_6x6, vec![1e-8, 2e-8, 3e-8]);
        assert!(result.is_err());
    }

    #[test]
    fn test_n_free_params() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-8;
        let us = UncertainState::from_state(&state.clone(), &cov, vec![]).unwrap();
        assert_eq!(us.n_free_params(), 0);

        let cov = DMatrix::identity(9, 9) * 1e-8;
        let us = UncertainState::from_state(&state, &cov, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(us.n_free_params(), 3);
    }

    #[test]
    fn test_sample_no_free_params() {
        let state = test_state();
        let cov = DMatrix::identity(6, 6) * 1e-12;
        let us = UncertainState::from_state(&state.clone(), &cov, vec![]).unwrap();
        let samples: Vec<(State<Equatorial>, Vec<f64>)> = us.sample(100, Some(42)).unwrap();
        assert_eq!(samples.len(), 100);
        for (s, params) in &samples {
            assert!(params.is_empty());
            // Samples should be close to nominal with tiny covariance.
            assert!((s.pos[0] - state.pos[0]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_sample_with_free_params() {
        let state = test_state();
        let nominal_params = vec![1e-8, 2e-8, 3e-8];
        let cov = DMatrix::identity(9, 9) * 1e-16;
        let us = UncertainState::from_state(&state, &cov, nominal_params.clone()).unwrap();
        let samples: Vec<(State<Equatorial>, Vec<f64>)> = us.sample(10, Some(42)).unwrap();
        for (_, params) in &samples {
            assert_eq!(params.len(), 3);
            // Tiny covariance -> sampled params close to nominal.
            for (sampled, nominal) in params.iter().zip(nominal_params.iter()) {
                assert!((sampled - nominal).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_sample_zero_covariance_returns_nominal() {
        let state = test_state();
        // Zero covariance (positive semi-definite, all eigenvalues zero).
        let cov = DMatrix::zeros(6, 6);
        let us = UncertainState::from_state(&state.clone(), &cov, vec![]).unwrap();
        let samples: Vec<(State<Equatorial>, Vec<f64>)> = us.sample(5, Some(42)).unwrap();
        assert_eq!(samples.len(), 5);
        // Every sample should reproduce the nominal state. Not bit for bit: a zero step
        // still goes through the cartesian -> element -> cartesian round trip, which is
        // exact only to a few ulp.
        for (s, _) in &samples {
            for i in 0..3 {
                assert!((s.pos[i] - state.pos[i]).abs() < 1e-14 * state.pos.norm());
                assert!((s.vel[i] - state.vel[i]).abs() < 1e-14 * state.vel.norm());
            }
        }
    }

    #[test]
    fn test_from_cometary_round_trip() {
        // Build a state, convert to cometary elements, then round-trip
        // through from_cometary with an identity-like covariance.
        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.clone().into_frame();
        let elements = CometElements::from_state(&state_ecl).unwrap();

        // Tiny diagonal covariance in element space.
        let cov_elem = DMatrix::identity(6, 6) * 1e-20;
        let us = UncertainState::from_cometary(&elements, &cov_elem, vec![]).unwrap();

        // The recovered state should match the original.
        for i in 0..3 {
            assert!(
                (us.state::<Equatorial>().unwrap().pos[i] - state_eq.pos[i]).abs() < 1e-10,
                "pos[{i}] mismatch: {} vs {}",
                us.state::<Equatorial>().unwrap().pos[i],
                state_eq.pos[i]
            );
            assert!(
                (us.state::<Equatorial>().unwrap().vel[i] - state_eq.vel[i]).abs() < 1e-10,
                "vel[{i}] mismatch: {} vs {}",
                us.state::<Equatorial>().unwrap().vel[i],
                state_eq.vel[i]
            );
        }
    }

    #[test]
    fn test_from_cometary_with_free_params() {
        let state_eq = test_state();
        let state_ecl: State<Ecliptic> = state_eq.into_frame();
        let elements = CometElements::from_state(&state_ecl).unwrap();

        let cov_elem = DMatrix::identity(9, 9) * 1e-20;
        let us =
            UncertainState::from_cometary(&elements, &cov_elem, vec![1e-8, 2e-8, 3e-8]).unwrap();

        assert_eq!(us.cov_matrix.nrows(), 9);
        assert_eq!(us.cov_matrix.ncols(), 9);
        assert_eq!(us.free_params.len(), 3);
    }

    /// Validate the Jacobian by comparing `J * delta_elem` against the
    /// actual Cartesian-space change for a known perturbation.
    ///
    /// Uses a general elliptical orbit (e=0.3, q=1.5 AU, i=20 deg) with
    /// no special symmetries so every Jacobian column is exercised.
    #[test]
    fn test_jacobian_accuracy() {
        let epoch = Time::new(2460000.5);
        let elements = CometElements {
            desig: Desig::Empty,
            epoch,
            eccentricity: 0.3,
            peri_dist: 1.5,
            // 100 days before epoch
            peri_time: Time::new(2459900.5),
            // 45 deg
            lon_of_ascending: std::f64::consts::FRAC_PI_4,
            // 60 deg
            peri_arg: std::f64::consts::FRAC_PI_3,
            inclination: 20.0_f64.to_radians(),
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };

        // Both sides of the cometary Jacobian are Ecliptic, the frame `CometElements` is
        // defined in.
        let jac = cometary_to_cartesian_jacobian(&elements).unwrap();
        let nominal = elements.try_to_state().unwrap();

        // Test each column against an independent central difference at a different step
        // from the one the Jacobian itself uses. A one-sided difference would carry an
        // O(h^2) term the analytic value does not have, which is unbounded relative to any
        // entry whose own derivative is near zero - and a nearly planar orbit has several.
        let elem_names = ["e", "q", "tp", "Omega", "omega", "i"];
        let perturbation = 1e-4;
        let _ = nominal;

        for col in 0..6 {
            let plus = perturb_element(&elements, col, perturbation)
                .try_to_state()
                .unwrap();
            let minus = perturb_element(&elements, col, -perturbation)
                .try_to_state()
                .unwrap();

            // Tolerance is set from the column's own scale rather than from each entry:
            // an entry a thousand times smaller than its neighbours is not determined to
            // the same relative accuracy and requiring it to be is not a real check.
            let scale: f64 = (0..6).map(|r| jac[(r, col)].powi(2)).sum::<f64>().sqrt();
            for row in 0..6 {
                let difference = if row < 3 {
                    plus.pos[row] - minus.pos[row]
                } else {
                    plus.vel[row - 3] - minus.vel[row - 3]
                } / (2.0 * perturbation);

                let err = (jac[(row, col)] - difference).abs();
                assert!(
                    err < 1e-4 * scale + 1e-14,
                    "Jacobian[{row},{}] (d cart / d {}): \
                     analytic={:.6e}, difference={difference:.6e}, err={err:.2e}",
                    col,
                    elem_names[col],
                    jac[(row, col)]
                );
            }
        }
    }

    /// Same Jacobian test but for a nearly planar orbit (`lon_of_ascending`, `peri_arg`
    /// and inclination all near zero) to exercise the step-size floor.
    #[test]
    fn test_jacobian_near_planar_orbit() {
        let epoch = Time::new(2460000.5);
        let elements = CometElements {
            desig: Desig::Empty,
            epoch,
            eccentricity: 0.05,
            peri_dist: 1.0,
            peri_time: Time::new(2459950.5),
            // nearly zero
            lon_of_ascending: 1e-6,
            // nearly zero
            peri_arg: 1e-6,
            // nearly equatorial
            inclination: 1e-4,
            center_id: 10,
            gm_sqrt: GMS_SQRT,
        };

        // Both sides of the cometary Jacobian are Ecliptic, the frame `CometElements` is
        // defined in.
        let jac = cometary_to_cartesian_jacobian(&elements).unwrap();
        let nominal = elements.try_to_state().unwrap();

        let perturbation = 1e-4;
        let _ = nominal;
        for col in 0..6 {
            let plus = perturb_element(&elements, col, perturbation)
                .try_to_state()
                .unwrap();
            let minus = perturb_element(&elements, col, -perturbation)
                .try_to_state()
                .unwrap();

            let scale: f64 = (0..6).map(|r| jac[(r, col)].powi(2)).sum::<f64>().sqrt();
            for row in 0..6 {
                let difference = if row < 3 {
                    plus.pos[row] - minus.pos[row]
                } else {
                    plus.vel[row - 3] - minus.vel[row - 3]
                } / (2.0 * perturbation);

                let err = (jac[(row, col)] - difference).abs();
                assert!(
                    err < 1e-4 * scale + 1e-14,
                    "Near-planar Jacobian[{row},{col}]: analytic={:.6e}, \
                     difference={difference:.6e}, err={err:.2e}",
                    jac[(row, col)]
                );
            }
        }
    }

    /// Conditioning of the cometary reporting boundary as `e -> 0` and `i -> 0`.
    ///
    /// [`EquinoctialElements`] is regular at both, but reporting `sigma_e`, `sigma_omega`
    /// and friends means pushing a covariance through the classical-angle Jacobian, which
    /// is not. This prints how badly, because the claim that the degeneracy "returns at the
    /// reporting boundary" is worth showing rather than asserting.
    ///
    /// `cond(J)` for the cometary Jacobian grows as `1/e` and as `1/i`, diverging outright
    /// at either degeneracy, while `cond(K)` for the equinoctial elements stays flat across
    /// the whole sweep. The last column is the norm of the `peri_arg` row of `J^-1`, the
    /// factor by which a unit cartesian uncertainty inflates into the reported
    /// `sigma_omega`.
    ///
    /// That divergence is correct rather than a defect: the argument of periapsis of a
    /// circular orbit genuinely is undefined, so its uncertainty genuinely is unbounded.
    /// It is a property of the output format. The information is intact in the
    /// equinoctial elements, and only the classical coordinates cannot express it.
    /// Anything added later that reports classical sigmas has to guard on `e` and `i`
    /// rather than return the number this Jacobian produces.
    #[test]
    #[ignore = "diagnostic, prints a table"]
    fn cometary_reporting_boundary_conditioning() {
        let epoch = Time::new(2460000.5);
        println!(
            "{:>10}  {:>10}  {:>12}  {:>12}  {:>12}",
            "e", "i (deg)", "cond(J)", "cond(K)", "sigma_w ratio"
        );
        let cases: [(f64, f64); 10] = [
            // (eccentricity, inclination in degrees)
            (0.3, 20.0),
            (1e-2, 20.0),
            (1e-4, 20.0),
            (1e-6, 20.0),
            (1e-8, 20.0),
            (0.0, 20.0),
            (0.3, 1e-2),
            (0.3, 1e-4),
            (0.3, 1e-6),
            (0.3, 0.0),
        ];
        for &(ecc, incl_deg) in &cases {
            let elements = CometElements {
                desig: Desig::Empty,
                epoch,
                eccentricity: ecc,
                peri_dist: 1.5,
                peri_time: Time::new(2459900.5),
                lon_of_ascending: std::f64::consts::FRAC_PI_4,
                peri_arg: std::f64::consts::FRAC_PI_3,
                inclination: incl_deg.to_radians(),
                center_id: 10,
                gm_sqrt: GMS_SQRT,
            };
            let Ok(jac) = cometary_to_cartesian_jacobian(&elements) else {
                println!("{ecc:>10.1e}  {incl_deg:>10.1e}   construction failed");
                continue;
            };
            let svd = jac.clone().svd(false, false);
            let sv = svd.singular_values;
            let (smax, smin) = (sv.max(), sv.min());
            let cond_j = if smin > 0.0 {
                smax / smin
            } else {
                f64::INFINITY
            };

            // The same orbit through the equinoctial elements, for contrast.
            let equinoctial =
                EquinoctialElements::from_state(&elements.try_to_state().unwrap()).unwrap();
            let k = equinoctial.state_jacobian::<Ecliptic>().unwrap();
            let ksvd = k.svd(false, false);
            let ksv = ksvd.singular_values;
            let cond_k = ksv.max() / ksv.min();

            // How much a unit cartesian uncertainty inflates in the omega row.
            let sigma_w = jac.clone().try_inverse().map_or(f64::INFINITY, |inv| {
                inv.row(4).iter().map(|v| v * v).sum::<f64>().sqrt()
            });

            println!(
                "{ecc:>10.1e}  {incl_deg:>10.1e}  {cond_j:>12.3e}  {cond_k:>12.3e}  {sigma_w:>12.3e}"
            );
        }
    }

    /// The conversion divergence is the size of the quadratic term the linear conversion
    /// drops, so it must scale as the square of the probe distance and linearly with the
    /// width of the input covariance.
    #[test]
    fn conversion_divergence_scales_with_width() {
        let state: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&state).unwrap();

        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-10;
            cov[(i + 3, i + 3)] = 1e-14;
        }

        let d1 = equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov, 1.0).unwrap();
        let d2 = equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov, 2.0).unwrap();
        let ratio = d2 / d1;
        println!("1 sigma {d1:e}, 2 sigma {d2:e}, ratio {ratio}");
        assert!(d1 > 0.0, "divergence must be positive, got {d1:e}");
        assert!(
            (3.5..4.5).contains(&ratio),
            "probe scaling {ratio} is not quadratic"
        );

        // Widening every sigma by 100x scales the divergence by 100x.
        let wide = &cov * 1e4;
        let d_wide = equinoctial_conversion_divergence::<Ecliptic>(&elements, &wide, 1.0).unwrap();
        let ratio = d_wide / d1;
        println!("100x wider: {d_wide:e}, ratio {ratio}");
        assert!((50.0..200.0).contains(&ratio), "width scaling {ratio}");
    }

    /// A well observed orbit converts faithfully; an uncertainty comparable to the orbit
    /// itself does not, and the divergence must separate the two on the splitting scale.
    #[test]
    fn conversion_divergence_separates_faithful_from_lossy() {
        let state: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&state).unwrap();

        // Distinct eigenvalues, so the probe directions are unique and the self-measured
        // comparison below is well posed; with a degenerate eigenspace the probe basis
        // inside it is arbitrary.
        let mut small = DMatrix::<f64>::zeros(6, 6);
        let mut huge = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            #[allow(clippy::cast_precision_loss, reason = "index is at most 2")]
            let scale = (i + 1) as f64;
            small[(i, i)] = scale * 1e-14;
            small[(i + 3, i + 3)] = scale * 1e-18;
            // Sigma of half the orbit radius and a third of the circular speed.
            huge[(i, i)] = 0.25;
            huge[(i + 3, i + 3)] = 2.5e-5;
        }

        let faithful =
            equinoctial_conversion_divergence::<Ecliptic>(&elements, &small, 1.0).unwrap();
        let lossy = equinoctial_conversion_divergence::<Ecliptic>(&elements, &huge, 1.0).unwrap();
        // At three sigma the quadratic term has grown ninefold, which is where an
        // uncertainty this wide crosses the splitting threshold.
        let lossy_tail =
            equinoctial_conversion_divergence::<Ecliptic>(&elements, &huge, 3.0).unwrap();
        println!("faithful {faithful:e}, lossy {lossy:e}, lossy at 3 sigma {lossy_tail:e}");
        assert!(faithful < 1e-2, "well observed case read as {faithful:e}");
        assert!(lossy > 0.1, "orbit-sized uncertainty read as {lossy:e}");
        assert!(
            lossy_tail > 3.0,
            "orbit-sized uncertainty at 3 sigma read as {lossy_tail:e}"
        );

        // The method on the state reports the same quantity for the stored covariance,
        // since the cartesian image it probes is the exact linear image of the storage.
        let us = UncertainState::from_state(&state, &small, vec![]).unwrap();
        let self_measured = us.conversion_divergence(1.0).unwrap();
        let relative = (self_measured - faithful).abs() / faithful;
        println!("self measured {self_measured:e}, relative {relative:e}");
        assert!(relative < 1e-6, "self measurement differs by {relative:e}");
    }

    /// The divergence is a Mahalanobis distance between physical distributions, so the
    /// frame the input covariance is written in must not move it.
    #[test]
    fn conversion_divergence_is_frame_invariant() {
        let ecliptic: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&ecliptic).unwrap();

        // Distinct eigenvalues, so the probe directions are the same physical directions
        // in both frames rather than an arbitrary basis of a degenerate eigenspace.
        let mut cov_ecl = DMatrix::<f64>::zeros(6, 6);
        for (i, v) in [9e-11, 4e-11, 1e-11, 9e-15, 4e-15, 1e-15]
            .iter()
            .enumerate()
        {
            cov_ecl[(i, i)] = *v;
        }
        let rotation = Ecliptic::rotation_to_frame::<Equatorial>();
        let mut rot6 = DMatrix::<f64>::zeros(6, 6);
        rot6.view_mut((0, 0), (3, 3)).copy_from(rotation.matrix());
        rot6.view_mut((3, 3), (3, 3)).copy_from(rotation.matrix());
        let cov_eq = &rot6 * &cov_ecl * rot6.transpose();

        let d_ecl =
            equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov_ecl, 1.0).unwrap();
        let d_eq =
            equinoctial_conversion_divergence::<Equatorial>(&elements, &cov_eq, 1.0).unwrap();
        let relative = (d_ecl - d_eq).abs() / d_ecl;
        println!("ecliptic {d_ecl:e}, equatorial {d_eq:e}, relative {relative:e}");
        assert!(relative < 1e-6, "frame dependence {relative:e}");
    }

    /// Force-model parameters convert exactly, so the parameter block must not move the
    /// divergence at all.
    #[test]
    fn conversion_divergence_ignores_the_parameter_block() {
        let state: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&state).unwrap();

        let mut cov6 = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov6[(i, i)] = 1e-10;
            cov6[(i + 3, i + 3)] = 1e-14;
        }
        let mut cov8 = DMatrix::<f64>::zeros(8, 8);
        cov8.view_mut((0, 0), (6, 6)).copy_from(&cov6);
        cov8[(6, 6)] = 1e-16;
        cov8[(7, 7)] = 1e-16;
        cov8[(0, 6)] = 1e-14;
        cov8[(6, 0)] = 1e-14;

        let d6 = equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov6, 1.0).unwrap();
        let d8 = equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov8, 1.0).unwrap();
        assert_eq!(d6, d8, "the parameter block moved the divergence");
    }

    /// Input validation of the entry measurement.
    #[test]
    fn conversion_divergence_validates_inputs() {
        let state: State<Ecliptic> = test_state().into_frame();
        let elements = EquinoctialElements::from_state(&state).unwrap();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                equinoctial_conversion_divergence::<Ecliptic>(&elements, &cov, bad).is_err(),
                "sigma_factor {bad} was accepted"
            );
        }
        let wrong_shape = DMatrix::<f64>::identity(5, 5);
        assert!(
            equinoctial_conversion_divergence::<Ecliptic>(&elements, &wrong_shape, 1.0).is_err()
        );
        let zero = DMatrix::<f64>::zeros(6, 6);
        assert!(equinoctial_conversion_divergence::<Ecliptic>(&elements, &zero, 1.0).is_err());
    }
}
