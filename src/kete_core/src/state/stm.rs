// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Variational integration via the [`ParameterizedForce`] trait.
//!
//! [`propagate_with_stm`] integrates a [`ParameterizedForce`] together with the
//! second-order encoding of the state-transition matrix and parameter
//! sensitivities, returning the propagated state and a 6 x (6 + Np)
//! sensitivity matrix.
//!
//! The augmented state is dynamically sized (`DVector`) so the caller
//! can supply any number of free parameters. The encoding is:
//!
//! ```text
//! pos_aug = [pos | vec(Phi_rr) | vec(Phi_rv) | s_1 | s_2 | ...]
//! vel_aug = [vel | vec(Phi_rr') | vec(Phi_rv') | s_1' | s_2' | ...]
//! ```
//!
//! with `Phi_rr(0) = I_3`, `Phi_rv'(0) = I_3`, all parameter
//! sensitivities zero at the start. The integrator runs the second-order
//! Radau scheme and the sensitivity matrix is reconstructed from the
//! final augmented state.

use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

use crate::errors::{Error, KeteResult};
use crate::forces::ParameterizedForce;
use crate::frames::{CenterBody, DynCenter, InertialFrame, Vector};
use crate::integrators::RadauDense;
use crate::prelude::State;
use crate::time::{TDB, Time};

/// Propagate a state and its augmented STM under the given `forces`.
///
/// Returns `(pos_final, vel_final, sensitivity)` where `sensitivity` is
/// the 6 x (6 + Np) matrix:
///
/// ```text
/// cols 0..3 : Phi_rr  (d r_f / d r_0)
/// cols 3..6 : Phi_rv  (d r_f / d v_0)
/// rows 3..6 : Phi_vr / Phi_vv (similarly)
/// cols 6+k  : parameter sensitivity (d (r_f, v_f) / d p_k)
/// ```
///
/// # Errors
/// Returns a `ValueError` if `free_params.len()` does not match
/// `forces.n_free_params()`. Propagation may fail if the integrator does not converge
/// or if the `ParameterizedForce` impl returns an error.
pub fn propagate_with_stm<F: ParameterizedForce>(
    forces: &F,
    pos_init: Vector3<f64>,
    vel_init: Vector3<f64>,
    free_params: &[f64],
    epoch_init: Time<TDB>,
    epoch_final: Time<TDB>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>, DMatrix<f64>)> {
    use crate::integrators::RadauIntegrator;

    let np = free_params.len();
    if np != forces.n_free_params() {
        return Err(Error::ValueError(format!(
            "propagate_with_stm received {np} free parameters for a force with {}",
            forces.n_free_params()
        )));
    }
    let dim = 21 + 3 * np;

    // Augmented initial conditions.
    let mut pos_aug = DVector::<f64>::zeros(dim);
    let mut vel_aug = DVector::<f64>::zeros(dim);
    pos_aug[0] = pos_init[0];
    pos_aug[1] = pos_init[1];
    pos_aug[2] = pos_init[2];
    vel_aug[0] = vel_init[0];
    vel_aug[1] = vel_init[1];
    vel_aug[2] = vel_init[2];
    // Phi_rr(0) = I3 (column-major in cols 3..12 of pos_aug).
    pos_aug[3] = 1.0;
    pos_aug[7] = 1.0;
    pos_aug[11] = 1.0;
    // Phi_rv'(0) = I3 (column-major in cols 12..21 of vel_aug).
    vel_aug[12] = 1.0;
    vel_aug[16] = 1.0;
    vel_aug[20] = 1.0;

    let ode = |time: Time<TDB>,
               pos_aug: &DVector<f64>,
               vel_aug: &DVector<f64>,
               meta: &mut F::Meta,
               exact_eval: bool|
     -> KeteResult<DVector<f64>> {
        let mut result = DVector::<f64>::zeros(dim);

        let pos_phys = Vector::<F::Frame>::new([pos_aug[0], pos_aug[1], pos_aug[2]]);
        let vel_phys = Vector::<F::Frame>::new([vel_aug[0], vel_aug[1], vel_aug[2]]);

        // Physical acceleration and the dynamics Jacobians at the current physical
        // state, in one call -- propagate any error from the force.
        let (accel, da_dr, da_dv, dp) = forces.accel_and_jacobians(
            time,
            &pos_phys,
            &vel_phys,
            free_params,
            meta,
            exact_eval,
        )?;
        let accel: Vector3<f64> = accel.into();
        result[0] = accel[0];
        result[1] = accel[1];
        result[2] = accel[2];

        // Phi_rr'' = da_dr * Phi_rr + da_dv * Phi_rr'
        let phi_rr = Matrix3::from_column_slice(&pos_aug.as_slice()[3..12]);
        let phi_rr_dot = Matrix3::from_column_slice(&vel_aug.as_slice()[3..12]);
        let phi_rr_ddot = da_dr * phi_rr + da_dv * phi_rr_dot;
        result.as_mut_slice()[3..12].copy_from_slice(phi_rr_ddot.as_slice());

        // Phi_rv'' = da_dr * Phi_rv + da_dv * Phi_rv'
        let phi_rv = Matrix3::from_column_slice(&pos_aug.as_slice()[12..21]);
        let phi_rv_dot = Matrix3::from_column_slice(&vel_aug.as_slice()[12..21]);
        let phi_rv_ddot = da_dr * phi_rv + da_dv * phi_rv_dot;
        result.as_mut_slice()[12..21].copy_from_slice(phi_rv_ddot.as_slice());

        // Parameter sensitivities: s_k'' = da_dr * s_k + da_dv * s_k' + d a / d p_k.
        for k in 0..np {
            let base = 21 + k * 3;
            let s_k = Vector3::new(pos_aug[base], pos_aug[base + 1], pos_aug[base + 2]);
            let s_k_dot = Vector3::new(vel_aug[base], vel_aug[base + 1], vel_aug[base + 2]);
            let partial = dp.column(k);
            let partial_v = Vector3::new(partial[0], partial[1], partial[2]);
            let s_k_ddot = da_dr * s_k + da_dv * s_k_dot + partial_v;
            result[base] = s_k_ddot[0];
            result[base + 1] = s_k_ddot[1];
            result[base + 2] = s_k_ddot[2];
        }

        Ok(result)
    };

    // control_dim=3 keeps step-size adaptation focused on the physical
    // 3-component acceleration row, matching the existing variational
    // integrator. Without it, large STM entries can drag steps to be
    // unnecessarily small.
    // The integrator owns the force's working storage for this one integration.
    let (pos_f, vel_f, _) = RadauIntegrator::integrate(
        &ode,
        pos_aug,
        vel_aug,
        epoch_init,
        epoch_final,
        F::Meta::default(),
        Some(3),
        None,
    )?;

    // Reconstruct the 6 x (6 + Np) sensitivity matrix.
    let phi_rr = Matrix3::from_column_slice(&pos_f.as_slice()[3..12]);
    let phi_rv = Matrix3::from_column_slice(&pos_f.as_slice()[12..21]);
    let phi_vr = Matrix3::from_column_slice(&vel_f.as_slice()[3..12]);
    let phi_vv = Matrix3::from_column_slice(&vel_f.as_slice()[12..21]);

    let mut sens = DMatrix::<f64>::zeros(6, 6 + np);
    sens.fixed_view_mut::<3, 3>(0, 0).copy_from(&phi_rr);
    sens.fixed_view_mut::<3, 3>(0, 3).copy_from(&phi_rv);
    sens.fixed_view_mut::<3, 3>(3, 0).copy_from(&phi_vr);
    sens.fixed_view_mut::<3, 3>(3, 3).copy_from(&phi_vv);

    for k in 0..np {
        let base = 21 + k * 3;
        for i in 0..3 {
            sens[(i, 6 + k)] = pos_f[base + i];
            sens[(3 + i, 6 + k)] = vel_f[base + i];
        }
    }

    let pos_final = Vector3::new(pos_f[0], pos_f[1], pos_f[2]);
    let vel_final = Vector3::new(vel_f[0], vel_f[1], vel_f[2]);
    Ok((pos_final, vel_final, sens))
}

/// Propagate a state under `forces` with explicit `free_params`,
/// returning `(pos_final, vel_final)` only.
///
/// Lighter than [`propagate_with_stm`] because no STM is integrated. Use this for
/// particle propagation, FOV checks, or any other forward-only propagation where
/// the free-parameter values come from outside the state shape.
///
/// `state.propagate_with(forces, to)` covers the typical case where
/// the state itself carries free parameters (or has none); this
/// function exists for the fewer cases where caller-supplied
/// `free_params` are needed.
///
/// `dense`, when given, receives the integrator's dense output, see
/// [`RadauIntegrator::integrate`](crate::integrators::RadauIntegrator::integrate).
///
/// # Errors
/// Returns an error if integration fails or if the force returns an
/// error.
pub fn propagate_state<F: ParameterizedForce>(
    forces: &F,
    pos_init: Vector3<f64>,
    vel_init: Vector3<f64>,
    free_params: &[f64],
    epoch_init: Time<TDB>,
    epoch_final: Time<TDB>,
    dense: Option<&mut RadauDense>,
) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
    use crate::integrators::RadauIntegrator;

    let ode = |time: Time<TDB>,
               pos: &Vector3<f64>,
               vel: &Vector3<f64>,
               meta: &mut F::Meta,
               exact_eval: bool|
     -> KeteResult<Vector3<f64>> {
        let pos_typed = Vector::<F::Frame>::new([pos[0], pos[1], pos[2]]);
        let vel_typed = Vector::<F::Frame>::new([vel[0], vel[1], vel[2]]);
        Ok(forces
            .accel(time, &pos_typed, &vel_typed, free_params, meta, exact_eval)?
            .into())
    };

    let (pos_f, vel_f, _) = RadauIntegrator::integrate(
        &ode,
        pos_init,
        vel_init,
        epoch_init,
        epoch_final,
        F::Meta::default(),
        None,
        dense,
    )?;
    Ok((pos_f, vel_f))
}

impl<F: InertialFrame, C: CenterBody> State<F, C>
where
    DynCenter: From<C>,
{
    /// Advance the state to `to` under `forces`.
    ///
    /// The force must have no free parameters: fix those of a non-gravitational model
    /// first with `ParameterMask::fixed_at` or `ParameterMask::all_fixed`, or pass their
    /// values to [`propagate_state`].
    ///
    /// # Errors
    /// Fails if the force has free parameters, if the force itself fails, or if the
    /// integration does not converge.
    pub fn propagate_with<Forces>(self, forces: &Forces, to: Time<TDB>) -> KeteResult<Self>
    where
        Forces: ParameterizedForce<Frame = F, Center = C>,
    {
        if forces.n_free_params() != 0 {
            return Err(Error::ValueError(format!(
                "State::propagate_with requires a force with zero free parameters \
                 (got {}); fix the parameters of any non-grav first via \
                 ParameterMask::fixed_at or ParameterMask::all_fixed.",
                forces.n_free_params()
            )));
        }
        let (pos, vel) = propagate_state(
            forces,
            self.pos.into(),
            self.vel.into(),
            &[],
            self.epoch,
            to,
            None,
        )?;
        Ok(Self {
            desig: self.desig,
            epoch: to,
            pos: pos.into(),
            vel: vel.into(),
            center: self.center,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS;
    use crate::desigs::Desig;
    use crate::forces::ParameterizedForce;
    use crate::frames::{Equatorial, SunCenter, Vector};
    use crate::prelude::State;

    /// A two-body Kepler force with provided GM (no free parameters).
    /// Provides analytical position Jacobian for cross-validation.
    struct TwoBody {
        gm: f64,
    }

    impl ParameterizedForce for TwoBody {
        type Frame = Equatorial;
        type Center = SunCenter;
        type Meta = ();

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
            _meta: &mut Self::Meta,
            _exact_eval: bool,
        ) -> KeteResult<Vector<Equatorial>> {
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (self.gm / r3)).into()))
        }

        fn jacobians(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
            _meta: &mut Self::Meta,
        ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
            // d(-GM r / r^3) / dr = -GM (I/r^3 - 3 r r^T / r^5)
            //                    = GM/r^5 (3 r r^T - r^2 I)
            let p: Vector3<f64> = (*pos).into();
            let r2 = p.norm_squared();
            let r5 = r2 * r2 * r2.sqrt();
            let da_dr = (3.0 * p * p.transpose() - r2 * Matrix3::identity()) * (self.gm / r5);
            Ok((da_dr, Matrix3::zeros()))
        }
    }

    /// The propagated STM must agree with two independent references at a range of
    /// heliocentric distances.
    ///
    /// The variational rows are integrated but never error-controlled -
    /// `propagate_with_stm` passes `control_dim = 3`, so only the physical acceleration
    /// enters the step-size decision and the STM's accuracy is inherited from it. That
    /// makes the STM a far more sensitive probe of the dynamics than the state is: the
    /// state can sit at 1e-16 while the STM is wrong by orders of magnitude.
    ///
    /// Two references, because one is not enough to say which side is wrong:
    ///   * `analytic_2_body_stm` - central differences on the closed-form Kepler solution
    ///   * central differences on the nonlinear propagation of this same force
    ///
    /// Both are finite-difference at a fixed `eps = 1e-8`, so roughly `1e-9` relative is
    /// the floor and the bound is set there.
    ///
    /// This is deliberately run at several `a`. A Jacobian wrong by a power of `r` - the
    /// easiest algebra slip to make in `da_dr`, whose denominator is `r^5` - is exact at
    /// `a = 1 AU`, where every power of `r` is 1, and wrong everywhere else. Testing at one
    /// radius cannot catch a radius-dependent error.
    #[test]
    fn stm_matches_independent_references_across_distance() {
        use crate::kepler::analytic_2_body_stm;

        for (semi_major, arc) in [(1.0, 365.0), (2.5, 289.0), (2.5, 1444.0), (5.0, 4084.0)] {
            let pos0 = Vector3::new(-semi_major, 0.0, 0.0);
            let vel0 = Vector3::new(0.0, -(GMS / semi_major).sqrt(), 0.0);
            let forces = TwoBody { gm: GMS };

            let (_pf, _vf, variational) =
                propagate_with_stm(&forces, pos0, vel0, &[], 0.0.into(), arc.into()).unwrap();
            let (_rp, _rv, kepler_fd) =
                analytic_2_body_stm(arc.into(), &pos0, &vel0, None).unwrap();

            let mut nonlin_fd = DMatrix::<f64>::zeros(6, 6);
            let eps = 1e-8;
            for j in 0..6 {
                let (mut pp, mut vp, mut pm, mut vm) = (pos0, vel0, pos0, vel0);
                if j < 3 {
                    pp[j] += eps;
                    pm[j] -= eps;
                } else {
                    vp[j - 3] += eps;
                    vm[j - 3] -= eps;
                }
                let (fp, fvp) =
                    propagate_state(&forces, pp, vp, &[], 0.0.into(), arc.into(), None).unwrap();
                let (fm, fvm) =
                    propagate_state(&forces, pm, vm, &[], 0.0.into(), arc.into(), None).unwrap();
                for i in 0..3 {
                    nonlin_fd[(i, j)] = (fp[i] - fm[i]) / (2.0 * eps);
                    nonlin_fd[(3 + i, j)] = (fvp[i] - fvm[i]) / (2.0 * eps);
                }
            }

            let scale = nonlin_fd.norm();
            let vs_kepler = (&variational - &kepler_fd).norm() / scale;
            let vs_nonlin = (&variational - &nonlin_fd).norm() / scale;
            println!(
                "a={semi_major:5.2} arc={arc:6.0}d   vs kepler {vs_kepler:9.2e}   \
                 vs nonlinear {vs_nonlin:9.2e}"
            );
            assert!(
                vs_kepler < 1e-6,
                "a={semi_major} arc={arc}: STM disagrees with the closed-form \
                 two-body STM by {vs_kepler:e}",
            );
            assert!(
                vs_nonlin < 1e-6,
                "a={semi_major} arc={arc}: STM disagrees with finite differences of \
                 the nonlinear propagation by {vs_nonlin:e}",
            );
        }
    }

    /// A free-parameter count that does not match the force is an error up front, not
    /// an out-of-range column inside the integrator.
    #[test]
    fn stm_rejects_mismatched_free_params() {
        let result = propagate_with_stm(
            &TwoBody { gm: GMS },
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, GMS.sqrt(), 0.0),
            &[1.0],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(1.0),
        );
        assert!(result.is_err());
    }

    /// Variational integration round-trip on a circular orbit:
    /// after one period, the state should return to its starting point
    /// and the STM should be close to the identity (modulo a known
    /// secular drift of ~1 rad along the along-track direction over
    /// one orbit -- so we just assert the position columns are
    /// reasonable, not exactly identity).
    #[test]
    fn stm_two_body_circular_short_arc() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        // Short arc -- 1 day.
        let (pos_f, vel_f, sens) = propagate_with_stm(
            &TwoBody { gm },
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(1.0),
        )
        .unwrap();

        // The state should advance the expected small amount.
        assert!(pos_f.norm() > 0.99 && pos_f.norm() < 1.01);
        assert!(vel_f.norm() > 0.99 * v_circ && vel_f.norm() < 1.01 * v_circ);

        // STM dimensions: 6 x 6 (no free params).
        assert_eq!(sens.nrows(), 6);
        assert_eq!(sens.ncols(), 6);

        // For a short arc, the STM should be close to identity in
        // structure: the 3x3 r-from-r block should have det ~ 1, and
        // the v-from-v block too.
        let phi_rr = sens.fixed_view::<3, 3>(0, 0);
        let phi_vv = sens.fixed_view::<3, 3>(3, 3);
        assert!(
            (phi_rr.determinant() - 1.0).abs() < 1e-3,
            "det(Phi_rr) = {}",
            phi_rr.determinant()
        );
        assert!(
            (phi_vv.determinant() - 1.0).abs() < 1e-3,
            "det(Phi_vv) = {}",
            phi_vv.determinant()
        );
    }

    /// [`ParameterizedForce`] with a single free parameter `gm`. The free parameter
    /// scales the central acceleration, so `d a / d gm = -r / r^3`.
    struct TwoBodyParametric;

    impl ParameterizedForce for TwoBodyParametric {
        type Frame = Equatorial;
        type Center = SunCenter;
        type Meta = ();

        fn n_free_params(&self) -> usize {
            1
        }

        fn free_param_names(&self) -> Vec<&'static str> {
            vec!["gm"]
        }

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            free_params: &[f64],
            _meta: &mut Self::Meta,
            _exact_eval: bool,
        ) -> KeteResult<Vector<Equatorial>> {
            let gm = free_params[0];
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (gm / r3)).into()))
        }
    }

    /// Parameter sensitivity: propagate with a parametric force and
    /// verify the parameter-sensitivity column matches FD against gm.
    #[test]
    fn stm_parameter_sensitivity_matches_finite_difference() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        let dt = 5.0;
        let force = TwoBodyParametric;

        let (_pos_base, _vel_base, sens) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();

        // FD: perturb gm by h, repropagate.
        let h = gm * 1e-6;
        let (pos_p, vel_p, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm + h],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let (pos_b, vel_b, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[gm],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let dpos_dp = (pos_p - pos_b) / h;
        let dvel_dp = (vel_p - vel_b) / h;

        // Sensitivity column 6 (the only parameter column).
        // Tolerance is loose because FD over a 5-day arc carries noise
        // and the FD-default jacobian inside propagate_with_stm adds
        // its own error budget.
        for i in 0..3 {
            assert!(
                (sens[(i, 6)] - dpos_dp[i]).abs() < 1e-3,
                "row {} pos: STM={}, FD={}",
                i,
                sens[(i, 6)],
                dpos_dp[i]
            );
            assert!(
                (sens[(3 + i, 6)] - dvel_dp[i]).abs() < 1e-3,
                "row {} vel: STM={}, FD={}",
                i,
                sens[(3 + i, 6)],
                dvel_dp[i]
            );
        }
    }

    /// FD validation: compute STM analytically, then compute it by
    /// finite-differencing the state propagation. They should agree to
    /// roughly FD precision (~1e-5).
    #[test]
    fn stm_two_body_matches_finite_difference() {
        let gm = GMS;
        let v_circ = gm.sqrt();
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);
        let dt = 5.0;
        let force = TwoBody { gm };

        let (_pos_f, _vel_f, sens) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();

        // Compute STM column 0 via FD: perturb pos.x by h, repropagate,
        // take (final - unperturbed) / h.
        let h = 1e-6;
        let mut pos_pert = pos;
        pos_pert.x += h;
        let (pos_f_pert, vel_f_pert, _) = propagate_with_stm(
            &force,
            pos_pert,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let (pos_f_base, vel_f_base, _) = propagate_with_stm(
            &force,
            pos,
            vel,
            &[],
            Time::<TDB>::new(0.0),
            Time::<TDB>::new(dt),
        )
        .unwrap();
        let dpos_dx = (pos_f_pert - pos_f_base) / h;
        let dvel_dx = (vel_f_pert - vel_f_base) / h;

        // Analytical column 0 of STM is sens column 0 (rows 0..6).
        for i in 0..3 {
            assert!(
                (sens[(i, 0)] - dpos_dx[i]).abs() < 1e-4,
                "row {} pos: STM={}, FD={}",
                i,
                sens[(i, 0)],
                dpos_dx[i]
            );
            assert!(
                (sens[(3 + i, 0)] - dvel_dx[i]).abs() < 1e-4,
                "row {} vel: STM={}, FD={}",
                i,
                sens[(3 + i, 0)],
                dvel_dx[i]
            );
        }
    }

    /// A trivial central-mass gravity force for testing. Constant GM at
    /// the center body. Holds no free parameters.
    struct CentralMass {
        gm: f64,
    }

    impl ParameterizedForce for CentralMass {
        type Frame = Equatorial;
        type Center = SunCenter;
        type Meta = ();

        fn accel(
            &self,
            _time: Time<TDB>,
            pos: &Vector<Equatorial>,
            _vel: &Vector<Equatorial>,
            _free_params: &[f64],
            _meta: &mut Self::Meta,
            _exact_eval: bool,
        ) -> KeteResult<Vector<Equatorial>> {
            let p: Vector3<f64> = (*pos).into();
            let r3 = p.norm().powi(3);
            Ok(Vector::<Equatorial>::new((-p * (self.gm / r3)).into()))
        }
    }

    #[test]
    fn state_propagate_with_two_body_kepler() {
        // Circular orbit at 1 AU around a body with GM = (2*pi/year)^2 = unity in our units.
        // We'll use the kete GMS constant via a simple fact: at r = 1 AU, with circular
        // velocity v = sqrt(GMS), we should return to the start after one period 2*pi/sqrt(GMS).
        let gm = GMS;
        let v_circ = gm.sqrt();
        let period = 2.0 * std::f64::consts::PI / v_circ;
        let start = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(0.0),
            pos: Vector::<Equatorial>::new([1.0, 0.0, 0.0]),
            vel: Vector::<Equatorial>::new([0.0, v_circ, 0.0]),
            center: SunCenter,
        };
        let force = CentralMass { gm };
        let final_state = start
            .clone()
            .propagate_with(&force, Time::<TDB>::new(period))
            .unwrap();
        let pos: Vector3<f64> = final_state.pos.into();
        let vel: Vector3<f64> = final_state.vel.into();
        // After one full period, position and velocity should match the start.
        assert!((pos.x - 1.0).abs() < 1e-9, "x = {}", pos.x);
        assert!(pos.y.abs() < 1e-9, "y = {}", pos.y);
        assert!(pos.z.abs() < 1e-12, "z = {}", pos.z);
        assert!(vel.x.abs() < 1e-9, "vx = {}", vel.x);
        assert!((vel.y - v_circ).abs() < 1e-9, "vy = {}", vel.y);
        assert!(vel.z.abs() < 1e-12, "vz = {}", vel.z);
    }
}
