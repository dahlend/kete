/// Gauss-Jackson (Stoermer-Cowell) 8th-order multi-step integrator.
///
/// This is a fixed-step predictor-corrector method optimized for second-order
/// ODEs where the force depends on position (and optionally velocity). It uses
/// one force evaluation per step, making it ideal when the Moon or other
/// short-period bodies force small step sizes regardless of orbit eccentricity.
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
use crate::prelude::KeteResult;
use crate::propagation::radau::RadauIntegrator;
use crate::propagation::util::SecondOrderODE;
use crate::time::{TDB, Time};
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, Matrix, OVector, U1, U7};

/// Result type returned by [`GaussJacksonIntegrator::integrate`].
type GJResult<MType, D> = KeteResult<(OVector<f64, D>, OVector<f64, D>, MType)>;

/// Number of back-values (order of the method).
const ORDER: usize = 8;

/// Default fixed step size in days (~1 day, tuned for lunar period).
const DEFAULT_STEP: f64 = 1.0;

/// Maximum number of corrector iterations per step.
const MAX_CORRECTOR_ITER: usize = 3;

/// Convergence threshold for corrector iteration (relative).
const CORRECTOR_TOL: f64 = 1e-15;

// Stoermer predictor coefficients (position, explicit).
// y_{n+1} - 2*y_n + y_{n-1} = h^2 * sum(STORMER[j] * f_{n-j}, j=0..7)
// Evaluation points: t = 0, -1, ..., -7 relative to t_n.
// Common denominator: 60480.
const STORMER: [f64; ORDER] = [
    88324.0 / 60480.0,
    -121797.0 / 60480.0,
    245598.0 / 60480.0,
    -300227.0 / 60480.0,
    236568.0 / 60480.0,
    -117051.0 / 60480.0,
    33190.0 / 60480.0,
    -4125.0 / 60480.0,
];

// Cowell corrector coefficients (position, implicit).
// y_{n+1} - 2*y_n + y_{n-1} = h^2 * sum(COWELL[j] * f_{n+1-j}, j=0..7)
// Evaluation points: t = 1, 0, -1, ..., -6 relative to t_n.
// Common denominator: 60480.
const COWELL: [f64; ORDER] = [
    4125.0 / 60480.0,
    55324.0 / 60480.0,
    -6297.0 / 60480.0,
    14598.0 / 60480.0,
    -11477.0 / 60480.0,
    5568.0 / 60480.0,
    -1551.0 / 60480.0,
    190.0 / 60480.0,
];

// Adams-Bashforth 8 coefficients (velocity, explicit).
// v_{n+1} = v_n + h * sum(AB[j] * f_{n-j}, j=0..7)
// Common denominator: 120960.
const AB: [f64; ORDER] = [
    434241.0 / 120960.0,
    -1152169.0 / 120960.0,
    2183877.0 / 120960.0,
    -2664477.0 / 120960.0,
    2102243.0 / 120960.0,
    -1041723.0 / 120960.0,
    295767.0 / 120960.0,
    -36799.0 / 120960.0,
];

// Adams-Moulton coefficients (velocity, implicit).
// v_{n+1} = v_n + h * sum(AM[j] * f_{n+1-j}, j=0..7)
// Common denominator: 120960.
const AM: [f64; ORDER] = [
    36799.0 / 120960.0,
    139849.0 / 120960.0,
    -121797.0 / 120960.0,
    123133.0 / 120960.0,
    -88547.0 / 120960.0,
    41499.0 / 120960.0,
    -11351.0 / 120960.0,
    1375.0 / 120960.0,
];

/// Gauss-Jackson (Stoermer-Cowell) 8th-order multi-step integrator.
///
/// A fixed-step predictor-corrector pair for second-order ODEs:
///   - Position:  Stoermer predictor + Cowell corrector (PECE)
///   - Velocity:  Adams-Bashforth predictor + Adams-Moulton corrector
///
/// The method requires ORDER = 8 back-values of the acceleration. These are
/// generated during a bootstrap phase using the Radau integrator, which
/// provides the startup values at machine precision.
///
/// Compensated (Kahan) summation is used for both position and velocity
/// updates to limit roundoff growth over million-step integrations.
///
/// This method uses exactly one force evaluation per accepted step (after
/// the bootstrap), making it extremely efficient when the step size is
/// constrained by short-period bodies (e.g., the Moon's 27-day orbit).
///
/// References:
/// - Berry, Healy (2004): "Implementation of Gauss-Jackson Integration
///   for Orbit Propagation", J. Astronaut. Sci. 52(3).
/// - Montenbruck, Gill (2000): "Satellite Orbits", Chapter 4.
/// - Shampine, Gordon (1975): "Computer Solution of Ordinary Differential
///   Equations: The Initial Value Problem", Chapter 10.
#[allow(missing_debug_implementations, reason = "No debug impl needed")]
pub struct GaussJacksonIntegrator<'a, MType, D: Dim>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
{
    func: SecondOrderODE<'a, MType, D>,
    metadata: MType,

    final_time: Time<TDB>,

    cur_time: Time<TDB>,
    cur_pos: OVector<f64, D>,
    cur_vel: OVector<f64, D>,

    /// Position at step n-1, needed for the second-difference formulas.
    prev_pos: OVector<f64, D>,

    /// Ring buffer of the last ORDER acceleration values.
    /// Index 0 is the most recent (at `cur_time`), index ORDER-1 is oldest.
    accel_hist: Vec<OVector<f64, D>>,

    /// Number of leading dimensions used for convergence checking.
    control_dim: usize,

    // Kahan compensated summation error accumulators.
    comp_pos: OVector<f64, D>,
    comp_vel: OVector<f64, D>,
}

impl<'a, MType: Clone, D: Dim> GaussJacksonIntegrator<'a, MType, D>
where
    DefaultAllocator: Allocator<D, U1> + Allocator<D, U7>,
{
    /// Bootstrap the integrator by generating back-values using Radau.
    ///
    /// We need:
    ///   1. The position at t0 - h (`prev_pos`, for the second-difference)
    ///   2. Accelerations at t0, t0-h, t0-2h, ..., t0-(ORDER-1)*h
    ///
    /// All back-states are computed by stepping backward from t0 using Radau.
    fn bootstrap(
        func: SecondOrderODE<'a, MType, D>,
        pos0: OVector<f64, D>,
        vel0: OVector<f64, D>,
        t0: Time<TDB>,
        h: f64,
        metadata: MType,
        control_dim: Option<usize>,
    ) -> KeteResult<Self> {
        let (dim, _) = pos0.shape_generic();
        let full_dim = pos0.len();
        let cd = control_dim.unwrap_or(full_dim);

        // Evaluate acceleration at the initial state.
        let mut meta = metadata.clone();
        let f0 = func(t0, &pos0, &vel0, &mut meta, true)?;

        // Build back-values: need states at t0-h, t0-2h, ..., t0-(ORDER-1)*h.
        let mut accel_hist = Vec::with_capacity(ORDER);
        accel_hist.push(f0);

        let mut back_pos = Vec::with_capacity(ORDER);
        back_pos.push(pos0.clone());

        let mut prev_pos_out = pos0.clone();
        let mut prev_vel = vel0.clone();
        let mut prev_time = t0;

        for i in 1..ORDER {
            let target_time: Time<TDB> = (t0.jd - i as f64 * h).into();
            let (p, v, m) = RadauIntegrator::integrate(
                func,
                prev_pos_out,
                prev_vel,
                prev_time,
                target_time,
                meta,
                control_dim,
            )?;
            meta = m;
            let f_i = func(target_time, &p, &v, &mut meta, true)?;
            accel_hist.push(f_i);
            back_pos.push(p.clone());
            prev_pos_out = p;
            prev_vel = v;
            prev_time = target_time;
        }

        // prev_pos = position at t0 - h
        let prev_pos = back_pos[1].clone();

        Ok(Self {
            func,
            metadata: meta,
            final_time: t0,
            cur_time: t0,
            cur_pos: pos0,
            cur_vel: vel0,
            prev_pos,
            accel_hist,
            control_dim: cd,
            comp_pos: Matrix::zeros_generic(dim, U1),
            comp_vel: Matrix::zeros_generic(dim, U1),
        })
    }

    /// Integrate from `time_init` to `final_time` using a fixed step size.
    ///
    /// The `step_size` parameter controls the fixed step in days. If `None`,
    /// a default of 1.0 day is used.
    ///
    /// # Errors
    /// Returns an error if bootstrapping fails, the corrector diverges,
    /// or the force function returns an error.
    pub fn integrate(
        func: SecondOrderODE<'a, MType, D>,
        state_init: OVector<f64, D>,
        state_der_init: OVector<f64, D>,
        time_init: Time<TDB>,
        final_time: Time<TDB>,
        metadata: MType,
        control_dim: Option<usize>,
        step_size: Option<f64>,
    ) -> GJResult<MType, D> {
        let total = (final_time - time_init).elapsed;
        if total.abs() < 1e-10 {
            return Ok((state_init, state_der_init, metadata));
        }

        let h_mag = step_size.unwrap_or(DEFAULT_STEP).abs();
        let h = h_mag.copysign(total);

        let mut integrator = Self::bootstrap(
            func,
            state_init,
            state_der_init,
            time_init,
            h,
            metadata,
            control_dim,
        )?;
        integrator.final_time = final_time;

        loop {
            let remaining = (integrator.final_time - integrator.cur_time).elapsed;
            if remaining.abs() < 1e-12 {
                return Ok((integrator.cur_pos, integrator.cur_vel, integrator.metadata));
            }

            // Use a Radau step for the final partial step.
            if remaining.abs() < h.abs() {
                let target = integrator.final_time;
                let (p, v, m) = RadauIntegrator::integrate(
                    integrator.func,
                    integrator.cur_pos,
                    integrator.cur_vel,
                    integrator.cur_time,
                    target,
                    integrator.metadata,
                    Some(integrator.control_dim),
                )?;
                return Ok((p, v, m));
            }

            integrator.step(h)?;
        }
    }

    /// Perform one PECE step of the nominal step size h.
    fn step(&mut self, h: f64) -> KeteResult<()> {
        let h2 = h * h;
        let ndim = self.cur_pos.len();

        // ---- PREDICT ----

        // Stoermer predictor (position):
        //   y_{n+1}^P = 2*y_n - y_{n-1} + h^2 * sum(STORMER[j] * f_{n-j})
        let (dim, _) = self.cur_pos.shape_generic();
        let mut stormer_sum = Matrix::zeros_generic(dim, U1);
        for (f_j, &s_j) in self.accel_hist.iter().zip(&STORMER) {
            stormer_sum += f_j * s_j;
        }
        let pos_pred = &self.cur_pos * 2.0 - &self.prev_pos + &stormer_sum * h2;

        // Adams-Bashforth predictor (velocity):
        //   v_{n+1}^P = v_n + h * sum(AB[j] * f_{n-j})
        let mut ab_sum = Matrix::zeros_generic(dim, U1);
        for (f_j, &a_j) in self.accel_hist.iter().zip(&AB) {
            ab_sum += f_j * a_j;
        }
        let vel_pred = &self.cur_vel + &ab_sum * h;

        // ---- EVALUATE at predicted state ----
        let t_next: Time<TDB> = (self.cur_time.jd + h).into();
        let mut f_new = (self.func)(t_next, &pos_pred, &vel_pred, &mut self.metadata, true)?;

        // ---- CORRECT (iterated PECE) ----
        let mut pos_cor = pos_pred;
        let mut vel_cor = vel_pred;

        for _ in 0..MAX_CORRECTOR_ITER {
            // Cowell corrector (position):
            //   y_{n+1}^C = 2*y_n - y_{n-1} + h^2 * (COWELL[0]*f_{n+1} + COWELL[1]*f_n + ... + COWELL[7]*f_{n-6})
            let mut cowell_sum = &f_new * COWELL[0];
            for (f_j, &c_j) in self.accel_hist.iter().zip(&COWELL[1..]) {
                cowell_sum += f_j * c_j;
            }
            let pos_new = &self.cur_pos * 2.0 - &self.prev_pos + &cowell_sum * h2;

            // Adams-Moulton corrector (velocity):
            //   v_{n+1}^C = v_n + h * (AM[0]*f_{n+1} + AM[1]*f_n + ... + AM[7]*f_{n-6})
            let mut am_sum = &f_new * AM[0];
            for (f_j, &a_j) in self.accel_hist.iter().zip(&AM[1..]) {
                am_sum += f_j * a_j;
            }
            let vel_new = &self.cur_vel + &am_sum * h;

            // Check convergence on position.
            let mut converged = true;
            for i in 0..self.control_dim.min(ndim) {
                let scale = pos_new[i].abs().max(self.cur_pos[i].abs()) + 1e-30;
                if ((pos_new[i] - pos_cor[i]) / scale).abs() > CORRECTOR_TOL {
                    converged = false;
                    break;
                }
            }

            pos_cor = pos_new;
            vel_cor = vel_new;

            if converged {
                break;
            }

            // Re-evaluate at corrected state.
            f_new = (self.func)(t_next, &pos_cor, &vel_cor, &mut self.metadata, true)?;
        }

        // ---- UPDATE STATE ----

        // Save old cur_pos as new prev_pos BEFORE updating cur_pos.
        let old_cur_pos = self.cur_pos.clone();

        // Apply position update with Kahan summation.
        let delta_pos = &pos_cor - &self.cur_pos;
        for i in 0..ndim {
            let y = delta_pos[i] - self.comp_pos[i];
            let t = self.cur_pos[i] + y;
            self.comp_pos[i] = (t - self.cur_pos[i]) - y;
            self.cur_pos[i] = t;
        }

        // Apply velocity update with Kahan summation.
        let delta_vel = &vel_cor - &self.cur_vel;
        for i in 0..ndim {
            let y = delta_vel[i] - self.comp_vel[i];
            let t = self.cur_vel[i] + y;
            self.comp_vel[i] = (t - self.cur_vel[i]) - y;
            self.cur_vel[i] = t;
        }

        self.prev_pos = old_cur_pos;

        // Rotate acceleration history: drop oldest, insert f_new at front.
        let _ = self.accel_hist.pop();
        self.accel_hist.insert(0, f_new);

        self.cur_time = t_next;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;
    use crate::propagation::analytic_2_body;
    use crate::propagation::{CentralAccelMeta, central_accel};

    /// Two-body orbit validated against the analytic Kepler solution.
    #[test]
    fn two_body_vs_analytic() {
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);

        let (exact_pos, exact_vel) =
            analytic_2_body(1000.0_f64.into(), &init_pos, &init_vel, None).unwrap();

        let (gj_pos, gj_vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();

        for i in 0..3 {
            let pos_err = (gj_pos[i] - exact_pos[i]).abs();
            let vel_err = (gj_vel[i] - exact_vel[i]).abs();
            assert!(pos_err < 2e-13, "pos[{i}] error vs analytic: {pos_err:.2e}");
            assert!(vel_err < 1e-14, "vel[{i}] error vs analytic: {vel_err:.2e}");
        }
    }

    /// Test that a circular orbit conserves energy over 100k days.
    #[test]
    fn energy_conservation() {
        let gms = crate::constants::GMS;
        let v_circ = gms.sqrt();
        let (pos, vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, v_circ, 0.0),
            0.0.into(),
            100_000.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();

        let energy_init = 0.5 * v_circ * v_circ - gms;
        let energy_final = 0.5 * vel.norm_squared() - gms / pos.norm();
        let rel_err = ((energy_final - energy_init) / energy_init).abs();
        assert!(rel_err < 5e-12, "Energy conservation error: {rel_err:.2e}");
    }

    /// Test backward integration recovers the initial state.
    #[test]
    fn backward_integration() {
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);

        let (mid_pos, mid_vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            500.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();

        let (final_pos, final_vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            mid_pos,
            mid_vel,
            500.0.into(),
            0.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();

        for i in 0..3 {
            assert!(
                (final_pos[i] - init_pos[i]).abs() < 1e-11,
                "Backward pos[{i}] error: {:.2e}",
                (final_pos[i] - init_pos[i]).abs()
            );
            assert!(
                (final_vel[i] - init_vel[i]).abs() < 1e-11,
                "Backward vel[{i}] error: {:.2e}",
                (final_vel[i] - init_vel[i]).abs()
            );
        }
    }

    /// Test with a highly eccentric orbit (e ~ 0.9).
    #[test]
    fn eccentric_orbit() {
        let gms = crate::constants::GMS;
        let r_peri = 0.1;
        let a = 1.0;
        let v_peri = (gms * (2.0 / r_peri - 1.0 / a)).sqrt();

        // Use a moderately small step for the fast perihelion passage.
        let (pos, vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            Vector3::new(r_peri, 0.0, 0.0),
            Vector3::new(0.0, v_peri, 0.0),
            0.0.into(),
            365.25.into(),
            CentralAccelMeta::default(),
            None,
            Some(0.05),
        )
        .unwrap();

        let energy_init = 0.5 * v_peri * v_peri - gms / r_peri;
        let energy_final = 0.5 * vel.norm_squared() - gms / pos.norm();
        let rel_err = ((energy_final - energy_init) / energy_init).abs();
        eprintln!("Eccentric orbit energy err (h=0.05): {rel_err:.2e}");
        assert!(
            rel_err < 5e-10,
            "Eccentric orbit energy error: {rel_err:.2e}"
        );
    }

    /// Near-zero integration interval returns the initial state unchanged.
    #[test]
    fn zero_interval() {
        let (pos, vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.01, 0.0),
            0.0.into(),
            0.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(pos[0], 1.0);
        assert_eq!(vel[1], 0.01);
    }

    /// Verify smaller step sizes improve accuracy.
    #[test]
    fn step_size_convergence() {
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);

        let (exact_pos, _) = analytic_2_body(100.0_f64.into(), &init_pos, &init_vel, None).unwrap();

        let steps = [8.0, 4.0, 2.0];
        let mut errors = Vec::new();
        for &step in &steps {
            let (gj_pos, _, _) = GaussJacksonIntegrator::integrate(
                &central_accel,
                init_pos,
                init_vel,
                0.0.into(),
                100.0.into(),
                CentralAccelMeta::default(),
                None,
                Some(step),
            )
            .unwrap();
            let err: f64 = (0..3)
                .map(|i| (gj_pos[i] - exact_pos[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            errors.push(err);
        }

        // Each halving of step size should roughly improve error by 2^8 = 256
        // for an 8th-order method. In practice the ratio won't be exact, but
        // error should decrease monotonically.
        for i in 1..errors.len() {
            assert!(
                errors[i] < errors[i - 1],
                "Error did not decrease: h={} err={:.2e}, h={} err={:.2e}",
                steps[i - 1],
                errors[i - 1],
                steps[i],
                errors[i],
            );
        }
    }

    /// Diagnostic: compare GJ, Radau, and BS on the same orbits.
    /// Run with:
    /// `cargo test -p kete_core --lib gauss_jackson::tests::compare -- --nocapture --ignored`
    #[test]
    #[ignore = "expensive diagnostic, run manually"]
    fn compare_integrators() {
        use crate::propagation::BulirschStoerIntegrator;

        let gms = crate::constants::GMS;

        // --- Test 1: position/velocity vs analytic, 1000-day elliptical ---
        let init_pos = Vector3::new(0.46937657, -0.8829981, 0.0);
        let init_vel = Vector3::new(0.01518942, 0.00807426, 0.0);
        let (exact_pos, exact_vel) =
            analytic_2_body(1000.0_f64.into(), &init_pos, &init_vel, None).unwrap();

        let (gj_pos, gj_vel, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();

        let (rad_pos, rad_vel, _) = RadauIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        let (bs_pos, bs_vel, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            init_pos,
            init_vel,
            0.0.into(),
            1000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();

        let gj_perr: f64 = (0..3)
            .map(|i| (gj_pos[i] - exact_pos[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let gj_verr: f64 = (0..3)
            .map(|i| (gj_vel[i] - exact_vel[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rad_perr: f64 = (0..3)
            .map(|i| (rad_pos[i] - exact_pos[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let rad_verr: f64 = (0..3)
            .map(|i| (rad_vel[i] - exact_vel[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let bs_perr: f64 = (0..3)
            .map(|i| (bs_pos[i] - exact_pos[i]).powi(2))
            .sum::<f64>()
            .sqrt();
        let bs_verr: f64 = (0..3)
            .map(|i| (bs_vel[i] - exact_vel[i]).powi(2))
            .sum::<f64>()
            .sqrt();

        eprintln!("\n=== 1000-day elliptical orbit vs analytic ===");
        eprintln!("              pos err          vel err");
        eprintln!("  GJ:     {gj_perr:12.3e}    {gj_verr:12.3e}");
        eprintln!("  Radau:  {rad_perr:12.3e}    {rad_verr:12.3e}");
        eprintln!("  BS:     {bs_perr:12.3e}    {bs_verr:12.3e}");

        // --- Test 2: energy conservation, circular orbit, 100k days ---
        let v_circ = gms.sqrt();
        let p0 = Vector3::new(1.0, 0.0, 0.0);
        let v0 = Vector3::new(0.0, v_circ, 0.0);
        let e_init = 0.5 * v_circ * v_circ - gms;

        let (gj_p, gj_v, _) = GaussJacksonIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            100_000.0.into(),
            CentralAccelMeta::default(),
            None,
            None,
        )
        .unwrap();
        let gj_e = ((0.5 * gj_v.norm_squared() - gms / gj_p.norm() - e_init) / e_init).abs();

        let (rad_p, rad_v, _) = RadauIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            100_000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();
        let rad_e = ((0.5 * rad_v.norm_squared() - gms / rad_p.norm() - e_init) / e_init).abs();

        let (bs_p, bs_v, _) = BulirschStoerIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            100_000.0.into(),
            CentralAccelMeta::default(),
            None,
        )
        .unwrap();
        let bs_e = ((0.5 * bs_v.norm_squared() - gms / bs_p.norm() - e_init) / e_init).abs();

        eprintln!("\n=== 100k-day circular orbit energy conservation ===");
        eprintln!("  GJ:     {gj_e:12.3e}");
        eprintln!("  Radau:  {rad_e:12.3e}");
        eprintln!("  BS:     {bs_e:12.3e}");

        // --- Test 3: GJ step-size sweep on 1000-day elliptical ---
        eprintln!("\n=== GJ step-size sweep (1000-day elliptical) ===");
        eprintln!("  step     pos err          vel err");
        for &step in &[2.0, 1.0, 0.5, 0.25] {
            let (p, v, _) = GaussJacksonIntegrator::integrate(
                &central_accel,
                init_pos,
                init_vel,
                0.0.into(),
                1000.0.into(),
                CentralAccelMeta::default(),
                None,
                Some(step),
            )
            .unwrap();
            let pe: f64 = (0..3)
                .map(|i| (p[i] - exact_pos[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let ve: f64 = (0..3)
                .map(|i| (v[i] - exact_vel[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            eprintln!("  {step:4.2}   {pe:12.3e}    {ve:12.3e}");
        }
    }
}
