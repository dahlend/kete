//! Integration tests for the state transition matrix and analytical Jacobians.
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
//

use kete_core::forces::{
    DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, NonGravKind, NonGravMask, ParameterMask,
    a_over_m_from_physical, lambda_0_from_physical,
};
use kete_core::frames::{Equatorial, SSB, Vector};
use kete_core::prelude::{Desig, KeteResult};
use kete_core::propagation::{NBody, compute_state_transition};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use kete_spice::ephemeris::SpiceEphemeris;

/// Test helper: propagate a heliocentric SSB-centered state with
/// optional non-grav values frozen into the force.
fn propagate(
    state: State<Equatorial, SSB>,
    jd_final: Time<TDB>,
    non_grav: Option<&NonGravMask>,
) -> KeteResult<State<Equatorial, SSB>> {
    let eph = SpiceEphemeris::loaded()?;
    match non_grav {
        None => state.propagate_with(&NBody::new(&eph, false), jd_final),
        Some(frozen) => {
            let force = NBody::with_non_grav(&eph, false, Some(frozen.clone()));
            state.propagate_with(&force, jd_final)
        }
    }
}

/// Helper: build a frozen JPL-comet non-grav model.
fn jpl_comet_entry(a1: f64, a2: f64, a3: f64) -> NonGravMask {
    let kind = NonGravKind::JplComet(JplCometNonGrav::standard_comet());
    ParameterMask::all_fixed(kind, vec![a1, a2, a3]).unwrap()
}

/// Helper: build a frozen Dust non-grav model.
fn dust_entry(beta: f64) -> NonGravMask {
    let kind = NonGravKind::Dust(DustNonGrav);
    ParameterMask::all_fixed(kind, vec![beta]).unwrap()
}

/// Helper: create a test state at ~1 AU from the Sun (solar-system barycenter centered).
fn test_state() -> State<Equatorial, SSB> {
    State {
        desig: Desig::Name("Test".into()),
        epoch: 2451545.0.into(),
        pos: [1.0, 0.0, 0.0].into(),
        vel: [0.0, 0.01720209895, 0.0].into(),
        center: SSB,
    }
}

#[test]
fn stm_n_body_finite_difference_validation() {
    kete_spice::test_data::ensure_test_spk();
    // Validate the variational STM against finite-difference-of-trajectory.
    let state = test_state();
    // 30 days
    let jd_final = (2451545.0 + 30.0).into();

    let (_final_state, sens) = compute_state_transition::<_, NonGravKind>(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        None,
    )
    .unwrap();

    // Build STM via finite differences of Radau propagations.
    // eps = 1e-4 AU balances FD truncation (smaller is better) against
    // round-off noise (larger is better) for ~30-day trajectories near
    // 1 AU. Out-of-plane (z) perturbations couple weakly to in-plane
    // motion (cells like dy/dz are O(1e-5)); a smaller eps such as
    // 1e-5 lets FP round-off in the position differences dominate the
    // FD signal at those cells, even though the variational STM is
    // correct to ~5 digits there.
    let eps = 1e-4;

    for col in 0..6 {
        let mut pos_p: [f64; 3] = state.pos.into();
        let mut vel_p: [f64; 3] = state.vel.into();
        let mut pos_m: [f64; 3] = state.pos.into();
        let mut vel_m: [f64; 3] = state.vel.into();
        if col < 3 {
            pos_p[col] += eps;
            pos_m[col] -= eps;
        } else {
            vel_p[col - 3] += eps;
            vel_m[col - 3] -= eps;
        }
        let state_p = State {
            desig: Desig::Name("P".into()),
            epoch: state.epoch,
            pos: pos_p.into(),
            vel: vel_p.into(),
            center: SSB,
        };
        let state_m = State {
            desig: Desig::Name("M".into()),
            epoch: state.epoch,
            pos: pos_m.into(),
            vel: vel_m.into(),
            center: SSB,
        };
        let res_p = propagate(state_p, jd_final, None).unwrap();
        let res_m = propagate(state_m, jd_final, None).unwrap();

        let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
        let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

        for row in 0..6 {
            let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps);
            let var = sens[(row, col)];
            let abs_err = (fd - var).abs();
            let scale = fd.abs().max(1e-10);
            assert!(
                abs_err / scale < 1e-3,
                "STM mismatch at ({}, {}): variational={:.10e}, fd={:.10e}, rel_err={:.4e}",
                row,
                col,
                var,
                fd,
                abs_err / scale
            );
        }
    }
}

#[test]
fn stm_determinant_conservative() {
    kete_spice::test_data::ensure_test_spk();
    // For conservative forces (no non-grav), det(STM) should be ~1.
    let state = test_state();
    let jd_final = (2451545.0 + 30.0).into();

    let (_final_state, sens) = compute_state_transition::<_, NonGravKind>(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        None,
    )
    .unwrap();

    // Extract the 6x6 STM
    let stm = sens.fixed_view::<6, 6>(0, 0);
    let det = stm.determinant();
    assert!(
        (det - 1.0).abs() < 1e-4,
        "STM determinant should be ~1 for conservative forces, got {det}"
    );
}

#[test]
fn stm_jpl_comet_param_sensitivity() {
    kete_spice::test_data::ensure_test_spk();
    // Validate parameter sensitivity columns for JplComet model via finite diffs.
    let a1 = 1e-8;
    let a2 = 1e-9;
    let a3 = 1e-10;
    let model = jpl_comet_entry(a1, a2, a3);
    let state = test_state();
    let jd_final = (2451545.0 + 30.0).into();

    let (_final_state, sens) = compute_state_transition(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        Some((model.inner(), model.fixed_values().unwrap())),
    )
    .unwrap();

    // Finite-difference test for each A parameter.
    // eps_a = 1e-10 is near-optimal: large enough that position differences
    // are well above trajectory round-off (~1e-12 AU), small enough that
    // nonlinear FD truncation is manageable. Cross-coupling terms (e.g.
    // dx/dA3 from out-of-plane A3) have inherently lower FD accuracy because
    // their signal is weak, so we use a combined relative + absolute threshold.
    let eps_a = 1e-10;
    // Absolute tolerance roughly equals the FD noise floor for cross-coupling:
    // trajectory noise ~2e-12 AU / (2 * eps_a) ~= 0.01.
    let abs_tol = 0.05;
    let rel_tol = 1e-2;
    let a_vals = [a1, a2, a3];
    for k in 0..3 {
        let mut a_p = a_vals;
        let mut a_m = a_vals;
        a_p[k] += eps_a;
        a_m[k] -= eps_a;

        let model_p = jpl_comet_entry(a_p[0], a_p[1], a_p[2]);
        let model_m = jpl_comet_entry(a_m[0], a_m[1], a_m[2]);

        let res_p = propagate(state.clone(), jd_final, Some(&model_p)).unwrap();
        let res_m = propagate(state.clone(), jd_final, Some(&model_m)).unwrap();

        let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
        let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

        for row in 0..6 {
            let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps_a);
            let var = sens[(row, 6 + k)];
            let abs_err = (fd - var).abs();
            let scale = fd.abs().max(var.abs()).max(1e-10);
            let threshold = (scale * rel_tol).max(abs_tol);
            assert!(
                abs_err < threshold,
                "Param sensitivity mismatch for A{} at row {}: var={:.8e}, fd={:.8e}, abs_err={:.4e}, thr={:.4e}",
                k + 1,
                row,
                var,
                fd,
                abs_err,
                threshold
            );
        }
    }
}

#[test]
fn stm_dust_param_sensitivity() {
    kete_spice::test_data::ensure_test_spk();
    // Validate parameter sensitivity column for the Dust (beta) model via FD.
    let beta = 0.01;
    let model = dust_entry(beta);

    let state = test_state();
    let jd_final = (2451545.0 + 30.0).into();

    let (_final_state, sens) = compute_state_transition(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        Some((model.inner(), model.fixed_values().unwrap())),
    )
    .unwrap();

    // Sensitivity matrix should be 6x7 (6 state + 1 beta parameter)
    assert_eq!(sens.ncols(), 7, "Expected 6+1 columns for Dust model");

    // Finite-difference perturbation of beta
    let eps_beta = 1e-6;
    let model_p = dust_entry(beta + eps_beta);
    let model_m = dust_entry(beta - eps_beta);

    let res_p = propagate(state.clone(), jd_final, Some(&model_p)).unwrap();
    let res_m = propagate(state.clone(), jd_final, Some(&model_m)).unwrap();

    let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
    let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

    for row in 0..6 {
        let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps_beta);
        // column 6 = beta sensitivity
        let var = sens[(row, 6)];
        let abs_err = (fd - var).abs();
        let scale = fd.abs().max(var.abs()).max(1e-10);
        assert!(
            abs_err / scale < 1e-2,
            "Dust beta sensitivity mismatch at row {}: var={:.8e}, fd={:.8e}, rel={:.4e}",
            row,
            var,
            fd,
            abs_err / scale
        );
    }
}

#[test]
fn stm_long_arc_90_day() {
    kete_spice::test_data::ensure_test_spk();
    // Validate STM over a 90-day arc against finite-difference-of-trajectory.
    let state = test_state();
    // 90 days
    let jd_final = (2451545.0 + 90.0).into();

    let (_final_state, sens) = compute_state_transition::<_, NonGravKind>(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        None,
    )
    .unwrap();

    // Finite-difference validation of each STM column
    let eps = 1e-5;

    for col in 0..6 {
        let mut pos_p: [f64; 3] = state.pos.into();
        let mut vel_p: [f64; 3] = state.vel.into();
        let mut pos_m: [f64; 3] = state.pos.into();
        let mut vel_m: [f64; 3] = state.vel.into();
        if col < 3 {
            pos_p[col] += eps;
            pos_m[col] -= eps;
        } else {
            vel_p[col - 3] += eps;
            vel_m[col - 3] -= eps;
        }
        let state_p = State {
            desig: Desig::Name("P".into()),
            epoch: state.epoch,
            pos: pos_p.into(),
            vel: vel_p.into(),
            center: SSB,
        };
        let state_m = State {
            desig: Desig::Name("M".into()),
            epoch: state.epoch,
            pos: pos_m.into(),
            vel: vel_m.into(),
            center: SSB,
        };
        let res_p = propagate(state_p, jd_final, None).unwrap();
        let res_m = propagate(state_m, jd_final, None).unwrap();

        let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
        let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

        for row in 0..6 {
            let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps);
            let var = sens[(row, col)];
            let abs_err = (fd - var).abs();
            let scale = fd.abs().max(1e-10);
            // Relax tolerance to 1% for a longer arc; FD accuracy degrades
            // over long arcs due to trajectory divergence.
            assert!(
                abs_err / scale < 1e-2,
                "Long-arc STM mismatch at ({}, {}): var={:.10e}, fd={:.10e}, rel={:.4e}",
                row,
                col,
                var,
                fd,
                abs_err / scale
            );
        }
    }

    // Determinant check: should still be ~1 for conservative forces
    let stm = sens.fixed_view::<6, 6>(0, 0);
    let det = stm.determinant();
    assert!(
        (det - 1.0).abs() < 1e-3,
        "Long-arc STM determinant should be ~1, got {det}"
    );
}

fn radiation_test_model() -> NonGravMask {
    // Realistic 1998 KY26-like inputs.
    let pole = Vector::<Equatorial>::from_ra_dec(49_f64.to_radians(), -28_f64.to_radians());
    let flattening = 0.71_f64;
    let a_over_m = a_over_m_from_physical(2000.0, 0.030, flattening);
    let lambda_0 = lambda_0_from_physical(200.0, 0.9, 0.71, flattening, 5.351 / 60.0);
    let kind =
        NonGravKind::Farnocchia(FarnocchiaNonGrav::new(0.52, 0.71, flattening, pole).unwrap());
    ParameterMask::all_fixed(kind, vec![a_over_m, lambda_0]).unwrap()
}

#[test]
fn stm_radiation_param_sensitivity() {
    kete_spice::test_data::ensure_test_spk();
    // Validate the parameter sensitivity columns of the Farnocchia model against
    // finite differences of full-trajectory propagation.
    let model = radiation_test_model();
    let state = test_state();
    let jd_final = (2451545.0 + 30.0).into();

    let (_final_state, sens) = compute_state_transition(
        &SpiceEphemeris::loaded().unwrap(),
        &state,
        jd_final,
        false,
        Some((model.inner(), model.fixed_values().unwrap())),
    )
    .unwrap();

    // Columns 6 and 7 are d/d(a_over_m) and d/d(lambda_0).
    assert_eq!(
        sens.ncols(),
        8,
        "Expected 6+2 columns for FarnocchiaNonGrav"
    );

    // Pick the FD eps per parameter so the trajectory perturbation lies
    // well above the n-body integrator's round-off floor (~1e-12 AU over
    // 30 days).  `a_over_m` is order 1e-8 (m^2/kg) and the radiation
    // acceleration is exactly linear in it, so we perturb by 100% to lift
    // the FD signal off the noise floor.  `lambda_0` is order 1 and
    // mildly nonlinear, so a small relative step is appropriate.
    let base_params = model.fixed_values().unwrap().to_vec();
    let eps_rel = [1.0_f64, 0.1];
    for k in 0..base_params.len() {
        let eps = base_params[k] * eps_rel[k];
        let mut p_plus = base_params.clone();
        let mut p_minus = base_params.clone();
        p_plus[k] += eps;
        p_minus[k] -= eps;

        let model_p = ParameterMask::all_fixed(model.inner().clone(), p_plus).unwrap();
        let model_m = ParameterMask::all_fixed(model.inner().clone(), p_minus).unwrap();

        let res_p = propagate(state.clone(), jd_final, Some(&model_p)).unwrap();
        let res_m = propagate(state.clone(), jd_final, Some(&model_m)).unwrap();

        let vec_p: Vec<f64> = res_p.pos.into_iter().chain(res_p.vel).collect();
        let vec_m: Vec<f64> = res_m.pos.into_iter().chain(res_m.vel).collect();

        for row in 0..6 {
            let fd = (vec_p[row] - vec_m[row]) / (2.0 * eps);
            let var = sens[(row, 6 + k)];
            let abs_err = (fd - var).abs();
            let scale = fd.abs().max(var.abs()).max(1e-12);
            let threshold = (scale * 5e-2).max(1e-10);
            assert!(
                abs_err < threshold,
                "Radiation param {} sensitivity mismatch row {}: var={:.6e}, fd={:.6e}, rel={:.4e}",
                if k == 0 { "a_over_m" } else { "lambda_0" },
                row,
                var,
                fd,
                abs_err / scale
            );
        }
    }
}
