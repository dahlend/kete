// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Tests for the Wisdom-Holman integrator.
//!
//! Each test prints the quantities it measures (orders, drift rates, residuals)
//! alongside the tolerance it asserts, so a run can be inspected rather than
//! just passed or failed.
//!
//! Tests marked `#[ignore]` are long running (1 Myr integrations and
//! throughput measurements), and are run explicitly with:
//! `cargo test --release -p kete_core wisdom_holman -- --ignored --nocapture`

use std::f64::consts::TAU;

use nalgebra::{DVector, Rotation3, Vector3};

use super::{
    EMB_QUAD_J2R2, LostReason, SUN_RADIUS_AU, WisdomHolman, impact_radius, sun_oblateness,
};
use crate::analysis::hill_radius;
use crate::constants::{F0_OVER_C_AU_DAY2, GMS, SUN_J2};
use crate::desigs::Desig;
use crate::forces::{
    DustNonGrav, FarnocchiaNonGrav, GravParams, JplCometNonGrav, NonGravKind, NonGravMask,
    ParameterMask, ParameterizedForce, a_over_m_from_physical, apply_gr_correction,
    radiation_accel,
};
use crate::frames::{Ecliptic, Equatorial, SSB, Vector};
use crate::integrators::RadauIntegrator;
use crate::kepler::analytic_2_body;
use crate::kepler::{compute_peri_dist, compute_semi_major};
use crate::state::State;
use crate::time::{TDB, Time};

const J2000: f64 = 2451545.0;

/// GM of a body in AU^3/Day^2, from the built-in mass table (EMB merged).
fn gm(naif_id: i32) -> f64 {
    GravParams::simplified_planets()
        .iter()
        .find(|p| p.naif_id == naif_id)
        .expect("body missing from mass table")
        .mass
}

fn make_state(id: u32, pos: Vector3<f64>, vel: Vector3<f64>) -> State<Ecliptic, SSB> {
    State::new(Desig::Perm(id), Time::new(J2000), pos, vel, SSB)
}

fn sun() -> State<Ecliptic, SSB> {
    make_state(10, Vector3::zeros(), Vector3::zeros())
}

/// Heliocentric state at perihelion for the given elements, rotated by an
/// inclination about x and then a phase angle about z.
fn peri_pos_vel(a: f64, ecc: f64, inc_deg: f64, phase: f64) -> (Vector3<f64>, Vector3<f64>) {
    let peri_dist = a * (1.0 - ecc);
    let v_peri = (GMS * (1.0 + ecc) / peri_dist).sqrt();
    let rot = |v: Vector3<f64>| -> Vector3<f64> {
        let (sin_i, cos_i) = inc_deg.to_radians().sin_cos();
        let tilted = Vector3::new(v.x, v.y * cos_i - v.z * sin_i, v.y * sin_i + v.z * cos_i);
        let (sin_p, cos_p) = phase.sin_cos();
        Vector3::new(
            tilted.x * cos_p - tilted.y * sin_p,
            tilted.x * sin_p + tilted.y * cos_p,
            tilted.z,
        )
    };
    (
        rot(Vector3::new(peri_dist, 0.0, 0.0)),
        rot(Vector3::new(0.0, v_peri, 0.0)),
    )
}

fn planet(id: u32, a: f64, ecc: f64, inc_deg: f64, phase: f64) -> State<Ecliptic, SSB> {
    let (pos, vel) = peri_pos_vel(a, ecc, inc_deg, phase);
    make_state(id, pos, vel)
}

/// The 8 planet system (EMB in place of Earth+Moon) as parallel states and
/// GMs, heliocentric, with arbitrary fixed orbital phases. Mean elements are
/// adequate here, since the map's properties do not depend on using a real
/// ephemeris epoch.
fn solar_system() -> (Vec<State<Ecliptic, SSB>>, Vec<f64>) {
    (
        vec![
            sun(),
            planet(1, 0.38710, 0.20563, 7.005, 0.0),
            planet(2, 0.72333, 0.00677, 3.395, 2.4),
            planet(3, 1.00000, 0.01671, 0.0, 4.8),
            planet(4, 1.52368, 0.09340, 1.850, 0.9),
            planet(5, 5.20260, 0.04849, 1.303, 3.3),
            planet(6, 9.55491, 0.05551, 2.489, 5.7),
            planet(7, 19.21845, 0.04630, 0.773, 1.8),
            planet(8, 30.11039, 0.00899, 1.770, 4.2),
        ],
        vec![GMS, gm(1), gm(2), gm(3), gm(4), gm(5), gm(6), gm(7), gm(8)],
    )
}

fn outer_solar_system() -> (Vec<State<Ecliptic, SSB>>, Vec<f64>) {
    let (states, gms) = solar_system();
    let keep = |idx: usize| idx == 0 || idx >= 5;
    (
        states
            .into_iter()
            .enumerate()
            .filter_map(|(idx, s)| keep(idx).then_some(s))
            .collect(),
        gms.into_iter()
            .enumerate()
            .filter_map(|(idx, g)| keep(idx).then_some(g))
            .collect(),
    )
}

/// Shift a state list into its own center-of-mass frame so results are
/// directly comparable with an inertial-frame reference integration.
fn to_com_frame(states: &[State<Ecliptic, SSB>], gms: &[f64]) -> Vec<State<Ecliptic, SSB>> {
    let total: f64 = gms.iter().sum();
    let mut com_pos = Vector3::zeros();
    let mut com_vel = Vector3::zeros();
    for (state, gm) in states.iter().zip(gms) {
        com_pos += Vector3::from(state.pos) * *gm;
        com_vel += Vector3::from(state.vel) * *gm;
    }
    com_pos /= total;
    com_vel /= total;
    states
        .iter()
        .map(|state| {
            State::new(
                state.desig.clone(),
                state.epoch,
                Vector3::from(state.pos) - com_pos,
                Vector3::from(state.vel) - com_vel,
                SSB,
            )
        })
        .collect()
}

/// Integrate the system with the WH map for `t_final` days at step `dt` and
/// return the barycentric position of massive body `idx` (0 = Sun).
fn wh_position_after(
    states: &[State<Ecliptic, SSB>],
    gms: &[f64],
    dt: f64,
    t_final: f64,
    idx: usize,
) -> Vector3<f64> {
    let mut sim = WisdomHolman::new(states, gms, &[], &[], dt, false, false, false).unwrap();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test values"
    )]
    let n_steps = (t_final / dt).round() as u64;
    sim.integrate_n_steps(n_steps).unwrap();
    assert!(
        (sim.epoch().jd() - (J2000 + t_final)).abs() < 1e-6,
        "step count did not land on the target time"
    );
    sim.massive_states()[idx].pos.into()
}

/// Reference N-body integration with the 15th order Radau integrator in
/// inertial barycentric coordinates. Returns the position of body `idx`.
fn radau_position_after(
    states: &[State<Ecliptic, SSB>],
    gms: &[f64],
    t_final: f64,
    idx: usize,
) -> Vector3<f64> {
    let n = states.len();
    let gms = gms.to_vec();
    let mut pos0 = DVector::<f64>::zeros(3 * n);
    let mut vel0 = DVector::<f64>::zeros(3 * n);
    for (i, state) in states.iter().enumerate() {
        let p = Vector3::from(state.pos);
        let v = Vector3::from(state.vel);
        for k in 0..3 {
            pos0[3 * i + k] = p[k];
            vel0[3 * i + k] = v[k];
        }
    }
    let func = |_t: Time<TDB>,
                pos: &DVector<f64>,
                _vel: &DVector<f64>,
                _meta: &mut (),
                _exact: bool|
     -> crate::errors::KeteResult<DVector<f64>> {
        let mut accel = DVector::<f64>::zeros(pos.len());
        let n = pos.len() / 3;
        for i in 0..n {
            let ri = Vector3::new(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
            for j in (i + 1)..n {
                let rj = Vector3::new(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
                let sep = rj - ri;
                let r2 = sep.norm_squared();
                let inv_r3 = (r2 * r2.sqrt()).recip();
                for k in 0..3 {
                    accel[3 * i + k] += sep[k] * gms[j] * inv_r3;
                    accel[3 * j + k] -= sep[k] * gms[i] * inv_r3;
                }
            }
        }
        Ok(accel)
    };
    let (pos, _, ()) = RadauIntegrator::integrate(
        &func,
        pos0,
        vel0,
        Time::new(J2000),
        Time::new(J2000 + t_final),
        (),
        None,
        None,
    )
    .unwrap();
    Vector3::new(pos[3 * idx], pos[3 * idx + 1], pos[3 * idx + 2])
}

/// Heliocentric position/velocity of a state relative to a sun state.
fn heliocentric(
    state: &State<Ecliptic, SSB>,
    sun_state: &State<Ecliptic, SSB>,
) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::from(state.pos) - Vector3::from(sun_state.pos),
        Vector3::from(state.vel) - Vector3::from(sun_state.vel),
    )
}

/// Laplace-Runge-Lenz (eccentricity) vector from a heliocentric state.
fn ecc_vector(pos: &Vector3<f64>, vel: &Vector3<f64>) -> Vector3<f64> {
    vel.cross(&pos.cross(vel)) / GMS - pos / pos.norm()
}

/// Sample the massive-system energy every `sample_every` steps for `n_steps`.
/// Returns (relative peak-to-peak amplitude, relative difference between the
/// mean of the first and second halves of the samples).
#[allow(clippy::cast_precision_loss, reason = "test statistics")]
fn energy_drift_stats(
    sim: &mut WisdomHolman<Ecliptic>,
    n_steps: u64,
    sample_every: u64,
) -> (f64, f64) {
    let e0 = sim.energy();
    let mut samples = vec![e0];
    let mut taken = 0;
    while taken < n_steps {
        let chunk = sample_every.min(n_steps - taken);
        sim.integrate_n_steps(chunk).unwrap();
        taken += chunk;
        samples.push(sim.energy());
    }
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let half = samples.len() / 2;
    let mean_first = samples[..half].iter().sum::<f64>() / half as f64;
    let mean_second = samples[half..].iter().sum::<f64>() / (samples.len() - half) as f64;
    (
        (max - min) / e0.abs(),
        (mean_second - mean_first).abs() / e0.abs(),
    )
}

/// Sun + one test particle is a pure Kepler orbit: the jump and kick vanish,
/// so the map should reproduce the analytic two-body solution to solver tolerance.
#[test]
fn kepler_limit() {
    let period = TAU * (1.0 / GMS).sqrt();
    let dt = period / 40.0;
    let n_orbits = 1000_u64;
    let n_steps = n_orbits * 40;
    let t_total = dt * 40_000.0;

    let (pos0, vel0) = peri_pos_vel(1.0, 0.2, 0.0, 0.0);
    let tp = make_state(1000, pos0, vel0);
    let mut sim = WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], dt, false, false, false).unwrap();
    sim.integrate_n_steps(n_steps).unwrap();

    let states = sim.test_particle_states();
    let sun_state = &sim.massive_states()[0];
    let (pos, vel) = heliocentric(&states[0], sun_state);

    let (exact_pos, exact_vel) = analytic_2_body(t_total.into(), &pos0, &vel0).unwrap();
    let pos_err = (pos - exact_pos).norm();
    let vel_err = (vel - exact_vel).norm();

    let energy0 = 0.5 * vel0.norm_squared() - GMS / pos0.norm();
    let energy1 = 0.5 * vel.norm_squared() - GMS / pos.norm();
    let energy_rel = ((energy1 - energy0) / energy0).abs();

    println!("kepler_limit: {n_orbits} orbits, dt = P/40");
    println!("  position error vs analytic: {pos_err:.3e} AU");
    println!("  velocity error vs analytic: {vel_err:.3e} AU/day");
    println!("  relative energy error:      {energy_rel:.3e}");

    // Solver tolerance random walk over 4e4 steps.
    assert!(pos_err < 1e-8, "position error {pos_err:e} exceeds budget");
    assert!(
        energy_rel < 1e-12,
        "energy error {energy_rel:e} exceeds budget"
    );
}

/// The map is second order: global error is O(dt^2). Measured under step halving
/// against a small-step WH reference, plus an independent equations check
/// against Radau.
#[test]
fn convergence_order() {
    let gms = [GMS, gm(5), gm(6)];
    let bodies = to_com_frame(
        &[
            sun(),
            planet(5, 5.20260, 0.04849, 1.303, 3.3),
            planet(6, 9.55491, 0.05551, 2.489, 5.7),
        ],
        &gms,
    );
    let t_final = 3200.0;

    let reference = wh_position_after(&bodies, &gms, 0.25, t_final, 1);
    let step_sizes = [16.0, 8.0, 4.0, 2.0];
    let errors: Vec<f64> = step_sizes
        .iter()
        .map(|&dt| (wh_position_after(&bodies, &gms, dt, t_final, 1) - reference).norm())
        .collect();

    println!("convergence_order: Jupiter position error vs WH(dt=0.25) after {t_final} days");
    for (dt, err) in step_sizes.iter().zip(&errors) {
        println!("  dt = {dt:5.2} d: error = {err:.6e} AU");
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        println!("  measured order: {order:.3}");
        assert!(
            (1.85..=2.15).contains(&order),
            "order {order} outside 2.0 +/- 0.15"
        );
    }

    let radau = radau_position_after(&bodies, &gms, t_final, 1);
    let cross_err = (reference - radau).norm();
    println!("  WH(dt=0.25) vs Radau: {cross_err:.3e} AU");
    assert!(
        cross_err < 1e-7,
        "independent reference disagrees: {cross_err:e} AU"
    );
}

/// The 8 planet system energy stays in a bounded band with
/// no secular trend. Fast version; the long version is `#[ignore]`d below.
#[test]
fn energy_planets() {
    let (states, gms) = solar_system();
    let mut sim = WisdomHolman::new(&states, &gms, &[], &[], 4.0, false, false, false).unwrap();
    let ang_mom0 = sim.angular_momentum();
    let (amplitude, half_drift) = energy_drift_stats(&mut sim, 100_000, 100);
    let ang_mom1 = sim.angular_momentum();
    let ang_rel = (ang_mom1 - ang_mom0).norm() / ang_mom0.norm();

    println!("energy_planets: 8 planets, dt = 4 d, 1e5 steps (~1.1 kyr)");
    println!("  relative energy amplitude (peak-to-peak): {amplitude:.3e}");
    println!("  relative half-mean drift:                 {half_drift:.3e}");
    println!("  relative angular momentum change:         {ang_rel:.3e}");

    assert!(
        amplitude < 1e-6,
        "energy amplitude {amplitude:e} exceeds budget"
    );
    assert!(
        half_drift < 0.5 * amplitude,
        "secular energy drift {half_drift:e} vs amplitude {amplitude:e}"
    );
    assert!(ang_rel < 1e-12, "angular momentum drift {ang_rel:e}");
}

/// Run the reversibility experiment (Sun+Jupiter+Saturn+tp, 5000 steps of
/// 8 days forward, then backward from the evolved states) and return the
/// worst massive-body and test-particle position return errors.
fn reversibility_errors(use_correctors: bool) -> (f64, f64) {
    let gms = [GMS, gm(5), gm(6)];
    let bodies = to_com_frame(
        &[
            sun(),
            planet(5, 5.20260, 0.04849, 1.303, 3.3),
            planet(6, 9.55491, 0.05551, 2.489, 5.7),
        ],
        &gms,
    );
    let (tp_pos, tp_vel) = peri_pos_vel(2.5, 0.1, 5.0, 1.0);
    let tp = make_state(1000, tp_pos, tp_vel);

    let n_steps = 5000_u64;
    let mut forward =
        WisdomHolman::new(&bodies, &gms, &[tp], &[], 8.0, false, false, use_correctors).unwrap();
    forward.integrate_n_steps(n_steps).unwrap();

    let mut backward = WisdomHolman::new(
        &forward.massive_states(),
        &gms,
        &forward.test_particle_states(),
        &[],
        -8.0,
        false,
        false,
        use_correctors,
    )
    .unwrap();
    backward.integrate_n_steps(n_steps).unwrap();

    let final_massive = backward.massive_states();
    let mut worst_massive = 0.0_f64;
    for (start, end) in bodies.iter().zip(&final_massive) {
        let diff = (Vector3::from(start.pos) - Vector3::from(end.pos)).norm();
        worst_massive = worst_massive.max(diff);
    }
    let tp_return = (Vector3::from(backward.test_particle_states()[0].pos) - tp_pos).norm();
    (worst_massive, tp_return)
}

/// The map is time-reversible to the roundoff floor.
#[test]
fn reversibility() {
    let (worst_massive, tp_return) = reversibility_errors(false);

    println!("reversibility: Sun+Jupiter+Saturn+tp, dt = 8 d, 5000 steps each way");
    println!("  max massive body position return error: {worst_massive:.3e} AU");
    println!("  test particle position return error:    {tp_return:.3e} AU");

    assert!(
        worst_massive < 1e-9,
        "massive return error {worst_massive:e}"
    );
    assert!(tp_return < 1e-9, "test particle return error {tp_return:e}");
}

/// The corrector is canonical and near-identity, so the
/// corrected map remains time-reversible to a roundoff-level floor (slightly
/// above the kernel's, from the extra corrector stages).
#[test]
fn corrector_reversibility() {
    let (worst_massive, tp_return) = reversibility_errors(true);

    println!("corrector_reversibility: dt = 8 d, 5000 steps each way, order-17 correctors");
    println!("  max massive body position return error: {worst_massive:.3e} AU");
    println!("  test particle position return error:    {tp_return:.3e} AU");

    assert!(
        worst_massive < 1e-9,
        "massive return error {worst_massive:e}"
    );
    assert!(tp_return < 1e-9, "test particle return error {tp_return:e}");
}

/// The order-17 corrector removes the dominant
/// O(eps) oscillating energy error. Measures the kernel/corrected amplitude
/// ratio directly; a mismatched kernel arrangement or bad coefficients
/// cannot reach the required ratio.
#[test]
fn corrector_improvement() {
    let (states, gms) = outer_solar_system();
    let measure = |use_correctors: bool| -> f64 {
        let mut sim =
            WisdomHolman::new(&states, &gms, &[], &[], 100.0, false, false, use_correctors)
                .unwrap();
        let mut samples = vec![sim.energy()];
        for _ in 0..100 {
            sim.integrate_n_steps(1000).unwrap();
            samples.push(sim.energy());
        }
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        (max - min) / samples[0].abs()
    };

    let kernel = measure(false);
    let corrected = measure(true);
    let ratio = kernel / corrected;

    println!("corrector_improvement: J/S/U/N, dt = 100 d, 1e5 steps");
    println!("  kernel energy amplitude:    {kernel:.3e}");
    println!("  corrected energy amplitude: {corrected:.3e}");
    println!("  improvement ratio:          {ratio:.1}");

    assert!(corrected < 2e-7, "corrected amplitude {corrected:e}");
    assert!(ratio > 20.0, "improvement ratio only {ratio}");
}

/// Largest position difference, in AU, between two runs' massive bodies and
/// test particles.
fn max_state_diff(a: &WisdomHolman<Ecliptic>, b: &WisdomHolman<Ecliptic>) -> f64 {
    let diff = |x: &[State<Ecliptic, SSB>], y: &[State<Ecliptic, SSB>]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| (Vector3::from(p.pos) - Vector3::from(q.pos)).norm())
            .fold(0.0_f64, f64::max)
    };
    diff(&a.massive_states(), &b.massive_states())
        .max(diff(&a.test_particle_states(), &b.test_particle_states()))
}

/// Jupiter, Saturn and a test particle, with correctors, for the tests of
/// when the corrector is applied.
fn corrected_sim() -> WisdomHolman<Ecliptic> {
    let gms = [GMS, gm(5), gm(6)];
    let bodies = [
        sun(),
        planet(5, 5.20260, 0.04849, 1.303, 3.3),
        planet(6, 9.55491, 0.05551, 2.489, 5.7),
    ];
    let (pos, vel) = peri_pos_vel(2.5, 0.1, 5.0, 1.0);
    WisdomHolman::new(
        &bodies,
        &gms,
        &[make_state(1000, pos, vel)],
        &[],
        8.0,
        false,
        false,
        true,
    )
    .unwrap()
}

/// With correctors, the corrector is applied once for a run however it is
/// split into calls: single steps match one batched call to the roundoff of
/// the unfused half drifts, and reading the state between steps, which works
/// on a synchronized copy, leaves the run bit for bit unchanged.
#[test]
fn corrector_applied_once_per_run() {
    let n = 500;
    let mut batched = corrected_sim();
    batched.integrate_n_steps(n).unwrap();
    let mut stepped = corrected_sim();
    let mut read = corrected_sim();
    for _ in 0..n {
        stepped.step().unwrap();
        read.step().unwrap();
        let _ = (
            read.energy(),
            read.massive_states(),
            read.test_particle_states(),
        );
    }
    let split = max_state_diff(&batched, &stepped);
    println!("corrector_applied_once_per_run: batched vs single steps {split:.3e} AU");
    assert!(split < 1e-12, "{split:e}");
    stepped.synchronize().unwrap();
    read.synchronize().unwrap();
    for (a, b) in stepped.massive_states().iter().zip(&read.massive_states()) {
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.vel, b.vel);
    }
    assert_eq!(
        stepped.test_particle_states()[0].pos,
        read.test_particle_states()[0].pos
    );
}

/// A change to the state synchronizes it first, so it acts on the physical
/// state the accessors showed; synchronizing twice is a no-op.
#[test]
fn corrector_synchronizes_before_changes() {
    let mut sim = corrected_sim();
    sim.integrate_n_steps(100).unwrap();
    let shown = sim.test_particle_states();
    let massive = sim.massive_states();
    sim.kick_test_particles(&[Vector3::zeros()]).unwrap();
    assert_eq!(sim.test_particle_states()[0].pos, shown[0].pos);
    assert_eq!(sim.test_particle_states()[0].vel, shown[0].vel);
    sim.synchronize().unwrap();
    for (a, b) in sim.massive_states().iter().zip(&massive) {
        assert_eq!(a.pos, b.pos);
    }
    let removed = sim.remove_test_particles(&[0]).unwrap();
    assert_eq!(removed[0].pos, shown[0].pos);
}

/// The corrector with its adjacent drifts fused matches the plain sequence
/// of elementary operators `Z(a, b) = X(a) Y(-b) X(-2a) Y(b) X(a)`, forward
/// and inverse, to roundoff.
#[test]
fn corrector_fused_drifts_match_operators() {
    let plain = |sim: &mut WisdomHolman<Ecliptic>, inv: f64| {
        let alpha = super::CORRECTOR_A_BASE.sqrt() * sim.dt;
        let dt = sim.dt;
        let mut z = |a: f64, b: f64| {
            sim.kepler(a, None).unwrap();
            sim.perturbation(-b).unwrap();
            sim.kepler(-2.0 * a, None).unwrap();
            sim.perturbation(b).unwrap();
            sim.kepler(a, None).unwrap();
        };
        for (j, b) in super::CORRECTOR_B17.iter().enumerate() {
            z(-((8 - j) as f64) * alpha, -inv * b * dt);
        }
        for (j, b) in super::CORRECTOR_B17.iter().rev().enumerate() {
            z(((j + 1) as f64) * alpha, inv * b * dt);
        }
    };
    for inv in [1.0, -1.0] {
        let mut fused = corrected_sim();
        let mut reference = corrected_sim();
        fused.apply_corrector(inv).unwrap();
        plain(&mut reference, inv);
        let diff = max_state_diff(&fused, &reference);
        println!("corrector_fused_drifts_match_operators: inv = {inv}: {diff:.3e} AU");
        assert!(diff < 1e-13, "{diff:e}");
    }
}

/// Merging the inner planets into the Sun, with the summed GM at their
/// barycenter, follows a distant particle far better than dropping them. The
/// reference keeps all eight planets at a 4 day step. At that step the merge
/// alone is about a thousand times closer than dropping them; at a 100 day
/// step, which the inner planets would not allow, the step's own error
/// dominates and it is still far closer.
#[test]
fn merged_central_mass() {
    let (states, gms) = solar_system();
    let (pos, vel) = peri_pos_vel(40.0, 0.1, 10.0, 2.0);
    let tp = make_state(4000, pos, vel);
    let span = 20_000.0;
    let run = |states: &[State<Ecliptic, SSB>], gms: &[f64], dt: f64| {
        let mut sim = WisdomHolman::new(
            states,
            gms,
            std::slice::from_ref(&tp),
            &[],
            dt,
            false,
            false,
            false,
        )
        .unwrap();
        sim.integrate_to(Time::new(J2000 + span)).unwrap();
        Vector3::from(sim.test_particle_states()[0].pos)
    };
    let reference = run(&states, &gms, 4.0);

    // Sun, Mercury, Venus, EMB and Mars become one body.
    let inner = 0..5;
    let gm_sum: f64 = gms[inner.clone()].iter().sum();
    let weighted = |f: fn(&State<Ecliptic, SSB>) -> Vector3<f64>| {
        inner
            .clone()
            .map(|i| f(&states[i]) * gms[i])
            .sum::<Vector3<f64>>()
            / gm_sum
    };
    let merged_sun = State::new(
        states[0].desig.clone(),
        states[0].epoch,
        weighted(|s| Vector3::from(s.pos)),
        weighted(|s| Vector3::from(s.vel)),
        SSB,
    );
    let mut merged_states = vec![merged_sun];
    merged_states.extend_from_slice(&states[5..]);
    let mut merged_gms = vec![gm_sum];
    merged_gms.extend_from_slice(&gms[5..]);
    let merged = (run(&merged_states, &merged_gms, 100.0) - reference).norm();
    let merged_fine = (run(&merged_states, &merged_gms, 4.0) - reference).norm();

    let mut dropped_states = vec![states[0].clone()];
    dropped_states.extend_from_slice(&states[5..]);
    let mut dropped_gms = vec![GMS];
    dropped_gms.extend_from_slice(&gms[5..]);
    let dropped = (run(&dropped_states, &dropped_gms, 100.0) - reference).norm();

    println!(
        "merged_central_mass: merged {merged_fine:.3e} AU at 4 d, {merged:.3e} AU at 100 d; \
         dropped {dropped:.3e} AU"
    );
    assert!(
        merged_fine < 1e-2 * dropped,
        "merged {merged_fine:e}, dropped {dropped:e}"
    );
    assert!(
        merged < 0.1 * dropped,
        "merged {merged:e}, dropped {dropped:e}"
    );
}

/// The corrector on a pure Kepler system is the identity up to solver
/// roundoff: the kepler-limit budgets must hold unchanged.
#[test]
fn corrector_kepler_limit() {
    let period = TAU * (1.0 / GMS).sqrt();
    let dt = period / 40.0;
    let n_steps = 40_000_u64;
    let t_total = dt * 40_000.0;

    let (pos0, vel0) = peri_pos_vel(1.0, 0.2, 0.0, 0.0);
    let tp = make_state(1000, pos0, vel0);
    let mut sim = WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], dt, false, false, true).unwrap();
    sim.integrate_n_steps(n_steps).unwrap();

    let states = sim.test_particle_states();
    let sun_state = &sim.massive_states()[0];
    let (pos, vel) = heliocentric(&states[0], sun_state);

    let (exact_pos, _) = analytic_2_body(t_total.into(), &pos0, &vel0).unwrap();
    let pos_err = (pos - exact_pos).norm();
    let energy0 = 0.5 * vel0.norm_squared() - GMS / pos0.norm();
    let energy1 = 0.5 * vel.norm_squared() - GMS / pos.norm();
    let energy_rel = ((energy1 - energy0) / energy0).abs();

    println!("corrector_kepler_limit: 1000 orbits, dt = P/40, correctors on");
    println!("  position error vs analytic: {pos_err:.3e} AU");
    println!("  relative energy error:      {energy_rel:.3e}");
    assert!(pos_err < 1e-8, "position error {pos_err:e} exceeds budget");
    assert!(energy_rel < 1e-12, "energy error {energy_rel:e}");
}

/// A test particle and a vanishingly small massive body
/// with the same initial conditions follow the same trajectory.
#[test]
fn test_particle_consistency() {
    let (ast_pos, ast_vel) = peri_pos_vel(2.5, 0.1, 5.0, 1.0);
    let base = vec![sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)];
    let base_gms = [GMS, gm(5)];

    let tp = make_state(1000, ast_pos, ast_vel);
    let mut sim_tp =
        WisdomHolman::new(&base, &base_gms, &[tp], &[], 4.0, false, false, false).unwrap();
    sim_tp.integrate_n_steps(10_000).unwrap();

    let mut with_massive = base.clone();
    with_massive.push(make_state(1000, ast_pos, ast_vel));
    let gms = [GMS, gm(5), 1e-14 * GMS];
    let mut sim_massive =
        WisdomHolman::new(&with_massive, &gms, &[], &[], 4.0, false, false, false).unwrap();
    sim_massive.integrate_n_steps(10_000).unwrap();

    let tp_final: Vector3<f64> = sim_tp.test_particle_states()[0].pos.into();
    let massive_final: Vector3<f64> = sim_massive.massive_states()[2].pos.into();
    let diff = (tp_final - massive_final).norm();

    println!("test_particle_consistency: 10^4 steps at dt = 4 d");
    println!("  position difference: {diff:.3e} AU");
    assert!(diff < 1e-9, "code paths disagree by {diff:e} AU");
}

/// The GR potential reproduces the secular apsidal
/// precession 6 pi GMS / (c^2 a (1 - e^2)) per orbit.
#[test]
fn gr_precession() {
    let a: f64 = 0.387098;
    let ecc = 0.205630;
    let period = TAU * (a.powi(3) / GMS).sqrt();
    let n_orbits = 400.0;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test values"
    )]
    let n_steps = (n_orbits * period).round() as u64; // dt = 1 day
    let (pos0, vel0) = peri_pos_vel(a, ecc, 0.0, 0.0);

    let measure = |include_gr: bool, use_correctors: bool| -> f64 {
        let tp = make_state(1000, pos0, vel0);
        let mut sim = WisdomHolman::new(
            &[sun()],
            &[GMS],
            &[tp],
            &[],
            1.0,
            include_gr,
            false,
            use_correctors,
        )
        .unwrap();
        sim.integrate_n_steps(n_steps).unwrap();
        let sun_state = &sim.massive_states()[0];
        let (pos, vel) = heliocentric(&sim.test_particle_states()[0], sun_state);
        let e_start = ecc_vector(&pos0, &vel0);
        let e_end = ecc_vector(&pos, &vel);
        e_end.y.atan2(e_end.x) - e_start.y.atan2(e_start.x)
    };

    let with_gr = measure(true, false);
    let with_gr_corrected = measure(true, true);
    let without_gr = measure(false, false);
    #[allow(clippy::cast_precision_loss, reason = "test values")]
    let orbits_elapsed = n_steps as f64 / period;
    let expected = 6.0 * std::f64::consts::PI * GMS
        / (crate::constants::C_AU_PER_DAY * crate::constants::C_AU_PER_DAY)
        * orbits_elapsed
        / (a * (1.0 - ecc * ecc));

    println!("gr_precession: Mercury-like orbit, {n_orbits} orbits, dt = 1 d");
    println!("  measured precession with GR:  {with_gr:.6e} rad");
    println!("  with GR and correctors:       {with_gr_corrected:.6e} rad");
    println!("  expected (theory):            {expected:.6e} rad");
    println!("  measured without GR:          {without_gr:.3e} rad");

    assert!(
        (with_gr / expected - 1.0).abs() < 0.01,
        "GR precession off by more than 1%: measured {with_gr:e}, expected {expected:e}"
    );
    assert!(
        (with_gr_corrected / expected - 1.0).abs() < 0.01,
        "corrected GR precession off: measured {with_gr_corrected:e}, expected {expected:e}"
    );
    assert!(
        without_gr.abs() < 1e-6,
        "Newtonian control shows spurious precession {without_gr:e}"
    );
}

/// The GR kick reproduces the Schwarzschild dynamics it splits: a WH run with
/// GR converges at second order to a Radau reference integrating the same
/// shared acceleration. With only the Sun massive the Kepler part of the map
/// is exact, so the error measured here is purely the GR-kick splitting.
#[test]
fn gr_matches_radau() {
    let a = 0.387;
    let ecc = 0.2;
    let (pos0, vel0) = peri_pos_vel(a, ecc, 5.0, 0.7);
    let tp = make_state(1000, pos0, vel0);
    let t_final = 10_000.0;

    let wh_helio = |dt: f64| -> Vector3<f64> {
        let mut sim = WisdomHolman::new(
            &[sun()],
            &[GMS],
            std::slice::from_ref(&tp),
            &[],
            dt,
            true,
            false,
            false,
        )
        .unwrap();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test"
        )]
        let n = (t_final / dt).round() as u64;
        sim.integrate_n_steps(n).unwrap();
        let sun_end = &sim.massive_states()[0];
        heliocentric(&sim.test_particle_states()[0], sun_end).0
    };

    let func = |_t: Time<TDB>,
                pos: &DVector<f64>,
                vel: &DVector<f64>,
                _m: &mut (),
                _e: bool|
     -> crate::errors::KeteResult<DVector<f64>> {
        let p = Vector3::new(pos[0], pos[1], pos[2]);
        let v = Vector3::new(vel[0], vel[1], vel[2]);
        let mut accel = -GMS / p.norm().powi(3) * p;
        apply_gr_correction(&mut accel, &p, &v, GMS);
        Ok(DVector::from_row_slice(&[accel.x, accel.y, accel.z]))
    };
    let (rpos, _, ()) = RadauIntegrator::integrate(
        &func,
        DVector::from_row_slice(&[pos0.x, pos0.y, pos0.z]),
        DVector::from_row_slice(&[vel0.x, vel0.y, vel0.z]),
        Time::new(J2000),
        Time::new(J2000 + t_final),
        (),
        None,
        None,
    )
    .unwrap();
    let radau = Vector3::new(rpos[0], rpos[1], rpos[2]);

    let step_sizes = [8.0, 4.0, 2.0];
    let errors: Vec<f64> = step_sizes
        .iter()
        .map(|&dt| (wh_helio(dt) - radau).norm())
        .collect();
    println!("gr_matches_radau: Mercury-like orbit, {t_final} days");
    for (dt, err) in step_sizes.iter().zip(&errors) {
        println!("  dt = {dt:4.1} d: WH-GR vs Radau {err:.3e} AU");
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        println!("  measured order: {order:.3}");
        assert!(
            (1.7..=2.3).contains(&order),
            "GR kick does not converge to the Schwarzschild dynamics at second order: {order}"
        );
    }
    assert!(
        errors[2] < 1e-7,
        "WH-GR vs Radau floor too high: {:e} AU",
        errors[2]
    );
}

/// With the velocity-dependent GR kick enabled, the 8 planet system keeps its
/// energy in a bounded band with no secular trend (the kick is only
/// approximately symplectic, so this is the empirical gate on that trade),
/// and the total angular momentum oscillates only at the 1PN scale.
#[test]
#[allow(clippy::cast_precision_loss, reason = "test statistics")]
fn gr_invariants() {
    let (states, gms) = solar_system();
    let mut sim = WisdomHolman::new(&states, &gms, &[], &[], 4.0, true, false, false).unwrap();
    let l0 = sim.angular_momentum();
    let l_norm = l0.norm();
    let mut energies = vec![sim.energy()];
    let mut l_devs = vec![0.0_f64];
    for _ in 0..500 {
        sim.integrate_n_steps(100).unwrap();
        energies.push(sim.energy());
        l_devs.push((sim.angular_momentum() - l0).norm() / l_norm);
    }
    let max = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let amplitude = (max - min) / energies[0].abs();
    let half = energies.len() / 2;
    let mean_first = energies[..half].iter().sum::<f64>() / half as f64;
    let mean_second = energies[half..].iter().sum::<f64>() / (energies.len() - half) as f64;
    let half_drift = ((mean_second - mean_first) / energies[0]).abs();
    let l_first = l_devs[..half].iter().copied().fold(0.0_f64, f64::max);
    let l_second = l_devs[half..].iter().copied().fold(0.0_f64, f64::max);

    println!("gr_invariants: 8 planets, dt = 4 d, 5e4 steps, GR on");
    println!("  relative energy amplitude: {amplitude:.3e}");
    println!("  relative half-mean drift:  {half_drift:.3e}");
    println!("  max relative L wander, first half:  {l_first:.3e}");
    println!("  max relative L wander, second half: {l_second:.3e}");
    assert!(amplitude < 1e-6, "energy amplitude {amplitude:e}");
    assert!(half_drift < 0.5 * amplitude, "secular drift {half_drift:e}");
    // The velocity-dependent kick perturbs L by ~3e-15 relative per step
    // (splitting error, not the continuous 1PN oscillation); its quasi
    // periodic accumulation stays below the e-9 scale over this window.
    assert!(l_second < 5e-9, "angular momentum wander {l_second:e}");
    // Bounded, not secular: the second-half wander must not outgrow the
    // first half by more than the oscillation envelope.
    assert!(
        l_second < 2.0 * l_first.max(1e-12),
        "angular momentum wander grows: {l_first:e} -> {l_second:e}"
    );
}

/// A massive body resolving to the Earth-Moon barycenter receives the
/// orbit-averaged lunar quadrupole: on an exactly circular in-ecliptic orbit
/// its angular rate gains `0.75 J2R2_eff / r^2` relative to a plain body, a
/// closed form the map must reproduce. The correction keys on the
/// designation, and the name form must resolve identically to the NAIF id.
#[test]
fn emb_quadrupole() {
    let r0 = 1.0_f64;
    let quad = *EMB_QUAD_J2R2;
    // The coefficient itself: (mu/M) a_m^2 (1 + 3/2 e^2)(1 - 3/2 sin^2 i)/2
    // with table masses is ~3.9e-8 AU^2.
    assert!(
        (3e-8..5e-8).contains(&quad),
        "EMB quadrupole coefficient {quad:e} is far from its physical scale"
    );

    let period = TAU * (r0.powi(3) / GMS).sqrt();
    let dt = period / 100.0;
    let n_steps = 10_000_u64; // 100 orbits
    #[allow(clippy::cast_precision_loss, reason = "test values")]
    let t_total = dt * n_steps as f64;

    let run = |desig: Desig, v0: f64| -> Vector3<f64> {
        let state = State::new(
            desig,
            Time::new(J2000),
            Vector3::new(r0, 0.0, 0.0),
            Vector3::new(0.0, v0, 0.0),
            SSB,
        );
        let mut sim = WisdomHolman::new(
            &[sun(), state],
            &[GMS, 1e-30],
            &[],
            &[],
            dt,
            false,
            false,
            false,
        )
        .unwrap();
        sim.integrate_n_steps(n_steps).unwrap();
        let states = sim.massive_states();
        Vector3::from(states[1].pos) - Vector3::from(states[0].pos)
    };

    // Each run starts on the exact circular orbit of its own force law, so
    // both stay at radius r0 and the lead between them is (v_emb - v_ctrl) t.
    let v_ctrl = (GMS / r0).sqrt();
    let v_emb = (GMS / r0 * (1.0 + 1.5 * quad / (r0 * r0))).sqrt();
    let ctrl = run(Desig::Perm(9999), v_ctrl);
    let emb = run(Desig::Naif(3), v_emb);
    let named = run(Desig::Name("earth barycenter".into()), v_emb);

    assert_eq!(
        emb, named,
        "name and NAIF id forms must resolve identically"
    );

    let predicted = (v_emb - v_ctrl) * t_total;
    let measured = (emb - ctrl).norm();
    let rel = (measured / predicted - 1.0).abs();
    println!("emb_quadrupole: 100 orbits at 1 AU, coefficient {quad:.4e} AU^2");
    println!("  measured lead: {measured:.6e} AU, predicted {predicted:.6e} AU");
    println!("  relative difference: {rel:.3e}");
    assert!(
        rel < 0.02,
        "quadrupole drift off: measured {measured:e} vs predicted {predicted:e}"
    );
    assert!(
        ctrl.cross(&emb).z > 0.0,
        "corrected body must lead the control (stronger effective attraction)"
    );
}

/// The solar J2 term reproduces the secular nodal regression and apsidal
/// precession of an inclined orbit:
/// `dOmega/dt = -(3/2) n J2 (R/p)^2 cos(i)` and
/// `domega/dt = (3/4) n J2 (R/p)^2 (5 cos^2(i) - 1)`,
/// with the inclination, node and apse measured from the solar spin pole.
#[test]
fn j2_precession() {
    let a = 1.0_f64;
    let ecc = 0.1;
    let inc_deg = 45.0_f64;
    // The formulas hold about the J2 axis, so the orbit is built, and its
    // node and apse read, in a frame whose z axis is the solar spin pole.
    let (_, pole) = sun_oblateness::<Ecliptic>().unwrap();
    let to_ecliptic = Rotation3::rotation_between(&Vector3::z(), &pole).unwrap();
    let (pos0, vel0) = peri_pos_vel(a, ecc, inc_deg, 0.0);
    let tp = make_state(1000, to_ecliptic * pos0, to_ecliptic * vel0);
    let period = TAU * (a.powi(3) / GMS).sqrt();
    let n_samples = 2000;

    // Slopes of the node angle and argument of perihelion, in rad/day,
    // sampled once per orbit so the osculating oscillation drops out.
    let measure = |include_j2: bool| -> (f64, f64) {
        let mut sim = WisdomHolman::new(
            &[sun()],
            &[GMS],
            std::slice::from_ref(&tp),
            &[],
            period / 40.0,
            false,
            include_j2,
            false,
        )
        .unwrap();
        let mut times = Vec::with_capacity(n_samples);
        let mut nodes = Vec::with_capacity(n_samples);
        let mut apses = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let sun_state = &sim.massive_states()[0];
            let (pos, vel) = heliocentric(&sim.test_particle_states()[0], sun_state);
            let (pos, vel) = (to_ecliptic.inverse() * pos, to_ecliptic.inverse() * vel);
            let h = pos.cross(&vel);
            let node = Vector3::new(-h.y, h.x, 0.0).normalize();
            let e_hat = ecc_vector(&pos, &vel).normalize();
            let h_hat = h.normalize();
            times.push(sim.epoch().jd() - J2000);
            nodes.push(node.y.atan2(node.x));
            // Argument of perihelion as a smooth signed angle from the node,
            // well behaved even though the orbit starts with omega = 0.
            apses.push(node.cross(&e_hat).dot(&h_hat).atan2(node.dot(&e_hat)));
            sim.integrate_n_steps(40).unwrap();
        }
        (linear_slope(&times, &nodes), linear_slope(&times, &apses))
    };

    let (node_rate, apse_rate) = measure(true);
    let (node_ctrl, apse_ctrl) = measure(false);

    let mean_motion = TAU / period;
    let semi_latus = a * (1.0 - ecc * ecc);
    let scale = SUN_J2 * (*SUN_RADIUS_AU / semi_latus).powi(2) * mean_motion;
    let cos_i = inc_deg.to_radians().cos();
    let node_expected = -1.5 * scale * cos_i;
    let apse_expected = 0.75 * scale * (5.0 * cos_i * cos_i - 1.0);

    println!("j2_precession: a = {a} AU, e = {ecc}, i = {inc_deg} deg, {n_samples} orbits");
    println!("  node rate measured: {node_rate:.6e} rad/day, expected {node_expected:.6e}");
    println!("  apse rate measured: {apse_rate:.6e} rad/day, expected {apse_expected:.6e}");
    println!("  without J2: node {node_ctrl:.3e}, apse {apse_ctrl:.3e} rad/day");

    // Budget: second-order splitting at dt = P/40 contributes O((dt/P)^2)
    // ~ 6e-4 relative; the linear-fit noise floor is far below the signal.
    assert!(
        (node_rate / node_expected - 1.0).abs() < 0.01,
        "nodal regression off: measured {node_rate:e}, expected {node_expected:e}"
    );
    assert!(
        (apse_rate / apse_expected - 1.0).abs() < 0.01,
        "apsidal precession off: measured {apse_rate:e}, expected {apse_expected:e}"
    );
    assert!(
        node_ctrl.abs() < 0.01 * node_expected.abs(),
        "control run shows spurious node motion {node_ctrl:e}"
    );
    assert!(
        apse_ctrl.abs() < 0.01 * apse_expected.abs(),
        "control run shows spurious apse motion {apse_ctrl:e}"
    );
}

/// With the solar J2 term on, the 8 planet system conserves energy within the
/// kernel budget and the solar-pole component of the angular momentum to
/// roundoff; the transverse components precess by design. The Earth stands in
/// for the Earth-Moon barycenter so the lunar quadrupole, symmetric about the
/// ecliptic pole instead, is off.
#[test]
fn j2_invariants() {
    let (mut states, gms) = solar_system();
    states[3] = planet(399, 1.00000, 0.01671, 0.0, 4.8);
    let (_, pole) = sun_oblateness::<Ecliptic>().unwrap();
    let mut sim = WisdomHolman::new(&states, &gms, &[], &[], 4.0, false, true, false).unwrap();
    let l0 = sim.angular_momentum();
    let (amplitude, half_drift) = energy_drift_stats(&mut sim, 50_000, 100);
    let l1 = sim.angular_momentum();
    let lz_rel = ((l1 - l0).dot(&pole) / l0.norm()).abs();

    println!("j2_invariants: 8 planets, dt = 4 d, 5e4 steps, J2 on");
    println!("  relative energy amplitude: {amplitude:.3e}");
    println!("  relative half-mean drift:  {half_drift:.3e}");
    println!("  pole-component L change:   {lz_rel:.3e}");
    assert!(amplitude < 1e-6, "energy amplitude {amplitude:e}");
    assert!(half_drift < 0.5 * amplitude, "secular drift {half_drift:e}");
    assert!(lz_rel < 1e-12, "pole angular momentum drift {lz_rel:e}");
}

/// Surface properties shared by the Yarkovsky tests: a dark, spherical body.
const ALBEDO: f64 = 0.15;
const ABSORPTIVITY: f64 = 0.9;
const DENSITY: f64 = 2500.0;

/// Frozen Farnocchia (Yarkovsky) force from an explicit area-to-mass ratio,
/// with the spin pole given on ecliptic axes (the frame the tests integrate
/// in; the pole is stored equatorial and the map rotates it back).
fn yark_raw(a_over_m: f64, lambda_0: f64, pole: Vector3<f64>) -> NonGravMask {
    let pole_eq = Vector::<Ecliptic>::new(pole.into()).into_frame::<Equatorial>();
    ParameterMask::all_fixed(
        NonGravKind::Farnocchia(
            FarnocchiaNonGrav::new(ALBEDO, ABSORPTIVITY, 1.0, pole_eq).unwrap(),
        ),
        vec![a_over_m, lambda_0],
    )
    .unwrap()
}

/// Frozen Farnocchia (Yarkovsky) force for a spherical body, `diameter` in km.
fn yark(diameter: f64, lambda_0: f64, pole: Vector3<f64>) -> NonGravMask {
    yark_raw(
        a_over_m_from_physical(DENSITY, diameter, 1.0),
        lambda_0,
        pole,
    )
}

/// Frozen dust force with the given radiation-pressure ratio.
fn dust(beta: f64) -> NonGravMask {
    ParameterMask::all_fixed(NonGravKind::Dust(DustNonGrav), vec![beta]).unwrap()
}

/// Circular prograde orbit of radius `a` in the xy plane (angular momentum +z).
fn circular(a: f64) -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(a, 0.0, 0.0),
        Vector3::new(0.0, (GMS / a).sqrt(), 0.0),
    )
}

/// Least-squares slope of `y` against `x`.
#[allow(clippy::cast_precision_loss, reason = "sample counts are small")]
fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let num: f64 = x
        .iter()
        .zip(y)
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum();
    let den: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
    num / den
}

/// Measured secular drift of the test particle's osculating semi-major axis, in
/// AU/day, as the least-squares slope of `a` against time.
///
/// Sampling every `steps_per_sample` steps, chosen as one orbital period, keeps
/// every sample at the same orbital phase, so the osculating oscillation enters
/// as a constant offset and drops out of the slope entirely. Differencing
/// endpoints instead would alias that oscillation.
fn measure_drift(sim: &mut WisdomHolman<Ecliptic>, n_samples: usize, steps_per_sample: u64) -> f64 {
    let mut times = Vec::with_capacity(n_samples);
    let mut axes = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let massive = sim.massive_states();
        let particle = &sim.test_particle_states()[0];
        let (pos, vel) = heliocentric(particle, &massive[0]);
        times.push(sim.epoch().jd() - J2000);
        axes.push(compute_semi_major(&pos, &vel, GMS));
        sim.integrate_n_steps(steps_per_sample).unwrap();
    }
    linear_slope(&times, &axes)
}

/// Run one test particle on a circular orbit with the given frozen Yarkovsky
/// force and return the measured drift in AU/day.
fn circular_drift(a: f64, yarkovsky: NonGravMask, use_correctors: bool, dt_frac: u64) -> f64 {
    let (pos, vel) = circular(a);
    let particle = make_state(1000, pos, vel);
    let period = TAU * (a.powi(3) / GMS).sqrt();
    #[allow(clippy::cast_precision_loss, reason = "small integer")]
    let dt = period / dt_frac as f64;
    let mut sim = WisdomHolman::new(
        &[sun()],
        &[GMS],
        &[particle],
        &[Some(yarkovsky)],
        dt,
        false,
        false,
        use_correctors,
    )
    .unwrap();
    measure_drift(&mut sim, 100, dt_frac)
}

/// Predicted secular drift from the Gauss planetary equation for a circular
/// orbit under a constant transverse acceleration: `da/dt = 2 F_t / n`.
fn gauss_drift(a: f64, transverse_accel: f64) -> f64 {
    let n = (GMS / a.powi(3)).sqrt();
    2.0 * transverse_accel / n
}

/// Transverse (along-track) component of the radiation acceleration for a
/// circular orbit of radius `a`, evaluated on ecliptic axes from a frozen
/// Farnocchia force (the same rotation the map performs).
fn transverse_accel(a: f64, frozen: &NonGravMask) -> f64 {
    let NonGravKind::Farnocchia(force) = frozen.inner() else {
        panic!("test helper expects a Farnocchia force");
    };
    let (pos, vel) = circular(a);
    let pole: Vector3<f64> = force.spin_pole.into_frame::<Ecliptic>().into();
    let accel = radiation_accel(
        &pos,
        &pole,
        force.albedo,
        force.absorptivity,
        force.flattening,
        frozen.fixed_values().unwrap()[0],
        frozen.fixed_values().unwrap()[1],
    );
    accel.dot(&vel.normalize())
}

/// With `a_over_m = 0` the whole radiation model vanishes (it is linear in
/// `a_over_m`), so the map must reproduce the gravity-only trajectory exactly.
#[test]
fn yarkovsky_zero_is_identity() {
    let gms = [GMS, gm(5)];
    let bodies = to_com_frame(&[sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)], &gms);
    let (pos, vel) = peri_pos_vel(2.5, 0.1, 5.0, 1.0);
    let state = make_state(1000, pos, vel);

    let mut plain = WisdomHolman::new(
        &bodies,
        &gms,
        std::slice::from_ref(&state),
        &[],
        4.0,
        true,
        false,
        true,
    )
    .unwrap();

    // Every surface property populated, but zero area-to-mass.
    let zero_yark = yark_raw(0.0, 0.5, Vector3::new(0.1, -0.2, 0.97));
    let mut zeroed = WisdomHolman::new(
        &bodies,
        &gms,
        &[state],
        &[Some(zero_yark)],
        4.0,
        true,
        false,
        true,
    )
    .unwrap();

    plain.integrate_n_steps(2000).unwrap();
    zeroed.integrate_n_steps(2000).unwrap();

    let a: Vector3<f64> = plain.test_particle_states()[0].pos.into();
    let b: Vector3<f64> = zeroed.test_particle_states()[0].pos.into();
    println!(
        "yarkovsky_zero_is_identity: position difference {:.3e} AU",
        (a - b).norm()
    );
    assert_eq!(
        a, b,
        "a_over_m = 0 must reproduce the gravity-only map exactly"
    );
}

/// The measured secular drift matches the Gauss planetary equation
/// `da/dt = 2 F_t / n`. On a circular orbit with the spin pole on the orbit
/// normal, r and the pole angle never change, so the transverse acceleration is
/// constant around the orbit and the prediction is closed form.
#[test]
fn yarkovsky_drift_matches_gauss() {
    let a = 2.4;
    let lambda_0 = 0.188; // near the peak of Lambda_2 at this distance
    let y = yark(1.0, lambda_0, Vector3::new(0.0, 0.0, 1.0));

    // Cross-check the transverse term against its closed form at this geometry.
    // Sphere and pole on the orbit normal give psi_x = 1 and J2 = 1, leaving
    // F_t = 4/9 * alpha * Lambda_2 * scale.
    let a_over_m = a_over_m_from_physical(DENSITY, 1.0, 1.0);
    let scale = a_over_m * F0_OVER_C_AU_DAY2 / (a * a);
    let lambda = lambda_0 * a.powf(1.5);
    let big_lambda_2 = lambda / (1.0 + 2.0 * lambda + 2.0 * lambda * lambda);
    let closed_form = 4.0 / 9.0 * ABSORPTIVITY * scale * big_lambda_2;
    let f_t = transverse_accel(a, &y);
    let closed_rel = ((f_t - closed_form) / closed_form).abs();

    let predicted = gauss_drift(a, f_t);
    let measured = circular_drift(a, y, false, 50);
    let rel = ((measured - predicted) / predicted).abs();

    // AU/day -> AU/Myr for a number comparable with the literature.
    let per_myr = measured * 365.25e6;
    println!("yarkovsky_drift_matches_gauss: D = 1 km, a = {a} AU, 100 orbits");
    println!("  transverse accel:      {f_t:.6e} AU/day^2");
    println!("  closed-form agreement: {closed_rel:.3e}");
    println!("  predicted da/dt:       {predicted:.6e} AU/day");
    println!("  measured da/dt:        {measured:.6e} AU/day");
    println!("  relative difference:   {rel:.3e}");
    println!("  measured drift:        {per_myr:.3e} AU/Myr");

    assert!(
        closed_rel < 1e-12,
        "transverse term disagrees with its closed form: {closed_rel:e}"
    );
    // Measured 4.5e-9; anchored ~20x above that. The prediction is first
    // order in the perturbation, whose size here is a_yark/a_grav ~ 1e-11,
    // so there is no second-order floor anywhere near this level.
    assert!(rel < 1e-7, "measured drift differs from Gauss by {rel:e}");
    assert!(
        measured > 0.0,
        "prograde spin must increase the semi-major axis"
    );
    // Order-of-magnitude gate against the literature scale for a km-size
    // main-belt body (~1e-4 AU/Myr). Deliberately loose: this pins the units
    // and the size scaling, not the model.
    assert!(
        (1e-5..1e-3).contains(&per_myr),
        "drift {per_myr:e} AU/Myr is far from the km-size main-belt scale"
    );
}

/// The drift is independent of the step size: it is a property of the dynamics,
/// not of the discretization.
#[test]
fn yarkovsky_drift_step_independent() {
    let a = 2.4;
    let coarse = circular_drift(a, yark(1.0, 0.188, Vector3::new(0.0, 0.0, 1.0)), false, 25);
    let fine = circular_drift(a, yark(1.0, 0.188, Vector3::new(0.0, 0.0, 1.0)), false, 100);
    let rel = ((coarse - fine) / fine).abs();
    println!("yarkovsky_drift_step_independent: dt = P/25 vs P/100");
    println!("  coarse {coarse:.6e}, fine {fine:.6e} AU/day, relative {rel:.3e}");
    // Measured 8.6e-8 between dt = P/25 and P/100.
    assert!(rel < 1e-5, "drift depends on step size: {rel:e}");
}

/// Obliquity sets the sign and size of the drift: a pole on the orbit normal
/// drifts outward, the reverse pole drifts inward by the same amount, and a
/// pole in the orbit plane produces no secular drift.
#[test]
fn yarkovsky_obliquity_law() {
    let a = 2.4;
    let prograde = circular_drift(a, yark(1.0, 0.188, Vector3::new(0.0, 0.0, 1.0)), false, 50);
    let retrograde = circular_drift(a, yark(1.0, 0.188, Vector3::new(0.0, 0.0, -1.0)), false, 50);
    let in_plane = circular_drift(a, yark(1.0, 0.188, Vector3::new(1.0, 0.0, 0.0)), false, 50);

    let antisymmetry = ((prograde + retrograde) / prograde).abs();
    let in_plane_frac = (in_plane / prograde).abs();
    println!("yarkovsky_obliquity_law: a = {a} AU, D = 1 km");
    println!("  obliquity   0 deg: {prograde:.6e} AU/day");
    println!("  obliquity 180 deg: {retrograde:.6e} AU/day");
    println!("  obliquity  90 deg: {in_plane:.6e} AU/day");
    println!("  antisymmetry residual: {antisymmetry:.3e}");
    println!("  in-plane / prograde:   {in_plane_frac:.3e}");

    assert!(prograde > 0.0, "obliquity 0 must drift outward");
    assert!(retrograde < 0.0, "obliquity 180 must flip the drift inward");
    // The residual is the difference of two independently roundoff-limited
    // slope measurements; its noise envelope is ~1e-7 relative (it scatters
    // from 0 to ~3e-8 under sampling-cadence changes alone, and shifts with
    // any ulp-level code change). A genuine asymmetry defect in any term of
    // the force produces an O(0.1+) residual, so 1e-6 discriminates with
    // orders of magnitude to spare on both sides.
    assert!(
        antisymmetry < 1e-6,
        "reversing the pole must flip the drift: {antisymmetry:e}"
    );
    assert!(
        in_plane_frac < 1e-5, // measured 1.0e-7
        "a pole in the orbit plane must not drive a secular drift: {in_plane_frac:e}"
    );
}

/// `a_over_m` scales as 1 / diameter, and the model is linear in `a_over_m`, so
/// the drift scales as 1 / diameter. This is the link that carries an absolute
/// magnitude through to a drift rate.
#[test]
fn yarkovsky_diameter_scaling() {
    let a = 2.4;
    let pole = Vector3::new(0.0, 0.0, 1.0);
    let big = circular_drift(a, yark(1.0, 0.188, pole), false, 50);
    let small = circular_drift(a, yark(0.1, 0.188, pole), false, 50);
    let ratio = small / big;
    println!("yarkovsky_diameter_scaling: D = 1 km vs 0.1 km");
    println!("  drift ratio: {ratio:.6} (expected 10)");
    assert!(
        (ratio - 10.0).abs() < 1e-4, // measured 1e-6
        "drift must scale as 1/diameter, got ratio {ratio}"
    );
}

/// The model is built from dot and cross products of the position and pole
/// directions, so it is frame covariant: the same physical setup expressed in
/// the equatorial frame must drift identically.
#[test]
fn yarkovsky_frame_covariance() {
    let a = 2.4;
    let (pos, _) = circular(a);
    let pole = Vector3::new(0.0, 0.0, 1.0);
    let a_over_m = a_over_m_from_physical(DENSITY, 1.0, 1.0);

    let ecliptic = radiation_accel(&pos, &pole, ALBEDO, ABSORPTIVITY, 1.0, a_over_m, 0.188);

    // Rotate both inputs into the equatorial frame and evaluate there.
    let pos_eq: Vector3<f64> = Vector::<Ecliptic>::new(pos.into())
        .into_frame::<Equatorial>()
        .into();
    let pole_eq: Vector3<f64> = Vector::<Ecliptic>::new(pole.into())
        .into_frame::<Equatorial>()
        .into();
    let equatorial = radiation_accel(
        &pos_eq,
        &pole_eq,
        ALBEDO,
        ABSORPTIVITY,
        1.0,
        a_over_m,
        0.188,
    );

    // Rotating the result back must recover the ecliptic evaluation.
    let back: Vector3<f64> = Vector::<Equatorial>::new(equatorial.into())
        .into_frame::<Ecliptic>()
        .into();
    let rel = (back - ecliptic).norm() / ecliptic.norm();
    println!("yarkovsky_frame_covariance: relative difference {rel:.3e}");
    assert!(rel < 1e-14, "model is not frame covariant: {rel:e}");
}

/// The symplectic correctors are derived for a Hamiltonian perturbation, which
/// the radiation force is not. Within each corrector operator the perturbation
/// flow appears as Y(-b) then Y(+b), so the radiation impulses cancel to first
/// order and the measured drift is unaffected.
#[test]
fn yarkovsky_corrector_agnostic() {
    let a = 2.4;
    let pole = Vector3::new(0.0, 0.0, 1.0);
    let kernel = circular_drift(a, yark(1.0, 0.188, pole), false, 50);
    let corrected = circular_drift(a, yark(1.0, 0.188, pole), true, 50);
    let rel = ((corrected - kernel) / kernel).abs();
    println!("yarkovsky_corrector_agnostic: kernel {kernel:.6e}, corrected {corrected:.6e} AU/day");
    println!("  relative difference: {rel:.3e}");
    assert!(
        rel < 1e-5, // measured 4.5e-7
        "correctors changed the secular drift by {rel:e}; they must not"
    );
}

/// Radiation descriptions stay attached to their particles when other particles
/// are removed mid-run. A misaligned retain would silently give a particle
/// someone else's spin pole.
#[test]
fn yarkovsky_survives_particle_loss() {
    let a = 2.4;
    let (pos, vel) = circular(a);
    // A doomed particle first, then the one we track.
    let plunging = make_state(
        1,
        Vector3::new(0.05, 0.0, 0.0),
        Vector3::new(0.0, 1e-4, 0.0),
    );
    let tracked = make_state(1000, pos, vel);
    let non_gravs = [
        Some(yark(1.0, 0.188, Vector3::new(0.0, 0.0, -1.0))),
        Some(yark(1.0, 0.188, Vector3::new(0.0, 0.0, 1.0))),
    ];
    let period = TAU * (a.powi(3) / GMS).sqrt();
    let mut sim = WisdomHolman::new(
        &[sun()],
        &[GMS],
        &[plunging, tracked],
        &non_gravs,
        period / 50.0,
        false,
        false,
        false,
    )
    .unwrap();

    sim.integrate_n_steps(50).unwrap();
    assert_eq!(sim.n_test_particles(), 1, "plunging particle not removed");
    assert_eq!(sim.lost_particles()[0].desig, Desig::Perm(1));

    // The survivor must keep its own prograde pole, not inherit the retrograde
    // one from the removed particle.
    let drift = measure_drift(&mut sim, 100, 50);
    println!("yarkovsky_survives_particle_loss: surviving drift {drift:.6e} AU/day");
    assert!(
        drift > 0.0,
        "survivor kept the wrong spin pole after a particle was removed"
    );
}

/// Yarkovsky fields are public, so a description can be built as a struct
/// literal without the checks of `Yarkovsky::new`. The constructor re-validates
/// each description: invalid values are rejected loudly and a non-unit spin
/// pole is normalized rather than silently mis-scaling the force.
#[test]
fn yarkovsky_revalidated_at_construction() {
    let (pos, vel) = circular(2.4);
    let state = make_state(1000, pos, vel);
    let build = |non_grav: NonGravMask| {
        WisdomHolman::new(
            &[sun()],
            &[GMS],
            std::slice::from_ref(&state),
            &[Some(non_grav)],
            4.0,
            false,
            false,
            false,
        )
    };
    let a_over_m = a_over_m_from_physical(DENSITY, 1.0, 1.0);

    // A zero spin pole smuggled in via the public fields must be rejected.
    let bad = ParameterMask::all_fixed(
        NonGravKind::Farnocchia(FarnocchiaNonGrav {
            albedo: ALBEDO,
            absorptivity: ABSORPTIVITY,
            flattening: 1.0,
            spin_pole: Vector::<Equatorial>::new([0.0, 0.0, 0.0]),
        }),
        vec![a_over_m, 0.188],
    )
    .unwrap();
    assert!(
        build(bad).is_err(),
        "zero spin pole must be rejected at construction"
    );

    // A value count that does not match the kind cannot be built.
    assert!(ParameterMask::all_fixed(NonGravKind::Dust(DustNonGrav), Vec::new()).is_err());

    // A parameter left free cannot be simulated.
    assert!(
        build(ParameterMask::all_free(NonGravKind::Dust(DustNonGrav))).is_err(),
        "a free beta must be rejected"
    );

    // NaN values (left free for orbit fitting) cannot be simulated.
    assert!(
        build(yark_raw(f64::NAN, 0.188, Vector3::new(0.0, 0.0, 1.0))).is_err(),
        "NaN a_over_m must be rejected"
    );

    // An unbound grain must be rejected.
    assert!(build(dust(1.0)).is_err(), "beta = 1 must be rejected");

    // The A1/A2/A3 model is supported only in its un-lagged form.
    let lagged = ParameterMask::all_fixed(
        NonGravKind::JplComet(JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 30.0)),
        vec![1e-13, 1e-13, 0.0],
    )
    .unwrap();
    assert!(
        build(lagged).is_err(),
        "time-lagged outgassing must be rejected loudly"
    );
    // A partly fixed comet model is rejected: a3 is still free.
    let partly = ParameterMask::new(
        NonGravKind::JplComet(JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 0.0)),
        vec![Some(0.0), Some(0.0), None],
    )
    .unwrap();
    assert!(build(partly).is_err(), "a free a3 must be rejected");

    // A non-unit pole must integrate identically to its normalized form.
    let unit = yark(1.0, 0.188, Vector3::new(0.0, 0.0, 1.0));
    let NonGravKind::Farnocchia(unit_force) = unit.inner() else {
        unreachable!()
    };
    let doubled: Vector3<f64> = Vector3::from(unit_force.spin_pole) * 2.0;
    let scaled = ParameterMask::all_fixed(
        NonGravKind::Farnocchia(FarnocchiaNonGrav {
            albedo: ALBEDO,
            absorptivity: ABSORPTIVITY,
            flattening: 1.0,
            spin_pole: Vector::<Equatorial>::new(doubled.into()),
        }),
        vec![a_over_m, 0.188],
    )
    .unwrap();
    let mut sim_scaled = build(scaled).unwrap();
    let mut sim_unit = build(unit).unwrap();
    sim_scaled.integrate_n_steps(500).unwrap();
    sim_unit.integrate_n_steps(500).unwrap();
    let a: Vector3<f64> = sim_scaled.test_particle_states()[0].pos.into();
    let b: Vector3<f64> = sim_unit.test_particle_states()[0].pos.into();
    println!(
        "yarkovsky_revalidated_at_construction: position difference {:.3e} AU",
        (a - b).norm()
    );
    assert_eq!(
        a, b,
        "non-unit spin pole must be normalized at construction"
    );
}

/// A dust grain with `beta = 0` feels no radiation force and must reproduce a
/// plain gravity-only test particle exactly (its drift reduces to `mu = GMS`
/// and its kick to the plain additive kick).
#[test]
fn dust_zero_beta_is_test_particle() {
    let gms = [GMS, gm(5)];
    let bodies = to_com_frame(&[sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)], &gms);
    let (pos, vel) = peri_pos_vel(2.5, 0.15, 8.0, 1.0);
    let state = make_state(1000, pos, vel);

    let mut as_tp = WisdomHolman::new(
        &bodies,
        &gms,
        std::slice::from_ref(&state),
        &[],
        4.0,
        false,
        false,
        false,
    )
    .unwrap();
    let mut as_dust = WisdomHolman::new(
        &bodies,
        &gms,
        &[state],
        &[Some(dust(0.0))],
        4.0,
        false,
        false,
        false,
    )
    .unwrap();

    as_tp.integrate_n_steps(3000).unwrap();
    as_dust.integrate_n_steps(3000).unwrap();

    let tp: Vector3<f64> = as_tp.test_particle_states()[0].pos.into();
    let grain: Vector3<f64> = as_dust.test_particle_states()[0].pos.into();
    println!(
        "dust_zero_beta: position difference {:.3e} AU",
        (tp - grain).norm()
    );
    assert_eq!(tp, grain, "beta = 0 dust must equal a plain test particle");
}

/// Poynting-Robertson inspiral of a circular grain through the WH map. As in
/// the force certification, `a^2` decays linearly at `d(a^2)/dt = -4 beta GMS
/// / c`; here it exercises the reduced-gravity drift and the exact drag flow in
/// the kick together.
#[test]
fn dust_pr_inspiral_rate() {
    let beta = 0.02;
    let a0 = 2.5;
    let mu_eff = (1.0 - beta) * GMS;
    let v0 = (mu_eff / a0).sqrt();
    let grain = make_state(1000, Vector3::new(a0, 0.0, 0.0), Vector3::new(0.0, v0, 0.0));
    let period = TAU * (a0.powi(3) / mu_eff).sqrt();
    let steps_per_orbit = 40_u32;
    let dt = period / f64::from(steps_per_orbit);
    let mut sim = WisdomHolman::new(
        &[sun()],
        &[GMS],
        &[grain],
        &[Some(dust(beta))],
        dt,
        false,
        false,
        false,
    )
    .unwrap();

    let mut times = Vec::new();
    let mut a_sq = Vec::new();
    let sun_state = &sim.massive_states()[0];
    for _ in 0..200 {
        let g = &sim.test_particle_states()[0];
        let (pos, vel) = heliocentric(g, sun_state);
        times.push(sim.epoch().jd() - J2000);
        a_sq.push(compute_semi_major(&pos, &vel, mu_eff).powi(2));
        sim.integrate_n_steps(u64::from(steps_per_orbit)).unwrap();
    }
    let slope = linear_slope(&times, &a_sq);
    let predicted = -4.0 * beta * GMS / crate::constants::C_AU_PER_DAY;
    let rel = ((slope - predicted) / predicted).abs();
    println!("dust_pr_inspiral: beta = {beta}, a0 = {a0} AU");
    println!("  measured d(a^2)/dt {slope:.6e}, predicted {predicted:.6e}, rel {rel:.3e}");
    assert!(rel < 0.01, "map PR inspiral rate off by {rel:e}");
}

/// Full-trajectory cross-check of the WH dust dynamics against the Radau
/// integrator, which handles the velocity-dependent force natively. Sun +
/// Jupiter + a dust grain: this exercises the reduced-gravity drift, the drag
/// flow, and the planetary perturbation on the grain all at once.
#[test]
fn dust_matches_radau() {
    let beta = 0.05;
    let gms = [GMS, gm(5)];
    let bodies = to_com_frame(&[sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)], &gms);
    let (dpos, dvel) = peri_pos_vel(2.2, 0.1, 3.0, 0.5);
    let grain = make_state(1000, dpos, dvel);
    let t_final = 4000.0;

    let wh_helio = |dt: f64| -> Vector3<f64> {
        let mut sim = WisdomHolman::new(
            &bodies,
            &gms,
            std::slice::from_ref(&grain),
            &[Some(dust(beta))],
            dt,
            false,
            false,
            false,
        )
        .unwrap();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test"
        )]
        let n = (t_final / dt).round() as u64;
        sim.integrate_n_steps(n).unwrap();
        let sun_end = &sim.massive_states()[0];
        heliocentric(&sim.test_particle_states()[0], sun_end).0
    };

    // Radau reference: Sun + Jupiter (mutual gravity) and a massless grain
    // feeling both plus radiation pressure and drag, in the same COM frame.
    let gm_sun = GMS;
    let gm_jup = gm(5);
    let mut pos0 = DVector::<f64>::zeros(9);
    let mut vel0 = DVector::<f64>::zeros(9);
    for (i, s) in [&bodies[0], &bodies[1]].iter().enumerate() {
        let p = Vector3::from(s.pos);
        let v = Vector3::from(s.vel);
        for k in 0..3 {
            pos0[3 * i + k] = p[k];
            vel0[3 * i + k] = v[k];
        }
    }
    for k in 0..3 {
        pos0[6 + k] = dpos[k];
        vel0[6 + k] = dvel[k];
    }
    let func = |_t: Time<TDB>,
                pos: &DVector<f64>,
                vel: &DVector<f64>,
                _m: &mut (),
                _e: bool|
     -> crate::errors::KeteResult<DVector<f64>> {
        let get = |b: usize, p: &DVector<f64>| Vector3::new(p[3 * b], p[3 * b + 1], p[3 * b + 2]);
        let (r_sun, r_jup, r_dust) = (get(0, pos), get(1, pos), get(2, pos));
        let grav = |from: Vector3<f64>, gm: f64, to: Vector3<f64>| {
            let sep = from - to;
            sep * (gm / sep.norm().powi(3))
        };
        let mut a = DVector::<f64>::zeros(9);
        let a_sun = grav(r_jup, gm_jup, r_sun);
        let a_jup = grav(r_sun, gm_sun, r_jup);
        let mut a_dust = grav(r_sun, gm_sun, r_dust) + grav(r_jup, gm_jup, r_dust);
        // Radiation pressure and drag, from the Sun-relative state. The force
        // is frame covariant, so the Ecliptic components are reinterpreted as
        // Equatorial (no rotation) and the result is read back in-frame.
        let v_dust = get(2, vel);
        let v_sun = get(0, vel);
        let rel_pos = Vector::<Equatorial>::new((r_dust - r_sun).into());
        let rel_vel = Vector::<Equatorial>::new((v_dust - v_sun).into());
        let dust_force: Vector3<f64> = DustNonGrav
            .accel(
                Time::new(0.0),
                &rel_pos,
                &rel_vel,
                &[beta],
                &mut Default::default(),
                false,
            )?
            .into();
        a_dust += dust_force;
        for k in 0..3 {
            a[k] = a_sun[k];
            a[3 + k] = a_jup[k];
            a[6 + k] = a_dust[k];
        }
        Ok(a)
    };
    let (rpos, _, ()) = RadauIntegrator::integrate(
        &func,
        pos0,
        vel0,
        Time::new(J2000),
        Time::new(J2000 + t_final),
        (),
        None,
        None,
    )
    .unwrap();
    let radau_helio = Vector3::new(rpos[6] - rpos[0], rpos[7] - rpos[1], rpos[8] - rpos[2]);

    // WH dust must converge to the (essentially exact) Radau trajectory as the
    // step shrinks, at the map's second order. A wrong drift mu or drag would
    // instead leave a dt-independent floor.
    let errors: Vec<f64> = [8.0, 4.0, 2.0]
        .iter()
        .map(|&dt| (wh_helio(dt) - radau_helio).norm())
        .collect();
    println!("dust_matches_radau: beta = {beta}, {t_final} days");
    for (dt, err) in [8.0, 4.0, 2.0].iter().zip(&errors) {
        println!("  dt = {dt:4.1} d: WH-dust vs Radau {err:.3e} AU");
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        println!("  measured order: {order:.3}");
        assert!(
            (1.7..=2.3).contains(&order),
            "WH dust does not converge to Radau at second order: {order}"
        );
    }
    assert!(
        errors[2] < 1e-6,
        "WH dust vs Radau floor too high: {:e} AU",
        errors[2]
    );
}

/// The transverse `A2` of the JPL model with the asteroid `1/r^2` falloff
/// drives the secular semi-major drift of the Gauss planetary equation,
/// `da/dt = 2 F_t / n` with `F_t = a2 / r^2` -- the parameterization JPL
/// orbit solutions use for asteroid Yarkovsky detections. A radial `A1` of
/// the same size must drive no secular drift on a circular orbit.
#[test]
fn jpl_comet_a2_matches_gauss() {
    let a = 1.27_f64; // Phaethon-like semi-major axis
    let a2 = 1e-13;
    let asteroid_form = || JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 0.0);
    let frozen = |values: Vec<f64>| {
        ParameterMask::all_fixed(NonGravKind::JplComet(asteroid_form()), values).unwrap()
    };

    let measured = circular_drift(a, frozen(vec![0.0, a2, 0.0]), false, 50);
    let n = (GMS / a.powi(3)).sqrt();
    let predicted = 2.0 * (a2 / (a * a)) / n;
    let rel = ((measured - predicted) / predicted).abs();

    let radial = circular_drift(a, frozen(vec![a2, 0.0, 0.0]), false, 50);
    let radial_frac = (radial / measured).abs();

    println!("jpl_comet_a2_matches_gauss: a = {a} AU, a2 = {a2:e} AU/day^2");
    println!("  measured da/dt: {measured:.6e} AU/day, predicted {predicted:.6e}");
    println!("  relative difference: {rel:.3e}");
    println!("  radial-only drift fraction: {radial_frac:.3e}");
    assert!(
        rel < 1e-6,
        "A2 drift differs from the Gauss equation by {rel:e}"
    );
    assert!(
        measured > 0.0,
        "positive A2 must increase the semi-major axis"
    );
    assert!(
        radial_frac < 1e-5,
        "radial A1 must not drive a secular drift: {radial_frac:e}"
    );
}

/// Full-trajectory cross-check of the map's A1/A2/A3 kick against Radau
/// integrating the same trait force. With only the Sun massive the Kepler
/// part is exact, so this isolates the comet-kick splitting; a wrong RTN
/// basis or `g(r)` wiring would appear as a dt-independent error floor.
#[test]
fn jpl_comet_matches_radau() {
    let (pos0, vel0) = peri_pos_vel(1.27, 0.3, 5.0, 0.7);
    let tp = make_state(1000, pos0, vel0);
    let t_final = 10_000.0;
    let comet = JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 0.0);
    let params = [2e-10, -1.5e-10, 1e-10];

    let wh_helio = |dt: f64| -> Vector3<f64> {
        let frozen =
            ParameterMask::all_fixed(NonGravKind::JplComet(comet.clone()), params.to_vec())
                .unwrap();
        let mut sim = WisdomHolman::new(
            &[sun()],
            &[GMS],
            std::slice::from_ref(&tp),
            &[Some(frozen)],
            dt,
            false,
            false,
            false,
        )
        .unwrap();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "test"
        )]
        let n = (t_final / dt).round() as u64;
        sim.integrate_n_steps(n).unwrap();
        let sun_end = &sim.massive_states()[0];
        heliocentric(&sim.test_particle_states()[0], sun_end).0
    };

    // The force is frame covariant, so the Ecliptic components are
    // reinterpreted as Equatorial for the trait call and read back in-frame.
    let func = |_t: Time<TDB>,
                pos: &DVector<f64>,
                vel: &DVector<f64>,
                _m: &mut (),
                _e: bool|
     -> crate::errors::KeteResult<DVector<f64>> {
        let p = Vector3::new(pos[0], pos[1], pos[2]);
        let v = Vector3::new(vel[0], vel[1], vel[2]);
        let mut accel = -GMS / p.norm().powi(3) * p;
        let ng: Vector3<f64> = comet
            .accel(
                Time::new(0.0),
                &Vector::<Equatorial>::new(p.into()),
                &Vector::<Equatorial>::new(v.into()),
                &params,
                &mut Default::default(),
                false,
            )?
            .into();
        accel += ng;
        Ok(DVector::from_row_slice(&[accel.x, accel.y, accel.z]))
    };
    let (rpos, _, ()) = RadauIntegrator::integrate(
        &func,
        DVector::from_row_slice(&[pos0.x, pos0.y, pos0.z]),
        DVector::from_row_slice(&[vel0.x, vel0.y, vel0.z]),
        Time::new(J2000),
        Time::new(J2000 + t_final),
        (),
        None,
        None,
    )
    .unwrap();
    let radau = Vector3::new(rpos[0], rpos[1], rpos[2]);

    let step_sizes = [8.0, 4.0, 2.0];
    let errors: Vec<f64> = step_sizes
        .iter()
        .map(|&dt| (wh_helio(dt) - radau).norm())
        .collect();
    println!("jpl_comet_matches_radau: {t_final} days, a1/a2/a3 all nonzero");
    for (dt, err) in step_sizes.iter().zip(&errors) {
        println!("  dt = {dt:4.1} d: WH-comet vs Radau {err:.3e} AU");
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        println!("  measured order: {order:.3}");
        assert!(
            (1.7..=2.3).contains(&order),
            "comet kick does not converge to the reference at second order: {order}"
        );
    }
    assert!(
        errors[2] < 1e-4,
        "WH-comet vs Radau floor too high: {:e} AU",
        errors[2]
    );
}

/// Test particles that fall into the Sun are removed and recorded.
#[test]
fn sun_impact_loss() {
    let tp = make_state(
        1000,
        Vector3::new(0.1, 0.0, 0.0),
        Vector3::new(0.0, 1e-4, 0.0),
    );
    let mut sim =
        WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], 0.05, false, false, false).unwrap();
    sim.integrate_n_steps(100).unwrap();

    assert_eq!(sim.n_test_particles(), 0, "plunging particle not removed");
    assert_eq!(sim.lost_particles().len(), 1);
    assert_eq!(sim.lost_particles()[0].reason, LostReason::SunImpact);
    assert_eq!(sim.lost_particles()[0].desig, Desig::Perm(1000));
    println!(
        "sun_impact_loss: particle removed at jd = {}",
        sim.lost_particles()[0].epoch.jd()
    );
}

/// Sub-3-Hill-radius approaches are tracked and reported.
#[test]
fn encounter_tracking() {
    let jupiter = planet(5, 5.20260, 0.0, 0.0, 0.0);
    let hill = hill_radius(5.2026, 0.0, gm(5), GMS);
    let jup_pos: Vector3<f64> = jupiter.pos.into();
    let jup_vel: Vector3<f64> = jupiter.vel.into();
    let tp = make_state(1000, jup_pos + Vector3::new(0.1 * hill, 0.0, 0.0), jup_vel);

    let mut sim = WisdomHolman::new(
        &[sun(), jupiter],
        &[GMS, gm(5)],
        &[tp],
        &[],
        1.0,
        false,
        false,
        false,
    )
    .unwrap();
    sim.integrate_n_steps(1).unwrap();

    let enc = sim.closest_encounter().expect("encounter not recorded");
    println!(
        "encounter_tracking: ratio = {:.3} between {:?} and {:?}",
        enc.hill_ratio, enc.first, enc.second
    );
    assert!(
        enc.hill_ratio < 0.2,
        "expected ~0.1, got {}",
        enc.hill_ratio
    );
    assert_eq!(enc.first, Desig::Perm(5));
    assert_eq!(enc.second, Desig::Perm(1000));
}

/// Constructor rejects invalid inputs loudly.
#[test]
fn constructor_validation() {
    let jupiter = planet(5, 5.2026, 0.048, 1.3, 0.0);

    // First body must be the Sun.
    assert!(
        WisdomHolman::new(
            std::slice::from_ref(&jupiter),
            &[gm(5)],
            &[],
            &[],
            1.0,
            false,
            false,
            false
        )
        .is_err()
    );
    // The Sun's GM may grow by merged planets, within 1%.
    for (factor, ok) in [
        (1.005, true),
        (0.995, true),
        (1.02, false),
        (f64::NAN, false),
    ] {
        assert_eq!(
            WisdomHolman::new(
                &[sun()],
                &[GMS * factor],
                &[],
                &[],
                1.0,
                false,
                false,
                false
            )
            .is_ok(),
            ok,
            "{factor}"
        );
    }
    // Empty body list.
    assert!(WisdomHolman::<Ecliptic>::new(&[], &[], &[], &[], 1.0, false, false, false).is_err());
    // dt of zero.
    assert!(WisdomHolman::new(&[sun()], &[GMS], &[], &[], 0.0, false, false, false).is_err());
    // Mismatched massive and gms lengths.
    assert!(
        WisdomHolman::new(&[sun()], &[GMS, gm(5)], &[], &[], 1.0, false, false, false).is_err()
    );
    // Non-empty non_gravs must match the test particle count.
    let tp = make_state(
        1000,
        Vector3::new(2.0, 0.0, 0.0),
        Vector3::new(0.0, 0.01, 0.0),
    );
    assert!(
        WisdomHolman::new(
            &[sun()],
            &[GMS],
            &[tp],
            &[None, None],
            1.0,
            false,
            false,
            false
        )
        .is_err()
    );
    // Mismatched epoch.
    let mut late = jupiter;
    late.epoch = Time::new(J2000 + 1.0);
    assert!(
        WisdomHolman::new(
            &[sun(), late],
            &[GMS, gm(5)],
            &[],
            &[],
            1.0,
            false,
            false,
            false
        )
        .is_err()
    );
}

/// The step count based `integrate_to` lands on the nearest whole step.
#[test]
fn integrate_to_whole_steps() {
    let (states, gms) = outer_solar_system();
    let mut sim = WisdomHolman::new(&states, &gms, &[], &[], 10.0, false, false, false).unwrap();
    sim.integrate_to(Time::new(J2000 + 1004.0)).unwrap();
    assert_eq!(sim.steps_taken(), 100);
    assert!((sim.epoch().jd() - (J2000 + 1000.0)).abs() < 1e-9);
    // Asking for a time behind the current epoch errors.
    assert!(sim.integrate_to(Time::new(J2000)).is_err());
}

/// Outer solar system for 1 Myr: energy bounded, no secular drift,
/// semi-major axes bounded (sampled throughout the run, since osculating
/// elements of a stable system genuinely oscillate at the ~1% level).
/// Budgets are passed in by the two test wrappers.
#[allow(clippy::cast_precision_loss, reason = "test statistics")]
fn outer_ss_1myr(use_correctors: bool, amp_budget: f64, drift_factor: f64) {
    let (states, gms) = outer_solar_system();
    let mut sim =
        WisdomHolman::new(&states, &gms, &[], &[], 100.0, false, false, use_correctors).unwrap();

    let start_states = sim.massive_states();
    let mut a_start = Vec::new();
    for state in &start_states[1..] {
        let (pos, vel) = heliocentric(state, &start_states[0]);
        a_start.push(compute_semi_major(&pos, &vel, GMS));
    }
    let ang_mom0 = sim.angular_momentum();

    let n_steps = 3_652_500_u64; // 1 Myr at dt = 100 d
    let sample_every = 10_000_u64;
    let mut energy_samples = vec![sim.energy()];
    let mut a_lo = a_start.clone();
    let mut a_hi = a_start.clone();
    let timer = std::time::Instant::now();
    let mut taken = 0;
    while taken < n_steps {
        sim.integrate_n_steps(sample_every.min(n_steps - taken))
            .unwrap();
        taken += sample_every;
        energy_samples.push(sim.energy());
        let states = sim.massive_states();
        for (idx, state) in states[1..].iter().enumerate() {
            let (pos, vel) = heliocentric(state, &states[0]);
            let a = compute_semi_major(&pos, &vel, GMS);
            a_lo[idx] = a_lo[idx].min(a);
            a_hi[idx] = a_hi[idx].max(a);
        }
    }
    let elapsed = timer.elapsed().as_secs_f64();

    let e0 = energy_samples[0];
    let max = energy_samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min = energy_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let amplitude = (max - min) / e0.abs();
    let half = energy_samples.len() / 2;
    let mean_first = energy_samples[..half].iter().sum::<f64>() / half as f64;
    let mean_second =
        energy_samples[half..].iter().sum::<f64>() / (energy_samples.len() - half) as f64;
    let half_drift = (mean_second - mean_first).abs() / e0.abs();
    let ang_rel = (sim.angular_momentum() - ang_mom0).norm() / ang_mom0.norm();

    println!(
        "outer_ss_1myr (correctors: {use_correctors}): J/S/U/N, dt = 100 d, {n_steps} steps = 1 Myr"
    );
    println!(
        "  wall time: {elapsed:.1} s ({:.2e} steps/s)",
        n_steps as f64 / elapsed
    );
    println!("  relative energy amplitude:        {amplitude:.3e}");
    println!("  relative half-mean energy drift:  {half_drift:.3e}");
    println!("  relative angular momentum change: {ang_rel:.3e}");
    for (idx, ((a0, lo), hi)) in a_start.iter().zip(&a_lo).zip(&a_hi).enumerate() {
        let spread = (hi - lo) / a0;
        println!(
            "  body {}: a0 = {a0:.5} AU, sampled range [{lo:.5}, {hi:.5}] (spread {spread:.2e})",
            idx + 5
        );
        assert!(
            (hi - a0).max(a0 - lo) / a0 < 0.02,
            "semi-major axis left the 2% band"
        );
    }

    assert!(amplitude < amp_budget, "energy amplitude {amplitude:e}");
    assert!(
        half_drift < drift_factor * amplitude,
        "secular drift {half_drift:e}"
    );
    assert!(ang_rel < 1e-11, "angular momentum {ang_rel:e}");
}

/// Outer solar system, 1 Myr, kernel map.
#[test]
#[ignore = "long running, 1 Myr integration"]
fn outer_ss_1myr_kernel() {
    outer_ss_1myr(false, 4e-6, 0.1);
}

/// Outer solar system, 1 Myr, order-17 correctors.
/// Budget follows the eps^2 (n dt)^2 scale, with a drift factor of 0.5
/// since the corrected band sits near the roundoff floor.
#[test]
#[ignore = "long running, 1 Myr integration"]
fn outer_ss_1myr_corrected() {
    outer_ss_1myr(true, 2e-7, 0.5);
}

/// The full 8 planet system for 100 kyr at dt = 4 d.
#[test]
#[ignore = "long running, 1 Myr integration"]
fn energy_planets_long() {
    let (states, gms) = solar_system();
    let mut sim = WisdomHolman::new(&states, &gms, &[], &[], 4.0, false, false, false).unwrap();
    let ang_mom0 = sim.angular_momentum();

    let n_steps = 9_131_250_u64; // 100 kyr at dt = 4 d
    let timer = std::time::Instant::now();
    let (amplitude, half_drift) = energy_drift_stats(&mut sim, n_steps, 10_000);
    let elapsed = timer.elapsed().as_secs_f64();
    let ang_rel = (sim.angular_momentum() - ang_mom0).norm() / ang_mom0.norm();

    println!("energy_planets_long: 8 planets, dt = 4 d, {n_steps} steps = 100 kyr");
    #[allow(clippy::cast_precision_loss, reason = "test statistics")]
    let rate = n_steps as f64 / elapsed;
    println!("  wall time: {elapsed:.1} s ({rate:.2e} steps/s)");
    println!("  relative energy amplitude:        {amplitude:.3e}");
    println!("  relative half-mean energy drift:  {half_drift:.3e}");
    println!("  relative angular momentum change: {ang_rel:.3e}");

    assert!(amplitude < 1e-6, "energy amplitude {amplitude:e}");
    assert!(half_drift < 0.1 * amplitude, "secular drift {half_drift:e}");
    assert!(ang_rel < 1e-11, "angular momentum {ang_rel:e}");
}

/// 209 massive bodies (planets + 200 asteroids) and `n_tp` test particles,
/// the production-scale configuration used by the perf and profile tests.
#[allow(
    clippy::type_complexity,
    reason = "test helper returning parallel lists"
)]
fn production_config(
    n_tp: u32,
) -> (
    Vec<State<Ecliptic, SSB>>,
    Vec<f64>,
    Vec<State<Ecliptic, SSB>>,
) {
    let (mut bodies, mut gms) = solar_system();
    for idx in 0..200_u32 {
        let a = 2.1 + 0.004 * f64::from(idx);
        let phase = f64::from(idx) * 2.399963;
        let (pos, vel) = peri_pos_vel(a, 0.1, 5.0, phase);
        bodies.push(make_state(20000 + idx, pos, vel));
        gms.push(1e-11 * GMS);
    }
    let test_particles = (0..n_tp)
        .map(|idx| {
            let a = 2.2 + 0.0006 * f64::from(idx);
            let phase = f64::from(idx) * 0.71;
            let (pos, vel) = peri_pos_vel(a, 0.15, 8.0, phase);
            make_state(100_000 + idx, pos, vel)
        })
        .collect();
    (bodies, gms, test_particles)
}

/// Steps per second for `n_tp` test particles at production scale.
#[allow(clippy::cast_precision_loss, reason = "test statistics")]
fn throughput(n_tp: u32, n_steps: u64, use_correctors: bool) -> f64 {
    let (bodies, gms, tp) = production_config(n_tp);
    let mut sim =
        WisdomHolman::new(&bodies, &gms, &tp, &[], 4.0, true, true, use_correctors).unwrap();
    let timer = std::time::Instant::now();
    sim.integrate_n_steps(n_steps).unwrap();
    n_steps as f64 / timer.elapsed().as_secs_f64()
}

/// Throughput smoke test at production scale: 200 massive asteroids plus
/// 2000 test particles.
#[test]
#[ignore = "long running, throughput measurement"]
fn perf_production_scale() {
    let (bodies, gms, test_particles) = production_config(2000);
    let mut sim =
        WisdomHolman::new(&bodies, &gms, &test_particles, &[], 4.0, true, true, true).unwrap();
    let e0 = sim.energy();
    let n_steps = 5000_u64;
    let timer = std::time::Instant::now();
    sim.integrate_n_steps(n_steps).unwrap();
    let elapsed = timer.elapsed().as_secs_f64();
    let e_rel = ((sim.energy() - e0) / e0).abs();

    #[allow(clippy::cast_precision_loss, reason = "test statistics")]
    let rate = n_steps as f64 / elapsed;
    println!("perf_production_scale: 209 massive + 2000 tp, dt = 4 d");
    println!(
        "  {rate:.0} steps/s ({:.1} kyr of simulation per wall-clock hour)",
        rate * 3600.0 * 4.0 / 365.25 / 1000.0
    );
    println!("  relative energy change over {n_steps} steps: {e_rel:.3e}");
    assert!(e_rel < 1e-6);
}

/// Test-particle scaling: wall-time per step against test-particle count.
/// The marginal cost per particle converges to the compute floor as the count
/// grows; the tp pass is compute/parallelism-bound at the target scales (tens
/// of thousands and up).

#[test]
#[ignore = "long running, profiling measurement"]
fn profile_test_particle_scaling() {
    // Warm up the thread pool and caches.
    let _ = throughput(2000, 200, false);

    println!("profile_test_particle_scaling: 209 massive, dt = 4 d, correctors off");
    let cases = [
        (0_u32, 3000_u64),
        (2000, 2000),
        (8000, 800),
        (25000, 250),
        (100_000, 60),
    ];
    let mut prev: Option<(f64, f64)> = None; // (n_tp, us_per_step)
    for &(n_tp, n_steps) in &cases {
        let us_per_step = 1e6 / throughput(n_tp, n_steps, false);
        let marginal = prev.map(|(pn, pu)| (us_per_step - pu) / (f64::from(n_tp) - pn));
        match marginal {
            Some(m) => println!(
                "  n_tp = {n_tp:6}: {us_per_step:9.1} us/step, marginal {m:.4} us/step/particle"
            ),
            None => println!("  n_tp = {n_tp:6}: {us_per_step:9.1} us/step (massive-only)"),
        }
        prev = Some((f64::from(n_tp), us_per_step));
    }
}

// ---------------------------------------------------------------------------
// Adding, removing, kicking and reading test particles in place
// ---------------------------------------------------------------------------

/// A few test particles on varied orbits among the outer planets.
fn some_particles() -> Vec<State<Ecliptic, SSB>> {
    (0..6_u32)
        .map(|k| {
            let (pos, vel) = peri_pos_vel(
                6.0 + 4.0 * f64::from(k),
                0.2,
                10.0 * f64::from(k),
                f64::from(k),
            );
            make_state(3000 + k, pos, vel)
        })
        .collect()
}

/// Removing particles and adding the returned states back changes nothing
/// beyond roundoff: the run matches one that never removed them.
#[test]
fn remove_and_add_round_trip() {
    let (states, gms) = outer_solar_system();
    let tps = some_particles();
    let build = || WisdomHolman::new(&states, &gms, &tps, &[], 40.0, false, false, false).unwrap();
    let mut whole = build();
    whole.integrate_n_steps(2000).unwrap();

    let mut split = build();
    split.integrate_n_steps(1000).unwrap();
    let removed = split.remove_test_particles(&[4, 1]).unwrap();
    assert_eq!(split.n_test_particles(), 4);
    assert_eq!(removed[0].desig, tps[4].desig);
    assert_eq!(removed[1].desig, tps[1].desig);
    split.add_test_particles(&removed, &[]).unwrap();
    split.integrate_n_steps(1000).unwrap();

    let mut worst = 0.0_f64;
    for a in whole.test_particle_states() {
        let b = split
            .test_particle_states()
            .into_iter()
            .find(|s| s.desig == a.desig)
            .unwrap();
        worst = worst.max((Vector3::from(a.pos) - Vector3::from(b.pos)).norm());
    }
    println!(
        "remove_and_add_round_trip: worst position difference {worst:.3e} AU after 2000 steps"
    );
    assert!(
        worst < 1e-10,
        "round trip changed the trajectory by {worst:e} AU"
    );

    assert!(split.remove_test_particles(&[9]).is_err());
    assert!(split.remove_test_particles(&[0, 0]).is_err());
    let stale = make_state(
        4000,
        Vector3::new(10.0, 0.0, 0.0),
        Vector3::new(0.0, 0.005, 0.0),
    );
    assert!(
        split.add_test_particles(&[stale], &[]).is_err(),
        "epoch mismatch accepted"
    );
}

/// A kick equals starting from the kicked velocity, and the heliocentric
/// accessor equals the barycentric states minus the Sun's.
#[test]
fn kick_and_heliocentric_accessor() {
    let (states, gms) = outer_solar_system();
    let tps = some_particles();
    let dv: Vec<Vector3<f64>> = (0..tps.len())
        .map(|k| Vector3::new(1e-5, -2e-5, 3e-6) * (1.0 + f64::from(u32::try_from(k).unwrap())))
        .collect();
    let mut kicked =
        WisdomHolman::new(&states, &gms, &tps, &[], 40.0, false, false, false).unwrap();
    kicked.kick_test_particles(&dv).unwrap();
    let moved: Vec<_> = tps
        .iter()
        .zip(&dv)
        .map(|(s, d)| {
            State::new(
                s.desig.clone(),
                s.epoch,
                Vector3::from(s.pos),
                Vector3::from(s.vel) + d,
                SSB,
            )
        })
        .collect();
    let mut direct =
        WisdomHolman::new(&states, &gms, &moved, &[], 40.0, false, false, false).unwrap();
    kicked.integrate_n_steps(1000).unwrap();
    direct.integrate_n_steps(1000).unwrap();
    let worst = kicked
        .test_particle_states()
        .iter()
        .zip(direct.test_particle_states())
        .map(|(a, b)| (Vector3::from(a.pos) - Vector3::from(b.pos)).norm())
        .fold(0.0, f64::max);
    println!("kick_and_heliocentric_accessor: kick vs direct start, worst {worst:.3e} AU");
    assert!(
        worst < 1e-10,
        "kick differs from a direct start by {worst:e} AU"
    );
    assert!(kicked.kick_test_particles(&dv[..2]).is_err());

    let sun = &kicked.massive_states()[0];
    for ((pos, vel), state) in kicked
        .test_particle_heliocentric()
        .iter()
        .zip(kicked.test_particle_states())
    {
        let (p, v) = heliocentric(&state, sun);
        assert!((pos - p).norm() < 1e-12 && (vel - v).norm() < 1e-15);
    }
    assert_eq!(kicked.test_particle_desigs().len(), tps.len());
}

// ---------------------------------------------------------------------------
// Removal bounds
// ---------------------------------------------------------------------------

/// The perihelion bound removes a particle passing perihelion inside it and
/// keeps one outside it; the distance bound removes a particle moving past it;
/// an orbit shorter than one step is caught without a sign change.
#[test]
fn removal_bounds() {
    // Two inbound near-parabolic particles, perihelia 10 and 20 AU, a year out.
    let inbound = |q: f64, id: u32| {
        let (pos, vel) = peri_pos_vel(5000.0, 1.0 - q / 5000.0, 30.0, 0.0);
        let mut s = make_state(id, pos, vel);
        (s.pos, s.vel) =
            analytic_2_body((-365.25).into(), &Vector3::from(pos), &Vector3::from(vel))
                .map(|(p, v)| (p.into(), v.into()))
                .unwrap();
        s
    };
    let outbound = make_state(
        6002,
        Vector3::new(990.0, 0.0, 0.0),
        Vector3::new(0.05, 0.0, 0.0),
    );
    // An inner orbit (a = 20 AU, q = 4 AU, period ~ 89 yr) and a step of 200 yr.
    let (pi, vi) = peri_pos_vel(20.0, 0.8, 5.0, 0.0);
    let short = make_state(6003, pi, vi);

    let tps = vec![inbound(10.0, 6000), inbound(20.0, 6001), outbound];
    let mut sim = WisdomHolman::new(&[sun()], &[GMS], &tps, &[], 5.0, false, false, false).unwrap();
    sim.set_removal_bounds(Some(15.0), Some(1000.0)).unwrap();
    sim.integrate_n_steps(150).unwrap();
    let lost: Vec<_> = sim
        .lost_particles()
        .iter()
        .map(|l| (l.desig.clone(), l.reason))
        .collect();
    println!("removal_bounds: lost {lost:?}");
    assert!(lost.contains(&(Desig::Perm(6000), LostReason::PerihelionBound)));
    assert!(lost.contains(&(Desig::Perm(6002), LostReason::DistanceBound)));
    assert!(!lost.iter().any(|(d, _)| *d == Desig::Perm(6001)));
    assert_eq!(sim.n_test_particles(), 1);

    let mut coarse = WisdomHolman::new(
        &[sun()],
        &[GMS],
        &[short],
        &[],
        200.0 * 365.25,
        false,
        false,
        false,
    )
    .unwrap();
    coarse.set_removal_bounds(Some(15.0), None).unwrap();
    coarse.integrate_n_steps(1).unwrap();
    assert_eq!(
        coarse.lost_particles().len(),
        1,
        "an orbit shorter than the step was not caught"
    );
    assert!(sim.set_removal_bounds(Some(-1.0), None).is_err());
}

/// A particle on an orbit with perihelion 1.5 AU, `days` from perihelion
/// (negative before it).
fn near_perihelion(id: u32, days: f64) -> State<Ecliptic, SSB> {
    let (pos, vel) = peri_pos_vel(3.0, 0.5, 10.0, 0.0);
    let (pos, vel) = analytic_2_body(days.into(), &pos, &vel).unwrap();
    make_state(id, pos, vel)
}

/// The symplectic corrector drifts every particle several steps backward and
/// forward, which are not motion of the run: a particle that starts just past
/// perihelion inside the perihelion bound passes no perihelion in the run,
/// and is kept.
#[test]
fn removal_bounds_ignore_corrector_drifts() {
    let tp = near_perihelion(6100, 1.5);
    let mut sim = WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], 1.0, false, false, true).unwrap();
    sim.set_removal_bounds(Some(5.0), None).unwrap();
    sim.integrate_n_steps(10).unwrap();
    assert!(
        sim.lost_particles().is_empty(),
        "removed by a corrector drift: {:?}",
        sim.lost_particles()
    );
}

/// A perihelion passage whose radial velocity changes sign between two
/// drifts, here in a radial kick between integration calls, is caught.
#[test]
fn removal_bounds_passage_between_drifts() {
    let tp = near_perihelion(6101, -1.0001);
    let mut sim =
        WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], 1.0, false, false, false).unwrap();
    sim.set_removal_bounds(Some(5.0), None).unwrap();
    // Ends 1e-4 days before perihelion, moving inward.
    sim.integrate_n_steps(1).unwrap();
    assert!(sim.lost_particles().is_empty());
    let (pos, _) = sim.test_particle_heliocentric()[0];
    sim.kick_test_particles(&[pos.normalize() * 1e-6]).unwrap();
    sim.integrate_n_steps(1).unwrap();
    let lost: Vec<_> = sim.lost_particles().iter().map(|l| l.reason).collect();
    assert_eq!(lost, [LostReason::PerihelionBound]);
}

/// A drift longer than half an orbit whose ends both move outward has passed
/// perihelion and aphelion; with a step of 0.97 orbits, the second drift of
/// this run does so and the particle is removed there.
#[test]
fn removal_bounds_drift_over_half_an_orbit() {
    let (pos, vel) = peri_pos_vel(1.0, 0.8, 10.0, 0.0);
    let (pos, vel) = analytic_2_body(1.0.into(), &pos, &vel).unwrap();
    let tp = make_state(6102, pos, vel);
    let period = TAU / GMS.sqrt();
    let mut sim = WisdomHolman::new(
        &[sun()],
        &[GMS],
        &[tp],
        &[],
        0.97 * period,
        false,
        false,
        false,
    )
    .unwrap();
    sim.set_removal_bounds(Some(0.5), None).unwrap();
    sim.integrate_n_steps(3).unwrap();
    let lost: Vec<_> = sim.lost_particles().iter().map(|l| l.reason).collect();
    assert_eq!(lost, [LostReason::PerihelionBound]);
}

/// A removal found in a drift is recorded at the start of that drift: the
/// drifts of a call start at the call's epoch and then half a step into each
/// step, so a passage 3.3 days in, with a step of a day, is in the drift that
/// starts at 2.5 days.
#[test]
fn removal_recorded_at_drift_start() {
    let tp = near_perihelion(6103, -3.3);
    let mut sim =
        WisdomHolman::new(&[sun()], &[GMS], &[tp], &[], 1.0, false, false, false).unwrap();
    sim.set_removal_bounds(Some(5.0), None).unwrap();
    sim.integrate_n_steps(10).unwrap();
    let lost = sim.lost_particles();
    assert_eq!(lost.len(), 1);
    let days = lost[0].epoch.jd() - J2000;
    assert!((days - 2.5).abs() < 1e-9, "recorded at {days} days");
}

/// Sun, Jupiter (as NAIF 5, so its radius is known) and a test particle
/// passing Jupiter at `miss` AU, at closest approach at J2000 with relative
/// speed `speed` AU/day. Returns the three states `before` days earlier, from
/// a backward Radau integration of the full system.
fn jupiter_flyby(miss: f64, speed: f64, before: f64) -> (Vec<State<Ecliptic, SSB>>, [f64; 3]) {
    let gms = [GMS, gm(5), 0.0];
    let jup = planet(5, 5.20260, 0.04849, 1.303, 3.3);
    let (jpos, jvel) = (Vector3::from(jup.pos), Vector3::from(jup.vel));
    // Relative velocity radial, offset along the ecliptic normal: the two are
    // perpendicular, so J2000 is the planetocentric periapsis.
    let tp = make_state(
        7000,
        jpos + Vector3::z() * miss,
        jvel + jpos.normalize() * speed,
    );
    let jup = State::new(Desig::Naif(5), jup.epoch, jup.pos, jup.vel, SSB);
    let start = radau_states(&[sun(), jup, tp], &gms, -before);
    (start, gms)
}

/// Full-system Radau integration of `states` (all at the same epoch) for
/// `span` days, returning the end states.
fn radau_states(
    states: &[State<Ecliptic, SSB>],
    gms: &[f64],
    span: f64,
) -> Vec<State<Ecliptic, SSB>> {
    let n = states.len();
    let mut pos0 = DVector::<f64>::zeros(3 * n);
    let mut vel0 = DVector::<f64>::zeros(3 * n);
    for (i, state) in states.iter().enumerate() {
        for k in 0..3 {
            pos0[3 * i + k] = state.pos[k];
            vel0[3 * i + k] = state.vel[k];
        }
    }
    let gms = gms.to_vec();
    let func = |_t: Time<TDB>,
                pos: &DVector<f64>,
                _vel: &DVector<f64>,
                _meta: &mut (),
                _exact: bool|
     -> crate::errors::KeteResult<DVector<f64>> {
        let mut accel = DVector::<f64>::zeros(pos.len());
        for i in 0..n {
            let ri = Vector3::new(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
            for j in (i + 1)..n {
                let rj = Vector3::new(pos[3 * j], pos[3 * j + 1], pos[3 * j + 2]);
                let sep = rj - ri;
                let inv_r3 = sep.norm().powi(-3);
                for k in 0..3 {
                    accel[3 * i + k] += sep[k] * gms[j] * inv_r3;
                    accel[3 * j + k] -= sep[k] * gms[i] * inv_r3;
                }
            }
        }
        Ok(accel)
    };
    let epoch = states[0].epoch;
    let (pos, vel, ()) =
        RadauIntegrator::integrate(&func, pos0, vel0, epoch, epoch + span, (), None, None).unwrap();
    states
        .iter()
        .enumerate()
        .map(|(i, state)| {
            let get = |v: &DVector<f64>| Vector3::new(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
            State::new(state.desig.clone(), epoch + span, get(&pos), get(&vel), SSB)
        })
        .collect()
}

/// Particle heliocentric position after a WH run through the flyby, and the
/// number of steps resolved as encounters.
fn wh_flyby(
    start: &[State<Ecliptic, SSB>],
    gms: &[f64; 3],
    dt: f64,
    span: f64,
    encounters: Option<f64>,
) -> (WisdomHolman<Ecliptic>, Vector3<f64>) {
    let mut sim = WisdomHolman::new(
        &start[..2],
        &gms[..2],
        &start[2..],
        &[],
        dt,
        false,
        false,
        false,
    )
    .unwrap();
    if let Some(k) = encounters {
        sim = sim.with_close_encounters(k).unwrap();
    }
    sim.integrate_to(start[0].epoch + span).unwrap();
    let helio = if sim.n_test_particles() == 1 {
        heliocentric(&sim.test_particle_states()[0], &sim.massive_states()[0]).0
    } else {
        Vector3::repeat(f64::NAN)
    };
    (sim, helio)
}

/// A deep Jupiter flyby (0.02 AU, about 0.06 Hill radii) integrated through
/// with resolved encounters matches a full Radau integration of the system,
/// and is far better than the plain map, which cannot resolve it. The
/// remaining difference is the map's own error, on Jupiter and on the
/// particle outside the encounter radius, amplified by the encounter, so it
/// falls at second order as the step shrinks. A first step resolved in full
/// matches Radau to roundoff.
#[test]
fn close_encounter_matches_radau() {
    let (start, gms) = jupiter_flyby(0.02, 0.008, 400.0);
    let span = 800.0;
    let end = radau_states(&start, &gms, span);
    let reference = Vector3::from(end[2].pos) - Vector3::from(end[0].pos);

    println!("close_encounter_matches_radau: miss 0.02 AU, {span} days");
    let mut errors = Vec::new();
    for dt in [40.0, 20.0, 10.0] {
        let (sim, resolved) = wh_flyby(&start, &gms, dt, span, Some(3.0));
        let (_, plain) = wh_flyby(&start, &gms, dt, span, None);
        let err = (resolved - reference).norm();
        let plain_err = (plain - reference).norm();
        println!("  dt = {dt:4.1} d: resolved {err:.3e} AU; plain map {plain_err:.3e} AU");
        // Every approach within 3 Hill radii was resolved, so none is
        // recorded as unresolved.
        assert!(sim.closest_encounter().is_none());
        assert_eq!(sim.close_approaches().len(), 1);
        assert!(
            err < 1e-2 * plain_err,
            "resolved encounter is not far better than the plain map"
        );
        errors.push(err);
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        println!("  measured order: {order:.3}");
        assert!((1.7..=2.3).contains(&order), "not second order: {order}");
    }
    assert!(
        errors[2] < 2e-5,
        "resolved flyby vs Radau too far: {:e}",
        errors[2]
    );

    // One step, far from Jupiter, resolved in full: only the Radau tolerance.
    let end = radau_states(&start, &gms, 20.0);
    let reference = Vector3::from(end[2].pos) - Vector3::from(end[0].pos);
    let (_, resolved) = wh_flyby(&start, &gms, 20.0, 20.0, Some(1e6));
    let err = (resolved - reference).norm();
    println!("  one step resolved in full: {err:.3e} AU");
    assert!(
        err < 1e-12,
        "a fully resolved step does not match Radau: {err:e}"
    );
}

/// With encounters enabled but none occurring, the map is unchanged to the bit.
#[test]
fn close_encounter_inactive_is_identical() {
    let (states, gms) = outer_solar_system();
    let run = |enc: Option<f64>| {
        // Beyond Neptune, never within 3 of its Hill radii.
        let particles: Vec<_> = (0..4_u32)
            .map(|k| {
                let (pos, vel) = peri_pos_vel(45.0 + 5.0 * f64::from(k), 0.05, 20.0, f64::from(k));
                make_state(3100 + k, pos, vel)
            })
            .collect();
        let mut sim =
            WisdomHolman::new(&states, &gms, &particles, &[], 30.0, true, true, true).unwrap();
        if let Some(k) = enc {
            sim = sim.with_close_encounters(k).unwrap();
        }
        sim.integrate_n_steps(500).unwrap();
        (
            sim.close_approaches().len(),
            sim.test_particle_heliocentric(),
        )
    };
    let (passages, with) = run(Some(3.0));
    let (_, without) = run(None);
    assert_eq!(passages, 0);
    for (a, b) in with.iter().zip(&without) {
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }
}

/// A particle falling straight into Jupiter from 0.5 AU is removed as a
/// planet impact in the step it reaches the planet; a nonpositive radius is
/// rejected.
#[test]
fn close_encounter_planet_impact() {
    let jup = planet(5, 5.20260, 0.04849, 1.303, 3.3);
    let (jpos, jvel) = (Vector3::from(jup.pos), Vector3::from(jup.vel));
    let jup = State::new(Desig::Naif(5), jup.epoch, jup.pos, jup.vel, SSB);
    let offset = Vector3::new(0.3, -0.4, 0.0);
    let tp = make_state(7001, jpos + offset, jvel - offset * 0.016);
    let (fpos, fvel) = peri_pos_vel(2.5, 0.1, 5.0, 1.0);
    let far = make_state(7002, fpos, fvel);
    let run = |particles: &[State<Ecliptic, SSB>]| {
        let mut sim = WisdomHolman::new(
            &[sun(), jup.clone()],
            &[GMS, gm(5)],
            particles,
            &[],
            20.0,
            false,
            false,
            false,
        )
        .unwrap()
        .with_close_encounters(3.0)
        .unwrap();
        sim.integrate_n_steps(5).unwrap();
        sim
    };
    let sim = run(&[tp, far.clone()]);
    let lost = sim.lost_particles();
    println!("close_encounter_planet_impact: lost {lost:?}");
    assert_eq!(lost.len(), 1);
    assert_eq!(lost[0].reason, LostReason::PlanetImpact);
    // Falls in at 0.008 AU/day plus Jupiter's pull: about two months. The
    // impact is recorded when it happens, not at the start of its step.
    let days = lost[0].epoch.jd() - J2000;
    assert!((40.0..=80.0).contains(&days));
    assert!(
        (days / 20.0).fract().abs() > 1e-6,
        "impact at a step start: {days}"
    );
    // The particle after it in the arrays is untouched by the removal.
    assert_eq!(sim.test_particle_desigs(), &[Desig::Perm(7002)]);
    let alone = run(&[far]);
    assert_eq!(
        sim.test_particle_heliocentric(),
        alone.test_particle_heliocentric()
    );

    let (states, gms) = outer_solar_system();
    let sim = WisdomHolman::new(&states, &gms, &[], &[], 30.0, false, false, false).unwrap();
    assert!(sim.clone().with_close_encounters(0.0).is_err());
    assert!(sim.with_close_encounters(f64::NAN).is_err());
}

/// Steps resolved in full carry the same physics as the map: with GR, J2 and
/// each kind of non-gravitational force, particles whose every step goes through Radau follow a
/// fine-step run of the plain map far more closely than a plain run at the
/// same step does. A term missing or wrong in the encounter path would
/// instead leave a difference of the size of that term.
#[test]
fn close_encounter_physics_matches_map() {
    let gms = [GMS, gm(5)];
    let bodies = to_com_frame(&[sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)], &gms);
    let jpl = ParameterMask::all_fixed(
        NonGravKind::JplComet(JplCometNonGrav::new(1.0, 1.0, 2.0, 1.0, 0.0, 0.0)),
        vec![2e-8, -1e-8, 5e-9],
    )
    .unwrap();
    let non_gravs = vec![
        None,
        Some(dust(0.05)),
        Some(yark_raw(
            1e-3,
            0.5,
            Vector3::new(0.3, -0.2, 0.9).normalize(),
        )),
        Some(jpl),
    ];
    let particles: Vec<_> = (0..4_u32)
        .map(|k| {
            let (pos, vel) = peri_pos_vel(2.5 + 0.3 * f64::from(k), 0.15, 8.0, 1.3 * f64::from(k));
            make_state(7100 + k, pos, vel)
        })
        .collect();
    let span = 400.0;
    let run = |dt: f64, encounters: bool| {
        let mut sim =
            WisdomHolman::new(&bodies, &gms, &particles, &non_gravs, dt, true, true, false)
                .unwrap();
        if encounters {
            sim = sim.with_close_encounters(1e6).unwrap();
        }
        sim.integrate_to(Time::new(J2000 + span)).unwrap();
        let sun_end = sim.massive_states()[0].clone();
        sim.test_particle_states()
            .iter()
            .map(|s| heliocentric(s, &sun_end).0)
            .collect::<Vec<_>>()
    };
    let fine = run(0.25, false);
    let plain = run(8.0, false);
    let resolved = run(8.0, true);
    println!("close_encounter_physics_matches_map: {span} days, dt = 8 d vs fine map");
    for k in 0..4 {
        let err_plain = (plain[k] - fine[k]).norm();
        let err_resolved = (resolved[k] - fine[k]).norm();
        println!("  particle {k}: plain {err_plain:.3e} AU, resolved {err_resolved:.3e} AU");
        assert!(
            err_resolved < 0.1 * err_plain,
            "particle {k}: resolved steps do not follow the map's physics"
        );
    }
}

/// Integrating back through the flyby recovers the start: the resolved
/// steps, the planet paths and the detection all hold for a negative step.
#[test]
fn close_encounter_backward() {
    let (start, gms) = jupiter_flyby(0.02, 0.008, 400.0);
    let span = 800.0;
    let end = radau_states(&start, &gms, span);
    let reference = Vector3::from(start[2].pos) - Vector3::from(start[0].pos);
    let mut errors = Vec::new();
    for dt in [-40.0, -20.0, -10.0] {
        let (sim, back) = wh_flyby(&end, &gms, dt, -span, Some(3.0));
        let err = (back - reference).norm();
        println!("close_encounter_backward: dt = {dt}: {err:.3e} AU");
        assert_eq!(sim.close_approaches().len(), 1);
        errors.push(err);
    }
    for pair in errors.windows(2) {
        let order = (pair[0] / pair[1]).log2();
        assert!((1.7..=2.3).contains(&order), "not second order: {order}");
    }
}

/// The flyby's passage of Jupiter is recorded once, forward and backward,
/// with the fixture's periapsis: 0.02 AU at 0.008 AU/day relative, at J2000.
/// What remains is the map's own error on the approach, which falls with
/// the step. Taking the record empties it, and a run without resolved
/// encounters records nothing.
#[test]
fn close_encounter_records_passage() {
    let (start, gms) = jupiter_flyby(0.02, 0.008, 400.0);
    let span = 800.0;
    let end = radau_states(&start, &gms, span);
    for (from, dt) in [(&start, 20.0), (&start, 10.0), (&end, -10.0)] {
        let (mut sim, _) = wh_flyby(from, &gms, dt, span * dt.signum(), Some(3.0));
        let passages = sim.close_approaches();
        assert_eq!(passages.len(), 1, "dt = {dt}: {passages:?}");
        let pass = &passages[0];
        let t_err = (pass.epoch - Time::<TDB>::new(J2000)).elapsed;
        println!(
            "close_encounter_records_passage: dt = {dt}: distance {:.9} AU, speed {:.9} AU/day, epoch {t_err:+.2e} d",
            pass.distance, pass.speed
        );
        assert_eq!(pass.desig, Desig::Perm(7000));
        assert_eq!(pass.body, Desig::Naif(5));
        assert!((pass.distance - 0.02).abs() < 1e-6, "{}", pass.distance);
        assert!((pass.speed - 0.008).abs() < 1e-7, "{}", pass.speed);
        assert!(t_err.abs() < 1e-3, "{t_err}");
        assert_eq!(sim.take_close_approaches().len(), 1);
        assert!(sim.close_approaches().is_empty());
    }
    let (sim, _) = wh_flyby(&start, &gms, 10.0, span, None);
    assert!(sim.close_approaches().is_empty());
}

/// A passage at 0.6 AU, about 1.7 of Jupiter's Hill radii, is inside the
/// 3 Hill radius encounter radius but is only recorded once the close
/// approach radius is widened past it from its default of 1 Hill radius.
#[test]
fn close_approach_radius_filters_passages() {
    let (start, gms) = jupiter_flyby(0.6, 0.008, 400.0);
    let run = |approach: Option<f64>| {
        let mut sim = WisdomHolman::new(
            &start[..2],
            &gms[..2],
            &start[2..],
            &[],
            10.0,
            false,
            false,
            false,
        )
        .unwrap()
        .with_close_encounters(3.0)
        .unwrap();
        if let Some(k) = approach {
            sim = sim.with_close_approach_radius(k).unwrap();
        }
        sim.integrate_to(start[0].epoch + 800.0).unwrap();
        sim.take_close_approaches()
    };
    assert!(run(None).is_empty());
    let passages = run(Some(2.0));
    assert_eq!(passages.len(), 1, "{passages:?}");
    assert!((passages[0].distance - 0.6).abs() < 1e-2, "{passages:?}");

    let sim = wh_flyby(&start, &gms, 10.0, 0.0, None).0;
    assert_eq!(sim.close_approach_radius(), 1.0);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(sim.clone().with_close_approach_radius(bad).is_err());
    }
}

/// A particle temporarily captured by Jupiter (0.005 AU at 0.008 AU/day, so
/// within the encounter radius at both ends of every call) is as accurate
/// with the symplectic corrector enabled as without it, even with one step
/// per call: the corrector leaves it out and moves it with Jupiter.
#[test]
fn close_encounter_with_correctors() {
    let (start, gms) = jupiter_flyby(0.005, 0.008, 200.0);
    let span = 400.0;
    let end = radau_states(&start, &gms, span);
    let reference = Vector3::from(end[2].pos) - Vector3::from(end[0].pos);
    let error = |use_correctors: bool, per_call: u64| {
        let mut sim = WisdomHolman::new(
            &start[..2],
            &gms[..2],
            &start[2..],
            &[],
            10.0,
            false,
            false,
            use_correctors,
        )
        .unwrap()
        .with_close_encounters(3.0)
        .unwrap();
        for _ in 0..(40 / per_call) {
            sim.integrate_n_steps(per_call).unwrap();
        }
        let helio = heliocentric(&sim.test_particle_states()[0], &sim.massive_states()[0]).0;
        (helio - reference).norm()
    };
    let off = error(false, 40);
    println!("close_encounter_with_correctors: corrector off {off:.3e} AU");
    for per_call in [40, 1] {
        let on = error(true, per_call);
        println!("  corrector on, {per_call} steps per call: {on:.3e} AU");
        assert!(
            on < 2.0 * off,
            "the corrector damages the captured particle"
        );
    }
}

/// The solar J2 term in a resolved step has the effect it has in the map:
/// for a particle at 0.3 AU, the change J2 makes over 400 days is the same
/// with every step resolved as in a fine-step run of the plain map.
#[test]
fn close_encounter_j2_matches_map() {
    let gms = [GMS, gm(5)];
    let bodies = to_com_frame(&[sun(), planet(5, 5.20260, 0.04849, 1.303, 3.3)], &gms);
    let (pos, vel) = peri_pos_vel(0.3, 0.1, 20.0, 0.4);
    let tp = [make_state(7200, pos, vel)];
    let run = |dt: f64, include_j2: bool, resolve: bool| {
        let mut sim =
            WisdomHolman::new(&bodies, &gms, &tp, &[], dt, false, include_j2, false).unwrap();
        if resolve {
            sim = sim.with_close_encounters(1e6).unwrap();
        }
        sim.integrate_to(Time::new(J2000 + 400.0)).unwrap();
        heliocentric(&sim.test_particle_states()[0], &sim.massive_states()[0]).0
    };
    let map_effect = run(0.05, true, false) - run(0.05, false, false);
    let resolved_effect = run(2.0, true, true) - run(2.0, false, true);
    let rel = (resolved_effect - map_effect).norm() / map_effect.norm();
    println!(
        "close_encounter_j2_matches_map: J2 moves the particle {:.3e} AU; resolved differs by {rel:.2e} of it",
        map_effect.norm()
    );
    assert!(rel < 1e-2, "J2 in resolved steps differs from the map's");
}

/// A flyby that turns an inbound particle outward is not a perihelion
/// passage: Jupiter turns this one onto an orbit with perihelion inside the
/// bound (about 4 AU), but its distance from the Sun never drops below
/// about 4.94 AU, so it is kept.
#[test]
fn close_encounter_flyby_is_not_a_perihelion() {
    let gms = [GMS, gm(5), 0.0];
    let jup = planet(5, 5.20260, 0.04849, 1.303, 3.3);
    let (jpos, jvel) = (Vector3::from(jup.pos), Vector3::from(jup.vel));
    // Planetocentric periapsis 0.01 AU sunward of Jupiter, moving along the
    // ecliptic normal, on a hyperbola (e = 1.155) that turns the particle by
    // 120 degrees: in toward the Sun before, out after.
    let rp = 0.01;
    let tp = make_state(
        7300,
        jpos - jpos.normalize() * rp,
        jvel + Vector3::z() * (gm(5) * 2.155 / rp).sqrt(),
    );
    let jup = State::new(Desig::Naif(5), jup.epoch, jup.pos, jup.vel, SSB);
    let start = radau_states(&[sun(), jup, tp], &gms, -40.0);
    let mut sim = WisdomHolman::new(
        &start[..2],
        &gms[..2],
        &start[2..],
        &[],
        20.0,
        false,
        false,
        false,
    )
    .unwrap()
    .with_close_encounters(3.0)
    .unwrap();
    sim.set_removal_bounds(Some(4.6), None).unwrap();
    sim.integrate_n_steps(4).unwrap();
    assert_eq!(sim.close_approaches().len(), 1);
    assert!(
        sim.lost_particles().is_empty(),
        "the flyby was taken for a perihelion: {:?}",
        sim.lost_particles()
    );
    let (pos, vel) = heliocentric(&sim.test_particle_states()[0], &sim.massive_states()[0]);
    let peri = compute_peri_dist(&pos, &vel, GMS);
    println!("close_encounter_flyby_is_not_a_perihelion: perihelion now {peri:.3} AU");
    assert!(pos.dot(&vel) > 0.0 && peri < 4.6);
}

/// The corrector kicks bodies at positions along their orbits that the run
/// never visits, so it records no approaches: with every encounter step of
/// this flyby resolved, stepping one step per call with the corrector on
/// reports none, as with it off.
#[test]
fn corrector_records_no_encounters() {
    let (start, gms) = jupiter_flyby(0.02, 0.008, 400.0);
    for (use_correctors, per_call) in [(false, 1), (true, 40), (true, 1)] {
        let mut sim = WisdomHolman::new(
            &start[..2],
            &gms[..2],
            &start[2..],
            &[],
            20.0,
            false,
            false,
            use_correctors,
        )
        .unwrap()
        .with_close_encounters(3.0)
        .unwrap();
        for _ in 0..(40 / per_call) {
            sim.integrate_n_steps(per_call).unwrap();
        }
        assert_eq!(sim.close_approaches().len(), 1);
        assert!(
            sim.closest_encounter().is_none(),
            "correctors {use_correctors}, {per_call} steps per call: {:?}",
            sim.closest_encounter()
        );
    }
}

/// The Earth-Moon barycenter's impact radius is the Earth's, not the span of
/// the lunar orbit its mass-table entry carries.
#[test]
fn close_encounter_emb_radius() {
    let emb = State::new(
        Desig::Naif(3),
        Time::new(J2000),
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, (GMS).sqrt(), 0.0),
        SSB,
    );
    let sim = WisdomHolman::new(
        &[sun(), emb],
        &[GMS, gm(3)],
        &[],
        &[],
        1.0,
        false,
        false,
        false,
    )
    .unwrap();
    let earth = GravParams::known_masses()
        .iter()
        .find(|p| p.naif_id == 399)
        .unwrap()
        .radius;
    assert_eq!(sim.radii, vec![f64::from(earth)]);
}

/// A registered mass with a radius sets a body's impact radius, ahead of the
/// mass table; one without a radius falls back to the table.
#[test]
fn impact_radius_prefers_registered_masses() {
    let known = GravParams::known_masses();
    let table = |id: i32| f64::from(known.iter().find(|p| p.naif_id == id).unwrap().radius);
    let custom = GravParams::new(2_000_433, 1e-20, 1e-6);
    assert_eq!(
        impact_radius(2_000_433, std::slice::from_ref(&custom), known),
        f64::from(1e-6_f32)
    );
    assert_eq!(impact_radius(2_000_433, &[], known), 0.0);
    let no_radius = GravParams::new(5, gm(5), 0.0);
    assert_eq!(impact_radius(5, &[no_radius], known), table(5));
    let bigger = GravParams::new(5, gm(5), 1e-3);
    assert_eq!(impact_radius(5, &[bigger], known), f64::from(1e-3_f32));
    assert_eq!(impact_radius(3, &[], known), table(399));
}
