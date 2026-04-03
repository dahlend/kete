/// Stress tests for numerical integrators.
///
/// Each scenario defines a second-order IVP y'' = f(t, y, y') with a known
/// analytic solution, then runs every integrator through it and reports errors.
///
/// Run all scenarios (prints a comparison table):
///   `cargo test -p kete_core --lib propagation::stress_tests -- --nocapture --ignored`
use std::time::Instant;

use nalgebra::{SMatrix, Vector3};

use super::{
    BulirschStoerIntegrator, CentralAccelMeta, GaussJacksonIntegrator, PC15, RadauIntegrator,
    analytic_2_body, central_accel,
};
use crate::constants::GMS;
use crate::time::{TDB, Time};

/// Two-body analytic init for the second-order Picard integrator.
fn picard_2body_init<const N: usize>(
    times: &[Time<TDB>; N],
    init_pos: &Vector3<f64>,
    init_vel: &Vector3<f64>,
) -> (SMatrix<f64, 3, N>, SMatrix<f64, 3, N>) {
    let t0 = times[0];
    let mut pos_mat: SMatrix<f64, 3, N> = SMatrix::zeros();
    let mut vel_mat: SMatrix<f64, 3, N> = SMatrix::zeros();
    pos_mat.set_column(0, init_pos);
    vel_mat.set_column(0, init_vel);
    for (idx, t) in times.iter().enumerate().skip(1) {
        let dt = *t - t0;
        let (p, v) = analytic_2_body(dt, init_pos, init_vel, None).unwrap();
        pos_mat.set_column(idx, &p);
        vel_mat.set_column(idx, &v);
    }
    (pos_mat, vel_mat)
}

/// Result of one integrator on one scenario.
struct RunResult {
    name: &'static str,
    pos_err: f64,
    vel_err: f64,
    energy_err: f64,
    elapsed_ms: f64,
    evals: usize,
}

/// A test scenario: initial conditions + integration span + analytic reference.
struct Scenario {
    label: &'static str,
    pos: Vector3<f64>,
    vel: Vector3<f64>,
    t_days: f64,
    gj_step: f64,
    picard_step: f64,
}

impl Scenario {
    /// Compute the analytic two-body solution at `t_days`.
    fn exact(&self) -> (Vector3<f64>, Vector3<f64>) {
        analytic_2_body(self.t_days.into(), &self.pos, &self.vel, None).unwrap()
    }

    /// Orbital energy (specific).
    fn energy(pos: &Vector3<f64>, vel: &Vector3<f64>) -> f64 {
        0.5 * vel.norm_squared() - GMS / pos.norm()
    }

    /// Run all integrators and return results.
    fn run(&self) -> Vec<RunResult> {
        let (exact_p, exact_v) = self.exact();
        let e0 = Self::energy(&self.pos, &self.vel);
        let mut results = Vec::new();

        // --- Radau ---
        {
            let mut meta = CentralAccelMeta::default();
            let t0 = Instant::now();
            let res = RadauIntegrator::integrate(
                &central_accel,
                self.pos,
                self.vel,
                0.0.into(),
                self.t_days.into(),
                meta.clone(),
                None,
            );
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Ok((p, v, m)) = res {
                meta = m;
                results.push(RunResult {
                    name: "Radau",
                    pos_err: (p - exact_p).norm(),
                    vel_err: (v - exact_v).norm(),
                    energy_err: ((Self::energy(&p, &v) - e0) / e0).abs(),
                    elapsed_ms,
                    evals: meta.eval_count,
                });
            } else {
                results.push(RunResult {
                    name: "Radau",
                    pos_err: f64::NAN,
                    vel_err: f64::NAN,
                    energy_err: f64::NAN,
                    elapsed_ms,
                    evals: 0,
                });
            }
        }

        // --- Bulirsch-Stoer ---
        {
            let mut meta = CentralAccelMeta::default();
            let t0 = Instant::now();
            let res = BulirschStoerIntegrator::integrate(
                &central_accel,
                self.pos,
                self.vel,
                0.0.into(),
                self.t_days.into(),
                meta.clone(),
                None,
            );
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Ok((p, v, m)) = res {
                meta = m;
                results.push(RunResult {
                    name: "BS",
                    pos_err: (p - exact_p).norm(),
                    vel_err: (v - exact_v).norm(),
                    energy_err: ((Self::energy(&p, &v) - e0) / e0).abs(),
                    elapsed_ms,
                    evals: meta.eval_count,
                });
            } else {
                results.push(RunResult {
                    name: "BS",
                    pos_err: f64::NAN,
                    vel_err: f64::NAN,
                    energy_err: f64::NAN,
                    elapsed_ms,
                    evals: 0,
                });
            }
        }

        // --- Gauss-Jackson ---
        {
            let mut meta = CentralAccelMeta::default();
            let t0 = Instant::now();
            let res = GaussJacksonIntegrator::integrate(
                &central_accel,
                self.pos,
                self.vel,
                0.0.into(),
                self.t_days.into(),
                meta.clone(),
                None,
                Some(self.gj_step),
            );
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Ok((p, v, m)) = res {
                meta = m;
                results.push(RunResult {
                    name: "GJ",
                    pos_err: (p - exact_p).norm(),
                    vel_err: (v - exact_v).norm(),
                    energy_err: ((Self::energy(&p, &v) - e0) / e0).abs(),
                    elapsed_ms,
                    evals: meta.eval_count,
                });
            } else {
                results.push(RunResult {
                    name: "GJ",
                    pos_err: f64::NAN,
                    vel_err: f64::NAN,
                    energy_err: f64::NAN,
                    elapsed_ms,
                    evals: 0,
                });
            }
        }

        // --- Picard-Chebyshev ---
        {
            let mut meta = CentralAccelMeta::default();
            let t0 = Instant::now();
            let res = PC15.integrate_second_order(
                &central_accel,
                &picard_2body_init,
                self.pos,
                self.vel,
                0.0.into(),
                self.t_days.into(),
                self.picard_step,
                &mut meta,
            );
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            if let Ok((p, v)) = res {
                results.push(RunResult {
                    name: "PC15",
                    pos_err: (p - exact_p).norm(),
                    vel_err: (v - exact_v).norm(),
                    energy_err: ((Self::energy(&p, &v) - e0) / e0).abs(),
                    elapsed_ms,
                    evals: meta.eval_count,
                });
            } else {
                results.push(RunResult {
                    name: "PC15",
                    pos_err: f64::NAN,
                    vel_err: f64::NAN,
                    energy_err: f64::NAN,
                    elapsed_ms,
                    evals: 0,
                });
            }
        }

        results
    }
}

fn print_table(scenarios: &[Scenario]) {
    eprintln!(
        "\n{:<40}  {:<6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>8}",
        "Scenario", "Integ", "pos err", "vel err", "|dE/E|", "time(ms)", "evals"
    );
    eprintln!("{}", "-".repeat(112));

    for s in scenarios {
        let results = s.run();
        for (i, r) in results.iter().enumerate() {
            let label = if i == 0 { s.label } else { "" };
            eprintln!(
                "{:<40}  {:<6}  {:>12.3e}  {:>12.3e}  {:>12.3e}  {:>10.2}  {:>8}",
                label, r.name, r.pos_err, r.vel_err, r.energy_err, r.elapsed_ms, r.evals
            );
        }
        eprintln!();
    }
}

/// Build orbital initial conditions from Keplerian elements.
/// Returns (position, velocity) in the orbital plane (i=0) or rotated.
fn keplerian_ic(a: f64, e: f64, inc_deg: f64, omega_deg: f64) -> (Vector3<f64>, Vector3<f64>) {
    // Start at perihelion: r = a(1-e), v = sqrt(GMS * (2/r - 1/a))
    let r_peri = a * (1.0 - e);
    let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();

    let mut pos = Vector3::new(r_peri, 0.0, 0.0);
    let mut vel = Vector3::new(0.0, v_peri, 0.0);

    // Rotate by argument of perihelion (in orbital plane)
    let w = omega_deg.to_radians();
    if w.abs() > 1e-15 {
        let cw = w.cos();
        let sw = w.sin();
        pos = Vector3::new(cw * pos[0] - sw * pos[1], sw * pos[0] + cw * pos[1], 0.0);
        vel = Vector3::new(cw * vel[0] - sw * vel[1], sw * vel[0] + cw * vel[1], 0.0);
    }

    // Rotate by inclination (around x-axis)
    let i = inc_deg.to_radians();
    if i.abs() > 1e-15 {
        let ci = i.cos();
        let si = i.sin();
        pos = Vector3::new(pos[0], ci * pos[1] - si * pos[2], si * pos[1] + ci * pos[2]);
        vel = Vector3::new(vel[0], ci * vel[1] - si * vel[2], si * vel[1] + ci * vel[2]);
    }

    (pos, vel)
}

// ========================================================================
// Stress test scenarios
// ========================================================================

#[test]
#[ignore = "expensive diagnostic, run manually"]
fn integrator_stress_comparison() {
    let scenarios = vec![
        // 1. Mild ellipse (e=0.5) -- baseline sanity check, 1000 days
        {
            let (p, v) = keplerian_ic(1.0, 0.5, 0.0, 0.0);
            Scenario {
                label: "e=0.5, a=1 AU, 1000d",
                pos: p,
                vel: v,
                t_days: 1000.0,
                gj_step: 0.5,
                picard_step: 10.0,
            }
        },
        // 2. High eccentricity (e=0.95) -- sharp perihelion curvature
        {
            let (p, v) = keplerian_ic(1.0, 0.95, 0.0, 0.0);
            Scenario {
                label: "e=0.95, a=1 AU, 365d",
                pos: p,
                vel: v,
                t_days: 365.25,
                gj_step: 0.02,
                picard_step: 2.0,
            }
        },
        // 3. Very high eccentricity (e=0.99) -- extreme perihelion
        {
            let (p, v) = keplerian_ic(2.0, 0.99, 0.0, 0.0);
            Scenario {
                label: "e=0.99, a=2 AU, 1000d",
                pos: p,
                vel: v,
                t_days: 1000.0,
                gj_step: 0.01,
                picard_step: 1.0,
            }
        },
        // 4. Sun-grazer (r_peri = 0.01 AU, e=0.999)
        {
            // a = r_peri / (1-e) = 0.01 / 0.001 = 10 AU
            let (p, v) = keplerian_ic(10.0, 0.999, 0.0, 0.0);
            Scenario {
                label: "Sun-grazer e=0.999, 100d",
                pos: p,
                vel: v,
                t_days: 100.0,
                gj_step: 0.001,
                picard_step: 0.5,
            }
        },
        // 5. Circular orbit, long integration (100k days ~ 274 years)
        {
            let (p, v) = keplerian_ic(1.0, 0.0, 0.0, 0.0);
            Scenario {
                label: "Circular, 100k days",
                pos: p,
                vel: v,
                t_days: 100_000.0,
                gj_step: 1.0,
                picard_step: 20.0,
            }
        },
        // 6. Circular orbit, very long (1M days ~ 2740 years)
        {
            let (p, v) = keplerian_ic(1.0, 0.0, 0.0, 0.0);
            Scenario {
                label: "Circular, 1M days",
                pos: p,
                vel: v,
                t_days: 1_000_000.0,
                gj_step: 1.0,
                picard_step: 20.0,
            }
        },
        // 7. 3D inclined eccentric orbit -- tests all 3 components
        {
            let (p, v) = keplerian_ic(1.5, 0.7, 30.0, 45.0);
            Scenario {
                label: "e=0.7, i=30, w=45, 1000d",
                pos: p,
                vel: v,
                t_days: 1000.0,
                gj_step: 0.1,
                picard_step: 5.0,
            }
        },
        // 8. Retrograde highly eccentric
        {
            let (p, v) = keplerian_ic(1.0, 0.9, 150.0, 0.0);
            Scenario {
                label: "e=0.9, i=150 (retro), 500d",
                pos: p,
                vel: v,
                t_days: 500.0,
                gj_step: 0.05,
                picard_step: 2.0,
            }
        },
        // 9. Tight inner orbit (a=0.1 AU, e=0.3) -- fast period (~11.5 days)
        {
            let (p, v) = keplerian_ic(0.1, 0.3, 10.0, 0.0);
            Scenario {
                label: "a=0.1 AU, e=0.3, 100d",
                pos: p,
                vel: v,
                t_days: 100.0,
                gj_step: 0.05,
                picard_step: 1.0,
            }
        },
        // 10. Wide slow orbit (a=30 AU, e=0.1) -- ~60k day period
        {
            let (p, v) = keplerian_ic(30.0, 0.1, 5.0, 0.0);
            Scenario {
                label: "a=30 AU, e=0.1, 10000d",
                pos: p,
                vel: v,
                t_days: 10_000.0,
                gj_step: 2.0,
                picard_step: 30.0,
            }
        },
    ];

    print_table(&scenarios);
}

/// Forward-backward reversibility test.
/// Integrates forward t days, then backward t days, and checks that the
/// initial state is recovered.
#[test]
#[ignore = "expensive diagnostic, run manually"]
#[allow(clippy::collapsible_if, reason = "clarity over brevity here")]
fn integrator_reversibility() {
    let cases = [
        ("e=0.5, 500d", 1.0_f64, 0.5, 500.0, 0.5, 10.0),
        ("e=0.9, 200d", 1.0, 0.9, 200.0, 0.05, 2.0),
        ("circular, 10000d", 1.0, 0.0, 10_000.0, 1.0, 20.0),
    ];

    eprintln!(
        "\n{:<25}  {:<6}  {:>14}  {:>14}",
        "Scenario", "Integ", "pos roundtrip", "vel roundtrip"
    );
    eprintln!("{}", "-".repeat(65));

    for (label, a, e, t, gj_h, pc_h) in &cases {
        let (p0, v0) = keplerian_ic(*a, *e, 0.0, 0.0);

        // Radau
        if let Ok((pm, vm, _)) = RadauIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            (*t).into(),
            CentralAccelMeta::default(),
            None,
        ) {
            if let Ok((pf, vf, _)) = RadauIntegrator::integrate(
                &central_accel,
                pm,
                vm,
                (*t).into(),
                0.0.into(),
                CentralAccelMeta::default(),
                None,
            ) {
                eprintln!(
                    "{:<25}  {:<6}  {:>14.3e}  {:>14.3e}",
                    label,
                    "Radau",
                    (pf - p0).norm(),
                    (vf - v0).norm()
                );
            }
        }

        // BS
        if let Ok((pm, vm, _)) = BulirschStoerIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            (*t).into(),
            CentralAccelMeta::default(),
            None,
        ) {
            if let Ok((pf, vf, _)) = BulirschStoerIntegrator::integrate(
                &central_accel,
                pm,
                vm,
                (*t).into(),
                0.0.into(),
                CentralAccelMeta::default(),
                None,
            ) {
                eprintln!(
                    "{:<25}  {:<6}  {:>14.3e}  {:>14.3e}",
                    "",
                    "BS",
                    (pf - p0).norm(),
                    (vf - v0).norm()
                );
            }
        }

        // GJ
        if let Ok((pm, vm, _)) = GaussJacksonIntegrator::integrate(
            &central_accel,
            p0,
            v0,
            0.0.into(),
            (*t).into(),
            CentralAccelMeta::default(),
            None,
            Some(*gj_h),
        ) {
            if let Ok((pf, vf, _)) = GaussJacksonIntegrator::integrate(
                &central_accel,
                pm,
                vm,
                (*t).into(),
                0.0.into(),
                CentralAccelMeta::default(),
                None,
                Some(*gj_h),
            ) {
                eprintln!(
                    "{:<25}  {:<6}  {:>14.3e}  {:>14.3e}",
                    "",
                    "GJ",
                    (pf - p0).norm(),
                    (vf - v0).norm()
                );
            }
        }

        // Picard
        if let Ok((pm, vm)) = PC15.integrate_second_order(
            &central_accel,
            &picard_2body_init,
            p0,
            v0,
            0.0.into(),
            (*t).into(),
            *pc_h,
            &mut CentralAccelMeta::default(),
        ) {
            if let Ok((pf, vf)) = PC15.integrate_second_order(
                &central_accel,
                &picard_2body_init,
                pm,
                vm,
                (*t).into(),
                0.0.into(),
                *pc_h,
                &mut CentralAccelMeta::default(),
            ) {
                eprintln!(
                    "{:<25}  {:<6}  {:>14.3e}  {:>14.3e}",
                    "",
                    "PC15",
                    (pf - p0).norm(),
                    (vf - v0).norm()
                );
            }
        }

        eprintln!();
    }
}
