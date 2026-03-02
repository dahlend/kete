//! Initial Orbit Determination (IOD).
//!
//! Given a small number of optical observations, compute an approximate
//! heliocentric state that can seed the batch least-squares differential
//! corrector.  Two classical methods are provided:
//!
//! - **Gauss**: classical method from exactly 3 optical observations.
//! - **Laplace**: derivative-based method from 3+ optical observations.

use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, Vector};
use kete_core::prelude::{Error, KeteResult, State};

use crate::Observation;

/// Classical Gauss method for IOD from exactly 3 optical observations.
///
/// Returns all physically valid candidate states (SSB-centered, Equatorial).
/// The 8th-degree range polynomial may have multiple roots; each valid root
/// produces a separate candidate.
///
/// # Errors
/// - Fewer than 3 optical observations.
/// - No valid roots found.
/// - Non-optical observations passed.
///
/// References:
///   - Curtis, "Orbital Mechanics for Engineering Students", Ch. 5
///   - Bate, Mueller & White, "Fundamentals of Astrodynamics", Ch. 5
pub fn gauss_iod(obs: &[Observation]) -> KeteResult<Vec<State<Equatorial>>> {
    if obs.len() < 3 {
        return Err(Error::ValueError(
            "Gauss IOD requires at least 3 optical observations".into(),
        ));
    }

    // Pick the first, middle, and last observations.
    let i1 = 0;
    let i2 = obs.len() / 2;
    let i3 = obs.len() - 1;

    let (ra1, dec1, obs1) = obs[i1].as_optical()?;
    let (ra2, dec2, obs2) = obs[i2].as_optical()?;
    let (ra3, dec3, obs3) = obs[i3].as_optical()?;
    // Line-of-sight unit vectors
    let rho1 = Vector::<Equatorial>::from_ra_dec(ra1, dec1);
    let rho2 = Vector::<Equatorial>::from_ra_dec(ra2, dec2);
    let rho3 = Vector::<Equatorial>::from_ra_dec(ra3, dec3);

    // Observer positions (SSB-centered ~ heliocentric for IOD)
    let r_obs1 = obs1.pos;
    let r_obs2 = obs2.pos;
    let r_obs3 = obs3.pos;

    // Time intervals (days)
    let t1 = obs1.epoch.jd;
    let t2 = obs2.epoch.jd;
    let t3 = obs3.epoch.jd;
    let tau1 = t1 - t2; // negative
    let tau3 = t3 - t2; // positive
    let tau = tau3 - tau1; // t3 - t1

    // Cross products of line-of-sight vectors
    let p1 = rho2.cross(&rho3);
    let p2 = rho1.cross(&rho3);
    let p3 = rho1.cross(&rho2);

    // Scalar triple product D0 = rho1 . (rho2 x rho3)
    let d0 = rho1.dot(&p1);
    if d0.abs() < 1e-14 {
        return Err(Error::ValueError(
            "Gauss IOD: coplanar lines of sight (D0 ~ 0)".into(),
        ));
    }

    // D matrix: D[i][j] = R_i . p_{j+1}
    let d = [
        [r_obs1.dot(&p1), r_obs1.dot(&p2), r_obs1.dot(&p3)],
        [r_obs2.dot(&p1), r_obs2.dot(&p2), r_obs2.dot(&p3)],
        [r_obs3.dot(&p1), r_obs3.dot(&p2), r_obs3.dot(&p3)],
    ];

    // Coefficients A and B for the range polynomial
    let a_coeff = (-d[0][1] * (tau3 / tau) + d[1][1] + d[2][1] * (tau1 / tau)) / d0;

    let b_coeff = (d[0][1] * (tau * tau - tau3 * tau3) * (tau3 / tau)
        + d[2][1] * (tau * tau - tau1 * tau1) * (tau1 / tau))
        / (6.0 * d0);

    // E = R2 . rho2, R2_sq = |R2|^2
    let e_coeff = r_obs2.dot(&rho2);
    let r2_sq = r_obs2.dot(&r_obs2);

    // Solve 8th-degree polynomial in r2:
    //   r2^8 - (A^2 + 2*A*E + R2^2) * r2^6 - 2*mu*B*(A+E) * r2^3 - mu^2 * B^2 = 0
    let c6 = -(a_coeff * a_coeff + 2.0 * a_coeff * e_coeff + r2_sq);
    let c3 = -2.0 * GMS * b_coeff * (a_coeff + e_coeff);
    let c0 = -(GMS * b_coeff).powi(2);

    let roots = solve_r2_polynomial(c6, c3, c0);

    // For each valid root, recover slant ranges and full state
    let mut results = Vec::new();
    for r2 in roots {
        if r2 < 0.01 {
            continue;
        }

        // Slant ranges (topocentric distances)
        let r2_cubed = r2 * r2 * r2;
        let c1 = tau3 / tau * (1.0 + GMS / (6.0 * r2_cubed) * (tau * tau - tau3 * tau3));
        let c3_coeff = -tau1 / tau * (1.0 + GMS / (6.0 * r2_cubed) * (tau * tau - tau1 * tau1));

        let rho_mag1 = (-c1 * d[0][0] + d[1][0] - c3_coeff * d[2][0]) / (c1 * d0);
        let rho_mag2 = a_coeff + GMS * b_coeff / r2_cubed;
        let rho_mag3 = (-c1 * d[0][2] + d[1][2] - c3_coeff * d[2][2]) / (c3_coeff * d0);

        // All slant ranges must be positive
        if rho_mag1 < 0.0 || rho_mag2 < 0.0 || rho_mag3 < 0.0 {
            continue;
        }

        // Heliocentric position vectors
        let r1 = r_obs1 + rho1 * rho_mag1;
        let r2_vec = r_obs2 + rho2 * rho_mag2;
        let r3 = r_obs3 + rho3 * rho_mag3;

        // Lagrange f and g coefficients (series approximation)
        let (f1, g1) = lagrange_fg(tau1, r2_cubed);
        let (f3, g3) = lagrange_fg(tau3, r2_cubed);

        // Velocity at observation 2: v2 = (f1 * r3 - f3 * r1) / (f1 * g3 - f3 * g1)
        let denom = f1 * g3 - f3 * g1;
        if denom.abs() < 1e-20 {
            continue;
        }
        let v2 = (r3 * f1 - r1 * f3) / denom;

        let state = State::new(
            kete_core::desigs::Desig::Empty,
            obs2.epoch,
            r2_vec,
            v2,
            0, // SSB centered
        );
        results.push(state);
    }

    if results.is_empty() {
        return Err(Error::ValueError("Gauss IOD: no valid roots found".into()));
    }
    Ok(results)
}

/// Laplace method for IOD from 3+ optical observations.
///
/// Returns all physically valid candidate states (SSB-centered, Equatorial).
/// Uses finite-difference estimates of the time derivatives of the line-of-sight
/// direction to solve for the geocentric distance at the middle observation.
///
/// # Errors
/// - Fewer than 3 optical observations.
/// - No valid roots found.
/// - Non-optical observations passed.
pub fn laplace_iod(obs: &[Observation]) -> KeteResult<Vec<State<Equatorial>>> {
    if obs.len() < 3 {
        return Err(Error::ValueError(
            "Laplace IOD requires at least 3 optical observations".into(),
        ));
    }

    // Pick first, middle, last
    let i1 = 0;
    let i2 = obs.len() / 2;
    let i3 = obs.len() - 1;

    let (ra1, dec1, obs1) = obs[i1].as_optical()?;
    let (ra2, dec2, obs2) = obs[i2].as_optical()?;
    let (ra3, dec3, obs3) = obs[i3].as_optical()?;

    let rho1 = Vector::<Equatorial>::from_ra_dec(ra1, dec1);
    let rho2 = Vector::<Equatorial>::from_ra_dec(ra2, dec2);
    let rho3 = Vector::<Equatorial>::from_ra_dec(ra3, dec3);

    // Time intervals
    let t1 = obs1.epoch.jd;
    let t2 = obs2.epoch.jd;
    let t3 = obs3.epoch.jd;
    let tau1 = t1 - t2;
    let tau3 = t3 - t2;

    // Line-of-sight time derivatives at observation 2 via finite differences
    let dt = t3 - t1;
    let rho_dot = (rho3 - rho1) / dt;
    let rho_ddot =
        (rho3 * tau1 - rho1 * tau3 + rho2 * (tau3 - tau1)) * (2.0 / (tau1 * tau3 * (tau3 - tau1)));

    // Observer position and acceleration at epoch 2
    let r_obs = obs2.pos;
    // Observer acceleration: approximate from two-body around Sun
    // a_obs = -GMS * R / |R|^3
    let r_obs_mag = r_obs.norm();
    let r_obs_mag_cubed = r_obs_mag * r_obs_mag * r_obs_mag;
    let a_obs = r_obs * (-GMS / r_obs_mag_cubed);

    // Form the Laplace determinants
    // D = rho2 . (rho_dot x rho_ddot)
    let d_det = rho2.dot(&rho_dot.cross(&rho_ddot));
    if d_det.abs() < 1e-20 {
        return Err(Error::ValueError(
            "Laplace IOD: singular geometry (D ~ 0)".into(),
        ));
    }

    // D_R: replace rho_ddot column with (-a_obs) in the triple product
    let d_r = rho2.dot(&rho_dot.cross(&(-a_obs)));

    // D_rho: replace rho_ddot column with R_obs
    let d_rho = rho2.dot(&rho_dot.cross(&r_obs));

    let alpha = d_r / d_det;
    let beta_coeff = -d_rho / d_det;
    let e_dot = r_obs.dot(&rho2);

    // r^8 + c6*r^6 + c3*r^3 + c0 = 0
    let c6 = -(alpha * alpha + 2.0 * alpha * e_dot + r_obs_mag * r_obs_mag);
    let c3 = -2.0 * GMS * beta_coeff * (alpha + e_dot);
    let c0 = -(GMS * beta_coeff).powi(2);

    let roots = solve_r2_polynomial(c6, c3, c0);

    let mut results = Vec::new();
    for r_mag in roots {
        if r_mag < 0.01 {
            continue;
        }

        let rho_scalar = alpha + GMS * beta_coeff / (r_mag * r_mag * r_mag);
        if rho_scalar < 0.0 {
            continue;
        }

        // Heliocentric position at epoch 2
        let r_vec = r_obs + rho2 * rho_scalar;
        let r_actual = r_vec.norm();
        let r_actual_cubed = r_actual * r_actual * r_actual;

        // Velocity via the Laplace determinant for rho_dot.
        //
        // From the equation of motion rearranged as:
        //   rho'' L + 2 rho' L' + rho (L'' + mu L/r^3) = -mu R/r^3 - a_obs
        //
        // Cramer's rule (replacing L' column with the RHS) gives:
        //   2 rho_dot * D = L . (a_star x L'')
        // where a_star = -mu R / r^3 - a_obs  (and D' = D since L.(L' x L) = 0).
        let a_star = r_obs * (-GMS / r_actual_cubed) - a_obs;
        let rho_dot_scalar = rho2.dot(&a_star.cross(&rho_ddot)) / (2.0 * d_det);

        // v = v_obs + rho_dot * L_hat + rho * L_hat_dot
        let v_obs = obs2.vel;
        let v2 = v_obs + rho2 * rho_dot_scalar + rho_dot * rho_scalar;

        let state = State::new(kete_core::desigs::Desig::Empty, obs2.epoch, r_vec, v2, 0);
        results.push(state);
    }

    if results.is_empty() {
        return Err(Error::ValueError(
            "Laplace IOD: no valid roots found".into(),
        ));
    }
    Ok(results)
}

/// Lagrange f and g coefficients (two-body series approximation).
///
/// Given a time offset `tau` (days) and `r_cubed` = |r|^3 at the reference
/// epoch, returns `(f, g)` such that `r(t) ~= f * r_0 + g * v_0`.
fn lagrange_fg(tau: f64, r_cubed: f64) -> (f64, f64) {
    let f = 1.0 - GMS / (2.0 * r_cubed) * tau * tau;
    let g = tau - GMS / (6.0 * r_cubed) * tau * tau * tau;
    (f, g)
}

/// Solve the IOD distance polynomial:
///   x^8 + c6*x^6 + c3*x^3 + c0 = 0
///
/// Returns all real positive roots found by companion-matrix eigenvalue
/// decomposition. This is a sparse polynomial (only terms x^8, x^6, x^3, x^0)
/// so we solve via bisection on a bracketed search after sign analysis.
fn solve_r2_polynomial(c6: f64, c3: f64, c0: f64) -> Vec<f64> {
    // Evaluate p(x) = x^8 + c6*x^6 + c3*x^3 + c0
    let poly = |x: f64| -> f64 {
        let x3 = x * x * x;
        let x6 = x3 * x3;
        let x8 = x6 * x * x;
        x8 + c6 * x6 + c3 * x3 + c0
    };

    // Derivative for Newton refinement
    let dpoly = |x: f64| -> f64 {
        let x2 = x * x;
        let x5 = x2 * x2 * x;
        let x7 = x5 * x * x;
        8.0 * x7 + 6.0 * c6 * x5 + 3.0 * c3 * x2
    };

    // Scan for sign changes in [0.01, 200] AU
    let n_scan = 10000;
    let x_min = 0.01_f64;
    let x_max = 200.0_f64;
    let dx = (x_max - x_min) / f64::from(n_scan);

    let mut roots = Vec::new();
    let mut x_prev = x_min;
    let mut f_prev = poly(x_prev);

    for i in 1..=n_scan {
        let x_cur = x_min + f64::from(i) * dx;
        let f_cur = poly(x_cur);

        if f_prev * f_cur < 0.0 {
            // Sign change -- bisect to find root
            let root = bisect_newton(poly, dpoly, x_prev, x_cur, 60);
            roots.push(root);
        } else if f_cur.abs() < 1e-30 {
            roots.push(x_cur);
        }

        x_prev = x_cur;
        f_prev = f_cur;
    }

    roots
}

/// Bisection followed by Newton polishing.
fn bisect_newton(
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    max_iter: usize,
) -> f64 {
    // Bisect to narrow the bracket
    for _ in 0..40 {
        let m = 0.5 * (a + b);
        if f(a) * f(m) <= 0.0 {
            b = m;
        } else {
            a = m;
        }
        if (b - a) < 1e-12 {
            break;
        }
    }
    // Newton polish from midpoint
    let mut x = 0.5 * (a + b);
    for _ in 0..max_iter {
        let fx = f(x);
        let dfx = df(x);
        if dfx.abs() < 1e-30 {
            break;
        }
        let dx = fx / dfx;
        x -= dx;
        if dx.abs() < 1e-14 * x.abs() {
            break;
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;
    use kete_core::propagation::propagate_two_body;
    use kete_core::time::{TDB, Time};

    /// Helper: build a State from arrays.
    fn make_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial> {
        State::new(Desig::Empty, jd.into(), pos.into(), vel.into(), 0)
    }

    /// Synthesize optical observations from a known orbit.
    ///
    /// Propagates the object to each epoch using two-body, computes the
    /// topocentric RA/Dec, and returns Optical observations. The observer
    /// is placed on a circular Earth-like orbit at 1 AU.
    fn synth_optical(obj: &State<Equatorial>, epochs: &[f64]) -> Vec<Observation> {
        // Earth-like circular orbit at 1 AU
        let r_earth = 1.0;
        let v_earth = (GMS / r_earth).sqrt();
        // Small inclination so LOS vectors are not perfectly coplanar
        let earth_incl = 0.05_f64; // ~3 degrees
        let earth_ref = make_state(
            [r_earth, 0.0, 0.0],
            [0.0, v_earth * earth_incl.cos(), v_earth * earth_incl.sin()],
            epochs[0],
        );

        epochs
            .iter()
            .map(|&jd| {
                let obj_at = propagate_two_body(obj, Time::<TDB>::new(jd))
                    .expect("two-body propagation failed");
                let observer = propagate_two_body(&earth_ref, Time::<TDB>::new(jd))
                    .expect("earth propagation failed");
                let d = obj_at.pos - observer.pos;
                let (ra, dec) = d.to_ra_dec();
                Observation::Optical {
                    observer,
                    ra,
                    dec,
                    sigma_ra: 1e-6,
                    sigma_dec: 1e-6,
                }
            })
            .collect()
    }

    #[test]
    fn test_gauss_circular_orbit() {
        // Object on a roughly circular orbit at ~1.5 AU
        // v_circ = sqrt(GMS / r) for circular orbit
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let obj = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        // Three observations spread over ~30 days
        let epochs = [2460000.5, 2460015.5, 2460030.5];
        let observations = synth_optical(&obj, &epochs);

        let results = gauss_iod(&observations).unwrap();
        assert!(!results.is_empty(), "Should find at least one root");

        // Check the best candidate against the true state at the middle epoch
        let best = &results[0];

        // The IOD state is at the middle epoch, so propagate the true object there
        let obj_mid = propagate_two_body(&obj, Time::<TDB>::new(epochs[1])).unwrap();
        let pos_err_mid = (best.pos - obj_mid.pos).norm();

        // IOD should recover position to within ~10%
        let r_mid = obj_mid.pos.norm();
        assert!(
            pos_err_mid / r_mid < 0.1,
            "Position error {pos_err_mid:.4} too large relative to r={r_mid:.4}"
        );
    }

    #[test]
    fn test_gauss_elliptical_orbit() {
        // Moderately eccentric orbit (e ~ 0.3)
        // a = 2.0, r_peri = a*(1-e) = 1.4, v at peri = sqrt(GMS*(2/r - 1/a))
        let a = 2.0;
        let r_peri = 1.4;
        let v_peri = (GMS * (2.0 / r_peri - 1.0 / a)).sqrt();
        let obj = make_state([r_peri, 0.0, 0.0], [0.0, v_peri, 0.0], 2460000.5);

        let epochs = [2460000.5, 2460020.5, 2460040.5];
        let observations = synth_optical(&obj, &epochs);

        let results = gauss_iod(&observations).unwrap();
        assert!(!results.is_empty());

        // IOD state is at middle epoch
        let obj_mid = propagate_two_body(&obj, Time::<TDB>::new(epochs[1])).unwrap();
        let best = &results[0];
        let pos_err = (best.pos - obj_mid.pos).norm();
        let r_mid = obj_mid.pos.norm();
        assert!(
            pos_err / r_mid < 0.15,
            "Position error {pos_err:.6} too large relative to r={r_mid:.4}"
        );
    }

    #[test]
    fn test_gauss_inclined_orbit() {
        // Inclined circular orbit at 1.5 AU, i ~ 30 deg
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let i = 30.0_f64.to_radians();
        let obj = make_state([r, 0.0, 0.0], [0.0, v * i.cos(), v * i.sin()], 2460000.5);

        let epochs = [2460000.5, 2460015.5, 2460030.5];
        let observations = synth_optical(&obj, &epochs);

        let results = gauss_iod(&observations).unwrap();
        assert!(!results.is_empty());

        let obj_mid = propagate_two_body(&obj, Time::<TDB>::new(epochs[1])).unwrap();
        let best = &results[0];
        let pos_err = (best.pos - obj_mid.pos).norm();
        let r_mid = obj_mid.pos.norm();
        assert!(
            pos_err / r_mid < 0.1,
            "Position error {pos_err:.6} too large for inclined orbit"
        );
    }

    #[test]
    fn test_polynomial_solver_basic() {
        // x^8 = 0 has root x=0 (but we filter x < 0.01)
        let _roots = solve_r2_polynomial(0.0, 0.0, 0.0);
        // Should find root(s) near zero, all filtered
        // No assertion on count -- just verify it does not panic.

        // A polynomial with a known root: set up so x=2 is a root.
        // p(2) = 256 + c6*64 + c3*8 + c0 = 0
        // Pick c6 = -1, c3 = -10: 256 - 64 - 80 + c0 = 0 => c0 = -112
        let roots = solve_r2_polynomial(-1.0, -10.0, -112.0);
        let has_near_2 = roots.iter().any(|&r| (r - 2.0).abs() < 0.01);
        assert!(has_near_2, "Should find root near x=2, got: {roots:?}");
    }

    #[test]
    fn test_laplace_circular_orbit() {
        // Same as Gauss circular test but using Laplace
        let r = 1.5;
        let v = (GMS / r).sqrt();
        let obj = make_state([r, 0.0, 0.0], [0.0, v, 0.0], 2460000.5);

        let epochs = [2460000.5, 2460015.5, 2460030.5];
        let observations = synth_optical(&obj, &epochs);

        let results = laplace_iod(&observations).unwrap();
        assert!(!results.is_empty(), "Laplace should find at least one root");

        let obj_mid = propagate_two_body(&obj, Time::<TDB>::new(epochs[1])).unwrap();
        let best = &results[0];
        let pos_err = (best.pos - obj_mid.pos).norm();
        let r_mid = obj_mid.pos.norm();
        // Laplace can be less accurate; allow 20% error
        assert!(
            pos_err / r_mid < 0.2,
            "Laplace position error {pos_err:.4} too large relative to r={r_mid:.4}"
        );
    }
}
