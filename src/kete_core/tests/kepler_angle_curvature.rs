//! Does a free-Kepler cloud accumulate sigma-point divergence purely from the
//! choice of `true_lon` as the angle coordinate?
//!
//! # Theory
//!
//! Under pure two-body motion five of the six modified equinoctial elements are
//! constants of the motion; only `true_lon` evolves.  The element-to-element map
//! over a time span is therefore the identity on `(p, f, g, h, k)` and some scalar
//! map on `L`.  That scalar map is not linear, and it has two separable sources:
//!
//! - **(A) mean-motion shear.** A spread in `p` is a spread in `a`, hence in
//!   `n ~ a^-3/2`.  Nonlinear in any angle variable, secular in time.
//! - **(B) equation of the center.** At *fixed* orbit (zero spread in
//!   `p, f, g, h, k`) a spread purely in `L` is a phase spread along one orbit.
//!   In mean longitude `dM` is constant forever, so the map is exactly linear.
//!   In true longitude `dL(t)` breathes over the orbit, so it is not.  The
//!   distortion is bounded and periodic: it must return to exactly linear at
//!   integer multiples of the period, since both particles return to their
//!   starting `L`.
//!
//! # Measurement
//!
//! For a probe pair at `+/- delta`, with `F(s)` the final element offset of the
//! displaced particle from the propagated base:
//!
//! ```text
//! odd  = (F(+) - F(-)) / 2     linear response, error O(delta^3)
//! even = (F(+) + F(-)) / 2     leading curvature, O(delta^2)
//! ```
//!
//! No Jacobian and no STM enter, so nothing here can be contaminated by a
//! finite-difference step choice.  `even[5] / |odd[5]|` is the linear prediction
//! error as a fraction of the propagated spread along `L` -- the production
//! sigma-point divergence restricted to that coordinate, directly comparable to
//! `SplitConfig::split_threshold` (default 0.1).
//!
//! # Certification
//!
//! - `dt = 0`: `even = 0`, `odd = delta`.
//! - `dt = P` (case B): `even ~ 0`, `odd ~ delta` -- both particles return.
//! - Case B leaves `(p, f, g, h, k)` untouched to round-off.
//! - `even` scales as `delta^2` under halving.
//!
//! Error budget: the two-body solver and the state <-> element round trip are the
//! floor.  `equinoctial_comet_conversion_roundtrip` pins the round trip at 1e-10;
//! tolerances below are set from that, not tuned to pass.

use kete_core::constants::{GMS, GMS_SQRT};
use kete_core::desigs::Desig;
use kete_core::elements::EquinoctialElements;
use kete_core::frames::Ecliptic;
use kete_core::state::State;
use nalgebra::{Vector3, Vector6};

const EPOCH: f64 = 2451545.0;

/// A planar conic with perihelion `q`, eccentricity `e`, at true anomaly `nu`.
///
/// Longitude of perihelion is zero, so `f = e`, `g = 0`, and `L_0 = nu`.
fn base_orbit(q: f64, e: f64, nu: f64) -> EquinoctialElements {
    EquinoctialElements {
        desig: Desig::Name("probe".into()),
        epoch: EPOCH.into(),
        semi_latus: q * (1.0 + e),
        ecc_f: e,
        ecc_g: 0.0,
        pole_h: 0.0,
        pole_k: 0.0,
        true_lon: nu,
        center_id: 10,
        gm_sqrt: GMS_SQRT,
    }
}

/// Orbital period in days for a bound conic.
fn period(q: f64, e: f64) -> f64 {
    let a = q / (1.0 - e);
    std::f64::consts::TAU * a.powf(1.5) / GMS.sqrt()
}

/// Propagate one element set by pure two-body motion.
fn propagate(elem: &EquinoctialElements, dt: f64) -> EquinoctialElements {
    let start = elem.try_to_state().expect("elements must decode");
    let (pos, vel) = kete_core::kepler::analytic_2_body(
        dt.into(),
        &Vector3::from(start.pos),
        &Vector3::from(start.vel),
        None,
    )
    .expect("two body must converge");
    let final_state =
        State::<Ecliptic>::new(elem.desig.clone(), EPOCH + dt, pos, vel, elem.center_id);
    EquinoctialElements::from_state(&final_state).expect("state must encode")
}

/// Odd (linear) and even (curvature) parts of the propagated probe-pair offsets.
fn probe_pair(
    base: &EquinoctialElements,
    delta: &Vector6<f64>,
    dt: f64,
) -> (Vector6<f64>, Vector6<f64>) {
    let base_f = propagate(base, dt);
    let plus = base_f.offset_to(&propagate(&base.displaced_by(delta), dt));
    let minus = base_f.offset_to(&propagate(&base.displaced_by(&(-delta)), dt));
    ((plus - minus) * 0.5, (plus + minus) * 0.5)
}

fn along_track(sigma: f64) -> Vector6<f64> {
    Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, sigma)
}

/// Harness certification.  Everything below rests on these.
#[test]
fn harness_reproduces_the_closed_form_cases() {
    let (q, e) = (1.0, 0.9);
    let base = base_orbit(q, e, 0.0);
    let delta = along_track(1e-2);

    // dt = 0: the map is the identity.
    let (odd, even) = probe_pair(&base, &delta, 0.0);
    println!("dt=0     odd[5] {:.16e}  even[5] {:.3e}", odd[5], even[5]);
    assert!((odd[5] - delta[5]).abs() < 1e-14, "odd {}", odd[5]);
    assert!(even.norm() < 1e-14, "even {}", even.norm());

    // dt = one period: both particles return to their starting longitude, so the
    // equation-of-center distortion has closed.  This is the sharpest check that
    // the harness measures the map and not an artifact.
    let p = period(q, e);
    let (odd, even) = probe_pair(&base, &delta, p);
    println!(
        "dt=P     odd[5] {:.16e}  even[5] {:.3e}   (P = {p:.3} d)",
        odd[5], even[5]
    );
    assert!(
        (odd[5] - delta[5]).abs() < 1e-9,
        "one period must return the spread exactly: {}",
        odd[5]
    );
    assert!(even.norm() < 1e-9, "one period curvature {}", even.norm());

    // Case B moves no constant of the motion.
    let (odd, even) = probe_pair(&base, &delta, 0.31 * p);
    let constants = (0..5)
        .map(|i| odd[i].abs().max(even[i].abs()))
        .fold(0.0_f64, f64::max);
    println!("dt=0.31P constants of motion worst |offset| {constants:.3e}");
    assert!(constants < 1e-10, "constants of motion moved {constants:e}");
}

/// The curvature is second order in the probe scale, which is what makes it a
/// genuine nonlinearity of the map rather than a bug or a round-off floor.
#[test]
fn curvature_is_second_order_in_the_probe_scale() {
    let (q, e) = (1.0, 0.9);
    let base = base_orbit(q, e, 0.0);
    let dt = 0.31 * period(q, e);

    let mut prev: Option<f64> = None;
    for k in 0..4 {
        let sigma = 1e-2 / 2.0_f64.powi(k);
        let (_, even) = probe_pair(&base, &along_track(sigma), dt);
        let mag = even[5].abs();
        if let Some(p) = prev {
            let ratio = p / mag;
            println!("sigma {sigma:.3e}  even[5] {mag:.6e}  ratio vs 2x {ratio:.3}");
            assert!(
                (3.0..5.0).contains(&ratio),
                "curvature must fall 4x per halving, got {ratio}"
            );
        } else {
            println!("sigma {sigma:.3e}  even[5] {mag:.6e}");
        }
        prev = Some(mag);
    }
}

/// The headline measurement: linear-prediction error as a fraction of the
/// propagated spread, for a cloud under *no perturbations at all*.
#[test]
fn free_kepler_along_track_spread_divergence() {
    let q = 1.0;
    println!(
        "\ncase B: pure phase spread, fixed orbit -- coordinate curvature only\n\
         {:>5} {:>10} {:>8} {:>12} {:>12} {:>10}",
        "e", "sigma_L", "dt/P", "odd[5]", "even[5]", "sigma"
    );
    for &e in &[0.1, 0.5, 0.9] {
        let p = period(q, e);
        let base = base_orbit(q, e, 0.0);
        for &sigma in &[1e-3_f64, 1e-2, 1e-1] {
            for &frac in &[0.05, 0.25, 0.5, 0.75] {
                let (odd, even) = probe_pair(&base, &along_track(sigma), frac * p);
                println!(
                    "{e:>5.1} {sigma:>10.0e} {frac:>8.2} {:>12.3e} {:>12.3e} {:>10.3}",
                    odd[5],
                    even[5],
                    even[5].abs() / odd[5].abs()
                );
            }
        }
    }

    println!(
        "\ncase A: spread in semi-latus only -- mean-motion shear, any angle variable\n\
         {:>5} {:>10} {:>8} {:>12} {:>12} {:>10}",
        "e", "sigma_p", "dt/P", "odd[5]", "even[5]", "sigma"
    );
    for &e in &[0.1, 0.9] {
        let p = period(q, e);
        let base = base_orbit(q, e, 0.0);
        for &sigma in &[1e-6_f64, 1e-4] {
            for &frac in &[0.05, 0.25, 0.5, 0.75] {
                let delta = Vector6::new(sigma, 0.0, 0.0, 0.0, 0.0, 0.0);
                let (odd, even) = probe_pair(&base, &delta, frac * p);
                println!(
                    "{e:>5.1} {sigma:>10.0e} {frac:>8.2} {:>12.3e} {:>12.3e} {:>10.3}",
                    odd[5],
                    even[5],
                    even[5].abs() / odd[5].abs()
                );
            }
        }
    }
}

/// Fallible propagation -- the universal Kepler solver does not converge for very
/// wide phase spreads on near-parabolic orbits, and those grid points are skipped
/// rather than silently reported as zero.
fn try_propagate(elem: &EquinoctialElements, dt: f64) -> Option<EquinoctialElements> {
    let start = elem.try_to_state().ok()?;
    let (pos, vel) = kete_core::kepler::analytic_2_body(
        dt.into(),
        &Vector3::from(start.pos),
        &Vector3::from(start.vel),
        None,
    )
    .ok()?;
    let final_state =
        State::<Ecliptic>::new(elem.desig.clone(), EPOCH + dt, pos, vel, elem.center_id);
    EquinoctialElements::from_state(&final_state).ok()
}

/// Mean anomaly of an element set, via the cometary conversion.
fn mean_anomaly(elem: &EquinoctialElements) -> Option<f64> {
    kete_core::elements::CometElements::try_from(elem)
        .ok()
        .map(|c| c.mean_anomaly())
}

fn wrap(x: f64) -> f64 {
    x - std::f64::consts::TAU * (x / std::f64::consts::TAU).round()
}

/// Elements at a given mean anomaly on the planar conic of [`base_orbit`].
fn at_mean_anomaly(q: f64, e: f64, m: f64) -> Option<EquinoctialElements> {
    let nu = kete_core::kepler::compute_true_anomaly(e, m, q).ok()?;
    Some(base_orbit(q, e, nu))
}

/// The decisive comparison: the same physical cloud, probed in the coordinate it is
/// actually stored in.
///
/// A probe pair must be symmetric in the coordinate being scored, or its even part
/// is nonzero before anything is propagated.  So the two columns use different pairs
/// by construction:
///
/// - `L`: pair at `L_0 +/- sigma`, scored in true longitude.  This is what the
///   production sigma-point probes do today, since an along-track-dominated
///   covariance has its dominant eigenvector essentially along `L`.
/// - `M`: pair at `M_0 +/- sigma`, scored in mean anomaly.  This is what they would
///   do if the angle coordinate were the mean longitude.
///
/// Both pairs sit on the same orbit as the base, so under two-body motion every
/// particle conserves its mean-anomaly separation exactly.  The `M` column must
/// therefore be zero to solver round-off at every grid point, and any excess in the
/// `L` column is attributable to the angle coordinate rather than to the dynamics.
#[test]
fn true_longitude_vs_mean_anomaly_probes() {
    let peri_q = 1.0;
    let fracs: Vec<f64> = (1..40).map(|i| f64::from(i) * 0.025).collect();

    println!(
        "\npeak relative linear-prediction error over dt in (0, P), free Kepler\n\
         {:>5} {:>6} {:>7} | {:>10} {:>10} {:>9} | {:>9} | {:>7}",
        "e", "M_0", "sigma", "odd_L", "even_L", "err_L", "err_M", "at dt/P"
    );
    let mut worst_m = 0.0_f64;
    for &ecc in &[0.1, 0.5, 0.9, 0.99] {
        let per = period(peri_q, ecc);
        for &(label, m0) in &[
            ("peri", 0.0),
            ("quad", std::f64::consts::FRAC_PI_2),
            ("apo", std::f64::consts::PI),
        ] {
            let Some(base) = at_mean_anomaly(peri_q, ecc, m0) else {
                continue;
            };
            for &sigma in &[0.1_f64, 0.3, 1.0] {
                // Pair symmetric in L, and pair symmetric in M.
                let l_pair = [
                    Some(base.displaced_by(&along_track(sigma))),
                    Some(base.displaced_by(&along_track(-sigma))),
                ];
                let m_pair = [
                    at_mean_anomaly(peri_q, ecc, m0 + sigma),
                    at_mean_anomaly(peri_q, ecc, m0 - sigma),
                ];

                let (mut peak, mut at, mut odd_at, mut even_at, mut m_at) =
                    (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
                let mut any = false;
                for &frac in &fracs {
                    let dt = frac * per;
                    let Some(bf) = try_propagate(&base, dt) else {
                        continue;
                    };

                    // True-longitude column.
                    let (Some(l_plus), Some(l_minus)) = (
                        l_pair[0].as_ref().and_then(|x| try_propagate(x, dt)),
                        l_pair[1].as_ref().and_then(|x| try_propagate(x, dt)),
                    ) else {
                        continue;
                    };
                    let (dp, dm) = (bf.offset_to(&l_plus)[5], bf.offset_to(&l_minus)[5]);
                    let (odd_l, even_l) = ((dp - dm) * 0.5, f64::midpoint(dp, dm));

                    // Mean-anomaly column.
                    let (Some(m_plus), Some(m_minus)) = (
                        m_pair[0].as_ref().and_then(|x| try_propagate(x, dt)),
                        m_pair[1].as_ref().and_then(|x| try_propagate(x, dt)),
                    ) else {
                        continue;
                    };
                    let (Some(base_m), Some(plus_m), Some(minus_m)) = (
                        mean_anomaly(&bf),
                        mean_anomaly(&m_plus),
                        mean_anomaly(&m_minus),
                    ) else {
                        continue;
                    };
                    let (ep, em) = (wrap(plus_m - base_m), wrap(minus_m - base_m));
                    let (odd_m, even_m) = ((ep - em) * 0.5, f64::midpoint(ep, em));
                    let err_m = even_m.abs() / odd_m.abs().max(1e-300);
                    worst_m = worst_m.max(err_m);

                    any = true;
                    let err_l = even_l.abs() / odd_l.abs().max(1e-300);
                    if err_l > peak {
                        (peak, at, odd_at, even_at, m_at) = (err_l, frac, odd_l, even_l, err_m);
                    }
                }
                if any {
                    println!(
                        "{ecc:>5.2} {label:>6} {sigma:>7.1} | {odd_at:>10.3e} {even_at:>10.3e} \
                         {peak:>9.3} | {m_at:>9.2e} | {at:>7.3}"
                    );
                } else {
                    println!("{ecc:>5.2} {label:>6} {sigma:>7.1} |   solver did not converge");
                }
            }
        }
    }

    println!("\nworst mean-anomaly relative curvature anywhere on the grid: {worst_m:.3e}");
    assert!(
        worst_m < 1e-6,
        "mean-anomaly separation is a two-body invariant; got relative curvature {worst_m:e}"
    );
}
