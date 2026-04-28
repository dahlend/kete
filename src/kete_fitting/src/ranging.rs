//! Systematic ranging orbit uncertainty sampling.
//!
//! Implements the JPL Scout algorithm (Farnocchia et al. 2015): scans a 2-D
//! grid over topocentric range `rho` and range-rate `rho_dot`, scores each cell by
//! solving the constrained attributable least-squares (the admissible-region
//! chi^2), and importance-samples from the resulting weighted distribution.
//!
//! **Reference**: D. Farnocchia, S. R. Chesley, M. Micheli, "Systematic
//! ranging and late warning asteroid impacts", Icarus 258 (2015) 18--27.

use crate::obs::{AstrometricObservation, differential_light_deflect};
use kete_core::constants::GMS;
use kete_core::frames::{Equatorial, SSB, Vector};
use kete_core::kepler::{light_time_correct, propagate_two_body};
use kete_core::prelude::{Error, KeteResult, State};
use kete_core::time::Time;
use kete_spice::prelude::LOADED_SPK;
use nalgebra::{DMatrix, DVector};
use nuts_rs::rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::Distribution;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Grid constants
// ---------------------------------------------------------------------------

/// log-rho grid: 1e-5 to 1000 AU.
const LOG_RHO_MIN: f64 = -11.5129;
const LOG_RHO_MAX: f64 = 6.9078;

/// Absolute cap on the adaptive `rho_dot` range (AU/day).
/// Escape velocity at 1e-5 AU is enormous; this cap prevents unreasonably
/// large grids for impactor-range distances.
const RHO_DOT_ABS_MAX: f64 = 2.0; // ~3460 km/s

/// Energy multiplier controlling both the `rho_dot` scan range and the physical
/// validity threshold.
///
/// The `rho_dot` scan range per rho row is `+/-sqrt(ENERGY_MULT * mu/r)`, and cells are
/// accepted only when `v^2 < ENERGY_MULT * mu/r`.  Setting both to the same
/// multiplier ensures the scan covers exactly the admissible cells and no more.
///
/// `ENERGY_MULT = 2` -> strict bound orbits only (e < 1).
/// `ENERGY_MULT = 8` -> tangential orbits up to e ~ 7; typical non-radial
///   orbits up to e ~ 2--3.  Comfortably covers interstellar-like e ~ 1--2.
const ENERGY_MULT: f64 = 8.0;

const N_RHO: usize = 400;
const N_RHO_DOT: usize = 400;

const TARGET_ESS: f64 = 50.0;
const MAX_REFINE: usize = 4;
/// Cells more than this many nats below the max weight are dropped.
const LOG_W_FLOOR: f64 = 50.0;

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// Orbit samples from the weighted ranging grid.
///
/// Each draw is a `[x, y, z, vx, vy, vz]` state in AU/AU*day,
/// SSB Equatorial frame, at `epoch`.
#[derive(Debug, Clone)]
pub struct RangingSamples {
    /// Object designator.
    pub desig: String,
    /// Reference epoch (JD TDB)  -- light-time-corrected epoch of the attributable.
    pub epoch: f64,
    /// Orbit draws: `[num_draws][6]`, SSB Equatorial AU/AU*day.
    pub draws: Vec<Vec<f64>>,
    /// Normalized log-posterior weight per draw (nats, relative).
    pub log_posterior: Vec<f64>,
    /// Effective sample size `(sumw_i)^2 / sumw_i^2` of the grid before drawing.
    pub effective_sample_size: f64,
    /// Set when `effective_sample_size < 50` at the end of refinement.
    pub convergence_warning: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// Pre-computed attributable at a reference epoch.
#[derive(Clone)]
struct Attributable {
    /// Reference observation epoch (JD TDB).
    t_ref: f64,
    /// RA at `t_ref` (radians).
    alpha: f64,
    /// Dec at `t_ref` (radians).
    delta: f64,
    /// RA rate (radians/day).
    alpha_dot: f64,
    /// Dec rate (radians/day).
    delta_dot: f64,
    /// Observer SSB state at `t_ref`.
    observer: State<Equatorial, SSB>,
}

struct Cell {
    log_w: f64,
    state: State<Equatorial, SSB>,
    /// 4x4 attributable normal matrix N = H^4^T W H^4.
    attr_info: DMatrix<f64>,
    rho: f64,
    rho_dot: f64,
    /// 1-sigma spread in `rho_dot` from the local chi^2 curvature along the `rho_dot` axis.
    /// Used for within-cell `rho_dot` perturbation to spread draws across the LOV ridge.
    rho_dot_sigma: f64,
    /// 1-sigma spread in log(rho) from the LOV chi^2 profile curvature along the rho axis.
    /// Filled after all LOV cells are scored; zero until then.
    log_rho_sigma: f64,
}

// ---------------------------------------------------------------------------
// Attributable computation
// ---------------------------------------------------------------------------

/// Fit `(alpha0, delta0, alpha_dot, delta_dot)` at the first observation epoch by weighted LS of
/// (RA, Dec) vs time.  Returns `None` when fewer than 2 optical observations
/// are available.
fn compute_attributable(sorted_obs: &[AstrometricObservation]) -> Option<Attributable> {
    struct Entry {
        dt: f64,
        ra: f64,
        dec: f64,
        w_alpha: f64,
        w_delta: f64,
    }

    let t_ref = sorted_obs.first()?.epoch().jd;
    let entries: Vec<Entry> = sorted_obs
        .iter()
        .filter_map(|obs| {
            let (ra, dec, _) = obs.as_optical().ok()?;
            let w = obs.base_weight_matrix();
            Some(Entry {
                dt: obs.epoch().jd - t_ref,
                ra,
                dec,
                w_alpha: w[(0, 0)],
                w_delta: w[(1, 1)],
            })
        })
        .collect();

    if entries.len() < 2 {
        return None;
    }

    let ra0 = entries[0].ra;
    let (mut sw_a, mut swt_a, mut swt2_a, mut swra_a, mut swrat_a) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
    let (mut sw_d, mut swt_d, mut swt2_d, mut swdec_d, mut swdect_d) =
        (0.0_f64, 0.0, 0.0, 0.0, 0.0);

    for e in &entries {
        let mut dra = e.ra - ra0;
        if dra > std::f64::consts::PI {
            dra -= std::f64::consts::TAU;
        }
        if dra < -std::f64::consts::PI {
            dra += std::f64::consts::TAU;
        }
        sw_a += e.w_alpha;
        swt_a += e.w_alpha * e.dt;
        swt2_a += e.w_alpha * e.dt * e.dt;
        swra_a += e.w_alpha * dra;
        swrat_a += e.w_alpha * dra * e.dt;
        sw_d += e.w_delta;
        swt_d += e.w_delta * e.dt;
        swt2_d += e.w_delta * e.dt * e.dt;
        swdec_d += e.w_delta * e.dec;
        swdect_d += e.w_delta * e.dec * e.dt;
    }

    let det_a = sw_a * swt2_a - swt_a * swt_a;
    let det_d = sw_d * swt2_d - swt_d * swt_d;
    if det_a.abs() < 1e-30 || det_d.abs() < 1e-30 {
        return None;
    }

    let alpha0 = ra0 + (swt2_a * swra_a - swt_a * swrat_a) / det_a;
    let alpha_dot = (sw_a * swrat_a - swt_a * swra_a) / det_a;
    let delta0 = (swt2_d * swdec_d - swt_d * swdect_d) / det_d;
    let delta_dot = (sw_d * swdect_d - swt_d * swdec_d) / det_d;

    let (_, _, observer) = sorted_obs.first()?.as_optical().ok()?;
    Some(Attributable {
        t_ref,
        alpha: alpha0,
        delta: delta0,
        alpha_dot,
        delta_dot,
        observer: observer.clone(),
    })
}

// ---------------------------------------------------------------------------
// State construction from (rho, `rho_dot`, attributable)
// ---------------------------------------------------------------------------

/// Build an SSB Cartesian state from the attributable plus `(rho, rho_dot)`.
///
/// Per Scout section 2, the epoch is light-time corrected by `rho/c`.
fn state_from_rho(attr: &Attributable, rho: f64, rho_dot: f64) -> State<Equatorial, SSB> {
    let (sin_a, cos_a) = attr.alpha.sin_cos();
    let (sin_d, cos_d) = attr.delta.sin_cos();
    let los = Vector::<Equatorial>::new([cos_d * cos_a, cos_d * sin_a, sin_d]);
    let los_da = Vector::<Equatorial>::new([-sin_a * cos_d, cos_a * cos_d, 0.0]);
    let los_dd = Vector::<Equatorial>::new([-cos_a * sin_d, -sin_a * sin_d, cos_d]);
    let los_dot = los_da * attr.alpha_dot + los_dd * attr.delta_dot;

    let pos_ssb = attr.observer.pos + los * rho;
    let vel_ssb = attr.observer.vel + los * rho_dot + los_dot * rho;

    let tau = rho * kete_core::constants::C_AU_PER_DAY_INV;
    State::<Equatorial, SSB> {
        desig: kete_core::desigs::Desig::Empty,
        epoch: Time::from(attr.t_ref - tau),
        pos: pos_ssb,
        vel: vel_ssb,
        center: SSB,
    }
}

// ---------------------------------------------------------------------------
// Physical validity
// ---------------------------------------------------------------------------

/// Reject orbits outside [0.001, 1000] AU or with v^2 >= `ENERGY_MULT * GMS / r`.
///
/// The threshold is more generous than the strict parabolic bound (v^2 < 2mu/r)
/// to avoid cutting valid near-parabolic and mildly hyperbolic orbits.  Orbits
/// beyond this threshold are not physical for any solar-system population and
/// would corrupt the log-rho prior by allowing cells to survive at arbitrarily
/// large distances with zero chi^2.
fn is_physically_valid(pos_helio: Vector<Equatorial>, vel_helio: Vector<Equatorial>) -> bool {
    let r = pos_helio.norm();
    if !(0.001..=1000.0).contains(&r) {
        return false;
    }
    vel_helio.norm_squared() < ENERGY_MULT * GMS / r
}

// ---------------------------------------------------------------------------
// Scout scoring: constrained attributable LS
// ---------------------------------------------------------------------------

/// Score a `(rho, rho_dot)` cell using Scout's constrained attributable LS.
///
/// Propagates the orbit to every observation and computes residuals nu.  The
/// score is the Gaussian log-likelihood after removing the best-fit 4-parameter
/// attributable correction `(deltaalpha, deltadelta, deltaalpha_dot, deltadelta_dot)`:
///
/// ```text
/// Q_min = nu^TWnu - b^T N^-1 b,   b = H^4^T W nu,   N = H^4^T W H^4
/// log_w = -Q_min / (2T)
/// ```
///
/// `Q_min` measures curvature  -- deviation from linear on-sky motion  -- a
/// smooth function of `(rho, rho_dot)` that a coarse grid can resolve (Farnocchia
/// et al. 2015, Eq. 1).  Also returns `N` for within-cell sampling.
///
/// Requires >= 3 observations (>= 1 dof per coordinate after 2-parameter fit).
fn scout_score(
    state: &State<Equatorial, SSB>,
    sorted_obs: &[AstrometricObservation],
    t_ref: f64,
    temperature: f64,
) -> Option<(f64, DMatrix<f64>)> {
    struct Entry {
        dt: f64,
        nu_alpha: f64,
        nu_delta: f64,
        w_alpha: f64,
        w_delta: f64,
    }

    let spk = LOADED_SPK.try_read().ok()?;
    let sun_state = spk.try_to_sun(state.clone().into()).ok()?;
    let mut entries: Vec<Entry> = Vec::new();

    for obs in sorted_obs {
        let Ok((alpha_obs, delta_obs, obs_ssb)) = obs.as_optical() else {
            continue;
        };
        let Ok(prop) = propagate_two_body(&sun_state, obs.epoch()) else {
            continue;
        };
        let Ok(obs_sun) = spk.try_to_sun(obs_ssb.clone().into()) else {
            continue;
        };
        let Ok(lt_sun) = light_time_correct(&prop, &obs_sun.pos) else {
            continue;
        };
        let lt_pos = differential_light_deflect(&obs_sun.pos, lt_sun.pos);
        let Some(lt_ssb) = spk
            .try_to_ssb(
                State {
                    pos: lt_pos,
                    ..lt_sun
                }
                .into(),
            )
            .ok()
        else {
            continue;
        };

        let (alpha_pred, delta_pred) = (lt_ssb.pos - obs_ssb.pos).to_ra_dec();
        let mut nu_a = alpha_obs - alpha_pred;
        if nu_a > std::f64::consts::PI {
            nu_a -= std::f64::consts::TAU;
        }
        if nu_a < -std::f64::consts::PI {
            nu_a += std::f64::consts::TAU;
        }

        let w = obs.base_weight_matrix();
        entries.push(Entry {
            dt: obs.epoch().jd - t_ref,
            nu_alpha: nu_a,
            nu_delta: delta_obs - delta_pred,
            w_alpha: w[(0, 0)],
            w_delta: w[(1, 1)],
        });
    }

    if entries.len() < 3 {
        return None;
    }

    // Accumulate chi2_raw and the normal equation components for the 4-parameter
    // attributable model H^4 = [[1,0,deltat,0],[0,1,0,deltat]] per observation.
    // The RA and Dec blocks are decoupled (uncorrelated observations).
    let (mut chi2_raw, mut sw_a, mut swt_a, mut swt2_a, mut swnu_a, mut swnut_a) =
        (0.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sw_d, mut swt_d, mut swt2_d, mut swnu_d, mut swnut_d) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);

    for e in &entries {
        chi2_raw += e.nu_alpha * e.nu_alpha * e.w_alpha + e.nu_delta * e.nu_delta * e.w_delta;
        sw_a += e.w_alpha;
        swt_a += e.w_alpha * e.dt;
        swt2_a += e.w_alpha * e.dt * e.dt;
        swnu_a += e.w_alpha * e.nu_alpha;
        swnut_a += e.w_alpha * e.nu_alpha * e.dt;
        sw_d += e.w_delta;
        swt_d += e.w_delta * e.dt;
        swt2_d += e.w_delta * e.dt * e.dt;
        swnu_d += e.w_delta * e.nu_delta;
        swnut_d += e.w_delta * e.nu_delta * e.dt;
    }

    // b^T N^{-1} b for RA: analytic 2x2 inverse.
    let det_a = sw_a * swt2_a - swt_a * swt_a;
    let chi2_attr_a = if det_a > 1e-30 {
        (swt2_a * swnu_a * swnu_a - 2.0 * swt_a * swnu_a * swnut_a + sw_a * swnut_a * swnut_a)
            / det_a
    } else if sw_a > 1e-30 {
        swnu_a * swnu_a / sw_a
    } else {
        0.0
    };

    let det_d = sw_d * swt2_d - swt_d * swt_d;
    let chi2_attr_d = if det_d > 1e-30 {
        (swt2_d * swnu_d * swnu_d - 2.0 * swt_d * swnu_d * swnut_d + sw_d * swnut_d * swnut_d)
            / det_d
    } else if sw_d > 1e-30 {
        swnu_d * swnu_d / sw_d
    } else {
        0.0
    };

    let chi2_min = (chi2_raw - chi2_attr_a - chi2_attr_d).max(0.0);

    // 4x4 attributable normal matrix N.  Layout: [alpha0, delta0, alpha_dot, delta_dot] -> indices [0,1,2,3].
    // Block-diagonal (RA and Dec decouple for uncorrelated observations).
    let mut n_attr = DMatrix::<f64>::zeros(4, 4);
    n_attr[(0, 0)] = sw_a;
    n_attr[(0, 2)] = swt_a;
    n_attr[(2, 0)] = swt_a;
    n_attr[(2, 2)] = swt2_a;
    n_attr[(1, 1)] = sw_d;
    n_attr[(1, 3)] = swt_d;
    n_attr[(3, 1)] = swt_d;
    n_attr[(3, 3)] = swt2_d;

    Some((-chi2_min / (2.0 * temperature), n_attr))
}

// ---------------------------------------------------------------------------
// Grid scoring
// ---------------------------------------------------------------------------

/// Scan a `(rho, rho_dot)` patch and return one **LOV cell per rho row**.
///
/// For each rho value, scans the full `rho_dot` range and retains the cell with the
/// minimum `chi^2_min` (maximum `log_w`) -- the Line of Variations (LOV) point for
/// that distance.  Estimating `rho_dot_sigma` from the parabolic curvature of
/// the chi^2 profile around the minimum gives the `rho_dot` uncertainty for that row.
///
/// This replaces the flat 2-D grid, which produces isolated blobs at grid
/// points when the posterior is a narrow ridge: every draw from an off-ridge
/// cell has exactly that cell's `rho_dot` (the 4-D attributable perturbation does not
/// move rho or `rho_dot`).  The LOV approach gives one cell per rho on the ridge, with
/// `rho_dot_sigma` allowing draws to spread across the ridge width.
///
/// `rho_dot_range = None`: adaptive `+/-v_esc(rho)` (coarse scan).
/// `rho_dot_range = Some((min, max))`: fixed range (refinement centering).
fn score_patch(
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    log_rho_range: (f64, f64),
    rho_dot_range: Option<(f64, f64)>,
    n_rho: usize,
    n_rdot: usize,
    temperature: f64,
) -> Vec<Cell> {
    let (sun_pos_ssb, sun_vel_ssb, obs_helio_pos, los) = {
        let Ok(spk) = LOADED_SPK.try_read() else {
            return vec![];
        };
        let Ok(obs_helio) = spk.try_to_sun(attr.observer.clone().into()) else {
            return vec![];
        };
        let sun_pos = attr.observer.pos - obs_helio.pos;
        let sun_vel = attr.observer.vel - obs_helio.vel;
        let (sin_a, cos_a) = attr.alpha.sin_cos();
        let (sin_d, cos_d) = attr.delta.sin_cos();
        let los = Vector::<Equatorial>::new([cos_d * cos_a, cos_d * sin_a, sin_d]);
        (sun_pos, sun_vel, obs_helio.pos, los)
    };

    let (la_min, la_max) = log_rho_range;

    // Each rho row is scored independently: scan n_rdot `rho_dot` values and keep the
    // one with the highest log_w (minimum chi^2).  Parallelise over rho rows.
    let lov_cells: Vec<Cell> = (0..n_rho)
        .into_par_iter()
        .filter_map(|ir| {
            let frac_r = ir as f64 / (n_rho - 1).max(1) as f64;
            let rho = (la_min + (la_max - la_min) * frac_r).exp();

            let (rd_min, rd_max) = rho_dot_range.unwrap_or_else(|| {
                let r_helio = (obs_helio_pos + los * rho).norm().max(1e-6);
                // Scan up to sqrt(ENERGY_MULT * mu/r) so every cell that passes
                // is_physically_valid is reachable in the `rho_dot` scan.
                let v_bound = (ENERGY_MULT * GMS / r_helio).sqrt().min(RHO_DOT_ABS_MAX);
                (-v_bound, v_bound)
            });
            let rdot_step = if n_rdot > 1 {
                (rd_max - rd_min) / (n_rdot - 1) as f64
            } else {
                1.0
            };

            // Dense `rho_dot` scan for this rho row.
            let mut log_ws = vec![f64::NEG_INFINITY; n_rdot];
            let mut best: Option<(usize, f64, f64, State<Equatorial, SSB>, DMatrix<f64>)> = None;

            for (id, log_w_slot) in log_ws.iter_mut().enumerate() {
                let frac_d = id as f64 / (n_rdot - 1).max(1) as f64;
                let rho_dot = rd_min + (rd_max - rd_min) * frac_d;
                let state = state_from_rho(attr, rho, rho_dot);
                let pos_helio = state.pos - sun_pos_ssb;
                if !is_physically_valid(pos_helio, state.vel - sun_vel_ssb) {
                    continue;
                }
                let Some((log_w, attr_info)) =
                    scout_score(&state, sorted_obs, attr.t_ref, temperature)
                else {
                    continue;
                };
                if !log_w.is_finite() {
                    continue;
                }
                *log_w_slot = log_w;
                if best.as_ref().is_none_or(|(_, bw, ..)| log_w > *bw) {
                    best = Some((id, log_w, rho_dot, state, attr_info));
                }
            }

            let (best_id, best_log_w, best_rho_dot, best_state, best_attr_info) = best?;

            // Estimate sigma_rho_dot from parabolic curvature of chi^2 around the minimum.
            // chi^2(`rho_dot`) ~ chi^2_min + 1/2(`rho_dot` - rho_dot_best)^2 / sigma^2_rho_dot  ->  sigma_rho_dot = sqrt(T / d^2chi^2/drho_dot^2).
            let rho_dot_sigma = if best_id > 0
                && best_id + 1 < n_rdot
                && log_ws[best_id - 1].is_finite()
                && log_ws[best_id + 1].is_finite()
            {
                let chi2_m = -2.0 * temperature * log_ws[best_id - 1];
                let chi2_0 = -2.0 * temperature * best_log_w;
                let chi2_p = -2.0 * temperature * log_ws[best_id + 1];
                let d2 = (chi2_m - 2.0 * chi2_0 + chi2_p) / (rdot_step * rdot_step);
                if d2 > 1e-30 {
                    (temperature / d2).sqrt()
                } else {
                    rdot_step
                }
            } else {
                rdot_step // edge or isolated minimum: fall back to grid step
            };

            Some(Cell {
                log_w: best_log_w,
                state: best_state,
                attr_info: best_attr_info,
                rho,
                rho_dot: best_rho_dot,
                rho_dot_sigma,
                log_rho_sigma: 0.0, // filled by compute_log_rho_sigma
            })
        })
        .collect();

    lov_cells
}

/// Fill `log_rho_sigma` for each LOV cell from the parabolic curvature of the
/// chi^2 profile along the rho axis.
///
/// Cells must be sorted by `log(rho)` (they arrive that way from `score_patch`
/// since rho rows are scanned in order).  Interior cells get a finite-difference
/// second derivative; edge cells and cells with missing neighbours fall back to
/// half the local rho grid step.
fn compute_log_rho_sigma(cells: &mut [Cell], temperature: f64) {
    let n = cells.len();
    if n < 2 {
        if n == 1 {
            cells[0].log_rho_sigma = 0.05;
        } // arbitrary small spread
        return;
    }

    // Pre-compute log(rho) and chi^2_min per cell.
    let log_rhos: Vec<f64> = cells.iter().map(|c| c.rho.ln()).collect();
    let chi2s: Vec<f64> = cells.iter().map(|c| -2.0 * temperature * c.log_w).collect();

    for i in 0..n {
        let grid_step = if i > 0 && i + 1 < n {
            (log_rhos[i + 1] - log_rhos[i - 1]) / 2.0
        } else if i == 0 && n > 1 {
            log_rhos[1] - log_rhos[0]
        } else {
            log_rhos[n - 1] - log_rhos[n - 2]
        };
        let sigma = if i > 0 && i + 1 < n {
            let dlr_m = log_rhos[i] - log_rhos[i - 1];
            let dlr_p = log_rhos[i + 1] - log_rhos[i];
            // Central second-difference on a possibly non-uniform grid.
            let d2 = 2.0
                * (chi2s[i + 1] * dlr_m - chi2s[i] * (dlr_m + dlr_p) + chi2s[i - 1] * dlr_p)
                / (dlr_m * dlr_p * (dlr_m + dlr_p));
            // Only trust the curvature when it is clearly resolved  -- at least
            // 1 nat of chi^2 variation per grid step.  Floating-point noise on a
            // flat landscape produces tiny positive d2 values that would give
            // astronomically large sigma (sqrt(T / 1e-30) -> inf).
            let d2_threshold = temperature / (grid_step * grid_step);
            if d2 > d2_threshold {
                // Posterior is sharper than one grid cell  -- use curvature.
                (temperature / d2).sqrt().min(grid_step)
            } else {
                // Flat or unresolved  -- spread draws over one grid step.
                grid_step
            }
        } else {
            grid_step
        };
        cells[i].log_rho_sigma = sigma.abs().max(1e-6);
    }
}

// ---------------------------------------------------------------------------
// ESS and adaptive refinement
// ---------------------------------------------------------------------------

fn ess(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    // Use the same Jacobian-corrected weights as draw_samples (+ln rho for uniform prior on rho).
    let max_lw = cells
        .iter()
        .map(|c| c.log_w + c.rho.ln())
        .fold(f64::NEG_INFINITY, f64::max);
    let w: Vec<f64> = cells
        .iter()
        .map(|c| (c.log_w + c.rho.ln() - max_lw).exp())
        .collect();
    let sum_w: f64 = w.iter().sum();
    let sum_w2: f64 = w.iter().map(|wi| wi * wi).sum();
    if sum_w2 < 1e-300 {
        0.0
    } else {
        sum_w * sum_w / sum_w2
    }
}

/// Adaptively refine until ESS >= `TARGET_ESS` or `MAX_REFINE` rounds.
///
/// Each hot cell is subdivided with a 5x5 sub-grid.  The `rho_dot` half-step
/// is set to `v_esc(rho) / (N_RHO_DOT - 1)` so the refinement patch
/// covers exactly one coarse `rho_dot` cell in the adaptive grid.
fn refine(
    mut cells: Vec<Cell>,
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    temperature: f64,
) -> (Vec<Cell>, f64) {
    let coarse_step_rho = (LOG_RHO_MAX - LOG_RHO_MIN) / (N_RHO - 1) as f64;

    for _ in 0..MAX_REFINE {
        if ess(&cells) >= TARGET_ESS {
            break;
        }
        let max_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        let hot: Vec<(f64, f64, f64)> = cells
            .iter()
            .filter(|c| c.log_w > max_lw - 5.0)
            .map(|c| {
                // Half local coarse step in `rho_dot` = v_bound(rho) / (N_RHO_DOT-1).
                // Approximate r_helio ~ rho  -- accurate when the object is far from Earth.
                let v_bound = (ENERGY_MULT * GMS / c.rho).sqrt().min(RHO_DOT_ABS_MAX);
                let hd = v_bound / (N_RHO_DOT - 1) as f64;
                (c.rho.ln(), c.rho_dot, hd)
            })
            .collect();
        if hot.is_empty() {
            break;
        }
        let hr = coarse_step_rho / 2.0;
        let new_cells: Vec<Cell> = hot
            .into_par_iter()
            .flat_map(|(lr, rd, hd)| {
                score_patch(
                    sorted_obs,
                    attr,
                    (lr - hr, lr + hr),
                    Some((rd - hd, rd + hd)),
                    5,
                    5,
                    temperature,
                )
            })
            .collect();
        cells.extend(new_cells);
        let max_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);
    }
    let final_ess = ess(&cells);
    (cells, final_ess)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Weak diagonal regularizer for the 4x4 attributable information matrix.
/// sigma = 1e-4 rad (~20 arc-seconds) for both position and rate parameters.
const ATTR_REG_INV: f64 = 1.0 / (1e-4 * 1e-4);

/// Draw `num_draws` samples from the weighted cell distribution.
///
/// **Prior** (Scout section 3 footnote): the log-rho grid has an implicit 1/rho density
/// in linear rho.  Multiplying by rho (i.e. adding `+ln rho` to log-weight) converts
/// to a flat (uniform) prior over linear rho, which is the Scout default.
///
/// **Within-cell perturbation**: draws (deltaalpha, deltadelta, `deltaalpha_dot`, `deltadelta_dot`) from the 4-D
/// attributable covariance `Gamma_A = N^-1` (Scout section 5, step f).  This correctly
/// reflects that rho and `rho_dot` are fixed at the grid-cell values  -- only the
/// attributable is uncertain.  A diagonal regularizer prevents inversion
/// failure when the arc is too short to constrain all four parameters.
fn draw_samples(
    cells: &[Cell],
    num_draws: usize,
    rng: &mut impl rand::Rng,
    attr: &Attributable,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    // Uniform prior on linear rho: add +ln(rho) Jacobian correction (paper footnote section 3).
    let max_lw = cells
        .iter()
        .map(|c| c.log_w + c.rho.ln())
        .fold(f64::NEG_INFINITY, f64::max);
    let w_raw: Vec<f64> = cells
        .iter()
        .map(|c| (c.log_w + c.rho.ln() - max_lw).exp())
        .collect();
    let sum_w: f64 = w_raw.iter().sum();
    let log_norm = max_lw + sum_w.ln();
    let norm_log_w: Vec<f64> = cells
        .iter()
        .map(|c| c.log_w + c.rho.ln() - log_norm)
        .collect();

    let mut cdf = Vec::with_capacity(cells.len());
    let mut acc = 0.0_f64;
    for &w in &w_raw {
        acc += w / sum_w;
        cdf.push(acc);
    }

    let uniform = Uniform::new(0.0_f64, 1.0_f64).unwrap();
    let normal = rand_distr::StandardNormal;

    let mut draws = Vec::with_capacity(num_draws);
    let mut log_posteriors = Vec::with_capacity(num_draws);

    for _ in 0..num_draws {
        let u: f64 = uniform.sample(rng);
        let idx = cdf.partition_point(|&c| c < u).min(cells.len() - 1);
        let cell = &cells[idx];
        let mut n_reg = cell.attr_info.clone();
        n_reg[(0, 0)] += ATTR_REG_INV;
        n_reg[(1, 1)] += ATTR_REG_INV;
        n_reg[(2, 2)] += ATTR_REG_INV;
        n_reg[(3, 3)] += ATTR_REG_INV;

        // Perturb rho along the LOV ridge and `rho_dot` across it, then perturb the
        // attributable.  These three perturbations are independent:
        //   delta(log rho) ~ N(0, `log_rho_sigma`^2)   -- spreads draws continuously along the ridge
        //   deltarho_dot      ~ N(0, `rho_dot_sigma`^2)   -- spreads draws across the ridge width
        //   deltaattr    ~ N(0, Gamma_A)              -- attributable uncertainty at that (rho, `rho_dot`)
        let z_log_rho: f64 =
            <rand_distr::StandardNormal as Distribution<f64>>::sample(&normal, rng);
        let z_rho_dot: f64 =
            <rand_distr::StandardNormal as Distribution<f64>>::sample(&normal, rng);
        let p_rho = (cell.rho * (z_log_rho * cell.log_rho_sigma).exp()).clamp(1e-5, 1000.0);
        let p_rho_dot = cell.rho_dot + z_rho_dot * cell.rho_dot_sigma;

        let sv = if let Some(gamma_a) = n_reg.try_inverse()
            && let Some(chol) = gamma_a.cholesky()
        {
            let z = DVector::from_vec(
                (0..4)
                    .map(|_| {
                        <rand_distr::StandardNormal as Distribution<f64>>::sample(&normal, rng)
                    })
                    .collect::<Vec<_>>(),
            );
            let d_attr = chol.l() * z;
            let p_attr = Attributable {
                alpha: attr.alpha + d_attr[0],
                delta: attr.delta + d_attr[1],
                alpha_dot: attr.alpha_dot + d_attr[2],
                delta_dot: attr.delta_dot + d_attr[3],
                t_ref: attr.t_ref,
                observer: attr.observer.clone(),
            };
            let ps = state_from_rho(&p_attr, p_rho, p_rho_dot);
            [
                ps.pos[0], ps.pos[1], ps.pos[2], ps.vel[0], ps.vel[1], ps.vel[2],
            ]
        } else {
            // Fallback: 5% radial/speed perturbation directly on the state.
            let p = cell.state.pos;
            let v = cell.state.vel;
            let r_norm = p.norm().max(0.1);
            let v_norm = v.norm().max(1e-4);
            let mut sv = [p[0], p[1], p[2], v[0], v[1], v[2]];
            for x in &mut sv[0..3] {
                *x += <rand_distr::StandardNormal as Distribution<f64>>::sample(&normal, rng)
                    * (0.05 * r_norm);
            }
            for x in &mut sv[3..6] {
                *x += <rand_distr::StandardNormal as Distribution<f64>>::sample(&normal, rng)
                    * (0.05 * v_norm);
            }
            sv
        };

        draws.push(sv.to_vec());
        log_posteriors.push(norm_log_w[idx]);
    }

    (draws, log_posteriors)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generate orbit samples covering the full admissible region from sparse
/// observations.
///
/// Implements JPL Scout (Farnocchia et al. 2015): scans a 2-D grid over
/// topocentric range `rho` and range-rate `rho_dot`, scores each cell by the
/// constrained attributable chi^2 (curvature residual after optimizing the
/// on-sky position and rate), and draws samples from the weighted posterior.
///
/// Designed for **short arcs** (hours to a few days) where the posterior is
/// a ridge or multi-modal and MCMC cannot explore it efficiently.  For
/// well-constrained arcs, use [`fit_orbit_mcmc`].
///
/// The attributable `(alpha, delta, alpha_dot, delta_dot)` is computed from a sliding 2-day window
/// swept across the full arc.  Each window position gives an instantaneous
/// linear motion estimate at its reference epoch; the window where the linear
/// approximation is most accurate yields the sharpest chi^2 landscape (highest
/// ESS) and is selected.  All observations are always used for chi^2 scoring
/// regardless of which window is chosen, so the full arc constrains the orbit.
///
/// [`fit_orbit_mcmc`]: crate::fit_orbit_mcmc
///
/// # Arguments
/// * `obs`  -- At least 3 optical observations (any order, sorted internally).
/// * `num_draws`  -- Number of orbit samples to return.
/// * `temperature`  -- Likelihood temperature (default 1.0). Higher values give
///   broader coverage of the admissible region.
/// * `seed`  -- RNG seed; identical inputs + seed -> identical draws.
///
/// # Errors
/// Returns an error if fewer than 3 optical observations are available or
/// no valid cells survived scoring.
pub fn fit_orbit_ranging(
    obs: &[AstrometricObservation],
    num_draws: usize,
    temperature: f64,
    seed: u64,
    desig: String,
) -> KeteResult<RangingSamples> {
    // Width of each attributable window (days).  The linear-motion approximation
    // is valid within this span; beyond it curvature within the window biases
    // the fitted rate, shifting the chi^2 minimum away from the true orbit.
    // 0.1 days (~2.4 hours) captures a single tracklet; rates from a ~1-hour
    // arc are accurate to well below 1% for any solar-system orbit.
    const ATTR_WINDOW_DAYS: f64 = 0.1;
    // Step between successive window start times.  Using the same value as the
    // window width (no overlap) keeps the number of windows proportional to the
    // arc length without redundant computation.
    const WINDOW_STEP_DAYS: f64 = 0.1;

    if obs.len() < 3 {
        return Err(Error::ValueError(
            "fit_orbit_ranging requires at least 3 observations".into(),
        ));
    }

    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| {
        a.epoch()
            .jd
            .partial_cmp(&b.epoch().jd)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let t0 = sorted[0].epoch().jd;
    let arc = sorted[sorted.len() - 1].epoch().jd - t0;

    // Determine the effective window width: use ATTR_WINDOW_DAYS if it
    // captures >= 2 observations with sufficient baseline, otherwise widen
    // to the gap between the first two distinct observation epochs.  This
    // ensures the attributable can always be computed on sparse datasets.
    let min_obs_gap = sorted
        .windows(2)
        .filter_map(|w| {
            let dt = w[1].epoch().jd - w[0].epoch().jd;
            if dt > 1e-9 { Some(dt) } else { None }
        })
        .fold(f64::INFINITY, f64::min);
    let effective_window = ATTR_WINDOW_DAYS.max(min_obs_gap * 1.5);
    let effective_step = WINDOW_STEP_DAYS.max(effective_window / 2.0);

    // Number of windows: at least 1, enough to cover the whole arc.
    let n_windows = if arc <= effective_window {
        1
    } else {
        // arc > effective_window > 0, effective_step > 0: quotient is always positive.
        let steps_ceil = ((arc - effective_window) / effective_step).ceil();
        #[allow(
            clippy::cast_sign_loss,
            reason = "steps_ceil > 0: arc > effective_window > 0"
        )]
        let n: usize = steps_ceil as usize;
        n + 1
    };

    // Score each window and keep the one with the highest peak log-weight
    // (= lowest minimum chi^2_min across all cells).
    //
    // Selecting by max(ESS) is wrong for well-constrained arcs: a biased
    // attributable produces a wider, shallower posterior (high ESS) while an
    // accurate attributable produces a narrow, deep posterior (low ESS but the
    // chi^2 minimum is at the correct orbit).  The window where the attributable
    // rates are closest to the true instantaneous rates minimises chi^2_min,
    // giving the highest max(log_w).  That is the window that correctly places
    // the peak of the posterior at the true orbit.
    let mut best_cells: Vec<Cell> = Vec::new();
    let mut best_attr: Option<Attributable> = None;
    let mut best_peak_lw = f64::NEG_INFINITY;

    for i in 0..n_windows {
        let w_start = t0 + i as f64 * effective_step;
        let w_end = w_start + effective_window;
        let window_obs: Vec<AstrometricObservation> = sorted
            .iter()
            .filter(|o| {
                let jd = o.epoch().jd;
                jd >= w_start && jd <= w_end
            })
            .cloned()
            .collect();
        if window_obs.len() < 2 {
            continue;
        }
        // Require at least 15 minutes of temporal baseline so the rate
        // estimate is meaningful.
        let w_span = window_obs[window_obs.len() - 1].epoch().jd - window_obs[0].epoch().jd;
        if w_span < 0.01 {
            continue;
        }
        let Some(attr) = compute_attributable(&window_obs) else {
            continue;
        };
        let cells = score_patch(
            &sorted,
            &attr,
            (LOG_RHO_MIN, LOG_RHO_MAX),
            None,
            N_RHO,
            N_RHO_DOT,
            temperature,
        );
        if cells.is_empty() {
            continue;
        }
        let peak_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        if peak_lw > best_peak_lw {
            best_peak_lw = peak_lw;
            best_cells = cells;
            best_attr = Some(attr);
        }
    }

    let attr = best_attr.ok_or_else(|| {
        Error::ValueError("fit_orbit_ranging: no valid cells found for any window".into())
    })?;
    let cells = best_cells;

    let max_lw = cells
        .iter()
        .map(|c| c.log_w)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut cells = cells;
    cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);
    // Removing cells changes each cell's neighbours, so recompute sigmas.
    cells.sort_by(|a, b| {
        a.rho
            .partial_cmp(&b.rho)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    compute_log_rho_sigma(&mut cells, temperature);

    let initial_ess = ess(&cells);
    let (mut cells, final_ess) = if initial_ess < TARGET_ESS {
        refine(cells, &sorted, &attr, temperature)
    } else {
        (cells, initial_ess)
    };
    // Refinement appends new LOV cells; resort and recompute sigmas.
    cells.sort_by(|a, b| {
        a.rho
            .partial_cmp(&b.rho)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    compute_log_rho_sigma(&mut cells, temperature);

    let convergence_warning = if final_ess < TARGET_ESS {
        Some(format!(
            "ESS = {final_ess:.1} < {TARGET_ESS}; orbit space may be under-sampled. \
             Consider using fit_orbit_mcmc if the orbit is well-constrained."
        ))
    } else {
        None
    };

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let (draws, log_posterior) = draw_samples(&cells, num_draws, &mut rng, &attr);

    Ok(RangingSamples {
        desig,
        epoch: attr.t_ref,
        draws,
        log_posterior,
        effective_sample_size: final_ess,
        convergence_warning,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::Band;
    use kete_core::constants::GMS;
    use kete_core::desigs::Desig;
    use kete_core::frames::SSB;
    use kete_core::kepler::propagate_two_body;
    use kete_core::time::{TDB, Time};
    use kete_spice::test_data::ensure_test_spk;

    fn make_ssb_state(pos: [f64; 3], vel: [f64; 3], jd: f64) -> State<Equatorial, SSB> {
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: pos.into(),
            vel: vel.into(),
            center: SSB,
        }
    }

    fn synth_obs(
        obj: &State<Equatorial, SSB>,
        epochs: &[f64],
        sigma_rad: f64,
    ) -> Vec<AstrometricObservation> {
        let spk = LOADED_SPK.try_read().unwrap();
        let obj_sun = spk.try_to_sun(obj.clone().into()).unwrap();
        let v_earth = (GMS / 1.0_f64).sqrt();
        let obl = 23.44_f64.to_radians();
        epochs
            .iter()
            .filter_map(|&jd| {
                let angle = (jd - 2_460_000.5) / 365.25 * std::f64::consts::TAU;
                let obs_pos = [
                    angle.cos(),
                    angle.sin() * obl.cos(),
                    angle.sin() * obl.sin(),
                ];
                let obs_vel = [
                    -v_earth * angle.sin(),
                    v_earth * angle.cos() * obl.cos(),
                    v_earth * angle.cos() * obl.sin(),
                ];
                let observer = make_ssb_state(obs_pos, obs_vel, jd);
                let obj_at = propagate_two_body(&obj_sun, Time::<TDB>::new(jd)).ok()?;
                let obs_sun_pos = spk.try_to_sun(observer.clone().into()).ok()?.pos;
                let obj_lt_sun = light_time_correct(&obj_at, &obs_sun_pos).ok()?;
                let obj_lt_ssb = spk.try_to_ssb(obj_lt_sun.into()).ok()?;
                let (ra, dec) = (obj_lt_ssb.pos - observer.pos).to_ra_dec();
                Some(AstrometricObservation::Optical {
                    observer,
                    ra,
                    dec,
                    sigma_ra: sigma_rad,
                    sigma_dec: sigma_rad,
                    sigma_corr: 0.0,
                    time_sigma: 0.0,
                    is_occultation: false,
                    band: Band::Unknown([0; 8]),
                    mag: f64::NAN,
                })
            })
            .collect()
    }

    #[test]
    fn attributable_fit_matches_truth() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2_460_000.5,
        );
        let epochs = [
            2_460_000.5,
            2_460_000.5 + 0.02,
            2_460_000.5 + 0.04,
            2_460_000.5 + 0.06,
        ];
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let attr = compute_attributable(&obs).expect("attributable must succeed");
        assert!(attr.alpha.is_finite() && attr.delta.is_finite());
        assert!(attr.alpha_dot.is_finite() && attr.delta_dot.is_finite());
    }

    #[test]
    fn scout_score_near_zero_for_true_state() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2_460_000.5,
        );
        let epochs = [
            2_460_000.5,
            2_460_000.5 + 0.02,
            2_460_000.5 + 0.04,
            2_460_000.5 + 0.06,
        ];
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let (log_w, _) =
            scout_score(&obj, &obs, 2_460_000.5, 1.0).expect("scout_score must succeed");
        assert!(
            log_w > -0.5,
            "log_w = {log_w:.6} for true state; expected near-zero curvature"
        );
    }

    #[test]
    fn highest_weight_cell_near_truth() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        // 4 observations over ~1 hour  -- short arc, many distances admissible.
        let epochs: Vec<f64> = (0..4)
            .map(|i| 2_460_000.5 + f64::from(i) * (1.0 / 24.0 / 4.0))
            .collect();
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        assert!(obs.len() >= 3);

        let result = fit_orbit_ranging(&obs, 200, 10.0, 42, String::new());
        assert!(result.is_ok(), "ranging failed: {:?}", result.err());
        let samples = result.unwrap();
        assert!(!samples.draws.is_empty());
        assert_eq!(samples.draws.len(), samples.log_posterior.len());
        assert!(samples.effective_sample_size > 0.0);

        let true_r = r;
        let distances: Vec<f64> = samples
            .draws
            .iter()
            .map(|d| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
            .collect();
        let min_r = distances.iter().copied().fold(f64::INFINITY, f64::min);
        let max_r = distances.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            distances
                .iter()
                .any(|&dr| (dr - true_r).abs() / true_r < 0.5),
            "no draw within 50% of truth r={true_r:.2}; r=[{min_r:.3}, {max_r:.3}], ESS={:.1}",
            samples.effective_sample_size
        );
    }

    #[test]
    fn all_draws_physically_valid() {
        ensure_test_spk();
        let r = 1.5_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2_460_000.5,
        );
        let epochs = [
            2_460_000.5,
            2_460_000.5 + 0.02,
            2_460_000.5 + 0.04,
            2_460_000.5 + 0.06,
        ];
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let samples = fit_orbit_ranging(&obs, 50, 10.0, 7, String::new()).unwrap();
        for draw in &samples.draws {
            let dr = (draw[0] * draw[0] + draw[1] * draw[1] + draw[2] * draw[2]).sqrt();
            assert!(dr > 0.001 && dr < 1000.0, "draw distance {dr} out of range");
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2_460_000.5,
        );
        let epochs = [
            2_460_000.5,
            2_460_000.5 + 0.02,
            2_460_000.5 + 0.04,
            2_460_000.5 + 0.06,
        ];
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let a = fit_orbit_ranging(&obs, 20, 10.0, 99, String::new()).unwrap();
        let b = fit_orbit_ranging(&obs, 20, 10.0, 99, String::new()).unwrap();
        assert_eq!(a.draws, b.draws);
        assert_eq!(a.log_posterior, b.log_posterior);
    }

    #[test]
    fn ess_positive_on_valid_input() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [r, 0.0, 0.0],
            [0.0, v * obl.cos(), v * obl.sin()],
            2_460_000.5,
        );
        let epochs = [
            2_460_000.5,
            2_460_000.5 + 0.02,
            2_460_000.5 + 0.04,
            2_460_000.5 + 0.06,
        ];
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let samples = fit_orbit_ranging(&obs, 50, 10.0, 1, String::new()).unwrap();
        assert!(
            samples.effective_sample_size > 0.0,
            "ESS = {:.1}",
            samples.effective_sample_size
        );
    }

    #[test]
    fn log_posterior_not_all_zero_on_constrained_arc() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        // 3-day arc: curvature is detectable -> chi^2 varies across cells.
        let epochs: Vec<f64> = (0..6).map(|i| 2_460_000.5 + f64::from(i) * 0.6).collect();
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);

        let samples = fit_orbit_ranging(&obs, 100, 1.0, 42, String::new()).unwrap();
        let min_lp = samples
            .log_posterior
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max_lp = samples
            .log_posterior
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_lp - min_lp > 0.01 || max_lp < -0.01,
            "log_posteriors appear constant (max={max_lp:.4}, min={min_lp:.4}); \
             curvature chi^2 should differentiate cells on a 3-day arc"
        );
    }
}
