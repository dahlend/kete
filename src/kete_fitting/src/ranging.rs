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
use kete_spice::spk::SpkCollection;
use nalgebra::{DMatrix, DVector};
use nuts_rs::rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::Distribution;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// Grid constants
// ---------------------------------------------------------------------------

/// log-rho grid: ~5e-7 AU (~75 km) to 1000 AU.
const LOG_RHO_MIN: f64 = -14.5129;
const LOG_RHO_MAX: f64 = 6.9078;

/// Absolute cap on the adaptive `rho_dot` range (AU/day).
/// Escape velocity at 1e-5 AU is enormous; this cap prevents unreasonably
/// large grids for impactor-range distances.
const RHO_DOT_ABS_MAX: f64 = 2.0; // ~3460 km/s

/// Energy multiplier controlling the `rho_dot` scan range (parabolic boundary).
///
/// `ENERGY_MULT = 2` -> strict bound orbits only (e < 1).
/// `ENERGY_MULT = 8` -> tangential orbits up to e ~ 7; typical non-radial
///   orbits up to e ~ 2--3.  Comfortably covers interstellar-like e ~ 1--2.
const ENERGY_MULT: f64 = 5.0;

/// Energy multiplier for the physical-validity hard ceiling.  Acts as a
/// runaway guard; the actual posterior shape at the parabolic boundary is
/// produced by the soft energy prior (see `ENERGY_PRIOR_SIGMA`), not by this
/// cutoff.  Set well past `ENERGY_PRIOR_SIGMA` so the soft taper kills cells
/// long before they hit this ceiling.
const ENERGY_MULT_VALID: f64 = 20.0;

/// Sigma of the Gaussian energy prior past the parabolic boundary, expressed
/// in units of `v^2 * r / GMS` (= 2 at parabolic).  Cell weights are multiplied
/// by `exp(-0.5 * ((ratio - 2) / sigma)^2)` for `ratio > 2`.  Smaller sigma
/// gives a sharper falloff; larger sigma admits more hyperbolic orbits.
const ENERGY_PRIOR_SIGMA: f64 = 1.0;

const N_RHO: usize = 500;
const N_RHO_DOT: usize = 500;

const TARGET_ESS: f64 = 50.0;
/// Largest allowed change in log density over half a cell step within the cells
/// holding most of the posterior mass; see `max_half_step_change`.
const MAX_HALF_STEP_LOG_CHANGE: f64 = 1.0;
/// Largest allowed shift of the posterior, in standard deviations, when dominant
/// cells are scored by their iterated corrected orbits; see `linear_model_check`.
const MAX_LINEAR_MODEL_SHIFT_SIGMA: f64 = 0.5;
/// Best-orbit chi^2 above `dof + FIT_CHI2_SIGMAS * sqrt(2 dof)` raises a warning.
const FIT_CHI2_SIGMAS: f64 = 5.0;
const MAX_REFINE: usize = 4;
/// Cells whose log weight is more than this below the maximum are dropped.
const LOG_W_FLOOR: f64 = 50.0;

// ---------------------------------------------------------------------------
// Public output type
// ---------------------------------------------------------------------------

/// Orbit samples from the weighted ranging grid.
///
/// Each draw is a `[x, y, z, vx, vy, vz]` state in AU/AU*day,
/// SSB Equatorial frame, at `epoch`.  Draws are distributed according to the
/// posterior and are equally weighted.
#[derive(Debug, Clone)]
pub struct RangingSamples {
    /// Epoch of every draw (JD TDB): the attributable reference epoch.  Each
    /// state is constructed at its own light-time-corrected emission epoch and
    /// propagated two-body to this epoch.
    pub epoch: f64,
    /// Orbit draws: `[num_draws][6]`, SSB Equatorial AU/AU*day.
    pub draws: Vec<Vec<f64>>,
    /// Log posterior density per unit `(rho, rho_dot)` of the grid cell each draw
    /// came from, as a natural log relative to the maximum across draws.  This
    /// describes where a draw sits in the posterior; it is not an importance
    /// weight.
    pub log_posterior: Vec<f64>,
    /// Effective sample size `(sumw_i)^2 / sumw_i^2` over the grid cells before
    /// drawing, with `w_i` the cell masses.
    ///
    /// This counts cells, not independent orbit solutions.  Refinement splits
    /// cells, so it grows with grid resolution and can be much larger than the
    /// number of distinct solutions the observations allow.
    pub effective_sample_size: f64,
    /// Set when, at the end of refinement, `effective_sample_size < 50`, the grid is
    /// still coarser than the posterior structure it samples, the linear attributable
    /// model does not describe the observations, or the best orbit fits the
    /// observations far worse than their uncertainties allow.
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
    /// Tempered 4x4 attributable normal matrix `N / T`, `N = H^4^T W H^4`.
    attr_info: DMatrix<f64>,
    /// Best-fit attributable correction `N^-1 b` at this cell's `(rho, rho_dot)`,
    /// ordered `[alpha, delta, alpha_dot, delta_dot]`.
    attr_delta: [f64; 4],
    rho: f64,
    rho_dot: f64,
    /// Cell extent in log(rho); uniform across the grid for a given patch.
    log_rho_step: f64,
    /// Cell extent in `rho_dot`; varies per row in the adaptive scan.
    rho_dot_step: f64,
}

impl Cell {
    /// Log posterior mass of the cell under a uniform prior in `(rho, rho_dot)`.
    ///
    /// `log_w` is a density per unit `(rho, rho_dot)`; the cell spans
    /// `rho * log_rho_step` in `rho` and `rho_dot_step` in `rho_dot`.  Cells of
    /// different sizes (per-row `rho_dot` ranges, refined sub-cells) therefore
    /// carry mass in proportion to their area.
    fn log_mass(&self) -> f64 {
        self.log_w + self.rho.ln() + self.log_rho_step.ln() + self.rho_dot_step.ln()
    }
}

// ---------------------------------------------------------------------------
// Attributable computation
// ---------------------------------------------------------------------------

/// Fit `(alpha0, delta0, alpha_dot, delta_dot)` at the first observation epoch by weighted LS of
/// (RA, Dec) vs time.  Returns `None` when fewer than 2 optical observations
/// are available.
///
/// The stored observer keeps the first observation's position but carries the
/// secant velocity across the arc, so that it describes the same averaged motion
/// as the fitted rates.  See the comment at the end of the function.
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
    let mut observer = observer.clone();

    // `alpha_dot`/`delta_dot` above are the slope of a straight line through the
    // whole arc, so they describe the average on-sky rate rather than the rate at
    // `t_ref`.  `state_from_rho` pairs them with `observer.vel` in
    // `v = v_obs + rho_dot * los + rho * d(los)/dt`, and that identity only holds
    // when both halves refer to the same observer motion.  An observer whose
    // non-inertial motion is small next to its heliocentric motion satisfies this
    // either way -- a ground station's rotation is under 2% of Earth's orbital
    // velocity -- but one in low Earth orbit does not: its instantaneous velocity
    // carries a ~0.0044 AU/day term, a quarter of Earth's orbital velocity, that a
    // fit spanning many revolutions averages away.
    //
    // Use the secant velocity of the observer across the arc, which is the motion
    // the straight-line fit actually saw.  It needs only the observer positions
    // already carried by the observations, requires no knowledge of what kind of
    // observer produced them, and tends to the instantaneous velocity as the arc
    // shortens.
    let last_optical = sorted_obs
        .iter()
        .rev()
        .find_map(|obs| obs.as_optical().ok().map(|(_, _, obs)| obs));
    if let Some(last) = last_optical {
        let dt = last.epoch.jd - t_ref;
        if dt > 0.0 {
            observer.vel = (last.pos - observer.pos) / dt;
        }
    }

    Some(Attributable {
        t_ref,
        alpha: alpha0,
        delta: delta0,
        alpha_dot,
        delta_dot,
        observer,
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

/// Reject orbits outside [0.001, 1000] AU or with v^2 >= `ENERGY_MULT_VALID * GMS / r`.
///
/// The validity ceiling is wider than the scan's parabolic boundary so cells in
/// the padded region beyond the admissible curve survive scoring; the chi^2
/// landscape then produces a soft falloff instead of a hard polygonal edge at
/// the boundary.  Orbits beyond this ceiling are not physical for any
/// solar-system population.
fn is_physically_valid(pos_helio: Vector<Equatorial>, vel_helio: Vector<Equatorial>) -> bool {
    let r = pos_helio.norm();
    if !(0.001..=1000.0).contains(&r) {
        return false;
    }
    vel_helio.norm_squared() < ENERGY_MULT_VALID * GMS / r
}

// ---------------------------------------------------------------------------
// Scout scoring: constrained attributable LS
// ---------------------------------------------------------------------------

/// Residual of one optical observation against a two-body orbit.
struct Residual {
    /// Observation epoch minus the attributable epoch (days).
    dt: f64,
    /// Observed minus predicted RA (radians, RA coordinate direction).
    nu_alpha: f64,
    /// Observed minus predicted Dec (radians).
    nu_delta: f64,
    /// RA weight, the diagonal of the base weight matrix.
    w_alpha: f64,
    /// Dec weight, the diagonal of the base weight matrix.
    w_delta: f64,
}

/// Residuals of `state`, propagated two-body and light-time corrected, against every
/// optical observation that can be evaluated.  Returns `None` if the state cannot be
/// referred to the Sun.
fn optical_residuals(
    spk: &SpkCollection,
    state: &State<Equatorial, SSB>,
    sorted_obs: &[AstrometricObservation],
    t_ref: f64,
) -> Option<Vec<Residual>> {
    let sun_state = spk.try_to_sun(state.clone()).ok()?;
    let mut entries = Vec::with_capacity(sorted_obs.len());

    for obs in sorted_obs {
        let Ok((alpha_obs, delta_obs, obs_ssb)) = obs.as_optical() else {
            continue;
        };
        let Ok(prop) = propagate_two_body(&sun_state, obs.epoch()) else {
            continue;
        };
        let Ok(obs_sun) = spk.try_to_sun(obs_ssb.clone()) else {
            continue;
        };
        let Ok(lt_sun) = light_time_correct(&prop, &obs_sun.pos) else {
            continue;
        };
        let lt_pos = differential_light_deflect(&obs_sun.pos, lt_sun.pos);
        let Some(lt_ssb) = spk
            .try_to_ssb(State {
                pos: lt_pos,
                ..lt_sun
            })
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
        entries.push(Residual {
            dt: obs.epoch().jd - t_ref,
            nu_alpha: nu_a,
            nu_delta: delta_obs - delta_pred,
            w_alpha: w[(0, 0)],
            w_delta: w[(1, 1)],
        });
    }

    Some(entries)
}

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
/// et al. 2015, Eq. 1).  Also returns the tempered information `N / T` for
/// within-cell sampling and the
/// best-fit correction `N^-1 b`, which `draw_samples` adds to the attributable
/// so that the drawn states are the constrained LS solutions rather than the
/// uncorrected straight-line attributable.  The correction absorbs curvature
/// over the arc and any mismatch between the attributable's reference observer
/// state and the per-observation observer states the residuals are computed
/// against.
///
/// Requires >= 3 observations (>= 1 dof per coordinate after 2-parameter fit).
/// `spk` is passed in so that one read guard covers a whole scan.
fn scout_score(
    spk: &SpkCollection,
    state: &State<Equatorial, SSB>,
    sorted_obs: &[AstrometricObservation],
    t_ref: f64,
    temperature: f64,
) -> Option<(f64, DMatrix<f64>, [f64; 4])> {
    let entries = optical_residuals(spk, state, sorted_obs, t_ref)?;
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

    // b^T N^{-1} b and N^{-1} b for RA: analytic 2x2 inverse.
    let det_a = sw_a * swt2_a - swt_a * swt_a;
    let (chi2_attr_a, d_alpha, d_alpha_dot) = if det_a > 1e-30 {
        (
            (swt2_a * swnu_a * swnu_a - 2.0 * swt_a * swnu_a * swnut_a + sw_a * swnut_a * swnut_a)
                / det_a,
            (swt2_a * swnu_a - swt_a * swnut_a) / det_a,
            (sw_a * swnut_a - swt_a * swnu_a) / det_a,
        )
    } else if sw_a > 1e-30 {
        (swnu_a * swnu_a / sw_a, swnu_a / sw_a, 0.0)
    } else {
        (0.0, 0.0, 0.0)
    };

    let det_d = sw_d * swt2_d - swt_d * swt_d;
    let (chi2_attr_d, d_delta, d_delta_dot) = if det_d > 1e-30 {
        (
            (swt2_d * swnu_d * swnu_d - 2.0 * swt_d * swnu_d * swnut_d + sw_d * swnut_d * swnut_d)
                / det_d,
            (swt2_d * swnu_d - swt_d * swnut_d) / det_d,
            (sw_d * swnut_d - swt_d * swnu_d) / det_d,
        )
    } else if sw_d > 1e-30 {
        (swnu_d * swnu_d / sw_d, swnu_d / sw_d, 0.0)
    } else {
        (0.0, 0.0, 0.0)
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

    Some((
        -chi2_min / (2.0 * temperature),
        n_attr / temperature,
        [d_alpha, d_delta, d_alpha_dot, d_delta_dot],
    ))
}

// ---------------------------------------------------------------------------
// Grid scoring
// ---------------------------------------------------------------------------

/// Scan a `(rho, rho_dot)` patch and return scored cells.
///
/// `rho_dot_range = None`: adaptive range per row, centered on the observer's
/// heliocentric radial velocity, with `n_rdot` points per row (coarse scan).
/// `rho_dot_range = Some(f)`: row `i` of the `n_rho` rows spans `f(i) = (min,
/// max, n)`, and is skipped when `f(i)` is `None`; `n_rdot` is unused.
///
/// Cell extents are the spacing between grid points, so `n_rho` and every row's
/// point count must be at least 2; rows with fewer points are skipped and a
/// patch with `n_rho < 2` is empty.  Returns no cells if the SPK cannot be read.
/// One SPK read guard is held for the whole scan.
fn score_patch(
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    log_rho_range: (f64, f64),
    rho_dot_range: Option<&(dyn Fn(usize) -> Option<(f64, f64, usize)> + Sync)>,
    n_rho: usize,
    n_rdot: usize,
    temperature: f64,
) -> Vec<Cell> {
    if n_rho < 2 {
        return vec![];
    }
    let Ok(spk_guard) = LOADED_SPK.try_read() else {
        return vec![];
    };
    let spk: &SpkCollection = &spk_guard;
    let (sun_pos_ssb, sun_vel_ssb, obs_helio_pos, obs_helio_vel, los, los_dot) = {
        let Ok(obs_helio) = spk.try_to_sun(attr.observer.clone()) else {
            return vec![];
        };
        let sun_pos = attr.observer.pos - obs_helio.pos;
        let sun_vel = attr.observer.vel - obs_helio.vel;
        let (sin_a, cos_a) = attr.alpha.sin_cos();
        let (sin_d, cos_d) = attr.delta.sin_cos();
        let los = Vector::<Equatorial>::new([cos_d * cos_a, cos_d * sin_a, sin_d]);
        let los_da = Vector::<Equatorial>::new([-sin_a * cos_d, cos_a * cos_d, 0.0]);
        let los_dd = Vector::<Equatorial>::new([-cos_a * sin_d, -sin_a * sin_d, cos_d]);
        let los_dot = los_da * attr.alpha_dot + los_dd * attr.delta_dot;
        (sun_pos, sun_vel, obs_helio.pos, obs_helio.vel, los, los_dot)
    };

    let (la_min, la_max) = log_rho_range;
    let log_rho_step = (la_max - la_min) / (n_rho - 1) as f64;

    // Each rho row produces multiple cells (every admissible rho_dot).
    let cells: Vec<Cell> = (0..n_rho)
        .into_par_iter()
        .flat_map_iter(|ir| {
            let frac_r = ir as f64 / (n_rho - 1) as f64;
            let rho = (la_min + (la_max - la_min) * frac_r).exp();

            let row_range = rho_dot_range.map_or_else(
                || {
                    // Center the scan on the heliocentric radial-velocity zero point and size
                    // it from the parabolic boundary, then pad outward so cells exist past the
                    // admissible region.  Without padding the grid clips at the energy curve and
                    // the sampler produces hard polygonal edges; the chi^2 landscape and
                    // is_physically_valid below give the true soft falloff.
                    //
                    // The half-width is the along-LOS speed that reaches an energy ratio
                    // `v^2 r / GMS` of `ENERGY_MULT * RHO_DOT_PAD^2`.  Subtracting the
                    // transverse speed inside the square root (rather than padding a
                    // half-width computed without it) keeps rows whose transverse speed
                    // is near the admissible limit from collapsing to zero width.
                    const RHO_DOT_PAD: f64 = 1.5;
                    let r_helio = (obs_helio_pos + los * rho).norm().max(1e-6);
                    let v_perp = obs_helio_vel + los_dot * rho;
                    let v_along = v_perp[0] * los[0] + v_perp[1] * los[1] + v_perp[2] * los[2];
                    let v_trans_sq = (v_perp.norm_squared() - v_along * v_along).max(0.0);
                    let padded = (ENERGY_MULT * RHO_DOT_PAD * RHO_DOT_PAD * GMS / r_helio
                        - v_trans_sq)
                        .max(0.0)
                        .sqrt()
                        .min(RHO_DOT_ABS_MAX);
                    Some((-v_along - padded, -v_along + padded, n_rdot))
                },
                |f| f(ir),
            );
            let Some((rd_min, rd_max, n_row)) = row_range else {
                return Vec::new();
            };
            // A zero-width row has no area and would otherwise produce `n_row`
            // identical cells.  It only occurs when the transverse speed alone
            // exceeds the energy cap, where the energy prior is negligible.
            if rd_max <= rd_min || n_row < 2 {
                return Vec::new();
            }
            let rdot_step = (rd_max - rd_min) / (n_row - 1) as f64;

            let mut row_cells: Vec<Cell> = Vec::with_capacity(n_row);
            for id in 0..n_row {
                let frac_d = id as f64 / (n_row - 1) as f64;
                let rho_dot = rd_min + (rd_max - rd_min) * frac_d;
                let Some((log_w, attr_info, attr_delta)) = score_point(
                    spk,
                    sorted_obs,
                    attr,
                    rho,
                    rho_dot,
                    temperature,
                    (sun_pos_ssb, sun_vel_ssb),
                ) else {
                    continue;
                };
                row_cells.push(Cell {
                    log_w,
                    attr_info,
                    attr_delta,
                    rho,
                    rho_dot,
                    log_rho_step,
                    rho_dot_step: rdot_step,
                });
            }
            row_cells
        })
        .collect();

    cells
}

/// Score a single `(rho, rho_dot)` point: the tempered constrained attributable
/// log-likelihood from [`scout_score`] plus the soft energy prior.
///
/// `sun_ssb` is the Sun's SSB `(position, velocity)` at the attributable epoch.
/// Returns `None` for physically invalid orbits or failed scoring.
fn score_point(
    spk: &SpkCollection,
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    rho: f64,
    rho_dot: f64,
    temperature: f64,
    sun_ssb: (Vector<Equatorial>, Vector<Equatorial>),
) -> Option<(f64, DMatrix<f64>, [f64; 4])> {
    let state = state_from_rho(attr, rho, rho_dot);
    let pos_helio = state.pos - sun_ssb.0;
    let vel_helio = state.vel - sun_ssb.1;
    if !is_physically_valid(pos_helio, vel_helio) {
        return None;
    }
    let (log_w, attr_info, attr_delta) =
        scout_score(spk, &state, sorted_obs, attr.t_ref, temperature)?;
    if !log_w.is_finite() {
        return None;
    }
    // Soft Gaussian energy prior past the parabolic boundary.  Chi^2 alone has no
    // gradient in the energy direction over short arcs, so without this prior the
    // posterior would be flat to the validity ceiling and produce a hard polygonal
    // edge there.
    let r_h = pos_helio.norm();
    let energy_ratio = vel_helio.norm_squared() * r_h / GMS;
    let log_prior_energy = if energy_ratio > 2.0 {
        let excess = (energy_ratio - 2.0) / ENERGY_PRIOR_SIGMA;
        -0.5 * excess * excess
    } else {
        0.0
    };
    Some((log_w + log_prior_energy, attr_info, attr_delta))
}

// ---------------------------------------------------------------------------
// ESS, resolution, and adaptive refinement
// ---------------------------------------------------------------------------

fn ess(cells: &[Cell]) -> f64 {
    if cells.is_empty() {
        return 0.0;
    }
    // Same cell masses as draw_samples.
    let max_lm = cells
        .iter()
        .map(Cell::log_mass)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lm.is_finite() {
        return 0.0;
    }
    let w: Vec<f64> = cells
        .iter()
        .map(|c| (c.log_mass() - max_lm).exp())
        .collect();
    let sum_w: f64 = w.iter().sum();
    let sum_w2: f64 = w.iter().map(|wi| wi * wi).sum();
    if sum_w2 < 1e-300 {
        0.0
    } else {
        sum_w * sum_w / sum_w2
    }
}

/// The cells holding 99% of the posterior mass, largest mass first.
fn dominant_cells(cells: &[Cell]) -> Vec<&Cell> {
    const MASS_FRACTION: f64 = 0.99;
    let max_lm = cells
        .iter()
        .map(Cell::log_mass)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lm.is_finite() {
        return Vec::new();
    }
    let mut order: Vec<(f64, &Cell)> = cells
        .iter()
        .map(|c| ((c.log_mass() - max_lm).exp(), c))
        .collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let total: f64 = order.iter().map(|(w, _)| w).sum();
    let mut acc = 0.0;
    order
        .into_iter()
        .take_while(|(w, _)| {
            let before = acc;
            acc += w;
            before < MASS_FRACTION * total
        })
        .map(|(_, c)| c)
        .collect()
}

/// Result of [`linear_model_check`].
struct LinearModelCheck {
    /// Largest shift of the posterior mean of `ln(rho)` or `rho_dot`, in posterior
    /// standard deviations, when the dominant cells are scored by their iterated
    /// corrected orbits instead of the linear model.
    shift_sigma: f64,
    /// Number of dominant cells whose iterated correction did not settle.
    unconverged: usize,
    /// Smallest chi^2 (untempered) among the iterated corrected orbits.
    best_chi2: f64,
}

/// Check the linear attributable model on the cells holding 99% of the posterior
/// mass.
///
/// `scout_score` scores a cell by the chi^2 that the linear attributable model
/// `H = [[1, 0, dt, 0], [0, 1, 0, dt]]` predicts after one correction.  When the
/// model describes the arc, the corrected orbit has that chi^2.  Over long arcs, or
/// with observers far apart, it does not, and cells are ranked by a chi^2 their
/// corrected orbits do not have.
///
/// For each dominant cell the correction is repeated, each step linearized about
/// the previous corrected orbit, until its chi^2 changes by less than
/// `SETTLE_CHI2` or `MAX_ITER` steps.  The posterior mean of `ln(rho)` and
/// `rho_dot` over the dominant cells is then compared between the linear scores and
/// the iterated chi^2.  A disagreement shared by all cells cancels in the
/// normalization and does not move the posterior; only its variation across cells
/// does, which is what the shift measures.
///
/// Returns `None` if the SPK cannot be read or no cell can be evaluated.
fn linear_model_check(
    cells: &[Cell],
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    temperature: f64,
) -> Option<LinearModelCheck> {
    const MAX_ITER: usize = 6;
    const SETTLE_CHI2: f64 = 0.2;

    let spk_guard = LOADED_SPK.try_read().ok()?;
    let spk: &SpkCollection = &spk_guard;
    let chi2_of = |a: &Attributable, rho: f64, rho_dot: f64| -> Option<f64> {
        let entries =
            optical_residuals(spk, &state_from_rho(a, rho, rho_dot), sorted_obs, a.t_ref)?;
        Some(
            entries
                .iter()
                .map(|e| e.nu_alpha * e.nu_alpha * e.w_alpha + e.nu_delta * e.nu_delta * e.w_delta)
                .sum(),
        )
    };
    let corrected = |a: &Attributable, d: [f64; 4]| Attributable {
        alpha: a.alpha + d[0],
        delta: a.delta + d[1],
        alpha_dot: a.alpha_dot + d[2],
        delta_dot: a.delta_dot + d[3],
        t_ref: a.t_ref,
        observer: a.observer.clone(),
    };

    let dominant = dominant_cells(cells);
    // Per dominant cell: (log mass from the linear score, log mass from the iterated
    // chi^2, iterated chi^2, settled).
    let rows: Vec<(&Cell, f64, f64, f64, bool)> = dominant
        .par_iter()
        .filter_map(|c| {
            let (log_w, _, delta) = scout_score(
                spk,
                &state_from_rho(attr, c.rho, c.rho_dot),
                sorted_obs,
                attr.t_ref,
                temperature,
            )?;
            let mut a = corrected(attr, delta);
            let mut chi2 = chi2_of(&a, c.rho, c.rho_dot)?;
            let mut settled = false;
            for _ in 0..MAX_ITER {
                let (_, _, d) = scout_score(
                    spk,
                    &state_from_rho(&a, c.rho, c.rho_dot),
                    sorted_obs,
                    a.t_ref,
                    temperature,
                )?;
                let next = corrected(&a, d);
                let next_chi2 = chi2_of(&next, c.rho, c.rho_dot)?;
                let change = (chi2 - next_chi2).abs();
                if next_chi2 < chi2 {
                    a = next;
                    chi2 = next_chi2;
                }
                if change < SETTLE_CHI2 {
                    settled = true;
                    break;
                }
            }
            // `log_w` holds the energy prior on top of the linear log-likelihood.
            let prior = c.log_w - log_w;
            let area = c.rho.ln() + c.log_rho_step.ln() + c.rho_dot_step.ln();
            Some((
                *c,
                c.log_mass(),
                -chi2 / (2.0 * temperature) + prior + area,
                chi2,
                settled,
            ))
        })
        .collect();
    if rows.is_empty() {
        return None;
    }

    let moments = |pick: &dyn Fn(&(&Cell, f64, f64, f64, bool)) -> f64| {
        let max = rows.iter().map(pick).fold(f64::NEG_INFINITY, f64::max);
        let (mut sw, mut lr, mut lr2, mut rd, mut rd2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for row in &rows {
            let w = (pick(row) - max).exp();
            let (x, y) = (row.0.rho.ln(), row.0.rho_dot);
            sw += w;
            lr += w * x;
            lr2 += w * x * x;
            rd += w * y;
            rd2 += w * y * y;
        }
        let (mean_lr, mean_rd) = (lr / sw, rd / sw);
        [
            mean_lr,
            (lr2 / sw - mean_lr * mean_lr).max(0.0).sqrt(),
            mean_rd,
            (rd2 / sw - mean_rd * mean_rd).max(0.0).sqrt(),
        ]
    };
    let linear = moments(&|r| r.1);
    let iterated = moments(&|r| r.2);
    let shift = |m: usize, s: usize| {
        let sigma = linear[s].max(iterated[s]);
        if sigma > 0.0 {
            (iterated[m] - linear[m]).abs() / sigma
        } else {
            0.0
        }
    };

    Some(LinearModelCheck {
        shift_sigma: shift(0, 1).max(shift(2, 3)),
        unconverged: rows.iter().filter(|r| !r.4).count(),
        best_chi2: rows.iter().map(|r| r.3).fold(f64::INFINITY, f64::min),
    })
}

/// Largest change in log density, over the cells holding 99% of the posterior
/// mass, between a cell center and the points half a cell step away along each
/// axis.
///
/// `draw_samples` jitters draws by a Gaussian with sigma equal to half the cell
/// step along each axis independently.  When the log density changes by much
/// more than 1 over that distance the jitter does not follow the posterior: a
/// ridge that is thin across and tilted in `(log_rho, rho_dot)` is smeared
/// across, and the draws show one offset segment per cell.  ESS alone does not
/// detect this, since a well-spread but under-resolved grid can have high ESS.
///
/// Offset points that are physically invalid or fail to score are skipped.
/// Returns infinity if the SPK cannot be read.
fn max_half_step_change(
    cells: &[Cell],
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    temperature: f64,
) -> f64 {
    let Ok(spk_guard) = LOADED_SPK.try_read() else {
        return f64::INFINITY;
    };
    let spk: &SpkCollection = &spk_guard;
    let Ok(obs_helio) = spk.try_to_sun(attr.observer.clone()) else {
        return f64::INFINITY;
    };
    let sun_ssb = (
        attr.observer.pos - obs_helio.pos,
        attr.observer.vel - obs_helio.vel,
    );

    dominant_cells(cells)
        .par_iter()
        .map(|c| {
            let lr = c.rho.ln();
            let offsets = [
                ((lr + 0.5 * c.log_rho_step).exp(), c.rho_dot),
                ((lr - 0.5 * c.log_rho_step).exp(), c.rho_dot),
                (c.rho, c.rho_dot + 0.5 * c.rho_dot_step),
                (c.rho, c.rho_dot - 0.5 * c.rho_dot_step),
            ];
            offsets
                .iter()
                .filter_map(|&(rho, rho_dot)| {
                    score_point(spk, sorted_obs, attr, rho, rho_dot, temperature, sun_ssb)
                })
                .map(|(log_w, _, _)| (log_w - c.log_w).abs())
                .fold(0.0, f64::max)
        })
        .reduce(|| 0.0, f64::max)
}

/// Adaptively refine until ESS >= `TARGET_ESS` and [`max_half_step_change`] is
/// at most `MAX_HALF_STEP_LOG_CHANGE`, or `MAX_REFINE` rounds.
///
/// Each round takes the cells whose log density is within `REFINE_WINDOW` of the
/// peak and re-scores the region around them on a finer grid:
///
/// * Target steps are `1 / N_SUB` of the peak cell's steps.
/// * Rows are uniform in `log_rho`, spanning the window cells plus
///   `REFINE_MARGIN` cell extents.
/// * Each row's `rho_dot` range covers only the window cells within
///   `REFINE_MARGIN` extents of that row, plus the same margin, with as many
///   points as its width needs at the target step.  The region therefore
///   follows a posterior ridge that is tilted in `(log_rho, rho_dot)` instead of
///   enclosing it in a box.
/// * If the region would exceed `MAX_REFINE_CELLS`, both target steps are scaled
///   up together; if that leaves them no finer than the peak cell, refinement
///   stops.
/// * The new region is scored first; if that yields no cells the round stops
///   with the existing cells unchanged.  Otherwise every existing cell whose
///   extent overlaps the new region is removed before the new cells are added,
///   so cells never overlap and [`Cell::log_mass`] stays consistent across cell
///   sizes.
///
/// Re-scanning the region, rather than subdividing hot cells in place, also
/// resolves posterior mass inside coarse cells whose centers fell below the
/// weight floor; a ridge narrower than the coarse step often does.  The window is
/// wide so that the tails of a ridge are resolved along with its peak; cells
/// outside it carry negligible mass.
///
/// Returns the cells, their ESS, and their final [`max_half_step_change`].
fn refine(
    cells: Vec<Cell>,
    sorted_obs: &[AstrometricObservation],
    attr: &Attributable,
    temperature: f64,
) -> (Vec<Cell>, f64, f64) {
    const N_SUB: f64 = 5.0;
    const REFINE_WINDOW: f64 = 15.0;
    const REFINE_MARGIN: f64 = 2.0;
    const MAX_REFINE_CELLS: usize = N_RHO * N_RHO_DOT;

    /// Grid points needed to span `width` at `step`, at least 2.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "width and step are positive; the count is clamped below usize::MAX"
    )]
    fn n_points(width: f64, step: f64) -> usize {
        ((width / step).ceil().min(1e12) as usize + 1).max(2)
    }

    let mut cells = cells;
    let mut resolution = max_half_step_change(&cells, sorted_obs, attr, temperature);
    for _ in 0..MAX_REFINE {
        if ess(&cells) >= TARGET_ESS && resolution <= MAX_HALF_STEP_LOG_CHANGE {
            break;
        }
        let Some(peak) = cells.iter().max_by(|a, b| a.log_w.total_cmp(&b.log_w)) else {
            break;
        };
        let (peak_lr_step, peak_rd_step) = (peak.log_rho_step, peak.rho_dot_step);
        let max_lw = peak.log_w;

        // Window cells as (log_rho, log_rho_step, rho_dot, rho_dot_step), sorted by
        // log_rho so each row can find its neighbors by binary search.
        let mut window: Vec<(f64, f64, f64, f64)> = cells
            .iter()
            .filter(|c| c.log_w > max_lw - REFINE_WINDOW)
            .map(|c| (c.rho.ln(), c.log_rho_step, c.rho_dot, c.rho_dot_step))
            .collect();
        window.sort_by(|a, b| a.0.total_cmp(&b.0));
        let lr_lo = window
            .iter()
            .map(|w| w.0 - REFINE_MARGIN * w.1)
            .fold(f64::INFINITY, f64::min);
        let lr_hi = window
            .iter()
            .map(|w| w.0 + REFINE_MARGIN * w.1)
            .fold(f64::NEG_INFINITY, f64::max);
        let reach = REFINE_MARGIN * window.iter().map(|w| w.1).fold(0.0, f64::max);
        if lr_hi.partial_cmp(&lr_lo) != Some(std::cmp::Ordering::Greater) {
            break;
        }

        // `rho_dot` range covered at `log_rho = x` by the window cells near it.
        let range_at = |x: f64| {
            let start = window.partition_point(|w| w.0 < x - reach);
            let (lo, hi) = window[start..]
                .iter()
                .take_while(|w| w.0 <= x + reach)
                .filter(|w| (w.0 - x).abs() <= REFINE_MARGIN * w.1)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), w| {
                    (
                        lo.min(w.2 - REFINE_MARGIN * w.3),
                        hi.max(w.2 + REFINE_MARGIN * w.3),
                    )
                });
            (hi > lo).then_some((lo, hi))
        };

        // Find target steps whose region fits in MAX_REFINE_CELLS.
        let mut scale = 1.0;
        let mut plan = None;
        for _ in 0..8 {
            let lr_target = scale * peak_lr_step / N_SUB;
            let rd_target = scale * peak_rd_step / N_SUB;
            let n_lr = n_points(lr_hi - lr_lo, lr_target);
            if n_lr > MAX_REFINE_CELLS {
                scale *= 2.0;
                continue;
            }
            let lr_step = (lr_hi - lr_lo) / (n_lr - 1) as f64;
            let rows: Vec<Option<(f64, f64, usize)>> = (0..n_lr)
                .map(|i| {
                    range_at(lr_lo + i as f64 * lr_step)
                        .map(|(lo, hi)| (lo, hi, n_points(hi - lo, rd_target)))
                })
                .collect();
            let total: usize = rows.iter().flatten().map(|r| r.2).sum();
            if total <= MAX_REFINE_CELLS {
                plan = Some((lr_step, rows));
                break;
            }
            scale *= (total as f64 / MAX_REFINE_CELLS as f64).sqrt() * 1.05;
        }
        let Some((lr_step, rows)) = plan else {
            break;
        };
        if scale >= N_SUB || lr_step >= peak_lr_step {
            break;
        }

        // Score the new region before touching the existing cells.  An empty result
        // (SPK unreadable, or no valid orbit anywhere in the region) leaves the
        // current cells in place rather than deleting the region's mass.
        let new_cells = score_patch(
            sorted_obs,
            attr,
            (lr_lo, lr_hi),
            Some(&|i: usize| rows[i]),
            rows.len(),
            0,
            temperature,
        );
        if new_cells.is_empty() {
            break;
        }

        // Remove every cell whose extent overlaps a new row's covered extent: each
        // row spans half a step past its end points in both coordinates.
        #[allow(
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation,
            reason = "row indices are clamped to [0, rows.len()] before the cast"
        )]
        let row_index = |x: f64| ((x - lr_lo) / lr_step).clamp(0.0, rows.len() as f64) as usize;
        cells.retain(|c| {
            let lr = c.rho.ln();
            let half = f64::midpoint(c.log_rho_step, lr_step);
            let first = row_index((lr - half).max(lr_lo));
            let last = (row_index(lr + half) + 1).min(rows.len());
            !(first..last).any(|i| {
                let Some((lo, hi, n)) = rows[i] else {
                    return false;
                };
                let x = lr_lo + i as f64 * lr_step;
                let half_rd = 0.5 * (hi - lo) / (n - 1) as f64;
                (lr - x).abs() < half
                    && c.rho_dot + 0.5 * c.rho_dot_step > lo - half_rd
                    && c.rho_dot - 0.5 * c.rho_dot_step < hi + half_rd
            })
        });
        cells.extend(new_cells);

        let max_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);
        resolution = max_half_step_change(&cells, sorted_obs, attr, temperature);
    }
    let final_ess = ess(&cells);
    (cells, final_ess, resolution)
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

/// Weak diagonal regularizer for the 4x4 attributable information matrix.
/// sigma = 1e-4 rad (~20 arc-seconds) for both position and rate parameters.
const ATTR_REG_INV: f64 = 1.0 / (1e-4 * 1e-4);

/// Generate equally weighted orbital samples from the grid, in parallel.
///
/// Selects cells proportional to their posterior mass ([`Cell::log_mass`]),
/// applies Gaussian jitter (sigma = half cell width) within each cell's
/// `(log_rho, rho_dot)` extent, shifts the attributable by the cell's constrained
/// LS correction, and perturbs it by `N(0, Gamma_A)` using the stored tempered
/// information.  Each state is built at its emission epoch `t_ref - rho/c` and
/// propagated two-body to `t_ref`, so all draws share one epoch.
/// Fully parallelized via rayon; no `scout_score` calls at draw time.
fn draw_samples(
    cells: &[Cell],
    num_draws: usize,
    rng: &mut impl rand::Rng,
    attr: &Attributable,
) -> KeteResult<(Vec<Vec<f64>>, Vec<f64>)> {
    if cells.is_empty() || num_draws == 0 {
        return Ok((vec![], vec![]));
    }

    let max_lm = cells
        .iter()
        .map(Cell::log_mass)
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_lm.is_finite() {
        return Ok((vec![], vec![]));
    }
    let weights: Vec<f64> = cells
        .iter()
        .map(|c| (c.log_mass() - max_lm).exp())
        .collect();
    let sum_w: f64 = weights.iter().sum();
    if sum_w < 1e-300 {
        return Ok((vec![], vec![]));
    }

    // Build CDF for O(log n) weighted cell selection per draw.
    let mut cdf = Vec::with_capacity(cells.len());
    let mut acc = 0.0_f64;
    for &w in &weights {
        acc += w / sum_w;
        cdf.push(acc);
    }

    // Generate per-draw seeds from the caller's RNG so output is deterministic
    // and draws are independent across threads.
    let seeds: Vec<u64> = (0..num_draws).map(|_| rng.next_u64()).collect();

    let uniform = Uniform::new(0.0_f64, 1.0_f64).unwrap();

    let mut results: Vec<(Vec<f64>, f64)> = seeds
        .into_par_iter()
        .map(|seed| -> KeteResult<(Vec<f64>, f64)> {
            let mut local_rng = rand::rngs::SmallRng::seed_from_u64(seed);

            // Select a cell proportional to its posterior mass.
            let u = uniform.sample(&mut local_rng);
            let idx = cdf.partition_point(|&c| c < u).min(cells.len() - 1);
            let cell = &cells[idx];

            // Jitter within the cell with a Gaussian (sigma = half cell width).
            // Adjacent cells' Gaussian densities overlap and blend across their shared
            // boundaries, eliminating the polygonal staircase that uniform jitter
            // produces at the surviving-cell boundary.
            let dz_log = <rand_distr::StandardNormal as Distribution<f64>>::sample(
                &rand_distr::StandardNormal,
                &mut local_rng,
            );
            let dz_rd = <rand_distr::StandardNormal as Distribution<f64>>::sample(
                &rand_distr::StandardNormal,
                &mut local_rng,
            );
            let log_rho_j = cell.rho.ln() + 0.5 * cell.log_rho_step * dz_log;
            let rho_j = log_rho_j.exp();
            let rho_dot_j = cell.rho_dot + 0.5 * cell.rho_dot_step * dz_rd;

            // Center on the cell's constrained LS solution, then add N(0, Gamma_A)
            // noise using the cell's Fisher info.
            let mut p_attr = Attributable {
                alpha: attr.alpha + cell.attr_delta[0],
                delta: attr.delta + cell.attr_delta[1],
                alpha_dot: attr.alpha_dot + cell.attr_delta[2],
                delta_dot: attr.delta_dot + cell.attr_delta[3],
                t_ref: attr.t_ref,
                observer: attr.observer.clone(),
            };
            let mut n_reg = cell.attr_info.clone();
            n_reg[(0, 0)] += ATTR_REG_INV;
            n_reg[(1, 1)] += ATTR_REG_INV;
            n_reg[(2, 2)] += ATTR_REG_INV;
            n_reg[(3, 3)] += ATTR_REG_INV;
            if let Some(gamma_a) = n_reg.try_inverse()
                && let Some(chol) = gamma_a.cholesky()
            {
                let z = DVector::from_vec(
                    (0..4)
                        .map(|_| {
                            <rand_distr::StandardNormal as Distribution<f64>>::sample(
                                &rand_distr::StandardNormal,
                                &mut local_rng,
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                let d_attr = chol.l() * z;
                p_attr.alpha += d_attr[0];
                p_attr.delta += d_attr[1];
                p_attr.alpha_dot += d_attr[2];
                p_attr.delta_dot += d_attr[3];
            }

            // `state_from_rho` returns the state at the emission epoch, which
            // depends on rho; bring every draw to the common epoch `t_ref`.
            let emitted = state_from_rho(&p_attr, rho_j, rho_dot_j);
            let spk = LOADED_SPK.try_read()?;
            let helio = spk.try_to_sun(emitted)?;
            let at_ref = propagate_two_body(&helio, Time::from(attr.t_ref))?;
            let ps = spk.try_to_ssb(at_ref)?;
            Ok((
                vec![
                    ps.pos[0], ps.pos[1], ps.pos[2], ps.vel[0], ps.vel[1], ps.vel[2],
                ],
                cell.log_w,
            ))
        })
        .collect::<KeteResult<_>>()?;

    // Normalize so the maximum log-posterior across draws is 0.
    let lp_max = results
        .iter()
        .map(|(_, lp)| *lp)
        .fold(f64::NEG_INFINITY, f64::max);
    if lp_max.is_finite() {
        for (_, lp) in &mut results {
            *lp -= lp_max;
        }
    }

    Ok(results.into_iter().unzip())
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Generate orbit samples covering the admissible region from sparse observations.
///
/// Scans a 2-D grid over topocentric range `rho` and range-rate `rho_dot`, scores
/// each cell by the constrained attributable chi^2, and draws equally weighted
/// samples in proportion to each cell's posterior mass.  Designed for short arcs (hours to a few days)
/// where the posterior is a ridge or multi-modal.  For well-constrained arcs use
/// [`fit_orbit_mcmc`].
///
/// A short sliding window is swept across each observer's observations to find
/// the attributable epoch where the linear-motion approximation best fits.  Each
/// window holds observations from a single observer, identified by the
/// designation of its observer state (the observatory code for MPC data); all
/// observations from every observer are used for chi^2 scoring regardless of
/// which window is chosen.
///
/// [`fit_orbit_mcmc`]: crate::fit_orbit_mcmc
///
/// # Arguments
/// * `obs` -- At least 3 optical observations (any order, sorted internally).
/// * `num_draws` -- Number of orbit samples to return.
/// * `temperature` -- Likelihood temperature (1.0 = nominal). Higher values broaden coverage.
/// * `seed` -- RNG seed; identical inputs + seed -> identical draws.
///
/// # Errors
/// Returns an error if fewer than 3 optical observations are provided, no valid
/// cells survive scoring, or a draw cannot be propagated to the reference epoch.
pub fn fit_orbit_ranging(
    obs: &[AstrometricObservation],
    num_draws: usize,
    temperature: f64,
    seed: u64,
) -> KeteResult<RangingSamples> {
    const ATTR_WINDOW_DAYS: f64 = 0.1;
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

    // Attributables are built from one observer at a time.  The attributable is a
    // straight-line fit of RA/Dec against time paired with the secant velocity of
    // the observer, and both are only meaningful for a single observer: observers
    // at different locations see different parallax, and the secant between two
    // of them is not a velocity.  Observations are grouped by their observer's
    // designation (the observatory code for MPC data), preserving time order.
    let mut groups: Vec<(kete_core::desigs::Desig, Vec<AstrometricObservation>)> = Vec::new();
    for o in &sorted {
        let Ok((_, _, observer)) = o.as_optical() else {
            continue;
        };
        match groups.iter_mut().find(|(d, _)| *d == observer.desig) {
            Some((_, members)) => members.push(o.clone()),
            None => groups.push((observer.desig.clone(), vec![o.clone()])),
        }
    }

    // Select the window with the highest peak log-weight (lowest chi^2_min).
    // Selecting by ESS is wrong: a biased attributable produces a wider,
    // shallower posterior with high ESS while the correct one has a sharp peak.
    let mut best_cells: Vec<Cell> = Vec::new();
    let mut best_attr: Option<Attributable> = None;
    let mut best_peak_lw = f64::NEG_INFINITY;

    for (_, group) in &groups {
        if group.len() < 2 {
            continue;
        }
        let t0 = group[0].epoch().jd;
        let arc = group[group.len() - 1].epoch().jd - t0;

        // Widen the window if this observer's observations are more spread than
        // ATTR_WINDOW_DAYS.
        let min_obs_gap = group
            .windows(2)
            .filter_map(|w| {
                let dt = w[1].epoch().jd - w[0].epoch().jd;
                if dt > 1e-9 { Some(dt) } else { None }
            })
            .fold(f64::INFINITY, f64::min);
        if !min_obs_gap.is_finite() {
            continue;
        }
        let effective_window = ATTR_WINDOW_DAYS.max(min_obs_gap * 1.5);
        let effective_step = WINDOW_STEP_DAYS.max(effective_window / 2.0);

        let n_windows = if arc <= effective_window {
            1
        } else {
            // arc > effective_window > 0, effective_step > 0: quotient is always positive.
            let steps_ceil = ((arc - effective_window) / effective_step).ceil();
            #[allow(
                clippy::cast_sign_loss,
                clippy::cast_possible_truncation,
                reason = "steps_ceil > 0: arc > effective_window > 0"
            )]
            let n: usize = steps_ceil as usize;
            n + 1
        };

        for i in 0..n_windows {
            let w_start = t0 + i as f64 * effective_step;
            let w_end = w_start + effective_window;
            let window_obs: Vec<AstrometricObservation> = group
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
            let w_span = window_obs[window_obs.len() - 1].epoch().jd - window_obs[0].epoch().jd;
            if w_span < 1e-5 {
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
    }

    let attr = best_attr.ok_or_else(|| {
        Error::ValueError(
            "fit_orbit_ranging: no valid cells found for any window; an attributable \
             needs at least two observations from the same observer"
                .into(),
        )
    })?;
    let cells = best_cells;

    let max_lw = cells
        .iter()
        .map(|c| c.log_w)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut cells = cells;
    cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);

    let (cells, final_ess, resolution) = refine(cells, &sorted, &attr, temperature);

    let mut warnings = Vec::new();
    if final_ess < TARGET_ESS {
        warnings.push(format!(
            "ESS = {final_ess:.1} < {TARGET_ESS}; orbit space may be under-sampled. \
             Consider using fit_orbit_mcmc if the orbit is well-constrained."
        ));
    }
    if resolution > MAX_HALF_STEP_LOG_CHANGE {
        warnings.push(format!(
            "Grid under-resolved: log posterior changes by {resolution:.1} over half a \
             cell (limit {MAX_HALF_STEP_LOG_CHANGE}); draws may be spread across narrow \
             posterior structure."
        ));
    }
    if let Some(check) = linear_model_check(&cells, &sorted, &attr, temperature) {
        if check.unconverged > 0 || check.shift_sigma > MAX_LINEAR_MODEL_SHIFT_SIGMA {
            warnings.push(format!(
                "The linear attributable model does not describe these observations: \
                 re-scoring the dominant cells with iterated corrections moves the posterior \
                 by {:.1} sigma (limit {MAX_LINEAR_MODEL_SHIFT_SIGMA}), and {} of them did \
                 not converge. Cell weights and draws are unreliable; the arc is likely \
                 too long, or its observers too far apart, for ranging. Consider fit_orbit \
                 or fit_orbit_mcmc.",
                check.shift_sigma, check.unconverged
            ));
        }
        let n_optical = sorted.iter().filter(|o| o.as_optical().is_ok()).count();
        #[allow(clippy::cast_precision_loss, reason = "observation counts are small")]
        let dof = (2 * n_optical).saturating_sub(6) as f64;
        if dof > 0.0 && check.best_chi2 > dof + FIT_CHI2_SIGMAS * (2.0 * dof).sqrt() {
            warnings.push(format!(
                "The best orbit found has chi2 = {:.0} for {dof:.0} degrees of freedom; \
                 observation uncertainties may be too small, or the observations may not \
                 all belong to one object on a two-body orbit.",
                check.best_chi2
            ));
        }
    }
    let convergence_warning = (!warnings.is_empty()).then(|| warnings.join(" "));

    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
    let (draws, log_posterior) = draw_samples(&cells, num_draws, &mut rng, &attr)?;

    Ok(RangingSamples {
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
        synth_obs_offset(obj, epochs, sigma_rad, [0.0; 3], &Desig::Empty)
    }

    /// Like `synth_obs`, but the observer is displaced by a constant `offset` (AU)
    /// from the analytic Earth-like path and carries designation `desig`.
    fn synth_obs_offset(
        obj: &State<Equatorial, SSB>,
        epochs: &[f64],
        sigma_rad: f64,
        offset: [f64; 3],
        desig: &Desig,
    ) -> Vec<AstrometricObservation> {
        let spk = LOADED_SPK.try_read().unwrap();
        let obj_sun = spk.try_to_sun(obj.clone()).unwrap();
        let v_earth = (GMS / 1.0_f64).sqrt();
        let obl = 23.44_f64.to_radians();
        epochs
            .iter()
            .filter_map(|&jd| {
                let angle = (jd - 2_460_000.5) / 365.25 * std::f64::consts::TAU;
                let obs_pos = [
                    angle.cos() + offset[0],
                    angle.sin() * obl.cos() + offset[1],
                    angle.sin() * obl.sin() + offset[2],
                ];
                let obs_vel = [
                    -v_earth * angle.sin(),
                    v_earth * angle.cos() * obl.cos(),
                    v_earth * angle.cos() * obl.sin(),
                ];
                let mut observer = make_ssb_state(obs_pos, obs_vel, jd);
                observer.desig = desig.clone();
                let obj_at = propagate_two_body(&obj_sun, Time::<TDB>::new(jd)).ok()?;
                let obs_sun_pos = spk.try_to_sun(observer.clone()).ok()?.pos;
                let obj_lt_sun = light_time_correct(&obj_at, &obs_sun_pos).ok()?;
                let obj_lt_ssb = spk.try_to_ssb(obj_lt_sun).ok()?;
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

    /// Synthetic observations from an observer in a circular low Earth orbit.
    ///
    /// The observer follows the same analytic Earth-like path `synth_obs` uses,
    /// plus a 6900 km polar circular orbit with a 95 minute period -- WISE's
    /// geometry.  Returns the observations along with the observer's true
    /// instantaneous states, so callers can compare the attributable's stored
    /// observer velocity with the instantaneous one.
    fn synth_obs_leo(
        obj: &State<Equatorial, SSB>,
        epochs: &[f64],
        sigma_rad: f64,
    ) -> (Vec<AstrometricObservation>, Vec<State<Equatorial, SSB>>) {
        let spk = LOADED_SPK.try_read().unwrap();
        let obj_sun = spk.try_to_sun(obj.clone()).unwrap();
        let v_earth = (GMS / 1.0_f64).sqrt();
        let obl = 23.44_f64.to_radians();

        let r_leo = 6900.0 / kete_core::constants::AU_KM;
        let period = 95.0 / 1440.0;
        let omega = std::f64::consts::TAU / period;

        let mut obs = Vec::new();
        let mut observers = Vec::new();
        for &jd in epochs {
            let angle = (jd - 2_460_000.5) / 365.25 * std::f64::consts::TAU;
            let theta = omega * (jd - epochs[0]);
            // Polar orbit in the x-z plane, so the wobble enters both RA and Dec.
            let obs_pos = [
                angle.cos() + r_leo * theta.cos(),
                angle.sin() * obl.cos(),
                angle.sin() * obl.sin() + r_leo * theta.sin(),
            ];
            let obs_vel = [
                -v_earth * angle.sin() - r_leo * omega * theta.sin(),
                v_earth * angle.cos() * obl.cos(),
                v_earth * angle.cos() * obl.sin() + r_leo * omega * theta.cos(),
            ];
            let observer = make_ssb_state(obs_pos, obs_vel, jd);
            let Ok(obj_at) = propagate_two_body(&obj_sun, Time::<TDB>::new(jd)) else {
                continue;
            };
            let Ok(obs_sun_pos) = spk.try_to_sun(observer.clone()).map(|s| s.pos) else {
                continue;
            };
            let Ok(obj_lt_sun) = light_time_correct(&obj_at, &obs_sun_pos) else {
                continue;
            };
            let Ok(obj_lt_ssb) = spk.try_to_ssb(obj_lt_sun) else {
                continue;
            };
            let (ra, dec) = (obj_lt_ssb.pos - observer.pos).to_ra_dec();
            obs.push(AstrometricObservation::Optical {
                observer: observer.clone(),
                ra,
                dec,
                sigma_ra: sigma_rad,
                sigma_dec: sigma_rad,
                sigma_corr: 0.0,
                time_sigma: 0.0,
                is_occultation: false,
                band: Band::Unknown([0; 8]),
                mag: f64::NAN,
            });
            observers.push(observer);
        }
        (obs, observers)
    }

    /// The attributable fits a straight line across the arc, so the observer
    /// velocity it stores must be the arc's secant velocity.  For an observer in
    /// low Earth orbit the instantaneous velocity differs from that by a
    /// substantial fraction of Earth's orbital velocity, and pairing it with the
    /// averaged on-sky rates corrupts every reconstructed orbit.
    #[test]
    fn attributable_observer_velocity_is_the_arc_secant() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        // 9 observations over 20 hours -- a WISE tracklet, about 12.6 revolutions.
        let epochs: Vec<f64> = (0..9)
            .map(|i| 2_460_000.5 + f64::from(i) * (20.0 / 24.0) / 8.0)
            .collect();
        let (obs, observers) = synth_obs_leo(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        assert_eq!(obs.len(), 9);

        let attr = compute_attributable(&obs).expect("attributable must succeed");

        // The stored velocity is the secant across the arc, to round-off.
        let dt = epochs[8] - epochs[0];
        let secant = (observers[8].pos - observers[0].pos) / dt;
        let err = (attr.observer.vel - secant).norm();
        assert!(
            err < 1e-12,
            "stored velocity is not the arc secant: error {err:.3e} AU/day"
        );

        // The position is still the first observation's, untouched.
        assert!((attr.observer.pos - observers[0].pos).norm() < 1e-15);

        // The instantaneous velocity differs from it by the orbital motion the
        // straight-line fit averaged away: 6900 km at a 95 minute period.
        let leo_speed =
            std::f64::consts::TAU * (6900.0 / kete_core::constants::AU_KM) / (95.0 / 1440.0);
        let offset = (observers[0].vel - attr.observer.vel).norm();
        assert!(
            (offset - leo_speed).abs() < 0.2 * leo_speed,
            "expected the secant to differ from the instantaneous velocity by about \
             {leo_speed:.5} AU/day, got {offset:.5}"
        );

        // Reconstructing the object's velocity from the averaged rates works with
        // the secant velocity and fails with the instantaneous one.  Both use the
        // same (rho, rho_dot), so the only difference is the pairing.
        let rho_of = |k: usize| (obj_at_epoch(&obj, epochs[k]).pos - observers[k].pos).norm();
        let rho = rho_of(0);
        let rho_dot = (rho_of(8) - rho_of(0)) / dt;

        let fixed = state_from_rho(&attr, rho, rho_dot);
        let mut attr_instantaneous = attr.clone();
        attr_instantaneous.observer.vel = observers[0].vel;
        let instantaneous = state_from_rho(&attr_instantaneous, rho, rho_dot);

        let err_fixed = (fixed.vel - obj.vel).norm();
        let err_instantaneous = (instantaneous.vel - obj.vel).norm();
        assert!(
            err_fixed < 0.1 * leo_speed,
            "secant pairing should recover the velocity: error {err_fixed:.5} AU/day"
        );
        assert!(
            err_instantaneous > 5.0 * err_fixed,
            "instantaneous pairing should be much worse: {err_instantaneous:.5} vs \
             {err_fixed:.5} AU/day"
        );
    }

    /// Drawn states are centered on each cell's constrained LS solution, not on
    /// the raw attributable, and are reported at `t_ref` rather than at their
    /// emission epoch.  For a LEO observer the raw attributable's reference
    /// position is thousands of km off the arc-averaged one, which shows up as a
    /// transverse offset of the drawn state; the correction removes it.  At 2 AU
    /// the object moves tens of thousands of km during the light time, which the
    /// propagation to `t_ref` removes.
    #[test]
    fn draws_are_centered_on_constrained_ls_solution() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        let epochs: Vec<f64> = (0..9)
            .map(|i| 2_460_000.5 + f64::from(i) * (20.0 / 24.0) / 8.0)
            .collect();
        let (obs, observers) = synth_obs_leo(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let attr = compute_attributable(&obs).expect("attributable must succeed");

        // Cell at the true (rho, rho_dot), light-time corrected range.
        let rho_of = |k: usize| {
            let obs_pos = observers[k].pos;
            let mut rho = (obj_at_epoch(&obj, epochs[k]).pos - obs_pos).norm();
            for _ in 0..3 {
                let t_emit = epochs[k] - rho * kete_core::constants::C_AU_PER_DAY_INV;
                rho = (obj_at_epoch(&obj, t_emit).pos - obs_pos).norm();
            }
            rho
        };
        let rho = rho_of(0);
        let rho_dot = (rho_of(8) - rho_of(0)) / (epochs[8] - epochs[0]);
        let state = state_from_rho(&attr, rho, rho_dot);
        let (log_w, _, attr_delta) = scout_score(
            &LOADED_SPK.try_read().unwrap(),
            &state,
            &obs,
            attr.t_ref,
            1.0,
        )
        .expect("scout_score must succeed");

        // Negligible cell extent and an effectively infinite Fisher information so
        // the draw is deterministic: attributable + correction, nothing else.
        let cell = Cell {
            log_w,
            attr_info: DMatrix::<f64>::identity(4, 4) * 1e30,
            attr_delta,
            rho,
            rho_dot,
            log_rho_step: 1e-15,
            rho_dot_step: 1e-15,
        };
        let mut rng = rand::rngs::SmallRng::seed_from_u64(3);
        let (draws, _) =
            draw_samples(std::slice::from_ref(&cell), 4, &mut rng, &attr).expect("draws");
        assert_eq!(draws.len(), 4);

        let t_emit = attr.t_ref - rho * kete_core::constants::C_AU_PER_DAY_INV;
        let truth_emit = obj_at_epoch(&obj, t_emit);
        let truth_ref = obj_at_epoch(&obj, attr.t_ref);
        let los = (state.pos - attr.observer.pos).normalize();
        let transverse = |dp: Vector<Equatorial>| {
            let radial = dp.dot(&los);
            (dp - los * radial).norm() * kete_core::constants::AU_KM
        };

        let uncorrected = transverse(state.pos - truth_emit.pos);
        assert!(
            uncorrected > 1000.0,
            "raw attributable should be off by thousands of km: {uncorrected:.1} km"
        );
        for d in &draws {
            let pos = Vector::<Equatorial>::new([d[0], d[1], d[2]]);
            let drawn = transverse(pos - truth_ref.pos);
            assert!(
                drawn < 50.0,
                "drawn state transverse error {drawn:.1} km (uncorrected {uncorrected:.1} km)"
            );
            let pos_err = (pos - truth_ref.pos).norm() * kete_core::constants::AU_KM;
            let emit_err = (pos - truth_emit.pos).norm() * kete_core::constants::AU_KM;
            assert!(
                pos_err < 100.0 && emit_err > 10_000.0,
                "draw must be at t_ref: error {pos_err:.1} km vs truth at t_ref, \
                 {emit_err:.1} km vs truth at emission"
            );
            let vel_err = (Vector::<Equatorial>::new([d[3], d[4], d[5]]) - truth_ref.vel).norm();
            assert!(vel_err < 5e-5, "drawn velocity error {vel_err:.3e} AU/day");
        }
    }

    /// Scores a synthetic arc on the full coarse grid, dropping cells below the
    /// weight floor as `fit_orbit_ranging` does.
    fn coarse_cells(
        n_obs: u32,
        span_days: f64,
        temperature: f64,
    ) -> (Vec<AstrometricObservation>, Attributable, Vec<Cell>) {
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        let epochs: Vec<f64> = (0..n_obs)
            .map(|i| 2_460_000.5 + f64::from(i) * span_days / f64::from(n_obs - 1))
            .collect();
        let obs = synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0);
        let attr = compute_attributable(&obs).expect("attributable must succeed");
        let mut cells = score_patch(
            &obs,
            &attr,
            (LOG_RHO_MIN, LOG_RHO_MAX),
            None,
            N_RHO,
            N_RHO_DOT,
            temperature,
        );
        let max_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);
        (obs, attr, cells)
    }

    /// Mass-weighted mean of `ln(rho)` over a set of cells.
    fn mean_log_rho(cells: &[Cell]) -> f64 {
        let max_lm = cells
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sw, mut s) = (0.0, 0.0);
        for c in cells {
            let w = (c.log_mass() - max_lm).exp();
            sw += w;
            s += w * c.rho.ln();
        }
        s / sw
    }

    /// Rounded `(ln rho, rho_dot)` key identifying a cell center.
    #[allow(clippy::cast_possible_truncation, reason = "test-only rounding key")]
    fn cell_key(c: &Cell) -> (i64, i64) {
        (
            (c.rho.ln() * 1e8).round() as i64,
            (c.rho_dot * 1e11).round() as i64,
        )
    }

    /// No coarse row collapses to zero `rho_dot` width, so no cell is repeated.
    #[test]
    fn coarse_grid_has_no_collapsed_rows() {
        ensure_test_spk();
        let (_, _, cells) = coarse_cells(4, 1.0 / 24.0, 10.0);
        assert!(cells.iter().all(|c| c.rho_dot_step > 0.0));
        let unique: std::collections::HashSet<_> = cells.iter().map(cell_key).collect();
        assert_eq!(unique.len(), cells.len(), "coarse grid repeats cells");
    }

    /// The per-row `rho_dot` step varies across the coarse grid, so cell masses
    /// must include cell area.  A fine grid with one uniform `rho_dot` step needs
    /// no area correction and serves as the reference.
    #[test]
    fn cell_mass_matches_uniform_grid_reference() {
        ensure_test_spk();
        let (obs, attr, coarse) = coarse_cells(4, 1.0 / 24.0, 10.0);

        let max_lm = coarse
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let significant: Vec<&Cell> = coarse
            .iter()
            .filter(|c| c.log_mass() > max_lm - 30.0)
            .collect();
        let step_min = significant
            .iter()
            .map(|c| c.rho_dot_step)
            .fold(f64::INFINITY, f64::min);
        let step_max = significant
            .iter()
            .map(|c| c.rho_dot_step)
            .fold(0.0, f64::max);
        assert!(
            step_max > 2.0 * step_min,
            "test needs varying row widths: [{step_min:.3e}, {step_max:.3e}]"
        );
        let fold = |f: fn(f64, f64) -> f64, init: f64, g: &dyn Fn(&Cell) -> f64| {
            significant.iter().map(|c| g(c)).fold(init, f)
        };
        let lr_lo = fold(f64::min, f64::INFINITY, &|c| c.rho.ln());
        let lr_hi = fold(f64::max, f64::NEG_INFINITY, &|c| c.rho.ln());
        let rd_lo = fold(f64::min, f64::INFINITY, &|c| c.rho_dot);
        let rd_hi = fold(f64::max, f64::NEG_INFINITY, &|c| c.rho_dot);
        let pad = 0.1 * (rd_hi - rd_lo);
        let fine = score_patch(
            &obs,
            &attr,
            (lr_lo - 0.05, lr_hi + 0.05),
            Some(&|_| Some((rd_lo - pad, rd_hi + pad, 1500))),
            200,
            1500,
            10.0,
        );

        let reference = mean_log_rho(&fine).exp();
        let coarse_mean = mean_log_rho(&coarse).exp();
        assert!(
            (coarse_mean / reference - 1.0).abs() < 0.02,
            "coarse posterior mean rho {coarse_mean:.4} AU vs uniform-grid reference \
             {reference:.4} AU"
        );
    }

    /// Mass-weighted mean and standard deviation of `ln(rho)` and `rho_dot`,
    /// including each cell's own extent.
    fn cell_moments(cells: &[Cell]) -> [f64; 4] {
        let max_lm = cells
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sum_w, mut lr_sum, mut lr_sq, mut rd_sum, mut rd_sq) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for c in cells {
            let weight = (c.log_mass() - max_lm).exp();
            let (lr, rd) = (c.rho.ln(), c.rho_dot);
            sum_w += weight;
            lr_sum += weight * lr;
            lr_sq += weight * (lr * lr + c.log_rho_step.powi(2) / 12.0);
            rd_sum += weight * rd;
            rd_sq += weight * (rd * rd + c.rho_dot_step.powi(2) / 12.0);
        }
        let (lr_mean, rd_mean) = (lr_sum / sum_w, rd_sum / sum_w);
        [
            lr_mean,
            (lr_sq / sum_w - lr_mean * lr_mean).sqrt(),
            rd_mean,
            (rd_sq / sum_w - rd_mean * rd_mean).sqrt(),
        ]
    }

    /// Log of the total cell mass.
    fn total_log_mass(cells: &[Cell]) -> f64 {
        let max_lm = cells
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        max_lm
            + cells
                .iter()
                .map(|c| (c.log_mass() - max_lm).exp())
                .sum::<f64>()
                .ln()
    }

    /// Refinement never repeats cells and reproduces a converged uniform-grid
    /// reference.  This arc's posterior ridge is narrower than the coarse step and
    /// has tails well past the peak, so neither keeping parent cells nor
    /// subdividing only the peak cells reproduces it.
    #[test]
    fn refinement_matches_uniform_grid_reference() {
        ensure_test_spk();
        let (obs, attr, cells) = coarse_cells(12, 10.0, 1.0);
        let initial = ess(&cells);
        assert!(
            initial < TARGET_ESS,
            "test needs refinement: ESS {initial:.1}"
        );

        let pad = |f: fn(f64, f64) -> f64, init: f64, g: &dyn Fn(&Cell) -> f64| {
            cells.iter().map(g).fold(init, f)
        };
        let lr_lo = pad(f64::min, f64::INFINITY, &|c| {
            c.rho.ln() - 8.0 * c.log_rho_step
        });
        let lr_hi = pad(f64::max, f64::NEG_INFINITY, &|c| {
            c.rho.ln() + 8.0 * c.log_rho_step
        });
        let rd_lo = pad(f64::min, f64::INFINITY, &|c| {
            c.rho_dot - 8.0 * c.rho_dot_step
        });
        let rd_hi = pad(f64::max, f64::NEG_INFINITY, &|c| {
            c.rho_dot + 8.0 * c.rho_dot_step
        });
        let reference = score_patch(
            &obs,
            &attr,
            (lr_lo, lr_hi),
            Some(&|_| Some((rd_lo, rd_hi, 500))),
            500,
            500,
            1.0,
        );

        let (refined, final_ess, _) = refine(cells, &obs, &attr, 1.0);
        assert!(final_ess >= TARGET_ESS, "ESS {final_ess:.1}");
        let unique: std::collections::HashSet<_> = refined.iter().map(cell_key).collect();
        assert_eq!(unique.len(), refined.len(), "refinement repeats cells");

        let [ra, rsa, rb, rsb] = cell_moments(&reference);
        let [fa, fsa, fb, fsb] = cell_moments(&refined);
        assert!(
            ((fa - ra) / rsa).abs() < 0.05 && (fsa / rsa - 1.0).abs() < 0.05,
            "ln rho: refined {fa:.5} +- {fsa:.5}, reference {ra:.5} +- {rsa:.5}"
        );
        assert!(
            ((fb - rb) / rsb).abs() < 0.05 && (fsb / rsb - 1.0).abs() < 0.05,
            "rho_dot: refined {fb:.4e} +- {fsb:.3e}, reference {rb:.4e} +- {rsb:.3e}"
        );
        let mass_diff = total_log_mass(&refined) - total_log_mass(&reference);
        assert!(mass_diff.abs() < 0.01, "log mass differs by {mass_diff:.4}");
    }

    /// Temperature scales the attributable information by `1 / T`, matching the
    /// tempered likelihood used for the cell weights.
    #[test]
    fn temperature_scales_attributable_information() {
        ensure_test_spk();
        let (obs, attr, cells) = coarse_cells(6, 2.0, 1.0);
        let best = cells
            .iter()
            .max_by(|a, b| a.log_w.total_cmp(&b.log_w))
            .expect("cells");
        let state = state_from_rho(&attr, best.rho, best.rho_dot);
        let (lw1, n1, d1) = scout_score(
            &LOADED_SPK.try_read().unwrap(),
            &state,
            &obs,
            attr.t_ref,
            1.0,
        )
        .expect("score");
        let (lw4, n4, d4) = scout_score(
            &LOADED_SPK.try_read().unwrap(),
            &state,
            &obs,
            attr.t_ref,
            4.0,
        )
        .expect("score");
        assert!((lw1 - 4.0 * lw4).abs() <= 1e-9 * lw1.abs().max(1.0));
        assert!((&n1 - &n4 * 4.0).norm() <= 1e-9 * n1.norm());
        assert_eq!(d1, d4, "the LS correction does not depend on temperature");
    }

    /// Nine NEOCP observations of `P12pZsW` (F51 and H21, 2026-09-13 to 09-16).
    ///
    /// Observer states are SSB Equatorial, computed by kete from the observatory
    /// codes; sigmas are the values kete assigns to these MPC lines.
    fn p12pzsw_obs() -> Vec<AstrometricObservation> {
        // (jd TDB, ra, dec, sigma_ra, sigma_dec) in radians, observer pos, observer vel.
        let data: [(f64, f64, f64, f64, f64, [f64; 3], [f64; 3]); 9] = [
            (
                2461296.9431087407,
                0.4022988289170059,
                0.15509868397847631,
                4.6796254786817753e-07,
                3.807272603516711e-07,
                [
                    0.9905260509140534,
                    -0.16036728563473468,
                    -0.06939872488439142,
                ],
                [
                    0.0026461486718229493,
                    0.015759125806241572,
                    0.006722942651140843,
                ],
            ),
            (
                2461296.954351741,
                0.4023551885074349,
                0.1550928662143028,
                4.679625478681775e-07,
                3.807272603516711e-07,
                [
                    0.9905556833237148,
                    -0.1601900967059782,
                    -0.06932313736048522,
                ],
                [
                    0.002625104430370301,
                    0.015760529983108645,
                    0.0067232056665094105,
                ],
            ),
            (
                2461296.965897741,
                0.40241489331226354,
                0.15508510919540514,
                4.679625478681774e-07,
                3.807272603516711e-07,
                [
                    0.9905858678786339,
                    -0.16000812360445477,
                    -0.06924550966992793,
                ],
                [
                    0.0026034738042102286,
                    0.015760658715754012,
                    0.006723475588729995,
                ],
            ),
            (
                2461296.9775367407,
                0.40247452539504003,
                0.15507880661755058,
                4.6796254786817753e-07,
                3.807272603516711e-07,
                [
                    0.9906160431448335,
                    -0.1598246910632488,
                    -0.06916725355665505,
                ],
                [
                    0.0025817465180420227,
                    0.015759443735856006,
                    0.006723747244875391,
                ],
            ),
            (
                2461299.7865797407,
                0.41896628694123883,
                0.15277254793651276,
                9.818474745373877e-07,
                8.465514645945206e-07,
                [
                    0.9968078850272505,
                    -0.11609736233379504,
                    -0.05020006999822096,
                ],
                [
                    0.0017627660825640919,
                    0.015821229088739997,
                    0.006769543409903565,
                ],
            ),
            (
                2461299.789466741,
                0.41898155857219377,
                0.15276769979970134,
                9.818474745373877e-07,
                8.465514645945206e-07,
                [
                    0.9968129675533154,
                    -0.11605168727094252,
                    -0.05018052625416701,
                ],
                [
                    0.001758209632065055,
                    0.015820642297893088,
                    0.006769591886790371,
                ],
            ),
            (
                2461299.792353741,
                0.4189975574236704,
                0.1527652757312961,
                9.818474745373877e-07,
                8.465514645945206e-07,
                [
                    0.9968180369424512,
                    -0.11600601400704244,
                    -0.05016098237339583,
                ],
                [
                    0.001753666109779852,
                    0.015819988010832474,
                    0.006769640315202754,
                ],
            ),
            (
                2461297.067890741,
                0.4029364558704012,
                0.15502159860317957,
                4.6796254786817753e-07,
                3.807272603516711e-07,
                [0.9908420482275389, -0.158402599540279, -0.06855964200824495],
                [
                    0.0024263488745346925,
                    0.015706442140421103,
                    0.006725813270044478,
                ],
            ),
            (
                2461297.068706741,
                0.4029401646950618,
                0.15502062897581748,
                4.679625478681775e-07,
                3.807272603516711e-07,
                [
                    0.9908440276293031,
                    -0.15838978340687454,
                    -0.06855415373615226,
                ],
                [
                    0.002425125807669756,
                    0.015705642565053482,
                    0.006725831390497616,
                ],
            ),
        ];
        let mut obs: Vec<AstrometricObservation> = data
            .iter()
            .map(
                |&(jd, ra, dec, sigma_ra, sigma_dec, pos, vel)| AstrometricObservation::Optical {
                    observer: make_ssb_state(pos, vel, jd),
                    ra,
                    dec,
                    sigma_ra,
                    sigma_dec,
                    sigma_corr: 0.0,
                    time_sigma: 0.5,
                    is_occultation: false,
                    band: Band::Unknown([0; 8]),
                    mag: f64::NAN,
                },
            )
            .collect();
        obs.sort_by(|a, b| a.epoch().jd.total_cmp(&b.epoch().jd));
        obs
    }

    /// Spread of `rho_dot` about the line `a + b ln(rho)`, as the sampler draws it:
    /// cell mass weights plus each cell's Gaussian jitter.
    fn cross_ridge_sigma(cells: &[Cell], a: f64, b: f64) -> f64 {
        let max_lm = cells
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sum_w, mut sum_r, mut sum_r2) = (0.0, 0.0, 0.0);
        for c in cells {
            let weight = (c.log_mass() - max_lm).exp();
            let resid = c.rho_dot - (a + b * c.rho.ln());
            let jitter_var = (0.5 * c.rho_dot_step).powi(2) + (b * 0.5 * c.log_rho_step).powi(2);
            sum_w += weight;
            sum_r += weight * resid;
            sum_r2 += weight * (resid * resid + jitter_var);
        }
        (sum_r2 / sum_w - (sum_r / sum_w).powi(2)).sqrt()
    }

    /// A tracklet whose posterior is a thin ridge tilted in `(log_rho, rho_dot)`.
    /// The coarse grid has ample ESS, so ESS alone never triggers refinement, but
    /// the cells are wider than the ridge and the per-axis jitter smears draws
    /// across it, which showed as banding in the orbit cloud.  Refinement must be
    /// triggered by resolution and must reproduce a dense uniform-grid reference.
    #[test]
    fn refinement_resolves_thin_tilted_ridge() {
        const TEMPERATURE: f64 = 10.0;
        ensure_test_spk();
        let obs = p12pzsw_obs();
        let first_night: Vec<AstrometricObservation> = obs
            .iter()
            .filter(|o| o.epoch().jd < obs[0].epoch().jd + 0.1)
            .cloned()
            .collect();
        let attr = compute_attributable(&first_night).expect("attributable");
        let mut cells = score_patch(
            &obs,
            &attr,
            (LOG_RHO_MIN, LOG_RHO_MAX),
            None,
            N_RHO,
            N_RHO_DOT,
            TEMPERATURE,
        );
        let max_lw = cells
            .iter()
            .map(|c| c.log_w)
            .fold(f64::NEG_INFINITY, f64::max);
        cells.retain(|c| c.log_w > max_lw - LOG_W_FLOOR);
        let coarse_resolution = max_half_step_change(&cells, &obs, &attr, TEMPERATURE);
        assert!(ess(&cells) >= TARGET_ESS, "test needs high coarse ESS");
        assert!(
            coarse_resolution > MAX_HALF_STEP_LOG_CHANGE,
            "test needs an under-resolved coarse grid: log change {coarse_resolution:.2}"
        );

        // Dense uniform reference over the significant region, and the ridge line
        // through it.
        let max_lm = cells
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let significant: Vec<&Cell> = cells
            .iter()
            .filter(|c| c.log_mass() > max_lm - 15.0)
            .collect();
        let lr_lo = significant
            .iter()
            .map(|c| c.rho.ln())
            .fold(f64::INFINITY, f64::min)
            - 0.15;
        let lr_hi = significant
            .iter()
            .map(|c| c.rho.ln())
            .fold(f64::NEG_INFINITY, f64::max)
            + 0.15;
        let rd_lo = significant
            .iter()
            .map(|c| c.rho_dot)
            .fold(f64::INFINITY, f64::min)
            - 3e-3;
        let rd_hi = significant
            .iter()
            .map(|c| c.rho_dot)
            .fold(f64::NEG_INFINITY, f64::max)
            + 3e-3;
        let reference = score_patch(
            &obs,
            &attr,
            (lr_lo, lr_hi),
            Some(&|_| Some((rd_lo, rd_hi, 800))),
            800,
            0,
            TEMPERATURE,
        );
        let ref_lm = reference
            .iter()
            .map(Cell::log_mass)
            .fold(f64::NEG_INFINITY, f64::max);
        let (mut sw, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for c in &reference {
            let weight = (c.log_mass() - ref_lm).exp();
            let (lr, rd) = (c.rho.ln(), c.rho_dot);
            sw += weight;
            sx += weight * lr;
            sy += weight * rd;
            sxx += weight * lr * lr;
            sxy += weight * lr * rd;
        }
        let slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx);
        let intercept = (sy - slope * sx) / sw;
        let ref_sigma = cross_ridge_sigma(&reference, intercept, slope);

        let coarse_ratio = cross_ridge_sigma(&cells, intercept, slope) / ref_sigma;
        assert!(
            coarse_ratio > 1.15,
            "coarse grid should smear across the ridge: ratio {coarse_ratio:.3}"
        );

        let (refined, _, resolution) = refine(cells, &obs, &attr, TEMPERATURE);
        assert!(
            resolution <= MAX_HALF_STEP_LOG_CHANGE,
            "refinement left a log change of {resolution:.2} over half a cell"
        );
        let refined_ratio = cross_ridge_sigma(&refined, intercept, slope) / ref_sigma;
        assert!(
            (refined_ratio - 1.0).abs() < 0.03,
            "refined cross-ridge spread {refined_ratio:.3} of the reference"
        );
    }

    /// Synthetic arc of `n` noise-free observations spread over `span_days`.
    fn arc_obs(n: u32, span_days: f64) -> Vec<AstrometricObservation> {
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        let epochs: Vec<f64> = (0..n)
            .map(|i| 2_460_000.5 + f64::from(i) * span_days / f64::from(n - 1))
            .collect();
        synth_obs(&obj, &epochs, 1.0_f64.to_radians() / 3600.0)
    }

    /// A multi-night arc the linear attributable model describes raises no warning.
    #[test]
    fn ranging_no_warning_for_short_multi_night_arc() {
        ensure_test_spk();
        let samples = fit_orbit_ranging(&arc_obs(6, 2.0), 100, 1.0, 1).expect("ranging");
        assert_eq!(samples.convergence_warning, None);
    }

    /// Over a 20-day arc the linear attributable model mis-scores the cells enough
    /// to move the posterior, which must be reported.
    #[test]
    fn ranging_warns_when_linear_model_fails() {
        ensure_test_spk();
        let samples = fit_orbit_ranging(&arc_obs(20, 20.0), 100, 1.0, 1).expect("ranging");
        let warning = samples.convergence_warning.unwrap_or_default();
        assert!(
            warning.contains("linear attributable model"),
            "expected a linear model warning, got {warning:?}"
        );
    }

    /// Observations no orbit fits within their uncertainties are reported.
    #[test]
    fn ranging_warns_on_poor_fit() {
        ensure_test_spk();
        let mut obs = arc_obs(6, 2.0);
        if let AstrometricObservation::Optical { dec, .. } = &mut obs[3] {
            *dec += 60.0_f64.to_radians() / 3600.0;
        }
        let samples = fit_orbit_ranging(&obs, 100, 1.0, 1).expect("ranging");
        let warning = samples.convergence_warning.unwrap_or_default();
        assert!(
            warning.contains("chi2 ="),
            "expected a fit quality warning, got {warning:?}"
        );
    }

    /// Position of the object at `jd`, two-body.
    fn obj_at_epoch(obj: &State<Equatorial, SSB>, jd: f64) -> State<Equatorial, SSB> {
        let spk = LOADED_SPK.try_read().unwrap();
        let obj_sun = spk.try_to_sun(obj.clone()).unwrap();
        let moved = propagate_two_body(&obj_sun, Time::<TDB>::new(jd)).unwrap();
        spk.try_to_ssb(moved).unwrap()
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
        let (log_w, _, _) = scout_score(
            &LOADED_SPK.try_read().unwrap(),
            &obj,
            &obs,
            2_460_000.5,
            1.0,
        )
        .expect("scout_score must succeed");
        assert!(
            log_w > -0.5,
            "log_w = {log_w:.6} for true state; expected near-zero curvature"
        );
    }

    /// Observations from two observers about 1 AU apart, interleaved so that every
    /// short window contains both.  Attributables must be built per observer: a
    /// straight-line fit and secant velocity across the two observers describe no
    /// real motion and reject every candidate orbit.
    #[test]
    fn attributable_windows_use_one_observer() {
        ensure_test_spk();
        let r = 2.0_f64;
        let v = (GMS / r).sqrt();
        let obl = 23.44_f64.to_radians();
        let obj = make_ssb_state(
            [0.0, r, 0.0],
            [-v * obl.cos(), 0.0, v * obl.sin()],
            2_460_000.5,
        );
        let sigma = 1.0_f64.to_radians() / 3600.0;
        let near: Vec<f64> = (0..4).map(|i| 2_460_000.5 + f64::from(i) * 0.02).collect();
        let far: Vec<f64> = (0..4).map(|i| 2_460_000.51 + f64::from(i) * 0.02).collect();
        let mut obs = synth_obs_offset(
            &obj,
            &near,
            sigma,
            [0.0; 3],
            &Desig::ObservatoryCode("500".into()),
        );
        obs.extend(synth_obs_offset(
            &obj,
            &far,
            sigma,
            [0.16, 0.98, 0.43],
            &Desig::ObservatoryCode("C49".into()),
        ));

        let samples = fit_orbit_ranging(&obs, 200, 1.0, 3).expect("ranging must succeed");
        assert!(!samples.draws.is_empty());
        let spk = LOADED_SPK.try_read().unwrap();
        let obj_sun = spk.try_to_sun(obj.clone()).unwrap();
        let truth = spk
            .try_to_ssb(propagate_two_body(&obj_sun, Time::<TDB>::new(samples.epoch)).unwrap())
            .unwrap();
        let best = samples
            .draws
            .iter()
            .map(|d| (Vector::<Equatorial>::new([d[0], d[1], d[2]]) - truth.pos).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best < 0.05,
            "no draw within 0.05 AU of the true position: closest {best:.3} AU"
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

        let result = fit_orbit_ranging(&obs, 200, 10.0, 42);
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
        let samples = fit_orbit_ranging(&obs, 50, 10.0, 7).unwrap();
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
        let a = fit_orbit_ranging(&obs, 20, 10.0, 99).unwrap();
        let b = fit_orbit_ranging(&obs, 20, 10.0, 99).unwrap();
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
        let samples = fit_orbit_ranging(&obs, 50, 10.0, 1).unwrap();
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

        let samples = fit_orbit_ranging(&obs, 100, 1.0, 42).unwrap();
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
