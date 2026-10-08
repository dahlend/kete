// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Core types and helper functions shared across the fitting submodules.

use crate::common::ThermalGeometry;
use crate::{BandInfo, bond_albedo, hg_apparent_flux};
use kete_core::errors::{Error, KeteResult};
use nalgebra::Vector3;

/// Degrees of freedom for the Student-t likelihood.
pub(super) const STUDENT_NU: f64 = 5.0;

/// Steepness of logistic barriers (sharper = closer to hard wall).
pub(super) const BARRIER_K: f64 = 50.0;

/// Which model to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// Near-Earth Asteroid Thermal Model -- beaming is a free parameter.
    Neatm,
    /// Fast Rotating Model -- beaming fixed at pi.
    Frm,
    /// HG reflected-light model only -- fits H and G.
    Hg,
}

impl Model {
    /// Number of free parameters.
    ///
    /// NEATM: `[D, beaming, H, G, f_sigma, R_IR]` -> 6.
    /// FRM:   `[D, H, G, f_sigma, R_IR]`           -> 5.
    /// HG:    `[H, G, f_sigma]`                     -> 3.
    #[must_use]
    pub fn dim(self) -> usize {
        match self {
            Self::Neatm => 6,
            Self::Frm => 5,
            Self::Hg => 3,
        }
    }

    /// Whether this is the NEATM model.
    #[must_use]
    pub fn is_neatm(self) -> bool {
        matches!(self, Self::Neatm)
    }

    /// Whether this is the HG reflected-light-only model.
    #[must_use]
    pub fn is_hg(self) -> bool {
        matches!(self, Self::Hg)
    }

    /// Column names for posterior draw vectors (physical space).
    ///
    /// NEATM: `["diameter", "vis_albedo", "beaming", "h_mag", "g_param", "ir_albedo_ratio", "f_sigma"]`
    /// FRM:   `["diameter", "vis_albedo", "h_mag", "g_param", "ir_albedo_ratio", "f_sigma"]`
    /// HG:    `["h_mag", "g_param", "f_sigma"]`
    #[must_use]
    pub fn draw_column_names(self) -> &'static [&'static str] {
        match self {
            Self::Neatm => &[
                "diameter",
                "vis_albedo",
                "beaming",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Self::Frm => &[
                "diameter",
                "vis_albedo",
                "h_mag",
                "g_param",
                "ir_albedo_ratio",
                "f_sigma",
            ],
            Self::Hg => &["h_mag", "g_param", "f_sigma"],
        }
    }

    /// Decode the raw parameter vector into physical parameters.
    ///
    /// This is the **only** place that knows the `x: &[f64]` layout.
    /// Infeasible combinations are not rejected here; for example a non-positive
    /// diameter gives an infinite `vis_albedo`, which the priors then penalize.
    ///
    /// Layout (all linear):
    /// - NEATM: `[diameter, beaming, h_mag, g_param, f_sigma, r_ir]`
    /// - FRM:   `[diameter, h_mag, g_param, f_sigma, r_ir]`
    /// - HG:    `[h_mag, g_param, f_sigma]`
    pub(crate) fn unpack(self, x: &[f64], emissivity: f64, c_hg: f64) -> ModelParams {
        if self.is_hg() {
            let h_mag = x[0];
            let vis_albedo = 1.0;
            let diameter = c_hg * 10.0_f64.powf(-h_mag / 5.0);
            return ModelParams {
                diameter,
                beaming: f64::NAN,
                h_mag,
                g_param: x[1],
                emissivity,
                f_sigma: x[2],
                // Every band reflects with the visible albedo.
                r_ir: 1.0,
                vis_albedo,
            };
        }

        // Thermal models share the same trailing layout:
        //   [D, (beaming)?, H, G, f_sigma, R_IR]
        // The only difference is whether beaming is present.
        let diameter = x[0];
        let (beaming, h) = if self.is_neatm() {
            (x[1], 2)
        } else {
            (f64::NAN, 1)
        };

        let h_mag = x[h];
        // Compute raw (unclamped) albedo so the logistic-barrier prior can
        // see the true derived value and properly penalize out-of-bounds
        // regions. The clamped version in `albedo_from_h_mag_diam` would
        // create a flat plateau that fools the prior and produces ridge
        // artifacts in the posterior.
        let vis_albedo = if diameter > 0.0 {
            (c_hg * 10_f64.powf(-0.2 * h_mag) / diameter).powi(2)
        } else {
            f64::INFINITY
        };

        ModelParams {
            diameter,
            beaming,
            h_mag,
            g_param: x[h + 1],
            emissivity,
            f_sigma: x[h + 2],
            r_ir: x[h + 3],
            vis_albedo,
        }
    }
}

/// Physical parameters decoded from the parameter vector.
///
/// Produced by [`Model::unpack`]; consumed by the forward model, likelihood,
/// and prior functions.  No downstream code needs to know the raw vector
/// layout.
pub(crate) struct ModelParams {
    pub diameter: f64,
    pub beaming: f64,
    pub h_mag: f64,
    pub g_param: f64,
    pub emissivity: f64,
    pub f_sigma: f64,
    pub r_ir: f64,
    pub vis_albedo: f64,
}

impl ModelParams {
    /// Band albedo times the squared diameter in km^2, which scales the reflected
    /// flux. `vis_albedo * diameter^2 = (c_hg * 10^(-H/5))^2` depends only on H.
    fn reflecting_area(&self) -> f64 {
        self.r_ir * self.vis_albedo * self.diameter.powi(2)
    }

    /// Convert to a draw row in physical space.
    ///
    /// NEATM: `[diameter, vis_albedo, beaming, h_mag, g_param, r_ir, f_sigma]`
    /// FRM:   `[diameter, vis_albedo, h_mag, g_param, r_ir, f_sigma]`
    /// HG:    `[h_mag, g_param, f_sigma]`
    pub(crate) fn to_draw_row(&self, model: Model) -> Vec<f64> {
        if model.is_hg() {
            return vec![self.h_mag, self.g_param, self.f_sigma];
        }
        let mut row = vec![self.diameter, self.vis_albedo];
        if model.is_neatm() {
            row.push(self.beaming);
        }
        row.extend_from_slice(&[self.h_mag, self.g_param, self.r_ir, self.f_sigma]);
        row
    }
}

/// Positions of the physical parameters in a gradient, which uses the NEATM
/// parameter layout for every model.
const DIAMETER: usize = 0;
const BEAMING: usize = 1;
const H_MAG: usize = 2;
const G_PARAM: usize = 3;
const F_SIGMA: usize = 4;
const R_IR: usize = 5;

/// d ln(10^(-0.4 H)) / dH: the rate at which `vis_albedo * diameter^2` changes with H.
const LN_FLUX_PER_MAG: f64 = -0.4 * std::f64::consts::LN_10;

impl Model {
    /// Positions in the NEATM layout of each entry of this model's parameter vector.
    /// Must agree with [`Model::unpack`].
    fn neatm_positions(self) -> &'static [usize] {
        match self {
            Self::Neatm => &[DIAMETER, BEAMING, H_MAG, G_PARAM, F_SIGMA, R_IR],
            Self::Frm => &[DIAMETER, H_MAG, G_PARAM, F_SIGMA, R_IR],
            Self::Hg => &[H_MAG, G_PARAM, F_SIGMA],
        }
    }
}

/// Parameter-independent quantities of one observation, computed once per fit.
#[derive(Debug, Clone)]
struct ObsGeometry {
    /// Thermal geometry, `None` for the HG model.
    thermal: Option<ThermalGeometry>,

    /// Reflected flux in Jy per unit band albedo and per km^2 of squared diameter,
    /// at `G = 0` and `G = 1`. The HG phase curve is linear in `G`.
    reflected: [f64; 2],
}

impl ObsGeometry {
    fn new(model: Model, ob: &FluxObs) -> Self {
        let thermal = match model {
            Model::Neatm => Some(ThermalGeometry::neatm(&ob.sun2obj, &ob.sun2obs)),
            Model::Frm => Some(ThermalGeometry::frm(&ob.sun2obj, &ob.sun2obs)),
            Model::Hg => None,
        };
        let reflected = [0.0, 1.0].map(|g_param| {
            hg_apparent_flux(
                g_param,
                1.0,
                &ob.sun2obj,
                &ob.sun2obs,
                ob.band.wavelength,
                1.0,
            ) * ob.band.solar_correction
        });
        Self { thermal, reflected }
    }

    /// Model thermal and reflected flux of one observation in Jy.
    fn fluxes(&self, band: &BandInfo, params: &ModelParams) -> (f64, f64) {
        let thermal = self.thermal.as_ref().map_or(0.0, |geom| {
            geom.flux(
                band,
                params.diameter,
                params.vis_albedo,
                params.g_param,
                params.beaming,
                params.emissivity,
            )
        });
        let [refl_0, refl_1] = self.reflected;
        let phase_curve = (1.0 - params.g_param) * refl_0 + params.g_param * refl_1;
        (thermal, phase_curve * params.reflecting_area())
    }

    /// Model flux of one observation in Jy, and its gradient with respect to the
    /// physical parameters in the NEATM layout. Only NEATM fits the beaming.
    fn flux_and_gradient(
        &self,
        model: Model,
        band: &BandInfo,
        params: &ModelParams,
    ) -> (f64, [f64; 6]) {
        let mut grad = [0.0; 6];

        // Thermal flux and its derivative with respect to the log of the
        // sub-solar temperature.
        let (thermal, d_log_temp) = self.thermal.as_ref().map_or((0.0, 0.0), |geom| {
            geom.flux_and_log_derivative(
                band,
                params.diameter,
                params.vis_albedo,
                params.g_param,
                params.beaming,
                params.emissivity,
            )
        });
        if self.thermal.is_some() {
            grad[DIAMETER] = 2.0 * thermal / params.diameter;
        }
        if d_log_temp != 0.0 {
            // T_ss^4 is proportional to (1 - A) / beaming, with the Bond albedo
            // A = vis_albedo * q(G) and vis_albedo proportional to 10^(-0.4 H) / D^2.
            let bond = bond_albedo(params.vis_albedo, params.g_param);
            let d_bond = d_log_temp * -0.25 / (1.0 - bond);
            grad[DIAMETER] += d_bond * -2.0 * bond / params.diameter;
            grad[H_MAG] += d_bond * LN_FLUX_PER_MAG * bond;
            grad[G_PARAM] += d_bond
                * (bond_albedo(params.vis_albedo, 1.0) - bond_albedo(params.vis_albedo, 0.0));
            if model.is_neatm() {
                grad[BEAMING] = d_log_temp * -0.25 / params.beaming;
            }
        }

        let [refl_0, refl_1] = self.reflected;
        let area = params.reflecting_area();
        let phase_curve = (1.0 - params.g_param) * refl_0 + params.g_param * refl_1;
        let reflected = phase_curve * area;
        grad[H_MAG] += LN_FLUX_PER_MAG * reflected;
        grad[G_PARAM] += (refl_1 - refl_0) * area;
        grad[R_IR] = phase_curve * params.vis_albedo * params.diameter.powi(2);

        (thermal + reflected, grad)
    }
}

/// A model with its observations, priors and fixed constants, and the geometry of
/// each observation precomputed.
#[derive(Debug, Clone)]
pub(super) struct FitProblem {
    /// Model being fit.
    pub model: Model,
    /// Observations.
    pub obs: Vec<FluxObs>,
    /// Geometry of each observation.
    geometry: Vec<ObsGeometry>,
    /// Relationship constant for D-H-pV conversion (km).
    pub c_hg: f64,
    /// Fixed thermal emissivity.
    pub emissivity: f64,
    /// Priors.
    pub priors: FluxPriors,
}

impl FitProblem {
    pub(super) fn new(
        model: Model,
        obs: &[FluxObs],
        c_hg: f64,
        emissivity: f64,
        priors: &FluxPriors,
    ) -> Self {
        Self {
            model,
            obs: obs.to_vec(),
            geometry: obs.iter().map(|ob| ObsGeometry::new(model, ob)).collect(),
            c_hg,
            emissivity,
            priors: priors.clone(),
        }
    }

    /// Decode the raw parameter vector, see [`Model::unpack`].
    pub(super) fn unpack(&self, x: &[f64]) -> ModelParams {
        self.model.unpack(x, self.emissivity, self.c_hg)
    }

    /// Model flux and reflected fraction of every observation.
    pub(super) fn forward(&self, params: &ModelParams) -> ForwardModelResult {
        let (model_fluxes, reflected_frac) = self
            .obs
            .iter()
            .zip(&self.geometry)
            .map(|(ob, geom)| {
                let (thermal, reflected) = geom.fluxes(&ob.band, params);
                let total = thermal + reflected;
                (total, reflected / total)
            })
            .unzip();
        ForwardModelResult {
            model_fluxes,
            reflected_frac,
        }
    }

    /// Log-likelihood: the sum over observations of
    /// [`FluxObs::log_likelihood_term`].
    ///
    /// Returns `f64::NEG_INFINITY` for infeasible points.
    pub(super) fn log_likelihood(&self, params: &ModelParams) -> f64 {
        let ll: f64 = self
            .obs
            .iter()
            .zip(&self.geometry)
            .map(|(ob, geom)| {
                let (thermal, reflected) = geom.fluxes(&ob.band, params);
                ob.log_likelihood_term(thermal + reflected, params.f_sigma)
            })
            .sum();
        if ll.is_finite() {
            ll
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Log-prior, adding its gradient in the NEATM layout to `grad`.
    ///
    /// All models share H, G, and `f_sigma` priors. Thermal models add diameter,
    /// beaming, and `r_ir` priors plus a derived `vis_albedo` penalty.
    fn log_prior_and_gradient(&self, params: &ModelParams, grad: &mut [f64; 6]) -> f64 {
        let priors = &self.priors;
        let mut lp = 0.0;
        let mut add = |prior: &ParamPrior, value: f64, idx: usize| {
            let (p, d) = prior.log_prob_and_derivative(value);
            lp += p;
            grad[idx] += d;
        };

        // ----- Shared across all models -----
        add(&priors.h_mag, params.h_mag, H_MAG);
        add(&priors.g_param, params.g_param, G_PARAM);
        add(&priors.f_sigma, params.f_sigma, F_SIGMA);

        if self.model.is_hg() {
            return lp;
        }

        // ----- Thermal only (NEATM / FRM) -----
        add(&priors.diameter, params.diameter, DIAMETER);
        if self.model.is_neatm() {
            add(&priors.beaming, params.beaming, BEAMING);
        }
        add(&priors.r_ir, params.r_ir, R_IR);
        let (p, d) = priors.vis_albedo.log_prob_and_derivative(params.vis_albedo);
        lp += p;
        grad[DIAMETER] += d * -2.0 * params.vis_albedo / params.diameter;
        grad[H_MAG] += d * LN_FLUX_PER_MAG * params.vis_albedo;

        lp
    }

    /// Log-posterior = log-likelihood + log-prior, to be **maximized**.
    ///
    /// Returns `f64::NEG_INFINITY` for infeasible points.
    pub(super) fn log_posterior(&self, x: &[f64]) -> f64 {
        let params = self.unpack(x);
        let ll = self.log_likelihood(&params);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let val = ll + self.log_prior_and_gradient(&params, &mut [0.0; 6]);
        if val.is_finite() {
            val
        } else {
            f64::NEG_INFINITY
        }
    }

    /// [`Self::log_posterior`], writing its gradient with respect to `x` into `grad`.
    ///
    /// The gradient is only written when the log-posterior is finite.
    pub(super) fn log_posterior_and_gradient(&self, x: &[f64], grad: &mut [f64]) -> f64 {
        let params = self.unpack(x);
        let mut physical = [0.0; 6];
        let mut ll = 0.0;
        for (ob, geom) in self.obs.iter().zip(&self.geometry) {
            let (flux, flux_grad) = geom.flux_and_gradient(self.model, &ob.band, &params);
            let (cost, d_flux, d_f_sigma) =
                ob.log_likelihood_term_and_derivatives(flux, params.f_sigma);
            ll += cost;
            for (g, fg) in physical.iter_mut().zip(flux_grad) {
                *g += d_flux * fg;
            }
            physical[F_SIGMA] += d_f_sigma;
        }
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let val = ll + self.log_prior_and_gradient(&params, &mut physical);
        if !val.is_finite() {
            return f64::NEG_INFINITY;
        }
        for (g, &idx) in grad.iter_mut().zip(self.model.neatm_positions()) {
            *g = physical[idx];
        }
        val
    }
}

/// A single flux constraint at a known geometry.
///
/// The constraint on the model flux is a **sum of independent [`Penalty`]
/// terms** -- e.g. a hard [`Penalty::TopHat`] interval and/or a
/// [`Penalty::Center`] point estimate -- the same primitives a [`ParamPrior`]
/// composes for a parameter, here applied to the model flux (the projection
/// differs; the penalties are shared). Use the [`Self::detection`],
/// [`Self::upper_limit`], and [`Self::bounded`] constructors for the common
/// cases, or [`Self::from_parts`] for combinations.
#[derive(Debug, Clone)]
pub struct FluxObs {
    /// Independent penalty terms on the model flux; the total cost is their sum.
    pub(super) penalties: Vec<Penalty>,
    /// Band information for this observation.
    pub band: BandInfo,
    /// Sun-to-object vector in AU (Ecliptic frame).
    pub sun2obj: Vector3<f64>,
    /// Sun-to-observer vector in AU (Ecliptic frame).
    pub sun2obs: Vector3<f64>,
}

impl FluxObs {
    /// General constructor: an optional hard `(lo, hi)` interval and/or an
    /// optional point estimate `(mean, sigma_lo, sigma_hi)`, where each scale
    /// may be `None` to leave that side unconstrained (a one-sided limit). At
    /// least one of `bounds`/`point` should be `Some` to constrain anything.
    ///
    /// # Panics
    /// Panics if `bounds` is `Some((lo, hi))` with `hi <= lo`; callers exposed
    /// to user input should validate first and raise a proper error.
    #[must_use]
    pub fn from_parts(
        bounds: Option<(f64, f64)>,
        point: Option<(f64, Option<f64>, Option<f64>)>,
        band: BandInfo,
        sun2obj: Vector3<f64>,
        sun2obs: Vector3<f64>,
    ) -> Self {
        let mut penalties = Vec::new();
        if let Some((lo, hi)) = bounds {
            penalties.push(Penalty::top_hat(lo, hi));
        }
        if let Some((mean, scale_lo, scale_hi)) = point {
            // Flux measurements are heavy-tailed and inflate with f_sigma, so the
            // anchor (`normalize`) applies (suppressed internally if one-sided).
            penalties.push(Penalty::Center {
                mean,
                scale_lo,
                scale_hi,
                tail: Tail::StudentT,
                normalize: true,
            });
        }
        Self {
            penalties,
            band,
            sun2obj,
            sun2obs,
        }
    }

    /// A two-sided flux detection `flux +/- sigma` (symmetric error).
    #[must_use]
    pub fn detection(
        flux: f64,
        sigma: f64,
        band: BandInfo,
        sun2obj: Vector3<f64>,
        sun2obs: Vector3<f64>,
    ) -> Self {
        Self::detection_asym(flux, sigma, sigma, band, sun2obj, sun2obs)
    }

    /// A two-sided flux detection with asymmetric error: `sigma_lo` below the
    /// measured `flux`, `sigma_hi` above.
    #[must_use]
    pub fn detection_asym(
        flux: f64,
        sigma_lo: f64,
        sigma_hi: f64,
        band: BandInfo,
        sun2obj: Vector3<f64>,
        sun2obs: Vector3<f64>,
    ) -> Self {
        Self::from_parts(
            None,
            Some((flux, Some(sigma_lo), Some(sigma_hi))),
            band,
            sun2obj,
            sun2obs,
        )
    }

    /// A soft photometric upper limit: a non-detection at `threshold` with noise
    /// scale `sigma`. Only model fluxes exceeding the threshold are penalized
    /// (the below-threshold side is unconstrained).
    #[must_use]
    pub fn upper_limit(
        threshold: f64,
        sigma: f64,
        band: BandInfo,
        sun2obj: Vector3<f64>,
        sun2obs: Vector3<f64>,
    ) -> Self {
        Self::from_parts(
            None,
            Some((threshold, None, Some(sigma))),
            band,
            sun2obj,
            sun2obs,
        )
    }

    /// A hard flux interval `[lo, hi]` with no point estimate.
    #[must_use]
    pub fn bounded(
        lo: f64,
        hi: f64,
        band: BandInfo,
        sun2obj: Vector3<f64>,
        sun2obs: Vector3<f64>,
    ) -> Self {
        Self::from_parts(Some((lo, hi)), None, band, sun2obj, sun2obs)
    }

    /// The centering term's `(mean, scale_lo, scale_hi)`, if any.
    fn center(&self) -> Option<(f64, Option<f64>, Option<f64>)> {
        self.penalties.iter().find_map(Penalty::as_center)
    }

    /// Point-estimate flux (the center), or `None` for a bounds-only constraint.
    #[must_use]
    pub fn point_estimate(&self) -> Option<f64> {
        self.center().map(|(mean, _, _)| mean)
    }

    /// Lower-side 1-sigma scale, or `None` (no point estimate, or upper limit).
    #[must_use]
    pub fn sigma_lo(&self) -> Option<f64> {
        self.center().and_then(|(_, lo, _)| lo)
    }

    /// Upper-side 1-sigma scale, or `None` (no point estimate, or lower limit).
    #[must_use]
    pub fn sigma_hi(&self) -> Option<f64> {
        self.center().and_then(|(_, _, hi)| hi)
    }

    /// Hard `(lo, hi)` flux interval, or `None` if unbounded.
    #[must_use]
    pub fn bounds(&self) -> Option<(f64, f64)> {
        self.penalties.iter().find_map(Penalty::as_top_hat)
    }

    /// Whether this is a (one-sided) non-detection upper limit.
    #[must_use]
    pub fn is_upper_limit(&self) -> bool {
        matches!(self.center(), Some((_, None, Some(_))))
    }

    /// Whether this observation is a two-sided detection (counts toward the
    /// reduced-chi2 degrees of freedom).
    pub(super) fn is_detection(&self) -> bool {
        matches!(self.center(), Some((_, Some(_), Some(_))))
    }

    /// Standardized residual `(mean - model) / (f_sigma * sigma_eff)` for MAP
    /// diagnostics, or `None` when there is no point estimate, or the residual
    /// falls on an unconstrained side (e.g. below an upper-limit threshold).
    pub(super) fn standardized_residual(&self, model_flux: f64, f_sigma: f64) -> Option<f64> {
        let (mean, scale_lo, scale_hi) = self.center()?;
        let r = mean - model_flux;
        let sigma = f_sigma * centering_scale(scale_lo, scale_hi, r)?;
        Some(if sigma > 0.0 { r / sigma } else { 0.0 })
    }

    /// Log-likelihood contribution of this observation: the sum of its penalty
    /// terms. Hard `bounds` are walls not scaled by `f_sigma`; a point estimate
    /// is a Student-t term scaled by `f_sigma`. See [`Penalty::cost_and_derivatives`].
    pub(super) fn log_likelihood_term(&self, model_flux: f64, f_sigma: f64) -> f64 {
        self.log_likelihood_term_and_derivatives(model_flux, f_sigma)
            .0
    }

    /// [`Self::log_likelihood_term`] and its derivatives with respect to the model
    /// flux and `f_sigma`.
    pub(super) fn log_likelihood_term_and_derivatives(
        &self,
        model_flux: f64,
        f_sigma: f64,
    ) -> (f64, f64, f64) {
        self.penalties
            .iter()
            .fold((0.0, 0.0, 0.0), |(c, d_flux, d_f_sigma), p| {
                let (pc, pd_flux, pd_f_sigma) = p.cost_and_derivatives(model_flux, f_sigma);
                (c + pc, d_flux + pd_flux, d_f_sigma + pd_f_sigma)
            })
    }
}

/// Unnormalized Student-t(nu=[`STUDENT_NU`]) log kernel for residual `r` and
/// scale `sigma`: the residual-dependent part of the log-likelihood.
#[must_use]
fn student_t_log_kernel(r: f64, sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    -0.5 * (STUDENT_NU + 1.0) * (1.0 + r * r / (STUDENT_NU * sigma2)).ln()
}

/// Tail shape of a [`Penalty::Center`] term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Tail {
    /// Plain Gaussian -- used by priors.
    Gaussian,
    /// Heavy-tailed Student-t(nu = [`STUDENT_NU`]) -- used by robust measurements.
    StudentT,
}

/// One additive penalty term on a scalar quantity.
///
/// A constraint -- a [`ParamPrior`] on a parameter, or a [`FluxObs`] on a model
/// flux -- is a *sum* of these. The terms are independent costs with no shared
/// parameters; the caller supplies the scalar (the projection) and sums each
/// term's [`Penalty::cost_and_derivatives`]. New constraint shapes are new variants here.
#[derive(Debug, Clone)]
pub(super) enum Penalty {
    /// Hard interval: logistic-barrier walls of steepness `k`, flat (zero cost)
    /// inside `[lo, hi]`, falling off outside. Never scaled by `scale_factor`.
    TopHat { lo: f64, hi: f64, k: f64 },
    /// (Possibly asymmetric / one-sided) centering term toward `mean`.
    ///
    /// `scale_lo`/`scale_hi` are the below-/above-`mean` 1-sigma widths; a `None`
    /// side is left unconstrained (a one-sided limit). `normalize` adds the
    /// residual-independent `-(scale_factor * sigma_bar).ln()` anchor that pins a
    /// fitted scale -- applied only when both sides are present (off for priors,
    /// and automatically suppressed for a one-sided/censored term).
    Center {
        mean: f64,
        scale_lo: Option<f64>,
        scale_hi: Option<f64>,
        tail: Tail,
        normalize: bool,
    },
}

impl Penalty {
    /// A hard `[lo, hi]` interval with width-aware wall steepness.
    ///
    /// Steepness is `BARRIER_K` per unit for intervals wider than 1, and
    /// `BARRIER_K` per interval-width for narrower ones -- so a tight interval
    /// (flux bounds at ~1e-3 Jy, or the tight-bounds parameter-fixing idiom in
    /// [`ParamPrior`]) reads as a wall rather than a gentle slope, while wide
    /// intervals keep the legacy fixed steepness.
    ///
    /// # Panics
    /// Panics if `hi <= lo`; callers exposed to user input should validate
    /// first and raise a proper error.
    pub(super) fn top_hat(lo: f64, hi: f64) -> Self {
        assert!(
            hi > lo,
            "TopHat interval requires lo < hi, got ({lo}, {hi})"
        );
        let k = BARRIER_K / (hi - lo).min(1.0);
        Self::TopHat { lo, hi, k }
    }

    /// Value of [`Self::cost_and_derivatives`].
    #[cfg(test)]
    pub(super) fn cost(&self, x: f64, scale_factor: f64) -> f64 {
        self.cost_and_derivatives(x, scale_factor).0
    }

    /// Log-cost contribution at scalar `x`, and its derivatives with respect to `x`
    /// and `scale_factor`. `scale_factor` multiplies the centering scales (1.0 for
    /// priors, the fitted `f_sigma` for measurements); a [`Penalty::TopHat`]
    /// ignores it.
    pub(super) fn cost_and_derivatives(&self, x: f64, scale_factor: f64) -> (f64, f64, f64) {
        match *self {
            Self::TopHat { lo, hi, k } => {
                let (c, d_x) = logistic_barrier_and_derivative(x, lo, hi, k);
                (c, d_x, 0.0)
            }
            Self::Center {
                mean,
                scale_lo,
                scale_hi,
                tail,
                normalize,
            } => {
                let r = mean - x;
                let mut c = 0.0;
                let mut d_x = 0.0;
                let mut d_scale = 0.0;
                if let Some(scale) = centering_scale(scale_lo, scale_hi, r) {
                    let sigma = scale_factor * scale;
                    match tail {
                        Tail::Gaussian => {
                            let z = r / sigma;
                            c += -0.5 * z * z;
                            d_x += z / sigma;
                            d_scale += z * z / scale_factor;
                        }
                        Tail::StudentT => {
                            c += student_t_log_kernel(r, sigma);
                            let denom = STUDENT_NU * sigma * sigma + r * r;
                            d_x += (STUDENT_NU + 1.0) * r / denom;
                            d_scale += (STUDENT_NU + 1.0) * r * r / (denom * scale_factor);
                        }
                    }
                }
                if normalize && let (Some(lo), Some(hi)) = (scale_lo, scale_hi) {
                    let sigma_bar = f64::midpoint(lo, hi);
                    c += -(scale_factor * sigma_bar).ln();
                    d_scale -= scale_factor.recip();
                }
                (c, d_x, d_scale)
            }
        }
    }

    /// The `(mean, scale_lo, scale_hi)` of a [`Penalty::Center`], else `None`.
    pub(super) fn as_center(&self) -> Option<(f64, Option<f64>, Option<f64>)> {
        match *self {
            Self::Center {
                mean,
                scale_lo,
                scale_hi,
                ..
            } => Some((mean, scale_lo, scale_hi)),
            Self::TopHat { .. } => None,
        }
    }

    /// The `(lo, hi)` of a [`Penalty::TopHat`], else `None`.
    pub(super) fn as_top_hat(&self) -> Option<(f64, f64)> {
        match *self {
            Self::TopHat { lo, hi, .. } => Some((lo, hi)),
            Self::Center { .. } => None,
        }
    }
}

/// Effective 1-sigma scale on the side of residual `r = center - x` (before any
/// `scale_factor`), or `None` if that side is unconstrained (a one-sided limit).
///
/// Two-sided values form a split (two-piece) normal/Student-t: the side's scale
/// applies strictly, so a 1-sigma deviation always scores as exactly 1 sigma.
/// The kernel's gradient is still continuous at `r = 0` (it vanishes from both
/// sides); only the curvature jumps, which the finite-difference NUTS sampler
/// tolerates. At exactly `r = 0` the kernel is zero, so the returned mean scale
/// only affects the (also zero) standardized residual.
fn centering_scale(scale_lo: Option<f64>, scale_hi: Option<f64>, r: f64) -> Option<f64> {
    match (scale_lo, scale_hi) {
        (Some(lo), Some(hi)) => Some(if r > 0.0 {
            lo
        } else if r < 0.0 {
            hi
        } else {
            f64::midpoint(lo, hi)
        }),
        // Upper limit: only the above-center side (model above, r < 0) bites.
        (None, Some(hi)) => (r < 0.0).then_some(hi),
        // Lower limit: only the below-center side (model below, r > 0) bites.
        (Some(lo), None) => (r > 0.0).then_some(lo),
        (None, None) => None,
    }
}

/// Configuration for a single fitted parameter's prior.
///
/// Each parameter has:
/// - `bounds`: `(lo, hi)` logistic-barrier hard bounds.
/// - `gaussian`: Optional `(mean, sigma_lo, sigma_hi)` Gaussian centering prior.
///
/// When `gaussian` is `Some((mean, sigma_lo, sigma_hi))`, the posterior is
/// pulled toward `mean`. The prior may be asymmetric: `sigma_lo` applies when
/// the parameter is below `mean`, `sigma_hi` when it is above (the two are
/// equal for an ordinary symmetric prior). This lets an external measurement of
/// a parameter -- e.g. an optical H with a lopsided error bar -- be entered as a
/// prior. When `gaussian` is `None`, only the hard bounds apply (flat/uniform
/// prior within the bounded region).
///
/// To effectively fix a parameter to a value, set tight bounds around it
/// (e.g., `bounds = (val - 0.001, val + 0.001)`) -- the logistic barrier
/// will constrain samples to that narrow range.
#[derive(Debug, Clone)]
pub struct ParamPrior {
    /// (lo, hi) logistic-barrier bounds.
    pub bounds: (f64, f64),
    /// Optional Gaussian centering prior `(mean, sigma_lo, sigma_hi)`.
    /// `sigma_lo`/`sigma_hi` are the below-/above-mean 1-sigma widths (equal
    /// when symmetric). `None` means flat prior within bounds.
    pub gaussian: Option<(f64, f64, f64)>,
}

impl ParamPrior {
    /// Create a prior with only hard bounds (flat/uniform within the range).
    #[must_use]
    pub fn bounds_only(lo: f64, hi: f64) -> Self {
        Self {
            bounds: (lo, hi),
            gaussian: None,
        }
    }

    /// Create a prior with hard bounds and a symmetric Gaussian center.
    #[must_use]
    pub fn with_gaussian(lo: f64, hi: f64, mean: f64, sigma: f64) -> Self {
        Self {
            bounds: (lo, hi),
            gaussian: Some((mean, sigma, sigma)),
        }
    }

    /// Create a prior with hard bounds and an asymmetric Gaussian center.
    ///
    /// `sigma_lo` is the 1-sigma width below `mean`, `sigma_hi` the width above.
    #[must_use]
    pub fn with_gaussian_asym(lo: f64, hi: f64, mean: f64, sigma_lo: f64, sigma_hi: f64) -> Self {
        Self {
            bounds: (lo, hi),
            gaussian: Some((mean, sigma_lo, sigma_hi)),
        }
    }

    /// Value of [`Self::log_prob_and_derivative`].
    #[cfg(test)]
    pub(super) fn log_prob(&self, x: f64) -> f64 {
        self.log_prob_and_derivative(x).0
    }

    /// Log-prior contribution for this parameter, and its derivative with respect
    /// to `x`: a hard [`Penalty::top_hat`] wall plus, if present, a Gaussian
    /// [`Penalty::Center`]. Priors are never inflated by `f_sigma`
    /// (`scale_factor = 1.0`) and carry no anchor.
    pub(super) fn log_prob_and_derivative(&self, x: f64) -> (f64, f64) {
        let (mut lp, mut d_x, _) =
            Penalty::top_hat(self.bounds.0, self.bounds.1).cost_and_derivatives(x, 1.0);
        if let Some((mean, scale_lo, scale_hi)) = self.gaussian {
            let (c, d, _) = Penalty::Center {
                mean,
                scale_lo: Some(scale_lo),
                scale_hi: Some(scale_hi),
                tail: Tail::Gaussian,
                normalize: false,
            }
            .cost_and_derivatives(x, 1.0);
            lp += c;
            d_x += d;
        }
        (lp, d_x)
    }

    /// Midpoint of the bounds, or the Gaussian mean if set.
    pub(crate) fn center(&self) -> f64 {
        self.gaussian
            .map_or(f64::midpoint(self.bounds.0, self.bounds.1), |(m, _, _)| m)
    }
}

/// Prior configuration for model fitting.
///
/// Each fitted parameter is configured via a [`ParamPrior`] with hard bounds
/// and an optional Gaussian center.  All bounds and Gaussian parameters are
/// specified in linear/physical units.
///
/// The nuisance parameter `f_sigma` has a sensible fixed prior handled
/// internally.
#[derive(Debug, Clone)]
pub struct FluxPriors {
    /// Prior on diameter D (km).  Used by NEATM/FRM.
    pub diameter: ParamPrior,
    /// Prior on beaming parameter.  Used by NEATM.
    pub beaming: ParamPrior,
    /// Prior on IR-to-visible albedo ratio `r_ir`.  Used by NEATM/FRM.
    pub r_ir: ParamPrior,
    /// Prior on H magnitude.
    pub h_mag: ParamPrior,
    /// Prior on G parameter.
    pub g_param: ParamPrior,
    /// Prior on geometric albedo `vis_albedo` (linear scale).
    /// Used by thermal models to penalize infeasible albedos derived from D and H.
    pub vis_albedo: ParamPrior,
    /// Prior on `f_sigma`, the uncertainty scaling factor.
    pub f_sigma: ParamPrior,
}

impl Default for FluxPriors {
    /// Sensible defaults (all in linear/physical units):
    /// - diameter in [0.001, 1000] km (bounds only)
    /// - beaming in [0.5, 3.0], Gaussian(1.0, 0.3)
    /// - `r_ir` in [0.5, 2.0], Gaussian(1.6, 0.3)
    /// - `h_mag` in [-5, 35] (bounds only)
    /// - `g_param` in [-0.3, 0.7], Gaussian(0.2, 0.05)
    /// - `vis_albedo` in [0.01, 1] (bounds only)
    /// - `f_sigma` in [0.5, 5.0] (bounds only)
    fn default() -> Self {
        Self {
            diameter: ParamPrior::bounds_only(0.001, 1000.0),
            beaming: ParamPrior::with_gaussian(0.5, 3.0, 1.0, 0.3),
            r_ir: ParamPrior::with_gaussian(0.5, 2.0, 1.6, 0.3),
            h_mag: ParamPrior::bounds_only(-5.0, 35.0),
            g_param: ParamPrior::with_gaussian(-0.3, 0.7, 0.2, 0.05),
            vis_albedo: ParamPrior::bounds_only(0.01, 1.0),
            f_sigma: ParamPrior::bounds_only(0.5, 5.0),
        }
    }
}

impl FluxPriors {
    /// Check that every prior has bounds with `lo < hi`.
    ///
    /// # Errors
    /// [`Error::ValueError`] naming the first prior whose bounds are not ordered,
    /// including bounds that are NaN.
    pub fn validate(&self) -> KeteResult<()> {
        for (name, prior) in [
            ("diameter", &self.diameter),
            ("beaming", &self.beaming),
            ("r_ir", &self.r_ir),
            ("h_mag", &self.h_mag),
            ("g_param", &self.g_param),
            ("vis_albedo", &self.vis_albedo),
            ("f_sigma", &self.f_sigma),
        ] {
            let (lo, hi) = prior.bounds;
            if lo.is_nan() || hi.is_nan() || lo >= hi {
                return Err(Error::ValueError(format!(
                    "The {name} prior bounds must satisfy lo < hi, got ({lo}, {hi})."
                )));
            }
        }
        Ok(())
    }
}

/// Value of [`logistic_barrier_and_derivative`].
#[cfg(test)]
#[must_use]
pub(super) fn logistic_barrier(x: f64, lo: f64, hi: f64, k: f64) -> f64 {
    logistic_barrier_and_derivative(x, lo, hi, k).0
}

/// Logistic barrier prior, and its derivative with respect to `x`: a smooth wall
/// that is 0 in the interior and -> -inf at the boundaries.
///
/// $$\ln\sigma(k(x - lo)) + \ln\sigma(k(hi - x))$$
///
/// where $\sigma$ is the logistic sigmoid.
///
/// # Arguments
/// * `x`  -- parameter value
/// * `lo` -- lower bound
/// * `hi` -- upper bound
/// * `k`  -- steepness (larger = sharper wall; 30 is typical)
fn logistic_barrier_and_derivative(x: f64, lo: f64, hi: f64, k: f64) -> (f64, f64) {
    // ln(sigmoid(z)) = z - ln(1 + exp(z)) but for numerical stability use -ln(1+exp(-z))
    // which is equivalent and avoids overflow for large positive z. Its derivative is
    // sigmoid(-z).
    fn log_sigmoid(z: f64) -> (f64, f64) {
        if z > 0.0 {
            let e = (-z).exp();
            (-e.ln_1p(), e / (1.0 + e))
        } else {
            let e = z.exp();
            (z - e.ln_1p(), (1.0 + e).recip())
        }
    }
    let (lower, d_lower) = log_sigmoid(k * (x - lo));
    let (upper, d_upper) = log_sigmoid(k * (hi - x));
    (lower + upper, k * (d_lower - d_upper))
}

/// Result of evaluating the forward model at a parameter point.
pub(super) struct ForwardModelResult {
    /// Model flux per observation (Jy), in obs-global index order.
    pub model_fluxes: Vec<f64>,
    /// Reflected-light fraction per observation.
    pub reflected_frac: Vec<f64>,
}
