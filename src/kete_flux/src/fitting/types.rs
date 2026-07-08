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

//! Core types and helper functions shared across the fitting submodules.

use crate::{
    BandInfo, ModelResults, flux_to_mag, frm_total_flux, hg_apparent_flux, hg_apparent_mag,
    mag_to_flux, neatm_total_flux,
};
use kete_core::constants::V_MAG_ZERO;
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
    /// NEATM: `["diameter", "vis_albedo", "beaming", "h_mag", "g_param", "r_ir", "f_sigma"]`
    /// FRM:   `["diameter", "vis_albedo", "h_mag", "g_param", "r_ir", "f_sigma"]`
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
                "r_ir",
                "f_sigma",
            ],
            Self::Frm => &[
                "diameter",
                "vis_albedo",
                "h_mag",
                "g_param",
                "r_ir",
                "f_sigma",
            ],
            Self::Hg => &["h_mag", "g_param", "f_sigma"],
        }
    }

    /// Decode the raw parameter vector into physical parameters.
    ///
    /// This is the **only** place that knows the `x: &[f64]` layout.
    /// Returns `None` for infeasible parameter combinations (e.g.
    /// negative albedo).
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
                r_ir: f64::NAN,
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
            (std::f64::consts::PI, 1)
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

impl Model {
    /// Compute apparent total fluxes for the given geometry.
    ///
    /// Works for all three model variants (NEATM, FRM, HG).
    pub(crate) fn compute_fluxes(
        self,
        params: &ModelParams,
        bands: &[BandInfo],
        sun2obj: &Vector3<f64>,
        sun2obs: &Vector3<f64>,
    ) -> ModelResults {
        let band_albedos: Vec<f64> = bands
            .iter()
            .map(|_| params.r_ir * params.vis_albedo)
            .collect();
        match self {
            Self::Neatm => neatm_total_flux(
                bands,
                &band_albedos,
                params.diameter,
                params.vis_albedo,
                params.g_param,
                params.h_mag,
                params.beaming,
                params.emissivity,
                sun2obj,
                sun2obs,
            ),
            Self::Frm => frm_total_flux(
                bands,
                &band_albedos,
                params.diameter,
                params.vis_albedo,
                params.g_param,
                params.h_mag,
                params.emissivity,
                sun2obj,
                sun2obs,
            ),
            Self::Hg => {
                let mut hg_fluxes = Vec::with_capacity(bands.len());
                for band in bands {
                    let flux = hg_apparent_flux(
                        params.g_param,
                        params.diameter,
                        sun2obj,
                        sun2obs,
                        band.wavelength,
                        params.vis_albedo,
                    ) * band.solar_correction;
                    hg_fluxes.push(flux);
                }
                let magnitudes: Vec<f64> = bands
                    .iter()
                    .zip(&hg_fluxes)
                    .map(|(band, flux)| flux_to_mag(*flux, band.zero_mag))
                    .collect();
                let v_band_magnitude =
                    hg_apparent_mag(params.g_param, params.h_mag, sun2obj, sun2obs);
                let v_band_flux = mag_to_flux(v_band_magnitude, V_MAG_ZERO);
                ModelResults {
                    thermal_fluxes: vec![0.0; bands.len()],
                    magnitudes,
                    fluxes: hg_fluxes.clone(),
                    hg_fluxes,
                    v_band_magnitude,
                    v_band_flux,
                }
            }
        }
    }

    /// Evaluate the forward model for the given [`ModelParams`].
    ///
    /// Assumes `params` came from [`Model::unpack`], which already performed
    /// feasibility checks.
    pub(super) fn evaluate_forward_model(
        self,
        params: &ModelParams,
        obs: &[FluxObs],
    ) -> ForwardModelResult {
        let n = obs.len();
        let mut model_fluxes = Vec::with_capacity(n);
        let mut reflected_frac = Vec::with_capacity(n);

        for ob in obs {
            let bands = [ob.band];
            let result = self.compute_fluxes(params, &bands, &ob.sun2obj, &ob.sun2obs);
            let rf = result.reflected_fraction();
            model_fluxes.push(result.fluxes[0]);
            reflected_frac.push(rf[0]);
        }

        ForwardModelResult {
            model_fluxes,
            reflected_frac,
        }
    }

    /// Evaluate the Student-t(nu=5) log-likelihood for the given parameters.
    ///
    /// Returns a value to be **maximized** (negative of the NLL).
    /// Returns `f64::NEG_INFINITY` for infeasible points.
    pub(super) fn log_likelihood(self, params: &ModelParams, obs: &[FluxObs]) -> f64 {
        let fwd = self.evaluate_forward_model(params, obs);

        let ll: f64 = obs
            .iter()
            .zip(&fwd.model_fluxes)
            .map(|(ob, &mf)| ob.log_likelihood_term(mf, params.f_sigma))
            .sum();

        if ll.is_finite() {
            ll
        } else {
            f64::NEG_INFINITY
        }
    }

    /// Evaluate the log-prior for the given [`ModelParams`].
    ///
    /// All models share H, G, and `f_sigma` priors.  Thermal models add
    /// diameter, beaming, and `r_ir` priors plus a derived `vis_albedo` penalty.
    pub(super) fn log_prior(self, params: &ModelParams, priors: &FluxPriors) -> f64 {
        let mut lp = 0.0;

        // ----- Shared across all models -----
        lp += priors.h_mag.log_prob(params.h_mag);
        lp += priors.g_param.log_prob(params.g_param);
        lp += priors.f_sigma.log_prob(params.f_sigma);

        if self.is_hg() {
            return lp;
        }

        // ----- Thermal only (NEATM / FRM) -----
        lp += priors.diameter.log_prob(params.diameter);
        if self.is_neatm() {
            lp += priors.beaming.log_prob(params.beaming);
        }
        lp += priors.r_ir.log_prob(params.r_ir);
        lp += priors.vis_albedo.log_prob(params.vis_albedo);

        lp
    }

    /// Evaluate the full log-posterior = log-likelihood + log-prior.
    ///
    /// Returns a value to be **maximized**.
    /// Returns `f64::NEG_INFINITY` for infeasible points.
    pub(super) fn log_posterior(
        self,
        x: &[f64],
        obs: &[FluxObs],
        c_hg: f64,
        emissivity: f64,
        priors: &FluxPriors,
    ) -> f64 {
        let params = self.unpack(x, emissivity, c_hg);
        let ll = self.log_likelihood(&params, obs);
        if !ll.is_finite() {
            return f64::NEG_INFINITY;
        }
        let val = ll + self.log_prior(&params, priors);
        if val.is_finite() {
            val
        } else {
            f64::NEG_INFINITY
        }
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
    /// is a Student-t term scaled by `f_sigma`. See [`Penalty::cost`].
    pub(super) fn log_likelihood_term(&self, model_flux: f64, f_sigma: f64) -> f64 {
        self.penalties.iter().map(|p| p.cost(model_flux, f_sigma)).sum()
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
/// term's [`Penalty::cost`]. New constraint shapes are new variants here.
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
        assert!(hi > lo, "TopHat interval requires lo < hi, got ({lo}, {hi})");
        let k = BARRIER_K / (hi - lo).min(1.0);
        Self::TopHat { lo, hi, k }
    }

    /// Log-cost contribution at scalar `x`. `scale_factor` multiplies the
    /// centering scales (1.0 for priors, the fitted `f_sigma` for measurements);
    /// a [`Penalty::TopHat`] ignores it.
    pub(super) fn cost(&self, x: f64, scale_factor: f64) -> f64 {
        match *self {
            Self::TopHat { lo, hi, k } => logistic_barrier(x, lo, hi, k),
            Self::Center {
                mean,
                scale_lo,
                scale_hi,
                tail,
                normalize,
            } => {
                let r = mean - x;
                let mut c = 0.0;
                if let Some(scale) = centering_scale(scale_lo, scale_hi, r) {
                    let sigma = scale_factor * scale;
                    c += match tail {
                        Tail::Gaussian => {
                            let z = r / sigma;
                            -0.5 * z * z
                        }
                        Tail::StudentT => student_t_log_kernel(r, sigma),
                    };
                }
                if normalize && let (Some(lo), Some(hi)) = (scale_lo, scale_hi) {
                    let sigma_bar = 0.5 * (lo + hi);
                    c += -(scale_factor * sigma_bar).ln();
                }
                c
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
            0.5 * (lo + hi)
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

    /// Evaluate the log-prior contribution for this parameter: a hard
    /// [`Penalty::top_hat`] wall plus, if present, a Gaussian
    /// [`Penalty::Center`]. Priors are never inflated by `f_sigma`
    /// (`scale_factor = 1.0`) and carry no anchor.
    pub(super) fn log_prob(&self, x: f64) -> f64 {
        let mut lp = Penalty::top_hat(self.bounds.0, self.bounds.1).cost(x, 1.0);
        if let Some((mean, scale_lo, scale_hi)) = self.gaussian {
            lp += Penalty::Center {
                mean,
                scale_lo: Some(scale_lo),
                scale_hi: Some(scale_hi),
                tail: Tail::Gaussian,
                normalize: false,
            }
            .cost(x, 1.0);
        }
        lp
    }

    /// Midpoint of the bounds, or the Gaussian mean if set.
    pub(crate) fn center(&self) -> f64 {
        self.gaussian
            .map_or(0.5 * (self.bounds.0 + self.bounds.1), |(m, _, _)| m)
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

/// Logistic barrier prior: smooth wall that is 0 in the interior and -> -inf at
/// the boundaries.
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
#[must_use]
pub(super) fn logistic_barrier(x: f64, lo: f64, hi: f64, k: f64) -> f64 {
    // ln(sigmoid(z)) = z - ln(1 + exp(z)) but for numerical stability use -ln(1+exp(-z))
    // which is equivalent and avoids overflow for large positive z.
    fn log_sigmoid(z: f64) -> f64 {
        if z > 0.0 {
            -(-z).exp().ln_1p()
        } else {
            z - z.exp().ln_1p()
        }
    }
    log_sigmoid(k * (x - lo)) + log_sigmoid(k * (hi - x))
}

/// Result of evaluating the forward model at a parameter point.
pub(super) struct ForwardModelResult {
    /// Model flux per observation (Jy), in obs-global index order.
    pub model_fluxes: Vec<f64>,
    /// Reflected-light fraction per observation.
    pub reflected_frac: Vec<f64>,
}
