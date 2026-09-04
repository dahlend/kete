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
    BandInfo, EllipsoidTemplate, ModelResults, RoughnessCorrection, SpinState, ThermalParams,
    TpmFieldGrid, TpmShape, flux_to_mag, frm_total_flux, gamma_from_mean_slope, hg_apparent_flux,
    hg_apparent_mag, mag_to_flux, neatm_total_flux, sub_solar_temperature, tpm_total_flux_cached,
};
use kete_core::constants::V_MAG_ZERO;
use kete_core::errors::{Error, KeteResult};
use nalgebra::Vector3;
use std::sync::Arc;

/// Fixed surface roughness for a TPM fit: the precomputed [`RoughnessCorrection`]
/// applied to the smooth flux. Roughness is an input (it is strongly degenerate with
/// thermal inertia), not a fitted parameter.
///
/// User-facing roughness is the mean slope angle `theta_bar` (see
/// [`mean_slope_angle`](crate::mean_slope_angle)); `gamma` here is the internal crater
/// opening half-angle it maps to, which is what the correction table is keyed on.
#[derive(Debug, Clone)]
pub struct RoughnessFit {
    /// Internal crater opening half-angle (radians), derived from the configured mean
    /// slope angle via [`gamma_from_mean_slope`](crate::gamma_from_mean_slope).
    pub gamma: f64,
    /// Shared correction table covering the relevant `Theta`/phase range.
    pub correction: Arc<RoughnessCorrection>,
}

/// Fixed TPM configuration for fitting: spin state, body shape, and a precomputed
/// field grid.
///
/// Spin and shape are inputs (from lightcurve inversion), not fitted parameters. The
/// grid is shared (via `Arc`) across chains and observations to amortize the heat
/// solve.
#[derive(Debug, Clone)]
pub struct TpmConfig {
    /// Spin state (pole, rotation period, phase).
    pub spin: SpinState,
    /// Body-fixed shape (sphere, ellipsoid, or custom mesh).
    pub shape: TpmShape,
    /// Coarse (low-facet) shape used only as the cheap forward model for the
    /// Nelder-Mead seed and whitening; the posterior chains use the full `shape`.
    /// For custom meshes (which cannot be coarsened) this is just `shape`.
    pub seed_shape: TpmShape,
    /// Precomputed diurnal field grid covering the relevant `Theta` range.
    pub grid: Arc<TpmFieldGrid>,
    /// Optional surface roughness applied via the correction table.
    pub roughness: Option<RoughnessFit>,
    /// Fit the oblate axis ratio `c/a` (`b/a` fixed at 1) as a free parameter. When any
    /// shape/phase extra is fit, the shape is rebuilt per step from `template` at the
    /// sampled axis ratios, so `shape` / `seed_shape` are ignored. The
    /// `(Theta, sub-solar-lat)` cache is shape-independent, so this adds only a re-mesh
    /// + re-render, no heat solve.
    pub fit_c_a: bool,
    /// Fit the axis ratio `b/a` as a free parameter (a triaxial body). Implies fitting
    /// `c/a` too; a triaxial shape produces a rotational lightcurve.
    pub fit_b_a: bool,
    /// Fit the rotation phase `phase0` (radians) as a free parameter. Relevant only for
    /// a non-axisymmetric (triaxial) shape. Restricted to `[0, pi)` -- an ellipsoid is
    /// point-symmetric, so `phase0` and `phase0 + pi` are degenerate.
    pub fit_phase0: bool,
    /// Reusable ellipsoid tessellation used to rebuild the shape cheaply when an axis
    /// ratio is fit. Resolution should match `shape`/`seed_shape`.
    pub template: Option<EllipsoidTemplate>,
}

impl TpmConfig {
    /// Build the configuration for a TPM fit from the observations and priors:
    /// construct the sampling shapes, span a field grid over the `Theta` range
    /// implied by the thermal-inertia prior and the observation distances, and
    /// set up the roughness correction table when requested.
    ///
    /// `mean_slope` is the surface mean slope angle in radians (see
    /// [`gamma_from_mean_slope`]). `fit_shape` is `(fit_c_a, fit_b_a, fit_phase0)`.
    /// When `fit_roughness` is set the correction table is loaded even without a
    /// fixed `mean_slope`; the prior center seeds the fallback crater angle.
    ///
    /// # Errors
    /// Returns `ValueError` if no valid thermal-parameter range can be derived
    /// from the observations, or if the shipped roughness table does not cover
    /// the observation bands.
    #[allow(
        clippy::too_many_arguments,
        reason = "TPM configuration has many inputs"
    )]
    pub fn for_fit(
        obs: &[FluxObs],
        spin: SpinState,
        axis_ratios: Option<(f64, f64)>,
        mean_slope: Option<f64>,
        fit_roughness: bool,
        fit_shape: (bool, bool, bool),
        emissivity: f64,
        priors: &FluxPriors,
    ) -> KeteResult<Self> {
        // Sampling shape facet counts. A smooth convex disk-integrated flux converges
        // quickly in facet count: ~256 facets match the full 2048-facet mesh to <0.1%
        // in flux (see `facet_count_study`) at ~8x lower render cost, which dominates
        // the fit. The Nelder-Mead seed only locates the basin, so it uses an even
        // coarser mesh. (Triaxial shapes have a rotational lightcurve and may warrant
        // more facets; the common axisymmetric/oblate case is smooth and converges
        // fastest.)
        const FIT_SPHERE_FACETS: u32 = 256;
        const FIT_ELLIPSOID_DIV: u32 = 6; // 8 * 6^2 = 288 facets
        const SEED_SPHERE_FACETS: u32 = 128;
        const SEED_ELLIPSOID_DIV: u32 = 4; // 8 * 4^2 = 128 facets

        let (fit_c_a, fit_b_a, fit_phase0) = fit_shape;
        let (shape, seed_shape) = match axis_ratios {
            Some((b_over_a, c_over_a)) => (
                TpmShape::ellipsoid_with_div(FIT_ELLIPSOID_DIV, 1.0, b_over_a, c_over_a),
                TpmShape::ellipsoid_with_div(SEED_ELLIPSOID_DIV, 1.0, b_over_a, c_over_a),
            ),
            None => (
                TpmShape::sphere_with_facets(FIT_SPHERE_FACETS),
                TpmShape::sphere_with_facets(SEED_SPHERE_FACETS),
            ),
        };

        // When fitting any axis ratio, the shape is rebuilt per step from this
        // template at the sampled (b/a, c/a), so `shape`/`seed_shape` above are
        // unused.
        let template = (fit_c_a || fit_b_a).then(|| EllipsoidTemplate::new(FIT_ELLIPSOID_DIV));

        // Span the grid over the prior's thermal-inertia bounds across all
        // observation distances, padded so the sampler stays inside the grid. T_ss
        // depends only weakly on albedo, so a fiducial value is used for the range
        // estimate. Theta scales linearly with thermal inertia, so a unit-inertia
        // [`ThermalParams::thermal_parameter`] gives the per-observation scale.
        let (gamma_lo, gamma_hi) = priors.thermal_inertia.bounds;
        let unit_inertia = ThermalParams {
            thermal_inertia: 1.0,
            emissivity,
        };
        let mut theta_min = f64::INFINITY;
        let mut theta_max: f64 = 0.0;
        for ob in obs {
            let sun_dist = ob.sun2obj.norm();
            let t_ss = sub_solar_temperature(sun_dist, 0.1, 0.15, 1.0, emissivity);
            if t_ss <= 0.0 {
                continue;
            }
            let scale = unit_inertia.thermal_parameter(spin.period, t_ss);
            theta_min = theta_min.min(gamma_lo * scale);
            theta_max = theta_max.max(gamma_hi * scale);
        }
        if !theta_max.is_finite() || theta_max <= 0.0 {
            return Err(Error::ValueError(
                "Could not determine a valid thermal-parameter range from the observations.".into(),
            ));
        }
        // Padded range for the field grid (keeps the sampler off the grid edges).
        let grid_theta_min = (theta_min * 0.5).max(1e-4);
        let grid_theta_max = theta_max * 2.0;

        // Surface roughness: load the shipped correction table and apply it cheaply
        // per evaluation. When roughness is fitted the table is still required (the
        // sampled mean slope drives it); the stored `gamma` is only a fallback
        // placeholder, taken from the prior center.
        let roughness = if mean_slope.is_some() || fit_roughness {
            let theta_bar = mean_slope.unwrap_or_else(|| priors.roughness.center());
            let table = RoughnessCorrection::shipped();
            // Fail loudly if the table does not cover the observation bands (the
            // nearest-band snap would be a silently wrong correction). The Theta /
            // T_ss axes clamp gracefully at their asymptotic edges, so they are not
            // gated here.
            validate_roughness_coverage(&table, obs)?;
            Some(RoughnessFit {
                gamma: gamma_from_mean_slope(theta_bar),
                correction: Arc::new(table),
            })
        } else {
            None
        };

        Ok(Self {
            spin,
            shape,
            seed_shape,
            grid: Arc::new(TpmFieldGrid::new(grid_theta_min, grid_theta_max)?),
            roughness,
            fit_c_a,
            fit_b_a,
            fit_phase0,
            template,
        })
    }
}

/// Verify the shipped roughness correction table actually covers a fit's observation
/// bands. Outside its band set the table snaps to the nearest tabulated band, which
/// for a far-off wavelength is a silently wrong correction -- so surface that as a
/// clear up-front error. (The table's thermal-parameter and sub-solar-temperature
/// axes clamp *gracefully* at their edges -- those regimes are the smooth low/high
/// asymptotes -- so they are documented, not gated.)
fn validate_roughness_coverage(table: &RoughnessCorrection, obs: &[FluxObs]) -> KeteResult<()> {
    // The table's WISE bands are far apart, so require each observation to sit within
    // 15% of a tabulated band -- else the nearest-band snap is meaningless.
    const WAVELENGTH_TOL: f64 = 0.15;
    let table_wl = table.wavelengths();
    for ob in obs {
        let wl = ob.band.wavelength;
        let nearest = table_wl
            .iter()
            .copied()
            .min_by(|a, b| (a - wl).abs().total_cmp(&(b - wl).abs()))
            .unwrap_or(f64::NAN);
        if !nearest.is_finite() || (nearest - wl).abs() > WAVELENGTH_TOL * wl {
            return Err(Error::ValueError(format!(
                "roughness needs a correction table covering every observation band, but \
                 the shipped table (WISE bands {table_wl:?} nm) has none within \
                 {pct:.0}% of {wl:.0} nm. Fit without roughness, or use the on-the-fly \
                 rough forward model, which computes at the true wavelength.",
                pct = WAVELENGTH_TOL * 100.0,
            )));
        }
    }
    Ok(())
}

/// Which shape / phase quantities a TPM fit treats as free parameters. Appended (in
/// this order) to the end of the TPM parameter vector so the base indices stay stable.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ShapeFit {
    pub fit_c_a: bool,
    pub fit_b_a: bool,
    pub fit_phase0: bool,
}

impl ShapeFit {
    /// Derive the descriptor from an optional config (all-false for non-TPM fits).
    pub(crate) fn from_config(tpm: Option<&TpmConfig>) -> Self {
        tpm.map_or(Self::default(), |c| Self {
            fit_c_a: c.fit_c_a,
            fit_b_a: c.fit_b_a,
            fit_phase0: c.fit_phase0,
        })
    }

    /// Number of appended free parameters.
    pub(crate) fn n_extra(self) -> usize {
        usize::from(self.fit_c_a) + usize::from(self.fit_b_a) + usize::from(self.fit_phase0)
    }
}

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
    /// Thermophysical model -- thermal inertia is a free parameter; spin is a fixed
    /// input. Requires a [`TpmConfig`].
    Tpm,
    /// Thermophysical model with surface roughness (mean slope angle `theta_bar`) as an
    /// additional free parameter, applied via the correction table. Requires a
    /// [`TpmConfig`] carrying a [`RoughnessFit`]. Roughness is partially degenerate
    /// with thermal inertia, so it needs multi-band and phase-angle coverage to
    /// separate the two.
    TpmRough,
}

impl Model {
    /// Number of free parameters.
    ///
    /// NEATM:      `[D, beaming, H, G, f_sigma, R_IR]`                    -> 6.
    /// TPM:        `[D, thermal_inertia, H, G, f_sigma, R_IR]`            -> 6.
    /// `TpmRough`: `[D, thermal_inertia, roughness, H, G, f_sigma, R_IR]` -> 7.
    /// FRM:        `[D, H, G, f_sigma, R_IR]`                             -> 5.
    /// HG:         `[H, G, f_sigma]`                                      -> 3.
    pub(crate) fn dim(self, shape_fit: ShapeFit) -> usize {
        let base = match self {
            Self::TpmRough => 7,
            Self::Neatm | Self::Tpm => 6,
            Self::Frm => 5,
            Self::Hg => 3,
        };
        // Fitted shape/phase extras are appended at the end of the TPM vector.
        base + if self.is_tpm() {
            shape_fit.n_extra()
        } else {
            0
        }
    }

    /// Whether this is the NEATM model.
    #[must_use]
    pub fn is_neatm(self) -> bool {
        matches!(self, Self::Neatm)
    }

    /// Whether this is a TPM model (smooth or with fitted roughness).
    #[must_use]
    pub fn is_tpm(self) -> bool {
        matches!(self, Self::Tpm | Self::TpmRough)
    }

    /// Whether this TPM model fits surface roughness as a free parameter.
    #[must_use]
    pub fn fits_roughness(self) -> bool {
        matches!(self, Self::TpmRough)
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
    pub(crate) fn draw_column_names(self, shape_fit: ShapeFit) -> Vec<&'static str> {
        let mut cols: Vec<&'static str> = match self {
            Self::Neatm => vec![
                "diameter",
                "vis_albedo",
                "beaming",
                "h_mag",
                "g_param",
                "r_ir",
                "f_sigma",
            ],
            Self::Tpm => vec![
                "diameter",
                "vis_albedo",
                "thermal_inertia",
                "h_mag",
                "g_param",
                "r_ir",
                "f_sigma",
            ],
            Self::TpmRough => vec![
                "diameter",
                "vis_albedo",
                "thermal_inertia",
                "roughness",
                "h_mag",
                "g_param",
                "r_ir",
                "f_sigma",
            ],
            Self::Frm => vec![
                "diameter",
                "vis_albedo",
                "h_mag",
                "g_param",
                "r_ir",
                "f_sigma",
            ],
            Self::Hg => vec!["h_mag", "g_param", "f_sigma"],
        };
        // Fitted shape/phase extras are appended last, in the same order as
        // `to_draw_row` / `unpack` (c_a, b_a, phase0).
        if self.is_tpm() {
            if shape_fit.fit_c_a {
                cols.push("c_a");
            }
            if shape_fit.fit_b_a {
                cols.push("b_a");
            }
            if shape_fit.fit_phase0 {
                cols.push("phase0");
            }
        }
        cols
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
    pub(crate) fn unpack(
        self,
        x: &[f64],
        emissivity: f64,
        c_hg: f64,
        shape_fit: ShapeFit,
    ) -> ModelParams {
        if self.is_hg() {
            let h_mag = x[0];
            let vis_albedo = 1.0;
            let diameter = c_hg * 10.0_f64.powf(-h_mag / 5.0);
            return ModelParams {
                diameter,
                beaming: f64::NAN,
                thermal_inertia: f64::NAN,
                roughness: f64::NAN,
                h_mag,
                g_param: x[1],
                emissivity,
                f_sigma: x[2],
                r_ir: f64::NAN,
                vis_albedo,
                c_a: f64::NAN,
                b_a: f64::NAN,
                phase0: f64::NAN,
            };
        }

        // Thermal models share the same trailing layout:
        //   [D, (beaming | ln thermal_inertia)?, (roughness)?, H, G, f_sigma, R_IR]
        // NEATM has a free beaming at x[1]; TPM has a free thermal inertia at x[1],
        // sampled in log space (Gamma spans decades), so x[1] = ln(Gamma) and is
        // exponentiated here; TpmRough adds a free roughness (crater half-angle,
        // radians) at x[2]; FRM has neither (beaming fixed at pi). `h` is the index of
        // H magnitude.
        let diameter = x[0];
        let (beaming, thermal_inertia, roughness, h) = match self {
            Self::Neatm => (x[1], f64::NAN, f64::NAN, 2),
            Self::Tpm => (f64::NAN, x[1].exp(), f64::NAN, 2),
            Self::TpmRough => (f64::NAN, x[1].exp(), x[2], 3),
            Self::Frm | Self::Hg => (std::f64::consts::PI, f64::NAN, f64::NAN, 1),
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

        // Fitted shape/phase extras (TPM only) are appended after r_ir, in order
        // c_a, b_a, phase0, each present only if fit.
        let mut idx = h + 4;
        let mut take = |fit: bool| {
            if fit && self.is_tpm() {
                let v = x[idx];
                idx += 1;
                v
            } else {
                f64::NAN
            }
        };
        let c_a = take(shape_fit.fit_c_a);
        let b_a = take(shape_fit.fit_b_a);
        let phase0 = take(shape_fit.fit_phase0);

        ModelParams {
            diameter,
            beaming,
            thermal_inertia,
            roughness,
            h_mag,
            g_param: x[h + 1],
            emissivity,
            f_sigma: x[h + 2],
            r_ir: x[h + 3],
            vis_albedo,
            c_a,
            b_a,
            phase0,
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
    pub thermal_inertia: f64,
    /// Mean slope angle `theta_bar` (radians); `NaN` unless the model fits roughness.
    pub roughness: f64,
    pub h_mag: f64,
    pub g_param: f64,
    pub emissivity: f64,
    pub f_sigma: f64,
    pub r_ir: f64,
    pub vis_albedo: f64,
    /// Axis ratio `c/a`; `NaN` unless the TPM fits a shape.
    pub c_a: f64,
    /// Axis ratio `b/a`; `NaN` unless the TPM fits a triaxial shape.
    pub b_a: f64,
    /// Rotation phase `phase0` (radians); `NaN` unless the TPM fits it.
    pub phase0: f64,
}

impl ModelParams {
    /// Convert to a draw row in physical space.
    ///
    /// NEATM: `[diameter, vis_albedo, beaming, h_mag, g_param, r_ir, f_sigma]`
    /// FRM:   `[diameter, vis_albedo, h_mag, g_param, r_ir, f_sigma]`
    /// HG:    `[h_mag, g_param, f_sigma]`
    pub(crate) fn to_draw_row(&self, model: Model, shape_fit: ShapeFit) -> Vec<f64> {
        if model.is_hg() {
            return vec![self.h_mag, self.g_param, self.f_sigma];
        }
        let mut row = vec![self.diameter, self.vis_albedo];
        if model.is_neatm() {
            row.push(self.beaming);
        } else if model.is_tpm() {
            row.push(self.thermal_inertia);
            // Roughness draws are reported as the mean slope angle in degrees.
            if model.fits_roughness() {
                row.push(self.roughness.to_degrees());
            }
        }
        row.extend_from_slice(&[self.h_mag, self.g_param, self.r_ir, self.f_sigma]);
        // Fitted shape/phase extras appended last (matches `draw_column_names`).
        // phase0 is reported in degrees to match the angle convention.
        if model.is_tpm() {
            if shape_fit.fit_c_a {
                row.push(self.c_a);
            }
            if shape_fit.fit_b_a {
                row.push(self.b_a);
            }
            if shape_fit.fit_phase0 {
                row.push(self.phase0.to_degrees());
            }
        }
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
        epoch: f64,
        tpm: Option<&TpmConfig>,
    ) -> ModelResults {
        let band_albedos: Vec<f64> = bands
            .iter()
            .map(|_| params.r_ir * params.vis_albedo)
            .collect();
        match self {
            Self::Tpm | Self::TpmRough => {
                let cfg = tpm.expect("TPM model requires a TpmConfig");
                let thermal = ThermalParams {
                    thermal_inertia: params.thermal_inertia,
                    emissivity: params.emissivity,
                };
                // When fitting axis ratios, rebuild the shape cheaply from the template
                // at the sampled (b/a, c/a); the field cache is unaffected. A fitted
                // phase0 overrides the configured rotation phase.
                let fitted = if params.c_a.is_finite() || params.b_a.is_finite() {
                    let b_a = if params.b_a.is_finite() {
                        params.b_a
                    } else {
                        1.0
                    };
                    let c_a = if params.c_a.is_finite() {
                        params.c_a
                    } else {
                        1.0
                    };
                    cfg.template.as_ref().map(|t| t.shape(1.0, b_a, c_a))
                } else {
                    None
                };
                let shape = fitted.as_ref().unwrap_or(&cfg.shape);
                let spin = if params.phase0.is_finite() {
                    let mut s = cfg.spin.clone();
                    s.phase0 = params.phase0;
                    s
                } else {
                    cfg.spin.clone()
                };
                let mut result = tpm_total_flux_cached(
                    &cfg.grid,
                    bands,
                    &band_albedos,
                    &spin,
                    shape,
                    &thermal,
                    params.diameter,
                    params.vis_albedo,
                    params.g_param,
                    params.h_mag,
                    sun2obj,
                    sun2obs,
                    epoch,
                );
                // Apply the roughness correction (fast) to the smooth thermal flux,
                // preserving the reflected-light contribution.
                if let Some(rough) = &cfg.roughness {
                    // When roughness is fitted (TpmRough) the sampled mean slope angle
                    // drives the correction; otherwise the fixed config value is used.
                    // The correction table is keyed by the internal crater opening
                    // half-angle, so convert the mean slope angle to gamma first.
                    let gamma = if params.roughness.is_finite() {
                        gamma_from_mean_slope(params.roughness)
                    } else {
                        rough.gamma
                    };
                    let t_ss = sub_solar_temperature(
                        sun2obj.norm(),
                        params.vis_albedo,
                        params.g_param,
                        1.0,
                        params.emissivity,
                    );
                    if t_ss > 0.0 && cfg.spin.period > 0.0 {
                        let theta = thermal.thermal_parameter(cfg.spin.period, t_ss);
                        let obj2sun = (-sun2obj).normalize();
                        let obj2obs = (sun2obs - sun2obj).normalize();
                        let phase = obj2sun.dot(&obj2obs).clamp(-1.0, 1.0).acos();
                        for ((band, tf), total) in bands
                            .iter()
                            .zip(result.thermal_fluxes.iter_mut())
                            .zip(result.fluxes.iter_mut())
                        {
                            let refl = *total - *tf;
                            *tf *= rough.correction.factor_at_wavelength(
                                theta,
                                gamma,
                                phase,
                                t_ss,
                                band.wavelength,
                            );
                            *total = *tf + refl;
                        }
                    }
                }
                result
            }
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
        tpm: Option<&TpmConfig>,
    ) -> ForwardModelResult {
        let n = obs.len();
        let mut model_fluxes = Vec::with_capacity(n);
        let mut reflected_frac = Vec::with_capacity(n);

        for ob in obs {
            let bands = [ob.band];
            let result =
                self.compute_fluxes(params, &bands, &ob.sun2obj, &ob.sun2obs, ob.epoch, tpm);
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
    pub(super) fn log_likelihood(
        self,
        params: &ModelParams,
        obs: &[FluxObs],
        tpm: Option<&TpmConfig>,
    ) -> f64 {
        let fwd = self.evaluate_forward_model(params, obs, tpm);

        let nu = STUDENT_NU;
        let mut ll = 0.0;
        for (i, ob) in obs.iter().enumerate() {
            let mf = fwd.model_fluxes[i];
            let sigma_eff = params.f_sigma * ob.sigma;
            let sigma2 = sigma_eff * sigma_eff;
            if ob.is_upper_limit {
                if mf > ob.flux {
                    let r = mf - ob.flux;
                    ll += -0.5 * (nu + 1.0) * (1.0 + r * r / (nu * sigma2)).ln();
                }
            } else {
                let r = ob.flux - mf;
                ll += -sigma_eff.ln() - f64::midpoint(nu, 1.0) * (1.0 + r * r / (nu * sigma2)).ln();
            }
        }

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
        } else if self.is_tpm() {
            // Thermal inertia uses a log-uniform (Jeffreys) prior: a flat,
            // barrier-bounded density in log(Gamma). This is the standard choice for a
            // scale parameter spanning decades and matches how the literature reports
            // Gamma (log-space medians). The user-facing bounds are in linear Gamma and
            // converted to log here; Gamma is also sampled in log space (see `unpack`),
            // so the barrier acts directly on the sampled coordinate.
            let ln_gamma = params.thermal_inertia.ln();
            let (gamma_lo, gamma_hi) = priors.thermal_inertia.bounds;
            lp += logistic_barrier(
                ln_gamma,
                gamma_lo.max(f64::MIN_POSITIVE).ln(),
                gamma_hi.max(f64::MIN_POSITIVE).ln(),
                BARRIER_K,
            );
            if let Some((mean, sigma_lo, sigma_hi)) = priors.thermal_inertia.gaussian {
                // An optional Gaussian center on Gamma is applied in log space (mean in
                // linear Gamma, sigmas as log-widths).  The asymmetric-error refactor
                // replaced the standalone gaussian_log_prior with a Penalty::Center
                // term; Tail::Gaussian + normalize:false reproduces the old
                // -(x-mu)^2/(2 sigma^2) exactly for the symmetric case.
                lp += Penalty::Center {
                    mean: mean.max(f64::MIN_POSITIVE).ln(),
                    scale_lo: Some(sigma_lo),
                    scale_hi: Some(sigma_hi),
                    tail: Tail::Gaussian,
                    normalize: false,
                }
                .cost(ln_gamma, 1.0);
            }
            if self.fits_roughness() {
                lp += priors.roughness.log_prob(params.roughness);
            }
            // Fitted shape/phase extras are signaled by a finite value (NaN otherwise).
            if params.c_a.is_finite() {
                lp += priors.c_a.log_prob(params.c_a);
            }
            if params.b_a.is_finite() {
                lp += priors.b_a.log_prob(params.b_a);
            }
            if params.phase0.is_finite() {
                lp += priors.phase0.log_prob(params.phase0);
            }
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
        tpm: Option<&TpmConfig>,
    ) -> f64 {
        let params = self.unpack(x, emissivity, c_hg, ShapeFit::from_config(tpm));
        let ll = self.log_likelihood(&params, obs, tpm);
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
    /// Observation time (Julian date). Used by the TPM model to set the rotation
    /// phase; ignored by NEATM/FRM/HG.
    pub epoch: f64,
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
        epoch: f64,
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
            epoch,
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
        epoch: f64,
    ) -> Self {
        Self::detection_asym(flux, sigma, sigma, band, sun2obj, sun2obs, epoch)
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
        epoch: f64,
    ) -> Self {
        Self::from_parts(
            None,
            Some((flux, Some(sigma_lo), Some(sigma_hi))),
            band,
            sun2obj,
            sun2obs,
            epoch,
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
        epoch: f64,
    ) -> Self {
        Self::from_parts(
            None,
            Some((threshold, None, Some(sigma))),
            band,
            sun2obj,
            sun2obs,
            epoch,
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
        epoch: f64,
    ) -> Self {
        Self::from_parts(Some((lo, hi)), None, band, sun2obj, sun2obs, epoch)
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
        self.penalties
            .iter()
            .map(|p| p.cost(model_flux, f_sigma))
            .sum()
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
        assert!(
            hi > lo,
            "TopHat interval requires lo < hi, got ({lo}, {hi})"
        );
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
    #[must_use]
    pub fn center(&self) -> f64 {
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
    /// Prior on thermal inertia `Gamma` (SI units). Used by TPM. `Gamma` is sampled
    /// and prior'd in log space: the `bounds` are linear `Gamma` but the prior is
    /// log-uniform within them (the standard scale-parameter / Jeffreys choice). An
    /// optional `gaussian` center is therefore also interpreted in log space -- `mean`
    /// is a linear `Gamma` (the lognormal median) and `sigma` is a multiplicative /
    /// log-width, not a linear standard deviation.
    pub thermal_inertia: ParamPrior,
    /// Prior on surface roughness as the mean slope angle `theta_bar` (radians). Used by
    /// the roughness-fitting TPM variant. The cap geometry caps `theta_bar` at ~1 rad
    /// (57.3 deg), so bounds should stay below that ceiling.
    pub roughness: ParamPrior,
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
    /// Prior on the axis ratio `c/a`. Used only when the TPM fits a shape.
    pub c_a: ParamPrior,
    /// Prior on the axis ratio `b/a`. Used only when the TPM fits a triaxial shape.
    pub b_a: ParamPrior,
    /// Prior on the rotation phase `phase0` (radians). Used only when the TPM fits it.
    pub phase0: ParamPrior,
}

impl Default for FluxPriors {
    /// Sensible defaults (all in linear/physical units):
    /// - diameter in [0.001, 1000] km (bounds only)
    /// - beaming in [0.5, 3.0], Gaussian(1.0, 0.3)
    /// - `thermal_inertia` in [1, 2500] SI (log-uniform within bounds)
    /// - roughness (mean slope angle) in [0, 50] deg, bounds only (radians internally)
    /// - `r_ir` in [0.5, 2.0], Gaussian(1.6, 0.3)
    /// - `h_mag` in [-5, 35] (bounds only)
    /// - `g_param` in [-0.3, 0.7], Gaussian(0.2, 0.05)
    /// - `vis_albedo` in [0.01, 1] (bounds only)
    /// - `f_sigma` in [0.5, 5.0] (bounds only)
    fn default() -> Self {
        Self {
            diameter: ParamPrior::bounds_only(0.001, 1000.0),
            beaming: ParamPrior::with_gaussian(0.5, 3.0, 1.0, 0.3),
            thermal_inertia: ParamPrior::bounds_only(1.0, 2500.0),
            // Mean slope angle, 0-50 deg in radians (margin below the ~57 deg ceiling).
            roughness: ParamPrior::bounds_only(0.0_f64.to_radians(), 50.0_f64.to_radians()),
            r_ir: ParamPrior::with_gaussian(0.5, 2.0, 1.6, 0.3),
            h_mag: ParamPrior::bounds_only(-5.0, 35.0),
            g_param: ParamPrior::with_gaussian(-0.3, 0.7, 0.2, 0.05),
            vis_albedo: ParamPrior::bounds_only(0.01, 1.0),
            f_sigma: ParamPrior::bounds_only(0.5, 5.0),
            // oblate c/a in (0, 1], gently centered at 0.85.
            c_a: ParamPrior::with_gaussian(0.1, 1.0, 0.85, 0.2),
            // b/a in (0, 1], gently centered at 0.85.
            b_a: ParamPrior::with_gaussian(0.1, 1.0, 0.85, 0.2),
            // phase0 restricted to [0, pi) (the ellipsoid's point-symmetry domain).
            phase0: ParamPrior::bounds_only(0.0, std::f64::consts::PI),
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
