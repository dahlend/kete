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

use super::*;
use crate::fitting::types::{Penalty, Tail, logistic_barrier};
use crate::{BandInfo, frm_total_flux, neatm_total_flux, resolve_hg_params};
use kete_core::constants::C_V;
use nalgebra::Vector3;

/// Resolved HG parameters for testing.
struct TestHg {
    g_param: f64,
    h_mag: f64,
    vis_albedo: f64,
    diameter: f64,
}

impl TestHg {
    fn new(
        g_param: f64,
        h_mag: Option<f64>,
        vis_albedo: Option<f64>,
        diameter: Option<f64>,
    ) -> Self {
        let (h_mag, vis_albedo, diameter) =
            resolve_hg_params(h_mag, vis_albedo, diameter, None).unwrap();
        Self {
            g_param,
            h_mag,
            vis_albedo,
            diameter,
        }
    }
}

/// Build a negative-log-likelihood closure for a model.
fn make_neg_log_likelihood(
    model: Model,
    obs: &[FluxObs],
    c_hg: f64,
    emissivity: f64,
) -> impl Fn(&[f64]) -> f64 {
    let obs = obs.to_vec();
    move |x: &[f64]| -> f64 {
        let params = model.unpack(x, emissivity, c_hg);
        let ll = model.log_likelihood(&params, &obs);
        if ll.is_finite() { -ll } else { f64::MAX }
    }
}

/// Build a synthetic observation set: a 10 km asteroid at 2 AU, H=18, G=0.15.
fn synthetic_neatm_obs() -> (Vec<FluxObs>, TestHg) {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));

    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    // observer at Sun for simplicity
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);

    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;
    let beaming = 1.2;
    let r_ir = 1.0;
    let emissivity = 0.9;

    // Generate "observed" fluxes from the forward model
    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * vis_albedo).collect();
    let result = neatm_total_flux(
        &bands,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        beaming,
        emissivity,
        &sun2obj,
        &sun2obs,
    );

    let obs: Vec<FluxObs> = bands
        .into_iter()
        .zip(&result.fluxes)
        // 5% uncertainty
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();

    (obs, hg)
}

#[test]
fn test_logistic_barrier_interior() {
    // Well inside bounds -> close to 0
    let val = logistic_barrier(0.0, -5.0, 5.0, 30.0);
    assert!(val > -1e-3, "Interior barrier should be ~0, got {val}");
}

#[test]
fn test_logistic_barrier_boundary() {
    // At the boundary -> large negative
    let val = logistic_barrier(4.99, -5.0, 5.0, 30.0);
    let val_edge = logistic_barrier(5.5, -5.0, 5.0, 30.0);
    assert!(val_edge < val, "Past boundary should be more negative");
}

#[test]
fn test_asymmetric_gaussian_prior() {
    // Tight below the mean, loose above: a value the same distance below the
    // mean should be penalized far more than one above it.
    let prior = ParamPrior::with_gaussian_asym(-10.0, 10.0, 0.0, 0.1, 1.0);
    let below = prior.log_prob(-0.5);
    let above = prior.log_prob(0.5);
    assert!(
        below < above,
        "Below-mean deviation should cost more with a tight lower side: \
         below={below}, above={above}"
    );

    // The symmetric constructor is exactly sign-symmetric.
    let sym = ParamPrior::with_gaussian(-10.0, 10.0, 0.0, 0.5);
    assert!(
        (sym.log_prob(-0.5) - sym.log_prob(0.5)).abs() < 1e-12,
        "Symmetric Gaussian prior must be sign-symmetric"
    );

    // center() still reports the mean.
    assert!((prior.center() - 0.0).abs() < 1e-12);
}

#[test]
fn test_neatm_nll_at_truth() {
    let (obs, _hg) = synthetic_neatm_obs();
    let neg_log_lik = make_neg_log_likelihood(Model::Neatm, &obs, C_V, 0.9);

    // Truth: [D, beaming, H, G, f_sigma, R_IR]
    let truth = [10.0, 1.2, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_at_truth = neg_log_lik(&truth);

    assert!(
        neg_log_lik_at_truth.is_finite(),
        "NLL at truth should be finite"
    );

    let wrong = [20.0, 1.2, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong,
        "NLL at truth ({neg_log_lik_at_truth}) should be less than at wrong D ({neg_log_lik_wrong})"
    );
}

#[test]
fn test_neatm_fit_recovery() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    let n_obs = obs.len();
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50)
        .expect("MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "Fit diameter {d_median:.2} too far from truth 10.0",
    );

    // MAP diagnostic fields.
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(
        res.reduced_chi2.is_finite() || n_obs <= Model::Neatm.dim(),
        "reduced_chi2 should be finite when nobs > nparams",
    );
}

#[test]
fn test_frm_nll_at_truth() {
    // Build FRM observations similarly
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));

    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;
    let r_ir = 1.0;

    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * vis_albedo).collect();
    let result = frm_total_flux(
        &bands,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        0.9,
        &sun2obj,
        &sun2obs,
    );

    let obs: Vec<FluxObs> = bands
        .into_iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();

    let neg_log_lik = make_neg_log_likelihood(Model::Frm, &obs, C_V, 0.9);
    // Truth: [D, H, G, f_sigma, R_IR]
    let truth = [10.0, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_val = neg_log_lik(&truth);
    assert!(
        neg_log_lik_val.is_finite(),
        "FRM NLL at truth should be finite, got {neg_log_lik_val}"
    );

    // NLL at truth should be less than at a wrong diameter
    let wrong = [20.0, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_val < neg_log_lik_wrong,
        "FRM NLL at truth ({neg_log_lik_val}) should be less than at wrong D ({neg_log_lik_wrong})"
    );
}

#[test]
fn test_upper_limit_penalty() {
    let (obs, _hg) = synthetic_neatm_obs();

    // Turn all observations into upper limits with threshold = actual flux.
    let ul_obs: Vec<FluxObs> = obs
        .into_iter()
        .map(|o| {
            let threshold = o.point_estimate().unwrap();
            let sigma = o.sigma_lo().unwrap();
            FluxObs::upper_limit(threshold, sigma, o.band, o.sun2obj, o.sun2obs)
        })
        .collect();

    let neg_log_lik = make_neg_log_likelihood(Model::Neatm, &ul_obs, C_V, 0.9);

    // At truth, model == threshold -> NLL should be 0
    // (upper limits don't include the ln(sigma) term)
    let truth = [10.0, 1.2, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_at_truth = neg_log_lik(&truth);
    assert!(
        neg_log_lik_at_truth < 1e-6,
        "Upper limit NLL at truth should be ~0, got {neg_log_lik_at_truth}"
    );

    // Larger diameter -> more flux -> exceeds threshold -> penalty
    let bigger = [20.0, 1.2, 18.0, 0.15, 1.0, 1.0];
    let neg_log_lik_bigger = neg_log_lik(&bigger);
    assert!(
        neg_log_lik_bigger > neg_log_lik_at_truth,
        "Larger D should incur upper-limit penalty"
    );
}

#[test]
fn test_split_normal_calibration() {
    // Split (two-piece) kernel: each side's quoted sigma applies strictly, so a
    // 1-sigma deviation on either side scores exactly z = 1 -- calibrated in
    // the 1-2 sigma region where inference lives, not just the far tails.
    let center = Penalty::Center {
        mean: 0.0,
        scale_lo: Some(0.02),
        scale_hi: Some(0.20),
        tail: Tail::Gaussian,
        normalize: false,
    };
    // x below mean (r > 0) uses scale_lo; x above mean (r < 0) uses scale_hi.
    assert!(
        (center.cost(-0.02, 1.0) - (-0.5)).abs() < 1e-12,
        "1-sigma tight-side deviation must cost exactly -0.5"
    );
    assert!(
        (center.cost(0.20, 1.0) - (-0.5)).abs() < 1e-12,
        "1-sigma wide-side deviation must cost exactly -0.5"
    );
    // 2-sigma on the tight side.
    assert!((center.cost(-0.04, 1.0) - (-2.0)).abs() < 1e-12);

    // Continuous through r = 0: the kernel and its gradient vanish from both
    // sides, so the sampler's finite differences see no jump at the mean.
    let eps = 1e-9;
    assert!(center.cost(eps, 1.0).abs() < 1e-12);
    assert!(center.cost(-eps, 1.0).abs() < 1e-12);
    assert!(center.cost(0.0, 1.0).abs() < 1e-12);
}

#[test]
fn test_penalty() {
    // A centering term at 0 with the given per-side scales, tail, normalize.
    let center = |scale_lo, scale_hi, tail, normalize| Penalty::Center {
        mean: 0.0,
        scale_lo,
        scale_hi,
        tail,
        normalize,
    };

    // Two-sided Gaussian, no scaling, no anchor (prior-like).
    // z = (mean - x)/sigma = -x; at x = 1, sigma = 1 -> -0.5.
    let g = center(Some(1.0), Some(1.0), Tail::Gaussian, false);
    assert!((g.cost(1.0, 1.0) - (-0.5)).abs() < 1e-12);
    assert!((g.cost(1.0, 1.0) - g.cost(-1.0, 1.0)).abs() < 1e-12);

    // Asymmetric: tighter below center -> a value below center costs more.
    let a = center(Some(0.5), Some(2.0), Tail::Gaussian, false);
    assert!(a.cost(-1.0, 1.0) < a.cost(1.0, 1.0));

    // Upper limit (scale_lo = None): free below center, penalized above.
    let ul = center(None, Some(1.0), Tail::StudentT, true);
    assert!(ul.cost(-5.0, 1.0).abs() < 1e-12, "below center is free");
    assert!(ul.cost(5.0, 1.0) < 0.0, "above center is penalized");
    assert!(ul.cost(0.0, 1.0).abs() < 1e-12, "one-sided => no anchor at center");

    // Lower limit (scale_hi = None): mirror image.
    let ll = center(Some(1.0), None, Tail::StudentT, true);
    assert!(ll.cost(5.0, 1.0).abs() < 1e-12, "above center is free");
    assert!(ll.cost(-5.0, 1.0) < 0.0, "below center is penalized");

    // Normalization anchor: two-sided + normalize adds -ln(scale_factor*sigma_bar).
    // At r = 0 the kernel is 0, so the whole value is the anchor.
    let n_on = center(Some(2.0), Some(2.0), Tail::StudentT, true);
    let n_off = center(Some(2.0), Some(2.0), Tail::StudentT, false);
    assert!(n_off.cost(0.0, 1.0).abs() < 1e-12, "no anchor when normalize=false");
    assert!((n_on.cost(0.0, 1.0) - (-(2.0_f64).ln())).abs() < 1e-12);
    assert!((n_on.cost(0.0, 3.0) - (-(6.0_f64).ln())).abs() < 1e-12, "anchor scales");

    // Top-hat: flat (~0) inside, strongly negative outside, never scaled.
    let b = Penalty::TopHat { lo: 1.0, hi: 2.0, k: 50.0 };
    assert!(b.cost(1.5, 1.0) > -1e-3, "interior is flat");
    assert!(b.cost(0.0, 1.0) < -1.0, "outside is walled off");
    assert!(
        (b.cost(0.0, 1.0) - b.cost(0.0, 5.0)).abs() < 1e-12,
        "hard bounds are not scaled by scale_factor"
    );

    // Composition: a constraint is the sum of its terms.
    let total: f64 = [&b, &g].iter().map(|p| p.cost(1.5, 1.0)).sum();
    assert!((total - (b.cost(1.5, 1.0) + g.cost(1.5, 1.0))).abs() < 1e-12);
}

#[test]
fn test_top_hat_steepness() {
    // Narrow intervals get width-relative steepness: the documented
    // tight-bounds parameter-fixing idiom must actually confine.
    let tight = Penalty::top_hat(9.999, 10.001);
    assert!(tight.cost(10.0, 1.0) > -1e-6, "interior of a tight interval is flat");
    assert!(
        tight.cost(10.021, 1.0) < -100.0,
        "10 interval-widths outside a tight interval must be walled off, got {}",
        tight.cost(10.021, 1.0)
    );

    // Intervals wider than 1 unit keep the legacy fixed steepness exactly.
    let wide = Penalty::top_hat(-5.0, 35.0);
    let legacy = Penalty::TopHat { lo: -5.0, hi: 35.0, k: 50.0 };
    for x in [-6.0, -5.1, 0.0, 20.0, 34.9, 36.0] {
        assert!(
            (wide.cost(x, 1.0) - legacy.cost(x, 1.0)).abs() < 1e-12,
            "wide bounds must match legacy steepness at x={x}"
        );
    }
}

#[test]
#[should_panic(expected = "lo < hi")]
fn test_top_hat_rejects_inverted_bounds() {
    let _ = Penalty::top_hat(2.0, 1.0);
}

#[test]
#[should_panic(expected = "lo < hi")]
fn test_bounded_rejects_degenerate_interval() {
    let base = synthetic_neatm_obs().0;
    let _ = FluxObs::bounded(1.0, 1.0, base[0].band, base[0].sun2obj, base[0].sun2obs);
}

#[test]
fn test_student_t_split_calibration() {
    // The Student-t tail must select sides exactly like the Gaussian: the
    // asymmetric term at a 1-sigma deviation on each side matches a symmetric
    // term whose sigma is that side's scale.
    let center = |scale_lo: f64, scale_hi: f64| Penalty::Center {
        mean: 0.0,
        scale_lo: Some(scale_lo),
        scale_hi: Some(scale_hi),
        tail: Tail::StudentT,
        normalize: false,
    };
    let asym = center(0.02, 0.20);
    // x below mean (r > 0) uses scale_lo; x above mean (r < 0) uses scale_hi.
    assert!(
        (asym.cost(-0.02, 1.0) - center(0.02, 0.02).cost(-0.02, 1.0)).abs() < 1e-12,
        "tight side must use sigma_lo strictly"
    );
    assert!(
        (asym.cost(0.20, 1.0) - center(0.20, 0.20).cost(0.20, 1.0)).abs() < 1e-12,
        "wide side must use sigma_hi strictly"
    );
}

#[test]
fn test_standardized_residual_sides() {
    let base = synthetic_neatm_obs().0;
    let (band, s2o, s2obs) = (base[0].band, base[0].sun2obj, base[0].sun2obs);

    // Upper limit: no residual below the threshold (unconstrained side),
    // signed residual with the f_sigma-inflated scale above it.
    let ul = FluxObs::upper_limit(1.0, 0.1, band, s2o, s2obs);
    assert!(ul.standardized_residual(0.5, 1.0).is_none(), "below threshold is unconstrained");
    let r = ul.standardized_residual(1.5, 2.0).unwrap();
    assert!(((-0.5 / (2.0 * 0.1)) - r).abs() < 1e-12, "expected -2.5, got {r}");

    // Asymmetric detection: each side standardizes by its own sigma.
    let det = FluxObs::detection_asym(1.0, 0.1, 0.4, band, s2o, s2obs);
    let below = det.standardized_residual(0.9, 1.0).unwrap();
    assert!((below - 1.0).abs() < 1e-12, "model below mean uses sigma_lo, got {below}");
    let above = det.standardized_residual(1.2, 1.0).unwrap();
    assert!((above - (-0.5)).abs() < 1e-12, "model above mean uses sigma_hi, got {above}");
}

#[test]
fn test_flux_bounds_constraint() {
    let base = synthetic_neatm_obs().0;
    let ob = FluxObs::bounded(1.0, 2.0, base[0].band, base[0].sun2obj, base[0].sun2obs);

    // Inside the interval: ~flat (penalty ~0). Outside: strong negative penalty.
    let inside = ob.log_likelihood_term(1.5, 1.0);
    let below = ob.log_likelihood_term(0.5, 1.0);
    let above = ob.log_likelihood_term(2.5, 1.0);
    assert!(inside > below && inside > above, "model inside bounds should be preferred");
    assert!(inside > -1e-3, "interior should be ~flat (penalty ~0), got {inside}");

    // Hard bounds are immune to f_sigma inflation.
    assert!((ob.log_likelihood_term(0.5, 1.0) - ob.log_likelihood_term(0.5, 3.0)).abs() < 1e-12);
    // Bounds-only constraints have no standardized residual.
    assert!(ob.standardized_residual(1.5, 1.0).is_none());
}

#[test]
fn test_asymmetric_likelihood_penalizes_tight_side() {
    // Observations are generated from the forward model at `truth`, so evaluating
    // the model at `truth` reproduces each base flux exactly. Shifting the
    // Gaussian mean by +/-delta therefore produces a residual of exactly
    // +/-delta at `truth`.
    let (base, _hg) = synthetic_neatm_obs();
    let truth = [10.0, 1.2, 18.0, 0.15, 1.0, 1.0];

    // Tight lower side, wide upper side. `sign` shifts the estimate vs the model:
    //   sign = +1 -> mean above model -> r > 0 -> tight sigma_lo -> big penalty
    //   sign = -1 -> mean below model -> r < 0 -> wide  sigma_hi -> small penalty
    let make = |sign: f64, frac_lo: f64, frac_hi: f64| -> Vec<FluxObs> {
        base.iter()
            .map(|o| {
                let mean0 = o.point_estimate().unwrap();
                let delta = 0.10 * mean0;
                FluxObs::detection_asym(
                    mean0 + sign * delta,
                    frac_lo * mean0,
                    frac_hi * mean0,
                    o.band,
                    o.sun2obj,
                    o.sun2obs,
                )
            })
            .collect()
    };

    let nll_tight = make_neg_log_likelihood(Model::Neatm, &make(1.0, 0.02, 0.20), C_V, 0.9)(&truth);
    let nll_wide = make_neg_log_likelihood(Model::Neatm, &make(-1.0, 0.02, 0.20), C_V, 0.9)(&truth);
    assert!(
        nll_tight > nll_wide,
        "Equal-magnitude deviation on the tight side should cost more: \
         tight={nll_tight}, wide={nll_wide}"
    );

    // With symmetric errors the two signs must cost the same.
    let nll_pos = make_neg_log_likelihood(Model::Neatm, &make(1.0, 0.11, 0.11), C_V, 0.9)(&truth);
    let nll_neg = make_neg_log_likelihood(Model::Neatm, &make(-1.0, 0.11, 0.11), C_V, 0.9)(&truth);
    assert!(
        (nll_pos - nll_neg).abs() < 1e-9,
        "Symmetric errors must be sign-symmetric: {nll_pos} vs {nll_neg}"
    );
}

#[test]
fn test_frm_fit_recovery() {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;
    let r_ir = 1.0;

    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * vis_albedo).collect();
    let result = frm_total_flux(
        &bands,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        0.9,
        &sun2obj,
        &sun2obs,
    );

    let obs: Vec<FluxObs> = bands
        .into_iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50)
        .expect("MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "FRM fit diameter {d_median:.2} too far from truth 10.0",
    );
    assert!(!res.model.is_neatm(), "FRM should not be NEATM");

    // MAP diagnostic fields.
    let n_obs = obs.len();
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(
        res.reduced_chi2.is_finite() || n_obs <= Model::Frm.dim(),
        "reduced_chi2 should be finite when nobs > nparams",
    );
}

#[test]
fn test_neatm_batch() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let tasks = vec![
        FitTask {
            model: Model::Neatm,
            obs: obs.clone(),
            c_hg: C_V,
            emissivity: 0.9,
            priors: priors.clone(),
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
        FitTask {
            model: Model::Neatm,
            obs,
            c_hg: C_V,
            emissivity: 0.9,
            priors,
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
    ];
    let results = fit_batch(&tasks);
    assert_eq!(results.len(), 2, "batch should return one result per task");
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "batch task {i} should succeed");
    }
}

#[test]
fn test_frm_batch() {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;
    let r_ir = 1.0;
    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * vis_albedo).collect();
    let result = frm_total_flux(
        &bands,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        0.9,
        &sun2obj,
        &sun2obs,
    );
    let obs: Vec<FluxObs> = bands
        .into_iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let tasks = vec![FitTask {
        model: Model::Frm,
        obs,
        c_hg: C_V,
        emissivity: 0.9,
        priors,
        num_chains: 1,
        num_tune: 50,
        num_draws: 50,
    }];
    let results = fit_batch(&tasks);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_ok(), "single FRM batch task should succeed");
}

#[test]
fn test_coupling_consistency_neatm() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50).unwrap();

    // Every draw must satisfy D = c_hg / sqrt(pV) * 10^(-H/5).
    // H is now per-draw at index 3.
    let c_hg = C_V;
    for (i, draw) in res.draws.iter().enumerate() {
        let d = draw[0];
        let vis_albedo = draw[1];
        let h = draw[3];
        let expected_d = c_hg / vis_albedo.sqrt() * 10_f64.powf(-h / 5.0);
        let rel = (d - expected_d).abs() / expected_d;
        assert!(
            rel < 1e-6,
            "draw {i}: D={d:.6}, expected={expected_d:.6}, rel_err={rel:.2e}"
        );
    }
}

#[test]
fn test_coupling_consistency_frm() {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;
    let band_albedos: Vec<f64> = bands.iter().map(|_| vis_albedo).collect();
    let result = frm_total_flux(
        &bands,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        0.9,
        &sun2obj,
        &sun2obs,
    );
    let obs: Vec<FluxObs> = bands
        .into_iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50).unwrap();

    // H is now per-draw at index 2.
    let c_hg = C_V;
    for (i, draw) in res.draws.iter().enumerate() {
        let d = draw[0];
        let vis_albedo_draw = draw[1];
        let h = draw[2];
        let expected_d = c_hg / vis_albedo_draw.sqrt() * 10_f64.powf(-h / 5.0);
        let rel = (d - expected_d).abs() / expected_d;
        assert!(
            rel < 1e-6,
            "FRM draw {i}: D={d:.6}, expected={expected_d:.6}, rel_err={rel:.2e}"
        );
    }
}

#[test]
fn test_neatm_w3_w4_only() {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::WISE;
    let vis_albedo = hg.vis_albedo;

    // Generate truth from NEATM with W3 + W4 only.
    let w3w4: Vec<BandInfo> = vec![bands[2], bands[3]];
    let band_albedos: Vec<f64> = w3w4.iter().map(|_| vis_albedo).collect();
    let result = neatm_total_flux(
        &w3w4,
        &band_albedos,
        hg.diameter,
        vis_albedo,
        hg.g_param,
        hg.h_mag,
        1.0,
        0.9,
        &sun2obj,
        &sun2obs,
    );

    let obs: Vec<FluxObs> = w3w4
        .into_iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50)
        .expect("W3+W4 MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "W3+W4 fit diameter {d_median:.2} too far from truth 10.0",
    );
}

#[test]
fn test_all_upper_limits_no_panic() {
    let (obs, hg) = synthetic_neatm_obs();
    // Turn each detection into a high-threshold upper limit.
    let ul_obs: Vec<FluxObs> = obs
        .into_iter()
        .map(|o| {
            let mean = o.point_estimate().unwrap();
            let sigma = o.sigma_lo().unwrap();
            FluxObs::upper_limit(mean * 10.0, sigma, o.band, o.sun2obj, o.sun2obs)
        })
        .collect();

    // All-upper-limit input: MCMC may or may not converge.
    // The key contract is no panic.
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let _ = fit_mcmc(Model::Neatm, &ul_obs, C_V, 0.9, &priors, 1, 50, 50);
}

#[test]
fn test_mcmc_draw_column_counts() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    // NEATM: 7 columns [D, pV, beaming, H, G, R_IR, f_sigma].
    let neatm_res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50).unwrap();
    for (i, d) in neatm_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 7, "NEATM draw {i} should have 7 columns");
        // D, pV, beaming, R_IR, f_sigma must be positive.
        for &col in &[0, 1, 2, 5, 6] {
            assert!(
                d[col] > 0.0 && d[col].is_finite(),
                "NEATM draw {i} col {col} = {}",
                d[col]
            );
        }
        // pV should be physically reasonable.
        assert!(
            d[1] < 2.0,
            "NEATM draw {i}: pV = {:.4} out of physical range",
            d[1]
        );
        // H and G must be finite (can be <= 0).
        assert!(d[3].is_finite(), "NEATM draw {i} H = {} not finite", d[3]);
        assert!(d[4].is_finite(), "NEATM draw {i} G = {} not finite", d[4]);
    }

    // FRM: 6 columns [D, pV, H, G, R_IR, f_sigma].
    let frm_res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50).unwrap();
    for (i, d) in frm_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 6, "FRM draw {i} should have 6 columns");
        // D, pV, R_IR, f_sigma must be positive.
        for &col in &[0, 1, 4, 5] {
            assert!(
                d[col] > 0.0 && d[col].is_finite(),
                "FRM draw {i} col {col} = {}",
                d[col]
            );
        }
        assert!(d[2].is_finite(), "FRM draw {i} H = {} not finite", d[2]);
        assert!(d[3].is_finite(), "FRM draw {i} G = {} not finite", d[3]);
    }

    // HG: 3 columns [H, G, f_sigma].  H can be any real number, G bounded, f_sigma > 0.
    let (hg_obs, hg_hg) = synthetic_hg_obs();
    let hg_priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg_hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let hg_res = fit_mcmc(Model::Hg, &hg_obs, C_V, 0.9, &hg_priors, 1, 50, 50).unwrap();
    for (i, d) in hg_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 3, "HG draw {i} should have 3 columns");
        assert!(d[0].is_finite(), "HG draw {i} H = {} not finite", d[0]);
        assert!(d[1].is_finite(), "HG draw {i} G = {} not finite", d[1]);
        assert!(
            d[2] > 0.0 && d[2].is_finite(),
            "HG draw {i} f_sigma = {}",
            d[2]
        );
    }
}

// ---------------------------------------------------------------------------
// HG model tests
// ---------------------------------------------------------------------------

/// Build synthetic V-band observations for the HG model.
///
/// An H=18, G=0.15 asteroid observed in V band at two phase angles.
/// The observer is offset from the Sun so that the phase angle is non-zero.
fn synthetic_hg_obs() -> (Vec<FluxObs>, TestHg) {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let v_band = BandInfo::V;

    // Geometry 1: opposition-like (small phase angle).
    let sun2obj_1 = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs_1 = Vector3::new(-1.0, 0.0, 0.0);

    // Geometry 2: moderate phase angle.
    let sun2obj_2 = Vector3::new(1.5, 1.0, 0.0);
    let sun2obs_2 = Vector3::new(-1.0, 0.0, 0.0);

    let vis_albedo = hg.vis_albedo;
    let mut obs = Vec::new();
    for (&s2o, &s2obs) in [sun2obj_1, sun2obj_2]
        .iter()
        .zip([sun2obs_1, sun2obs_2].iter())
    {
        let flux = crate::hg_apparent_flux(
            hg.g_param,
            hg.diameter,
            &s2o,
            &s2obs,
            v_band.wavelength,
            vis_albedo,
        ) * v_band.solar_correction;
        obs.push(FluxObs::detection(flux, flux * 0.05, v_band, s2o, s2obs));
    }

    (obs, hg)
}

#[test]
fn test_hg_nll_at_truth() {
    let (obs, _hg) = synthetic_hg_obs();
    let neg_log_lik = make_neg_log_likelihood(Model::Hg, &obs, C_V, 0.9);

    // Truth: H=18, G=0.15, f_sigma=1.0.
    // [H, G, f_sigma]
    let truth = [18.0, 0.15, 1.0_f64];
    let neg_log_lik_at_truth = neg_log_lik(&truth);
    assert!(
        neg_log_lik_at_truth.is_finite(),
        "HG NLL at truth should be finite"
    );

    // Wrong H.
    let wrong = [15.0, 0.15, 1.0];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong,
        "HG NLL at truth ({neg_log_lik_at_truth}) should be less than at wrong H ({neg_log_lik_wrong})"
    );

    // Wrong in other direction.
    let wrong2 = [22.0, 0.15, 1.0];
    let neg_log_lik_wrong2 = neg_log_lik(&wrong2);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong2,
        "HG NLL at truth ({neg_log_lik_at_truth}) should be less than at H=22 ({neg_log_lik_wrong2})"
    );
}

#[test]
fn test_hg_fit_recovery() {
    let (obs, hg) = synthetic_hg_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    let res = fit_mcmc(Model::Hg, &obs, C_V, 0.9, &priors, 1, 100, 100)
        .expect("HG MCMC should produce a result");

    // H should recover near 18.0.
    let h_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };
    assert!(
        (h_median - 18.0).abs() < 1.0,
        "HG fit H {h_median:.2} too far from truth 18.0",
    );

    assert!(res.model.is_hg(), "Model flag should be HG");

    // MAP diagnostic fields.
    let n_obs = obs.len();
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(
        res.reduced_chi2.is_finite() || n_obs <= Model::Hg.dim(),
        "reduced_chi2 should be finite when nobs > nparams",
    );

    // For HG, reflected_frac should be 1.0 everywhere.
    for (i, &frac) in res.best_fit_reflected_frac.iter().enumerate() {
        assert!(
            (frac - 1.0).abs() < 1e-10,
            "HG reflected_frac[{i}] = {frac}, expected 1.0"
        );
    }
}

#[test]
fn test_hg_batch() {
    let (obs, hg) = synthetic_hg_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let tasks = vec![
        FitTask {
            model: Model::Hg,
            obs: obs.clone(),
            c_hg: C_V,
            emissivity: 0.9,
            priors: priors.clone(),
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
        FitTask {
            model: Model::Hg,
            obs,
            c_hg: C_V,
            emissivity: 0.9,
            priors,
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
    ];
    let results = fit_batch(&tasks);
    assert_eq!(results.len(), 2, "batch should return one result per task");
    for (i, r) in results.iter().enumerate() {
        assert!(r.is_ok(), "HG batch task {i} should succeed");
        assert!(r.as_ref().unwrap().model.is_hg());
    }
}

// ---------------------------------------------------------------------------
// Multi-geometry test (exercises the observation-batching optimization)
// ---------------------------------------------------------------------------

#[test]
fn test_multi_geometry_neatm() {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let vis_albedo = hg.vis_albedo;
    let beaming = 1.2;
    let r_ir = 1.0;
    let emissivity = 0.9;
    let bands = BandInfo::WISE;

    // Two different geometries.
    let geoms: Vec<(Vector3<f64>, Vector3<f64>)> = vec![
        (Vector3::new(2.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
        (Vector3::new(1.5, 1.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
    ];

    let mut obs = Vec::new();
    for (sun2obj, sun2obs) in &geoms {
        let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * vis_albedo).collect();
        let result = neatm_total_flux(
            &bands,
            &band_albedos,
            hg.diameter,
            vis_albedo,
            hg.g_param,
            hg.h_mag,
            beaming,
            emissivity,
            sun2obj,
            sun2obs,
        );
        for (band, &flux) in bands.into_iter().zip(&result.fluxes) {
            obs.push(FluxObs::detection(flux, flux * 0.05, band, *sun2obj, *sun2obs));
        }
    }

    // Should have 8 observations (4 bands x 2 geometries).
    assert_eq!(obs.len(), 8);

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50)
        .expect("Multi-geometry NEATM MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "Multi-geometry fit diameter {d_median:.2} too far from truth 10.0",
    );
    assert_eq!(res.nobs, 8, "Should count all 8 observations");
    assert_eq!(res.best_fit_fluxes.len(), 8);
}
