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
use super::types::log_likelihood;
use crate::{BandInfo, FrmParams, HGParams, NeatmParams};
use nalgebra::Vector3;

/// Build a negative-log-likelihood closure for a model.
fn make_neg_log_likelihood(
    model: Model,
    obs: &[FluxObs],
    hg: &HGParams,
    emissivity: f64,
) -> impl Fn(&[f64]) -> f64 {
    let obs = obs.to_vec();
    let hg = hg.clone();
    move |x: &[f64]| -> f64 {
        let ll = log_likelihood(model, x, &obs, &hg, emissivity);
        if ll.is_finite() { -ll } else { f64::MAX }
    }
}

/// Build a synthetic observation set: a 10 km asteroid at 2 AU, H=18, G=0.15.
fn synthetic_neatm_obs() -> (Vec<FluxObs>, HGParams) {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();

    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0); // observer at Sun for simplicity

    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();
    let beaming = 1.2;
    let r_ir = 1.0;
    let emissivity = 0.9;

    // Generate "observed" fluxes from the forward model
    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
    let params = NeatmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        beaming,
        hg_params: hg.clone(),
        emissivity,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();

    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05, // 5% uncertainty
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
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
fn test_neatm_nll_at_truth() {
    let (obs, hg) = synthetic_neatm_obs();
    let neg_log_lik = make_neg_log_likelihood(Model::Neatm, &obs, &hg, 0.9);

    // Truth: D=10, eta=1.2, f_sigma=1.0, R_IR=1.0
    let truth = [10.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_at_truth = neg_log_lik(&truth);

    // At truth the residual terms are zero, but the normalization
    // sum ln(f_sigma * sigma_i) = sum ln(sigma_i) remains.  Just verify it's finite
    // and smaller than at a wrong point.
    assert!(neg_log_lik_at_truth.is_finite(), "NLL at truth should be finite");

    let wrong = [20.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong,
        "NLL at truth ({neg_log_lik_at_truth}) should be less than at wrong D ({neg_log_lik_wrong})"
    );
}

#[test]
fn test_neatm_nll_away_from_truth() {
    let (obs, hg) = synthetic_neatm_obs();
    let neg_log_lik = make_neg_log_likelihood(Model::Neatm, &obs, &hg, 0.9);

    let truth = [10.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    // Wrong diameter
    let wrong = [20.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    let neg_log_lik_truth = neg_log_lik(&truth);
    assert!(
        neg_log_lik_wrong > neg_log_lik_truth,
        "NLL at wrong D ({neg_log_lik_wrong}) should exceed truth ({neg_log_lik_truth})"
    );
}

#[test]
fn test_neatm_fit_recovery() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();

    let res = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50)
        .expect("MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "Fit diameter {d_median:.2} too far from truth 10.0",
    );
}

#[test]
fn test_frm_nll_at_truth() {
    // Build FRM observations similarly
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();

    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();
    let r_ir = 1.0;

    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
    let params = FrmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();

    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();

    let neg_log_lik = make_neg_log_likelihood(Model::Frm, &obs, &hg, 0.9);
    let truth = [10.0_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_val = neg_log_lik(&truth);
    assert!(
        neg_log_lik_val.is_finite(),
        "FRM NLL at truth should be finite, got {neg_log_lik_val}"
    );

    // NLL at truth should be less than at a wrong diameter
    let wrong = [20.0_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_val < neg_log_lik_wrong,
        "FRM NLL at truth ({neg_log_lik_val}) should be less than at wrong D ({neg_log_lik_wrong})"
    );
}

#[test]
fn test_upper_limit_penalty() {
    let (obs, hg) = synthetic_neatm_obs();

    // Turn all observations into upper limits with threshold = actual flux
    let ul_obs: Vec<FluxObs> = obs
        .into_iter()
        .map(|mut o| {
            o.is_upper_limit = true;
            o
        })
        .collect();

    let neg_log_lik = make_neg_log_likelihood(Model::Neatm, &ul_obs, &hg, 0.9);

    // At truth, model == threshold -> NLL should be 0
    // (upper limits don't include the ln(sigma) term)
    let truth = [10.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_at_truth = neg_log_lik(&truth);
    assert!(
        neg_log_lik_at_truth < 1e-6,
        "Upper limit NLL at truth should be ~0, got {neg_log_lik_at_truth}"
    );

    // Larger diameter -> more flux -> exceeds threshold -> penalty
    let bigger = [20.0_f64.ln(), 1.2_f64.ln(), 1.0_f64.ln(), 1.0_f64.ln()];
    let neg_log_lik_bigger = neg_log_lik(&bigger);
    assert!(
        neg_log_lik_bigger > neg_log_lik_at_truth,
        "Larger D should incur upper-limit penalty"
    );
}

#[test]
fn test_frm_fit_recovery() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();
    let r_ir = 1.0;

    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
    let params = FrmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();

    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();

    let priors = Priors::default();
    let res = fit_mcmc(Model::Frm, &obs, &hg, 0.9, &priors, 1, 50, 50)
        .expect("MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "FRM fit diameter {d_median:.2} too far from truth 10.0",
    );
    assert!(!res.model.is_neatm(), "FRM should not be NEATM");
}

#[test]
fn test_neatm_mcmc_smoke() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();
    let n_obs = obs.len();

    let result = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50);
    assert!(result.is_some(), "MCMC should produce a result");
    let res = result.unwrap();
    assert!(!res.draws.is_empty(), "Should have draws");
    // Each NEATM draw: [D, pV, eta, f_sigma, R_IR] -> length 5
    assert_eq!(res.draws[0].len(), 5, "NEATM draws should have 5 columns");
    // Diameter median should be in a reasonable range.
    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        d_median > 1.0 && d_median < 100.0,
        "Diameter median {d_median} out of reasonable range",
    );

    // MAP diagnostic fields
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(res.chi2_best.is_finite(), "chi2_best should be finite");
}

#[test]
fn test_frm_mcmc_smoke() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();

    let band_albedos: Vec<f64> = bands.iter().map(|_| pv).collect();
    let params = FrmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();

    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();

    let priors = Priors::default();
    let n_obs = obs.len();
    let result = fit_mcmc(Model::Frm, &obs, &hg, 0.9, &priors, 1, 50, 50);
    assert!(result.is_some(), "FRM MCMC should produce a result");
    let res = result.unwrap();
    assert_eq!(res.draws[0].len(), 4, "FRM draws should have 4 columns");
    assert!(!res.model.is_neatm(), "FRM result should have is_neatm=false");

    // MAP diagnostic fields
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(res.chi2_best.is_finite(), "chi2_best should be finite");
}

#[test]
fn test_neatm_batch() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();
    let tasks = vec![
        FitTask {
            model: Model::Neatm,
            obs: obs.clone(),
            hg: hg.clone(),
            emissivity: 0.9,
            priors: priors.clone(),
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
        FitTask {
            model: Model::Neatm,
            obs,
            hg,
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
        assert!(r.is_some(), "batch task {i} should succeed");
    }
}

#[test]
fn test_frm_batch() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();
    let r_ir = 1.0;
    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
    let params = FrmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();
    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();
    let priors = Priors::default();
    let tasks = vec![FitTask {
        model: Model::Frm,
        obs,
        hg,
        emissivity: 0.9,
        priors,
        num_chains: 1,
        num_tune: 50,
        num_draws: 50,
    }];
    let results = fit_batch(&tasks);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_some(), "single FRM batch task should succeed");
}

#[test]
fn test_coupling_consistency_neatm() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();
    let res = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50).unwrap();

    // Every draw must satisfy D = c_hg / sqrt(pV) * 10^(-H/5).
    let c_hg = hg.c_hg();
    let h = hg.h_mag;
    for (i, draw) in res.draws.iter().enumerate() {
        let d = draw[0];
        let pv = draw[1];
        let expected_d = c_hg / pv.sqrt() * 10_f64.powf(-h / 5.0);
        let rel = (d - expected_d).abs() / expected_d;
        assert!(
            rel < 1e-6,
            "draw {i}: D={d:.6}, expected={expected_d:.6}, rel_err={rel:.2e}"
        );
    }
}

#[test]
fn test_coupling_consistency_frm() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();
    let band_albedos: Vec<f64> = bands.iter().map(|_| pv).collect();
    let params = FrmParams {
        obs_bands: bands.to_vec(),
        band_albedos,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();
    let obs: Vec<FluxObs> = bands
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();

    let priors = Priors::default();
    let res = fit_mcmc(Model::Frm, &obs, &hg, 0.9, &priors, 1, 50, 50).unwrap();

    let c_hg = hg.c_hg();
    let h = hg.h_mag;
    for (i, draw) in res.draws.iter().enumerate() {
        let d = draw[0];
        let pv_draw = draw[1];
        let expected_d = c_hg / pv_draw.sqrt() * 10_f64.powf(-h / 5.0);
        let rel = (d - expected_d).abs() / expected_d;
        assert!(
            rel < 1e-6,
            "FRM draw {i}: D={d:.6}, expected={expected_d:.6}, rel_err={rel:.2e}"
        );
    }
}

#[test]
fn test_neatm_w3_w4_only() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 0.0, 0.0);
    let bands = BandInfo::new_wise();
    let pv = hg.vis_albedo().unwrap();

    // Generate truth from NEATM with W3 + W4 only.
    let w3w4: Vec<BandInfo> = vec![bands[2].clone(), bands[3].clone()];
    let band_albedos: Vec<f64> = w3w4.iter().map(|_| pv).collect();
    let params = NeatmParams {
        obs_bands: w3w4.clone(),
        band_albedos,
        beaming: 1.0,
        hg_params: hg.clone(),
        emissivity: 0.9,
    };
    let result = params.apparent_total_flux(&sun2obj, &sun2obs).unwrap();

    let obs: Vec<FluxObs> = w3w4
        .iter()
        .zip(&result.fluxes)
        .map(|(band, &flux)| FluxObs {
            flux,
            sigma: flux * 0.05,
            band: band.clone(),
            is_upper_limit: false,
            sun2obj,
            sun2obs,
        })
        .collect();

    let priors = Priors::default();
    let res = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50)
        .expect("W3+W4 MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "W3+W4 fit diameter {d_median:.2} too far from truth 10.0",
    );
}

#[test]
fn test_all_upper_limits_no_panic() {
    let (obs, hg) = synthetic_neatm_obs();
    let ul_obs: Vec<FluxObs> = obs
        .into_iter()
        .map(|mut o| {
            o.flux *= 10.0;
            o.is_upper_limit = true;
            o
        })
        .collect();

    // All-upper-limit input: MCMC may or may not converge.
    // The key contract is no panic.
    let priors = Priors::default();
    let _ = fit_mcmc(Model::Neatm, &ul_obs, &hg, 0.9, &priors, 1, 50, 50);
}

#[test]
fn test_mcmc_draw_column_counts() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();

    // NEATM: 5 columns [D, pV, eta, f_sigma, R_IR], all positive.
    let neatm_res = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50).unwrap();
    for (i, d) in neatm_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 5, "NEATM draw {i} should have 5 columns");
        for (j, &v) in d.iter().enumerate() {
            assert!(v > 0.0 && v.is_finite(), "NEATM draw {i} col {j} = {v}");
        }
    }

    // FRM: 4 columns [D, pV, f_sigma, R_IR], all positive.
    let frm_res = fit_mcmc(Model::Frm, &obs, &hg, 0.9, &priors, 1, 50, 50).unwrap();
    for (i, d) in frm_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 4, "FRM draw {i} should have 4 columns");
        for (j, &v) in d.iter().enumerate() {
            assert!(v > 0.0 && v.is_finite(), "FRM draw {i} col {j} = {v}");
        }
    }

    // HG: 3 columns [H, G, f_sigma].  H can be any real number, G bounded, f_sigma > 0.
    let (hg_obs, hg_hg) = synthetic_hg_obs();
    let hg_res = fit_mcmc(Model::Hg, &hg_obs, &hg_hg, 0.9, &priors, 1, 50, 50).unwrap();
    for (i, d) in hg_res.draws.iter().enumerate() {
        assert_eq!(d.len(), 3, "HG draw {i} should have 3 columns");
        assert!(d[0].is_finite(), "HG draw {i} H = {} not finite", d[0]);
        assert!(d[1].is_finite(), "HG draw {i} G = {} not finite", d[1]);
        assert!(d[2] > 0.0 && d[2].is_finite(), "HG draw {i} f_sigma = {}", d[2]);
    }
}

// ---------------------------------------------------------------------------
// HG model tests
// ---------------------------------------------------------------------------

/// Build synthetic V-band observations for the HG model.
///
/// An H=18, G=0.15 asteroid observed in V band at two phase angles.
/// The observer is offset from the Sun so that the phase angle is non-zero.
fn synthetic_hg_obs() -> (Vec<FluxObs>, HGParams) {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let v_band = BandInfo::new_v();

    // Geometry 1: opposition-like (small phase angle).
    let sun2obj_1 = Vector3::new(2.0, 0.0, 0.0);
    let sun2obs_1 = Vector3::new(-1.0, 0.0, 0.0);

    // Geometry 2: moderate phase angle.
    let sun2obj_2 = Vector3::new(1.5, 1.0, 0.0);
    let sun2obs_2 = Vector3::new(-1.0, 0.0, 0.0);

    let pv = hg.vis_albedo().unwrap();
    let mut obs = Vec::new();
    for (&s2o, &s2obs) in [sun2obj_1, sun2obj_2].iter().zip([sun2obs_1, sun2obs_2].iter()) {
        let flux = hg
            .apparent_flux(&s2o, &s2obs, v_band.wavelength, pv)
            .unwrap()
            * v_band.solar_correction;
        obs.push(FluxObs {
            flux,
            sigma: flux * 0.05,
            band: v_band.clone(),
            is_upper_limit: false,
            sun2obj: s2o,
            sun2obs: s2obs,
        });
    }

    (obs, hg)
}

#[test]
fn test_hg_nll_at_truth() {
    let (obs, hg) = synthetic_hg_obs();
    let neg_log_lik = make_neg_log_likelihood(Model::Hg, &obs, &hg, 0.9);

    // Truth: H=18, G=0.15, f_sigma=1.0.
    let truth = [18.0, 0.15, 0.0_f64]; // [H, G, ln(1.0)]
    let neg_log_lik_at_truth = neg_log_lik(&truth);
    assert!(neg_log_lik_at_truth.is_finite(), "HG NLL at truth should be finite");

    // Wrong H.
    let wrong = [15.0, 0.15, 0.0];
    let neg_log_lik_wrong = neg_log_lik(&wrong);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong,
        "HG NLL at truth ({neg_log_lik_at_truth}) should be less than at wrong H ({neg_log_lik_wrong})"
    );

    // Wrong in other direction.
    let wrong2 = [22.0, 0.15, 0.0];
    let neg_log_lik_wrong2 = neg_log_lik(&wrong2);
    assert!(
        neg_log_lik_at_truth < neg_log_lik_wrong2,
        "HG NLL at truth ({neg_log_lik_at_truth}) should be less than at H=22 ({neg_log_lik_wrong2})"
    );
}

#[test]
fn test_hg_fit_recovery() {
    let (obs, hg) = synthetic_hg_obs();
    let priors = Priors::default();

    let res = fit_mcmc(Model::Hg, &obs, &hg, 0.9, &priors, 1, 100, 100)
        .expect("HG MCMC should produce a result");

    // H should recover near 18.0.
    let h_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        (h_median - 18.0).abs() < 1.0,
        "HG fit H {h_median:.2} too far from truth 18.0",
    );
}

#[test]
fn test_hg_mcmc_smoke() {
    let (obs, hg) = synthetic_hg_obs();
    let priors = Priors::default();
    let n_obs = obs.len();

    let res = fit_mcmc(Model::Hg, &obs, &hg, 0.9, &priors, 1, 50, 50)
        .expect("HG MCMC should produce a result");

    assert!(!res.draws.is_empty(), "Should have draws");
    assert_eq!(res.draws[0].len(), 3, "HG draws should have 3 columns");
    assert!(res.model.is_hg(), "Model flag should be HG");

    // MAP diagnostics.
    assert_eq!(res.nobs, n_obs, "nobs should match observation count");
    assert_eq!(res.best_fit_fluxes.len(), n_obs);
    assert_eq!(res.best_fit_residuals.len(), n_obs);
    assert_eq!(res.best_fit_reflected_frac.len(), n_obs);
    assert!(res.chi2_best.is_finite(), "chi2_best should be finite");

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
    let priors = Priors::default();
    let tasks = vec![
        FitTask {
            model: Model::Hg,
            obs: obs.clone(),
            hg: hg.clone(),
            emissivity: 0.9,
            priors: priors.clone(),
            num_chains: 1,
            num_tune: 50,
            num_draws: 50,
        },
        FitTask {
            model: Model::Hg,
            obs,
            hg,
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
        assert!(r.is_some(), "HG batch task {i} should succeed");
        assert!(r.as_ref().unwrap().model.is_hg());
    }
}

// ---------------------------------------------------------------------------
// Multi-geometry test (exercises the observation-batching optimisation)
// ---------------------------------------------------------------------------

#[test]
fn test_multi_geometry_neatm() {
    let hg = HGParams::try_new("test".into(), 0.15, Some(18.0), None, None, Some(10.0)).unwrap();
    let pv = hg.vis_albedo().unwrap();
    let beaming = 1.2;
    let r_ir = 1.0;
    let emissivity = 0.9;
    let bands = BandInfo::new_wise();

    // Two different geometries.
    let geoms: Vec<(Vector3<f64>, Vector3<f64>)> = vec![
        (Vector3::new(2.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
        (Vector3::new(1.5, 1.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)),
    ];

    let mut obs = Vec::new();
    for (sun2obj, sun2obs) in &geoms {
        let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * pv).collect();
        let params = NeatmParams {
            obs_bands: bands.to_vec(),
            band_albedos,
            beaming,
            hg_params: hg.clone(),
            emissivity,
        };
        let result = params.apparent_total_flux(sun2obj, sun2obs).unwrap();
        for (band, &flux) in bands.iter().zip(&result.fluxes) {
            obs.push(FluxObs {
                flux,
                sigma: flux * 0.05,
                band: band.clone(),
                is_upper_limit: false,
                sun2obj: *sun2obj,
                sun2obs: *sun2obs,
            });
        }
    }

    // Should have 8 observations (4 bands x 2 geometries).
    assert_eq!(obs.len(), 8);

    let priors = Priors::default();
    let res = fit_mcmc(Model::Neatm, &obs, &hg, emissivity, &priors, 1, 50, 50)
        .expect("Multi-geometry NEATM MCMC should produce a result");

    let d_median = {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[0]).collect();
        kete_stats::prelude::SortedData::try_from(vals).unwrap().median()
    };
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "Multi-geometry fit diameter {d_median:.2} too far from truth 10.0",
    );
    assert_eq!(res.nobs, 8, "Should count all 8 observations");
    assert_eq!(res.best_fit_fluxes.len(), 8);
}

// ---------------------------------------------------------------------------
// pV reasonableness check
// ---------------------------------------------------------------------------

#[test]
fn test_pv_draws_reasonable() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = Priors::default();
    let res = fit_mcmc(Model::Neatm, &obs, &hg, 0.9, &priors, 1, 50, 50).unwrap();

    for (i, draw) in res.draws.iter().enumerate() {
        let pv = draw[1];
        assert!(
            pv > 0.0 && pv < 2.0,
            "NEATM draw {i}: pV = {pv:.4} out of physical range (0, 2)"
        );
    }
}
