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
use crate::fitting::types::{Penalty, ShapeFit, Tail, logistic_barrier};
use crate::{
    BandInfo, EllipsoidTemplate, RoughnessCorrection, SpinState, ThermalParams, TpmFieldGrid,
    TpmShape, frm_total_flux, neatm_total_flux, resolve_hg_params, tpm_total_flux,
};
use kete_core::constants::C_V;
use nalgebra::{UnitVector3, Vector3};
use std::sync::Arc;

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
        let params = model.unpack(x, emissivity, c_hg, ShapeFit::default());
        let ll = model.log_likelihood(&params, &obs, None);
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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
        .collect();

    (obs, hg)
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
#[ignore = "pprof sampling profile of the fitting hot path; writes /tmp/fit_flamegraph.svg"]
fn profile_fit_hotpath() {
    use std::collections::HashMap;
    // Build a realistic per-evaluation workload: the exact log_posterior call NUTS
    // makes for every finite-difference gradient evaluation, with the 256-facet
    // sampling shape and a cached field grid (as the fitter uses).
    let (base, hg, spin, gamma) = synthetic_tpm_obs();
    let obs: Vec<FluxObs> = base.iter().cloned().cycle().take(160).collect();
    let cfg = TpmConfig {
        spin: spin.clone(),
        shape: TpmShape::sphere_with_facets(256),
        seed_shape: TpmShape::sphere_with_facets(128),
        grid: Arc::new(TpmFieldGrid::new(0.1, 50.0).unwrap()),
        roughness: None,
        fit_c_a: false,
        fit_b_a: false,
        fit_phase0: false,
        template: None,
    };
    let priors = FluxPriors::default();
    let x = [hg.diameter, gamma.ln(), hg.h_mag, hg.g_param, 1.0, 1.0];

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    let t0 = std::time::Instant::now();
    let mut iters = 0_u64;
    let mut acc = 0.0;
    while t0.elapsed().as_secs_f64() < 6.0 {
        acc += Model::Tpm.log_posterior(&x, &obs, C_V, 0.9, &priors, Some(&cfg));
        iters += 1;
    }
    assert!(acc.is_finite());

    let report = guard.report().build().unwrap();
    if let Ok(f) = std::fs::File::create("/tmp/fit_flamegraph.svg") {
        let _ = report.flamegraph(f);
    }

    // Self-time split: aggregate by the *innermost* inlined symbol at the leaf frame
    // (inlining collapses several functions into one address, so the whole inlined
    // chain at the leaf is inspected, not just the outermost symbol).
    let total: isize = report.data.values().sum();
    let mut self_time: HashMap<String, isize> = HashMap::new();
    for (frames, count) in &report.data {
        if let Some(leaf_chain) = frames.frames.first() {
            // Tally each symbol in the leaf's inlined chain so geometry (asin/atan2),
            // field sampling, and the Planck exp are all visible regardless of order.
            for sym in leaf_chain {
                *self_time.entry(format!("{sym}")).or_default() += *count;
            }
        }
    }
    let mut rows: Vec<_> = self_time.into_iter().collect();
    rows.sort_by_key(|(_, c)| -*c);
    eprintln!(
        "PROFILE iters={iters} total_samples={total} ({:.2} us/eval)",
        t0.elapsed().as_secs_f64() / iters as f64 * 1e6
    );
    eprintln!("-- symbols present in the leaf inlined chain (inclusive of inlining) --");
    for (name, c) in rows.iter().take(30) {
        eprintln!("  {:5.1}%  {name}", 100.0 * *c as f64 / total as f64);
    }
    // Also show the full inlined leaf chain for the hottest distinct leaf frames.
    let mut stacks: Vec<_> = report.data.iter().collect();
    stacks.sort_by_key(|(_, c)| -**c);
    eprintln!("-- hottest leaf frames (innermost ... outermost) --");
    for (frames, count) in stacks.into_iter().take(12) {
        let chain: Vec<String> = frames
            .frames
            .first()
            .map(|lvl| lvl.iter().map(|s| format!("{s}")).collect())
            .unwrap_or_default();
        eprintln!(
            "  {:5.1}%  {}",
            100.0 * *count as f64 / total as f64,
            chain.join("  <  ")
        );
    }
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
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50, None)
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
        res.reduced_chi2.is_finite() || n_obs <= Model::Neatm.dim(ShapeFit::default()),
        "reduced_chi2 should be finite when nobs > nparams",
    );
}

/// Build a synthetic multi-phase TPM observation set: a 10 km asteroid at 2 AU with
/// thermal inertia Gamma = 200 SI, a 6 h prograde rotation, observed at several phase
/// angles (so the night-side / thermal-lag signal constrains Gamma).
#[test]
fn test_tpm_oblate_layout_and_forward() {
    // Oblate c/a as a fitted parameter: the layout grows by one, c/a is decoded from
    // the appended slot, and the forward model rebuilds the shape via the template so a
    // flattened body differs from the sphere.
    let (base, _hg, spin, _gamma) = synthetic_tpm_obs();
    let cfg = TpmConfig {
        spin,
        shape: TpmShape::sphere_with_facets(256),
        seed_shape: TpmShape::sphere_with_facets(128),
        grid: Arc::new(TpmFieldGrid::new(0.1, 50.0).unwrap()),
        roughness: None,
        fit_c_a: true,
        fit_b_a: false,
        fit_phase0: false,
        template: Some(EllipsoidTemplate::new(12)),
    };

    // layout: [D, thermal_inertia, H, G, f_sigma, R_IR, c_a]
    assert_eq!(
        Model::Tpm.dim(ShapeFit {
            fit_c_a: true,
            ..ShapeFit::default()
        }),
        7
    );
    assert_eq!(Model::Tpm.dim(ShapeFit::default()), 6);
    assert_eq!(
        *Model::Tpm
            .draw_column_names(ShapeFit {
                fit_c_a: true,
                ..ShapeFit::default()
            })
            .last()
            .unwrap(),
        "c_a"
    );

    let x = [10.0, 150.0, 18.0, 0.15, 1.0, 1.6, 0.6];
    let p = Model::Tpm.unpack(
        &x,
        0.9,
        C_V,
        ShapeFit {
            fit_c_a: true,
            ..ShapeFit::default()
        },
    );
    assert!((p.c_a - 0.6).abs() < 1e-12);
    // the appended c/a flows through the draw row
    assert!(
        (*p.to_draw_row(
            Model::Tpm,
            ShapeFit {
                fit_c_a: true,
                ..ShapeFit::default()
            }
        )
        .last()
        .unwrap()
            - 0.6)
            .abs()
            < 1e-12
    );

    let ob = &base[0];
    let bands = [ob.band];
    let flat =
        Model::Tpm.compute_fluxes(&p, &bands, &ob.sun2obj, &ob.sun2obs, ob.epoch, Some(&cfg));

    // c/a = 1 is a sphere; the flattened body must give a different thermal flux.
    let mut xs = x;
    xs[6] = 1.0;
    let ps = Model::Tpm.unpack(
        &xs,
        0.9,
        C_V,
        ShapeFit {
            fit_c_a: true,
            ..ShapeFit::default()
        },
    );
    let sph =
        Model::Tpm.compute_fluxes(&ps, &bands, &ob.sun2obj, &ob.sun2obs, ob.epoch, Some(&cfg));

    let rel = (flat.fluxes[0] - sph.fluxes[0]).abs() / sph.fluxes[0];
    assert!(
        rel > 1e-3,
        "oblate flux {} should differ from sphere {}",
        flat.fluxes[0],
        sph.fluxes[0]
    );
}

fn synthetic_tpm_obs() -> (Vec<FluxObs>, TestHg, SpinState, f64) {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let spin = SpinState {
        pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        period: 6.0 * 3600.0,
        phase0: 0.0,
        epoch0: 0.0,
    };
    let gamma = 200.0;
    let thermal = ThermalParams {
        thermal_inertia: gamma,
        emissivity: 0.9,
    };
    let r_ir = 1.0;
    let bands = BandInfo::WISE;
    let band_albedos: Vec<f64> = bands.iter().map(|_| r_ir * hg.vis_albedo).collect();

    let sun2obj = Vector3::new(2.0, 0.0, 0.0);
    let obj2sun_hat = -sun2obj.normalize();

    let mut obs = Vec::new();
    for phase_deg in [0.0, 45.0, 90.0, 135.0] {
        let phi = phase_deg * std::f64::consts::PI / 180.0;
        let obj2obs_hat = Vector3::new(obj2sun_hat.x * phi.cos(), -phi.sin(), 0.0);
        let sun2obs = sun2obj + 1.0 * obj2obs_hat;

        // truth from the direct (exact) solver
        let result = tpm_total_flux(
            &bands,
            &band_albedos,
            &spin,
            &TpmShape::sphere(),
            &thermal,
            hg.diameter,
            hg.vis_albedo,
            hg.g_param,
            hg.h_mag,
            &sun2obj,
            &sun2obs,
            0.0,
        )
        .unwrap();
        for (band, &flux) in bands.into_iter().zip(&result.fluxes) {
            obs.push(FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0));
        }
    }
    (obs, hg, spin, gamma)
}

/// Synthetic TPM observations of a known oblate body (`c/a < 1`) seen across a spread
/// of aspects -- the sub-solar latitude is swept from equator-on toward pole-on, which
/// is what makes `c/a` identifiable (a single aspect is degenerate with diameter).
fn synthetic_oblate_obs(c_a: f64) -> (Vec<FluxObs>, TestHg, SpinState) {
    let hg = TestHg::new(0.15, Some(18.0), None, Some(10.0));
    let spin = SpinState {
        pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        period: 6.0 * 3600.0,
        phase0: 0.0,
        epoch0: 0.0,
    };
    let thermal = ThermalParams {
        thermal_inertia: 200.0,
        emissivity: 0.9,
    };
    let shape = TpmShape::ellipsoid(1.0, 1.0, c_a);
    let bands = BandInfo::WISE;
    let band_albedos: Vec<f64> = bands.iter().map(|_| hg.vis_albedo).collect();

    let mut obs = Vec::new();
    // sub-solar latitude sweep (pole is +z): equator-on to near pole-on.
    for lat_deg in [0.0_f64, 20.0, 40.0, 60.0, 75.0] {
        let lat = lat_deg.to_radians();
        let sun2obj = 2.0 * Vector3::new(lat.cos(), 0.0, lat.sin());
        let obj2sun_hat = -sun2obj.normalize();
        // observer at a ~20 deg phase angle, offset in the y direction.
        let obj2obs_hat = (obj2sun_hat + 0.36 * Vector3::<f64>::y()).normalize();
        let sun2obs = sun2obj + obj2obs_hat;

        let result = tpm_total_flux(
            &bands,
            &band_albedos,
            &spin,
            &shape,
            &thermal,
            hg.diameter,
            hg.vis_albedo,
            hg.g_param,
            hg.h_mag,
            &sun2obj,
            &sun2obs,
            0.0,
        )
        .unwrap();
        for (band, &flux) in bands.into_iter().zip(&result.fluxes) {
            obs.push(FluxObs::detection(flux, flux * 0.02, band, sun2obj, sun2obs, 0.0));
        }
    }
    (obs, hg, spin)
}

#[test]
fn test_tpm_oblate_fit_recovery() {
    // End-to-end: fit the oblate axis ratio c/a with NUTS and check it is recovered
    // (and clearly distinguished from a sphere). Multi-aspect data make it identifiable.
    let truth_c_a = 0.6;
    let (obs, hg, spin) = synthetic_oblate_obs(truth_c_a);
    let cfg = TpmConfig {
        spin,
        shape: TpmShape::sphere(),
        seed_shape: TpmShape::sphere_with_facets(512),
        grid: Arc::new(TpmFieldGrid::with_resolution(0.01, 100.0, 20, 15).unwrap()),
        roughness: None,
        fit_c_a: true,
        fit_b_a: false,
        fit_phase0: false,
        template: Some(EllipsoidTemplate::new(8)),
    };
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    let res = fit_mcmc(Model::Tpm, &obs, C_V, 0.9, &priors, 1, 60, 60, Some(&cfg))
        .expect("oblate TPM MCMC should produce a result");

    // c/a is the appended last draw column.
    assert_eq!(*res.column_names().last().unwrap(), "c_a");

    let median = |col: usize| {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[col]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };

    // diameter (col 0) stays well constrained by the total flux.
    let d_median = median(0);
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.35,
        "fit diameter {d_median:.2} too far from truth 10.0",
    );

    // c/a (last col) should be recovered near truth and clearly below a sphere.
    let c_a_median = median(res.column_names().len() - 1);
    assert!(
        (0.4..0.85).contains(&c_a_median),
        "fit c/a {c_a_median:.2} did not recover the oblate truth {truth_c_a}",
    );
}

/// Synthetic observations of a known triaxial body. The rotation is sampled across one
/// period at an equatorial aspect (which carries the `b/a` lightcurve and `phase0`),
/// plus a couple of off-aspect epochs that pin `c/a`.
fn synthetic_triaxial_obs(b_a: f64, c_a: f64, phase0: f64) -> (Vec<FluxObs>, SpinState) {
    let hg = TestHg::new(0.15, Some(15.0), Some(0.05), None);
    let period = 6.0 * 3600.0;
    let spin = SpinState {
        pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
        period,
        phase0,
        epoch0: 0.0,
    };
    let thermal = ThermalParams {
        thermal_inertia: 200.0,
        emissivity: 0.9,
    };
    let shape = TpmShape::ellipsoid(1.0, b_a, c_a);
    let bands = BandInfo::WISE;
    let albedos: Vec<f64> = bands.iter().map(|_| hg.vis_albedo).collect();
    let one_rotation_days = period / 86_400.0;

    let push = |s2o: &Vector3<f64>, s2b: &Vector3<f64>, epoch: f64, obs: &mut Vec<FluxObs>| {
        let r = tpm_total_flux(
            &bands,
            &albedos,
            &spin,
            &shape,
            &thermal,
            hg.diameter,
            hg.vis_albedo,
            hg.g_param,
            hg.h_mag,
            s2o,
            s2b,
            epoch,
        )
        .unwrap();
        for (band, &flux) in bands.into_iter().zip(&r.fluxes) {
            obs.push(FluxObs::detection(flux, flux * 0.02, band, *s2o, *s2b, epoch));
        }
    };

    let mut obs = Vec::new();
    // equatorial aspect, sampled across one rotation -> the b/a lightcurve + phase0.
    let s2o_eq = Vector3::new(2.0, 0.0, 0.0);
    let o2o = (-s2o_eq.normalize() + 0.36 * Vector3::<f64>::y()).normalize();
    let s2b_eq = s2o_eq + o2o;
    for k in 0..8 {
        let epoch = one_rotation_days * f64::from(k) / 8.0;
        push(&s2o_eq, &s2b_eq, epoch, &mut obs);
    }
    // off-aspect epochs (sub-solar latitude raised) -> constrain c/a.
    for lat_deg in [45.0_f64, 70.0] {
        let lat = lat_deg.to_radians();
        let s2o = 2.0 * Vector3::new(lat.cos(), 0.0, lat.sin());
        let o2o = (-s2o.normalize() + 0.36 * Vector3::<f64>::y()).normalize();
        let s2b = s2o + o2o;
        push(&s2o, &s2b, 0.0, &mut obs);
    }
    (obs, spin)
}

#[test]
#[ignore = "slow: full triaxial MCMC recovery (9-dim fit; run on demand)"]
fn test_tpm_triaxial_fit_recovery() {
    // Fit a full triaxial shape (b/a, c/a) plus the rotation phase phase0 and check
    // they are recovered. phase0 is restricted to [0, pi) and multi-started.
    let (truth_b_a, truth_c_a, truth_phase0) = (0.7, 0.5, 0.9);
    let (obs, spin) = synthetic_triaxial_obs(truth_b_a, truth_c_a, truth_phase0);
    let cfg = TpmConfig {
        spin: SpinState {
            phase0: 0.0,
            ..spin
        },
        shape: TpmShape::sphere(),
        seed_shape: TpmShape::sphere_with_facets(512),
        grid: Arc::new(TpmFieldGrid::with_resolution(0.01, 100.0, 20, 15).unwrap()),
        roughness: None,
        fit_c_a: true,
        fit_b_a: true,
        fit_phase0: true,
        // Coarse template (matches the binding's fit resolution) keeps the on-demand
        // run tractable; disk-integrated flux is converged at this facet count.
        template: Some(EllipsoidTemplate::new(8)),
    };
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, 15.0, 1.0),
        ..FluxPriors::default()
    };

    let res = fit_mcmc(Model::Tpm, &obs, C_V, 0.9, &priors, 1, 60, 60, Some(&cfg))
        .expect("triaxial TPM MCMC should produce a result");

    let cols = res.column_names();
    let idx = |name: &str| cols.iter().position(|c| *c == name).unwrap();
    let median = |col: usize| {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[col]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };

    let b_a = median(idx("b_a"));
    let c_a = median(idx("c_a"));
    let phase0_deg = median(idx("phase0"));
    assert!(
        (0.5..0.9).contains(&b_a),
        "fit b/a {b_a:.2} did not recover truth {truth_b_a}",
    );
    assert!(
        (0.35..0.7).contains(&c_a),
        "fit c/a {c_a:.2} did not recover truth {truth_c_a}",
    );
    // phase0 truth ~51.6 deg; allow a wide band (it is the most degenerate parameter).
    let truth_deg = truth_phase0.to_degrees();
    assert!(
        (phase0_deg - truth_deg).abs() < 30.0,
        "fit phase0 {phase0_deg:.0} deg far from truth {truth_deg:.0}",
    );
}

/// Build a `TpmConfig` whose grid covers the `Theta` range spanned by the
/// thermal-inertia prior at this geometry.
fn tpm_config_for(spin: &SpinState) -> TpmConfig {
    TpmConfig {
        spin: spin.clone(),
        shape: TpmShape::sphere(),
        seed_shape: TpmShape::sphere_with_facets(512),
        grid: Arc::new(TpmFieldGrid::with_resolution(0.01, 100.0, 20, 15).unwrap()),
        roughness: None,
        fit_c_a: false,
        fit_b_a: false,
        fit_phase0: false,
        template: None,
    }
}

#[test]
fn test_tpm_nll_at_truth() {
    let (obs, hg, spin, gamma) = synthetic_tpm_obs();
    let cfg = tpm_config_for(&spin);

    let nll = |g: f64| -> f64 {
        // [D, ln(thermal_inertia), H, G, f_sigma, R_IR]
        let x = [hg.diameter, g.ln(), hg.h_mag, hg.g_param, 1.0, 1.0];
        let params = Model::Tpm.unpack(&x, 0.9, C_V, ShapeFit::default());
        -Model::Tpm.log_likelihood(&params, &obs, Some(&cfg))
    };

    let at_truth = nll(gamma);
    assert!(at_truth.is_finite());
    // the likelihood should prefer the true inertia over very wrong values
    assert!(at_truth < nll(5.0), "truth should beat very low inertia");
    assert!(
        at_truth < nll(1500.0),
        "truth should beat very high inertia"
    );
}

#[test]
fn test_tpm_roughness_scales_thermal_flux() {
    // A constant roughness correction R = 1.2 should raise each model flux by scaling
    // the thermal part (reflected light is left unscaled), so smooth < rough <= 1.2x.
    let (obs, hg, spin, _gamma) = synthetic_tpm_obs();
    let smooth_cfg = tpm_config_for(&spin);

    let wl: Vec<f64> = BandInfo::WISE.iter().map(|b| b.wavelength).collect();
    let factors = vec![1.2; wl.len()];
    let rc = RoughnessCorrection::from_factors(&[1.0], &[0.5], &[0.0], &[300.0], &wl, factors);
    let mut rough_cfg = tpm_config_for(&spin);
    rough_cfg.roughness = Some(RoughnessFit {
        gamma: 0.5,
        correction: Arc::new(rc),
    });

    // [D, ln(thermal_inertia), H, G, f_sigma, R_IR]
    let x = [hg.diameter, 200.0_f64.ln(), hg.h_mag, hg.g_param, 1.0, 1.0];
    let params = Model::Tpm.unpack(&x, 0.9, C_V, ShapeFit::default());
    let smooth = Model::Tpm.evaluate_forward_model(&params, &obs, Some(&smooth_cfg));
    let rough = Model::Tpm.evaluate_forward_model(&params, &obs, Some(&rough_cfg));

    for (s, r) in smooth.model_fluxes.iter().zip(&rough.model_fluxes) {
        assert!(*r > *s, "roughness should raise flux: {r} vs {s}");
        assert!(
            *r <= s * 1.2 + 1e-9,
            "raise is bounded by R on the thermal part: {r} vs {}",
            s * 1.2
        );
    }
}

#[test]
fn test_tpm_fit_recovery() {
    let (obs, hg, spin, _gamma) = synthetic_tpm_obs();
    let cfg = tpm_config_for(&spin);
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    let res = fit_mcmc(Model::Tpm, &obs, C_V, 0.9, &priors, 1, 40, 40, Some(&cfg))
        .expect("TPM MCMC should produce a result");

    assert_eq!(res.column_names()[2], "thermal_inertia");

    let median = |col: usize| {
        let vals: Vec<f64> = res.draws.iter().map(|r| r[col]).collect();
        kete_stats::prelude::SortedData::try_from(vals)
            .unwrap()
            .median()
    };

    // diameter (column 0) is well constrained by the total flux
    let d_median = median(0);
    assert!(
        (d_median - 10.0).abs() / 10.0 < 0.3,
        "fit diameter {d_median:.2} too far from truth 10.0",
    );
    // thermal inertia (column 2) is weakly constrained; require only that it is
    // recovered to the right order of magnitude and not pinned at a prior edge.
    let gamma_median = median(2);
    assert!(
        (50.0..=800.0).contains(&gamma_median),
        "fit thermal inertia {gamma_median:.0} not in the expected range",
    );
}

#[test]
fn test_tpm_rough_layout_and_fitted_roughness() {
    // The roughness-fitting variant adds a 7th parameter (mean slope angle at x[2]),
    // exposes a "roughness" draw column, and the sampled roughness -- not the config
    // value -- drives the correction. The sampled mean slope angle is converted to the
    // internal crater opening half-angle before the correction table lookup, so this
    // also exercises that theta_bar -> gamma -> factor stays monotone.
    assert_eq!(Model::TpmRough.dim(ShapeFit::default()), 7);
    assert_eq!(
        Model::TpmRough.draw_column_names(ShapeFit::default())[3],
        "roughness"
    );

    // A correction table that increases with gamma (1.0 at 0.2 rad, 1.4 at 1.0 rad).
    let wl: Vec<f64> = BandInfo::WISE.iter().map(|b| b.wavelength).collect();
    let n_wl = wl.len();
    let gammas = [0.2_f64, 1.0];
    let phases = [0.0, std::f64::consts::PI];
    let mut factors = Vec::new();
    for &g in &gammas {
        for _ in &phases {
            for _ in 0..n_wl {
                factors.push(if g < 0.5 { 1.0 } else { 1.4 });
            }
        }
    }
    let rc = RoughnessCorrection::from_factors(&[1.0], &gammas, &phases, &[300.0], &wl, factors);

    let (obs, hg, spin, _gamma) = synthetic_tpm_obs();
    let mut cfg = tpm_config_for(&spin);
    cfg.roughness = Some(RoughnessFit {
        gamma: 0.5,
        correction: Arc::new(rc),
    });

    // [D, ln(thermal_inertia), roughness, H, G, f_sigma, R_IR]
    let low = [
        hg.diameter,
        200.0_f64.ln(),
        0.25,
        hg.h_mag,
        hg.g_param,
        1.0,
        1.0,
    ];
    let high = [
        hg.diameter,
        200.0_f64.ln(),
        0.95,
        hg.h_mag,
        hg.g_param,
        1.0,
        1.0,
    ];

    let p_low = Model::TpmRough.unpack(&low, 0.9, C_V, ShapeFit::default());
    assert!(
        (p_low.roughness - 0.25).abs() < 1e-12,
        "mean slope angle read from x[2]"
    );
    // draw row carries roughness in degrees at column 3.
    let row = p_low.to_draw_row(Model::TpmRough, ShapeFit::default());
    assert_eq!(row.len(), 8);
    assert!((row[3] - 0.25_f64.to_degrees()).abs() < 1e-9);

    let f_low = Model::TpmRough.evaluate_forward_model(&p_low, &obs, Some(&cfg));
    let p_high = Model::TpmRough.unpack(&high, 0.9, C_V, ShapeFit::default());
    let f_high = Model::TpmRough.evaluate_forward_model(&p_high, &obs, Some(&cfg));

    // Larger fitted roughness -> larger thermal correction -> higher flux.
    let mut any_increase = false;
    for (lo, hi) in f_low.model_fluxes.iter().zip(&f_high.model_fluxes) {
        assert!(
            *hi >= *lo - 1e-9,
            "fitted gamma must not lower flux: {hi} vs {lo}"
        );
        if *hi > *lo + 1e-9 {
            any_increase = true;
        }
    }
    assert!(any_increase, "the sampled roughness should drive the flux");
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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
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
            FluxObs::upper_limit(threshold, sigma, o.band, o.sun2obj, o.sun2obs, 0.0)
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
    let _ = FluxObs::bounded(1.0, 1.0, base[0].band, base[0].sun2obj, base[0].sun2obs, 0.0);
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
    let ul = FluxObs::upper_limit(1.0, 0.1, band, s2o, s2obs, 0.0);
    assert!(ul.standardized_residual(0.5, 1.0).is_none(), "below threshold is unconstrained");
    let r = ul.standardized_residual(1.5, 2.0).unwrap();
    assert!(((-0.5 / (2.0 * 0.1)) - r).abs() < 1e-12, "expected -2.5, got {r}");

    // Asymmetric detection: each side standardizes by its own sigma.
    let det = FluxObs::detection_asym(1.0, 0.1, 0.4, band, s2o, s2obs, 0.0);
    let below = det.standardized_residual(0.9, 1.0).unwrap();
    assert!((below - 1.0).abs() < 1e-12, "model below mean uses sigma_lo, got {below}");
    let above = det.standardized_residual(1.2, 1.0).unwrap();
    assert!((above - (-0.5)).abs() < 1e-12, "model above mean uses sigma_hi, got {above}");
}

#[test]
fn test_flux_bounds_constraint() {
    let base = synthetic_neatm_obs().0;
    let ob = FluxObs::bounded(1.0, 2.0, base[0].band, base[0].sun2obj, base[0].sun2obs, 0.0);

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
                    o.epoch,
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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50, None)
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
        res.reduced_chi2.is_finite() || n_obs <= Model::Frm.dim(ShapeFit::default()),
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
            tpm: None,
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
            tpm: None,
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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
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
        tpm: None,
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
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50, None).unwrap();

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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50, None).unwrap();

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
        .map(|(band, &flux)| FluxObs::detection(flux, flux * 0.05, band, sun2obj, sun2obs, 0.0))
        .collect();

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50, None)
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
            FluxObs::upper_limit(mean * 10.0, sigma, o.band, o.sun2obj, o.sun2obs, 0.0)
        })
        .collect();

    // All-upper-limit input: MCMC may or may not converge.
    // The key contract is no panic.
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let _ = fit_mcmc(Model::Neatm, &ul_obs, C_V, 0.9, &priors, 1, 50, 50, None);
}

#[test]
fn test_mcmc_draw_column_counts() {
    let (obs, hg) = synthetic_neatm_obs();
    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };

    // NEATM: 7 columns [D, pV, beaming, H, G, R_IR, f_sigma].
    let neatm_res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50, None).unwrap();
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
    let frm_res = fit_mcmc(Model::Frm, &obs, C_V, 0.9, &priors, 1, 50, 50, None).unwrap();
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
    let hg_res = fit_mcmc(Model::Hg, &hg_obs, C_V, 0.9, &hg_priors, 1, 50, 50, None).unwrap();
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
        obs.push(FluxObs::detection(flux, flux * 0.05, v_band, s2o, s2obs, 0.0));
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

    let res = fit_mcmc(Model::Hg, &obs, C_V, 0.9, &priors, 1, 100, 100, None)
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
        res.reduced_chi2.is_finite() || n_obs <= Model::Hg.dim(ShapeFit::default()),
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
            tpm: None,
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
            tpm: None,
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
            obs.push(FluxObs::detection(flux, flux * 0.05, band, *sun2obj, *sun2obs, 0.0));
        }
    }

    // Should have 8 observations (4 bands x 2 geometries).
    assert_eq!(obs.len(), 8);

    let priors = FluxPriors {
        h_mag: ParamPrior::with_gaussian(-5.0, 35.0, hg.h_mag, 1.0),
        ..FluxPriors::default()
    };
    let res = fit_mcmc(Model::Neatm, &obs, C_V, 0.9, &priors, 1, 50, 50, None)
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
