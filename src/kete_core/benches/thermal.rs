#![allow(missing_docs, reason = "Unnecessary for benchmarks")]
#![allow(unused_results, reason = "Unnecessary for benchmarks")]
#![allow(clippy::missing_assert_message, reason = "Unnecessary for benchmarks")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kete_core::flux::{FrmParams, HGParams, NeatmParams};
use pprof::criterion::{Output, PProfProfiler};
use std::hint::black_box;

fn neatm_bench(params: &NeatmParams) {
    assert!(
        params
            .apparent_thermal_flux(&[1.0, 0.0, 0.0].into(), &[0.0, 1.0, 0.0].into(),)
            .is_some()
    );
}

fn frm_bench(params: &FrmParams) {
    assert!(
        params
            .apparent_thermal_flux(&[1.0, 0.0, 0.0].into(), &[0.0, 1.0, 0.0].into(),)
            .is_some()
    );
}

pub fn neatm_benchmark(c: &mut Criterion) {
    let mut neatm_group = c.benchmark_group("NEATM");

    let hg_params = HGParams::try_new(
        "test".into(),
        0.15,
        Some(15.0),
        Some(1329.0),
        Some(0.2),
        None,
    )
    .unwrap();
    let wise_params = NeatmParams {
        obs_bands: kete_core::flux::ObserverBands::Wise,
        band_albedos: vec![0.2; 4],
        beaming: 1.0,
        hg_params: hg_params.clone(),
        emissivity: 0.9,
    };
    let generic_params = NeatmParams {
        obs_bands: kete_core::flux::ObserverBands::Generic {
            bands: vec![1000.0; 4],
            zero_mags: None,
            solar_correction: vec![1.0; 4],
        },
        band_albedos: vec![0.2; 4],
        beaming: 1.0,
        hg_params,
        emissivity: 0.9,
    };

    neatm_group.bench_function(BenchmarkId::new("neatm", "No Color Correction"), |b| {
        b.iter(|| neatm_bench(black_box(&generic_params)));
    });
    neatm_group.bench_function(BenchmarkId::new("neatm", "Wise Color Correction"), |b| {
        b.iter(|| neatm_bench(black_box(&wise_params)));
    });
}

pub fn frm_benchmark(c: &mut Criterion) {
    let mut frm_group = c.benchmark_group("FRM");

    let hg_params = HGParams::try_new(
        "test".into(),
        0.15,
        Some(15.0),
        Some(1329.0),
        Some(0.2),
        None,
    )
    .unwrap();
    let wise_params = FrmParams {
        obs_bands: kete_core::flux::ObserverBands::Wise,
        band_albedos: vec![0.2; 4],
        hg_params: hg_params.clone(),
        emissivity: 0.9,
    };
    let generic_params = FrmParams {
        obs_bands: kete_core::flux::ObserverBands::Generic {
            bands: vec![1000.0; 4],
            zero_mags: None,
            solar_correction: vec![1.0; 4],
        },
        band_albedos: vec![0.2; 4],
        hg_params,
        emissivity: 0.9,
    };

    frm_group.bench_function(BenchmarkId::new("frm", "No Color Correction"), |b| {
        b.iter(|| frm_bench(black_box(&generic_params)));
    });
    frm_group.bench_function(BenchmarkId::new("frm", "Wise Color Correction"), |b| {
        b.iter(|| frm_bench(black_box(&wise_params)));
    });
}

criterion_group!(name=thermal;
                config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
                targets=neatm_benchmark, frm_benchmark);
criterion_main!(thermal);
