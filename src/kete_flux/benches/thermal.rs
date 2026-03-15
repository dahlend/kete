#![allow(missing_docs, reason = "Unnecessary for benchmarks")]
#![allow(unused_results, reason = "Unnecessary for benchmarks")]
#![allow(clippy::missing_assert_message, reason = "Unnecessary for benchmarks")]

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kete_flux::{BandInfo, frm_thermal_flux, neatm_thermal_flux, resolve_hg_params};
use nalgebra::Vector3;
use pprof::criterion::{Output, PProfProfiler};
use std::hint::black_box;

fn neatm_bench(
    bands: &[BandInfo],
    diam: f64,
    vis_albedo: f64,
    g_param: f64,
    beaming: f64,
    emissivity: f64,
) {
    let sun2obj = Vector3::new(1.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 1.0, 0.0);
    let result = neatm_thermal_flux(
        bands, diam, vis_albedo, g_param, beaming, emissivity, &sun2obj, &sun2obs,
    );
    assert!(!result.is_empty());
}

fn frm_bench(bands: &[BandInfo], diam: f64, vis_albedo: f64, g_param: f64, emissivity: f64) {
    let sun2obj = Vector3::new(1.0, 0.0, 0.0);
    let sun2obs = Vector3::new(0.0, 1.0, 0.0);
    let result = frm_thermal_flux(
        bands, diam, vis_albedo, g_param, emissivity, &sun2obj, &sun2obs,
    );
    assert!(!result.is_empty());
}

#[allow(clippy::missing_panics_doc, reason = "Benchmarking only")]
pub fn neatm_benchmark(c: &mut Criterion) {
    let mut neatm_group = c.benchmark_group("NEATM");

    let (_h_mag, vis_albedo, diam, _c_hg) =
        resolve_hg_params(Some(15.0), Some(0.2), None, Some(1329.0)).unwrap();
    let vis_albedo = vis_albedo.unwrap();
    let diam = diam.unwrap();
    let g_param = 0.15;

    let wise_bands = BandInfo::new_wise().to_vec();

    let generic_bands: Vec<BandInfo> = [1000.0; 4]
        .iter()
        .zip([f64::NAN; 4])
        .map(|(wavelength, z_mag)| BandInfo::new(*wavelength, 1.0, z_mag, None))
        .collect();

    neatm_group.bench_function(BenchmarkId::new("neatm", "No Color Correction"), |b| {
        b.iter(|| {
            neatm_bench(
                black_box(&generic_bands),
                black_box(diam),
                black_box(vis_albedo),
                black_box(g_param),
                black_box(1.0),
                black_box(0.9),
            );
        });
    });
    neatm_group.bench_function(BenchmarkId::new("neatm", "Wise Color Correction"), |b| {
        b.iter(|| {
            neatm_bench(
                black_box(&wise_bands),
                black_box(diam),
                black_box(vis_albedo),
                black_box(g_param),
                black_box(1.0),
                black_box(0.9),
            );
        });
    });
}

#[allow(clippy::missing_panics_doc, reason = "Benchmarking only")]
pub fn frm_benchmark(c: &mut Criterion) {
    let mut frm_group = c.benchmark_group("FRM");

    let (_h_mag, vis_albedo, diam, _c_hg) =
        resolve_hg_params(Some(15.0), Some(0.2), None, Some(1329.0)).unwrap();
    let vis_albedo = vis_albedo.unwrap();
    let diam = diam.unwrap();
    let g_param = 0.15;

    let wise_bands = BandInfo::new_wise().to_vec();

    let generic_bands: Vec<BandInfo> = [1000.0; 4]
        .iter()
        .zip([f64::NAN; 4])
        .map(|(wavelength, z_mag)| BandInfo::new(*wavelength, 1.0, z_mag, None))
        .collect();

    frm_group.bench_function(BenchmarkId::new("frm", "No Color Correction"), |b| {
        b.iter(|| {
            frm_bench(
                black_box(&generic_bands),
                black_box(diam),
                black_box(vis_albedo),
                black_box(g_param),
                black_box(0.9),
            );
        });
    });
    frm_group.bench_function(BenchmarkId::new("frm", "Wise Color Correction"), |b| {
        b.iter(|| {
            frm_bench(
                black_box(&wise_bands),
                black_box(diam),
                black_box(vis_albedo),
                black_box(g_param),
                black_box(0.9),
            );
        });
    });
}

criterion_group!(name=thermal;
                config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
                targets=neatm_benchmark, frm_benchmark);
criterion_main!(thermal);
