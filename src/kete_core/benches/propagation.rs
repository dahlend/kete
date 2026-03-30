//! Benchmarks for propagation algorithms in the kete library.

#![allow(missing_docs, reason = "Unnecessary for benchmarks")]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kete_core::constants;
use kete_core::prelude::*;
use pprof::criterion::{Output, PProfProfiler};

static CIRCULAR: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Circular".into()),
        2451545.0.into(),
        [0.0, 1., 0.0].into(),
        [-constants::GMS_SQRT, 0.0, 0.0].into(),
        0,
    )
});
static ELLIPTICAL: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Elliptical".into()),
        2451545.0.into(),
        [0.0, 1.5, 0.0].into(),
        [-constants::GMS_SQRT, 0.0, 0.0].into(),
        0,
    )
});
static PARABOLIC: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Parabolic".into()),
        2451545.0.into(),
        [0.0, 2., 0.0].into(),
        [-constants::GMS_SQRT, 0.0, 0.0].into(),
        0,
    )
});

static HYPERBOLIC: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Hyperbolic".into()),
        2451545.0.into(),
        [0.0, 3., 0.0].into(),
        [-constants::GMS_SQRT, 0.0, 0.0].into(),
        0,
    )
});

fn prop_2_body_kepler(state: &State<Ecliptic>, dt: f64) {
    let _ = propagate_two_body(state, state.epoch + dt).unwrap();
}

/// Benchmark functions for the propagation algorithms
#[allow(clippy::missing_panics_doc, reason = "Benchmarking only")]
pub fn two_body_analytic(c: &mut Criterion) {
    let mut twobody_group = c.benchmark_group("2-Body-Analytic");

    for state in [
        CIRCULAR.clone(),
        ELLIPTICAL.clone(),
        PARABOLIC.clone(),
        HYPERBOLIC.clone(),
    ] {
        let Desig::Name(name) = &state.desig else {
            panic!()
        };
        let _ = twobody_group.bench_with_input(BenchmarkId::new("Single", name), &state, |b, s| {
            b.iter(|| prop_2_body_kepler(s, black_box(1000.0)));
        });
    }
}

criterion_group!(name=benches;
                 config = Criterion::default().sample_size(30).measurement_time(Duration::from_secs(15)).with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
                 targets=two_body_analytic);

criterion_main!(benches);
