//! Benchmarks for SPK-dependent propagation algorithms.

#![allow(missing_docs, reason = "Unnecessary for benchmarks")]

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use kete_core::constants;
use kete_core::forces::{DustNonGrav, ForceSet};
use kete_core::prelude::*;
use kete_core::state::StateLike;
use kete_core::state::propagate_with_stm;
use kete_spice::propagation::{Recenter, SpkNBody, propagate_n_body_vec};
use kete_spice::spk::LOADED_SPK;
use pprof::criterion::{Output, PProfProfiler};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

static CIRCULAR: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Circular".into()),
        2451545.0,
        [0.0, 1., 0.0],
        [-constants::GMS_SQRT, 0.0, 0.0],
        0,
    )
});
static ELLIPTICAL: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Elliptical".into()),
        2451545.0,
        [0.0, 1.5, 0.0],
        [-constants::GMS_SQRT, 0.0, 0.0],
        0,
    )
});
static PARABOLIC: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Parabolic".into()),
        2451545.0,
        [0.0, 2., 0.0],
        [-constants::GMS_SQRT, 0.0, 0.0],
        0,
    )
});

static HYPERBOLIC: std::sync::LazyLock<State<Ecliptic>> = std::sync::LazyLock::new(|| {
    State::new(
        Desig::Name("Hyperbolic".into()),
        2451545.0,
        [0.0, 3., 0.0],
        [-constants::GMS_SQRT, 0.0, 0.0],
        0,
    )
});

fn prop_n_body_radau(state: State<Ecliptic>, dt: f64) {
    let spk = LOADED_SPK.try_read().unwrap();
    let planets = kete_core::forces::GravParams::planets();
    let jd = state.epoch + dt;
    let eq_state: State<Equatorial, SSB> = state.into_frame().try_into().unwrap();
    let _ = eq_state
        .propagate_with(&SpkNBody::new(&spk, &planets), jd)
        .unwrap();
}

fn prop_n_body_vec_radau(state: State<Ecliptic>, dt: f64) {
    let spk = &LOADED_SPK.read().unwrap();
    let state_sun = spk.try_to_sun(state).unwrap();
    let jd = state_sun.epoch + dt;
    let states: Vec<State<Equatorial, SunCenter>> = vec![state_sun.into_frame(); 100];
    let non_gravs = vec![None; 100];
    let _ = propagate_n_body_vec(states, jd, None, non_gravs).unwrap();
}

fn prop_n_body_radau_par(state: &State<Ecliptic>, dt: f64) {
    let states: Vec<State<_>> = (0..100).map(|_| state.clone()).collect();
    let planets = kete_core::forces::GravParams::planets();
    let _tmp: Vec<_> = states
        .into_par_iter()
        .map(|s| {
            let spk = LOADED_SPK.try_read().unwrap();
            let jd = s.epoch + dt;
            let eq: State<Equatorial, SSB> = s.into_frame().try_into().unwrap();
            eq.propagate_with(&SpkNBody::new(&spk, &planets), jd)
                .unwrap()
        })
        .collect();
}

// New path: state.propagate_with(&SpkNBody, jd). Builds a fresh SpkNBody
// per call so the per-step cache state matches the legacy path's
// AccelSPKMeta lifetime (rebuilt per propagation in the legacy code too).
fn prop_n_body_force(state: State<Ecliptic>, dt: f64) {
    let spk = LOADED_SPK.try_read().unwrap();
    let planets = kete_core::forces::GravParams::planets();
    let jd = state.epoch + dt;
    let eq_state: State<Equatorial, SSB> = state.into_frame().try_into().unwrap();
    let force = SpkNBody::new(&spk, &planets);
    let _ = eq_state.propagate_with(&force, jd).unwrap();
}

// Dust non-grav benchmark: composed `ForceSet { SpkNBody,
// Recenter<SSB, DustNonGrav> }` exercising the variational integrator
// path with a single free parameter (beta).
const DUST_BETA: f64 = 1e-3;

fn prop_n_body_force_dust(state: State<Ecliptic>, dt: f64) {
    let spk = LOADED_SPK.try_read().unwrap();
    let planets = kete_core::forces::GravParams::planets();
    let jd = state.epoch + dt;
    let eq_state: State<Equatorial, SSB> = state.into_frame().try_into().unwrap();
    let force: ForceSet<'_, Equatorial, SSB> = ForceSet::new()
        .with(Box::new(SpkNBody::new(&spk, &planets)))
        .with(Box::new(Recenter::<SSB, _>::new(&spk, DustNonGrav)));
    let _ = propagate_with_stm(
        &force,
        eq_state.pos.into(),
        eq_state.vel.into(),
        &[DUST_BETA],
        eq_state.epoch,
        jd,
    )
    .unwrap();
}

fn prop_n_body_force_par(state: &State<Ecliptic>, dt: f64) {
    let states: Vec<State<_>> = (0..100).map(|_| state.clone()).collect();
    // Cache the planet list once across all tasks; SpkNBody borrows it,
    // SpkNBody borrows the slice for the duration of integration.
    let planets = kete_core::forces::GravParams::planets();
    let _tmp: Vec<_> = states
        .into_par_iter()
        .map(|s| {
            let spk = LOADED_SPK.try_read().unwrap();
            let jd = s.epoch + dt;
            let eq: State<Equatorial, SSB> = s.into_frame().try_into().unwrap();
            let force = SpkNBody::new(&spk, &planets);
            eq.propagate_with(&force, jd).unwrap()
        })
        .collect();
}

/// Benchmark functions for the propagation algorithms
#[allow(clippy::missing_panics_doc, reason = "Benchmarking only")]
fn n_body_prop(c: &mut Criterion) {
    let mut nbody_group = c.benchmark_group("N-Body");

    for state in [
        CIRCULAR.clone(),
        ELLIPTICAL.clone(),
        PARABOLIC.clone(),
        HYPERBOLIC.clone(),
    ] {
        let Desig::Name(name) = &state.desig else {
            panic!()
        };
        let _ = nbody_group.bench_with_input(BenchmarkId::new("Single", name), &state, |b, s| {
            b.iter(|| prop_n_body_radau(black_box(s.clone()), black_box(1000.0)));
        });

        let _ =
            nbody_group.bench_with_input(BenchmarkId::new("Single-Force", name), &state, |b, s| {
                b.iter(|| prop_n_body_force(black_box(s.clone()), black_box(1000.0)));
            });

        let _ = nbody_group.bench_with_input(BenchmarkId::new("Parallel", name), &state, |b, s| {
            b.iter(|| prop_n_body_radau_par(black_box(s), black_box(1000.0)));
        });

        let _ = nbody_group.bench_with_input(
            BenchmarkId::new("Parallel-Force", name),
            &state,
            |b, s| b.iter(|| prop_n_body_force_par(black_box(s), black_box(1000.0))),
        );

        // Dust non-grav: short arc (100 days) since variational
        // integration is ~3x more expensive than the bare propagation
        // and the question is per-step overhead, not arc length.
        let _ = nbody_group.bench_with_input(BenchmarkId::new("Dust", name), &state, |b, s| {
            b.iter(|| {
                prop_n_body_force_dust(black_box(s.clone()), black_box(100.0));
            });
        });
    }
}

/// Benchmark functions for the propagation algorithms
#[allow(clippy::missing_panics_doc, reason = "Benchmarking only")]
pub fn n_body_prop_vec(c: &mut Criterion) {
    let mut nbody_group = c.benchmark_group("N-Body-Vec");

    for state in [
        CIRCULAR.clone(),
        ELLIPTICAL.clone(),
        PARABOLIC.clone(),
        HYPERBOLIC.clone(),
    ] {
        let Desig::Name(name) = &state.desig else {
            panic!()
        };
        let _ = nbody_group.bench_with_input(BenchmarkId::new("Single", name), &state, |b, s| {
            b.iter(|| prop_n_body_vec_radau(black_box(s.clone()), black_box(1000.0)));
        });
    }
}

criterion_group!(name=benches;
                 config = Criterion::default().sample_size(30).measurement_time(Duration::from_secs(15)).with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
                 targets=n_body_prop_vec, n_body_prop);

criterion_main!(benches);
