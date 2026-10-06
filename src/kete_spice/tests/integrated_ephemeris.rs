// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! The integrated ephemeris of `kete_core` against the loaded DE440 kernels.

use kete_core::ephemeris::Ephemeris;
use kete_core::frames::{Equatorial, SSB};
use kete_core::propagation::{IntegratedEphemeris, NBody};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use kete_spice::ephemeris::SpiceEphemeris;
use nalgebra::Vector3;

const AU_KM: f64 = 149_597_870.7;

/// Bodies compared, with the largest heliocentric difference from DE440 allowed at
/// +/-99 years from J2000 (km); about twice what this model gives.
const LIMITS_KM: [(i32, &str, f64); 16] = [
    (1, "Mercury", 15.0),
    (2, "Venus", 15.0),
    (3, "Earth-Moon barycenter", 100.0),
    (399, "Earth", 110.0),
    (301, "Moon", 800.0),
    (4, "Mars", 25.0),
    (5, "Jupiter", 110.0),
    (6, "Saturn", 15.0),
    (7, "Uranus", 250.0),
    (8, "Neptune", 1_900.0),
    (9, "Pluto", 1_600.0),
    (20000001, "Ceres", 400.0),
    (20000002, "Pallas", 200.0),
    (20000004, "Vesta", 400.0),
    (20000010, "Hygiea", 100.0),
    (20000704, "Interamnia", 200.0),
];

fn helio_km(eph: &impl Ephemeris, id: i32, time: Time<TDB>) -> Vector3<f64> {
    let state = eph.try_get_state_with_center(id, time, 10).unwrap();
    Vector3::from(state.pos) * AU_KM
}

/// Every served body stays within a stated distance of DE440 over +/-99 years.
#[test]
fn integrated_ephemeris_tracks_de440() {
    let spice = SpiceEphemeris::loaded().unwrap();
    let integrated = IntegratedEphemeris::new();
    let t0 = Time::<TDB>::new(2_451_545.0);
    for (id, name, limit) in LIMITS_KM {
        let mut worst = 0.0_f64;
        for years in [-99.0, -50.0, -10.0, 10.0, 50.0, 99.0] {
            let t = t0 + years * 365.25;
            let diff = (helio_km(&integrated, id, t) - helio_km(&spice, id, t)).norm();
            worst = worst.max(diff);
        }
        println!("{name:<22} worst {worst:10.1} km (limit {limit} km)");
        assert!(worst < limit, "{name}: {worst} km from DE440");
    }
    // The Moon about the Earth, the quantity the lunar model limits.
    let mut worst = 0.0_f64;
    for years in [-99.0, -50.0, 50.0, 99.0] {
        let t = t0 + years * 365.25;
        let geo = |eph: &dyn Fn(i32) -> Vector3<f64>| eph(301) - eph(399);
        let ours = geo(&|id| helio_km(&integrated, id, t));
        let theirs = geo(&|id| helio_km(&spice, id, t));
        worst = worst.max((ours - theirs).norm());
    }
    println!(
        "{:<22} worst {worst:10.1} km (limit 800 km)",
        "Moon about the Earth"
    );
    assert!(worst < 800.0);
}

/// The barycenter of the integrated ephemeris (the center of mass of the bodies it
/// integrates) is the documented distance from DE440's, which also counts the Kuiper
/// belt: the Sun relative to it agrees with DE440 within 150 km over +/-99 years.
#[test]
fn barycenter_matches_de440() {
    let spice = SpiceEphemeris::loaded().unwrap();
    let integrated = IntegratedEphemeris::new();
    let t0 = Time::<TDB>::new(2_451_545.0);
    let mut worst = 0.0_f64;
    for years in [-99.0, -50.0, -10.0, 0.0, 10.0, 50.0, 99.0] {
        let t = t0 + years * 365.25;
        let sun = |eph: &dyn Ephemeris| -> Vector3<f64> {
            Vector3::from(eph.try_get_state_with_center(10, t, 0).unwrap().pos) * AU_KM
        };
        worst = worst.max((sun(&integrated) - sun(&spice)).norm());
    }
    println!("Sun about the barycenter: worst {worst:.1} km from DE440");
    assert!(worst < 150.0, "barycenters differ by {worst} km");
}

/// Heliocentric position (km) at `end` of `start` propagated with the planets of `eph`.
fn propagate(eph: &impl Ephemeris, start: &State<Equatorial>, end: Time<TDB>) -> Vector3<f64> {
    let ssb: State<Equatorial, SSB> = eph.try_to_ssb(start.clone()).unwrap();
    let moved = ssb.propagate_with(&NBody::new(eph, false), end).unwrap();
    let helio = eph.try_to_sun(moved.into()).unwrap();
    Vector3::from(helio.pos) * AU_KM
}

/// An asteroid propagated with the planets from the integrated ephemeris lands close
/// to the same propagation with the planets from DE440.
///
/// The start is heliocentric and each run converts it with its own ephemeris: the
/// two barycenters differ by up to about 140 km, so the same barycentric state would
/// be a different heliocentric one, and that grows along an orbit.
#[test]
fn n_body_propagation_with_integrated_ephemeris() {
    let spice = SpiceEphemeris::loaded().unwrap();
    let integrated = IntegratedEphemeris::new();
    let t0 = Time::<TDB>::new(2_451_545.0);
    let end = t0 + 20.0 * 365.25;
    // A main-belt orbit near 2.7 AU, and a Mars-crossing one.
    let starts = [
        ([2.7, 0.0, 0.05], [0.0, 0.0104, 0.0012]),
        ([1.3, 0.2, 0.0], [-0.002, 0.0175, 0.002]),
    ];
    for (pos, vel) in starts {
        let start = State::<Equatorial>::new(kete_core::desigs::Desig::Empty, t0, pos, vel, 10);
        let diff = (propagate(&spice, &start, end) - propagate(&integrated, &start, end)).norm();
        println!("asteroid from {pos:?} (heliocentric): {diff:.1} km apart after 20 yr");
        assert!(diff < 50.0, "propagations differ by {diff} km");
    }
}
