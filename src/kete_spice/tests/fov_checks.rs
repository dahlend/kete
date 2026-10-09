// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Visibility checks with body states from the loaded SPK files.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::constants::{GMS, GMS_SQRT};
use kete_core::desigs::Desig;
use kete_core::forces::{DustNonGrav, NonGravKind, ParameterMask};
use kete_core::fov::{FovLike, GenericCone, GenericRectangle, OmniDirectional};
use kete_core::fov::{check_ephemeris, check_visible};
use kete_core::frames::{Equatorial, Vector};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use kete_spice::ephemeris::SpiceEphemeris;
use kete_spice::spk::LOADED_SPK;

use kete_core::propagation::NBody;

#[test]
fn test_check_rectangle_visible() {
    kete_spice::test_data::ensure_test_spk();
    let circular = State::new(
        Desig::Empty,
        2451545.0,
        [0.0, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );
    let circular_back = State::<Equatorial>::new(
        Desig::Empty,
        2451545.0,
        [1.0, 0.0, 0.0],
        [0.0, GMS_SQRT, 0.0],
        10,
    );

    let circular_back_ssb = {
        let spk = LOADED_SPK.try_read().unwrap();
        spk.try_to_ssb(circular_back.clone()).unwrap()
    };

    for offset in [-10.0_f64, -5.0, 0.0, 5.0, 10.0] {
        let eph = SpiceEphemeris::loaded().unwrap();
        let force = NBody::new(&eph, false);
        let off_state = circular_back_ssb
            .clone()
            .propagate_with(&force, circular_back_ssb.epoch - offset)
            .unwrap();

        let vec = circular_back.pos - circular.pos;
        let fov = GenericRectangle::new(vec, 0.0001, 0.01, 0.01, circular.clone()).unwrap();
        let seen = check_visible(&eph, &[fov], &[off_state.into()], &[], false).unwrap();
        assert_eq!(seen.len(), 1);
    }
}

/// A slow object near the observer passes the linear pre-filter when its
/// state and the observer have different centers.
#[test]
fn linear_prefilter_matches_the_observer_center() {
    kete_spice::test_data::ensure_test_spk();
    let observer = State::<Equatorial>::new(
        Desig::Empty,
        2451545.0,
        [0.0, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );
    // At rest relative to the observer, so the pre-filter allows no slack.
    let object = State::<Equatorial>::new(
        Desig::Empty,
        2451545.0,
        [1e-3, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );
    let object_ssb: State<Equatorial> = {
        let spk = LOADED_SPK.try_read().unwrap();
        spk.try_to_ssb(object).unwrap().into()
    };
    let fov = GenericRectangle::new([1.0, 0.0, 0.0].into(), 0.0, 0.01, 0.01, observer).unwrap();
    let seen = check_visible(
        &SpiceEphemeris::loaded().unwrap(),
        &[fov],
        &[object_ssb],
        &[],
        false,
    )
    .unwrap();
    assert_eq!(seen.len(), 1);
}

/// Test the light delay computations for the different checks
#[test]
fn test_check_omni_visible() {
    kete_spice::test_data::ensure_test_spk();
    // Build an observer, and check the observability of an asteroid with different
    // offsets from the observer time.
    // this will exercise the position, velocity, and time offsets due to light delay.
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let observer = State::new(
        Desig::Empty,
        2451545.0,
        [0.0, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );

    for offset in [-10.0, -5.0, 0.0, 5.0, 10.0] {
        let asteroid = spk
            .try_get_state_with_center(20000042, observer.epoch + offset, 10)
            .unwrap();
        let fov = OmniDirectional::new(observer.clone());

        let seen =
            check_visible(&eph, std::slice::from_ref(&fov), &[asteroid], &[], false).unwrap();
        assert_eq!(seen.len(), 1);
        let n_body = &seen[0].2.states[0];
        let dist = (n_body.pos - observer.pos).norm();
        assert!(((observer.epoch - n_body.epoch).elapsed - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
        let exact = spk
            .try_get_state_with_center(20000042, n_body.epoch, 10)
            .unwrap();
        // check that we are within about 150m
        assert!((n_body.pos - exact.pos).norm() < 1e-9);

        // Check spk queries
        let spk_check = &check_ephemeris(&eph, &fov, &[20000042]).unwrap()[0];
        assert!(spk_check.is_some());
        let spk_check = &spk_check.as_ref().unwrap().states[0];
        assert!(
            ((observer.epoch - spk_check.epoch).elapsed - dist * C_AU_PER_DAY_INV).abs() < 1e-6
        );
        let exact = spk
            .try_get_state_with_center(20000042, spk_check.epoch, 10)
            .unwrap();
        // check that we are within about 150 micron
        assert!((spk_check.pos - exact.pos).norm() < 1e-12);
    }

    // The Sun is co-located with itself in a Sun-centered FOV check
    let sun_fov = OmniDirectional::new(observer.clone());
    let sun_check = &check_ephemeris(&eph, &sun_fov, &[10]).unwrap()[0];
    assert!(sun_check.is_some());
    let sun_state = &sun_check.as_ref().unwrap().states[0];
    // The Sun is always at the solar center.
    assert!(sun_state.pos.norm() < 1e-12);
}

/// The observer may have any center: the same FOV finds the same object whether its
/// observer is stored relative to the Sun or to the solar system barycenter.
#[test]
fn check_ephemeris_accepts_any_observer_center() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let observer = State::new(
        Desig::Empty,
        2451545.0,
        [0.0, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );
    let target = spk
        .try_get_state_with_center(20000042, observer.epoch, 10)
        .unwrap();
    let pointing = target.pos - observer.pos;

    let mut barycentric = observer.clone();
    spk.try_change_center(&mut barycentric, 0).unwrap();
    // A 1e-3 radian cone is much narrower than the Sun's offset from the barycenter
    // as seen from the asteroid, so a check that ignored the observer's center
    // would look in the wrong place.
    for obs in [observer, barycentric] {
        let fov = GenericCone::new(pointing, 1e-3, obs);
        let seen = &check_ephemeris(&eph, &fov, &[20000042]).unwrap()[0];
        assert!(seen.is_some());
    }
}

/// An object at rest relative to the observer is seen in FOVs at both edges of one
/// half-day group. The pre-filter places the object at the middle of the group,
/// where a quarter day of its own motion carries it across the line of sight, far
/// outside the FOVs, although it does not move relative to the observer.
#[test]
fn prefilter_allows_for_curvature() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let force = NBody::new(&eph, false);
    // An object on a circular orbit at 1 au. Each observer sits 0.005 au sunward of
    // it, looking out along the radius, across the direction of motion.
    let object = State::<Equatorial>::new(
        Desig::Name("companion".into()),
        2451545.0,
        [1.0, 0.0, 0.0],
        [0.0, GMS_SQRT, 0.0],
        10,
    );
    let object_ssb = eph.spk().try_to_ssb(object.clone()).unwrap();
    let offset = Vector::<Equatorial>::new([0.005, 0.0, 0.0]);
    // Two FOVs at the edges of one half-day group.
    let fovs: Vec<_> = [2451545.0, 2451545.0 + 0.49]
        .into_iter()
        .map(|jd| {
            let mut at: State<Equatorial> = object_ssb
                .clone()
                .propagate_with(&force, jd.into())
                .unwrap()
                .into();
            eph.spk().try_change_center(&mut at, 10).unwrap();
            let radial = at.pos.normalize() * offset.norm();
            let observer = State::<Equatorial>::new(Desig::Empty, jd, at.pos - radial, at.vel, 10);
            GenericRectangle::new(radial, 0.0, 0.01, 0.01, observer).unwrap()
        })
        .collect();
    let seen = check_visible(&eph, &fovs, &[object], &[], false).unwrap();
    assert_eq!(seen.len(), 2);
}

/// An object that impacts the Earth is still seen in the FOVs before the impact,
/// including those in the half-day group of the impact, whose middle is after it.
#[test]
fn seen_until_impact() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let epoch = Time::<TDB>::new(2451545.0);
    // 0.01 au from the Earth, closing at 0.01 au/day, so it hits about a day later.
    let earth = spk.try_get_state_with_center(399, epoch, 10).unwrap();
    let impactor = State::<Equatorial>::new(
        Desig::Name("impactor".into()),
        epoch,
        earth.pos + Vector::new([0.01, 0.0, 0.0]),
        earth.vel - Vector::new([0.01, 0.0, 0.0]),
        10,
    );
    let observer =
        |jd: f64| State::<Equatorial>::new(Desig::Empty, jd, [0.0, 1.5, 0.0], [0.0; 3], 10);
    // One half-day group holds every FOV, the impact falls between the second and the
    // third, and the middle of the group is after it.
    let fovs: Vec<_> = [0.80, 0.95, 1.29]
        .into_iter()
        .map(|dt| OmniDirectional::new(observer(2451545.0 + dt)))
        .collect();
    let seen = check_visible(&eph, &fovs, &[impactor], &[], false).unwrap();
    let fov_idx: Vec<usize> = seen.iter().map(|(idx, _, _)| *idx).collect();
    assert_eq!(fov_idx, vec![0, 1]);
}

/// Narrow FOVs tracking an asteroid through a night see it in every FOV, although
/// it moves farther than the width of a FOV between the middle of the night and the
/// first and last FOVs.
#[test]
fn seen_through_a_night() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let start = Time::<TDB>::new(2451545.0);
    let fovs: Vec<_> = (0..=10)
        .map(|hour| {
            let jd = start + f64::from(hour) / 24.0;
            let earth = spk.try_get_state_with_center(399, jd, 10).unwrap();
            let target = spk.try_get_state_with_center(20000042, jd, 10).unwrap();
            let light_time = (target.pos - earth.pos).norm() * C_AU_PER_DAY_INV;
            let target = spk
                .try_get_state_with_center(20000042, jd - light_time, 10)
                .unwrap();
            GenericRectangle::new(target.pos - earth.pos, 0.0, 2e-4, 2e-4, earth).unwrap()
        })
        .collect();
    let asteroid = spk.try_get_state_with_center(20000042, start, 10).unwrap();
    let seen = check_visible(&eph, &fovs, &[asteroid], &[], false).unwrap();
    assert_eq!(seen.len(), fovs.len());
}

/// An object falling into Jupiter against Jupiter's orbital motion is at rest relative
/// to the SSB in the middle of the night, so its speed there allows for none of its
/// motion. Narrow FOVs centered on it still see it in every FOV.
#[test]
fn seen_while_at_rest_relative_to_the_ssb() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let epoch = Time::<TDB>::new(2451545.0);
    let gm_jupiter = GMS / 1047.348644;
    let jupiter = spk.try_get_state_with_center(5, epoch, 0).unwrap();
    let unit = jupiter.vel / jupiter.vel.norm();
    // The distance at which the speed of a fall from rest equals Jupiter's speed.
    let dist = 2.0 * gm_jupiter / jupiter.vel.norm().powi(2);
    let falling = State::<Equatorial>::new(
        Desig::Name("falling".into()),
        epoch,
        jupiter.pos + unit * dist,
        jupiter.vel - unit * (2.0 * gm_jupiter / dist).sqrt(),
        0,
    );
    assert!(falling.vel.norm() < 1e-12);

    let fovs: Vec<_> = (-5..=5)
        .map(|step| {
            let earth = spk
                .try_get_state_with_center(399, epoch + f64::from(step) * 0.05, 10)
                .unwrap();
            let omni = OmniDirectional::new(earth.clone());
            let exact =
                check_visible(&eph, &[omni], std::slice::from_ref(&falling), &[], false).unwrap();
            let pos = exact[0].2.states[0].pos;
            GenericRectangle::new(pos - earth.pos, 0.0, 1e-5, 1e-5, earth).unwrap()
        })
        .collect();
    let seen = check_visible(&eph, &fovs, &[falling], &[], false).unwrap();
    assert_eq!(seen.len(), fovs.len());
}

/// A FOV whose observer the ephemeris cannot place is an error.
#[test]
fn unknown_observer_center_is_an_error() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let observer = State::<Equatorial>::new(Desig::Empty, 2451545.0, [0.0; 3], [0.0; 3], 987_654);
    let asteroid = eph
        .spk()
        .try_get_state_with_center(20000042, Time::<TDB>::new(2451545.0), 10)
        .unwrap();
    let fov = [OmniDirectional::new(observer)];
    assert!(check_visible(&eph, &fov, &[asteroid], &[], false).is_err());
}

/// States from several blocks of integration that land in one patch keep their
/// input order.
#[test]
fn patch_keeps_input_order() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let asteroid = spk
        .try_get_state_with_center(20000042, Time::<TDB>::new(2451545.0), 10)
        .unwrap();
    let states: Vec<_> = (0..300)
        .map(|idx| {
            let mut state = asteroid.clone();
            state.desig = Desig::Name(format!("copy {idx}"));
            state
        })
        .collect();
    let observer = State::<Equatorial>::new(Desig::Empty, 2451546.0, [0.0, 1.0, 0.0], [0.0; 3], 10);
    let fov = [OmniDirectional::new(observer)];
    let seen = check_visible(&eph, &fov, &states, &[], false).unwrap();
    assert_eq!(seen.len(), 1);
    let desigs: Vec<_> = seen[0].2.states.iter().map(|s| s.desig.clone()).collect();
    let expected: Vec<_> = states.iter().map(|s| s.desig.clone()).collect();
    assert_eq!(desigs, expected);
}

/// One propagation serves FOVs both before and after the state epoch, and each
/// reported state matches the SPK at the time light left the object.
#[test]
fn many_fovs_from_one_trajectory() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let epoch = Time::<TDB>::new(2451545.0);
    let asteroid = spk.try_get_state_with_center(20000042, epoch, 10).unwrap();

    // Weekly FOVs from Earth over a year centered on the state epoch, alternately
    // pointed at the asteroid and away from it. Given in reverse time order.
    let mut fovs = Vec::new();
    let mut expected = Vec::new();
    for week in (-26..26).rev() {
        let jd = epoch + f64::from(week) * 7.0;
        let earth = spk.try_get_state_with_center(399, jd, 10).unwrap();
        let target = spk.try_get_state_with_center(20000042, jd, 10).unwrap();
        let mut pointing = target.pos - earth.pos;
        if week % 2 == 0 {
            expected.push(fovs.len());
        } else {
            pointing = -pointing;
        }
        fovs.push(GenericRectangle::new(pointing, 0.0, 0.01, 0.01, earth).unwrap());
    }

    let seen = check_visible(&eph, &fovs, std::slice::from_ref(&asteroid), &[], false).unwrap();
    let fov_idx: Vec<usize> = seen.iter().map(|(idx, _, _)| *idx).collect();
    assert_eq!(fov_idx, expected);
    for (idx, _, patch) in &seen {
        let observer = fovs[*idx].observer();
        let state = &patch.states[0];
        let dist = (state.pos - observer.pos).norm();
        assert!(((observer.epoch - state.epoch).elapsed - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
        let exact = spk
            .try_get_state_with_center(20000042, state.epoch, 10)
            .unwrap();
        assert!((state.pos - exact.pos).norm() < 1e-8);
    }
}

/// A non-gravitational model must change the observed state, including when the
/// state epoch is close to the observer epoch.
#[test]
fn test_check_visible_non_grav() {
    kete_spice::test_data::ensure_test_spk();
    let observer = State::new(
        Desig::Empty,
        2451545.0,
        [0.0, 1., 0.0],
        [-GMS_SQRT, 0.0, 0.0],
        10,
    );
    let fov = [OmniDirectional::new(observer.clone())];

    // One day before the observation.
    let asteroid: State<Equatorial> = {
        let spk = LOADED_SPK.try_read().unwrap();
        spk.try_get_state_with_center(20000042, observer.epoch - 1.0, 10)
            .unwrap()
    };

    // beta = 0.5 removes half of the solar gravity, which over a day is a
    // deflection of order 1e-5 au.
    let dust = ParameterMask::all_fixed(NonGravKind::Dust(DustNonGrav), vec![0.5]).unwrap();

    let eph = SpiceEphemeris::loaded().unwrap();
    let states = [asteroid];
    let grav_only = check_visible(&eph, &fov, &states, &[], false).unwrap();
    let with_dust = check_visible(&eph, &fov, &states, &[Some(dust)], false).unwrap();

    let grav_pos = grav_only[0].2.states[0].pos;
    let dust_pos = with_dust[0].2.states[0].pos;
    assert!((grav_pos - dust_pos).norm() > 1e-6);

    // A non-empty non_gravs which does not cover every state is rejected.
    assert!(check_visible(&eph, &fov, &states, &[None, None], false).is_err());
}
