//! Visibility checks with body states from the loaded SPK files.

use kete_core::constants::C_AU_PER_DAY_INV;
use kete_core::constants::GMS_SQRT;
use kete_core::desigs::Desig;
use kete_core::forces::{DustNonGrav, NonGravKind, ParameterMask};
use kete_core::fov::{GenericRectangle, OmniDirectional};
use kete_core::fov::{check_ephemeris, check_n_body, check_two_body, check_visible};
use kete_core::frames::{Equatorial, SunCenter};
use kete_core::state::State;
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
        drop(eph);

        let vec = circular_back.pos - circular.pos;

        let fov = GenericRectangle::new(vec, 0.0001, 0.01, 0.01, circular.clone());
        let off_sun = {
            let spk = LOADED_SPK.try_read().unwrap();
            spk.try_to_sun(off_state.clone()).unwrap()
        };
        assert!(check_two_body(&fov, &off_sun).is_ok());
        assert!(
            check_n_body(
                &SpiceEphemeris::loaded().unwrap(),
                &fov,
                off_state.clone(),
                None,
                false
            )
            .is_ok()
        );

        let off_dyn: State<Equatorial> = off_state.into();
        assert!(
            check_visible(
                &SpiceEphemeris::loaded().unwrap(),
                &fov,
                &[off_dyn],
                &[],
                6.0,
                false
            )
            .unwrap()
            .first()
            .unwrap()
            .is_some()
        );
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
    let fov = GenericRectangle::new([1.0, 0.0, 0.0].into(), 0.0, 0.01, 0.01, observer);
    let seen = check_visible(
        &SpiceEphemeris::loaded().unwrap(),
        &fov,
        &[object_ssb],
        &[],
        3.0,
        false,
    )
    .unwrap();
    assert!(seen[0].is_some());
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

        // Check two body approximation calculation
        let asteroid_sun: State<_, SunCenter> = asteroid.clone().try_into().unwrap();
        let two_body = check_two_body(&fov, &asteroid_sun);
        assert!(two_body.is_ok());
        let (_, _, two_body) = two_body.unwrap();
        let dist = (two_body.pos - observer.pos).norm();
        assert!(((observer.epoch - two_body.epoch).elapsed - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
        let exact = spk
            .try_get_state_with_center(20000042, two_body.epoch, 10)
            .unwrap();
        // check that we are within about 150km - not bad for 2 body
        assert!((two_body.pos - exact.pos).norm() < 1e-6);

        // Check n body approximation calculation
        let asteroid_ssb = spk.try_to_ssb(asteroid.clone()).unwrap();
        let n_body = check_n_body(&eph, &fov, asteroid_ssb, None, false);
        assert!(n_body.is_ok());
        let (_, _, n_body) = n_body.unwrap();
        assert!(((observer.epoch - n_body.epoch).elapsed - dist * C_AU_PER_DAY_INV).abs() < 1e-6);
        let exact = spk
            .try_get_state_with_center(20000042, n_body.epoch, 10)
            .unwrap();
        // check that we are within about 150m
        assert!((n_body.pos - exact.pos).norm() < 1e-9);

        // Check spk queries
        let spk_check = &check_ephemeris(&eph, &fov, &[20000042])[0];
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

        assert!(
            check_visible(&eph, &fov, &[asteroid], &[], 6.0, false)
                .unwrap()
                .first()
                .unwrap()
                .is_some()
        );
    }

    // The Sun is co-located with itself in a Sun-centered FOV check
    let sun_fov = OmniDirectional::new(observer.clone());
    let sun_check = &check_ephemeris(&eph, &sun_fov, &[10])[0];
    assert!(sun_check.is_some());
    let sun_state = &sun_check.as_ref().unwrap().states[0];
    // The Sun is always at the solar center.
    assert!(sun_state.pos.norm() < 1e-12);
}

/// A non-gravitational model must change the observed state, including when the
/// state epoch is close enough to the observer that the two body check would
/// otherwise be used.
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
    let fov = OmniDirectional::new(observer.clone());

    // One day before the observation, well within the dt_limit used below.
    let asteroid: State<Equatorial> = {
        let spk = LOADED_SPK.try_read().unwrap();
        spk.try_get_state_with_center(20000042, observer.epoch - 1.0, 10)
            .unwrap()
    };

    // beta = 0.5 removes half of the solar gravity, which over a day is a
    // deflection of order 1e-5 au, far above the two body vs n-body difference.
    let dust = ParameterMask::all_fixed(NonGravKind::Dust(DustNonGrav), vec![0.5]).unwrap();

    let states = [asteroid];
    let grav_only = check_visible(
        &SpiceEphemeris::loaded().unwrap(),
        &fov,
        &states,
        &[],
        3.0,
        false,
    )
    .unwrap();
    let with_dust = check_visible(
        &SpiceEphemeris::loaded().unwrap(),
        &fov,
        &states,
        &[Some(dust)],
        3.0,
        false,
    )
    .unwrap();

    let grav_pos = grav_only[0].as_ref().unwrap().states[0].pos;
    let dust_pos = with_dust[0].as_ref().unwrap().states[0].pos;
    assert!((grav_pos - dust_pos).norm() > 1e-6);

    // A non-empty non_gravs which does not cover every state is rejected.
    assert!(
        check_visible(
            &SpiceEphemeris::loaded().unwrap(),
            &fov,
            &states,
            &[None, None],
            3.0,
            false
        )
        .is_err()
    );
}
