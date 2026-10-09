// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! `NBody` with body states from the loaded SPK files.

use kete_core::desigs::Desig;
use kete_core::errors::Error;
use kete_core::forces::{GravParams, ParameterizedForce};
use kete_core::frames::{Equatorial, FrameId, SSB, Vector};
use kete_core::propagation::NBody;
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use kete_spice::ephemeris::SpiceEphemeris;
use kete_spice::spk::SpkCollection;
use nalgebra::{Matrix3, Vector3};

/// Asteroid 42 (in the test SPK) as a polyhedron body: a 50 km box, oriented as
/// given, with the polyhedron field used within 0.01 AU.
fn polyhedron_42(orientation: kete_core::forces::Orientation) -> GravParams {
    use kete_core::forces::{Polyhedron, Shape};
    use kete_core::geometry::TriMesh;
    let size = 50.0 / 1.495_978_707e8;
    let v = (0..8_u32)
        .map(|i| {
            Vector3::new(
                if i & 1 == 0 { -1.3 } else { 1.1 },
                if (i >> 1) & 1 == 0 { -0.7 } else { 0.9 },
                if (i >> 2) & 1 == 0 { -0.45 } else { 0.35 },
            ) * size
        })
        .collect();
    let f = vec![
        [0, 2, 1],
        [1, 2, 3],
        [4, 5, 6],
        [5, 7, 6],
        [0, 1, 4],
        [1, 5, 4],
        [2, 6, 3],
        [3, 6, 7],
        [0, 4, 2],
        [2, 4, 6],
        [1, 3, 5],
        [3, 7, 5],
    ];
    let gm = 1e-15;
    let model = Polyhedron::new(TriMesh::new(v, f).unwrap(), gm).unwrap();
    let mut body = GravParams::new(20_000_042, gm, (100.0 / 1.495_978_707e8) as f32);
    body.shape = Shape::Polyhedron {
        model: std::sync::Arc::new(model),
        orientation,
        switch_radius: 0.01,
    };
    body
}

/// Near a polyhedron body, `NBody` adds exactly the rotated polyhedron field to
/// the other bodies' gravity, through all three evaluation paths.
#[test]
fn n_body_polyhedron_body() {
    use kete_core::forces::Orientation;
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let rot = *nalgebra::Rotation3::from_euler_angles(0.4, -0.9, 1.7).matrix();
    let body = polyhedron_42(Orientation::Fixed(rot));
    let base = NBody::new(&eph, false);
    let mut with_body = NBody::new(&eph, false);
    with_body.massive_obj.push(body.clone());

    let body_pos = Vector3::from(
        spk.try_get_state_with_center::<Equatorial>(20_000_042, time, 0)
            .unwrap()
            .pos,
    );
    let offset = Vector3::new(90.0, -40.0, 60.0) / 1.495_978_707e8;
    let pos = Vector::<Equatorial>::new((body_pos + offset).into());
    let vel = Vector::<Equatorial>::new([0.001, -0.004, 0.0005]);

    let a_base: Vector3<f64> = base
        .accel(time, &pos, &vel, &[], &mut Default::default(), false)
        .unwrap()
        .into();
    let a_body: Vector3<f64> = with_body
        .accel(time, &pos, &vel, &[], &mut Default::default(), false)
        .unwrap()
        .into();
    let kete_core::forces::Shape::Polyhedron { model, .. } = &body.shape else {
        unreachable!("built as a polyhedron")
    };
    let expect = rot * model.field(&(rot.transpose() * offset)).0;
    assert!(
        ((a_body - a_base) - expect).norm() < 1e-9 * expect.norm(),
        "accel adds the rotated field: {:?} vs {expect:?}",
        a_body - a_base
    );

    let (dr, dv) = with_body
        .jacobians(time, &pos, &vel, &[], &mut Default::default())
        .unwrap();
    let (a_all, dr_all, dv_all, _) = with_body
        .accel_and_jacobians(time, &pos, &vel, &[], &mut Default::default(), false)
        .unwrap();
    assert_eq!(Vector3::from(a_all), a_body, "combined path: accel");
    assert_eq!((dr_all, dv_all), (dr, dv), "combined path: jacobians");

    // The body's term dominates da/dr at 100 km. The object is ~3 AU from the SSB,
    // so a step much below ~0.1 km is lost to rounding of the position; at 0.2 km
    // both rounding and truncation are ~1e-6 relative. The exact comparison of the
    // polyhedron jacobian is in `kete_core::forces::gravity`; this checks wiring.
    let h = 0.2 / 1.495_978_707e8;
    for axis in 0..3 {
        let step = Vector3::ith(axis, h);
        let p1 = Vector::<Equatorial>::new((body_pos + offset + step).into());
        let p0 = Vector::<Equatorial>::new((body_pos + offset - step).into());
        let col = (Vector3::from(
            with_body
                .accel(time, &p1, &vel, &[], &mut Default::default(), false)
                .unwrap(),
        ) - Vector3::from(
            with_body
                .accel(time, &p0, &vel, &[], &mut Default::default(), false)
                .unwrap(),
        )) / (2.0 * h);
        assert!(
            (dr.column(axis) - col).norm() < 1e-4 * dr.norm(),
            "da/dr column {axis}: {}",
            (dr.column(axis) - col).norm() / dr.norm()
        );
    }
}

/// A CK-oriented polyhedron body with no CK loaded for its frame is an error
/// inside the switch radius, and costs nothing (no CK lookup) outside it.
#[test]
fn n_body_polyhedron_ck_without_kernel() {
    use kete_core::forces::Orientation;
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let mut force = NBody::new(&eph, false);
    force.massive_obj.push(polyhedron_42(Orientation::Frame {
        frame_id: FrameId(-987_654_000),
    }));
    let body_pos = Vector3::from(
        spk.try_get_state_with_center::<Equatorial>(20_000_042, time, 0)
            .unwrap()
            .pos,
    );
    let vel = Vector::<Equatorial>::new([0.001, -0.004, 0.0005]);
    let near = Vector::<Equatorial>::new((body_pos + Vector3::new(1e-6, 0.0, 0.0)).into());
    let far = Vector::<Equatorial>::new((body_pos + Vector3::new(0.5, 0.0, 0.0)).into());
    assert!(
        force
            .accel(time, &near, &vel, &[], &mut Default::default(), false)
            .is_err(),
        "inside: error"
    );
    assert!(
        force
            .accel(time, &far, &vel, &[], &mut Default::default(), false)
            .is_ok(),
        "outside: point mass"
    );
}

/// Analytical `jacobian_pos` matches FD of `accel` to FD precision.
#[test]
fn n_body_jacobian_pos_matches_finite_difference() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let force = NBody::new(&eph, false);
    let time = Time::<TDB>::new(2_451_545.0);
    let pos = Vector::<Equatorial>::new([1.5, 0.5, 0.1]);
    let vel = Vector::<Equatorial>::new([-0.008, 0.012, 0.001]);

    let (analytical, _) = force
        .jacobians(time, &pos, &vel, &[], &mut Default::default())
        .unwrap();

    let pos_v: Vector3<f64> = pos.into();
    let h = pos_v.norm() * 1e-6;
    let mut fd = Matrix3::<f64>::zeros();
    for j in 0..3 {
        let mut p_plus = pos_v;
        p_plus[j] += h;
        let mut p_minus = pos_v;
        p_minus[j] -= h;
        let a_plus: Vector3<f64> = force
            .accel(
                time,
                &Vector::<Equatorial>::new(p_plus.into()),
                &vel,
                &[],
                &mut Default::default(),
                false,
            )
            .unwrap()
            .into();
        let a_minus: Vector3<f64> = force
            .accel(
                time,
                &Vector::<Equatorial>::new(p_minus.into()),
                &vel,
                &[],
                &mut Default::default(),
                false,
            )
            .unwrap()
            .into();
        let col = (a_plus - a_minus) / (2.0 * h);
        for i in 0..3 {
            fd[(i, j)] = col[i];
        }
    }
    let max_err = (analytical - fd).abs().max();
    let analytical_scale = analytical.abs().max();
    assert!(
        max_err < 1e-6 * analytical_scale.max(1e-6),
        "max_err = {max_err}, analytical_scale = {analytical_scale}"
    );
}

/// SSB-relative test point and the same point relative to the Sun.
fn ssb_and_sun_relative(
    spk: &SpkCollection,
    time: Time<TDB>,
) -> (
    (Vector<Equatorial>, Vector<Equatorial>),
    (Vector<Equatorial>, Vector<Equatorial>),
) {
    let pos = Vector3::new(0.5, 1.0, 0.1);
    let vel = Vector3::new(-0.012, 0.008, 0.001);
    let sun = spk
        .try_get_state_with_center::<Equatorial>(10, time, 0)
        .unwrap();
    (
        (pos.into(), vel.into()),
        (
            (pos - Vector3::from(sun.pos)).into(),
            (vel - Vector3::from(sun.vel)).into(),
        ),
    )
}

/// The non-grav contribution is the non-grav force evaluated on the Sun-relative
/// state: the model with dust minus the model without equals `DustNonGrav` called
/// directly, for the acceleration and for both jacobians.
#[test]
fn non_grav_is_evaluated_on_sun_relative_state() {
    use kete_core::forces::DustNonGrav;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let beta = 0.001;
    let ((pos, vel), (pos_sun, vel_sun)) = ssb_and_sun_relative(spk, time);

    let gravity = NBody::new(&eph, false);
    let with_dust = NBody::with_non_grav(&eph, false, Some(DustNonGrav));

    let a_grav: Vector3<f64> = gravity
        .accel(time, &pos, &vel, &[], &mut Default::default(), false)
        .unwrap()
        .into();
    let a_both: Vector3<f64> = with_dust
        .accel(time, &pos, &vel, &[beta], &mut Default::default(), false)
        .unwrap()
        .into();
    let a_dust: Vector3<f64> = DustNonGrav
        .accel(
            time,
            &pos_sun,
            &vel_sun,
            &[beta],
            &mut Default::default(),
            false,
        )
        .unwrap()
        .into();
    // The difference is taken against the full solar acceleration, so it is good to
    // the rounding of that sum, not of the dust term.
    let err = ((a_both - a_grav) - a_dust).norm();
    println!(
        "non-grav accel difference {err:e} against gravity {:e}",
        a_grav.norm()
    );
    assert!(
        err < 1e-15 * a_grav.norm().max(1.0),
        "accel difference {err:e}"
    );

    let (g_dr, g_dv) = gravity
        .jacobians(time, &pos, &vel, &[], &mut Default::default())
        .unwrap();
    let (b_dr, b_dv) = with_dust
        .jacobians(time, &pos, &vel, &[beta], &mut Default::default())
        .unwrap();
    let (d_dr, d_dv) = DustNonGrav
        .jacobians(time, &pos_sun, &vel_sun, &[beta], &mut Default::default())
        .unwrap();
    let err_dr = ((b_dr - g_dr) - d_dr).abs().max();
    let err_dv = ((b_dv - g_dv) - d_dv).abs().max();
    println!("non-grav jacobian differences da/dr {err_dr:e}, da/dv {err_dv:e}");
    assert!(err_dr < 1e-15 * g_dr.abs().max().max(1.0));
    assert!(err_dv < 1e-15);

    let dp = with_dust
        .parameter_jacobian(time, &pos, &vel, &[beta], &mut Default::default())
        .unwrap();
    let dp_direct = DustNonGrav
        .parameter_jacobian(time, &pos_sun, &vel_sun, &[beta], &mut Default::default())
        .unwrap();
    assert_eq!(dp, dp_direct);
}

/// The single-pass evaluation returns exactly what the three separate methods do.
#[test]
fn accel_and_jacobians_matches_separate_calls() {
    use kete_core::forces::JplCometNonGrav;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let params = [1.0e-8, 2.0e-9, -3.0e-10];
    let ((pos, vel), _) = ssb_and_sun_relative(spk, time);

    let force = NBody::with_non_grav(&eph, true, Some(JplCometNonGrav::standard_comet()));
    let (accel, da_dr, da_dv, da_dp) = force
        .accel_and_jacobians(time, &pos, &vel, &params, &mut Default::default(), false)
        .unwrap();
    assert_eq!(
        accel,
        force
            .accel(time, &pos, &vel, &params, &mut Default::default(), false)
            .unwrap()
    );
    assert_eq!(
        (da_dr, da_dv),
        force
            .jacobians(time, &pos, &vel, &params, &mut Default::default())
            .unwrap()
    );
    assert_eq!(
        da_dp,
        force
            .parameter_jacobian(time, &pos, &vel, &params, &mut Default::default())
            .unwrap()
    );

    let gravity = NBody::new(&eph, true);
    let (accel, da_dr, da_dv, da_dp) = gravity
        .accel_and_jacobians(time, &pos, &vel, &[], &mut Default::default(), false)
        .unwrap();
    assert_eq!(
        accel,
        gravity
            .accel(time, &pos, &vel, &[], &mut Default::default(), false)
            .unwrap()
    );
    assert_eq!(
        (da_dr, da_dv),
        gravity
            .jacobians(time, &pos, &vel, &[], &mut Default::default())
            .unwrap()
    );
    assert_eq!(da_dp.ncols(), 0);
}

/// Evaluations sharing one `meta` across an integration return exactly what fresh ones
/// do, both when the cache is filled and when it is read back, and a new time is not
/// served stale states. `jacobians` reads the same cache.
#[test]
fn shared_meta_matches_fresh_meta() {
    use kete_core::forces::JplCometNonGrav;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let later = Time::<TDB>::new(2_451_545.5);
    let params = [1.0e-8, 2.0e-9, -3.0e-10];
    let ((pos, vel), _) = ssb_and_sun_relative(spk, time);

    let force = NBody::with_non_grav(&eph, true, Some(JplCometNonGrav::standard_comet()));
    let mut meta = Default::default();
    for t in [time, time, later, time] {
        assert_eq!(
            force
                .accel(t, &pos, &vel, &params, &mut meta, true)
                .unwrap(),
            force
                .accel(t, &pos, &vel, &params, &mut Default::default(), false)
                .unwrap()
        );
        assert_eq!(
            force
                .accel_and_jacobians(t, &pos, &vel, &params, &mut meta, false)
                .unwrap(),
            force
                .accel_and_jacobians(t, &pos, &vel, &params, &mut Default::default(), false)
                .unwrap()
        );
        assert_eq!(
            force.jacobians(t, &pos, &vel, &params, &mut meta).unwrap(),
            force
                .jacobians(t, &pos, &vel, &params, &mut Default::default())
                .unwrap()
        );
    }
}

/// A state aimed at the Earth's center ends in an impact with the Earth, and a
/// state that misses it does not.
#[test]
fn an_impact_is_reported() {
    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let earth = spk
        .try_get_state_with_center::<Equatorial>(399, time, 0)
        .unwrap();
    let earth_pos = Vector3::from(earth.pos);
    let earth_vel = Vector3::from(earth.vel);
    // 0.001 AU from the Earth, closing at 0.01 AU/day.
    let offset = Vector3::new(1e-3, 0.0, 0.0);
    let aimed: State<Equatorial, SSB> = State::<Equatorial>::new(
        Desig::Empty,
        time,
        earth_pos + offset,
        earth_vel - 10.0 * offset,
        0,
    )
    .try_into()
    .unwrap();
    let result = aimed.propagate_with(&NBody::new(&eph, false), time + 1.0);
    assert!(
        matches!(result, Err(Error::Impact(399, _))),
        "expected an Earth impact, got {result:?}"
    );

    // The same approach displaced sideways by 0.01 AU passes the Earth.
    let miss: State<Equatorial, SSB> = State::<Equatorial>::new(
        Desig::Empty,
        time,
        earth_pos + offset + Vector3::new(0.0, 1e-2, 0.0),
        earth_vel - 10.0 * offset,
        0,
    )
    .try_into()
    .unwrap();
    assert!(
        miss.propagate_with(&NBody::new(&eph, false), time + 1.0)
            .is_ok()
    );
}

/// Free parameters, names, and lower bounds of the model are those of the non-grav.
#[test]
fn free_parameters_are_those_of_the_non_grav() {
    use kete_core::forces::DustNonGrav;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let gravity = NBody::new(&eph, false);
    assert_eq!(gravity.n_free_params(), 0);
    assert!(gravity.lower_bounds().is_empty());

    let force = NBody::with_non_grav(&eph, false, Some(DustNonGrav));
    assert_eq!(force.n_free_params(), 1);
    assert_eq!(force.free_param_names(), vec!["beta"]);
    assert_eq!(force.lower_bounds(), vec![Some(0.0)]);
}

/// A non-grav cannot be evaluated without the Sun in the body list; that is an error
/// rather than a silently dropped force.
#[test]
fn non_grav_without_the_sun_is_an_error() {
    use kete_core::forces::DustNonGrav;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let spk = eph.spk();
    let time = Time::<TDB>::new(2_451_545.0);
    let ((pos, vel), _) = ssb_and_sun_relative(spk, time);

    let mut force = NBody::with_non_grav(&eph, false, Some(DustNonGrav));
    force.massive_obj.retain(|body| body.naif_id != 10);
    assert!(
        force
            .accel(time, &pos, &vel, &[0.001], &mut Default::default(), false)
            .is_err()
    );
    assert!(
        force
            .jacobians(time, &pos, &vel, &[0.001], &mut Default::default())
            .is_err()
    );
    assert!(
        force
            .accel_and_jacobians(time, &pos, &vel, &[0.001], &mut Default::default(), false)
            .is_err()
    );
}

/// Variational propagation with a three-parameter non-grav gives a 6 x 9 sensitivity.
#[test]
fn jpl_comet_non_grav_propagates_with_sensitivities() {
    use kete_core::forces::JplCometNonGrav;
    use kete_core::state::propagate_with_stm;

    kete_spice::test_data::ensure_test_spk();
    let eph = SpiceEphemeris::loaded().unwrap();
    let force = NBody::with_non_grav(&eph, false, Some(JplCometNonGrav::standard_comet()));
    let (pos_f, vel_f, sens_f) = propagate_with_stm(
        &force,
        Vector3::new(0.5, 1.0, 0.1),
        Vector3::new(-0.012, 0.008, 0.001),
        &[1.0e-8, 2.0e-9, -3.0e-10],
        Time::<TDB>::new(2_451_545.0),
        Time::<TDB>::new(2_451_545.0 + 5.0),
    )
    .unwrap();
    assert!(pos_f.iter().all(|v| v.is_finite()));
    assert!(vel_f.iter().all(|v| v.is_finite()));
    assert_eq!((sens_f.nrows(), sens_f.ncols()), (6, 9));
}
