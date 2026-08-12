//! Integration tests for adaptive diffuse-state propagation with SPICE forces.
mod tests {
    use kete_core::desigs::Desig;
    use kete_core::frames::Equatorial;
    use kete_core::prelude::{KeteResult, State, UncertainState};
    use kete_core::time::{TDB, Time};
    use nalgebra::Vector3;

    use crate::spk::SpkCollection;
    use kete_core::state::{
        DEFAULT_STEP_DAYS, DiffuseState, SplitConfig, Termination, propagate_diffuse_state,
        propagate_with_stm, step_diffuse_state,
    };
    use nalgebra::DMatrix;

    use crate::propagation::SpkNBody;

    /// A one AU circular orbit, referred to the Sun.
    ///
    /// Elements are defined about a gravitating body, so this is centered on NAIF 10 rather
    /// than on the barycenter - the barycenter has no body at it and no `mu`, and asking
    /// for elements about it is now an error rather than a silent substitution of the
    /// Sun's mass about the wrong focus.
    fn sun_centered_state() -> State<Equatorial> {
        State::<Equatorial>::new(
            Desig::Name("Test".into()),
            2451545.0,
            [1.0, 0.0, 0.0],
            [0.0, 0.01720209895, 0.0],
            10,
        )
    }

    /// Resolves the Sun against the barycenter, which is what the force models use.
    ///
    /// The element center and the integration center are different bodies, so a state has
    /// to cross between them at every epoch the propagation touches. The adaptive path
    /// reaches many intermediate epochs while splitting, hence a resolver rather than a
    /// pair of endpoint states.
    fn sun_resolver(
        spk: &SpkCollection,
    ) -> impl Fn(Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> + Sync + '_ {
        move |time| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        }
    }

    /// Single-component mixture propagation matches the standalone
    /// `propagate_with_stm` mean and covariance to floating-point noise.
    #[test]
    fn single_component_matches_propagate_with_stm() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-12_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let component_check = component.clone();
        let mixture = DiffuseState::from_uncertain(component);

        // The state-estimation threshold, pinned explicitly: this test
        // compares the single-component path against the STM path, so no
        // split may occur or the comparison stops being one-to-one.  The
        // density-calibrated default (0.15) splits this covariance.
        let config = SplitConfig {
            split_threshold: 3.0,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 30.0).into();
        let (propagated, _) = propagate_diffuse_state(
            &mixture,
            &forces,
            jd_final,
            &config,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(propagated.n_components(), 1);
        assert!((propagated.weights[0] - 1.0).abs() < 1e-15);

        // The reference: the same problem run entirely through the old cartesian path.
        // The element-native result must reproduce it, which is the check that a swapped
        // Jacobian direction or a transposed multiply anywhere in the element
        // covariance update would fail.
        let sun_0 = spk
            .try_get_state_with_center::<Equatorial>(10, state.epoch, 0)
            .unwrap();
        // The reference starts from the *element path's* initial condition, not the
        // original cartesian one. Reconstructing a state from elements is exact only to
        // rounding, and thirty days of integration amplifies that to about 4e-12 - which
        // is the round trip's cost, certified separately by the element suite, and not
        // what this test is about. Comparing
        // against the original state would measure that instead of the covariance path.
        let elem_start = component_check.state::<Equatorial>().unwrap();
        let (pos_f, vel_f, sens) = propagate_with_stm(
            &forces,
            Vector3::from(elem_start.pos) + Vector3::from(sun_0.pos),
            Vector3::from(elem_start.vel) + Vector3::from(sun_0.vel),
            &[],
            state.epoch,
            jd_final,
        )
        .unwrap();

        // Mean: the propagated elements, carried back out to barycentric coordinates.
        let got = propagated.component(0).unwrap();
        let sun_f = spk
            .try_get_state_with_center::<Equatorial>(10, jd_final, 0)
            .unwrap();
        let got_state = got.state::<Equatorial>().unwrap();
        let got_pos = Vector3::from(got_state.pos) + Vector3::from(sun_f.pos);
        let got_vel = Vector3::from(got_state.vel) + Vector3::from(sun_f.vel);
        let pos_err = (got_pos - pos_f).norm() / pos_f.norm();
        let vel_err = (got_vel - vel_f).norm() / vel_f.norm();
        println!("element vs cartesian mean: pos {pos_err:e}, vel {vel_err:e} relative");
        assert!(pos_err < 1e-12, "position mismatch {pos_err:e}");
        assert!(vel_err < 1e-12, "velocity mismatch {vel_err:e}");

        // Covariance: the element covariance converted back to cartesian must reproduce
        // `Phi P Phi^T`. The tolerance is the element round trip's, not the
        // propagation's -
        // the covariance crosses into element coordinates on the way in and back out here,
        // and each crossing costs the conditioning of the element Jacobian.
        let expected_cov = sens.view((0, 0), (6, 6)) * &cov * sens.view((0, 0), (6, 6)).transpose();
        let got_cov = got.cartesian_covariance::<Equatorial>().unwrap();
        let scale = expected_cov.norm();
        let relative = (got_cov.view((0, 0), (6, 6)) - &expected_cov).norm() / scale;
        println!("element vs cartesian covariance path: {relative:e} relative");
        assert!(
            relative < 1e-8,
            "covariance mismatch against the cartesian path: {relative:e}"
        );
    }

    /// A threshold out of reach and a budget of one: the leg reports the nonlinearity it
    /// found instead of splitting to remove it.
    fn measure_only() -> SplitConfig {
        SplitConfig {
            split_threshold: 1e30,
            max_components: 1,
        }
    }

    /// A compact covariance over a short arc reads as linear, and the residual behind
    /// that reading sits at the propagator's own resolution rather than above it.
    #[test]
    fn leg_nonlinearity_small_cov_short_arc_is_small() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let jd_final = (2451545.0 + 10.0).into();
        let (stepped, report) = step_diffuse_state(
            &mixture,
            &forces,
            jd_final,
            &measure_only(),
            &sun_resolver(&spk),
        )
        .unwrap();
        let eta = stepped.max_eta().unwrap();
        // Far below any usable split_threshold -- "under a hundredth of a sigma off in
        // the predicted position distribution".
        assert!(eta < 0.01, "expected the near-linear regime, got {eta}");
        // A few meters over ten days on a one AU circular orbit is the propagator's own
        // resolution, which is what this reading has to be to mean anything.
        let residual = stepped.residual_meters().unwrap();
        assert!(
            residual < 100.0,
            "residual {residual} m is far above the propagator resolution"
        );
        // The leg hands back a component carrying its own probes and numbers, which is
        // what makes the propagated mixture marchable rather than just readable.
        assert!(stepped.components[0].has_probes());
        assert_eq!(stepped.components[0].eta, Some(eta));
        assert_eq!(report.seeded, 1, "a fresh mixture places its own probes");
    }

    /// A covariance with no extent carries no representation error, and no split could
    /// change that. Every direction is skipped rather than probed at an invented scale.
    #[test]
    fn leg_nonlinearity_zero_cov_returns_zero() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::zeros(6, 6);
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let jd_final = (2451545.0 + 30.0).into();
        let (stepped, report) = step_diffuse_state(
            &mixture,
            &forces,
            jd_final,
            &measure_only(),
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(stepped.max_eta(), Some(0.0));
        assert_eq!(report.termination, Termination::Converged);
    }

    #[test]
    fn adaptive_propagation_does_not_split_in_linear_regime() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg = SplitConfig {
            split_threshold: 0.05,
            max_components: 27,
        };
        let jd_final = (2451545.0 + 10.0).into();
        let (result, report) = propagate_diffuse_state(
            &mixture,
            &forces,
            jd_final,
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(result.n_components(), 1);
        assert!((result.weights[0] - 1.0).abs() < 1e-15);

        // A mixture that never needed a split reports convergence, and the nonlinearity
        // it is carrying sits on the component rather than in a parallel array.
        assert_eq!(report.termination, Termination::Converged);
        assert_eq!(report.seeded, 1, "the input component had no probes yet");
        assert!(result.components[0].eta.is_some());
        assert!(result.max_eta().unwrap() <= cfg.split_threshold);
        assert_eq!(result.weight_above_eta(cfg.split_threshold), Some(0.0));
    }

    /// `eta` describes the arc rather than the last leg of it, so cutting the same arc
    /// into more legs does not change it.
    ///
    /// The probes are carried and differenced against the accumulated transition matrix,
    /// both of which compose exactly, so the leg length drops out. This is what lets
    /// `split_threshold` mean one thing at every `step_days`. Probes re-placed at the start
    /// of each leg would instead report the increment one leg adds, which falls with the
    /// step and hides nonlinearity that arrives slowly.
    #[test]
    fn eta_does_not_depend_on_the_leg_length() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        // About a thousand km of position spread, far enough above the propagator's own
        // resolution that what is being compared is curvature rather than numerical noise.
        for i in 0..3 {
            cov[(i, i)] = (6.685e-6_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-18;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let jd_final = (2451545.0 + 200.0).into();

        let mut seen = Vec::new();
        for step_days in [200.0, 90.0, 45.0, 20.0] {
            let cfg = SplitConfig {
                // High enough that nothing splits: what is under test is the number, not
                // the controller's response to it.
                split_threshold: 1e6,
                max_components: 1,
            };
            let (result, _) = propagate_diffuse_state(
                &mixture,
                &forces,
                jd_final,
                &cfg,
                step_days,
                &sun_resolver(&spk),
            )
            .unwrap();
            assert_eq!(result.n_components(), 1);
            let eta = result.max_eta().unwrap();
            println!("step {step_days:6.1} d   eta {eta:e}");
            seen.push(eta);
        }

        let smallest = seen.iter().copied().fold(f64::INFINITY, f64::min);
        let largest = seen.iter().copied().fold(0.0_f64, f64::max);
        assert!(smallest > 0.0, "no nonlinearity was measured at all");
        let spread = largest / smallest - 1.0;
        // The residual left over is the integrator restarting on different leg boundaries,
        // not a dependence on the step.
        assert!(
            spread < 1e-3,
            "eta moved by {spread:e} across leg lengths of 200, 90, 45 and 20 days: {seen:?}"
        );
    }

    /// Marching by hand is the same measurement as one call over the whole arc.
    ///
    /// This is the property the carried probes exist for. The components hold their own
    /// probes, so a caller stepping leg by leg accumulates exactly what the internal fold
    /// accumulates - same splits, same `eta` - and nothing resets between calls. Probes
    /// held by the propagation call instead of by the components would restart the
    /// measurement at every step, and a hand-driven march would silently under-split.
    #[test]
    fn a_manual_march_matches_a_single_call() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);
        let resolver = sun_resolver(&spk);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        // Wide enough that the arc splits: a march that agrees only because nothing
        // happened would prove nothing.
        for i in 0..3 {
            cov[(i, i)] = 1e-6;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let cfg = SplitConfig {
            split_threshold: 1e-4,
            max_components: 27,
        };

        // One call, cut into legs at 20, 110 and 200 days by a 90 day step.
        let (single, single_report) = propagate_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 200.0).into(),
            &cfg,
            90.0,
            &resolver,
        )
        .unwrap();

        // The same grid, driven by the caller.
        let mut marched = mixture.clone();
        let mut last = None;
        let mut seeded_total = 0;
        for offset in [20.0, 110.0, 200.0] {
            let (next, report) = step_diffuse_state(
                &marched,
                &forces,
                (2451545.0 + offset).into(),
                &cfg,
                &resolver,
            )
            .unwrap();
            seeded_total += report.seeded;
            marched = next;
            last = Some(report);
        }
        let last = last.unwrap();

        println!(
            "single call {} components eta {:e}; manual march {} components eta {:e}, {} seeded",
            single.n_components(),
            single.max_eta().unwrap(),
            marched.n_components(),
            marched.max_eta().unwrap(),
            seeded_total
        );
        assert!(
            single.n_components() > 1,
            "the arc has to split or this proves nothing"
        );
        assert_eq!(single.n_components(), marched.n_components());
        assert_eq!(single_report.termination, last.termination);
        let (a, b) = (single.max_eta().unwrap(), marched.max_eta().unwrap());
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()),
            "hand-driven march read {b:e} against {a:e} in one call"
        );
    }

    /// A component rebuilt mid-march loses its carried probes, and the step says so
    /// rather than leaving the reset to be inferred.
    ///
    /// This is the one way a hand-driven march can quietly restart its own measurement:
    /// take a mixture apart, put it back together, and the probes do not survive the trip.
    #[test]
    fn rebuilding_a_component_is_reported_as_a_reseed() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);
        let resolver = sun_resolver(&spk);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-6_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-18;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let cfg = SplitConfig {
            split_threshold: 1e6,
            max_components: 1,
        };

        let (first, first_report) = step_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 90.0).into(),
            &cfg,
            &resolver,
        )
        .unwrap();
        assert_eq!(first_report.seeded, 1);

        // Continue normally: nothing is seeded, and the measurement accumulates.
        let (carried, carried_report) =
            step_diffuse_state(&first, &forces, (2451545.0 + 180.0).into(), &cfg, &resolver)
                .unwrap();
        assert_eq!(carried_report.seeded, 0);

        // Continue from a rebuilt component instead. `UncertainState::new` produces a
        // state with no probes, so this restarts the measurement - and reports it.
        let stripped = UncertainState::new(
            first.components[0].elements.clone(),
            first.components[0].cov_matrix.clone(),
            first.components[0].free_params.clone(),
        )
        .unwrap();
        let (restarted, restart_report) = step_diffuse_state(
            &DiffuseState::from_uncertain(stripped),
            &forces,
            (2451545.0 + 180.0).into(),
            &cfg,
            &resolver,
        )
        .unwrap();
        assert_eq!(restart_report.seeded, 1);

        // The restarted march reads a smaller departure, because it only measures the
        // last leg. That is exactly the silent under-splitting the seeded count exposes.
        let (accumulated, one_leg) = (carried.max_eta().unwrap(), restarted.max_eta().unwrap());
        println!("carried probes {accumulated:e}, reseeded {one_leg:e}");
        assert!(
            one_leg < accumulated,
            "a reseeded march should read less than a carried one: {one_leg:e} vs {accumulated:e}"
        );
    }

    /// Probes belong to the component they were seeded from, and a mismatched pairing is
    /// an error rather than a number measured against an anchor that no longer exists.
    #[test]
    fn probes_from_another_epoch_are_rejected() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);
        let resolver = sun_resolver(&spk);

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-16;
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let cfg = SplitConfig::default();

        let (stepped, _) = step_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 90.0).into(),
            &cfg,
            &resolver,
        )
        .unwrap();

        // Move the epoch out from under the carried probes. This is the failure a
        // caller can actually produce: the probes are the controller's own, but the
        // elements are public and nothing stops one being edited while the other is
        // kept.
        let mut stale = stepped.components[0].clone();
        stale.elements.epoch = (2451545.0 + 45.0).into();
        let broken = DiffuseState::from_uncertain(stale);
        assert!(
            step_diffuse_state(
                &broken,
                &forces,
                (2451545.0 + 180.0).into(),
                &cfg,
                &resolver
            )
            .is_err()
        );
    }

    /// A mixture that has never been marched reports nothing, which is a different
    /// statement from reporting zero.
    #[test]
    fn a_hand_built_mixture_has_no_eta() {
        let state = sun_centered_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-16;
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        assert!(!mixture.components[0].has_probes());
        assert!(mixture.max_eta().is_none());
        assert!(mixture.residual_meters().is_none());
        assert!(mixture.weight_above_eta(0.1).is_none());
    }

    /// Propagating backwards lands on the epoch asked for rather than accumulating a leg
    /// at a time towards it, and reads the same nonlinearity as the forward arc.
    #[test]
    fn adaptive_propagation_runs_backwards() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let cfg = SplitConfig::default();

        let (back, _) = propagate_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 - 200.0).into(),
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert!((back.epoch().jd - (2451545.0 - 200.0)).abs() < 1e-9);

        let (forward, _) = propagate_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 200.0).into(),
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        // A quiet circular orbit is symmetric in time to well inside an order of
        // magnitude; this is checking the grid carries its sign, not the dynamics.
        let (a, b) = (back.max_eta().unwrap(), forward.max_eta().unwrap());
        assert!(
            a > 0.0 && b > 0.0 && (a / b) > 0.1 && (a / b) < 10.0,
            "backward {a} and forward {b} nonlinearity disagree by more than 10x"
        );
    }

    /// An arc of no length is no work, and says nothing it did not measure.
    #[test]
    fn a_zero_length_arc_changes_nothing() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-16;
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let (out, report) = propagate_diffuse_state(
            &mixture,
            &forces,
            2451545.0.into(),
            &SplitConfig::default(),
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(out.n_components(), 1);
        assert_eq!(report.seeded, 0);
        assert!(
            out.max_eta().is_none(),
            "no leg ran, so there is no measurement to report"
        );
    }

    /// A target epoch that is not a number is an error, not a mixture returned unchanged
    /// at the epoch it started from.
    #[test]
    fn a_non_finite_target_is_rejected() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-16;
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        assert!(
            propagate_diffuse_state(
                &mixture,
                &forces,
                f64::NAN.into(),
                &SplitConfig::default(),
                DEFAULT_STEP_DAYS,
                &sun_resolver(&spk),
            )
            .is_err()
        );
    }

    /// A component that has never split is whitened against itself, so nothing about the
    /// reference is visible until a split happens.
    ///
    /// This is what keeps the single-Gaussian behavior, and every number measured on it,
    /// exactly what it was before the reference existed.
    #[test]
    fn an_unsplit_component_is_whitened_against_itself() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-6_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-18;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        // Two legs, so the reference has been carried as well as seeded.
        let cfg = SplitConfig {
            split_threshold: 1e6,
            max_components: 1,
        };
        let (out, _) = propagate_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 180.0).into(),
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();

        let got = &out.components[0];
        let reference = got.whitening_cov.as_ref().unwrap();
        let relative = (reference - &got.cov_matrix).norm() / got.cov_matrix.norm();
        println!("unsplit reference vs own covariance: {relative:e} relative");
        assert!(
            relative < 1e-12,
            "an unsplit component's reference drifted from its own covariance: {relative:e}"
        );
    }

    /// A split child keeps measuring against the width its parent had.
    ///
    /// Whitening by the child's own covariance would shrink the denominator by exactly the
    /// amount the split shrank the component, so a split that removed curvature would
    /// barely move `eta` and the threshold would mean something different at every depth.
    /// The check is that the reference is the parent's rather than the child's, and that
    /// the difference is in the direction that matters: the reference is wider, so the
    /// number the controller thresholds on is smaller than the self-whitened one.
    #[test]
    fn split_children_are_whitened_against_the_parent() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-6;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let cfg = SplitConfig {
            split_threshold: 1e-4,
            max_components: 27,
        };

        let (out, _) = propagate_diffuse_state(
            &mixture,
            &forces,
            (2451545.0 + 200.0).into(),
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert!(
            out.n_components() > 1,
            "the arc has to split or this proves nothing"
        );

        for (index, child) in out.components.iter().enumerate() {
            let reference = child.whitening_cov.as_ref().unwrap();
            // The split halves the variance along one direction, so the two matrices must
            // differ - a child whitened against itself would fail here.
            let separation = (reference - &child.cov_matrix).norm() / child.cov_matrix.norm();
            assert!(
                separation > 1e-3,
                "component {index} is whitened against its own covariance: {separation:e}"
            );
            // And differ in the direction that makes the threshold depth-invariant: the
            // reference is the wider, pre-split distribution.
            let widened = reference.clone() - &child.cov_matrix;
            let smallest = widened
                .symmetric_eigenvalues()
                .iter()
                .fold(f64::INFINITY, |acc, v| acc.min(*v));
            println!(
                "component {index}: reference/own {separation:e}, \
                 smallest eigenvalue of the difference {smallest:e}"
            );
            assert!(
                smallest > -1e-12 * reference.norm(),
                "component {index} carries a reference narrower than its own covariance"
            );
        }
    }

    #[test]
    fn adaptive_propagation_splits_when_nonlinear() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-6;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg = SplitConfig {
            split_threshold: 1e-4,
            max_components: 27,
        };
        let jd_final = (2451545.0 + 200.0).into();
        let (result, report) = propagate_diffuse_state(
            &mixture,
            &forces,
            jd_final,
            &cfg,
            DEFAULT_STEP_DAYS,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert!(
            result.n_components() > 1,
            "expected splitting; got {} component(s)",
            result.n_components()
        );
        let total_w: f64 = result.weights.iter().sum();
        assert!((total_w - 1.0).abs() < 1e-10, "weights drifted: {total_w}");
        // The report describes the final leg, which is not where the splitting happened,
        // so what is checked here is the state: every component that came back has been
        // measured and carries the probes that measured it.
        println!(
            "{} components, final leg {:?}, {} seeded",
            result.n_components(),
            report.termination,
            report.seeded
        );
        assert!(
            result
                .components
                .iter()
                .all(|c| c.eta.is_some() && c.has_probes())
        );
    }
}
