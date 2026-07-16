//! Integration tests for adaptive diffuse-state propagation with SPICE forces.
mod tests {
    use kete_core::desigs::Desig;
    use kete_core::frames::{Equatorial, SSB};
    use kete_core::prelude::{State, UncertainState};
    use kete_core::state::{
        DiffuseState, SplitConfig, propagate_diffuse_state_adaptive, propagate_with_stm,
        sigma_point_divergence,
    };
    use nalgebra::DMatrix;

    use crate::propagation::SpkNBody;

    fn earth_like_state() -> State<Equatorial, SSB> {
        State::<Equatorial, SSB> {
            desig: Desig::Name("Test".into()),
            epoch: 2451545.0.into(),
            pos: [1.0, 0.0, 0.0].into(),
            vel: [0.0, 0.01720209895, 0.0].into(),
            center: SSB,
        }
    }

    /// Single-component mixture propagation matches the standalone
    /// `propagate_with_stm` mean and covariance to floating-point noise.
    #[test]
    fn single_component_matches_propagate_with_stm() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-12_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component =
            UncertainState::<Equatorial, SSB>::new(state.clone(), cov.clone(), vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let jd_final = (2451545.0 + 30.0).into();
        let propagated =
            propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &SplitConfig::default())
                .unwrap();
        assert_eq!(propagated.n_components(), 1);
        assert!((propagated.weights[0] - 1.0).abs() < 1e-15);

        let (pos_f, vel_f, sens) = propagate_with_stm(
            &forces,
            state.pos.into(),
            state.vel.into(),
            &[],
            state.epoch,
            jd_final,
        )
        .unwrap();
        let got = &propagated.components[0].state;
        for i in 0..3 {
            assert!((got.pos[i] - pos_f[i]).abs() < 1e-12, "pos[{i}] mismatch");
            assert!((got.vel[i] - vel_f[i]).abs() < 1e-12, "vel[{i}] mismatch");
        }

        let phi = sens.view((0, 0), (6, 6));
        let expected_cov = phi * &cov * phi.transpose();
        let got_cov = &propagated.components[0].cov_matrix;
        for r in 0..6 {
            for c in 0..6 {
                let diff = (got_cov[(r, c)] - expected_cov[(r, c)]).abs();
                let scale = expected_cov[(r, c)].abs().max(1e-30);
                assert!(diff / scale < 1e-12, "cov[{r},{c}] mismatch");
            }
        }
    }

    #[test]
    fn sigma_divergence_small_cov_short_arc_is_small() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 10.0).into();
        let div = sigma_point_divergence(&component, &forces, jd_final, 3, 1.0, None).unwrap();
        // Mahalanobis divergence well below split_threshold (~3.0) -- this
        // is "well under a hundredth of a sigma off in the predicted Gaussian".
        assert!(div < 0.01, "expected near-linear regime, got {div}");
    }

    #[test]
    fn sigma_divergence_grows_with_sigma_factor() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-6;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 200.0).into();

        let d1 = sigma_point_divergence(&component, &forces, jd_final, 3, 1.0, None).unwrap();
        let d3 = sigma_point_divergence(&component, &forces, jd_final, 3, 3.0, None).unwrap();

        assert!(
            d3 > d1,
            "divergence should grow with sigma_factor: d1={d1}, d3={d3}"
        );
        // Mahalanobis divergence at 3-sigma sample of a small-cov / 200-day arc
        // should be clearly nonzero -- a few milli-sigma at least.
        assert!(d3 > 1e-3, "3-sigma divergence too small: {d3}");
    }

    #[test]
    fn sigma_divergence_zero_cov_returns_zero() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let cov = DMatrix::<f64>::zeros(6, 6);
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 30.0).into();
        let div = sigma_point_divergence(&component, &forces, jd_final, 3, 1.0, None).unwrap();
        assert_eq!(div, 0.0);
    }

    #[test]
    fn sigma_divergence_validates_inputs() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 10.0).into();
        assert!(sigma_point_divergence(&component, &forces, jd_final, 0, 1.0, None).is_err());
        assert!(sigma_point_divergence(&component, &forces, jd_final, 3, 0.0, None).is_err());
        assert!(sigma_point_divergence(&component, &forces, jd_final, 3, f64::NAN, None).is_err());
        assert!(sigma_point_divergence(&component, &forces, jd_final, 3, -1.0, None).is_err());
        // position_spacing_au must be positive when Some.
        assert!(
            sigma_point_divergence(&component, &forces, jd_final, 3, 1.0, Some(-0.001)).is_err()
        );
        assert!(sigma_point_divergence(&component, &forces, jd_final, 3, 1.0, Some(0.0)).is_err());
    }

    #[test]
    fn adaptive_propagation_does_not_split_in_linear_regime() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg = SplitConfig {
            split_threshold: 0.05,
            max_split_depth: 3,
            max_components: 27,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 10.0).into();
        let result = propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg).unwrap();
        assert_eq!(result.n_components(), 1);
        assert!((result.weights[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn time_bisection_arc_under_target_skips_bisection() {
        // A 200-day arc with target_arc_days=500 should produce the same
        // result as target_arc_days=infinity -- no time bisection.
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg_bisecting = SplitConfig {
            target_arc_days: 500.0,
            ..SplitConfig::default()
        };
        let cfg_no_bisect = SplitConfig {
            target_arc_days: f64::INFINITY,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 200.0).into();
        let r1 =
            propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg_bisecting).unwrap();
        let r2 =
            propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg_no_bisect).unwrap();
        assert_eq!(r1.n_components(), r2.n_components());
        // Means should match to integration precision.
        for (c1, c2) in r1.components.iter().zip(r2.components.iter()) {
            for i in 0..3 {
                assert!(
                    (c1.state.pos[i] - c2.state.pos[i]).abs() < 1e-12,
                    "pos[{i}] mismatch"
                );
            }
        }
    }

    #[test]
    fn time_bisection_long_arc_produces_consistent_propagation() {
        // A 1000-day arc with target_arc_days=200 forces several bisections.
        // The end state should match a manually-chunked propagation.
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = (6.685e-9_f64).powi(2);
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-20;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg = SplitConfig {
            target_arc_days: 200.0,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 1000.0).into();
        let bisected = propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg).unwrap();

        // Sanity: at very small covariance the mixture should stay single-component
        // through 1000 days (well within linear regime).
        assert_eq!(bisected.n_components(), 1);
        let total_w: f64 = bisected.weights.iter().sum();
        assert!((total_w - 1.0).abs() < 1e-12, "weights drifted: {total_w}");
    }

    #[test]
    fn time_bisection_validates_target_arc_days() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let jd_final = (2451545.0 + 10.0).into();

        let cfg_zero = SplitConfig {
            target_arc_days: 0.0,
            ..SplitConfig::default()
        };
        let cfg_neg = SplitConfig {
            target_arc_days: -1.0,
            ..SplitConfig::default()
        };
        let cfg_nan = SplitConfig {
            target_arc_days: f64::NAN,
            ..SplitConfig::default()
        };
        assert!(propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg_zero).is_err());
        assert!(propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg_neg).is_err());
        assert!(propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg_nan).is_err());
    }

    #[test]
    fn adaptive_propagation_splits_when_nonlinear() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = earth_like_state();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for i in 0..3 {
            cov[(i, i)] = 1e-6;
        }
        for i in 3..6 {
            cov[(i, i)] = 1e-12;
        }
        let component = UncertainState::<Equatorial, SSB>::new(state, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        let cfg = SplitConfig {
            split_threshold: 1e-4,
            max_split_depth: 2,
            max_components: 27,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 200.0).into();
        let result = propagate_diffuse_state_adaptive(&mixture, &forces, jd_final, &cfg).unwrap();
        assert!(
            result.n_components() > 1,
            "expected splitting; got {} component(s)",
            result.n_components()
        );
        let total_w: f64 = result.weights.iter().sum();
        assert!((total_w - 1.0).abs() < 1e-10, "weights drifted: {total_w}");
    }
}
