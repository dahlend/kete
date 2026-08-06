//! Integration tests for adaptive diffuse-state propagation with SPICE forces.
mod tests {
    use kete_core::desigs::Desig;
    use kete_core::frames::Equatorial;
    use kete_core::prelude::{KeteResult, State, UncertainState};
    use kete_core::time::{TDB, Time};
    use nalgebra::Vector3;

    use crate::spk::SpkCollection;
    use kete_core::state::{
        DiffuseState, SplitConfig, propagate_diffuse_state_adaptive, propagate_with_stm,
        sigma_point_divergence,
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
        // density-calibrated default (0.1) splits this covariance.
        let config = SplitConfig {
            split_threshold: 3.0,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 30.0).into();
        let propagated = propagate_diffuse_state_adaptive(
            &mixture,
            &forces,
            jd_final,
            &config,
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

    #[test]
    fn sigma_divergence_small_cov_short_arc_is_small() {
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
        let jd_final = (2451545.0 + 10.0).into();
        let div = sigma_point_divergence(
            &component,
            &forces,
            jd_final,
            3,
            1.0,
            None,
            &sun_resolver(&spk),
        )
        .unwrap();
        // Mahalanobis divergence well below split_threshold (~3.0) -- this
        // is "well under a hundredth of a sigma off in the predicted Gaussian".
        assert!(div < 0.01, "expected near-linear regime, got {div}");
    }

    #[test]
    fn sigma_divergence_grows_with_sigma_factor() {
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
        let jd_final = (2451545.0 + 200.0).into();

        let d1 = sigma_point_divergence(
            &component,
            &forces,
            jd_final,
            3,
            1.0,
            None,
            &sun_resolver(&spk),
        )
        .unwrap();
        let d3 = sigma_point_divergence(
            &component,
            &forces,
            jd_final,
            3,
            3.0,
            None,
            &sun_resolver(&spk),
        )
        .unwrap();

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

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::zeros(6, 6);
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 30.0).into();
        let div = sigma_point_divergence(
            &component,
            &forces,
            jd_final,
            3,
            1.0,
            None,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(div, 0.0);
    }

    #[test]
    fn sigma_divergence_validates_inputs() {
        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);

        let state = sun_centered_state();
        let cov = DMatrix::<f64>::identity(6, 6) * 1e-12;
        let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();
        let jd_final = (2451545.0 + 10.0).into();
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                0,
                1.0,
                None,
                &sun_resolver(&spk)
            )
            .is_err()
        );
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                3,
                0.0,
                None,
                &sun_resolver(&spk)
            )
            .is_err()
        );
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                3,
                f64::NAN,
                None,
                &sun_resolver(&spk)
            )
            .is_err()
        );
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                3,
                -1.0,
                None,
                &sun_resolver(&spk)
            )
            .is_err()
        );
        // position_spacing_au must be positive when Some.
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                3,
                1.0,
                Some(-0.001),
                &sun_resolver(&spk)
            )
            .is_err()
        );
        assert!(
            sigma_point_divergence(
                &component,
                &forces,
                jd_final,
                3,
                1.0,
                Some(0.0),
                &sun_resolver(&spk)
            )
            .is_err()
        );
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
            max_split_depth: 3,
            max_components: 27,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 10.0).into();
        let result = propagate_diffuse_state_adaptive(
            &mixture,
            &forces,
            jd_final,
            &cfg,
            &sun_resolver(&spk),
        )
        .unwrap();
        assert_eq!(result.n_components(), 1);
        assert!((result.weights[0] - 1.0).abs() < 1e-15);
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
            max_split_depth: 2,
            max_components: 27,
            ..SplitConfig::default()
        };
        let jd_final = (2451545.0 + 200.0).into();
        let result = propagate_diffuse_state_adaptive(
            &mixture,
            &forces,
            jd_final,
            &cfg,
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
    }

    /// Prints where the sigma-point divergence actually comes from, probe by
    /// probe: which eigendirections of the propagated covariance carry the
    /// Mahalanobis mass, how wide those directions are relative to the widest,
    /// and how the residual splits into its odd (STM-error, first order in the
    /// probe) and even (curvature, second order) parts.
    ///
    /// The scenario is a dust-release patch on an eccentric comet orbit: a
    /// near-delta release position crossed with an anisotropic velocity patch,
    /// scanned over the release-position sigma. This is the configuration
    /// where the reported divergence was observed to scale as `1/pos_sigma`
    /// while the propagated density stayed accurate, and where splitting did
    /// not reduce the reported value.
    #[test]
    fn sigma_divergence_decomposition_by_eigendirection() {
        use kete_core::elements::EquinoctialElements;
        use kete_core::state::propagate_with_diagnosis;
        use nalgebra::{DVector, SymmetricEigen, Vector6};

        crate::test_data::ensure_test_spk();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let forces = SpkNBody::new(&spk, false);
        let resolver = sun_resolver(&spk);

        // Eccentric comet at perihelion: q = 1 AU, e = 0.9.
        let k = 0.01720209895_f64;
        let v_peri = k * (2.0_f64 - 0.1).sqrt();
        let state = State::<Equatorial>::new(
            Desig::Name("decomp".into()),
            2451545.0,
            [1.0, 0.0, 0.0],
            [0.0, v_peri, 0.0],
            10,
        );

        // Anisotropic velocity patch along the tangential direction.
        let d = Vector3::new(0.0, 1.0, 0.0);
        let (sigma_t, sigma_r) = (5e-6_f64, 2.3e-6_f64);
        let jd_final: Time<TDB> = (2451545.0 + 20.0).into();

        for pos_sigma in [1e-9_f64, 3e-8, 1e-6] {
            let mut cov = DMatrix::<f64>::zeros(6, 6);
            for i in 0..3 {
                cov[(i, i)] = pos_sigma * pos_sigma;
            }
            for r in 0..3 {
                for c in 0..3 {
                    let transverse = f64::from(u8::from(r == c)) - d[r] * d[c];
                    cov[(3 + r, 3 + c)] =
                        sigma_t.powi(2) * transverse + sigma_r.powi(2) * d[r] * d[c];
                }
            }
            let component = UncertainState::from_state(&state, &cov, vec![]).unwrap();

            let diag = propagate_with_diagnosis(
                &component,
                &forces,
                jd_final,
                3,
                1.0,
                Some(0.001),
                &resolver,
            )
            .unwrap();
            println!(
                "\npos_sigma = {pos_sigma:.0e}: reported divergence {:.3}",
                diag.divergence
            );

            // Rebuild the edge probes and decompose each residual in the
            // propagated covariance's eigenbasis.  `eps` mirrors the
            // internal metric's floor (trace * 1e-6); the unfloored
            // decomposition is printed alongside to show what the floor is
            // suppressing - the same absolute residual measured against a
            // near-null width.
            let p_f = &diag.propagated.cov_matrix;
            let f_eig = SymmetricEigen::new(p_f.clone());
            let lam_max = f_eig.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
            let eps = p_f.trace() * 1e-6;
            let eps_raw = p_f.trace() * 1e-12;

            let s_eig = SymmetricEigen::new(component.cov_matrix.clone());
            let mut order: Vec<usize> = (0..6).collect();
            order.sort_by(|&a, &b| {
                s_eig.eigenvalues[b]
                    .partial_cmp(&s_eig.eigenvalues[a])
                    .unwrap()
            });

            for &axis in order.iter().take(3) {
                let lambda = s_eig.eigenvalues[axis];
                if lambda <= 0.0 {
                    continue;
                }
                let v = s_eig.eigenvectors.column(axis).clone_owned();
                let mut residuals = Vec::new();
                for sign in [1.0_f64, -1.0] {
                    let delta = &v * (sign * lambda.sqrt());
                    let step = Vector6::from_iterator(delta.iter().copied());
                    let perturbed = component.elements.displaced_by(&step);
                    let start: State<Equatorial> = perturbed.try_to_state().unwrap().into_frame();
                    let (off_p, off_v) = resolver(component.elements.epoch).unwrap();
                    let (pos_f, vel_f, _sens) = propagate_with_stm(
                        &forces,
                        Vector3::from(start.pos) + off_p,
                        Vector3::from(start.vel) + off_v,
                        &[],
                        component.elements.epoch,
                        jd_final,
                    )
                    .unwrap();
                    let (foff_p, foff_v) = resolver(jd_final).unwrap();
                    let fstate = State::<Equatorial>::new(
                        Desig::Name("p".into()),
                        jd_final.jd,
                        pos_f - foff_p,
                        vel_f - foff_v,
                        10,
                    );
                    let fel = EquinoctialElements::from_state(&fstate.into_frame()).unwrap();
                    let nonlin = diag.propagated.elements.offset_to(&fel);
                    let lin = &diag.augmented_stm * &delta;
                    let diff = DVector::from_iterator(6, (0..6).map(|i| nonlin[i] - lin[i]));
                    residuals.push(diff);
                }
                let odd = (&residuals[0] - &residuals[1]) * 0.5;
                let even = (&residuals[0] + &residuals[1]) * 0.5;

                for (label, diff) in [("odd/STM", &odd), ("even/curv", &even)] {
                    // Mahalanobis mass by propagated-covariance eigendirection.
                    let mut contrib: Vec<(f64, f64, f64, f64)> = (0..6)
                        .map(|j| {
                            let u = f_eig.eigenvectors.column(j);
                            let proj = u.dot(diff);
                            let lam_j = f_eig.eigenvalues[j].max(0.0) + eps;
                            let lam_raw = f_eig.eigenvalues[j].max(0.0) + eps_raw;
                            (
                                proj * proj / lam_j,
                                (lam_raw / lam_max).sqrt(),
                                proj.abs(),
                                proj * proj / lam_raw,
                            )
                        })
                        .collect();
                    contrib.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
                    let total: f64 = contrib.iter().map(|c| c.0).sum();
                    let total_raw: f64 = contrib.iter().map(|c| c.3).sum();
                    let (_c0, w0, p0, r0) = contrib[0];
                    println!(
                        "  axis {axis} {label:9}: mahal floored {:6.3} / raw {:6.3} | \
                         raw top dir: {:4.0}% of raw mass, width {:8.1e} x widest, \
                         |resid| {:8.1e}",
                        total.sqrt(),
                        total_raw.sqrt(),
                        100.0 * r0 / total_raw.max(1e-300),
                        w0,
                        p0,
                    );
                }
            }
        }
    }
}
