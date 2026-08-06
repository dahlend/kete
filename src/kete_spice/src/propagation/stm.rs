//! State Transition matrix computation
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use kete_core::forces::{FrozenForce, ParameterizedForce, Sum};
use kete_core::frames::{Equatorial, SSB, SunCenter};
use kete_core::prelude::{KeteResult, State};
use kete_core::state::propagate_with_stm;
use kete_core::time::{TDB, Time};

use super::recenter::Recenter;
use super::spk_n_body::SpkNBody;
use crate::spk::LOADED_SPK;
use nalgebra::DMatrix;

/// Compute the state transition matrix and optional parameter sensitivities using the
/// Radau 15th-order integrator with full N-body physics.
///
/// The input state must be typed as `State<Equatorial, SSB>`, enforcing at compile
/// time that the center is the solar system barycenter.  The returned state is also
/// SSB-centered.
///
/// When `include_asteroids` is `true`, the force model includes asteroid
/// masses from `GravParams::selected_masses()`; otherwise only the
/// planets and Moon from `GravParams::planets()` are used.
///
/// When `non_grav` is `Some`, the STM gains parameter-sensitivity columns
/// for the frozen force's parameters.  The frozen values serve as the
/// nominal trajectory; the all-`None` variational mask is constructed
/// internally so the integrator computes `d(r_f, v_f) / d p_k`.
///
/// Returns the propagated [`State`] and a 6x(6+N) sensitivity matrix where N is
/// the number of free non-gravitational parameters (0 for none, 1 for `Dust`, 3 for
/// `JplComet`). Column ordering is:
///
/// ```text
/// cols 0-5  : 6x6 state transition matrix  d(r_f, v_f) / d(r_0, v_0)
/// col  6+k  : parameter sensitivity        d(r_f, v_f) / dp_k
/// ```
///
/// # Errors
/// Returns an error if SPK queries fail or integration does not converge.
pub fn compute_state_transition<F>(
    state: &State<Equatorial, SSB>,
    jd: Time<TDB>,
    include_extended: bool,
    non_grav: Option<&FrozenForce<F>>,
) -> KeteResult<(State<Equatorial, SSB>, DMatrix<f64>)>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter> + Clone,
{
    // Gravity-only path stays bare (`SpkNBody` directly).
    //
    // Non-grav path: build an all-`None` variational mask from the frozen
    // template so `propagate_with_stm` gains parameter-sensitivity columns,
    // then pass the frozen values as the nominal `free_params` slice.
    let spk = LOADED_SPK.try_read()?;
    let (pos_f, vel_f, sens) = match non_grav {
        None => propagate_with_stm(
            &SpkNBody::new(&spk, include_extended),
            state.pos.into(),
            state.vel.into(),
            &[],
            state.epoch,
            jd,
        )?,
        Some(frozen) => {
            let force = Sum::new(
                SpkNBody::new(&spk, include_extended),
                Recenter::<SSB, _>::new(&spk, frozen.inner.clone()),
            );
            propagate_with_stm(
                &force,
                state.pos.into(),
                state.vel.into(),
                &frozen.values,
                state.epoch,
                jd,
            )?
        }
    };

    let final_state = State {
        desig: state.desig.clone(),
        epoch: jd,
        pos: pos_f.into(),
        vel: vel_f.into(),
        center: SSB,
    };

    Ok((final_state, sens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::desigs::Desig;
    use kete_core::elements::{CometElements, EquinoctialElements};
    use kete_core::forces::JplCometNonGrav;
    use kete_core::frames::Ecliptic;

    use kete_core::prelude::UncertainState;
    use kete_core::state::{
        DiffuseState, SplitConfig, minimum_components_for_divergence,
        propagate_diffuse_state_adaptive, propagate_elements_with_sensitivity, propagate_state,
        propagate_with_diagnosis, sigma_point_divergence, split_for_propagation,
    };
    use nalgebra::{Matrix6, Vector3, Vector6};

    /// Heliocentric elements of a real main-belt asteroid, the Sun's barycentric state at
    /// the same epoch, and the epoch itself.
    ///
    /// The elements are referred to the Sun while the force model is referred to the
    /// solar system barycenter, which is the combination this composition exists to serve
    /// and the one the two-body tests in `kete_core` cannot exercise.
    fn setup() -> (EquinoctialElements, State<Equatorial, SSB>, Time<TDB>) {
        crate::test_data::ensure_test_spk();
        let epoch = Time::<TDB>::new(2_451_545.0);
        let spk = LOADED_SPK.try_read().unwrap();

        // 42 Isis, a genuine main-belt orbit rather than a constructed one.
        let helio = spk
            .try_get_state_with_center::<Equatorial>(20_000_042, epoch, 10)
            .unwrap();
        let sun = spk
            .try_get_state_with_center::<Equatorial>(10, epoch, 0)
            .unwrap();

        let state = State::<Equatorial>::new(Desig::Empty, epoch, helio.pos, helio.vel, 10);
        let elem = EquinoctialElements::from_state(&state.into_frame()).unwrap();
        let sun_ssb = State::<Equatorial, SSB> {
            desig: Desig::Empty,
            epoch,
            pos: sun.pos,
            vel: sun.vel,
            center: SSB,
        };
        (elem, sun_ssb, epoch)
    }

    /// Central differences of the element sensitivity taken through the real N-body
    /// propagation: planetary gravity from SPK, GR and J2, over a 400 day arc.
    ///
    /// This is the certification that matters. The `kete_core` tests establish the chain
    /// rule against a two-body force, which cannot exercise the perturbing bodies, the
    /// analytic `SpkNBody` jacobians, or the Sun-to-barycenter offset.
    #[test]
    fn element_sensitivity_matches_finite_difference_under_n_body() {
        const TOL: f64 = 1e-5;

        let (elem, sun_ssb, epoch) = setup();
        let epoch_final = Time::<TDB>::new(epoch.jd + 400.0);
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);

        let (_, _, sens) =
            propagate_elements_with_sensitivity(&force, &elem, &sun_ssb, &[], epoch_final).unwrap();
        assert_eq!(sens.ncols(), 6);

        let offset_pos = Vector3::from(sun_ssb.pos);
        let offset_vel = Vector3::from(sun_ssb.vel);
        let propagate = |elements: &EquinoctialElements| {
            // The sensitivity's rows are in the force model's frame, so the reference
            // states must be rotated into it. This is a rotation of a *state* and is
            // linear and exact; the Jacobian's own rotation is not, and lives inside
            // `EquinoctialElements::state_jacobian` rather than at any call site.
            let s: State<Equatorial> = elements.try_to_state().unwrap().into_frame();
            propagate_with_stm(
                &force,
                Vector3::from(s.pos) + offset_pos,
                Vector3::from(s.vel) + offset_vel,
                &[],
                epoch,
                epoch_final,
            )
            .unwrap()
        };

        let helio: State<Equatorial> = elem.try_to_state().unwrap().into_frame();
        let pos: Vector3<f64> = helio.pos.into();
        let vel: Vector3<f64> = helio.vel.into();

        // Step scales follow the coordinates' units, as in the elements suite: the
        // semi-latus rectum column by its own size, and the shape and phase columns by the
        // orbit equation `p / r` that sits in their denominators.
        let orbit_eq = elem.semi_latus / elem.epoch_distance();
        let steps = [
            1e-5 * elem.semi_latus,
            1e-5 * orbit_eq,
            1e-5 * orbit_eq,
            1e-5,
            1e-5,
            1e-5 * orbit_eq,
        ];

        let mut worst = 0.0_f64;
        let mut worst_col = 0;
        for (col, step) in steps.iter().copied().enumerate() {
            let mut delta = Vector6::zeros();
            delta[col] = step;
            let (p_pos, p_vel, _) = propagate(&elem.displaced_by(&delta));
            delta[col] = -step;
            let (m_pos, m_vel, _) = propagate(&elem.displaced_by(&delta));

            let mut fd = Vector6::zeros();
            for row in 0..3 {
                fd[row] = (p_pos[row] - m_pos[row]) / (2.0 * step);
                fd[row + 3] = (p_vel[row] - m_vel[row]) / (2.0 * step);
            }
            let analytic = sens.column(col);
            let rel = (fd - analytic).norm() / analytic.norm();
            if rel > worst {
                worst = rel;
                worst_col = col;
            }
        }

        // The same measurement on the cartesian sensitivity, same object and same arc.
        // The composition is `Phi . K` with `K` exact to rounding, so the two must be the
        // same size; that is the statement that the change of coordinates costs nothing,
        // and it holds the test to the variational STM rather than to a guessed constant.
        let cart_sens = propagate(&elem).2;
        let base = [
            pos[0] + offset_pos[0],
            pos[1] + offset_pos[1],
            pos[2] + offset_pos[2],
            vel[0] + offset_vel[0],
            vel[1] + offset_vel[1],
            vel[2] + offset_vel[2],
        ];
        let mut worst_cart = 0.0_f64;
        for col in 0..6 {
            let step = 1e-6 * if col < 3 { pos.norm() } else { vel.norm() };
            let mut fd = Vector6::zeros();
            for (sign, scale) in [(1.0, 1.0), (-1.0, -1.0)] {
                let mut shifted = base;
                shifted[col] += sign * step;
                let (p, v, _) = propagate_with_stm(
                    &force,
                    Vector3::new(shifted[0], shifted[1], shifted[2]),
                    Vector3::new(shifted[3], shifted[4], shifted[5]),
                    &[],
                    epoch,
                    epoch_final,
                )
                .unwrap();
                for row in 0..3 {
                    fd[row] += scale * p[row] / (2.0 * step);
                    fd[row + 3] += scale * v[row] / (2.0 * step);
                }
            }
            let analytic = cart_sens.column(col);
            worst_cart = worst_cart.max((fd - analytic).norm() / analytic.norm());
        }

        println!("N-body, 400 d: element {worst:e} (col {worst_col}), cartesian {worst_cart:e}");
        assert!(
            worst < TOL,
            "element sensitivity {worst:e} exceeded {TOL:e}"
        );
        assert!(
            worst < 10.0 * worst_cart.max(1e-9),
            "element sensitivity {worst:e} is worse than the cartesian STM it composes, \
             {worst_cart:e}"
        );
    }

    /// The offset between centers is time dependent, so one evaluated at the wrong epoch
    /// is wrong by the central body's own motion over the difference. For the Sun against
    /// the barycenter that is small enough that nothing downstream would notice, so it is
    /// rejected rather than trusted.
    #[test]
    fn element_sensitivity_rejects_center_state_at_wrong_epoch() {
        let (elem, mut sun_ssb, epoch) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);

        sun_ssb.epoch = Time::<TDB>::new(epoch.jd + 100.0);
        let result = propagate_elements_with_sensitivity(
            &force,
            &elem,
            &sun_ssb,
            &[],
            Time::<TDB>::new(epoch.jd + 400.0),
        );
        assert!(result.is_err());
    }

    /// The same, with non-gravitational forces active, so the parameter columns are
    /// non-empty and the composed matrix is the full `6 x (6 + 3)` design block.
    #[test]
    fn element_sensitivity_under_n_body_with_non_gravs() {
        const TOL: f64 = 1e-5;

        let (elem, sun_ssb, epoch) = setup();
        let epoch_final = Time::<TDB>::new(epoch.jd + 200.0);
        let spk = LOADED_SPK.try_read().unwrap();
        let force = Sum::new(
            SpkNBody::new(&spk, false),
            Recenter::<SSB, _>::new(&spk, JplCometNonGrav::standard_comet()),
        );
        let params = [2.0e-9, 5.0e-10, -1.0e-10];

        let (_, _, sens) =
            propagate_elements_with_sensitivity(&force, &elem, &sun_ssb, &params, epoch_final)
                .unwrap();
        assert_eq!(sens.ncols(), 9);

        let offset_pos = Vector3::from(sun_ssb.pos);
        let offset_vel = Vector3::from(sun_ssb.vel);
        let propagate = |elements: &EquinoctialElements| {
            // The sensitivity's rows are in the force model's frame, so the reference
            // states must be rotated into it. This is a rotation of a *state* and is
            // linear and exact; the Jacobian's own rotation is not, and lives inside
            // `EquinoctialElements::state_jacobian` rather than at any call site.
            let s: State<Equatorial> = elements.try_to_state().unwrap().into_frame();
            propagate_with_stm(
                &force,
                Vector3::from(s.pos) + offset_pos,
                Vector3::from(s.vel) + offset_vel,
                &params,
                epoch,
                epoch_final,
            )
            .unwrap()
        };

        // Step scales follow the coordinates' units, as in the elements suite: the
        // semi-latus rectum column by its own size, and the shape and phase columns by the
        // orbit equation `p / r` that sits in their denominators.
        let orbit_eq = elem.semi_latus / elem.epoch_distance();
        let steps = [
            1e-5 * elem.semi_latus,
            1e-5 * orbit_eq,
            1e-5 * orbit_eq,
            1e-5,
            1e-5,
            1e-5 * orbit_eq,
        ];

        let mut worst = 0.0_f64;
        let mut worst_col = 0;
        for (col, step) in steps.iter().copied().enumerate() {
            let mut delta = Vector6::zeros();
            delta[col] = step;
            let (p_pos, p_vel, _) = propagate(&elem.displaced_by(&delta));
            delta[col] = -step;
            let (m_pos, m_vel, _) = propagate(&elem.displaced_by(&delta));

            let mut fd = Vector6::zeros();
            for row in 0..3 {
                fd[row] = (p_pos[row] - m_pos[row]) / (2.0 * step);
                fd[row + 3] = (p_vel[row] - m_vel[row]) / (2.0 * step);
            }
            let analytic = sens.column(col);
            let rel = (fd - analytic).norm() / analytic.norm();
            if rel > worst {
                worst = rel;
                worst_col = col;
            }
        }
        println!("N-body + non-gravs, 200 d: element {worst:e} (col {worst_col})");
        assert!(
            worst < TOL,
            "element sensitivity {worst:e} exceeded {TOL:e}"
        );

        // Parameter columns are the cartesian ones untouched: free parameters of the
        // force model are not elements.
        let (_, _, cart_sens) = propagate(&elem);
        for col in 6..9 {
            let diff = (sens.column(col) - cart_sens.column(col)).norm();
            assert!(diff == 0.0, "parameter column {col} changed by {diff:e}");
        }
    }

    /// A well observed orbit's uncertainty, in element coordinates: shape and orientation
    /// to parts in `1e8`, and along-track timing to fifteen minutes.
    ///
    /// The phase coordinate is a true longitude in **radians**, so the timing figure is
    /// converted through the orbit's own true-longitude rate `dL/dt = sqrt(GM p) / r^2`
    /// rather than carried across as a number of days. Using `TIMING_DAYS` directly would
    /// give a distribution wider by the reciprocal of that rate and would say nothing about
    /// a well observed object.
    fn well_observed_sigma(elem: &EquinoctialElements) -> Vector6<f64> {
        const TIMING_DAYS: f64 = 1.04e-2;
        let rate = elem.gm_sqrt * elem.semi_latus.sqrt() / elem.epoch_distance().powi(2);
        Vector6::from_column_slice(&[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, TIMING_DAYS * rate])
    }

    /// Osculating equinoctial elements of a heliocentric Equatorial state.
    fn equinoctial_at(
        epoch: Time<TDB>,
        pos: Vector3<f64>,
        vel: Vector3<f64>,
    ) -> EquinoctialElements {
        let helio = State::<Equatorial>::new(
            Desig::Empty,
            epoch,
            [pos.x, pos.y, pos.z],
            [vel.x, vel.y, vel.z],
            10,
        );
        EquinoctialElements::from_state(&helio.into_frame()).expect("the test orbits are ordinary")
    }

    /// Shared driver for the linearity horizon measurement.
    ///
    /// Clones are placed at `k` sigma along each element axis, propagated through the true
    /// nonlinear flow, and compared against what the linear map predicts in each basis:
    ///
    /// ```text
    /// cartesian:    x_f  ~  x_nom + Phi        . dx_0
    /// element:      q_f  ~  q_nom + J_f Phi K  . q_0
    /// ```
    ///
    /// The residual is reported in units of the initial standard deviation, by mapping it
    /// back through the same transition matrix and whitening with the initial covariance.
    /// That makes the two bases directly comparable and the number directly readable: it
    /// is how many initial sigma the linear model is wrong by.
    ///
    /// Both transition matrices come from the same cartesian variational integration, so
    /// this measures the coordinates and nothing else. The element side uses osculating
    /// elements at the output time. Elements are stored in the Ecliptic and the
    /// propagation runs in the Equatorial, and the Jacobian is asked for in the frame it
    /// is composed with rather than rotated here.
    fn linearity_horizon(
        force: &SpkNBody<'_>,
        spk: &crate::spk::SpkCollection,
        elem: &EquinoctialElements,
        sun_pos: Vector3<f64>,
        sun_vel: Vector3<f64>,
        arcs: &[f64],
        sigma: &Vector6<f64>,
    ) {
        let epoch = elem.epoch;
        let nominal: State<Equatorial> = elem.try_to_state().unwrap().into_frame();
        let pos_0 = Vector3::from(nominal.pos) + sun_pos;
        let vel_0 = Vector3::from(nominal.vel) + sun_vel;

        let jac = elem.state_jacobian::<Equatorial>().unwrap();
        let covariance_0 = jac * Matrix6::from_diagonal(&sigma.map(|s| s * s)) * jac.transpose();
        let chol_0 = covariance_0
            .cholesky()
            .expect("the initial covariance is positive definite");

        println!(
            "{:>8}  {:>7}  {:>5}  {:>13}  {:>13}  {:>8}  {:>7}",
            "arc (d)", "orbits", "k", "cartesian", "element", "ratio", "off-domain"
        );
        let period = elem.orbital_period();

        for &arc in arcs {
            let epoch_final = Time::<TDB>::new(epoch.jd + arc);
            let (pos_f, vel_f, phi_full) =
                propagate_with_stm(force, pos_0, vel_0, &[], epoch, epoch_final).unwrap();
            let phi: Matrix6<f64> = phi_full.fixed_view::<6, 6>(0, 0).into();
            let sens = phi * jac;

            let sun_f = spk
                .try_get_state_with_center::<Equatorial>(10, epoch_final, 0)
                .unwrap();
            let sun_f_pos = Vector3::from(sun_f.pos);
            let sun_f_vel = Vector3::from(sun_f.vel);
            let elements_at = |pos: Vector3<f64>, vel: Vector3<f64>| {
                let helio = State::<Equatorial>::new(
                    Desig::Empty,
                    epoch_final,
                    pos - sun_f_pos,
                    vel - sun_f_vel,
                    10,
                );
                EquinoctialElements::from_state(&helio.into_frame())
            };

            let Ok(elem_f) = elements_at(pos_f, vel_f) else {
                println!("{arc:>8.0}  nominal orbit has no valid elements");
                continue;
            };
            let Ok(inverse) = elem_f.state_jacobian_inverse::<Equatorial>() else {
                println!("{arc:>8.0}  nominal orbit left the element domain entirely");
                continue;
            };
            let phi_elem = inverse * sens;
            let phi_elem_lu = phi_elem.lu();
            let phi_lu = phi.lu();

            for k in [1.0_f64, 100.0] {
                let (mut worst_cart, mut worst_elem) = (0.0_f64, 0.0_f64);
                let mut off_domain = 0_usize;
                for axis in 0..6 {
                    for sign in [1.0_f64, -1.0] {
                        let mut delta_0 = Vector6::zeros();
                        delta_0[axis] = sign * k * sigma[axis];

                        let clone = elem.displaced_by(&delta_0);
                        let Ok(clone_state) = clone.try_to_state() else {
                            off_domain += 1;
                            continue;
                        };
                        let clone_state: State<Equatorial> = clone_state.into_frame();
                        let clone_pos = Vector3::from(clone_state.pos) + sun_pos;
                        let clone_vel = Vector3::from(clone_state.vel) + sun_vel;
                        let (truth_pos, truth_vel) =
                            propagate_state(force, clone_pos, clone_vel, &[], epoch, epoch_final)
                                .unwrap();

                        // Cartesian. The exact initial offset is used, so the element
                        // displacement's own curvature is not charged against this basis.
                        let state_offset = Vector6::from_column_slice(&[
                            clone_pos[0] - pos_0[0],
                            clone_pos[1] - pos_0[1],
                            clone_pos[2] - pos_0[2],
                            clone_vel[0] - vel_0[0],
                            clone_vel[1] - vel_0[1],
                            clone_vel[2] - vel_0[2],
                        ]);
                        let truth = Vector6::from_column_slice(&[
                            truth_pos[0] - pos_f[0],
                            truth_pos[1] - pos_f[1],
                            truth_pos[2] - pos_f[2],
                            truth_vel[0] - vel_f[0],
                            truth_vel[1] - vel_f[1],
                            truth_vel[2] - vel_f[2],
                        ]);
                        let residual = truth - phi * state_offset;
                        let mapped = phi_lu.solve(&residual).expect("the STM is invertible");
                        // Whitening is a solve against the Cholesky *factor*, not against
                        // the covariance: `||L^-1 m||^2 = m^T P^-1 m` is the Mahalanobis
                        // norm, while `||P^-1 m||` is not a norm of anything.
                        let whitened = chol_0
                            .l()
                            .solve_lower_triangular(&mapped)
                            .expect("the Cholesky factor is nonsingular");
                        worst_cart = worst_cart.max(whitened.norm());

                        // Element, in osculating coordinates at the output time.
                        let Ok(truth_elements) = elements_at(truth_pos, truth_vel) else {
                            off_domain += 1;
                            continue;
                        };
                        let residual = elem_f.offset_to(&truth_elements) - phi_elem * delta_0;
                        let mapped = phi_elem_lu
                            .solve(&residual)
                            .expect("the element STM is invertible");
                        let whitened: Vector6<f64> = Vector6::from_iterator(
                            mapped
                                .iter()
                                .zip(sigma.iter())
                                .map(|(value, scale)| value / scale),
                        );
                        worst_elem = worst_elem.max(whitened.norm());
                    }
                }
                println!(
                    "{arc:>8.0}  {:>7.2}  {k:>5.0}  {worst_cart:>13.3e}  {worst_elem:>13.3e}  \
                     {:>8.1}  {off_domain:>7}",
                    arc / period,
                    worst_cart / worst_elem.max(f64::MIN_POSITIVE)
                );
            }
        }
    }

    /// **The measurement the whole element-space effort exists to justify**, on a quiet
    /// main-belt orbit. Does a distribution stay linear longer in element coordinates?
    ///
    /// This is the Gaussianity horizon, element space against cartesian, scored against a
    /// clone ensemble under the real N-body force model. It needs no element-space
    /// propagator: both transition matrices come from one cartesian variational
    /// integration.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates a clone ensemble"]
    fn element_vs_cartesian_linearity_horizon() {
        let (elem, sun_ssb, _) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);

        // A well observed orbit: shape and orientation known to parts in 1e8, timing to
        // about fifteen minutes.
        let sigma = well_observed_sigma(&elem);

        println!(
            "42 Isis, main belt, period {:.0} d. Linearization error in initial sigma.",
            elem.orbital_period()
        );
        println!(
            "  phase sigma {:.3e} rad = {:.3e} AU along track at r = {:.3} AU",
            sigma[5],
            sigma[5] * elem.epoch_distance(),
            elem.epoch_distance()
        );
        linearity_horizon(
            &force,
            &spk,
            &elem,
            Vector3::from(sun_ssb.pos),
            Vector3::from(sun_ssb.vel),
            &[400.0, 1600.0, 6400.0, 12800.0],
            &sigma,
        );
    }

    /// A covariance carried out to a cartesian one and back through equinoctial
    /// coordinates, under the real N-body force model, must reproduce `Phi P Phi^T`.
    ///
    /// The identity is exact as algebra, so what this measures is the arithmetic: two
    /// Jacobian inversions and three congruence transforms applied to a matrix carrying the
    /// wide dynamic range a real fitted orbit covariance has.
    #[test]
    fn equinoctial_covariance_round_trip_under_n_body() {
        let (elem, sun_ssb, epoch) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let epoch_final = Time::<TDB>::new(epoch.jd + 400.0);

        // A realistic fitted-orbit shape: shape and orientation to parts in 1e8,
        // along-track timing to fifteen minutes, carried out to cartesian.
        let sigma = well_observed_sigma(&elem);
        let jac = elem.state_jacobian::<Equatorial>().unwrap();
        let cov_cart = jac * Matrix6::from_diagonal(&sigma.map(|s| s * s)) * jac.transpose();

        let nominal: State<Equatorial> = elem.try_to_state().unwrap().into_frame();
        let (pos_f, vel_f, phi_full) = propagate_with_stm(
            &force,
            Vector3::from(nominal.pos) + Vector3::from(sun_ssb.pos),
            Vector3::from(nominal.vel) + Vector3::from(sun_ssb.vel),
            &[],
            epoch,
            epoch_final,
        )
        .unwrap();
        let phi: Matrix6<f64> = phi_full.fixed_view::<6, 6>(0, 0).into();

        let sun_f = spk
            .try_get_state_with_center::<Equatorial>(10, epoch_final, 0)
            .unwrap();

        let equi_0 = equinoctial_at(epoch, nominal.pos.into(), nominal.vel.into());
        let equi_f = equinoctial_at(
            epoch_final,
            pos_f - Vector3::from(sun_f.pos),
            vel_f - Vector3::from(sun_f.vel),
        );
        let jac_0 = equi_0.state_jacobian::<Equatorial>().unwrap();
        let jac_f = equi_f.state_jacobian::<Equatorial>().unwrap();

        // Out to element coordinates, forward under the transition matrix, and back. The
        // Jacobians are requested in the frame the transition matrix lives in, so nothing
        // here rotates anything.
        let inv_0 = equi_0.state_jacobian_inverse::<Equatorial>().unwrap();
        let inv_f = equi_f.state_jacobian_inverse::<Equatorial>().unwrap();
        let cov_equi_0 = inv_0 * cov_cart * inv_0.transpose();
        let phi_equi = inv_f * phi * jac_0;
        let cov_equi_f = phi_equi * cov_equi_0 * phi_equi.transpose();
        let got = jac_f * cov_equi_f * jac_f.transpose();

        let expected = phi * cov_cart * phi.transpose();
        let relative = (got - expected).norm() / expected.norm();
        println!("equinoctial covariance round trip under N-body: {relative:e} relative");
        assert!(relative < 1e-14, "covariance round trip {relative:e}");

        // The conditioning that number has to survive.
        let eig = cov_equi_0.symmetric_eigenvalues();
        println!(
            "  cond(P) cartesian {:e}, equinoctial {:e}",
            {
                let e = cov_cart.symmetric_eigenvalues();
                e.max() / e.min().abs()
            },
            eig.max() / eig.min().abs()
        );
    }

    /// Sigma-point divergence in equinoctial coordinates on a quiet main-belt arc,
    /// against the threshold above which the adaptive mixture splits.
    ///
    /// Sigma points are placed on the one-sigma shell along the three dominant
    /// eigenvectors of the equinoctial covariance, propagated through the true nonlinear
    /// flow, and scored as a Mahalanobis distance in the propagated covariance - the same
    /// metric the adaptive splitter uses. The whole arc is taken in one shot.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates sigma points"]
    fn equinoctial_sigma_point_divergence_on_a_quiet_arc() {
        let (elem, sun_ssb, epoch) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let config = SplitConfig::default();

        let sigma = well_observed_sigma(&elem);
        let jac = elem.state_jacobian::<Equatorial>().unwrap();
        let cov_cart = jac * Matrix6::from_diagonal(&sigma.map(|s| s * s)) * jac.transpose();

        let nominal: State<Equatorial> = elem.try_to_state().unwrap().into_frame();
        let sun_pos = Vector3::from(sun_ssb.pos);
        let sun_vel = Vector3::from(sun_ssb.vel);
        let pos_0 = Vector3::from(nominal.pos) + sun_pos;
        let vel_0 = Vector3::from(nominal.vel) + sun_vel;

        let equi_0 = equinoctial_at(epoch, nominal.pos.into(), nominal.vel.into());
        let jac_0 = equi_0.state_jacobian::<Equatorial>().unwrap();
        let inv_0 = equi_0.state_jacobian_inverse::<Equatorial>().unwrap();
        let cov_equi_0 = inv_0 * cov_cart * inv_0.transpose();

        let eigen = cov_equi_0.symmetric_eigen();
        let mut order: Vec<usize> = (0..6).collect();
        order.sort_by(|&a, &b| eigen.eigenvalues[b].total_cmp(&eigen.eigenvalues[a]));

        println!(
            "42 Isis, main belt, period {:.0} d. Equinoctial sigma-point divergence, \
             split threshold {:.1}.",
            elem.orbital_period(),
            config.split_threshold
        );
        println!(
            "{:>9}  {:>8}  {:>13}  {:>9}",
            "arc (d)", "orbits", "divergence", "verdict"
        );

        for arc in [400.0_f64, 1600.0, 6400.0, 12800.0] {
            let epoch_final = Time::<TDB>::new(epoch.jd + arc);
            let (pos_f, vel_f, phi_full) =
                propagate_with_stm(&force, pos_0, vel_0, &[], epoch, epoch_final).unwrap();
            let phi: Matrix6<f64> = phi_full.fixed_view::<6, 6>(0, 0).into();
            let sun_f = spk
                .try_get_state_with_center::<Equatorial>(10, epoch_final, 0)
                .unwrap();
            let sun_f_pos = Vector3::from(sun_f.pos);
            let sun_f_vel = Vector3::from(sun_f.vel);

            let equi_f = equinoctial_at(epoch_final, pos_f - sun_f_pos, vel_f - sun_f_vel);
            let inv_f = equi_f.state_jacobian_inverse::<Equatorial>().unwrap();
            let phi_equi = inv_f * phi * jac_0;

            // The propagated covariance, regularized the way the splitter regularizes it.
            let mut cov_f = phi_equi * cov_equi_0 * phi_equi.transpose();
            let reg = (cov_f.trace() * 1e-12).max(1e-30);
            for i in 0..6 {
                cov_f[(i, i)] += reg;
            }
            let inv_cov_f = cov_f.try_inverse().unwrap();

            let mut divergence = 0.0_f64;
            for &axis in order.iter().take(config.n_axes) {
                let lambda = eigen.eigenvalues[axis];
                if !lambda.is_finite() || lambda <= 0.0 {
                    continue;
                }
                let direction = eigen.eigenvectors.column(axis).into_owned();
                for sign in [1.0_f64, -1.0] {
                    let step = direction * (sign * config.sigma_factor * lambda.sqrt());
                    let start: State<Equatorial> = equi_0
                        .displaced_by(&step)
                        .try_to_state()
                        .unwrap()
                        .into_frame();
                    let (truth_pos, truth_vel) = propagate_state(
                        &force,
                        Vector3::from(start.pos) + sun_pos,
                        Vector3::from(start.vel) + sun_vel,
                        &[],
                        epoch,
                        epoch_final,
                    )
                    .unwrap();
                    let truth =
                        equinoctial_at(epoch_final, truth_pos - sun_f_pos, truth_vel - sun_f_vel);
                    let diff = equi_f.offset_to(&truth) - phi_equi * step;
                    divergence = divergence.max(
                        (diff.transpose() * inv_cov_f * diff)[(0, 0)]
                            .max(0.0)
                            .sqrt(),
                    );
                }
            }
            println!(
                "{arc:>9.0}  {:>8.2}  {divergence:>13.3e}  {:>9}",
                arc / elem.orbital_period(),
                if divergence < config.split_threshold {
                    "1 comp"
                } else {
                    "SPLITS"
                }
            );
        }
    }

    /// Construct an Earth-encountering NEO and its epoch, 200 days before closest
    /// approach.
    ///
    /// The test kernel carries no NEO, so one is built rather than searched for: the object
    /// is placed a set distance from the Earth at a chosen time with a velocity giving a
    /// plausible Earth-crossing orbit, then back-propagated to form the epoch state. That
    /// guarantees the encounter.
    fn encounter_neo(
        spk: &crate::spk::SpkCollection,
        force: &SpkNBody<'_>,
    ) -> (EquinoctialElements, Time<TDB>) {
        encounter_neo_at(spk, force, 0.003, true)
    }

    /// The same, with the miss distance chosen and the report optional.
    fn encounter_neo_at(
        spk: &crate::spk::SpkCollection,
        force: &SpkNBody<'_>,
        miss_au: f64,
        report: bool,
    ) -> (EquinoctialElements, Time<TDB>) {
        let encounter = Time::<TDB>::new(2_451_545.0 + 3000.0);
        let lead = 200.0;
        let epoch = Time::<TDB>::new(encounter.jd - lead);

        // Place the object near the Earth at the encounter, on an Earth-crossing orbit.
        let earth = spk
            .try_get_state_with_center::<Equatorial>(399, encounter, 10)
            .unwrap();
        let earth_pos = Vector3::from(earth.pos);
        let earth_vel = Vector3::from(earth.vel);

        // 0.003 AU, about eight lunar distances, offset across the Earth's track.
        let across = earth_vel.cross(&earth_pos).normalize();
        let helio_pos = earth_pos + across * miss_au;

        // Velocity from vis-viva for a 1.6 AU semi-major axis, tilted off the Earth's
        // velocity so the encounter has a realistic relative speed.
        let mu = kete_core::constants::GMS;
        let speed = (mu * (2.0 / helio_pos.norm() - 1.0 / 1.6)).sqrt();
        let pole = earth_pos.cross(&earth_vel).normalize();
        let tilt = nalgebra::Rotation3::from_scaled_axis(pole * 0.25);
        let helio_vel = tilt * earth_vel.normalize() * speed;

        // Back-propagate to the epoch, in barycentric coordinates.
        let sun_enc = spk
            .try_get_state_with_center::<Equatorial>(10, encounter, 0)
            .unwrap();
        let (pos_epoch, vel_epoch) = propagate_state(
            force,
            helio_pos + Vector3::from(sun_enc.pos),
            helio_vel + Vector3::from(sun_enc.vel),
            &[],
            encounter,
            epoch,
        )
        .unwrap();

        let sun_epoch = spk
            .try_get_state_with_center::<Equatorial>(10, epoch, 0)
            .unwrap();
        let sun_pos = Vector3::from(sun_epoch.pos);
        let sun_vel = Vector3::from(sun_epoch.vel);
        let state = State::<Equatorial>::new(
            Desig::Empty,
            epoch,
            pos_epoch - sun_pos,
            vel_epoch - sun_vel,
            10,
        );
        let elem = EquinoctialElements::from_state(&state.into_frame()).unwrap();

        // Report the encounter actually achieved, by walking the nominal trajectory.
        let (mut walk_pos, mut walk_vel) = (pos_epoch, vel_epoch);
        let (mut walk_time, mut closest, mut closest_at) = (epoch.jd, f64::INFINITY, epoch.jd);
        #[allow(unused_assignments, reason = "reported below")]
        while walk_time < epoch.jd + 400.0 {
            let next = walk_time + 2.0;
            let stepped = propagate_state(
                force,
                walk_pos,
                walk_vel,
                &[],
                Time::<TDB>::new(walk_time),
                Time::<TDB>::new(next),
            )
            .unwrap();
            (walk_pos, walk_vel) = stepped;
            walk_time = next;
            let earth_now = spk
                .try_get_state_with_center::<Equatorial>(399, Time::<TDB>::new(next), 0)
                .unwrap();
            let separation = (walk_pos - Vector3::from(earth_now.pos)).norm();
            if separation < closest {
                closest = separation;
                closest_at = next;
            }
        }

        if report {
            println!(
                "Constructed NEO: closest approach {closest:.5} AU ({:.1} lunar distances) \
                 at epoch + {:.0} d",
                closest / 0.00257,
                closest_at - epoch.jd
            );
        }
        (elem, epoch)
    }

    /// The case where element coordinates may buy nothing: an Earth-encountering NEO
    /// carried through closest approach.
    ///
    /// Close to the Earth the perturbing acceleration climbs to a substantial fraction of
    /// the solar monopole, so the `O(a_p)` suppression that makes the element flow nearly
    /// linear stops applying. A deep enough encounter can also drive the osculating
    /// heliocentric orbit past unit eccentricity, where the elements exclude a finite arc of
    /// true longitudes outright.
    ///
    /// The test kernel carries no NEO, so one is constructed: the object is placed a set
    /// distance from the Earth at a chosen time with a velocity giving a plausible
    /// Earth-crossing orbit, then back-propagated to form the epoch state. That guarantees
    /// the encounter rather than searching for one.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates a clone ensemble"]
    fn element_vs_cartesian_linearity_horizon_through_encounter() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);

        let (elem, epoch) = encounter_neo(&spk, &force);
        let sun_epoch = spk
            .try_get_state_with_center::<Equatorial>(10, epoch, 0)
            .unwrap();
        let sun_pos = Vector3::from(sun_epoch.pos);
        let sun_vel = Vector3::from(sun_epoch.vel);

        println!(
            "Constructed NEO: a={:.3} AU, e={:.3}, q={:.3} AU, period {:.0} d",
            elem.semi_major(),
            elem.eccentricity(),
            elem.peri_dist(),
            elem.orbital_period()
        );

        let sigma = well_observed_sigma(&elem);
        linearity_horizon(
            &force,
            &spk,
            &elem,
            sun_pos,
            sun_vel,
            &[100.0, 195.0, 205.0, 400.0],
            &sigma,
        );
    }

    /// **The operational claim the element migration exists to deliver**: adaptive
    /// splitting should fire at encounters and essentially nowhere else.
    ///
    /// The linearity horizon measures the gain: large on a quiet arc, and gone through a
    /// deep encounter. The adaptive splitter's divergence is computed in element
    /// coordinates, so that gain should show up directly as components not being created.
    /// This counts them.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates mixtures"]
    fn splitting_frequency_quiet_versus_encounter() {
        let (elem, sun_ssb, epoch) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };
        let _ = sun_ssb;

        // A well observed orbit, written in element coordinates where an orbit uncertainty is
        // naturally close to diagonal.
        let sigma = well_observed_sigma(&elem);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let component = UncertainState::new(elem.clone(), cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        let config = SplitConfig::default();

        println!("42 Isis, main belt. Components after adaptive propagation:");
        println!("{:>9}  {:>8}  {:>12}", "arc (d)", "orbits", "components");
        let period = elem.orbital_period();
        for arc in [400.0_f64, 1600.0, 6400.0, 12800.0] {
            let target = Time::<TDB>::new(epoch.jd + arc);
            let out =
                propagate_diffuse_state_adaptive(&mixture, &force, target, &config, &resolver)
                    .unwrap();
            println!(
                "{arc:>9.0}  {:>8.2}  {:>12}",
                arc / period,
                out.n_components()
            );
        }

        // The other half: the same question through a close planetary encounter, where the
        // linearity gain vanishes and splitting is the only remaining tool.
        let (neo, neo_epoch) = encounter_neo(&spk, &force);
        let neo_component = {
            let mut cov = DMatrix::<f64>::zeros(6, 6);
            for (i, s) in sigma.iter().enumerate() {
                cov[(i, i)] = s * s;
            }
            UncertainState::new(neo.clone(), cov, vec![]).unwrap()
        };
        let neo_mixture = DiffuseState::from_uncertain(neo_component);

        println!();
        println!("Constructed NEO, closest approach at epoch + 200 d:");
        println!("{:>9}  {:>12}", "arc (d)", "components");
        for arc in [100.0_f64, 195.0, 205.0, 400.0] {
            let target = Time::<TDB>::new(neo_epoch.jd + arc);
            let count =
                propagate_diffuse_state_adaptive(&neo_mixture, &force, target, &config, &resolver)
                    .map_or_else(
                        |e| format!("failed: {e}"),
                        |m| {
                            // Whether the count is convergence or saturation is the question. A
                            // component that could not be split further keeps its peak divergence,
                            // so the weight above the split threshold measures how much of the
                            // distribution the mixture failed to resolve.
                            format!(
                                "{:>6}   unresolved weight {:.3}",
                                m.n_components(),
                                m.unresolved_weight(config.split_threshold)
                            )
                        },
                    );
            println!("{arc:>9.0}  {count}");
        }
    }

    /// The adaptive splitter must respect `max_components`.
    ///
    /// A component count above the cap means either a budget that does not bind or a count
    /// arriving from somewhere the budget does not see. This drives the splitting path with
    /// a small cap and a covariance wide enough to force splitting, so it localizes any
    /// overshoot without needing a long encounter to expose it.
    #[test]
    fn adaptive_splitting_respects_the_component_budget() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        // The encounter is what actually demands splitting; a wide covariance on a quiet
        // main-belt arc does not, because the flow stays linear over it.
        let (elem, epoch) = encounter_neo(&spk, &force);
        let sigma = well_observed_sigma(&elem);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let component = UncertainState::new(elem, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        for cap in [4_usize, 12, 40] {
            let config = SplitConfig {
                max_components: cap,
                ..SplitConfig::default()
            };
            let out = propagate_diffuse_state_adaptive(
                &mixture,
                &force,
                Time::<TDB>::new(epoch.jd + 205.0),
                &config,
                &resolver,
            )
            .unwrap();
            println!(
                "cap {cap:>4} -> {:>4} components, unresolved weight {:.3}",
                out.n_components(),
                out.unresolved_weight(config.split_threshold)
            );
            assert!(
                out.n_components() <= cap,
                "cap {cap} exceeded: got {}",
                out.n_components()
            );
        }
    }

    /// **Does splitting actually reduce the divergence?**
    ///
    /// The adaptive loop assumes it does: a component whose linear prediction is bad gets
    /// subdivided so each child spans a narrower region where the flow is more linear. If
    /// that assumption fails the loop cannot converge - it splits until the budget runs out
    /// and every component stays unresolved, which is exactly what the encounter case does.
    /// Distinguishing "this encounter is genuinely unresolvable" from "we are splitting
    /// along a useless direction" needs the mechanism measured directly.
    ///
    /// The divergence should fall roughly linearly with the spread: the nonlinear error
    /// grows as the square of the perturbation while the normalizing sigma grows linearly,
    /// so a three-way split ought to buy a factor of order two.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn does_splitting_reduce_divergence() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        let (neo, neo_epoch) = encounter_neo(&spk, &force);
        let sigma = well_observed_sigma(&neo);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let base = UncertainState::new(neo, cov, vec![]).unwrap();
        let config = SplitConfig::default();

        println!(
            "{:>8}  {:>12}  {:>12}  {:>8}",
            "arc (d)", "parent div", "worst child", "ratio"
        );
        for arc in [195.0_f64, 200.0, 205.0, 400.0] {
            let target = Time::<TDB>::new(neo_epoch.jd + arc);
            let Ok(diag) = propagate_with_diagnosis(
                &base,
                &force,
                target,
                config.n_axes,
                config.sigma_factor,
                config.position_spacing_au,
                &resolver,
            ) else {
                println!("{arc:>8.0}  parent propagation failed");
                continue;
            };

            let Ok(parts) =
                split_for_propagation(&base, &diag.propagated.cov_matrix, &diag.augmented_stm)
            else {
                println!("{arc:>8.0}  {:>12.3e}  split failed", diag.divergence);
                continue;
            };

            let mut worst_child = 0.0_f64;
            for (_, child) in &parts {
                match propagate_with_diagnosis(
                    child,
                    &force,
                    target,
                    config.n_axes,
                    config.sigma_factor,
                    config.position_spacing_au,
                    &resolver,
                ) {
                    Ok(d) => worst_child = worst_child.max(d.divergence),
                    Err(_) => worst_child = f64::INFINITY,
                }
            }
            println!(
                "{arc:>8.0}  {:>12.3e}  {worst_child:>12.3e}  {:>8.2}",
                diag.divergence,
                diag.divergence / worst_child
            );
        }
    }

    /// Where is the boundary between "splitting can resolve this" and "it cannot"?
    ///
    /// A three-way split narrows each child along the split axis, so the component count
    /// needed to reach the split threshold grows exponentially in the divergence. The
    /// growth rate depends on how nonlinear the encounter is, which varies, so this
    /// reports the parameter-free lower bound from
    /// [`minimum_components_for_divergence`] rather than an extrapolation. This sweeps
    /// the two knobs a caller actually has, the encounter depth and how well the orbit is
    /// known, and reports where the requirement comes back inside a practical budget.
    ///
    /// Divergence only, no adaptive propagation: one diagnosis per cell is far cheaper than
    /// running the splitter to saturation.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn splitting_tractability_boundary() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };
        let config = SplitConfig::default();

        println!(
            "Lower bound on components needed to reach a divergence of {:.1}, at 5 days \
             past closest approach.",
            config.split_threshold
        );
        println!(
            "{:>10}  {:>8}  {:>12}  {:>12}  {:>12}",
            "miss (AU)", "lunar", "sigma scale", "divergence", "min comps"
        );

        for miss in [0.003_f64, 0.01, 0.03, 0.1] {
            let (neo, epoch) = encounter_neo_at(&spk, &force, miss, false);
            let target = Time::<TDB>::new(epoch.jd + 205.0);
            let sigma = well_observed_sigma(&neo);
            for scale in [1.0_f64, 0.1, 0.01] {
                let mut cov = DMatrix::<f64>::zeros(6, 6);
                for (i, s) in sigma.iter().enumerate() {
                    cov[(i, i)] = (s * scale) * (s * scale);
                }
                let component = UncertainState::new(neo.clone(), cov, vec![]).unwrap();
                let text = match propagate_with_diagnosis(
                    &component,
                    &force,
                    target,
                    config.n_axes,
                    config.sigma_factor,
                    config.position_spacing_au,
                    &resolver,
                ) {
                    Ok(diag) if diag.divergence <= config.split_threshold => {
                        format!("{:>12.3e}  {:>12}", diag.divergence, 1)
                    }
                    Ok(diag) if diag.divergence.is_finite() => format!(
                        "{:>12.3e}  {:>12.2e}",
                        diag.divergence,
                        minimum_components_for_divergence(diag.divergence, config.split_threshold)
                    ),
                    Ok(diag) => format!("{:>12.3e}  {:>12}", diag.divergence, "unbounded"),
                    Err(_) => format!("{:>12}  {:>12}", "failed", "-"),
                };
                println!(
                    "{miss:>10.3}  {:>8.1}  {scale:>12.2}  {text}",
                    miss / 0.00257
                );
            }
        }
    }

    /// Can a small enough covariance survive a deep encounter intact?
    ///
    /// `splitting_tractability_boundary` shows the divergence is linear in the covariance
    /// scale, and the component count exponential in the divergence, so shrinking the
    /// uncertainty should buy back tractability very fast even well inside the Hill sphere.
    /// This runs the adaptive propagation itself rather than inferring a count, at a miss
    /// distance of 1.2 lunar distances - roughly a third of Earth's Hill radius.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates mixtures"]
    fn small_covariance_survives_the_encounter() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        let (neo, epoch) = encounter_neo_at(&spk, &force, 0.003, false);
        let target = Time::<TDB>::new(epoch.jd + 205.0);
        let config = SplitConfig::default();

        println!("Miss distance 0.003 AU (1.2 lunar distances), 5 days past closest approach.");
        println!(
            "{:>12}  {:>12}  {:>10}  {:>12}  {:>12}  {:>10}",
            "sigma scale", "divergence", "bound", "components", "unresolved", "verdict"
        );
        let sigma = well_observed_sigma(&neo);
        for scale in [0.1_f64, 0.03, 0.01, 0.003, 0.001] {
            let mut cov = DMatrix::<f64>::zeros(6, 6);
            for (i, s) in sigma.iter().enumerate() {
                cov[(i, i)] = (s * scale) * (s * scale);
            }
            let component = UncertainState::new(neo.clone(), cov, vec![]).unwrap();

            // The bound is computed from the divergence of the single unsplit component --
            // exactly what a caller has in hand before deciding whether to attempt a cascade.
            let divergence = sigma_point_divergence(
                &component,
                &force,
                target,
                config.n_axes,
                config.sigma_factor,
                config.position_spacing_au,
                &resolver,
            )
            .unwrap();
            let bound = minimum_components_for_divergence(divergence, config.split_threshold);

            let mixture = DiffuseState::from_uncertain(component);
            match propagate_diffuse_state_adaptive(&mixture, &force, target, &config, &resolver) {
                Ok(out) => {
                    let unresolved = out.unresolved_weight(config.split_threshold);
                    let verdict = if unresolved > 1e-9 {
                        "saturated"
                    } else {
                        "resolved"
                    };
                    let actual = out.n_components();
                    println!(
                        "{scale:>12.3}  {divergence:>12.2}  {bound:>10.1}  {actual:>12}  \
                         {unresolved:>12.3}  {verdict:>10}"
                    );
                    // The whole point of the bound is that it never overstates the cost.
                    // Only meaningful where the cascade actually converged; a saturated run
                    // stopped at the budget rather than at the requirement.
                    if unresolved <= 1e-9 {
                        assert!(
                            bound <= actual as f64,
                            "bound {bound} exceeded the {actual} components actually needed \
                             at scale {scale}"
                        );
                    }
                }
                Err(e) => println!("{scale:>12.3}  failed: {e}"),
            }
        }
    }

    /// Cost of running the numerics-floor regime with the diminishing-returns check
    /// disabled: component count, wall time and settled divergence, against the same
    /// cascade with the check at its default.
    ///
    /// The check is an economics device, not a correctness one - splits preserve the
    /// mixture moments - so whether `min_split_improvement` needs a different default
    /// (or any companion mechanism) reduces to what the floor regime wastes without
    /// it. The floor case here is the one measured by `cascade_divergence_trace`:
    /// covariance scale 0.001 through the 0.003 AU encounter stalls at divergence
    /// 0.12-0.16 against the 0.1 density threshold.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn floor_churn_cost_without_improvement_check() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };
        let (neo, epoch) = encounter_neo_at(&spk, &force, 0.003, false);
        let target = Time::<TDB>::new(epoch.jd + 205.0);
        let sigma = well_observed_sigma(&neo);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = (s * 0.001) * (s * 0.001);
        }
        let component = UncertainState::new(neo, cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);

        println!(
            "{:>12} {:>10} {:>8} {:>7} {:>12} {:>10} {:>10} {:>8}",
            "improvement", "budget", "depth", "n_axes", "components", "max_div", "unresolved", "secs"
        );
        // The big-budget row is opt-in: in the floor regime nothing resolves, so it
        // pays a full diagnosis for every component of every generation and its cost
        // is the point of measuring it - run with KETE_FLOOR_BIG=1 and patience.
        let mut configs = vec![
            (0.1_f64, 50_000_usize, 10_u32, 3_usize),
            (0.1, 50_000, 10, 6),
        ];
        if std::env::var("KETE_FLOOR_BIG").is_ok() {
            configs.push((0.0, 100_000, 25, 3));
        }
        for (improvement, budget, depth, n_axes) in configs {
            let config = SplitConfig {
                split_threshold: 0.1,
                max_components: budget,
                max_split_depth: depth,
                min_split_improvement: improvement,
                n_axes,
                ..SplitConfig::default()
            };
            let start = std::time::Instant::now();
            match propagate_diffuse_state_adaptive(&mixture, &force, target, &config, &resolver) {
                Ok(out) => println!(
                    "{improvement:>12.2} {budget:>10} {depth:>8} {n_axes:>7} {:>12} {:>10.3} {:>10.3} {:>8.1}",
                    out.n_components(),
                    out.max_unresolved_divergence(),
                    out.unresolved_weight(config.split_threshold),
                    start.elapsed().as_secs_f64()
                ),
                Err(e) => println!("{improvement:>12.1} {budget:>10} {depth:>8} failed: {e}"),
            }
        }
    }

    /// TEMPORARY EVALUATION: residual scaling exponent as the settle discriminator.
    ///
    /// At each level of a worst-lineage walk, the top covariance axis is probed at
    /// the full one-sigma amplitude and at half that amplitude, and the whitened
    /// residual norms are compared.  A residual produced by smooth dynamics is a
    /// second-order Taylor term: half the probe gives a quarter the residual
    /// (ratio ~ 4, exponent ~ 2).  A residual at the measurement floor does not
    /// respond to probe amplitude (ratio ~ 1, exponent ~ 0).  The exponent is the
    /// settle criterion: structure that scales is resolvable by splitting,
    /// structure that does not is not.  The dichotomy is derived (Taylor order),
    /// not tuned.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn residual_scaling_probe() {
        use nalgebra::{DVector, SymmetricEigen};

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        // Scenario A: the numerics floor.
        let (neo, epoch) = encounter_neo_at(&spk, &force, 0.003, false);
        let sigma = well_observed_sigma(&neo);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = (s * 0.001) * (s * 0.001);
        }
        let floor = UncertainState::new(neo, cov, vec![]).unwrap();
        let floor_target = Time::<TDB>::new(epoch.jd + 205.0);

        // Scenario B: the JFC trail flyby valley.
        let t_enc = Time::<TDB>::new(2_461_000.0);
        let jup = spk
            .try_get_state_with_center::<Ecliptic>(5, t_enc, 10)
            .unwrap();
        let jpos = Vector3::from(jup.pos);
        let p_enc = jpos * (1.0 - 0.25 / jpos.norm());
        let r = p_enc.norm();
        let a = r / 1.55;
        let speed = (kete_core::constants::GMS * (2.0 / r - 1.0 / a)).sqrt();
        let tangent = Vector3::z().cross(&p_enc).normalize();
        let member: State<Equatorial> =
            State::<Ecliptic>::new(Desig::Empty, t_enc, p_enc, tangent * speed, 10).into_frame();
        let t0 = Time::<TDB>::new(t_enc.jd - 900.0);
        let (sp, sv) = resolver(t_enc).unwrap();
        let (p0, v0) = propagate_state(
            &force,
            Vector3::from(member.pos) + sp,
            Vector3::from(member.vel) + sv,
            &[],
            t_enc,
            t0,
        )
        .unwrap();
        let (s0p, s0v) = resolver(t0).unwrap();
        let helio_t0 = State::<Equatorial>::new(Desig::Empty, t0, p0 - s0p, v0 - s0v, 10);
        let mut elem = CometElements::from_state(&helio_t0.into_frame()).unwrap();
        elem.peri_time = (elem.peri_time.jd + 190.0).into();
        let mut tcov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in [2e-4, 5e-4, 95.0, 2e-5, 2e-5, 2e-5].iter().enumerate() {
            tcov[(i, i)] = s * s;
        }
        let valley = UncertainState::from_cometary(&elem, &tcov, vec![]).unwrap();
        let valley_target = Time::<TDB>::new(t_enc.jd + 300.0);

        for (label, start, target) in [
            ("floor", floor, floor_target),
            ("valley", valley, valley_target),
        ] {
            println!("{label}:");
            println!(
                "{:>3} {:>10} {:>10} {:>10} {:>7} {:>9}",
                "lvl", "d_own", "|r(d)|", "|r(d/2)|", "ratio", "exponent"
            );
            let mut current = start;
            for level in 0..10 {
                let diag = propagate_with_diagnosis(
                    &current, &force, target, 3, 1.0, Some(0.001), &resolver,
                )
                .unwrap();
                let pf = &diag.propagated.cov_matrix;
                let marg: Vec<f64> = (0..6).map(|i| pf[(i, i)].sqrt().max(1e-300)).collect();

                let sym = SymmetricEigen::new(current.cov_matrix.clone());
                let top = (0..6)
                    .max_by(|&x, &y| {
                        sym.eigenvalues[x].partial_cmp(&sym.eigenvalues[y]).unwrap()
                    })
                    .unwrap();
                let lam = sym.eigenvalues[top];
                let dir = sym.eigenvectors.column(top).clone_owned();
                let epoch_c = current.elements.epoch;
                let (e0p, e0v) = resolver(epoch_c).unwrap();
                let (efp, efv) = resolver(target).unwrap();

                let mut norms = [0.0_f64; 2];
                for (slot, amp) in [1.0_f64, 0.5].iter().enumerate() {
                    for sign in [1.0_f64, -1.0] {
                        let delta = &dir * (sign * amp * lam.sqrt());
                        let step = Vector6::from_iterator(delta.iter().copied());
                        let start_state: State<Equatorial> = current
                            .elements
                            .displaced_by(&step)
                            .try_to_state()
                            .unwrap()
                            .into_frame();
                        let (pf_, vf_) = propagate_state(
                            &force,
                            Vector3::from(start_state.pos) + e0p,
                            Vector3::from(start_state.vel) + e0v,
                            &[],
                            epoch_c,
                            target,
                        )
                        .unwrap();
                        let fin = equinoctial_at(target, pf_ - efp, vf_ - efv);
                        let nonlin = diag.propagated.elements.offset_to(&fin);
                        let lin = &diag.augmented_stm * &delta;
                        let w = DVector::from_iterator(
                            6,
                            (0..6).map(|i| (nonlin[i] - lin[i]) / marg[i]),
                        );
                        norms[slot] = norms[slot].max(w.norm());
                    }
                }
                let ratio = norms[0] / norms[1].max(1e-300);
                println!(
                    "{level:>3} {:>10.2} {:>10.3e} {:>10.3e} {ratio:>7.2} {:>9.2}",
                    diag.divergence,
                    norms[0],
                    norms[1],
                    ratio.log2()
                );

                let parts = split_for_propagation(
                    &current,
                    &diag.propagated.cov_matrix,
                    &diag.augmented_stm,
                )
                .unwrap();
                let mut best: Option<(f64, UncertainState)> = None;
                for (_, child) in parts {
                    let cd = propagate_with_diagnosis(
                        &child, &force, target, 3, 1.0, Some(0.001), &resolver,
                    )
                    .unwrap();
                    if best.as_ref().is_none_or(|(d, _)| cd.edge_divergence > *d) {
                        best = Some((cd.edge_divergence, child));
                    }
                }
                current = best.unwrap().1;
            }
        }
    }

    /// Does the adaptive split narrow the axis that carries the divergence?
    ///
    /// Walks the worst lineage of the JFC-trail flyby valley (the trail scenario from
    /// analysis/apophis_validation, rebuilt here), and per level prints each probed
    /// axis's measured divergence next to the fraction of that axis's variance the
    /// chosen split direction removes.  A stalled level whose worst-divergence axis
    /// receives no variance reduction is a direction mismatch, and choosing the split
    /// direction from the worst probe residual would help; the worst axis already
    /// being narrowed means the valley is genuine sub-component structure that no
    /// direction choice can shorten.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn valley_split_axis_alignment() {
        use nalgebra::{DVector, SymmetricEigen};

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        // The JFC trail from the analysis scripts, rebuilt: aphelion parked just
        // inside Jupiter's distance, trail = emission-time spread whose leading end
        // sweeps 0.18-0.35 AU from Jupiter while the mean stays ~0.9 AU clear.
        let t_enc = Time::<TDB>::new(2_461_000.0);
        let jup = spk
            .try_get_state_with_center::<Ecliptic>(5, t_enc, 10)
            .unwrap();
        let jpos = Vector3::from(jup.pos);
        let p_enc = jpos * (1.0 - 0.25 / jpos.norm());
        let r = p_enc.norm();
        let a = r / 1.55;
        let speed = (kete_core::constants::GMS * (2.0 / r - 1.0 / a)).sqrt();
        let tangent = Vector3::z().cross(&p_enc).normalize();
        let member: State<Equatorial> =
            State::<Ecliptic>::new(Desig::Empty, t_enc, p_enc, tangent * speed, 10).into_frame();

        let t0 = Time::<TDB>::new(t_enc.jd - 900.0);
        let (sp, sv) = resolver(t_enc).unwrap();
        let (p0, v0) = propagate_state(
            &force,
            Vector3::from(member.pos) + sp,
            Vector3::from(member.vel) + sv,
            &[],
            t_enc,
            t0,
        )
        .unwrap();
        let (s0p, s0v) = resolver(t0).unwrap();
        let helio_t0 = State::<Equatorial>::new(Desig::Empty, t0, p0 - s0p, v0 - s0v, 10);
        let mut elem = CometElements::from_state(&helio_t0.into_frame()).unwrap();
        // The mean lags the encountering member; the member sits at ~2 sigma of the
        // emission-time spread, matching the python scenario.
        elem.peri_time = (elem.peri_time.jd + 190.0).into();
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in [2e-4, 5e-4, 95.0, 2e-5, 2e-5, 2e-5].iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let trail = UncertainState::from_cometary(&elem, &cov, vec![]).unwrap();
        let t_final = Time::<TDB>::new(t_enc.jd + 300.0);

        println!(
            "Per level: divergence d and split variance-reduction fraction f, per\n\
             covariance axis ranked widest first (the loop probes the top 3).\n"
        );
        println!(
            "{:>3} {:>9}  {:>44}  {:>44}  {:>7} {:>7}",
            "lvl", "max_div", "d per axis (widest..)", "f per axis", "worst_d", "split_f"
        );
        let mut current = trail;
        for level in 0..12 {
            let diag =
                propagate_with_diagnosis(&current, &force, t_final, 3, 1.0, Some(0.001), &resolver)
                    .unwrap();

            // Regularized inverse of the propagated covariance, correlation form,
            // matching the metric in kete_core.
            let pf = &diag.propagated.cov_matrix;
            let marg: Vec<f64> = (0..6).map(|i| pf[(i, i)].sqrt()).collect();
            let mut corr = pf.clone();
            for row in 0..6 {
                for col in 0..6 {
                    corr[(row, col)] /= marg[row] * marg[col];
                }
            }
            for i in 0..6 {
                corr[(i, i)] += 1e-6;
            }
            let inv_corr = corr.try_inverse().unwrap();
            let mahal = |diff: &DVector<f64>| -> f64 {
                let w = DVector::from_iterator(6, (0..6).map(|i| diff[i] / marg[i]));
                (w.transpose() * &inv_corr * &w)[(0, 0)].max(0.0).sqrt()
            };

            // Per-axis probe divergences, all six axes ranked widest first.
            let sym = SymmetricEigen::new(current.cov_matrix.clone());
            let mut order: Vec<usize> = (0..6).collect();
            order.sort_by(|&x, &y| sym.eigenvalues[y].partial_cmp(&sym.eigenvalues[x]).unwrap());
            let epoch = current.elements.epoch;
            let (e0p, e0v) = resolver(epoch).unwrap();
            let (efp, efv) = resolver(t_final).unwrap();
            let mut d_axis = [0.0_f64; 6];
            for (rank, &ax) in order.iter().enumerate() {
                let lam = sym.eigenvalues[ax];
                if lam <= 0.0 {
                    continue;
                }
                let dir = sym.eigenvectors.column(ax).clone_owned();
                for sign in [1.0_f64, -1.0] {
                    let delta = &dir * (sign * lam.sqrt());
                    let step = Vector6::from_iterator(delta.iter().copied());
                    let start: State<Equatorial> = current
                        .elements
                        .displaced_by(&step)
                        .try_to_state()
                        .unwrap()
                        .into_frame();
                    let (pf_, vf_) = propagate_state(
                        &force,
                        Vector3::from(start.pos) + e0p,
                        Vector3::from(start.vel) + e0v,
                        &[],
                        epoch,
                        t_final,
                    )
                    .unwrap();
                    let fin = equinoctial_at(t_final, pf_ - efp, vf_ - efv);
                    let nonlin = diag.propagated.elements.offset_to(&fin);
                    let lin = &diag.augmented_stm * &delta;
                    let diff = DVector::from_iterator(6, (0..6).map(|i| nonlin[i] - lin[i]));
                    d_axis[rank] = d_axis[rank].max(mahal(&diff));
                }
            }

            // The split the loop would choose, and the variance fraction it removes
            // from each axis: f_k = (e_k . r)^2 / (2 lambda_k).
            let symf = SymmetricEigen::new(diag.propagated.cov_matrix.clone());
            let fmax = (0..6)
                .max_by(|&x, &y| {
                    symf.eigenvalues[x]
                        .partial_cmp(&symf.eigenvalues[y])
                        .unwrap()
                })
                .unwrap();
            let u = diag.augmented_stm.transpose() * symf.eigenvectors.column(fmax);
            let u_hat = &u / u.norm();
            let sig_u = (u_hat.transpose() * &current.cov_matrix * &u_hat)[(0, 0)].sqrt();
            let r_vec = (&current.cov_matrix * &u_hat) / sig_u;
            let mut f_axis = [0.0_f64; 6];
            for (rank, &ax) in order.iter().enumerate() {
                let lam = sym.eigenvalues[ax].max(1e-300);
                let proj = sym.eigenvectors.column(ax).dot(&r_vec);
                f_axis[rank] = 0.5 * proj * proj / lam;
            }

            let fmt = |vals: &[f64; 6]| -> String {
                vals.iter()
                    .map(|v| format!("{v:>6.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            let worst_d = (0..6)
                .max_by(|&x, &y| d_axis[x].total_cmp(&d_axis[y]))
                .unwrap();
            println!(
                "{level:>3} {:>9.1}  {}  {}  {worst_d:>7} {:>7.2}",
                diag.divergence,
                fmt(&d_axis),
                fmt(&f_axis),
                f_axis[worst_d]
            );

            // Descend into the worst child, as the trace does.
            let parts =
                split_for_propagation(&current, &diag.propagated.cov_matrix, &diag.augmented_stm)
                    .unwrap();
            let mut best: Option<(f64, UncertainState)> = None;
            for (_, child) in parts {
                let cd = propagate_with_diagnosis(
                    &child,
                    &force,
                    t_final,
                    3,
                    1.0,
                    Some(0.001),
                    &resolver,
                )
                .unwrap();
                if best.as_ref().is_none_or(|(d, _)| cd.edge_divergence > *d) {
                    best = Some((cd.edge_divergence, child));
                }
            }
            current = best.unwrap().1;
        }
    }

    /// Trace the worst-divergence lineage of a splitting cascade, one K=3 split per
    /// level, printing every child's edge divergence and which settle rule would fire.
    ///
    /// Exists to diagnose cascade stalls that the aggregate `unresolved_weight` of a
    /// full adaptive run cannot localize: it shows whether the divergence per level
    /// falls (healthy), stalls above the threshold (`no_improvement` settles it), or
    /// grows (a probe family or numerics floor artifact).
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn cascade_divergence_trace() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };

        let (neo, epoch) = encounter_neo_at(&spk, &force, 0.003, false);
        let target = Time::<TDB>::new(epoch.jd + 205.0);
        let config = SplitConfig::default();
        let sigma = well_observed_sigma(&neo);

        println!(
            "Worst-lineage trace, 0.003 AU miss. threshold {}, min improvement {}",
            config.split_threshold, config.min_split_improvement
        );
        for scale in [0.003_f64, 0.001, 3e-4] {
            let mut cov = DMatrix::<f64>::zeros(6, 6);
            for (i, s) in sigma.iter().enumerate() {
                cov[(i, i)] = (s * scale) * (s * scale);
            }
            let mut current = UncertainState::new(neo.clone(), cov, vec![]).unwrap();
            let mut parent_edge: Option<f64> = None;
            println!("sigma scale {scale}:");
            println!(
                "  {:>5}  {:>32}  {:>10}",
                "level", "child edge divergences", "worst rule"
            );
            for level in 0..8 {
                let diag = propagate_with_diagnosis(
                    &current,
                    &force,
                    target,
                    config.n_axes,
                    config.sigma_factor,
                    config.position_spacing_au,
                    &resolver,
                )
                .unwrap();
                let rule = if diag.divergence <= config.split_threshold {
                    "resolved"
                } else if parent_edge.is_some_and(|pd| {
                    pd > config.split_threshold
                        && diag.edge_divergence >= pd * (1.0 - config.min_split_improvement)
                }) {
                    "no improvement"
                } else {
                    "splits"
                };
                if level == 0 {
                    println!("  {level:>5}  {:>32.4}  {rule:>10}", diag.edge_divergence);
                }
                if rule != "splits" {
                    break;
                }

                // Split, diagnose all three children, follow the worst.
                let parts = split_for_propagation(
                    &current,
                    &diag.propagated.cov_matrix,
                    &diag.augmented_stm,
                )
                .unwrap();
                let child_diags: Vec<_> = parts
                    .iter()
                    .map(|(_, child)| {
                        propagate_with_diagnosis(
                            child,
                            &force,
                            target,
                            config.n_axes,
                            config.sigma_factor,
                            config.position_spacing_au,
                            &resolver,
                        )
                        .unwrap()
                    })
                    .collect();
                let edges: Vec<f64> = child_diags.iter().map(|d| d.edge_divergence).collect();
                let worst = edges
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(i, _)| i)
                    .unwrap();
                let worst_rule = if edges[worst] <= config.split_threshold {
                    "resolved"
                } else if edges[worst]
                    >= diag.edge_divergence * (1.0 - config.min_split_improvement)
                {
                    "no improvement"
                } else {
                    "splits"
                };
                println!(
                    "  {:>5}  {:>10.4} {:>10.4} {:>10.4}  {worst_rule:>10}",
                    level + 1,
                    edges[0],
                    edges[1],
                    edges[2]
                );
                parent_edge = Some(diag.edge_divergence);
                current = parts[worst].1.clone();
            }
        }
    }

    /// Shared driver for `mixture_matches_propagated_ensemble`: propagate a clone
    /// ensemble and the adaptive mixture to `target`, print the comparison, and
    /// return the whitened covariance errors `(split, unsplit)` against the ensemble.
    ///
    /// All scoring happens in element coordinates about the propagated mixture mean,
    /// whitened by the ensemble's own marginal scales - the units are mixed and the
    /// along-track spread would otherwise dominate every norm.
    fn mixture_vs_ensemble(
        label: &str,
        component: &UncertainState,
        target: Time<TDB>,
        n_clones: usize,
        config: &SplitConfig,
        force: &SpkNBody<'_>,
        spk: &crate::spk::SpkCollection,
    ) -> (f64, f64) {
        use kete_core::state::propagate_uncertain;
        use rayon::prelude::*;

        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };
        let epoch = component.elements.epoch;
        let (sun0_pos, sun0_vel) = resolver(epoch).unwrap();
        let (sunf_pos, sunf_vel) = resolver(target).unwrap();

        // The truth: clones drawn from the initial distribution and carried through the
        // full nonlinear flow, read back as osculating heliocentric elements.
        let samples: Vec<(State<Equatorial>, Vec<f64>)> =
            component.sample(n_clones, Some(1234)).unwrap();
        let finals: Vec<EquinoctialElements> = samples
            .par_iter()
            .map(|(s, _)| {
                let (pos_f, vel_f) = propagate_state(
                    force,
                    Vector3::from(s.pos) + sun0_pos,
                    Vector3::from(s.vel) + sun0_vel,
                    &[],
                    epoch,
                    target,
                )
                .unwrap();
                equinoctial_at(target, pos_f - sunf_pos, vel_f - sunf_vel)
            })
            .collect();

        // The two models: the adaptive mixture, and the same component propagated as a
        // single Gaussian with no splitting.
        let mixture = propagate_diffuse_state_adaptive(
            &DiffuseState::from_uncertain(component.clone()),
            force,
            target,
            config,
            &resolver,
        )
        .unwrap();
        let unsplit = propagate_uncertain(component, force, target, &resolver).unwrap();
        let (mix_mean, mix_cov) = mixture.mean_and_covariance().unwrap();

        // Ensemble moments about the mixture mean.  The element coordinates are one
        // global chart, so covariances of the six stored floats are comparable no
        // matter which orbit they were accumulated about.
        let offsets: Vec<Vector6<f64>> = finals
            .iter()
            .map(|f| mix_mean.elements.offset_to(f))
            .collect();
        let n = offsets.len() as f64;
        let mean_offset: Vector6<f64> = offsets.iter().sum::<Vector6<f64>>() / n;
        let mut sample_cov = Matrix6::<f64>::zeros();
        for offset in &offsets {
            let dev = offset - mean_offset;
            sample_cov += dev * dev.transpose();
        }
        sample_cov /= n;

        let scales = sample_cov.diagonal().map(f64::sqrt);
        let whitened_error = |cov: &Matrix6<f64>| -> f64 {
            let mut sum_sq = 0.0;
            for r in 0..6 {
                for c in 0..6 {
                    let e = (cov[(r, c)] - sample_cov[(r, c)]) / (scales[r] * scales[c]);
                    sum_sq += e * e;
                }
            }
            sum_sq.sqrt()
        };
        let as_matrix6 =
            |cov: &DMatrix<f64>| Matrix6::from_iterator(cov.view((0, 0), (6, 6)).iter().copied());
        let err_split = whitened_error(&as_matrix6(&mix_cov));
        let err_unsplit = whitened_error(&as_matrix6(&unsplit.cov_matrix));

        // Mean errors, in units of the ensemble spread.
        let whitened_norm = |v: &Vector6<f64>| -> f64 {
            (0..6)
                .map(|i| (v[i] / scales[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mean_err_split = whitened_norm(&mean_offset);
        let unsplit_offset = mix_mean.elements.offset_to(&unsplit.elements);
        let mean_err_unsplit = whitened_norm(&(mean_offset - unsplit_offset));

        // Containment of the clones inside the mixture's total covariance, against the
        // 6-D chi quantiles a faithful Gaussian would produce.  Computed in whitened
        // coordinates so the factorization sees a conditioned matrix.
        let mut white_cov = as_matrix6(&mix_cov);
        for r in 0..6 {
            for c in 0..6 {
                white_cov[(r, c)] /= scales[r] * scales[c];
            }
        }
        let chol = white_cov.cholesky().expect("mixture covariance is PD");
        let mut mahal: Vec<f64> = offsets
            .iter()
            .map(|offset| {
                let white = Vector6::from_iterator((0..6).map(|i| offset[i] / scales[i]));
                chol.solve(&white).dot(&white).max(0.0).sqrt()
            })
            .collect();
        mahal.sort_by(f64::total_cmp);
        let fraction_within = |d: f64| -> f64 {
            let idx = mahal.partition_point(|&m| m <= d);
            idx as f64 / n
        };

        // Occupancy: each clone assigned to the component with the highest weighted
        // density, the fractions compared against the mixture weights as a
        // total-variation distance.  The global whitening cancels in the argmax.
        let comps: Vec<(
            f64,
            EquinoctialElements,
            f64,
            nalgebra::Cholesky<f64, nalgebra::U6>,
        )> = (0..mixture.n_components())
            .map(|k| {
                let comp = mixture.component(k).unwrap();
                let mut white = as_matrix6(&comp.cov_matrix);
                for r in 0..6 {
                    for c in 0..6 {
                        white[(r, c)] /= scales[r] * scales[c];
                    }
                }
                let chol = white.cholesky().expect("component covariance is PD");
                let logdet = (0..6).map(|i| 2.0 * chol.l()[(i, i)].ln()).sum::<f64>();
                (mixture.weights[k].ln(), comp.elements.clone(), logdet, chol)
            })
            .collect();
        let mut occupancy = vec![0.0_f64; mixture.n_components()];
        for elements in &finals {
            let best = comps
                .iter()
                .enumerate()
                .map(|(k, (log_w, comp_elements, logdet, chol))| {
                    let offset = comp_elements.offset_to(elements);
                    let white = Vector6::from_iterator((0..6).map(|i| offset[i] / scales[i]));
                    let mahal_sq = chol.solve(&white).dot(&white);
                    (k, log_w - 0.5 * (logdet + mahal_sq))
                })
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(k, _)| k)
                .unwrap();
            occupancy[best] += 1.0 / n;
        }
        let tv_distance = 0.5
            * occupancy
                .iter()
                .zip(mixture.weights.iter())
                .map(|(o, w)| (o - w).abs())
                .sum::<f64>();

        println!("{label}");
        println!(
            "  components {:>5}   unresolved weight {:.3e}   clones {n_clones}",
            mixture.n_components(),
            mixture.unresolved_weight(config.split_threshold)
        );
        println!("  {:>28}  {:>10}  {:>10}", "", "mixture", "unsplit");
        println!(
            "  {:>28}  {mean_err_split:>10.4}  {mean_err_unsplit:>10.4}",
            "mean error (ensemble sigma)"
        );
        println!(
            "  {:>28}  {err_split:>10.4}  {err_unsplit:>10.4}",
            "covariance error (whitened)"
        );
        println!(
            "  containment d<2.31/3.26/3.55  {:.3}/{:.3}/{:.3}  (chi-6: 0.500/0.900/0.950)",
            fraction_within(2.313),
            fraction_within(3.263),
            fraction_within(3.548)
        );
        println!("  weight vs occupancy TV distance {tv_distance:.4}");
        (err_split, err_unsplit)
    }

    /// **The direct check on the mixture machinery**: does an adaptively split and
    /// propagated `DiffuseState` describe the distribution a nonlinearly propagated
    /// clone ensemble actually has?
    ///
    /// Everything else in this suite scores pieces - the STM against finite
    /// differences, the divergence metric against itself, the split against its own
    /// moment identities.  This scores the assembled machinery against the truth it
    /// claims to represent: the mixture's mean and total covariance against the
    /// ensemble's sample moments, clone containment against the chi quantiles a
    /// faithful description would produce, and component weights against the fraction
    /// of clones each component actually claims.
    ///
    /// Two scenarios: a quiet main-belt arc, where the single Gaussian is expected to
    /// suffice and the mixture must simply not degrade it, and a deep Earth encounter,
    /// where the unsplit Gaussian is expected to fail and the split mixture must beat
    /// it.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates a clone ensemble"]
    fn mixture_matches_propagated_ensemble() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let n_clones = 4096;

        // Quiet case: 42 Isis over 1600 days, the fitted-orbit covariance shape.
        let (isis, _, epoch) = setup();
        let sigma = well_observed_sigma(&isis);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let component = UncertainState::new(isis, cov, vec![]).unwrap();
        let config = SplitConfig::default();
        let (quiet_split, quiet_unsplit) = mixture_vs_ensemble(
            "42 Isis, 1600 day arc, density calibration:",
            &component,
            Time::<TDB>::new(epoch.jd + 1600.0),
            n_clones,
            &config,
            &force,
            &spk,
        );
        // On a quiet arc both descriptions must match the ensemble; the loose bound
        // covers sampling noise plus mild nonlinearity.
        assert!(
            quiet_split < 0.5,
            "quiet-arc mixture covariance error {quiet_split} exceeded 0.5"
        );
        assert!(
            quiet_unsplit < 0.5,
            "quiet-arc unsplit covariance error {quiet_unsplit} exceeded 0.5"
        );

        // Encounter case: 0.003 AU miss carried through closest approach, at the
        // state-estimation calibration so the cascade converges at a moderate count.
        let (neo, neo_epoch) = encounter_neo_at(&spk, &force, 0.003, false);
        let neo_sigma = well_observed_sigma(&neo);
        let mut neo_cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in neo_sigma.iter().enumerate() {
            neo_cov[(i, i)] = (s * 0.03) * (s * 0.03);
        }
        let neo_component = UncertainState::new(neo.clone(), neo_cov, vec![]).unwrap();
        let neo_config = SplitConfig {
            split_threshold: 3.0,
            ..SplitConfig::default()
        };
        let (enc_split, enc_unsplit) = mixture_vs_ensemble(
            "Constructed NEO, 0.003 AU miss, 5 days past closest approach:",
            &neo_component,
            Time::<TDB>::new(neo_epoch.jd + 205.0),
            n_clones,
            &neo_config,
            &force,
            &spk,
        );
        // Through the encounter the split must actually buy accuracy.
        assert!(
            enc_split < enc_unsplit,
            "the split mixture ({enc_split}) did not beat the unsplit Gaussian \
             ({enc_unsplit}) through the encounter"
        );

        // Stall regime: the same encounter at a covariance scale where the divergence
        // metric hits its trajectory-numerics floor and the improvement check settles
        // the cascade above the density threshold (`cascade_divergence_trace` shows
        // the stall; the settled components carry max_unresolved_divergence just
        // above 0.1, and unresolved_weight reads most of the mixture).  This scenario
        // answers whether that alarm reflects real density error: the settled mixture
        // must still match the ensemble, and beat the unsplit Gaussian.
        let mut stall_cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in neo_sigma.iter().enumerate() {
            stall_cov[(i, i)] = (s * 0.001) * (s * 0.001);
        }
        let stall_component = UncertainState::new(neo, stall_cov, vec![]).unwrap();
        let (stall_split, stall_unsplit) = mixture_vs_ensemble(
            "Same encounter, sigma scale 0.001, density calibration (stall regime):",
            &stall_component,
            Time::<TDB>::new(neo_epoch.jd + 205.0),
            n_clones,
            &SplitConfig::default(),
            &force,
            &spk,
        );
        assert!(
            stall_split < 0.5,
            "the stalled mixture does not match the ensemble: {stall_split}"
        );
        // At this scale the unsplit Gaussian is itself statistically indistinguishable
        // from the ensemble, so the mixture is required not to degrade it rather than
        // to beat it - the margin covers ensemble sampling noise.
        assert!(
            stall_split < stall_unsplit + 0.05,
            "the stalled mixture ({stall_split}) degraded the unsplit Gaussian \
             ({stall_unsplit})"
        );
    }
}
