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
    use kete_core::elements::EquinoctialElements;
    use kete_core::forces::JplCometNonGrav;

    use kete_core::prelude::UncertainState;
    use kete_core::state::{
        DEFAULT_STEP_DAYS, DiffuseState, SplitConfig, propagate_diffuse_state,
        propagate_elements_with_sensitivity, propagate_state, step_diffuse_state,
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

    /// Per-leg nonlinearity on a quiet main-belt arc, against the threshold above which
    /// the adaptive mixture splits.
    ///
    /// The whole arc is taken as a single leg, which is the pessimistic reading: the
    /// controller cuts an arc onto a grid and measures each leg over a much shorter span,
    /// so an arc that reads below threshold in one shot reads below it there too.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, propagates probes"]
    fn leg_nonlinearity_on_a_quiet_arc() {
        let (elem, _sun_ssb, epoch) = setup();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let resolver = |time: Time<TDB>| {
            let sun = spk.try_get_state_with_center::<Equatorial>(10, time, 0)?;
            Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
        };
        let config = SplitConfig::default();

        let sigma = well_observed_sigma(&elem);
        let mut cov = DMatrix::<f64>::zeros(6, 6);
        for (i, s) in sigma.iter().enumerate() {
            cov[(i, i)] = s * s;
        }
        let component = UncertainState::new(elem.clone(), cov, vec![]).unwrap();
        let mixture = DiffuseState::from_uncertain(component);
        // Measured, not acted on: the threshold is set out of reach so the leg reports the
        // nonlinearity it found rather than splitting to remove it.
        let measure_only = SplitConfig {
            split_threshold: 1e30,
            max_components: 1,
        };

        println!(
            "42 Isis, main belt, period {:.0} d. Split threshold {:.1}.",
            elem.orbital_period(),
            config.split_threshold
        );
        println!(
            "{:>9}  {:>8}  {:>11}  {:>13}  {:>8}",
            "arc (d)", "orbits", "eta", "residual (m)", "verdict"
        );

        for arc in [400.0_f64, 1600.0, 6400.0, 12800.0] {
            let target = Time::<TDB>::new(epoch.jd + arc);
            let (stepped, _) =
                step_diffuse_state(&mixture, &force, target, &measure_only, &resolver).unwrap();
            let eta = stepped.max_eta().unwrap();
            println!(
                "{arc:>9.0}  {:>8.2}  {:>11.3e}  {:>13.3e}  {:>8}",
                arc / elem.orbital_period(),
                eta,
                stepped.residual_meters().unwrap(),
                if eta < config.split_threshold {
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
            let (out, _) = propagate_diffuse_state(
                &mixture,
                &force,
                Time::<TDB>::new(epoch.jd + 205.0),
                &config,
                DEFAULT_STEP_DAYS,
                &resolver,
            )
            .unwrap();
            println!(
                "cap {cap:>4} -> {:>4} components, unresolved weight {:.3}",
                out.n_components(),
                out.weight_above_eta(config.split_threshold).unwrap_or(0.0)
            );
            assert!(
                out.n_components() <= cap,
                "cap {cap} exceeded: got {}",
                out.n_components()
            );
        }
    }

    /// **Does splitting actually reduce the nonlinearity it was chosen to remove?**
    ///
    /// The controller assumes it does: a component whose linear prediction is bad is
    /// subdivided so each child spans a narrower region where the flow is more linear.
    /// The stopped-helping termination is the test of that assumption at run time, and
    /// this measures the same thing directly, which is what distinguishes "this encounter
    /// is genuinely unresolvable" from "the split direction is useless".
    ///
    /// `eta` should fall roughly linearly with the spread: the residual grows as the
    /// square of the perturbation while the normalizing sigma grows linearly, so a
    /// three-way split ought to buy a factor of order two.
    ///
    /// Run with `cargo test -- --ignored --nocapture`.
    #[test]
    #[ignore = "measurement, prints a table"]
    fn does_splitting_reduce_eta() {
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
        let base = DiffuseState::from_uncertain(UncertainState::new(neo, cov, vec![]).unwrap());
        // The same leg taken twice: once with the threshold out of reach, so the parent
        // reports what it found, and once with it at zero and the budget at exactly one
        // split, so the children report what a single split bought. The controller picks
        // the direction either way, which is the thing being measured.
        let unsplit = SplitConfig {
            split_threshold: 1e30,
            max_components: 1,
        };
        let one_split = SplitConfig {
            split_threshold: 0.0,
            max_components: 3,
        };

        println!(
            "{:>8}  {:>12}  {:>12}  {:>8}",
            "arc (d)", "parent eta", "worst child", "ratio"
        );
        for arc in [195.0_f64, 200.0, 205.0, 400.0] {
            let target = Time::<TDB>::new(neo_epoch.jd + arc);
            let Ok((parent, _)) = step_diffuse_state(&base, &force, target, &unsplit, &resolver)
            else {
                println!("{arc:>8.0}  parent propagation failed");
                continue;
            };
            let eta = parent.max_eta().unwrap();

            let Ok((children, _)) =
                step_diffuse_state(&base, &force, target, &one_split, &resolver)
            else {
                println!("{arc:>8.0}  {eta:>12.3e}  split failed");
                continue;
            };
            if children.n_components() == 1 {
                println!("{arc:>8.0}  {eta:>12.3e}  no direction carried width");
                continue;
            }
            let worst_child = children.max_eta().unwrap();
            println!(
                "{arc:>8.0}  {eta:>12.3e}  {worst_child:>12.3e}  {:>8.2}",
                eta / worst_child
            );
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
        let (mixture, _) = propagate_diffuse_state(
            &DiffuseState::from_uncertain(component.clone()),
            force,
            target,
            config,
            DEFAULT_STEP_DAYS,
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
            mixture
                .weight_above_eta(config.split_threshold)
                .unwrap_or(0.0)
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

        // Noise-floor regime: the same encounter at a covariance scale so small that
        // the probe residuals sit at the propagator's own resolution.  This scenario
        // answers whether readings at that scale reflect real density error: the
        // mixture must still match the ensemble, and be no worse than the unsplit
        // Gaussian.
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
