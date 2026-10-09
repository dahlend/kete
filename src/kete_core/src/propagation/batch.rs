// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Batched N-body propagation over a packed `(planets | objects)` vector.
//!
//! [`vec_accel`] is the ephemeris-free ODE function: planets and objects are
//! integrated together as a single packed `DVector`, with massive bodies
//! occupying the leading `N` slots and test particles following.
//! [`propagate_n_body_vec`] is the high-level entry point that builds the
//! initial vector from an input list of states and returns the final
//! states at `jd_final`.

use crate::desigs::Desig;
use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::forces::{GravParams, ParameterMask, ParameterizedForce};
use crate::frames::{Equatorial, SunCenter};
use crate::integrators::{RadauDense, RadauIntegrator};
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::DVector;

/// Propagate objects with N-body mechanics and no ephemeris queries during the
/// integration.
///
/// The function integrates the Sun and the planet system barycenters together
/// with the objects. The Earth and the Moon enter as the Earth-Moon
/// barycenter. Thus the planet states can differ slightly from the SPK states.
///
/// `states` are the objects, all at one epoch. `jd_final` is the end time.
/// `planet_states` holds Sun-centered states of the bodies in
/// [`GravParams::simplified_planets`], in that order, at the epoch of
/// `states`. If it is `None`, the function reads these states from `ephem`.
/// `non_gravs` holds one optional non-gravitational force per object.
///
/// The function returns the final object states and the final planet states,
/// both Sun-centered at `jd_final`.
///
/// # Errors
/// - `Error::ValueError` if `states` is empty, if `non_gravs.len()` differs
///   from `states.len()`, or if the epochs of `states` differ.
/// - `Error::ValueError` if `planet_states` has the wrong length, or if its
///   first state has an epoch different from `states`.
/// - The error of an ephemeris query for the planet states when `planet_states`
///   is `None`.
/// - `Error::Impact` if an object impacts a massive body.
/// - `Error::Convergence` if the integrator does not converge.
///
/// # Panics
/// Does not panic. Each `unwrap` on `states.first()` and
/// `planet_states.first()` follows a length check that guarantees a first
/// element.
pub fn propagate_n_body_vec<E, F>(
    ephem: &E,
    states: Vec<State<Equatorial, SunCenter>>,
    jd_final: Time<TDB>,
    planet_states: Option<Vec<State<Equatorial>>>,
    non_gravs: Vec<Option<ParameterMask<F>>>,
) -> KeteResult<(Vec<State<Equatorial>>, Vec<State<Equatorial>>)>
where
    E: Ephemeris,
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    if states.is_empty() {
        Err(Error::ValueError(
            "State vector is empty, propagation cannot continue".into(),
        ))?;
    }

    if non_gravs.len() != states.len() {
        Err(Error::ValueError(
            "Number of non-grav models doesnt match the number of provided objects.".into(),
        ))?;
    }

    let jd_init = states.first().unwrap().epoch;

    let mut pos: Vec<f64> = Vec::new();
    let mut vel: Vec<f64> = Vec::new();
    let mut desigs: Vec<Desig> = Vec::new();
    let planets = GravParams::simplified_planets();
    let planet_states = if let Some(ps) = planet_states {
        ps
    } else {
        let mut planet_states = Vec::new();
        for obj in planets {
            let planet = ephem.try_get_state_with_center(obj.naif_id, jd_init, 10)?;
            planet_states.push(planet);
        }
        planet_states
    };

    if planet_states.len() != planets.len() {
        Err(Error::ValueError(
            "Input planet states must contain the correct number of states.".into(),
        ))?;
    }
    if !planet_states.first().unwrap().epoch.same_instant(&jd_init) {
        Err(Error::ValueError(
            "Planet states JD must match JD of input state.".into(),
        ))?;
    }
    for planet_state in planet_states {
        pos.append(&mut planet_state.pos.into());
        vel.append(&mut planet_state.vel.into());
        desigs.push(planet_state.desig);
    }

    for state in states {
        if !jd_init.same_instant(&state.epoch) {
            Err(Error::ValueError(
                "All input states must have the same JD".into(),
            ))?;
        }
        pos.append(&mut state.pos.into());
        vel.append(&mut state.vel.into());
        desigs.push(state.desig);
    }

    let (pos, vel) = integrate_packed(
        planets,
        non_gravs,
        DVector::from(pos),
        DVector::from(vel),
        jd_init,
        jd_final,
        None,
    )?;
    let sun_pos = pos.fixed_rows::<3>(0);
    let sun_vel = vel.fixed_rows::<3>(0);
    let mut all_states: Vec<State<_>> = Vec::new();
    for (idx, desig) in desigs.into_iter().enumerate() {
        let pos = pos.fixed_rows::<3>(idx * 3) - sun_pos;
        let vel = vel.fixed_rows::<3>(idx * 3) - sun_vel;
        let state = State::new(desig, jd_final, pos, vel, 10);
        all_states.push(state);
    }
    let final_states = all_states.split_off(planets.len());
    Ok((final_states, all_states))
}

/// Integrate a packed `(massive bodies | objects)` vector from `jd_init` to `jd_final`.
///
/// `pos` and `vel` hold absolute states (all relative to one inertial origin) of the
/// bodies of `massive_obj`, followed by the objects that `non_gravs` describes; the
/// result is in the same form. When the Earth (399) and the Moon (301) are both massive
/// bodies, the Moon is integrated relative to the Earth, which keeps its coordinates
/// and the integrator's error control on the scale of its orbit. `dense`, when given,
/// receives the dense output in that internal form.
///
/// # Errors
/// `Error::Impact` on an impact, and `Error::Convergence` if the integrator does not
/// converge.
pub(super) fn integrate_packed<F>(
    massive_obj: &[GravParams],
    non_gravs: Vec<Option<ParameterMask<F>>>,
    mut pos: DVector<f64>,
    mut vel: DVector<f64>,
    jd_init: Time<TDB>,
    jd_final: Time<TDB>,
    dense: Option<&mut RadauDense>,
) -> KeteResult<(DVector<f64>, DVector<f64>)>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    use std::ops::{AddAssign, SubAssign};

    let slot = |naif: i32| massive_obj.iter().position(|p| p.naif_id == naif);
    let moon_earth = slot(301).zip(slot(399));
    if let Some((moon, earth)) = moon_earth {
        for k in 0..3 {
            pos[moon * 3 + k] -= pos[earth * 3 + k];
            vel[moon * 3 + k] -= vel[earth * 3 + k];
        }
    }
    // Forces are evaluated on absolute states; the Moon's acceleration is returned
    // relative to the Earth's.
    let accel = |time: Time<TDB>,
                 pos: &DVector<f64>,
                 vel: &DVector<f64>,
                 meta: &mut AccelVecMeta<'_, F>,
                 exact_eval: bool| {
        let Some((moon, earth)) = moon_earth else {
            return vec_accel(time, pos, vel, meta, exact_eval);
        };
        let (mut pos_abs, mut vel_abs) = (pos.clone(), vel.clone());
        pos_abs
            .fixed_rows_mut::<3>(moon * 3)
            .add_assign(pos.fixed_rows::<3>(earth * 3));
        vel_abs
            .fixed_rows_mut::<3>(moon * 3)
            .add_assign(vel.fixed_rows::<3>(earth * 3));
        let mut accel = vec_accel(time, &pos_abs, &vel_abs, meta, exact_eval)?;
        let earth_accel = accel.fixed_rows::<3>(earth * 3).clone_owned();
        accel.fixed_rows_mut::<3>(moon * 3).sub_assign(earth_accel);
        Ok(accel)
    };
    let meta = AccelVecMeta {
        non_gravs,
        massive_obj,
    };
    let (mut pos, mut vel, _) =
        RadauIntegrator::integrate(&accel, pos, vel, jd_init, jd_final, meta, None, dense)?;
    if let Some((moon, earth)) = moon_earth {
        for k in 0..3 {
            pos[moon * 3 + k] += pos[earth * 3 + k];
            vel[moon * 3 + k] += vel[earth * 3 + k];
        }
    }
    Ok((pos, vel))
}

/// Metadata for [`vec_accel`]: the ephemeris-free bulk N-body ODE function.
///
/// Generic over the inner non-gravitational force `F`. Callers commit to
/// one concrete force type per batch.
struct AccelVecMeta<'a, F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>> {
    /// Per-object non-gravitational force with every parameter fixed, or
    /// `None`.
    pub non_gravs: Vec<Option<ParameterMask<F>>>,
    /// Massive bodies providing gravity, same order as the leading slots
    /// in the pos/vel vectors.
    pub massive_obj: &'a [GravParams],
}

impl<F> std::fmt::Debug for AccelVecMeta<'_, F>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelVecMeta")
            .field("n_objects", &self.non_gravs.len())
            .field("n_massive", &self.massive_obj.len())
            .finish()
    }
}

/// Compute the accel on a packed `(planets | objects)` pos/vel vector.
///
/// The first `N` objects in the vector are the massive bodies listed in
/// `meta.massive_obj` (in order). Objects beyond index `N` are test
/// particles subject to gravity from all massive bodies plus optional
/// non-gravitational forces.
///
/// # Errors
/// Fails on an impact.
fn vec_accel<D, F>(
    time: Time<TDB>,
    pos: &nalgebra::OVector<f64, D>,
    vel: &nalgebra::OVector<f64, D>,
    meta: &mut AccelVecMeta<'_, F>,
    exact_eval: bool,
) -> KeteResult<nalgebra::OVector<f64, D>>
where
    D: nalgebra::Dim,
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
    nalgebra::DefaultAllocator:
        nalgebra::allocator::Allocator<D> + nalgebra::allocator::Allocator<D, nalgebra::U2>,
{
    use crate::frames::Vector;
    use nalgebra::{U1, Vector3};
    use std::ops::AddAssign;

    let n_objects = pos.len() / 3;
    let n_massive = meta.massive_obj.len();
    let (dim, _) = pos.shape_generic();
    let mut accel = nalgebra::OVector::<f64, D>::zeros_generic(dim, U1);
    let mut accel_working = Vector3::zeros();

    for idx in 0..n_objects {
        let pos_idx = pos.fixed_rows::<3>(idx * 3);
        let vel_idx = vel.fixed_rows::<3>(idx * 3);
        for (idy, grav_params) in meta.massive_obj.iter().enumerate() {
            if idx == idy {
                continue;
            }
            accel_working.fill(0.0);
            let radius = grav_params.radius;
            let pos_idy = pos.fixed_rows::<3>(idy * 3);
            let vel_idy = vel.fixed_rows::<3>(idy * 3);
            let rel_pos = pos_idx - pos_idy;
            let rel_vel = vel_idx - vel_idy;
            if exact_eval && (rel_pos.norm() as f32 <= radius) {
                Err(Error::Impact(grav_params.naif_id, time))?;
            }
            // No orientation is passed: the massive bodies here are the simplified
            // planets, point masses or oblate, which need none. A body with a
            // frame-oriented shaped field would give an error inside its switch
            // radius, never a silent point mass.
            grav_params.add_acceleration(&mut accel_working, &rel_pos, &rel_vel, None)?;
            if (grav_params.naif_id == 10)
                && (idx >= n_massive)
                && let Some(frozen) = &meta.non_gravs[idx - n_massive]
            {
                let pos_vec = Vector::<Equatorial>::new([rel_pos[0], rel_pos[1], rel_pos[2]]);
                let vel_vec = Vector::<Equatorial>::new([rel_vel[0], rel_vel[1], rel_vel[2]]);
                let ng_accel = frozen.accel(
                    time,
                    &pos_vec,
                    &vel_vec,
                    &[],
                    &mut F::Meta::default(),
                    exact_eval,
                )?;
                let ng_v3: Vector3<f64> = ng_accel.into();
                accel_working += ng_v3;
            }
            accel.fixed_rows_mut::<3>(idx * 3).add_assign(accel_working);
        }
    }
    Ok(accel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS;
    use crate::ephemeris::test_ephemeris::SunAndOne;
    use crate::frames::Vector;
    use crate::propagation::NBody;

    /// The packed batch acceleration of a test particle equals the [`NBody`]
    /// acceleration with the same massive bodies at the same states.
    #[test]
    fn batch_accel_matches_n_body() {
        let eph = SunAndOne {
            radius: 5.2,
            rate: 0.0015,
        };
        let jd = Time::<TDB>::new(2_451_545.0);
        let massive = [
            GravParams::new(10, GMS, 0.004_65),
            GravParams::new(5, 1e-3 * GMS, 0.000_5),
        ];
        let mut pos: Vec<f64> = Vec::new();
        let mut vel: Vec<f64> = Vec::new();
        for obj in &massive {
            let s = eph.try_get_state_with_center(obj.naif_id, jd, 0).unwrap();
            pos.append(&mut s.pos.into());
            vel.append(&mut s.vel.into());
        }
        pos.extend([0.0, 0.0, 0.5]);
        vel.extend([0.0, 0.0, 1.0]);

        let accel = vec_accel(
            jd,
            &DVector::from(pos),
            &DVector::from(vel),
            &mut AccelVecMeta::<crate::forces::JplCometNonGrav> {
                non_gravs: vec![None],
                massive_obj: &massive,
            },
            false,
        )
        .unwrap();
        let mut force = NBody::new(&eph, false);
        force.massive_obj = massive.to_vec();
        let accel2 = force
            .accel(
                jd,
                &Vector::<Equatorial>::new([0.0, 0.0, 0.5]),
                &Vector::<Equatorial>::new([0.0, 0.0, 1.0]),
                &[],
                &mut Default::default(),
                false,
            )
            .unwrap();
        for i in 0..3 {
            assert!((accel[massive.len() * 3 + i] - accel2[i]).abs() < 1e-15);
        }
    }

    /// Integrating the Moon relative to the Earth gives the same motion as integrating
    /// it absolutely, which happens when the same bodies carry other ids.
    #[test]
    fn moon_relative_to_earth_matches_absolute() {
        let jd = Time::<TDB>::new(2_451_545.0);
        let (gm_earth, gm_moon) = (3.0e-6 * GMS, 3.7e-8 * GMS);
        let v_earth = GMS.sqrt();
        let v_moon = (gm_earth / 0.00257).sqrt();
        let pos = DVector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.00257, 0.0, 0.0]);
        let vel = DVector::from_vec(vec![
            0.0,
            0.0,
            0.0,
            0.0,
            v_earth,
            0.0,
            0.0,
            v_earth + v_moon,
            0.0,
        ]);
        let run = |earth_id: i32, moon_id: i32| {
            let mut earth = GravParams::new(399, gm_earth, 4.3e-5);
            let mut moon = GravParams::new(301, gm_moon, 1.2e-5);
            (earth.naif_id, moon.naif_id) = (earth_id, moon_id);
            let massive = [GravParams::new(10, GMS, 0.004_65), earth, moon];
            let (pos, _) = integrate_packed::<crate::forces::JplCometNonGrav>(
                &massive,
                Vec::new(),
                pos.clone(),
                vel.clone(),
                jd,
                jd + 100.0,
                None,
            )
            .unwrap();
            pos
        };
        let (relative, absolute) = (run(399, 301), run(398, 302));
        let moon_about_earth =
            |p: &DVector<f64>| p.fixed_rows::<3>(6).clone_owned() - p.fixed_rows::<3>(3);
        let diff = (moon_about_earth(&relative) - moon_about_earth(&absolute)).norm();
        println!("moon_relative_to_earth_matches_absolute: {diff:e} au apart");
        assert!(
            diff < 1e-10,
            "the Moon about the Earth differs by {diff} au"
        );
    }
}
