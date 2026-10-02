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
use crate::integrators::RadauIntegrator;
use crate::state::State;
use crate::time::{TDB, Time};
use nalgebra::DVector;

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

    #[allow(clippy::missing_panics_doc, reason = "not possible by construction.")]
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

    let meta = AccelVecMeta {
        non_gravs,
        massive_obj: planets,
    };

    let (pos, vel, _) = {
        RadauIntegrator::integrate(
            &vec_accel,
            DVector::from(pos),
            DVector::from(vel),
            jd_init,
            jd_final,
            meta,
            None,
        )?
    };
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
}
