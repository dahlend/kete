//! [`NBody`]: N-body gravity with body states from an [`Ephemeris`], with an optional
//! non-gravitational force.
//!
//! Every term of the model is a function of the object's state relative to one massive
//! body: point-mass gravity and the relativistic and oblateness corrections relative to
//! each body, a non-gravitational force relative to the Sun. `NBody` looks up each body
//! once per evaluation and hands the relative state to every term that needs it.
//!
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

use crate::ephemeris::Ephemeris;
use crate::errors::{Error, KeteResult};
use crate::forces::{GravParams, NonGravMask, ParameterizedForce};
use crate::frames::{Equatorial, SSB, SunCenter, Vector};
use crate::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3xX, Vector3};

/// NAIF id of the Sun, the body a non-gravitational force is evaluated relative to.
const SUN: i32 = 10;

/// N-body gravity [`ParameterizedForce`] with body states from the ephemeris `E`, and
/// an optional Sun-centered non-gravitational force `F`.
///
/// Borrows the ephemeris for the lifetime `'a`, so one provider (for the SPICE one, a
/// read guard on the loaded SPK files) is shared across all parallel tasks of a
/// propagation without being acquired again on each integration step.
///
/// `Center = SSB`. The non-gravitational force is written for `Center = SunCenter` and is
/// evaluated on the Sun-relative state, which the gravity loop has already computed. Its
/// free parameters are the free parameters of the model.
///
/// At a state the integrator has accepted (`exact_eval`), a position inside a massive
/// body returns [`Error::Impact`]. A body with a polyhedron field is tested against
/// its surface, a body with a spherical harmonic field against the larger of its
/// radius and the series' minimum radius, and any other body against its radius; see
/// [`GravParams::is_inside`]. Trial evaluations inside a step are not checked, so an
/// impact is reported only if an accepted state lies inside the body.
pub struct NBody<'a, E, F = NonGravMask> {
    /// Source of the massive body states, and of body-frame orientations.
    pub ephem: &'a E,
    /// Massive bodies whose gravity is included.
    pub massive_obj: Vec<GravParams>,
    /// Non-gravitational force, evaluated relative to the Sun.
    non_grav: Option<F>,
}

/// Number of distinct times an [`EphemerisCache`] holds before it is cleared.
///
/// One Radau step evaluates the force at 7 node times plus the end of the step, and
/// the next step uses none of them.
const CACHE_SLOTS: usize = 8;

/// Working storage of [`NBody`] for one integration: the massive body states,
/// `(position, velocity)` relative to the SSB, stored by the exact time they were
/// evaluated at.
///
/// Implicit integrators such as Radau evaluate the force several times at the same set
/// of times while iterating a step toward convergence, and the body states at those times
/// do not change between iterations. A lookup matches only an identical time value, so a
/// hit returns exactly the states that evaluating the ephemeris again would produce.
///
/// The integrator creates one per integration, as part of the force's
/// [`ParameterizedForce::Meta`]. The cache does not record which bodies it holds, so
/// it belongs to the one force that filled it.
#[derive(Debug, Default)]
pub struct EphemerisCache {
    entries: Vec<(Time<TDB>, Vec<(Vector3<f64>, Vector3<f64>)>)>,
}

impl EphemerisCache {
    /// Return the states stored for `time`, or compute them with `fill` and store them.
    ///
    /// # Errors
    /// Forwards the error from `fill`, in which case nothing is stored.
    fn get_or_try_insert(
        &mut self,
        time: Time<TDB>,
        fill: impl FnOnce() -> KeteResult<Vec<(Vector3<f64>, Vector3<f64>)>>,
    ) -> KeteResult<&[(Vector3<f64>, Vector3<f64>)]> {
        let idx = if let Some(idx) = self.entries.iter().position(|(t, _)| *t == time) {
            idx
        } else {
            let states = fill()?;
            if self.entries.len() == CACHE_SLOTS {
                self.entries.clear();
            }
            self.entries.push((time, states));
            self.entries.len() - 1
        };
        Ok(&self.entries[idx].1)
    }
}

impl<'a, E: Ephemeris> NBody<'a, E> {
    /// Gravity only, from the planets and the Moon, plus the registered massive bodies
    /// when `include_extended`.
    #[must_use]
    pub fn new(ephem: &'a E, include_extended: bool) -> Self {
        Self {
            ephem,
            massive_obj: massive_objects(include_extended),
            non_grav: None,
        }
    }
}

impl<'a, E, F> NBody<'a, E, F>
where
    E: Ephemeris,
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    /// Gravity plus, when given, a non-gravitational force evaluated relative to the Sun.
    #[must_use]
    pub fn with_non_grav(ephem: &'a E, include_extended: bool, non_grav: Option<F>) -> Self {
        Self {
            ephem,
            massive_obj: massive_objects(include_extended),
            non_grav,
        }
    }

    /// State of a massive body relative to the SSB.
    ///
    /// Inlined into the body loops: left to the compiler it stays out of line, which the
    /// `N-Body/Single` benchmarks show as a slower gravity-only propagation.
    #[inline(always)]
    fn body_state(
        &self,
        naif_id: i32,
        time: Time<TDB>,
    ) -> KeteResult<(Vector3<f64>, Vector3<f64>)> {
        let state = self.ephem.try_get_state_with_center(naif_id, time, 0)?;
        Ok((Vector3::from(state.pos), Vector3::from(state.vel)))
    }

    /// State of every massive body at `time` relative to the SSB, in the order of
    /// `massive_obj`, taken from `cache` when it already holds `time`.
    fn body_states<'c>(
        &self,
        cache: &'c mut EphemerisCache,
        time: Time<TDB>,
    ) -> KeteResult<&'c [(Vector3<f64>, Vector3<f64>)]> {
        cache.get_or_try_insert(time, || {
            let mut states = Vec::with_capacity(self.massive_obj.len());
            for grav_params in &self.massive_obj {
                states.push(self.body_state(grav_params.naif_id, time)?);
            }
            Ok(states)
        })
    }
}

/// A non-gravitational force is evaluated on the Sun-relative state, which the body loop
/// produces only if the Sun is in the list. `massive_obj` is public, so this is checked on
/// every evaluation rather than at construction.
fn missing_sun() -> Error {
    Error::ValueError(
        "NBody has a non-gravitational force but the Sun (NAIF id 10) is not in its list \
         of massive objects, so the force cannot be evaluated."
            .into(),
    )
}

/// The body-frame rotation a massive body needs at `rel_pos`, if any.
///
/// Only a shaped body with a frame orientation needs one, and only inside its switch
/// radius, so this costs nothing on orbits that never approach such a body. A missing
/// orientation is an error, never a silent point mass.
#[inline(always)]
fn orientation<E: Ephemeris>(
    ephem: &E,
    grav_params: &GravParams,
    rel_pos: &Vector3<f64>,
    time: Time<TDB>,
) -> KeteResult<Option<Matrix3<f64>>> {
    grav_params
        .orientation_needed(rel_pos)
        .map(|frame_id| {
            let (rot, _) = ephem.try_frame(frame_id, time)?.rotations_to_equatorial()?;
            Ok(*rot.matrix())
        })
        .transpose()
}

/// The massive bodies of the model: the planets and the Moon, plus the registered
/// asteroids when `include_extended`.
fn massive_objects(include_extended: bool) -> Vec<GravParams> {
    if include_extended {
        GravParams::selected_masses().clone()
    } else {
        GravParams::planets().to_vec()
    }
}

impl<E, F> std::fmt::Debug for NBody<'_, E, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NBody")
            .field("n_massive_obj", &self.massive_obj.len())
            .field("has_non_grav", &self.non_grav.is_some())
            .finish()
    }
}

impl<E, F> ParameterizedForce for NBody<'_, E, F>
where
    E: Ephemeris,
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    type Frame = Equatorial;
    type Center = SSB;
    /// The body-state cache, and the non-gravitational force's own storage.
    type Meta = (EphemerisCache, F::Meta);

    fn n_free_params(&self) -> usize {
        self.non_grav
            .as_ref()
            .map_or(0, ParameterizedForce::n_free_params)
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        self.non_grav
            .as_ref()
            .map_or_else(Vec::new, ParameterizedForce::free_param_names)
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        self.non_grav
            .as_ref()
            .map_or_else(Vec::new, ParameterizedForce::lower_bounds)
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<Vector<Equatorial>> {
        let (cache, non_grav_meta) = meta;
        let states = self.body_states(cache, time)?;
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let mut accel = Vector3::<f64>::zeros();
        let mut sun_relative = None;
        for (grav_params, (body_pos, body_vel)) in self.massive_obj.iter().zip(states) {
            let rel_pos = pos_v - body_pos;
            let rel_vel = vel_v - body_vel;
            let rot = orientation(self.ephem, grav_params, &rel_pos, time)?;
            grav_params.add_acceleration(&mut accel, &rel_pos, &rel_vel, rot.as_ref())?;
            if exact_eval && grav_params.is_inside(&rel_pos, rot.as_ref())? {
                return Err(Error::Impact(grav_params.naif_id, time));
            }
            if grav_params.naif_id == SUN {
                sun_relative = Some((rel_pos, rel_vel));
            }
        }
        // The non-grav is added after the loop, so the gravitational sum does not depend
        // on whether one is present.
        if let Some(non_grav) = &self.non_grav {
            let (rel_pos, rel_vel) = sun_relative.ok_or_else(missing_sun)?;
            let ng = non_grav.accel(
                time,
                &rel_pos.into(),
                &rel_vel.into(),
                free_params,
                non_grav_meta,
                exact_eval,
            )?;
            accel += Vector3::from(ng);
        }
        Ok(accel.into())
    }

    /// The derivative blocks of [`accel_and_jacobians`](Self::accel_and_jacobians).
    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        let (_, da_dr, da_dv, _) =
            self.accel_and_jacobians(time, pos, vel, free_params, meta, false)?;
        Ok((da_dr, da_dv))
    }

    /// The parameter block of [`accel_and_jacobians`](Self::accel_and_jacobians).
    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<Matrix3xX<f64>> {
        let (_, _, _, da_dp) =
            self.accel_and_jacobians(time, pos, vel, free_params, meta, false)?;
        Ok(da_dp)
    }

    /// One pass over the massive bodies: each body state feeds the acceleration and both
    /// jacobians.
    fn accel_and_jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<(
        Vector<Equatorial>,
        Matrix3<f64>,
        Matrix3<f64>,
        Matrix3xX<f64>,
    )> {
        let (cache, non_grav_meta) = meta;
        let states = self.body_states(cache, time)?;
        let pos_v: Vector3<f64> = (*pos).into();
        let vel_v: Vector3<f64> = (*vel).into();
        let mut accel = Vector3::<f64>::zeros();
        let mut da_dr = Matrix3::<f64>::zeros();
        let mut da_dv = Matrix3::<f64>::zeros();
        let mut da_dp = Matrix3xX::<f64>::zeros(0);
        let mut sun_relative = None;
        for (grav_params, (body_pos, body_vel)) in self.massive_obj.iter().zip(states) {
            let rel_pos = pos_v - body_pos;
            let rel_vel = vel_v - body_vel;
            let rot = orientation(self.ephem, grav_params, &rel_pos, time)?;
            grav_params.add_acceleration_and_jacobians(
                &mut accel,
                &mut da_dr,
                &mut da_dv,
                &rel_pos,
                &rel_vel,
                rot.as_ref(),
            )?;
            if exact_eval && grav_params.is_inside(&rel_pos, rot.as_ref())? {
                return Err(Error::Impact(grav_params.naif_id, time));
            }
            if grav_params.naif_id == SUN {
                sun_relative = Some((rel_pos, rel_vel));
            }
        }
        if let Some(non_grav) = &self.non_grav {
            let (rel_pos, rel_vel) = sun_relative.ok_or_else(missing_sun)?;
            let (ng, ng_dr, ng_dv, ng_dp) = non_grav.accel_and_jacobians(
                time,
                &rel_pos.into(),
                &rel_vel.into(),
                free_params,
                non_grav_meta,
                exact_eval,
            )?;
            accel += Vector3::from(ng);
            da_dr += ng_dr;
            da_dv += ng_dv;
            da_dp = ng_dp;
        }
        Ok((accel.into(), da_dr, da_dv, da_dp))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GMS;
    use crate::desigs::Desig;
    use crate::ephemeris::test_ephemeris::SunAndOne;
    use crate::forces::Shape;
    use crate::kepler::propagate_two_body;
    use crate::state::State;

    /// The Sun as a plain point mass, so the motion about it is exactly two-body.
    fn point_sun() -> GravParams {
        let mut sun = GravParams::new(10, GMS, 0.004_65);
        sun.relativistic = false;
        sun.shape = Shape::Point;
        sun
    }

    /// With a Sun-only ephemeris that is not SPICE, `NBody` reproduces two-body motion.
    #[test]
    fn non_spice_ephemeris_gives_two_body_motion() {
        let eph = SunAndOne {
            radius: 5.2,
            rate: 0.0015,
        };
        let mut force = NBody::new(&eph, false);
        force.massive_obj = vec![point_sun()];
        let start = State::<Equatorial, SSB> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(2_451_545.0),
            pos: Vector::<Equatorial>::new([1.1, 0.2, 0.05]),
            vel: Vector::<Equatorial>::new([-0.002, 0.016, 0.001]),
            center: SSB,
        };
        let target = Time::<TDB>::new(2_451_545.0 + 300.0);
        let n_body = start.clone().propagate_with(&force, target).unwrap();
        let sun_start = State::<Equatorial, SunCenter> {
            desig: Desig::Empty,
            epoch: start.epoch,
            pos: start.pos,
            vel: start.vel,
            center: SunCenter,
        };
        let two_body = propagate_two_body(&sun_start, target).unwrap();
        let err = (Vector3::from(n_body.pos) - Vector3::from(two_body.pos)).norm();
        assert!(err < 1e-10, "position difference {err:e} AU");
    }

    /// A second body from the ephemeris adds its point-mass pull at its own position.
    #[test]
    fn ephemeris_body_adds_its_gravity() {
        let eph = SunAndOne {
            radius: 5.2,
            rate: 0.0015,
        };
        let time = Time::<TDB>::new(2_451_600.0);
        let gm = 1e-3 * GMS;
        let mut sun_only = NBody::new(&eph, false);
        sun_only.massive_obj = vec![point_sun()];
        let mut body = GravParams::new(5, gm, 0.000_5);
        body.relativistic = false;
        body.shape = Shape::Point;
        let mut both = NBody::new(&eph, false);
        both.massive_obj = vec![point_sun(), body];
        let pos = Vector::<Equatorial>::new([4.0, 3.0, 0.2]);
        let vel = Vector::<Equatorial>::new([0.0, 0.0, 0.0]);
        let a0 = Vector3::from(
            sun_only
                .accel(time, &pos, &vel, &[], &mut Default::default(), false)
                .unwrap(),
        );
        let a1 = Vector3::from(
            both.accel(time, &pos, &vel, &[], &mut Default::default(), false)
                .unwrap(),
        );
        let body = eph.try_get_state_with_center(5, time, 0).unwrap();
        let rel = Vector3::from(pos) - Vector3::from(body.pos);
        let expect = -rel * gm / rel.norm().powi(3);
        // The body's pull is ~1% of the Sun's here, so the difference carries the Sun's
        // rounding at ~1e-14 of the body term.
        assert!(((a1 - a0) - expect).norm() < 1e-12 * expect.norm());
    }

    fn states(value: f64) -> Vec<(Vector3<f64>, Vector3<f64>)> {
        vec![(Vector3::repeat(value), Vector3::repeat(-value))]
    }

    #[test]
    fn a_repeated_time_returns_the_stored_states() {
        let mut cache = EphemerisCache::default();
        let _ = cache
            .get_or_try_insert(Time::new(1.0), || Ok(states(1.0)))
            .unwrap();
        let hit = cache
            .get_or_try_insert(Time::new(1.0), || panic!("fill must not run on a hit"))
            .unwrap();
        assert_eq!(hit, states(1.0).as_slice());
    }

    #[test]
    fn times_past_the_limit_are_still_correct() {
        let mut cache = EphemerisCache::default();
        for idx in 0..(3 * CACHE_SLOTS) {
            let t = f64::from(u32::try_from(idx).unwrap());
            let got = cache
                .get_or_try_insert(Time::new(t), || Ok(states(t)))
                .unwrap();
            assert_eq!(got, states(t).as_slice());
        }
    }

    #[test]
    fn a_failed_fill_is_not_stored() {
        let mut cache = EphemerisCache::default();
        let result = cache.get_or_try_insert(Time::new(2.0), || {
            Err(Error::ValueError("lookup failed".into()))
        });
        assert!(result.is_err());
        let got = cache
            .get_or_try_insert(Time::new(2.0), || Ok(states(2.0)))
            .unwrap();
        assert_eq!(got, states(2.0).as_slice());
    }
}
