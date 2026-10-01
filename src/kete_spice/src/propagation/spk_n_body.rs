//! [`SpkNBody`]: SPK-based N-body gravity, with an optional non-gravitational force.
//!
//! Every term of the model is a function of the object's state relative to one massive
//! body: point-mass gravity and the relativistic and oblateness corrections relative to
//! each body, a non-gravitational force relative to the Sun. `SpkNBody` looks up each
//! body once per evaluation and hands the relative state to every term that needs it.
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

use kete_core::errors::{Error, KeteResult};
use kete_core::forces::{GravParams, NonGravMask, ParameterizedForce};
use kete_core::frames::{Equatorial, SSB, SunCenter, Vector};
use kete_core::time::{TDB, Time};
use nalgebra::{Matrix3, Matrix3xX, Vector3};

use crate::ck::LOADED_CK;
use crate::frame_ext::rotations_to_equatorial_full;
use crate::spk::SpkCollection;

/// NAIF id of the Sun, the body a non-gravitational force is evaluated relative to.
const SUN: i32 = 10;

/// SPK-based N-body gravity [`ParameterizedForce`], with an optional Sun-centered
/// non-gravitational force `F`.
///
/// Borrows a `SpkCollection` for the lifetime `'a`; callers hold a read guard
/// from `LOADED_SPK.try_read()` for the duration of the propagation, which is
/// shared across all parallel tasks without re-acquiring the lock on each
/// integration step.
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
pub struct SpkNBody<'a, F = NonGravMask> {
    /// Borrowed reference to the loaded SPK collection.
    pub spk: &'a SpkCollection,
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

/// Working storage of [`SpkNBody`] for one integration: the massive body states,
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

impl<'a> SpkNBody<'a> {
    /// Gravity only, with the given SPK collection and massive body list.
    #[must_use]
    pub fn new(spk: &'a SpkCollection, include_extended: bool) -> Self {
        Self {
            spk,
            massive_obj: massive_objects(include_extended),
            non_grav: None,
        }
    }
}

impl<'a, F> SpkNBody<'a, F>
where
    F: ParameterizedForce<Frame = Equatorial, Center = SunCenter>,
{
    /// Gravity plus, when given, a non-gravitational force evaluated relative to the Sun.
    #[must_use]
    pub fn with_non_grav(
        spk: &'a SpkCollection,
        include_extended: bool,
        non_grav: Option<F>,
    ) -> Self {
        Self {
            spk,
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
        let state = self
            .spk
            .try_get_state_with_center::<Equatorial>(naif_id, time, 0)?;
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
        "SpkNBody has a non-gravitational force but the Sun (NAIF id 10) is not in its list \
         of massive objects, so the force cannot be evaluated."
            .into(),
    )
}

/// Rotation from a CK body frame to equatorial axes at `time`.
///
/// Used only for a polyhedron body with a CK orientation, and only inside its switch
/// radius, so it costs nothing on orbits that never approach such a body. The CK must
/// hold pointing at `time`; a gap is an error, never a silent point mass.
fn ck_body_to_equatorial(frame_id: i32, time: Time<TDB>) -> KeteResult<Matrix3<f64>> {
    let (frame_time, frame) = LOADED_CK.try_read()?.try_get_frame(time, frame_id)?;
    if (frame_time - time).elapsed.abs() > 1e-8 {
        return Err(Error::Bounds(format!(
            "CK frame {frame_id} has no pointing at JD {}.",
            time.jd()
        )));
    }
    let (rot, _) = rotations_to_equatorial_full(&frame)?;
    Ok(*rot.matrix())
}

/// The body-frame rotation a massive body needs at `rel_pos`, if any.
#[inline(always)]
fn orientation(
    grav_params: &GravParams,
    rel_pos: &Vector3<f64>,
    time: Time<TDB>,
) -> KeteResult<Option<Matrix3<f64>>> {
    grav_params
        .orientation_needed(rel_pos)
        .map(|frame_id| ck_body_to_equatorial(frame_id, time))
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

impl<F> std::fmt::Debug for SpkNBody<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpkNBody")
            .field("n_massive_obj", &self.massive_obj.len())
            .field("has_non_grav", &self.non_grav.is_some())
            .finish()
    }
}

impl<F> ParameterizedForce for SpkNBody<'_, F>
where
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
            let rot = orientation(grav_params, &rel_pos, time)?;
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
            let rot = orientation(grav_params, &rel_pos, time)?;
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
    use crate::spk::LOADED_SPK;
    use kete_core::desigs::Desig;
    use kete_core::state::State;

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

    /// Near a polyhedron body, `SpkNBody` adds exactly the rotated polyhedron field to
    /// the other bodies' gravity, through all three evaluation paths.
    #[test]
    fn spk_n_body_polyhedron_body() {
        use kete_core::forces::Orientation;
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let rot = *nalgebra::Rotation3::from_euler_angles(0.4, -0.9, 1.7).matrix();
        let body = polyhedron_42(Orientation::Fixed(rot));
        let base = SpkNBody::new(&spk, false);
        let mut with_body = SpkNBody::new(&spk, false);
        with_body.massive_obj.push(body.clone());

        let (body_pos, _) = base.body_state(20_000_042, time).unwrap();
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
    fn spk_n_body_polyhedron_ck_without_kernel() {
        use kete_core::forces::Orientation;
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let mut force = SpkNBody::new(&spk, false);
        force.massive_obj.push(polyhedron_42(Orientation::Ck {
            frame_id: -987_654_000,
        }));
        let (body_pos, _) = force.body_state(20_000_042, time).unwrap();
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
    fn spk_n_body_jacobian_pos_matches_finite_difference() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
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

    /// With analytical jacobians, the variational STM matches the
    /// existing `compute_state_transition` to working precision.
    #[test]
    fn spk_n_body_analytical_stm_matches_compute_state_transition() {
        use kete_core::state::propagate_with_stm;

        crate::test_data::ensure_test_spk();

        let start = State::<Equatorial, SSB> {
            desig: Desig::Empty,
            epoch: Time::<TDB>::new(2_451_545.0),
            pos: Vector::<Equatorial>::new([0.5, 1.0, 0.1]),
            vel: Vector::<Equatorial>::new([-0.012, 0.008, 0.001]),
            center: SSB,
        };
        let target = Time::<TDB>::new(2_451_545.0 + 5.0);

        let (legacy_state, legacy_stm) = crate::propagation::compute_state_transition::<
            kete_core::forces::JplCometNonGrav,
        >(&start, target, false, None)
        .unwrap();

        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::new(&spk, false);
        let pos_init: Vector3<f64> = start.pos.into();
        let vel_init: Vector3<f64> = start.vel.into();
        let (pos_new, vel_new, new_stm) =
            propagate_with_stm(&force, pos_init, vel_init, &[], start.epoch, target).unwrap();

        let legacy_pos: Vector3<f64> = legacy_state.pos.into();
        let legacy_vel: Vector3<f64> = legacy_state.vel.into();
        assert!((pos_new - legacy_pos).norm() < 1e-12);
        assert!((vel_new - legacy_vel).norm() < 1e-12);

        let max_diff = (legacy_stm - new_stm).abs().max();
        assert!(
            max_diff < 1e-10,
            "STM max element diff = {max_diff} (analytical jacobian)"
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

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let beta = 0.001;
        let ((pos, vel), (pos_sun, vel_sun)) = ssb_and_sun_relative(&spk, time);

        let gravity = SpkNBody::new(&spk, false);
        let with_dust = SpkNBody::with_non_grav(&spk, false, Some(DustNonGrav));

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

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let params = [1.0e-8, 2.0e-9, -3.0e-10];
        let ((pos, vel), _) = ssb_and_sun_relative(&spk, time);

        let force = SpkNBody::with_non_grav(&spk, true, Some(JplCometNonGrav::standard_comet()));
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

        let gravity = SpkNBody::new(&spk, true);
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

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let later = Time::<TDB>::new(2_451_545.5);
        let params = [1.0e-8, 2.0e-9, -3.0e-10];
        let ((pos, vel), _) = ssb_and_sun_relative(&spk, time);

        let force = SpkNBody::with_non_grav(&spk, true, Some(JplCometNonGrav::standard_comet()));
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

    /// A state aimed at the Earth's center ends in an impact with the Earth, and a
    /// state that misses it does not.
    #[test]
    fn an_impact_is_reported() {
        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
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
        let result = aimed.propagate_with(&SpkNBody::new(&spk, false), time + 1.0);
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
            miss.propagate_with(&SpkNBody::new(&spk, false), time + 1.0)
                .is_ok()
        );
    }

    /// Free parameters, names, and lower bounds of the model are those of the non-grav.
    #[test]
    fn free_parameters_are_those_of_the_non_grav() {
        use kete_core::forces::DustNonGrav;

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let gravity = SpkNBody::new(&spk, false);
        assert_eq!(gravity.n_free_params(), 0);
        assert!(gravity.lower_bounds().is_empty());

        let force = SpkNBody::with_non_grav(&spk, false, Some(DustNonGrav));
        assert_eq!(force.n_free_params(), 1);
        assert_eq!(force.free_param_names(), vec!["beta"]);
        assert_eq!(force.lower_bounds(), vec![Some(0.0)]);
    }

    /// A non-grav cannot be evaluated without the Sun in the body list; that is an error
    /// rather than a silently dropped force.
    #[test]
    fn non_grav_without_the_sun_is_an_error() {
        use kete_core::forces::DustNonGrav;

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let time = Time::<TDB>::new(2_451_545.0);
        let ((pos, vel), _) = ssb_and_sun_relative(&spk, time);

        let mut force = SpkNBody::with_non_grav(&spk, false, Some(DustNonGrav));
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

        crate::test_data::ensure_test_spk();
        let spk = LOADED_SPK.try_read().unwrap();
        let force = SpkNBody::with_non_grav(&spk, false, Some(JplCometNonGrav::standard_comet()));
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
}
