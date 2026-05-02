//! Body-centric propagation in the proximity regime of an extended body.
//!
//! [`propagate_near_body`] wraps the existing [`RadauIntegrator`] with
//! a force function that combines the polyhedron-gravity self-attraction
//! of an [`ExtendedBody`] with tidal third-body perturbations from
//! caller-specified perturber NAIF IDs (queried from loaded SPK
//! kernels at every integration evaluation).
//!
//! All integration is performed in body-centric coordinates: the
//! particle position is in the body's body-fixed frame, expressed in
//! body-natural length units (`AU / body.units.length_au`).  The time
//! axis remains TDB Julian Day to match the underlying integrator;
//! only spatial coordinates are scaled.  Body-natural lengths keep
//! near-body trajectories O(1) instead of O(1e-9), which is the
//! dominant numerical conditioning win for small-body proximity work.
//!
//! Surface contact (the field point coinciding with the body surface)
//! produces an [`Error::Impact`] tagged with the body's NAIF ID and
//! the current TDB time, mirroring the convention used by
//! [`crate::propagation::spk_accel`].

use kete_core::desigs::Desig;
use kete_core::errors::{Error, KeteResult};
use kete_core::forces::NonGravModel;
use kete_core::frames::{CenterBody, Equatorial, InertialFrame, Vector};
use kete_core::integrators::RadauIntegrator;
use kete_core::shape::{ExtendedBody, small_body_accel};
use kete_core::state::State;
use kete_core::time::{TDB, Time};
use nalgebra::{OVector, U3, Vector3};

use crate::spk::SpkCollection;

/// Particle state in the body-centric frame for [`propagate_near_body`].
///
/// `pos` and `vel` are expressed in the body's body-fixed frame in
/// body-natural length units (`AU / body.units.length_au`) and
/// `body_length` / day, respectively.  `epoch` is TDB.
#[derive(Debug, Clone)]
pub struct BodyRelativeState {
    /// Optional designation carried through propagation and
    /// regime-conversion helpers.  `None` means anonymous.
    pub desig: Option<Desig>,
    /// Position in body-fixed coordinates, body-natural length units.
    pub pos: Vector3<f64>,
    /// Velocity in body-fixed coordinates, `body_length` / day.
    pub vel: Vector3<f64>,
    /// TDB epoch of the state.
    pub epoch: Time<TDB>,
}

/// A perturbing body referenced by NAIF ID and gravitational parameter.
#[derive(Debug, Clone, Copy)]
pub struct Perturber {
    /// NAIF ID of the perturbing body (queried from loaded SPK kernels).
    pub naif_id: i32,
    /// Gravitational parameter `G * M` in AU^3 / day^2 (solar units).
    pub gm_solar: f64,
}

/// Metadata threaded through the integrator on every force evaluation.
struct ProxMeta<'a> {
    body: &'a ExtendedBody,
    body_naif_id: i32,
    perturbers: &'a [Perturber],
    /// Optional non-gravitational force acting on the particle.
    /// Evaluated in the inertial Sun-relative frame at every step
    /// (mirroring the heliocentric `propagate_n_body_spk` path) and
    /// transformed back into body-fixed body-natural units before
    /// being added to the integrator's acceleration.
    non_grav: Option<&'a NonGravModel>,
    spk: &'a SpkCollection,
    /// TDB Julian Date corresponding to the integrator's tau = 0.
    /// The integrator's `Time<TDB>` parameter actually carries body-
    /// natural time `tau` in its `jd` slot; this offset plus the body
    /// time scale lets us reconstruct the real TDB jd for SPK lookups.
    jd_zero_tdb: f64,
}

/// Force function evaluated by the Radau integrator.
///
/// The state vector here is the body-fixed position in body-natural
/// length units, and the integrator's time argument is body-natural
/// time `tau` (carried in the `Time<TDB>::jd` slot for convenience).
/// Returned acceleration is in body-natural units `L / tau^2`.
fn near_body_accel(
    time: Time<TDB>,
    pos: &OVector<f64, U3>,
    vel: &OVector<f64, U3>,
    meta: &mut ProxMeta<'_>,
    _exact_eval: bool,
) -> KeteResult<OVector<f64, U3>> {
    // Reconstruct the actual TDB time from the integrator's tau.
    let tau = time.jd;
    let time_day = meta.body.units.time_day;
    let length_au = meta.body.units.length_au;
    let jd_actual_tdb = meta.jd_zero_tdb + tau * time_day;
    let time_tdb: Time<TDB> = Time::new(jd_actual_tdb);

    // Build (gm, r_perturber_relative_to_body) list by querying SPK.
    // Positions come out in AU centered on `body_naif_id`.
    let mut p_states: Vec<(f64, Vector3<f64>)> = Vec::with_capacity(meta.perturbers.len());
    for p in meta.perturbers {
        let s = meta.spk.try_get_state_with_center::<Equatorial>(
            p.naif_id,
            time_tdb,
            meta.body_naif_id,
        )?;
        p_states.push((p.gm_solar, Vector3::from(s.pos)));
    }

    let r_body = Vector3::new(pos[0], pos[1], pos[2]);
    // `small_body_accel` returns acceleration in body_length / day^2.
    let mut accel_per_day2 = match small_body_accel(meta.body, jd_actual_tdb, r_body, &p_states) {
        Ok(a) => a,
        // Translate the body-natural surface-contact error into a
        // propagator-level Impact tagged with the body's NAIF ID and
        // the current TDB time, mirroring `spk_accel`'s convention.
        Err(Error::SurfaceImpact) => return Err(Error::Impact(meta.body_naif_id, time_tdb)),
        Err(e) => return Err(e),
    };

    // Optional non-gravitational acceleration.  `NonGravModel` is
    // defined in the inertial Sun-relative equatorial frame, so we
    // round-trip the body-centric body-natural particle state through
    // (1) length scale, (2) body-fixed -> inertial rotation including
    // the omega x r velocity correction, (3) addition of the body's
    // Sun-relative state from SPK, evaluate, then map the resulting
    // inertial acceleration back into body-fixed body-natural units.
    if let Some(ng) = meta.non_grav {
        let jd = jd_actual_tdb;

        let r_body_au = r_body * length_au;
        // `vel` is in body_length / tau; convert to body_length / day
        // (then to AU / day) so the inertial-frame conversion below
        // sees the correct physical velocity.
        let v_body_au_per_day = Vector3::new(vel[0], vel[1], vel[2]) * (length_au / time_day);

        let r_inert = meta.body.rotation.rotate_to_inertial(jd, r_body_au);
        let v_inert_from_body = meta.body.rotation.rotate_to_inertial(jd, v_body_au_per_day);
        let omega = meta.body.rotation.angular_velocity_inertial(jd);
        let v_inert = v_inert_from_body + omega.cross(&r_inert);

        // Body's Sun-relative state at this epoch.
        let body_helio =
            meta.spk
                .try_get_state_with_center::<Equatorial>(meta.body_naif_id, time_tdb, 10)?;
        let r_sun = Vector3::from(body_helio.pos) + r_inert;
        let v_sun = Vector3::from(body_helio.vel) + v_inert;

        let mut accel_ng = Vector3::<f64>::zeros();
        ng.add_acceleration(&mut accel_ng, &r_sun, &v_sun);

        // Inertial -> body-fixed, then divide by length_au to convert
        // AU/day^2 into body_length / day^2.  The day^2 -> tau^2
        // conversion is applied jointly below.
        let accel_ng_body = meta.body.rotation.rotate_to_body(jd, accel_ng);
        accel_per_day2 += accel_ng_body / length_au;
    }

    // Convert body_length / day^2 -> body_length / tau^2.
    let accel_per_tau2 = accel_per_day2 * (time_day * time_day);
    Ok(OVector::<f64, U3>::new(
        accel_per_tau2.x,
        accel_per_tau2.y,
        accel_per_tau2.z,
    ))
}

/// Propagate a particle in the body-centric proximity regime of an
/// [`ExtendedBody`] from the state's epoch to `jd_final`.
///
/// `body_naif_id` is the SPK ID of the central body itself; perturber
/// states are queried with that ID as the SPK center, so they are
/// already body-relative when fed to [`small_body_accel`].
///
/// The integration runs in body-fixed coordinates with body-natural
/// lengths and TDB-day time, using the existing [`RadauIntegrator`].
/// No regime hand-off is performed: the caller is responsible for
/// switching to heliocentric N-body propagation when the particle
/// leaves `body.proximity_radius_au`.
///
/// # Errors
/// Returns [`Error::Impact`] (tagged with `body_naif_id` and the TDB
/// time at the point of contact) if the particle's trajectory crosses
/// the body's polyhedron surface.  Other errors propagate from the
/// integrator (convergence, step rejection) or from SPK lookups
/// (missing kernel coverage).
pub fn propagate_near_body(
    body: &ExtendedBody,
    body_naif_id: i32,
    perturbers: &[Perturber],
    non_grav: Option<&NonGravModel>,
    initial_state: &BodyRelativeState,
    jd_final: Time<TDB>,
    spk: &SpkCollection,
) -> KeteResult<BodyRelativeState> {
    let time_day = body.units.time_day;
    let length_au = body.units.length_au;
    let jd_zero_tdb = initial_state.epoch.jd;

    let meta = ProxMeta {
        body,
        body_naif_id,
        perturbers,
        non_grav,
        spk,
        jd_zero_tdb,
    };

    let pos0 = OVector::<f64, U3>::new(
        initial_state.pos.x,
        initial_state.pos.y,
        initial_state.pos.z,
    );
    // `BodyRelativeState.vel` is in body_length / day; the integrator
    // works in body-natural time `tau`, so scale to body_length / tau.
    let vel0 = OVector::<f64, U3>::new(
        initial_state.vel.x * time_day,
        initial_state.vel.y * time_day,
        initial_state.vel.z * time_day,
    );

    // The integrator's `Time<TDB>` carries `tau` (body-natural time)
    // in its `jd` slot.  `near_body_accel` reconstructs the actual
    // TDB jd from `meta.jd_zero_tdb + tau * time_day` for SPK lookups.
    let tau_init: Time<TDB> = Time::new(0.0);
    let tau_final: Time<TDB> = Time::new((jd_final.jd - jd_zero_tdb) / time_day);

    let (pos, vel, _meta) = RadauIntegrator::integrate(
        &near_body_accel,
        pos0,
        vel0,
        tau_init,
        tau_final,
        meta,
        None,
    )?;

    // Suppress unused-variable warning when the conversion below uses
    // it only via the velocity scaling.
    let _ = length_au;

    Ok(BodyRelativeState {
        desig: initial_state.desig.clone(),
        pos: Vector3::new(pos[0], pos[1], pos[2]),
        // Convert back: body_length / tau -> body_length / day.
        vel: Vector3::new(vel[0] / time_day, vel[1] / time_day, vel[2] / time_day),
        epoch: jd_final,
    })
}

// ---------------------------------------------------------------------
// Regime hand-off helpers (Phase 6, manual path).
//
// These utilities let a caller convert particle states between the
// body-centric proximity regime (`BodyRelativeState`, body-fixed frame,
// body-natural lengths) and the standard heliocentric / inertial
// regime (`State<T, C>`, AU, AU/day) used by `propagate_n_body_spk`.
// They are intentionally SPK-free at the conversion layer: the caller
// supplies the body's own ephemeris state at the conversion epoch
// (typically obtained via one `SpkCollection::try_get_state_with_center`
// call), which keeps these functions deterministic and testable
// without kernel access.
// ---------------------------------------------------------------------

/// Convert a body-centric particle state to an inertial state with the
/// same `center` and frame as `body_helio_state`.
///
/// `body_helio_state` is the central body's own state at the same TDB
/// epoch as `state.epoch`, expressed in the desired output frame `T`
/// and centered on the desired center body `C` (typically SSB).
///
/// The conversion is:
/// ```text
/// r_inertial_au   = R_body->inertial * (r_body_units * length_au)
/// v_inertial_au_d = R_body->inertial * (v_body_units * length_au)
///                 + omega x r_inertial_au
/// pos_out         = body_helio_state.pos + r_inertial_au
/// vel_out         = body_helio_state.vel + v_inertial_au_d
/// ```
/// where `omega` is the body's inertial-frame angular-velocity vector
/// at `state.epoch` (zero for [`RotationModel::Fixed`]).
///
/// The body-fixed velocity in `BodyRelativeState` is the time
/// derivative of body-fixed position holding the body-fixed frame
/// constant, so the rotational `omega x r` term is added when
/// transferring to the inertial frame.
///
/// # Panics
/// Panics in debug builds if `state.epoch` differs from
/// `body_helio_state.epoch`.
pub fn body_relative_to_heliocentric<T: InertialFrame, C: CenterBody + Clone>(
    state: &BodyRelativeState,
    body: &ExtendedBody,
    body_helio_state: &State<T, C>,
) -> State<T, C>
where
    kete_core::frames::DynCenter: From<C>,
{
    debug_assert!(
        (state.epoch.jd - body_helio_state.epoch.jd).abs() < 1e-12,
        "epochs of particle and body states must agree"
    );
    let length_au = body.units.length_au;
    let jd = state.epoch.jd;

    // Particle position relative to body, expressed in the inertial
    // frame (rotate body-fixed -> inertial, then scale lengths).
    let r_body_au = state.pos * length_au;
    let r_inertial_au = body.rotation.rotate_to_inertial(jd, r_body_au);

    // Velocity transfer: body-fixed velocity rotated to inertial gives
    // the part due to motion within the body frame; add omega x r for
    // the rigid-body rotation contribution.
    let v_body_au_per_day = state.vel * length_au;
    let v_inertial_from_body = body.rotation.rotate_to_inertial(jd, v_body_au_per_day);
    let omega = body.rotation.angular_velocity_inertial(jd);
    let v_rotational = omega.cross(&r_inertial_au);
    let v_inertial = v_inertial_from_body + v_rotational;

    let pos = Vector::<T>::from(Vector3::from(body_helio_state.pos) + r_inertial_au);
    let vel = Vector::<T>::from(Vector3::from(body_helio_state.vel) + v_inertial);
    State {
        desig: state.desig.clone().unwrap_or(Desig::Empty),
        epoch: state.epoch,
        pos,
        vel,
        center: body_helio_state.center,
    }
}

/// Convert an inertial particle state to body-centric coordinates.
///
/// Inverse of [`body_relative_to_heliocentric`].  Subtracts the body's
/// translational state, rotates into the body-fixed frame, removes the
/// rigid-body rotation contribution from the velocity, and scales to
/// body-natural lengths.
///
/// # Panics
/// Panics in debug builds if `state.epoch` differs from
/// `body_helio_state.epoch`, or if the two states are not centered on
/// the same NAIF id.
pub fn heliocentric_to_body_relative<T: InertialFrame, C: CenterBody>(
    state: &State<T, C>,
    body: &ExtendedBody,
    body_helio_state: &State<T, C>,
) -> BodyRelativeState
where
    kete_core::frames::DynCenter: From<C>,
{
    debug_assert!(
        (state.epoch.jd - body_helio_state.epoch.jd).abs() < 1e-12,
        "epochs of particle and body states must agree"
    );
    debug_assert_eq!(
        state.center_id(),
        body_helio_state.center_id(),
        "particle and body states must share the same center"
    );
    let length_au = body.units.length_au;
    let jd = state.epoch.jd;

    let r_inertial_au = Vector3::from(state.pos) - Vector3::from(body_helio_state.pos);
    let v_inertial = Vector3::from(state.vel) - Vector3::from(body_helio_state.vel);

    // Remove rigid-body rotation contribution before rotating into
    // body-fixed frame: v_inertial = v_body_fixed_in_inertial + omega x r
    let omega = body.rotation.angular_velocity_inertial(jd);
    let v_inertial_from_body = v_inertial - omega.cross(&r_inertial_au);

    let r_body_au = body.rotation.rotate_to_body(jd, r_inertial_au);
    let v_body_au_per_day = body.rotation.rotate_to_body(jd, v_inertial_from_body);

    BodyRelativeState {
        desig: Some(state.desig.clone()),
        pos: r_body_au / length_au,
        vel: v_body_au_per_day / length_au,
        epoch: state.epoch,
    }
}

/// Predicate: is the particle inside `factor * body.proximity_radius_au`?
///
/// `factor` controls the threshold: pass a value < 1 to test the
/// "exit" condition (particle far enough to switch to heliocentric
/// propagation) and a value > 1 (or the same value) for the "enter"
/// condition.  Hysteresis is implemented by the caller picking two
/// different factors for entry and exit.
///
/// The comparison is done in inertial AU (not body-natural lengths)
/// so the threshold semantics are independent of the unit system.
#[must_use]
pub fn is_inside_proximity(state: &BodyRelativeState, body: &ExtendedBody, factor: f64) -> bool {
    let r_body_au = state.pos * body.units.length_au;
    r_body_au.norm() <= factor * body.proximity_radius_au
}

#[cfg(test)]
mod tests {
    use super::*;
    use kete_core::shape::{BodyUnits, Polyhedron, RotationModel};
    use std::f64::consts::TAU;

    fn icosphere_verts_faces(subdivisions: u32) -> (Vec<Vector3<f64>>, Vec<[u32; 3]>) {
        let phi = f64::midpoint(1.0, 5.0_f64.sqrt());
        let mut verts: Vec<Vector3<f64>> = vec![
            Vector3::new(-1.0, phi, 0.0),
            Vector3::new(1.0, phi, 0.0),
            Vector3::new(-1.0, -phi, 0.0),
            Vector3::new(1.0, -phi, 0.0),
            Vector3::new(0.0, -1.0, phi),
            Vector3::new(0.0, 1.0, phi),
            Vector3::new(0.0, -1.0, -phi),
            Vector3::new(0.0, 1.0, -phi),
            Vector3::new(phi, 0.0, -1.0),
            Vector3::new(phi, 0.0, 1.0),
            Vector3::new(-phi, 0.0, -1.0),
            Vector3::new(-phi, 0.0, 1.0),
        ]
        .into_iter()
        .map(|v| v.normalize())
        .collect();
        let mut faces: Vec<[u32; 3]> = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        for _ in 0..subdivisions {
            let mut new_faces = Vec::with_capacity(faces.len() * 4);
            let mut midpoints: std::collections::BTreeMap<(u32, u32), u32> =
                std::collections::BTreeMap::new();
            let mut midpoint = |a: u32, b: u32, verts: &mut Vec<Vector3<f64>>| -> u32 {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&i) = midpoints.get(&key) {
                    return i;
                }
                let m = ((verts[a as usize] + verts[b as usize]) * 0.5).normalize();
                let i = u32::try_from(verts.len()).unwrap();
                verts.push(m);
                let _ = midpoints.insert(key, i);
                i
            };
            for f in &faces {
                let a = midpoint(f[0], f[1], &mut verts);
                let b = midpoint(f[1], f[2], &mut verts);
                let c = midpoint(f[2], f[0], &mut verts);
                new_faces.push([f[0], a, c]);
                new_faces.push([f[1], b, a]);
                new_faces.push([f[2], c, b]);
                new_faces.push([a, b, c]);
            }
            faces = new_faces;
        }
        (verts, faces)
    }

    fn sphere_body(radius_au: f64, gm: f64) -> ExtendedBody {
        let (mut v, f) = icosphere_verts_faces(2);
        for vert in &mut v {
            *vert *= radius_au;
        }
        let p = Polyhedron::try_new(v, &f, gm).unwrap();
        let units = BodyUnits::try_new(radius_au, gm).unwrap();
        ExtendedBody::try_new(vec![p], RotationModel::identity(), units, 5.0 * radius_au).unwrap()
    }

    #[test]
    fn circular_orbit_no_perturbers_advances_along_orbit() {
        // Sphere body with a circular test orbit at r = 100 body radii,
        // integrated for 1/16th of a period.  In body_length/day^2
        // units the central acceleration scale is gm / length_au^3, so
        // v_circ = sqrt(GM_eff / r) and the period is
        // 2*pi*sqrt(r^3 / GM_eff).
        crate::test_data::ensure_test_spk();
        let radius = 1e-9; // ~150 m
        // Choose gm so the orbital period at r=100 body radii is ~1 day,
        // well above the integrator's MIN_STEP (5e-4 days) so Radau
        // has plenty of steps per orbit.
        let gm = 4e-20;
        let body = sphere_body(radius, gm);
        let gm_eff = gm / radius.powi(3); // body_length^3 / day^2

        let r0 = 100.0;
        let v0 = (gm_eff / r0).sqrt();
        let period_days = TAU * (r0.powi(3) / gm_eff).sqrt();

        let initial = BodyRelativeState {
            desig: None,
            pos: Vector3::new(r0, 0.0, 0.0),
            vel: Vector3::new(0.0, v0, 0.0),
            epoch: 2_451_545.0.into(),
        };
        // Integrate 1/16th of a period: small enough that the
        // polyhedron's quadrupole-induced precession has negligible
        // angular effect, but large enough to verify the orbit really
        // moves.
        let frac = 1.0 / 16.0;
        let final_jd: Time<TDB> = (2_451_545.0 + frac * period_days).into();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();
        let final_state = propagate_near_body(
            &body,
            10, // any NAIF id; perturbers list is empty so it is unused
            &[],
            None,
            &initial,
            final_jd,
            &spk,
        )
        .unwrap();

        let theta = TAU * frac;
        let expected_pos = Vector3::new(r0 * theta.cos(), r0 * theta.sin(), 0.0);
        let expected_vel = Vector3::new(-v0 * theta.sin(), v0 * theta.cos(), 0.0);
        let pos_err = (final_state.pos - expected_pos).norm() / r0;
        let vel_err = (final_state.vel - expected_vel).norm() / v0;
        // Tolerance accounts for both Radau truncation and polyhedron
        // multipole departure from a true sphere ((R/r)^2 ~ 1e-4).
        assert!(
            pos_err < 1e-2,
            "1/16-period position error too large: rel={pos_err:e}"
        );
        assert!(
            vel_err < 1e-2,
            "1/16-period velocity error too large: rel={vel_err:e}"
        );
        // Sanity: radial distance should be conserved at ~r0.
        let r_final = final_state.pos.norm();
        let r_drift = (r_final - r0).abs() / r0;
        assert!(r_drift < 5e-3, "radial drift too large: rel={r_drift:e}");
    }

    #[test]
    fn surface_impact_returns_impact_error() {
        // Aim a particle straight at the body so the integrator hits
        // the surface.  small_body_accel returns SurfaceImpact, which
        // the wrapper translates to Error::Impact tagged with the
        // supplied NAIF id.
        crate::test_data::ensure_test_spk();
        let radius = 1e-9;
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        let gm_eff = gm / radius.powi(3);

        // Start just outside the body surface (r = 2 body radii)
        // heading inward at 3x circular speed.
        let r0 = 2.0;
        let v_inward = -3.0 * (gm_eff / r0).sqrt();
        let initial = BodyRelativeState {
            desig: None,
            pos: Vector3::new(r0, 0.0, 0.0),
            vel: Vector3::new(v_inward, 0.0, 0.0),
            epoch: 2_451_545.0.into(),
        };
        let dt_days = (r0 / v_inward.abs()) * 4.0;
        let final_jd: Time<TDB> = (2_451_545.0 + dt_days).into();
        let spk = crate::spk::LOADED_SPK.try_read().unwrap();

        let result = propagate_near_body(&body, 10, &[], None, &initial, final_jd, &spk);
        match result {
            Err(Error::Impact(id, _)) => {
                assert_eq!(id, 10, "impact tagged with wrong NAIF id");
            }
            other => panic!("expected Error::Impact, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------
    // Phase 6 - regime hand-off helpers.
    // -----------------------------------------------------------------

    use kete_core::frames::{Equatorial, SSB};

    fn dummy_body_helio_state(jd: f64) -> State<Equatorial, SSB> {
        // Synthetic central-body state: 1.5 AU along +x, moving at
        // ~17.3 km/s ~ 0.01 AU/day along +y (rough Mars-like values).
        // The exact numbers do not matter for the round-trip test;
        // only consistency between forward and inverse conversion does.
        State {
            desig: Desig::Empty,
            epoch: jd.into(),
            pos: Vector::<Equatorial>::from([1.5, 0.0, 0.0]),
            vel: Vector::<Equatorial>::from([0.0, 0.01, 0.0]),
            center: SSB,
        }
    }

    #[test]
    fn round_trip_fixed_rotation() {
        // Round-trip a body-relative state through heliocentric and
        // back; with Fixed rotation the angular velocity is zero so
        // both position and velocity must come back exactly (modulo
        // floating-point round-off).
        let radius = 1e-9;
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        let jd = 2_451_545.0;
        let body_state = dummy_body_helio_state(jd);

        let initial = BodyRelativeState {
            desig: Some(Desig::Name("test_particle".to_string())),
            pos: Vector3::new(50.0, -10.0, 5.0),
            vel: Vector3::new(0.1, 0.2, -0.05),
            epoch: jd.into(),
        };

        let helio = body_relative_to_heliocentric(&initial, &body, &body_state);
        let back = heliocentric_to_body_relative(&helio, &body, &body_state);

        let pos_err = (back.pos - initial.pos).norm();
        let vel_err = (back.vel - initial.vel).norm();
        // Round-trip cancellation: body is at 1.5 AU and particle is
        // at ~5e-8 AU, so subtracting the body position introduces an
        // f64-epsilon error of ~eps * 1.5 / length_au in body-natural
        // units (~3e-7 here).  Allow 1e-6 to keep the test stable.
        assert!(pos_err < 1e-6, "fixed-rotation pos round-trip: {pos_err:e}");
        assert!(vel_err < 1e-6, "fixed-rotation vel round-trip: {vel_err:e}");
        // Designation propagated through the inverse path.
        assert_eq!(back.desig, Some(Desig::Name("test_particle".to_string())));
    }

    #[test]
    fn round_trip_constant_spin() {
        // With ConstantSpin the inertial-frame angular velocity is
        // non-zero; the inverse conversion must remove that
        // contribution to recover the original body-fixed velocity.
        let radius = 1e-9;
        let gm = 1e-20;
        let (mut v, f) = icosphere_verts_faces(2);
        for vert in &mut v {
            *vert *= radius;
        }
        let p = Polyhedron::try_new(v, &f, gm).unwrap();
        let units = BodyUnits::try_new(radius, gm).unwrap();
        let rotation = RotationModel::ConstantSpin {
            pole_ra: 0.7,
            pole_dec: 0.4,
            w0: 1.2,
            w_dot: 5.0, // rad/day, a fast spin to expose any error
            epoch_jd: 2_451_545.0,
        };
        let body = ExtendedBody::try_new(vec![p], rotation, units, 5.0 * radius).unwrap();
        let jd = 2_451_545.7;
        let body_state = dummy_body_helio_state(jd);

        let initial = BodyRelativeState {
            desig: None,
            pos: Vector3::new(80.0, 30.0, -20.0),
            vel: Vector3::new(-0.2, 0.05, 0.3),
            epoch: jd.into(),
        };

        let helio = body_relative_to_heliocentric(&initial, &body, &body_state);
        let back = heliocentric_to_body_relative(&helio, &body, &body_state);

        let pos_err = (back.pos - initial.pos).norm();
        let vel_err = (back.vel - initial.vel).norm();
        // Same cancellation budget as the fixed-rotation case;
        // additional rotation/cross-product round-tripping is itself
        // exact in f64 to ~eps.
        assert!(pos_err < 1e-6, "spin pos round-trip: {pos_err:e}");
        assert!(vel_err < 1e-6, "spin vel round-trip: {vel_err:e}");
    }

    #[test]
    fn proximity_predicate_with_hysteresis() {
        // Verify is_inside_proximity reports correctly relative to the
        // body's proximity_radius_au, both at the threshold and with
        // separate enter/exit factors (hysteresis).
        let radius = 1e-9;
        let gm = 1e-20;
        let body = sphere_body(radius, gm);
        // proximity_radius_au = 5 * radius = 5e-9 from sphere_body().
        // r_body_units * length_au = r_au.
        // r_body_units = 4.0 -> r_au = 4e-9 (inside both 0.8 and 1.0)
        let inside = BodyRelativeState {
            desig: None,
            pos: Vector3::new(4.0, 0.0, 0.0),
            vel: Vector3::zeros(),
            epoch: 2_451_545.0.into(),
        };
        // r_body_units = 6.0 -> r_au = 6e-9 (outside both 1.0 and 1.2)
        let outside = BodyRelativeState {
            desig: None,
            pos: Vector3::new(6.0, 0.0, 0.0),
            vel: Vector3::zeros(),
            epoch: 2_451_545.0.into(),
        };
        // r_body_units = 5.5 -> r_au = 5.5e-9 (between hysteresis bounds)
        let intermediate = BodyRelativeState {
            desig: None,
            pos: Vector3::new(5.5, 0.0, 0.0),
            vel: Vector3::zeros(),
            epoch: 2_451_545.0.into(),
        };

        // Plain test: factor == 1.0 means "inside the proximity sphere".
        assert!(is_inside_proximity(&inside, &body, 1.0));
        assert!(!is_inside_proximity(&outside, &body, 1.0));

        // Hysteresis: caller picks exit factor 1.2 (looser) and enter
        // factor 0.8 (tighter).  Intermediate point at r=5.5 is still
        // "inside" by the exit threshold but would not re-enter under
        // the enter threshold -> stays in the heliocentric regime once
        // it exited, but stays in the body-centric regime if it never
        // exited.  This is the standard hysteresis pattern.
        let exit_factor = 1.2;
        let enter_factor = 0.8;
        assert!(
            is_inside_proximity(&intermediate, &body, exit_factor),
            "intermediate should still be 'inside' under loose exit threshold"
        );
        assert!(
            !is_inside_proximity(&intermediate, &body, enter_factor),
            "intermediate should be 'outside' under tight enter threshold"
        );
    }
}
