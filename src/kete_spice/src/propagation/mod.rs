//! SPK-dependent propagation.
//!
//! These functions and types require loaded SPICE kernels (SPK files) to
//! query planet states. Pure-math integrators and force models live in
//! `kete_core::integrators`, `kete_core::forces`, and `kete_core::kepler`.
//!
//! - [`SpkNBody`]: N-body gravity using SPK ephemerides as the source of planet
//!   positions, with an optional Sun-centered non-gravitational force.
//! - [`compute_state_transition`]: state transition matrix between two epochs
//!   under SPK gravity.
//! - [`propagate_diffuse_state`](kete_core::state::propagate_diffuse_state): variational propagation of
//!   [`DiffuseState`](kete_core::state::DiffuseState) mixtures with adaptive
//!   sigma-point splitting.
//! - [`propagate_n_body_vec`] / [`closest_approach`]: batch propagation and
//!   close-encounter utilities.

mod analysis;
mod batch;
mod spk_n_body;
mod stm;

#[cfg(test)]
mod diffuse;
#[cfg(test)]
mod jacobian;

pub use analysis::closest_approach;
pub use batch::propagate_n_body_vec;
pub use spk_n_body::{EphemerisCache, SpkNBody};
pub use stm::compute_state_transition;

/// Position and velocity of the Sun relative to the SSB at a time, as the center resolver
/// the uncertain and diffuse propagation take.
///
/// Their elements are referred to the Sun while [`SpkNBody`] is barycentric, so a state
/// crosses between the two centers at every epoch the propagation touches.
pub fn sun_resolver(
    spk: &crate::spk::SpkCollection,
) -> impl Fn(
    kete_core::time::Time<kete_core::time::TDB>,
) -> kete_core::errors::KeteResult<(nalgebra::Vector3<f64>, nalgebra::Vector3<f64>)>
+ Sync
+ '_ {
    move |time| {
        let sun = spk.try_get_state_with_center::<kete_core::frames::Equatorial>(10, time, 0)?;
        Ok((
            nalgebra::Vector3::from(sun.pos),
            nalgebra::Vector3::from(sun.vel),
        ))
    }
}
