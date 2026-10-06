// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! N-body propagation with body states from an [`Ephemeris`].
//!
//! Pure-math integrators and force models live in [`integrators`](crate::integrators),
//! [`forces`](crate::forces), and [`kepler`](crate::kepler). The functions here need the
//! positions of the massive bodies over time, which they take from an ephemeris
//! provider; `kete_spice` implements one for its loaded SPK, PCK and CK files.
//!
//! - [`NBody`]: N-body gravity, with an optional Sun-centered non-gravitational force.
//! - [`compute_state_transition`]: state transition matrix between two epochs.
//! - [`propagate_n_body_vec`] / [`closest_approach`]: batch propagation and
//!   close-encounter utilities.
//! - [`IntegratedEphemeris`]: an [`Ephemeris`] of the
//!   planets that integrates them from saved states, needing no kernels.
//! - [`sun_resolver`]: the Sun's state over time, as the uncertain and diffuse
//!   propagation take it.

mod analysis;
mod batch;
mod integrated;
mod n_body;
mod stm;

pub use analysis::closest_approach;
pub use batch::propagate_n_body_vec;
pub use integrated::IntegratedEphemeris;
pub use n_body::{EphemerisCache, NBody};
pub use stm::compute_state_transition;

use crate::ephemeris::Ephemeris;
use crate::errors::KeteResult;
use crate::time::{TDB, Time};
use nalgebra::Vector3;

/// Position and velocity of the Sun relative to the SSB at a time, as the center resolver
/// the uncertain and diffuse propagation take.
///
/// Their elements are referred to the Sun while [`NBody`] is barycentric, so a state
/// crosses between the two centers at every epoch the propagation touches.
pub fn sun_resolver<E: Ephemeris>(
    ephem: &E,
) -> impl Fn(Time<TDB>) -> KeteResult<(Vector3<f64>, Vector3<f64>)> + Sync + '_ {
    move |time| {
        let sun = ephem.try_get_state_with_center(10, time, 0)?;
        Ok((Vector3::from(sun.pos), Vector3::from(sun.vel)))
    }
}
