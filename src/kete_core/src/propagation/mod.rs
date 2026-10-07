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
