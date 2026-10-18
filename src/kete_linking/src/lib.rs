// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Observation Linking
//!
//! Group observations of unknown moving objects into candidate objects, and
//! confirm them with orbit fits.
//!
//! The input is a list of [`kete_fitting::AstrometricObservation`]s, the
//! output a list of [`Confirmed`] objects, each the indices of the
//! observations its orbit fit kept and that [`kete_fitting::OrbitFit`].
//! [`search()`] does the whole job, and
//! [`SearchSettings::default()`](SearchSettings::default) suits surveys of
//! main belt objects from the ground:
//!
//! ```no_run
//! use kete_core::prelude::KeteResult;
//! use kete_fitting::AstrometricObservation;
//! use kete_linking::{SearchSettings, search};
//!
//! fn find(observations: &[AstrometricObservation]) -> KeteResult<()> {
//!     let found = search(observations, None, &SearchSettings::default())?;
//!     for object in &found {
//!         println!("{} observations, {}", object.sights.len(), object.fit.rms);
//!     }
//!     Ok(())
//! }
//! ```
//!
//! A search needs SPICE kernels loaded through `kete_spice`: the Sun at the
//! time of every observation, and the planets and the Moon over the span of
//! the observations, which the orbit fits of confirmation pull with.
//!
//! Two observations of one object give its direction on the sky at two times,
//! but not the two numbers that place it along the lines of sight. A
//! hypothesis asserts those two numbers for every object at once, as the
//! depth of the object along a fixed direction and its rate at a reference
//! time. Under it a pair of observations becomes one orbit, a run of several
//! close in time becomes the orbit that fits them best, and the orbits of one
//! object agree.
//!
//! The crate assumes nothing about the observer, the observing geometry, or
//! the cadence. The search uses two-body motion about the Sun, only bound
//! orbits are found, and the search window has to be short against the
//! orbital period.
//!
//! - [`Sight`] is one detection as a line of sight.
//! - [`form_units`] finds the units of linking: groups of sights which could
//!   be one object, either runs of sights close in time or pairs across
//!   gaps.
//! - [`Hypothesis`] is an assertion of the depth and depth rate of the
//!   objects along a fixed direction at a reference time, and
//!   [`hypothesis_grid`] lists the hypotheses of a search.
//! - [`link()`] groups the orbits of units into [`Candidate`]s.
//! - [`sight_from_observation`] and [`confirm()`] connect to observations
//!   and orbit fits, with the rules of [`ConfirmRules`].
//! - [`search()`] runs all of these in passes over observations, with
//!   [`SearchSettings`], each pass with the rate limit and grid steps of its
//!   [`SearchPass`]. With a label of the station of each observation, it
//!   scales the stated uncertainties of each station for confirmation.
//! - [`search_windows`] runs the passes over a whole survey one tile of the
//!   sky at a time, in windows of time which overlap by half, or in one
//!   window over the whole span. [`tiles()`] cuts the sky into the tiles. A
//!   tile is a core, one `HEALPix` pixel, and a halo around it, and holds
//!   every object whose direction at a reference time lies in its core, with
//!   [`parallax_margin`] for several observers.
//! - [`merge_pieces`] merges confirmed objects which are pieces of one
//!   object, which every search does last.
//!
//! The six fixed directions of the hypotheses are both ways along the axes
//! of the Equatorial frame. A unit is solved only under a direction which all
//! of its sights point within about 63 degrees of, see [`MIN_ALIGNMENT`], so
//! the sights of a unit must all be that close to the same one of the six.
//!
//! Only [`sight_from_observation`], [`confirm()`], [`merge_pieces`],
//! [`search()`] and [`search_windows`] need SPICE kernels loaded.

mod calibration;
mod confirm;
mod from_ranges;
mod grid;
mod hypothesis;
mod linalg;
mod link;
mod merge;
mod observe;
mod search;
mod sight;
#[cfg(test)]
mod tests;
mod tiles;
mod two_body_fit;
mod unit_fit;
mod units;
mod windows;

pub use confirm::{ConfirmRules, Confirmed, confirm, sight_from_observation};
pub use grid::{DEPTH_STEP, MAX_HYPOTHESES, RATE_STEP, hypothesis_grid};
pub use hypothesis::{Hypothesis, MIN_ALIGNMENT};
pub use link::{Candidate, LinkSettings, link};
pub use merge::merge_pieces;
pub use search::{SearchPass, SearchSettings, search};
pub use sight::Sight;
pub use tiles::{parallax_margin, tiles};
pub use units::form_units;
pub use windows::search_windows;
