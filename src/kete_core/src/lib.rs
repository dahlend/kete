// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # kete Core
//!
//! Simulation tools for calculating orbits of comets and asteroids.
//!
//! This is designed to predict of the positions of all known comets and asteroids
//! in the solar system within the next few centuries with high precision.
//!
//! ``kete_core`` is the central mathematical tools, however there is a python wrapper
//! ``kete`` which is intended to be more user friendly.
//!
//! This crate has no Python dependency, so it can be used directly from Rust or wrapped
//! for other languages in the same way ``kete`` wraps it for Python.
//!
//! ## Important Concepts
//!
//! There are a few core concepts which are important to understand.
//!
//! - [`frames::Vector`] - Cartesian vectors in 3D space, typically used to
//!   represent positions and velocities of objects in space.  
//! - [`frames::InertialFrame`] - A coordinate system which defines the
//!   cartesian axis. There are several commonly used coordinate systems, and kete
//!   contains conversion tools between them.
//! - [`state::State`] - The 'state' of an object at an instant of time, which
//!   contains the name, position, velocity, and time of the object.
//! - [`desigs::Desig`] - A designation for an object. There are many ways to
//!   refer to asteroids and comets, this provides representations and tools for
//!   parsing.
//! - [`time::Time`] - Representation of time, and allows conversions between
//!   different time systems. The most common time system used in orbital mechanics
//!   is TDB (Barycentric Dynamical Time), however there are many others.
//! - [`fov::FOV`] - A field of view, which is a representation of an
//!   area of sky that a telescope can see. This is used to calculate
//!   whether an object is visible from a given location at a given time.
//! - [`kepler::propagate_two_body`] - If only an approximate position is
//!   required over a short time period, this function can be used as it is about 50x
//!   faster.
//!

pub mod analysis;
pub mod bands;
pub use bands::{Band, BandInfo, ColorCorrFn};
pub mod cache;
pub mod constants;
pub mod desigs;
pub mod elements;
pub mod ephemeris;
pub mod errors;
pub mod forces;
pub mod fov;
pub mod frames;
pub mod geometry;
pub mod integrators;
pub mod io;
pub mod kepler;
pub mod moid;
pub mod propagation;
pub mod state;
pub mod time;
pub mod util;

/// Common useful imports
pub mod prelude {
    pub use crate::desigs::Desig;
    pub use crate::desigs::{NaifId, naif_ids_from_name, try_name_from_id};
    pub use crate::desigs::{OBS_CODES, ObsCode, try_obs_code_from_name};
    pub use crate::elements::CometElements;
    pub use crate::errors::{Error, KeteResult};
    pub use crate::frames::{
        CenterBody, DynCenter, EarthCenter, Ecliptic, Equatorial, FK4, Galactic, InertialFrame,
        NonInertialFrame, SSB, SunCenter,
    };
    pub use crate::kepler::propagate_two_body;
    pub use crate::state::{SimultaneousStates, State, UncertainState};
    pub use crate::time::{TDB, Time, UTC};
}
