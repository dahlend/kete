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
//! This crate is left as a stand alone Rust crate, completely independent of the
//! Python wrappers. This is done intentionally, as it makes these functions available
//! outside of the python module so that wrappers may be written for other languages
//! later.
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
//! - [`propagation::propagate_n_body_spk`] - The main function which should be
//!   used to propagate the state of an object for highest precision.
//! - [`propagation::propagate_two_body`] - If only an approximate position is
//!   required over a short time period, this function can be used as it is about 50x
//!   faster.
//! - [`spice`] - JPL maintains a very complex software tool called SPICE, which
//!   is used to save and load states of objects in the solar system. The original
//!   SPICE tool was written in Fortran, and is not very user friendly or terribly
//!   thread safe. Kete provides a custom implementation of a read-only interface to
//!   the files produces by SPICE that is thread safe and performant.
//!
//!
//
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

pub mod cache;
pub mod constants;
pub mod desigs;
pub mod elements;
pub mod errors;
pub mod fitting;
pub mod flux;
pub mod fov;
pub mod frames;
pub mod io;
pub mod propagation;
pub mod simult_states;
pub mod spice;
pub mod state;
pub mod stats;
pub mod time;
pub mod util;

/// Common useful imports
pub mod prelude {
    pub use crate::desigs::Desig;
    pub use crate::elements::CometElements;
    pub use crate::errors::{Error, KeteResult};
    pub use crate::flux::{
        CometMKParams, FrmParams, HGParams, NeatmParams, black_body_flux, frm_facet_temperature,
        lambertian_flux, neatm_facet_temperature,
    };
    pub use crate::frames::{Ecliptic, Equatorial, FK4, Galactic, NonInertialFrame};
    pub use crate::propagation::{propagate_n_body_spk, propagate_two_body};
    pub use crate::simult_states::SimultaneousStates;
    pub use crate::spice::{LOADED_PCK, LOADED_SPK};
    pub use crate::state::State;
}
