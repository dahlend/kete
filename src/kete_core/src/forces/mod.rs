// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Force Models
//!
//! A force is anything that contributes an acceleration to an orbiting body.
//! Gravity from the planets is one force; solar radiation pressure on a dust
//! grain is another; the outgassing thrust on a comet is a third.
//!
//! Each force is a plain struct. You call its `accel` method with the body's
//! current position, velocity, and time, and it returns an acceleration
//! (AU/day^2). Some forces also need one or more numbers that are not known
//! in advance and must be fitted from observations -- for example, a dust
//! grain's radiation pressure coefficient `beta`. Those numbers are passed as
//! a separate `free_params: &[f64]` slice so the same force struct can be
//! used with different parameter values during fitting.
//!
//! ## The force trait
//!
//! [`ParameterizedForce`] is implemented by every force. It accepts a `free_params`
//! slice, which is empty when the force has no fitted quantities. Propagating a plain
//! [`State`](crate::state::State) requires a force with no free parameters.
//!
//! ## Fixing parameters
//!
//! [`ParameterMask`] wraps a [`ParameterizedForce`] and fixes any subset of its
//! parameters at given values; the rest stay free and are passed through from the
//! caller.
//!
//! With every parameter free ([`ParameterMask::all_free`]) it is what gets stored on an
//! [`UncertainState`](crate::state::UncertainState) alongside its fitted values. With
//! some fixed it restricts a fit, e.g. holding `a1` and `a3` while fitting only `a2`.
//! With every parameter fixed ([`ParameterMask::all_fixed`]) it has no free parameters,
//! which is what you use to propagate a single trajectory from a best-fit estimate.
//!
//! ## Gravity
//!
//! Every term kete models is a function of the object's state relative to one massive
//! body. [`GravParams`] describes such a body: its `GM`, whether the relativistic
//! correction applies, and its [`Shape`] beyond a point mass.
//! [`GravParams::add_acceleration`] and [`GravParams::add_acceleration_and_jacobians`] evaluate all of a
//! body's terms on the relative state.
//!
//! The complete model used for n-body orbit propagation is
//! [`NBody`](crate::propagation::NBody): it looks up each massive body in an
//! [`Ephemeris`](crate::ephemeris::Ephemeris), evaluates that body's gravity, and
//! evaluates an optional non-grav force on the Sun-relative state.

mod gravity;
mod nongrav;
mod parameter_mask;
mod polyhedron;
mod spherical_harmonics;
mod traits;

pub use gravity::{
    GravParams, Orientation, Shape, known_masses, register_custom_mass, register_mass,
    registered_masses,
};
pub(crate) use gravity::{apply_gr_correction, j2_correction};
pub(crate) use nongrav::radiation_accel;
pub use nongrav::{
    DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, NonGravKind, RampedThrustNonGrav,
    a_over_m_from_physical, density_from_a_over_m, lambda_0_from_physical,
    thermal_inertia_from_lambda_0,
};
pub use parameter_mask::ParameterMask;
pub use polyhedron::Polyhedron;
pub use spherical_harmonics::SphericalHarmonics;
pub use traits::ParameterizedForce;

/// A [`ParameterMask`] over a [`NonGravKind`]: a bundled non-grav model with each of its
/// parameters fixed or free.
///
/// It rides along with an uncertain state with its fitted parameters free, and is handed
/// to plain `State` propagation with every parameter fixed.
pub type NonGravMask = ParameterMask<NonGravKind>;
