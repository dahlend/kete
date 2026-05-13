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
//! ## The two force traits
//!
//! [`ParameterizedForce`] is the base trait implemented by every force. It
//! accepts a `free_params` slice (which may be empty if the force has no
//! fitted quantities).
//!
//! [`Force`] is a narrower trait that additionally promises the force needs
//! *no* free parameters -- the slice is always empty and can be omitted.
//! Gravity (`SpkNBody`) and a non-grav model with its parameters already
//! fixed ([`FrozenForce`]) are examples. Actually propagating the orbit
//! of an object cannot have any free parameters - meaning that to propagate
//! a non-grav contribution, you must first freeze its parameters into a
//! [`FrozenForce`].
//!
//!
//! ## Adapters
//!
//! Sometimes you need to convert one type of force into another.
//! For example, you might have a non-grav model that you can fix all of its
//! parameters and it is no longer a `ParameterizedForce` but a `Force`.
//! For these situations where are a number of adapters:
//!
//! [`FrozenForce`] wraps any [`ParameterizedForce`] and stores a concrete
//! parameter vector alongside it. The result behaves like a [`Force`] with
//! no free parameters; every call uses the stored values. Use this when you
//! have a best-fit estimate and want to propagate a single trajectory.
//!
//! [`ParameterMask`] also wraps a [`ParameterizedForce`] but leaves some or
//! all parameters free so they are passed through from the caller. The common
//! case is an all-`None` mask -- every parameter remains free -- which is what
//! gets stored on an [`UncertainState`](crate::state::UncertainState) for
//! uncertainty propagation. You can also partially freeze parameters (e.g.
//! hold `a2` and `a3` fixed while fitting only `a1`).
//!
//! [`Sum`] composes two forces that share frame and center. Accelerations are
//! summed; free parameters from the two members are concatenated in order.
//! A complete force model is typically `Sum<gravity, non_grav_adapter>`.
//!
//! With these definitions it is possible to define all kinds of forces on
//! physical objects, and the numerical integrator will happily propagate
//! them.
//!
//! Note that `kete_spice` implements a number of additional forces, including
//! the primary one needed for efficient n-body orbit propagation: [`SpkNBody`].

mod frozen;
mod gravity;
mod nongrav;
mod parameter_mask;
mod sum;
mod traits;

pub use frozen::FrozenForce;
pub use gravity::{
    GravParams, MASSES_KNOWN, MASSES_SELECTED, analytical_jacobians, known_masses,
    register_custom_mass, register_mass, registered_masses,
};
pub use nongrav::{
    DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, NonGravKind, a_over_m_from_physical,
    density_from_a_over_m, lambda_0_from_physical, thermal_inertia_from_lambda_0,
};
pub use parameter_mask::ParameterMask;
pub use sum::Sum;
pub use traits::{Force, ParameterizedForce};

/// A [`ParameterMask`] over a [`NonGravKind`]: the variational template
/// used wherever a non-grav model rides along with an uncertain state
/// and its parameters need to be exposed for variational integration.
pub type NonGravMask = ParameterMask<NonGravKind>;

/// A [`NonGravKind`] with parameter values baked in. Used wherever a
/// single concrete parameter estimate drives a plain-`State` propagation
/// (batch propagation, covariance samples, orbit-fitter inner loop).
pub type FrozenNonGrav = FrozenForce<NonGravKind>;
