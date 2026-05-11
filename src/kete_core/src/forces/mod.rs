//! # Force Models
//!
//! A force is anything that contributes an acceleration to an orbiting body.
//! Gravity from the planets is one force; solar radiation pressure on a dust
//! grain is another; the outgassing thrust on a comet is a third.
//!
//! Each force is a plain struct. You call its `accel` method with the body's
//! current position, velocity, and time, and it returns an acceleration
//! (AU/day^2). Some forces also need one or more numbers that are not known
//! in advance and must be fitted from observations — for example, a dust
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
//! *no* free parameters — the slice is always empty and can be omitted.
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
//! case is an all-`None` mask — every parameter remains free — which is what
//! gets stored on an [`UncertainState`](crate::state::UncertainState) for
//! uncertainty propagation. You can also partially freeze parameters (e.g.
//! hold `a2` and `a3` fixed while fitting only `a1`).
//!
//! [`ForceSet`] composes multiple forces that share the same coordinate frame
//! and center body. Accelerations are summed; free parameters from each
//! member are concatenated in the order they were added.
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
mod set;
mod traits;

use std::sync::Arc;

use crate::frames::{Equatorial, SunCenter};

pub use frozen::FrozenForce;
pub use gravity::{
    GravParams, MASSES_KNOWN, MASSES_SELECTED, analytical_jacobians, known_masses,
    register_custom_mass, register_mass, registered_masses,
};
pub use nongrav::{
    DustNonGrav, FarnocchiaNonGrav, JplCometNonGrav, a_over_m_from_physical, density_from_a_over_m,
    lambda_0_from_physical, thermal_inertia_from_lambda_0,
};
pub use parameter_mask::ParameterMask;
pub use set::ForceSet;
pub use traits::{Force, ParameterizedForce};

/// Type-erased heliocentric non-gravitational force template.
///
/// `Arc` gives cheap `Clone` for shared ownership across propagation tasks.
pub type ArcForce = Arc<dyn ParameterizedForce<Frame = Equatorial, Center = SunCenter>>;

/// An all-`None` [`ParameterMask`] over an [`ArcForce`]: the variational
/// non-grav template stored on [`UncertainState`](crate::state::UncertainState)
/// wrappers. Free-parameter values come from `state.free_params` at integration
/// time; the mask itself holds no concrete values.
pub type NonGravMask = ParameterMask<ArcForce>;

/// An [`ArcForce`] with parameter values baked in, used wherever a single
/// concrete parameter estimate drives a plain-`State` propagation: batch
/// propagation, covariance samples, orbit-fitter inner loop.
pub type FrozenNonGrav = FrozenForce<ArcForce>;
