//! [`NonGravKind`]: closed sum type over the bundled Sun-centered
//! non-gravitational force models.
//!
//! `NonGravKind` aggregates the non-grav variants kete ships with -- dust
//! radiation pressure, JPL comet outgassing, the Farnocchia 2025 oblate
//! thermal recoil model, and a thrust fixed in the RTN frame with a linear ramp in
//! time. All share
//! `Frame = Equatorial` and `Center = SunCenter`, so the enum slots
//! into `ParameterMask` and `SpkNBody` interchangeably.
//!
//! This is a convenience aggregate for kete's bundled physics, not a
//! gate -- the public [`ParameterizedForce`](crate::forces::ParameterizedForce)
//! trait stays open, and external users can wrap their own concrete types in
//! [`ParameterMask`](crate::forces::ParameterMask) directly.

use nalgebra::{Matrix3, Matrix3xX};

use super::dust::DustNonGrav;
use super::farnocchia::FarnocchiaNonGrav;
use super::jpl_comet::JplCometNonGrav;
use super::ramped_thrust::RampedThrustNonGrav;
use crate::errors::KeteResult;
use crate::forces::ParameterizedForce;
use crate::frames::{Equatorial, SunCenter, Vector};
use crate::time::{TDB, Time};

/// Sun-centered non-gravitational force template.
///
/// The variants share `Frame = Equatorial` and `Center = SunCenter`,
/// so they compose into the same outer force model interchangeably.
#[derive(Debug, Clone)]
pub enum NonGravKind {
    /// Dust grain radiation pressure + Poynting-Robertson drag.
    Dust(DustNonGrav),
    /// JPL comet `(a1, a2, a3)` outgassing in RTN basis.
    JplComet(JplCometNonGrav),
    /// Farnocchia 2025 oblate-spheroid radiation + thermal recoil.
    Farnocchia(FarnocchiaNonGrav),
    /// Thrust fixed in the RTN frame with a linear ramp in time.
    RampedThrust(RampedThrustNonGrav),
}

impl From<DustNonGrav> for NonGravKind {
    fn from(value: DustNonGrav) -> Self {
        Self::Dust(value)
    }
}

impl From<JplCometNonGrav> for NonGravKind {
    fn from(value: JplCometNonGrav) -> Self {
        Self::JplComet(value)
    }
}

impl From<FarnocchiaNonGrav> for NonGravKind {
    fn from(value: FarnocchiaNonGrav) -> Self {
        Self::Farnocchia(value)
    }
}

impl From<RampedThrustNonGrav> for NonGravKind {
    fn from(value: RampedThrustNonGrav) -> Self {
        Self::RampedThrust(value)
    }
}

impl ParameterizedForce for NonGravKind {
    type Frame = Equatorial;
    type Center = SunCenter;
    type Meta = ();

    fn n_free_params(&self) -> usize {
        match self {
            Self::Dust(f) => f.n_free_params(),
            Self::JplComet(f) => f.n_free_params(),
            Self::Farnocchia(f) => f.n_free_params(),
            Self::RampedThrust(f) => f.n_free_params(),
        }
    }

    fn free_param_names(&self) -> Vec<&'static str> {
        match self {
            Self::Dust(f) => f.free_param_names(),
            Self::JplComet(f) => f.free_param_names(),
            Self::Farnocchia(f) => f.free_param_names(),
            Self::RampedThrust(f) => f.free_param_names(),
        }
    }

    fn lower_bounds(&self) -> Vec<Option<f64>> {
        match self {
            Self::Dust(f) => f.lower_bounds(),
            Self::JplComet(f) => f.lower_bounds(),
            Self::Farnocchia(f) => f.lower_bounds(),
            Self::RampedThrust(f) => f.lower_bounds(),
        }
    }

    fn accel(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
        exact_eval: bool,
    ) -> KeteResult<Vector<Equatorial>> {
        match self {
            Self::Dust(f) => f.accel(time, pos, vel, free_params, meta, exact_eval),
            Self::JplComet(f) => f.accel(time, pos, vel, free_params, meta, exact_eval),
            Self::Farnocchia(f) => f.accel(time, pos, vel, free_params, meta, exact_eval),
            Self::RampedThrust(f) => f.accel(time, pos, vel, free_params, meta, exact_eval),
        }
    }

    fn jacobians(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<(Matrix3<f64>, Matrix3<f64>)> {
        match self {
            Self::Dust(f) => f.jacobians(time, pos, vel, free_params, meta),
            Self::JplComet(f) => f.jacobians(time, pos, vel, free_params, meta),
            Self::Farnocchia(f) => f.jacobians(time, pos, vel, free_params, meta),
            Self::RampedThrust(f) => f.jacobians(time, pos, vel, free_params, meta),
        }
    }

    fn parameter_jacobian(
        &self,
        time: Time<TDB>,
        pos: &Vector<Equatorial>,
        vel: &Vector<Equatorial>,
        free_params: &[f64],
        meta: &mut Self::Meta,
    ) -> KeteResult<Matrix3xX<f64>> {
        match self {
            Self::Dust(f) => f.parameter_jacobian(time, pos, vel, free_params, meta),
            Self::JplComet(f) => f.parameter_jacobian(time, pos, vel, free_params, meta),
            Self::Farnocchia(f) => f.parameter_jacobian(time, pos, vel, free_params, meta),
            Self::RampedThrust(f) => f.parameter_jacobian(time, pos, vel, free_params, meta),
        }
    }
}
