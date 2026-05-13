//! Non-gravitational forces: typed [`ParameterizedForce`] impls and physical-input helpers.
//!
//! The three non-grav variants are separate concrete types, each holding only
//! its fixed physical constants; fitted parameters are passed through
//! `ParameterizedForce::accel`'s `&[f64]` argument and stored on the carrying state's
//! `free_params`.
//!
//! [`NonGravKind`] aggregates the three variants behind a single concrete
//! type. It is the convenient default for code that ships with kete (Horizons
//! reader, batch fitting helpers, Python wrapper) and slots into the open
//! composition machinery exactly like a hand-written `ParameterizedForce`.

mod dust;
mod farnocchia;
mod jpl_comet;
mod kind;

pub use dust::DustNonGrav;
pub use farnocchia::{
    FarnocchiaNonGrav, a_over_m_from_physical, density_from_a_over_m, lambda_0_from_physical,
    thermal_inertia_from_lambda_0,
};
pub use jpl_comet::JplCometNonGrav;
pub use kind::NonGravKind;
