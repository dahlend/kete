//! SPK-dependent propagation.
//!
//! These functions and types require loaded SPICE kernels (SPK files) to
//! query planet states. Pure-math integrators and force models live in
//! `kete_core::integrators`, `kete_core::forces`, and `kete_core::kepler`.
//!
//! - [`SpkNBody`]: N-body Newtonian gravity using SPK ephemerides as the
//!   source of planet positions.
//! - [`Recenter`]: a `ParameterizedForce` adapter that shifts the reference body of the
//!   input pos/vel before delegating to an inner `ParameterizedForce`.
//! - [`compute_state_transition`]: state transition matrix between two epochs
//!   under SPK gravity.
//! - [`propagate_diffuse_state_adaptive`]: variational propagation of
//!   [`DiffuseState`](kete_core::state::DiffuseState) mixtures with adaptive
//!   sigma-point splitting.
//! - [`propagate_n_body_vec`] / [`closest_approach`]: batch propagation and
//!   close-encounter utilities.

mod analysis;
mod batch;
mod recenter;
mod spk_n_body;
mod stm;

#[cfg(test)]
mod diffuse;
#[cfg(test)]
mod jacobian;

pub use analysis::closest_approach;
pub use batch::{AccelVecMeta, propagate_n_body_vec, vec_accel};
pub use kete_core::state::{
    LinearityDiagnosis, SplitConfig, mixture_sigma_point_divergence,
    propagate_diffuse_state_adaptive, propagate_with_diagnosis, sigma_point_divergence,
};
pub use recenter::Recenter;
pub use spk_n_body::{SpkNBody, SpkNonGravs};
pub use stm::compute_state_transition;
