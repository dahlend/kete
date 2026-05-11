//! State representations and state-shape polymorphism.
//!
//! [`State`] is the basic exact Cartesian state; [`UncertainState`] augments
//! it with a covariance matrix and fitted free parameters; [`DiffuseState`]
//! is a weighted mixture of `UncertainState` components.  [`SimultaneousStates`]
//! collects many `State` objects sharing the same epoch.
//!
//! [`StateLike`] is the propagation trait implemented by all three state shapes.
//!
//! [`propagate_with_stm`] / [`propagate_with_covariance`] are the low-level
//! STM integration primitives used by [`UncertainState`] and [`DiffuseState`].

mod adaptive;
mod cartesian;
mod diffuse;
mod simultaneous;
mod stm;
mod traits;
mod uncertain;

pub use adaptive::{
    LinearityDiagnosis, SplitConfig, mixture_sigma_point_divergence,
    propagate_diffuse_state_adaptive, propagate_with_diagnosis, sigma_point_divergence,
};
pub use cartesian::State;
pub use diffuse::{
    DiffuseState, HUBER_K3_MEANS, HUBER_K3_SIGMA, HUBER_K3_WEIGHTS, WEIGHT_SUM_TOL,
    split_for_propagation,
};
pub use simultaneous::SimultaneousStates;
pub use stm::{covariance_update, propagate_state, propagate_with_covariance, propagate_with_stm};
pub use traits::StateLike;
pub use uncertain::UncertainState;
