//! State representations and state-shape polymorphism.
//!
//! [`State`] is the basic exact Cartesian state. [`UncertainState`] is a best-fit
//! orbit as [`EquinoctialElements`](crate::elements::EquinoctialElements) plus a
//! covariance over those elements and any fitted free parameters; [`DiffuseState`]
//! is a weighted mixture of `UncertainState` components.  [`SimultaneousStates`]
//! collects many `State` objects sharing the same epoch.
//!
//! [`StateLike`] is the propagation trait for the exact Cartesian shapes.
//! `UncertainState` and `DiffuseState` deliberately do not implement it; see
//! `traits.rs` for why.
//!
//! [`propagate_with_stm`] / [`propagate_with_covariance`] are the low-level
//! STM integration primitives, and
//! [`propagate_elements_with_sensitivity`] composes the first with the element
//! Jacobian.

mod adaptive;
mod cartesian;
mod diffuse;
mod probes;
pub(crate) use probes::ProbeSet;
mod simultaneous;
mod stm;
mod traits;
mod uncertain;

pub use adaptive::{
    CenterResolver, DEFAULT_STEP_DAYS, SplitConfig, StepReport, Termination,
    propagate_diffuse_state, propagate_uncertain, step_diffuse_state,
};
pub use cartesian::State;
pub use diffuse::{
    DiffuseState, K3_SPLIT_MEANS, K3_SPLIT_SIGMA, K3_SPLIT_WEIGHTS, WEIGHT_SUM_TOL,
    split_axial_k3_along,
};
pub use simultaneous::SimultaneousStates;
pub use stm::{
    covariance_update, propagate_elements_with_sensitivity, propagate_state,
    propagate_with_covariance, propagate_with_stm,
};

pub use traits::StateLike;
pub use uncertain::{
    UncertainState, covariance_from_equinoctial, covariance_to_equinoctial,
    equinoctial_conversion_divergence, equinoctial_covariance_domain,
};
