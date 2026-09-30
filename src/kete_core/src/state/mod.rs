//! State representations and state-shape polymorphism.
//!
//! [`State`] is the basic exact Cartesian state. [`UncertainState`] is a best-fit
//! orbit as [`EquinoctialElements`](crate::elements::EquinoctialElements) plus a
//! covariance over those elements and any fitted free parameters; [`DiffuseState`]
//! is a weighted mixture of `UncertainState` components.  [`SimultaneousStates`]
//! collects many `State` objects sharing the same epoch.
//!
//! [`State::propagate_with`] advances an exact Cartesian state under a force.
//!
//! [`propagate_with_stm`] is the low-level STM integration primitive, and
//! [`propagate_elements_with_sensitivity`] composes the first with the element
//! Jacobian.

mod adaptive;
mod cartesian;
mod diffuse;
mod probes;
pub(crate) use probes::ProbeSet;
mod simultaneous;
mod stm;
mod uncertain;

pub use adaptive::{
    CenterResolver, DEFAULT_STEP_DAYS, SplitConfig, StepReport, Termination,
    propagate_diffuse_state, propagate_uncertain, step_diffuse_state,
};
pub use cartesian::State;
pub use diffuse::{
    DiffuseState, K3_SPLIT_MEANS, K3_SPLIT_SIGMA, K3_SPLIT_WEIGHTS, MAX_SPLIT_COUNT, SPLIT_SIZES,
    WEIGHT_SUM_TOL, split_axial_along, split_count_for_narrowing, split_narrowing,
};
pub use simultaneous::SimultaneousStates;
pub use stm::{propagate_elements_with_sensitivity, propagate_state, propagate_with_stm};

pub use uncertain::{
    UncertainState, covariance_from_equinoctial, covariance_to_equinoctial,
    equinoctial_conversion_divergence, equinoctial_covariance_domain,
};
