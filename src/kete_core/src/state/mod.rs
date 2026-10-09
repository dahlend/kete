// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! State representations and state-shape polymorphism.
//!
//! [`State`] is the basic exact Cartesian state. [`UncertainState`] is a best-fit
//! orbit as [`EquinoctialElements`](crate::elements::EquinoctialElements) plus a
//! covariance over those elements and any fitted free parameters.  [`SimultaneousStates`]
//! collects many `State` objects sharing the same epoch.
//!
//! [`State::propagate_with`] advances an exact Cartesian state under a force.
//!
//! [`propagate_with_stm`] is the low-level STM integration primitive.

mod cartesian;
mod simultaneous;
mod stm;
mod uncertain;

pub use cartesian::State;
pub use simultaneous::SimultaneousStates;
pub use stm::{propagate_state, propagate_with_stm};

pub use uncertain::{
    UncertainState, covariance_from_equinoctial, covariance_to_equinoctial,
    equinoctial_covariance_domain,
};
