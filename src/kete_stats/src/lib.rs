// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Basic Statistics for astronomical data.
//!
//! This handles NaN gracefully for astronomical data sets.
mod data;
pub mod fitting;
pub mod healpix;

/// export all stats functionality
pub mod prelude {
    pub use crate::data::{Data, DataError, SortedData, StatsResult, UncertainData};
}
