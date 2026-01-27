//! # Basic Statistics for astronomical data.
//!
//! This handles NaN gracefully for astronomical data sets.
mod data;
pub mod fitting;

/// export all stats functionality
pub mod prelude {
    pub use crate::data::{Data, DataError, SortedData, StatsResult, UncertainData};
}
