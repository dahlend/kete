//! # Statistics
//!

mod ks_test;
mod quantile;

pub use ks_test::two_sample_ks_statistic;
pub use quantile::{mad, median, quantile};
