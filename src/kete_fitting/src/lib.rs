//! # Orbit Determination and Fitting
//!
//! Batch least-squares differential correction with chained STM propagation,
//! initial orbit determination, and observation modeling for the Kete solar
//! system survey simulator.

mod batch;
mod iod;
mod obs;

pub use batch::{OrbitFit, differential_correction, differential_correction_with_rejection};
pub use iod::{gauss_iod, laplace_iod};
pub use obs::Observation;
