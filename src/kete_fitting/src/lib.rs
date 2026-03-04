//! # Orbit Determination and Fitting
//!
//! Batch least-squares differential correction with chained STM propagation,
//! initial orbit determination, and observation modeling for the Kete solar
//! system survey simulator.

mod diff_correction;
mod iod;
mod obs;

pub use diff_correction::{
    OrbitFit, differential_correction, differential_correction_with_rejection,
};
pub use iod::initial_orbit_determination;
pub use obs::Observation;
