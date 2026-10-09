// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Orbit Determination and Fitting
//!
//! Batch least-squares orbit fitting with chained STM propagation,
//! initial orbit determination, and observation modeling for Kete.

mod debias;
pub mod horizons;
mod iod;
mod lambert;
mod mcmc;
mod mpc;
mod obs;
mod orbit_fitting;
mod ranging;

pub use debias::{DEBIAS_EPOCH_JD, DEBIAS_N_TILES, DEBIAS_NSIDE, DebiasTable, DebiasVersion};
pub use horizons::HorizonsProperties;
pub use iod::initial_orbit_determination;
pub use lambert::lambert;
pub use mcmc::{OrbitSamples, fit_orbit_mcmc};
pub use mpc::{ObservatoryStats, get_observatory_stats};
pub use obs::AstrometricObservation;
pub use orbit_fitting::{OrbitFit, fit_orbit};
pub use ranging::{RangingSamples, fit_orbit_ranging};
