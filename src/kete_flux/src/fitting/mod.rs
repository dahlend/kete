// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Model fitting: MCMC posterior sampling, and parallel batch fitting.
//!
//! Supports NEATM, FRM (thermal), and HG (reflected-light) models.

mod mcmc;
mod types;

#[cfg(test)]
mod tests;

// Re-export the public API.
pub use mcmc::{FitResult, FitTask, fit_batch, fit_mcmc};
pub use types::{FluxObs, FluxPriors, Model, ParamPrior};
