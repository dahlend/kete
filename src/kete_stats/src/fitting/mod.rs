// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! # Fitting
//! Fitting tools, including root finding.

mod bisection;
mod golden_section;
mod halley;
mod nelder_mead;
mod newton;

pub use self::bisection::bisection;
pub use self::golden_section::golden_section_search;
pub use self::halley::halley;
pub use self::nelder_mead::{NelderMeadResult, nelder_mead};
pub use self::newton::newton_raphson;

/// Error type for fitting operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ConvergenceError {
    /// Maximum number of iterations reached without convergence.
    #[error("Maximum number of iterations reached without convergence")]
    Iterations,

    /// Non-finite value encountered during evaluation.
    #[error("Non-finite value encountered during evaluation")]
    NonFinite,

    /// Zero derivative encountered during evaluation.
    #[error("Zero derivative encountered during evaluation")]
    ZeroDerivative,

    /// Invalid input provided to the solver.
    #[error("Invalid input: {0}")]
    InvalidInput(&'static str),
}

/// Result type for fitting operations.
pub type FittingResult<T> = Result<T, ConvergenceError>;
