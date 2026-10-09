// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! # Errors
//! Errors emitted by ``kete_core``

/// Define all errors which may be raise by this crate, as well as optionally provide
/// conversion to pyo3 error types which allow for the errors to be raised in Python.
use chrono::ParseError;
use kete_stats::fitting::ConvergenceError;
use std::{error, fmt, io, sync::TryLockError};

/// kete specific result.
pub type KeteResult<T> = Result<T, Error>;

/// Possible Errors which may be raised by this crate.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error {
    /// Numerical method did not converge within the algorithms limits.
    Convergence(String),

    /// Input or variable exceeded expected or allowed bounds.
    ValueError(String),

    /// Attempting to query outside of data limits.
    Bounds(String),

    /// Error related to IO.
    IOError(String),

    /// Propagator detected an impact.
    Impact(i32, Time<TDB>),

    /// Failed to acquire lock on memory.
    LockFailed,
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Convergence(s) | Self::ValueError(s) | Self::Bounds(s) | Self::IOError(s) => {
                write!(f, "{s}")
            }
            Self::Impact(s, t) => {
                let t = t.jd();
                write!(f, "Propagation detected an impact with {s} at time {t}")
            }
            Self::LockFailed => {
                write!(f, "Failed to acquire lock on memory.")
            }
        }
    }
}

impl From<ConvergenceError> for Error {
    fn from(err: ConvergenceError) -> Self {
        match err {
            ConvergenceError::Iterations => {
                Self::Convergence("Maximum number of iterations reached without convergence".into())
            }
            ConvergenceError::NonFinite => {
                Self::Convergence("Non-finite value encountered during evaluation".into())
            }
            ConvergenceError::ZeroDerivative => {
                Self::Convergence("Zero derivative encountered during evaluation".into())
            }
            ConvergenceError::InvalidInput(msg) => Self::Convergence(msg.into()),
        }
    }
}

#[cfg(feature = "pyo3")]
use pyo3::{PyErr, exceptions};

use crate::time::{TDB, Time};

#[cfg(feature = "pyo3")]
impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        match err {
            Error::IOError(s) | Error::Bounds(s) | Error::ValueError(s) | Error::Convergence(s) => {
                Self::new::<exceptions::PyValueError, _>(s)
            }

            Error::LockFailed => {
                Self::new::<exceptions::PyValueError, _>("Failed to acquire lock on memory.")
            }

            Error::Impact(s, t) => Self::new::<exceptions::PyValueError, _>({
                let t = t.jd();
                format!("Propagation detected an impact with {s} at time {t}")
            }),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::IOError(error.to_string())
    }
}

impl From<std::num::ParseIntError> for Error {
    fn from(value: std::num::ParseIntError) -> Self {
        Self::IOError(value.to_string())
    }
}
impl From<std::num::ParseFloatError> for Error {
    fn from(value: std::num::ParseFloatError) -> Self {
        Self::IOError(value.to_string())
    }
}

impl From<ParseError> for Error {
    fn from(value: ParseError) -> Self {
        Self::IOError(value.to_string())
    }
}

impl<T> From<TryLockError<T>> for Error {
    fn from(_: TryLockError<T>) -> Self {
        Self::LockFailed
    }
}
