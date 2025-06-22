//! # Errors
//! Errors emitted by ``kete_core``

/// Define all errors which may be raise by this crate, as well as optionally provide
/// conversion to pyo3 error types which allow for the errors to be raised in Python.
use chrono::ParseError;
use std::{error, fmt, io};

/// kete specific result.
pub type KeteResult<T> = Result<T, Error>;

/// Possible Errors which may be raised by this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Numerical method did not converge within the algorithms limits.
    Convergence(String),

    /// Input or variable exceeded expected or allowed bounds.
    ValueError(String),

    /// Querying an SPK file failed due to it missing the requisite data.
    DAFLimits(String),

    /// Attempting to load or convert to/from an Frame of reference which is not known.
    UnknownFrame(i32),

    /// Error related to IO.
    IOError(String),

    /// Propagator detected an impact.
    Impact(i32, f64),
}

impl error::Error for Error {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Convergence(s) | Self::ValueError(s) | Self::DAFLimits(s) | Self::IOError(s) => {
                write!(f, "{s}")
            }
            Self::UnknownFrame(_) => {
                write!(f, "This reference frame is not supported.")
            }
            Self::Impact(s, t) => {
                write!(f, "Propagation detected an impact with {s} at time {t}")
            }
        }
    }
}

#[cfg(feature = "pyo3")]
use pyo3::{PyErr, exceptions};

#[cfg(feature = "pyo3")]
impl From<Error> for PyErr {
    fn from(err: Error) -> Self {
        match err {
            Error::IOError(s)
            | Error::DAFLimits(s)
            | Error::ValueError(s)
            | Error::Convergence(s) => Self::new::<exceptions::PyValueError, _>(s),

            Error::UnknownFrame(_) => {
                Self::new::<exceptions::PyValueError, _>("This reference frame is not supported.")
            }

            Error::Impact(s, t) => Self::new::<exceptions::PyValueError, _>(format!(
                "Propagation detected an impact with {s} at time {t}"
            )),
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
