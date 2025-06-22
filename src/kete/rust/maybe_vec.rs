//! MaybeVec
//! This allows Py03 functions to have polymorphic support for a single value or a vector of values.
//!

use std::fmt::Debug;

use pyo3::IntoPyObjectExt;
use pyo3::types::PyList;
use pyo3::{FromPyObject, IntoPyObject, PyObject, PyResult, Python};

/// Polymorphic support for a single value or a vector of values.
///
/// Functions can accept or return this type to handle the cases where
/// the python function can either take a single value of type `T` or a vector of values of type `T`.
///
/// If the function accepts a `MaybeVec<T>`, it makes sense that the returned
/// value maintains its "single or multiple" nature.
///
/// In order to handle this, the MaybeVec can be converted into a tuple of `(Vec<T>, bool)`
/// where the boolean indicates if the original input was a vector.
///
/// Then when returning a `MaybeVec<T>`, it can be constructed from a tuple of `(Vec<T>, bool)`
///
#[derive(Debug, FromPyObject, IntoPyObject)]
pub enum MaybeVec<T> {
    /// A single value of type T.
    Single(T),

    /// A vector of values of type T.
    Multiple(Vec<T>),
}

/// Return either a single value or a vector of values as a Python object.
///
/// This is the opposite of what MaybeVec does for polymorphic function input.
pub fn maybe_vec_to_pyobj<'py, T: IntoPyObject<'py>>(
    py: Python<'py>,
    value: Vec<T>,
    was_vec: bool,
) -> PyResult<PyObject>
where
    <T as IntoPyObject<'py>>::Output: IntoPyObject<'py>,
{
    if was_vec {
        PyList::new(py, value)?.into_py_any(py)
    } else {
        value.into_iter().next().unwrap().into_py_any(py)
    }
}

impl<T> From<MaybeVec<T>> for (Vec<T>, bool) {
    fn from(maybe_vec: MaybeVec<T>) -> Self {
        match maybe_vec {
            MaybeVec::Single(value) => (vec![value], false),
            MaybeVec::Multiple(vec) => (vec, true),
        }
    }
}

impl<T> From<MaybeVec<T>> for Vec<T> {
    fn from(maybe_vec: MaybeVec<T>) -> Self {
        match maybe_vec {
            MaybeVec::Single(value) => vec![value],
            MaybeVec::Multiple(vec) => vec,
        }
    }
}
