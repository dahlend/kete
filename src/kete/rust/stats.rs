//! Python wrapper for [`kete_stats::prelude::Data`].

use kete_stats::prelude::Data;
use pyo3::prelude::*;

/// Dataset with summary statistics.
///
/// Wraps a collection of floating-point values and exposes common
/// summary statistics.  Non-finite values are silently dropped during
/// construction, and the original insertion order is preserved.
///
/// Parameters
/// ----------
/// values :
///     Sequence of values.  Non-finite entries (NaN, inf) are removed.
///
/// Raises
/// ------
/// ValueError
///     If no finite values remain after filtering.
///
/// Examples
/// --------
/// >>> from kete import Data
/// >>> s = Data([3.0, 1.0, 2.0, 4.0, 5.0])
/// >>> s.median
/// 3.0
/// >>> s.quantile(0.5)
/// 3.0
#[pyclass(frozen, module = "kete", name = "Data")]
#[derive(Clone, Debug)]
pub struct PyData(pub Data<f64>);

#[pymethods]
impl PyData {
    #[new]
    fn new(values: Vec<f64>) -> PyResult<Self> {
        let data = Data::try_from(values).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "No finite values provided; cannot create Data.",
            )
        })?;
        Ok(Self(data))
    }

    /// Median value.
    #[getter]
    fn median(&self) -> f64 {
        self.0.clone().median()
    }

    /// Standard deviation.
    #[getter]
    fn std(&self) -> f64 {
        self.0.std()
    }

    /// Arithmetic mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.0.mean()
    }

    /// Lower bound of the 95% credible interval (2.5th percentile).
    #[getter]
    fn ci_low(&self) -> f64 {
        self.0.clone().quantile(0.025)
    }

    /// Upper bound of the 95% credible interval (97.5th percentile).
    #[getter]
    fn ci_high(&self) -> f64 {
        self.0.clone().quantile(0.975)
    }

    /// Minimum value.
    #[getter]
    fn min(&self) -> f64 {
        self.0.min()
    }

    /// Maximum value.
    #[getter]
    fn max(&self) -> f64 {
        self.0.max()
    }

    /// Median Absolute Deviation.
    #[getter]
    fn mad(&self) -> f64 {
        self.0.clone().mad()
    }

    /// MAD-based standard deviation estimate (MAD * 1.4826).
    #[getter]
    fn std_from_mad(&self) -> f64 {
        self.0.clone().std_from_mad()
    }

    /// Number of values.
    #[getter]
    fn n(&self) -> usize {
        self.0.len()
    }

    /// The values as a list (original insertion order).
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.0.as_slice().to_vec()
    }

    /// Return the value at a given quantile (0.0 to 1.0).
    fn quantile(&self, q: f64) -> f64 {
        self.0.clone().quantile(q)
    }

    fn __repr__(&self) -> String {
        format!("{self}")
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }
}

impl std::fmt::Display for PyData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut clone = self.0.clone();
        write!(
            f,
            "Data(median={:.4}, std={:.4}, ci=[{:.4}, {:.4}], n={})",
            clone.median(),
            self.0.std(),
            clone.quantile(0.025),
            clone.quantile(0.975),
            self.0.len()
        )
    }
}
