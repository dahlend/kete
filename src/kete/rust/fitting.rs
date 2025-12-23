//! Basic statistics
use kete_core::{fitting, stats};
use pyo3::{PyResult, pyfunction};

/// Perform a KS test between two vectors of values.
#[pyfunction]
#[pyo3(name = "ks_test")]
pub fn ks_test_py(sample_a: Vec<f64>, sample_b: Vec<f64>) -> PyResult<f64> {
    let sample_a: stats::ValidData = sample_a.try_into()?;
    let sample_b: stats::ValidData = sample_b.try_into()?;
    Ok(sample_a.two_sample_ks_statistic(&sample_b)?)
}

/// Fit the reduced chi squared value for a collection of data with uncertainties.
#[pyfunction]
#[pyo3(name = "fit_chi2")]
pub fn fit_chi2_py(data: Vec<f64>, sigmas: Vec<f64>) -> f64 {
    assert_eq!(data.len(), sigmas.len());
    fitting::fit_reduced_chi2(&data, &sigmas).unwrap()
}
