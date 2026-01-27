//! Basic statistics
use kete_stats::prelude::{Data, UncertainData};
use pyo3::{PyResult, pyfunction};

/// Perform a KS test between two vectors of values.
#[pyfunction]
#[pyo3(name = "ks_test")]
pub fn ks_test_py(sample_a: Vec<f64>, sample_b: Vec<f64>) -> PyResult<f64> {
    let sample_a: Data<f64> = sample_a
        .try_into()
        .expect("Sample A did not contain valid data.");
    let sample_b: Data<f64> = sample_b
        .try_into()
        .expect("Sample B did not contain valid data.");
    Ok(sample_a
        .into_sorted()
        .two_sample_ks_statistic(&sample_b.into_sorted()))
}

/// Fit the reduced chi squared value for a collection of data with uncertainties.
#[pyfunction]
#[pyo3(name = "fit_chi2")]
pub fn fit_chi2_py(data: Vec<f64>, sigmas: Vec<f64>) -> f64 {
    let data: UncertainData<f64> = (data, sigmas)
        .try_into()
        .expect("Data or sigmas did not contain valid data.");
    data.fit_reduced_chi2().unwrap()
}
