use super::newton_raphson;

/// Compute the reduced chi squared value from known values and standard deviations.
/// This computes the reduced chi squared against a single desired value.
#[inline(always)]
pub fn reduced_chi2(data: &[f64], sigmas: &[f64], val: f64) -> f64 {
    debug_assert_eq!(
        data.len(),
        sigmas.len(),
        "Data and sigmas must have the same length"
    );
    data.iter()
        .zip(sigmas)
        .map(|(d, sigma)| ((d - val) / sigma).powi(2))
        .sum::<f64>()
}

/// Compute the derivative of reduced chi squared value with respect to the set value.
#[inline(always)]
pub fn reduced_chi2_der(data: &[f64], sigmas: &[f64], val: f64) -> f64 {
    debug_assert_eq!(
        data.len(),
        sigmas.len(),
        "Data and sigmas must have the same length"
    );
    data.iter()
        .zip(sigmas)
        .map(|(d, sigma)| 2.0 * (val - d) / sigma.powi(2))
        .sum::<f64>()
}

/// Compute the second derivative of reduced chi squared value with respect to the set value.
#[inline(always)]
fn reduced_chi2_der_der(sigmas: &[f64]) -> f64 {
    sigmas.iter().map(|sigma| 2.0 / sigma.powi(2)).sum::<f64>()
}

/// Given a collection of data and standard deviations, fit the best reduced chi squared value
/// for the provided data.
pub fn fit_reduced_chi2(data: &[f64], sigmas: &[f64]) -> f64 {
    let cost = |val: f64| -> f64 { reduced_chi2_der(data, sigmas, val) / sigmas.len() as f64 };
    let der = |_: f64| -> f64 { reduced_chi2_der_der(sigmas) / sigmas.len() as f64 };
    newton_raphson(cost, der, data[0], 1e-8).unwrap()
}
