// BSD 3-Clause License
//
// Copyright (c) 2025, California Institute of Technology
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
