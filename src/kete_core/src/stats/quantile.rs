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

use crate::{errors::Error, prelude::KeteResult};

/// Calculate desired quantile of the provided data.
///
/// Quantile is effectively the same as percentile, but 0.5 quantile == 50% percentile.
///
/// This ignores non-finite values such as inf and nan.
///
/// Quantiles are linearly interpolated between the two closest ranked values.
///
/// If only one valid data point is provided, all quantiles evaluate to that value.
pub fn quantile(data: &[f64], quant: f64) -> KeteResult<f64> {
    if quant <= 0.0 || quant >= 1.0 {
        Err(Error::ValueError(
            "Quantile must be between 0.0 and 1.0".into(),
        ))?;
    }

    let mut data: Box<[f64]> = data
        .iter()
        .filter_map(|x| if x.is_finite() { Some(*x) } else { None })
        .collect();
    data.sort_by(f64::total_cmp);

    let n_data = data.len();

    if n_data == 0 {
        Err(Error::ValueError(
            "Data must have at least 1 finite value.".into(),
        ))?;
    } else if n_data == 1 {
        return Ok(data[0]);
    }

    let frac_idx = quant * (n_data - 1) as f64;
    let idx = frac_idx.floor() as usize;

    if idx as f64 == frac_idx {
        Ok(data[idx])
    } else {
        let diff = frac_idx - idx as f64;
        Ok(data[idx] * (1.0 - diff) + data[idx + 1] * diff)
    }
}

/// Compute the median value of the data.
///
/// This ignores non-finite values such as inf and nan.
pub fn median(data: &[f64]) -> KeteResult<f64> {
    quantile(data, 0.5)
}

/// Compute the MAD value of the data.
///
/// <https://en.wikipedia.org/wiki/Median_absolute_deviation>
///
pub fn mad(data: &[f64]) -> KeteResult<f64> {
    let median = quantile(data, 0.5)?;
    let abs_deviation_from_med: Box<[f64]> = data.iter().map(|d| d - median).collect();
    quantile(&abs_deviation_from_med, 0.5)
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn test_median() {
        let data = vec![
            0.5,
            0.6,
            0.6,
            0.6,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        assert!(median(&data).is_ok());
        assert!(median(&data).unwrap() == 0.6);
    }
    #[test]
    fn test_median_bad() {
        let data = vec![f64::NAN, f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert!(median(&data).is_err());

        let data2 = vec![];
        assert!(median(&data2).is_err());
    }
}
