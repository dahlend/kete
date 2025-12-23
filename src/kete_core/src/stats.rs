//! # Statistics
//!
//! Commonly used statistical methods.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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
use std::ops::Index;

use crate::{errors::Error, prelude::KeteResult};

/// Finite, sorted, nonempty dataset.
///
/// During construction, NaN and Infs are removed from the dataset.
///
/// Construction hints:
///
/// If data ownership can be given up, then a `Vec<f64>` will not incur any copy/clone
/// as all data will be manipulated in-place. However a slice may be used, but then the
/// data must be cloned/copied.
#[derive(Clone, Debug)]
pub struct ValidData(Box<[f64]>);

impl TryFrom<&[f64]> for ValidData {
    type Error = Error;
    fn try_from(value: &[f64]) -> Result<Self, Self::Error> {
        let mut data: Box<[f64]> = value
            .iter()
            .filter_map(|x| if x.is_finite() { Some(*x) } else { None })
            .collect();
        if data.is_empty() {
            Err(Error::ValueError(
                "Data was either empty or contained only non-finite values (NaN or inf).".into(),
            ))
        } else {
            data.sort_by(f64::total_cmp);
            Ok(Self(data))
        }
    }
}

impl TryFrom<Vec<f64>> for ValidData {
    type Error = Error;

    fn try_from(mut value: Vec<f64>) -> Result<Self, Self::Error> {
        // Switch all negative non-finites to positive inplace
        for x in &mut value {
            if !x.is_finite() & x.is_sign_negative() {
                *x = x.abs();
            }
        }

        // sort everything by total_cmp, which puts all positive non-finite values at the end.
        value.sort_by(f64::total_cmp);

        if let Some(idx) = value.iter().position(|x| !x.is_finite()) {
            value.truncate(idx);
        }

        if value.is_empty() {
            Err(Error::ValueError(
                "Data was either empty or contained only non-finite values (NaN or inf).".into(),
            ))
        } else {
            Ok(Self(value.into_boxed_slice()))
        }
    }
}

impl Index<usize> for ValidData {
    type Output = f64;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl ValidData {
    /// Calculate desired quantile of the provided data.
    ///
    /// Quantile is effectively the same as percentile, but 0.5 quantile == 50% percentile.
    ///
    /// This ignores non-finite values such as inf and nan.
    ///
    /// Quantiles are linearly interpolated between the two closest ranked values.
    ///
    /// If only one valid data point is provided, all quantiles evaluate to that value.
    ///
    /// # Errors
    /// Fails when quant is not between 0 and 1, or if the data does not have any finite
    /// values.
    pub fn quantile(&self, quant: f64) -> KeteResult<f64> {
        if !(0.0..=1.0).contains(&quant) {
            Err(Error::ValueError(
                "Quantile must be between 0.0 and 1.0".into(),
            ))?;
        }
        let data = self.as_slice();
        let n_data = self.len();

        let frac_idx = quant * (n_data - 1) as f64;
        #[allow(
            clippy::cast_sign_loss,
            reason = "By construction this is always positive."
        )]
        let idx = frac_idx.floor() as usize;

        if idx as f64 == frac_idx {
            // exactly on a data point
            Ok(unsafe { *data.get_unchecked(idx) })
        } else {
            // linear interpolation between two points
            let diff = frac_idx - idx as f64;
            unsafe {
                Ok(data.get_unchecked(idx) * (1.0 - diff) + data.get_unchecked(idx + 1) * diff)
            }
        }
    }

    /// Compute the median value of the data.
    #[must_use]
    pub fn median(&self) -> f64 {
        // 0.5 is well defined, infallible
        unsafe { self.quantile(0.5).unwrap_unchecked() }
    }

    /// Compute the mean value of the data.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let n: f64 = self.len() as f64;
        self.0.iter().sum::<f64>() / n
    }

    /// Compute the standard deviation of the data.
    #[must_use]
    pub fn std(&self) -> f64 {
        let n = self.len() as f64;
        let mean = self.mean();
        let mut val = 0.0;
        for v in self.as_slice() {
            val += v.powi(2);
        }
        val /= n;
        (val - mean.powi(2)).sqrt()
    }

    /// Compute the MAD value of the data.
    ///
    /// <https://en.wikipedia.org/wiki/Median_absolute_deviation>
    ///
    /// # Errors
    /// Fails when data does not contain any finite values.
    pub fn mad(&self) -> KeteResult<f64> {
        let median = self.quantile(0.5)?;
        let abs_deviation_from_med: Self = self
            .0
            .iter()
            .map(|d| d - median)
            .collect::<Vec<_>>()
            .as_slice()
            .try_into()?;
        abs_deviation_from_med.quantile(0.5)
    }

    /// Length of the dataset.
    #[must_use]
    #[allow(clippy::len_without_is_empty, reason = "Cannot have empty dataset.")]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Dataset as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Compute the KS Test two sample statistic.
    ///
    /// <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>
    ///
    /// # Errors
    /// Fails when data does not contain any finite values.
    pub fn two_sample_ks_statistic(&self, other: &Self) -> KeteResult<f64> {
        let len_a = self.len();
        let len_b = other.len();

        let mut stat = 0.0;
        let mut ida = 0;
        let mut idb = 0;
        let mut empirical_dist_func_a = 0.0;
        let mut empirical_dist_func_b = 0.0;

        // go through the sorted lists,
        while ida < len_a && idb < len_b {
            let val_a = &self[ida];
            while ida + 1 < len_a && *val_a == other[ida + 1] {
                ida += 1;
            }

            let val_b = &self[idb];
            while idb + 1 < len_b && *val_b == other[idb + 1] {
                idb += 1;
            }

            let min = &val_a.min(*val_b);

            if min == val_a {
                empirical_dist_func_a = (ida + 1) as f64 / (len_a as f64);
                ida += 1;
            }
            if min == val_b {
                empirical_dist_func_b = (idb + 1) as f64 / (len_b as f64);
                idb += 1;
            }

            let diff = (empirical_dist_func_a - empirical_dist_func_b).abs();
            if diff > stat {
                stat = diff;
            }
        }
        Ok(stat)
    }
}

#[cfg(test)]
mod tests {
    use super::{KeteResult, ValidData};

    #[test]
    fn test_median() {
        let data: ValidData = vec![
            -f64::NAN,
            f64::INFINITY,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NAN,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ]
        .as_slice()
        .try_into()
        .unwrap();

        assert!(data.median() == 3.0);
        assert!(data.mean() == 3.0);
        assert!((data.std() - 2_f64.sqrt()).abs() < 1e-13);
        assert!(data.quantile(0.0).unwrap() == 1.0);
        assert!(data.quantile(0.25).unwrap() == 2.0);
        assert!(data.quantile(0.5).unwrap() == 3.0);
        assert!(data.quantile(0.75).unwrap() == 4.0);
        assert!(data.quantile(1.0).unwrap() == 5.0);
        assert!(data.quantile(1.0 / 8.0).unwrap() == 1.5);
        assert!(data.quantile(1.0 / 8.0 + 0.75).unwrap() == 4.5);
    }
    #[test]
    fn test_finite_bad() {
        let data: KeteResult<ValidData> = [f64::NAN, f64::NEG_INFINITY, f64::INFINITY]
            .as_slice()
            .try_into();
        assert!(data.is_err());

        let data2: KeteResult<ValidData> = vec![].as_slice().try_into();
        assert!(data2.is_err());
    }
}
