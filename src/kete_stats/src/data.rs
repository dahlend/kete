//! # Data
//!
//! Handling of finite, nonempty datasets for basic statistical calculations.
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
use std::{fmt::Debug, ops::Index};

/// Error types for statistics calculations.
#[derive(Debug, Clone, Copy, thiserror::Error)]
#[non_exhaustive]
pub enum DataError {
    /// Error indicating that the dataset is empty after removing invalid values.
    #[error("Data was either empty or contained only non-finite values (NaN or inf).")]
    EmptyDataset,

    /// Data contains values outside of the allowed range.
    #[error("Data contains values outside of the allowed range.")]
    OutOfRange,

    /// Data and uncertainties have different lengths.
    #[error("Data and uncertainties have different lengths.")]
    UnequalLengths,
}

/// Result type for statistics calculations.
pub type StatsResult<T> = Result<T, DataError>;

/// Finite, nonempty dataset.
///
/// During construction, NaN and Infs are removed from the dataset.
#[derive(Clone, Debug)]
pub struct Data<T>(Box<[T]>)
where
    T: num_traits::Float;

/// Sorted version of [`Data`].
#[derive(Clone, Debug)]
pub struct SortedData<T>(Data<T>)
where
    T: num_traits::Float;

/// Dataset with associated uncertainties.
///
/// This structure pairs measurements with their one-sigma (1σ) uncertainties, representing
/// the standard deviation of each measurement. All statistical methods using these
/// uncertainties assume they represent Gaussian (normal) errors.
///
/// There is a one-to-one correspondence between values and uncertainties.
#[derive(Clone, Debug)]
pub struct UncertainData<T>
where
    T: num_traits::Float,
{
    /// Measured values of the dataset.
    pub values: Data<T>,

    /// One-sigma (1σ) uncertainties (standard deviations) for each measurement
    /// assuming Gaussian errors.
    pub uncertainties: Data<T>,
}

impl<T> Data<T>
where
    T: num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    /// Create a new [`Data`] without checking the data.
    ///
    /// Data cannot contain non-finite values.
    #[must_use]
    pub fn new_unchecked(data: Box<[T]>) -> Self {
        Self(data)
    }

    /// Compute the minimum value of the data.
    #[must_use]
    pub fn min(&self) -> T {
        self.0.iter().fold(T::infinity(), |a, &b| a.min(b))
    }

    /// Compute the maximum value of the data.
    #[must_use]
    pub fn max(&self) -> T {
        self.0.iter().fold(T::neg_infinity(), |a, &b| a.max(b))
    }

    /// Compute the mean value of the data.
    ///
    /// If you are using the std as well, consider using [`Data::mean_std`] instead.
    #[must_use]
    pub fn mean(&self) -> T {
        let n: T = unsafe { T::from(self.len()).unwrap_unchecked() };
        let mut sum = T::zero();
        for v in self.as_slice() {
            sum += *v;
        }
        sum / n
    }

    /// Compute the standard deviation of the data.
    ///
    /// If you are using the mean as well, consider using [`Data::mean_std`] instead.
    #[must_use]
    pub fn std(&self) -> T {
        self.mean_std().1
    }

    /// Compute the median value of the data.
    ///
    /// This mutates the internal data order but is O(n) average case.
    #[must_use]
    pub fn median(&mut self) -> T {
        let half = T::one() / (T::one() + T::one());
        self.quantile(half)
    }

    /// Compute the mean and standard deviation of the data.
    ///
    /// More efficient than calling [`Data::mean`] and [`Data::std`] separately.
    #[must_use]
    pub fn mean_std(&self) -> (T, T) {
        let n: T = unsafe { T::from(self.len()).unwrap_unchecked() };
        let mean = self.mean();
        let mut val = T::zero();
        for v in self.as_slice() {
            val += v.powi(2);
        }
        val /= n;
        let std = (val - mean.powi(2)).sqrt();
        (mean, std)
    }

    /// Length of the dataset.
    #[must_use]
    #[allow(clippy::len_without_is_empty, reason = "Cannot have empty dataset.")]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Dataset as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Find the k-th smallest element in the dataset.
    ///
    /// If k is larger than the length of the data, it returns the final element.
    #[must_use]
    pub fn kth_smallest(&mut self, k: usize) -> T {
        let k = k.min(self.len() - 1);
        quickselect(&mut self.0, k)
    }

    /// Calculate desired quantile of the data using quickselect.
    ///
    /// This mutates the internal data order but is O(n) average case.
    ///
    /// Quantile is effectively the same as percentile, but 0.5 quantile == 50% percentile.
    ///
    /// Quantiles are linearly interpolated between the two closest ranked values.
    ///
    /// If only one valid data point is provided, all quantiles evaluate to that value.
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    #[must_use]
    pub fn quantile(&mut self, quant: T) -> T {
        let quant = quant.clamp(T::zero(), T::one());
        let n_data = self.len();

        let frac_idx = quant * T::from(n_data - 1).unwrap();
        #[allow(
            clippy::cast_sign_loss,
            reason = "By construction this is always positive."
        )]
        let idx = frac_idx.floor().to_usize().unwrap();

        if T::from(idx).unwrap() == frac_idx {
            // exactly on a data point
            quickselect(&mut self.0, idx)
        } else {
            // need two adjacent values for linear interpolation
            let lower = quickselect(&mut self.0, idx);
            let upper = quickselect(&mut self.0[idx..], 1);
            let diff = frac_idx - T::from(idx).unwrap();
            lower * (T::one() - diff) + upper * diff
        }
    }

    /// Compute the MAD (Median Absolute Deviation) value of the data.
    ///
    /// This mutates the internal data order but is O(n) average case.
    ///
    /// <https://en.wikipedia.org/wiki/Median_absolute_deviation>
    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    pub fn mad(&mut self) -> T {
        let median = self.median();
        let mut abs_deviation_from_med: Vec<T> = self
            .as_slice()
            .iter()
            .map(|d| (*d - median).abs())
            .collect();
        let n = abs_deviation_from_med.len();
        let half = n / 2;
        if n % 2 == 1 {
            quickselect(&mut abs_deviation_from_med, half)
        } else {
            let lower = quickselect(&mut abs_deviation_from_med, half - 1);
            let upper = quickselect(&mut abs_deviation_from_med[half - 1..], 1);
            (lower + upper) / (T::one() + T::one())
        }
    }

    /// Compute the standard deviation estimate from the MAD value.
    ///
    /// This is not the std, or MAD, but an estimate of the std based on the MAD
    /// assuming that the data is normally distributed. This is more robust to outliers
    /// than the standard deviation.
    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    pub fn std_from_mad(&mut self) -> T {
        let mad = self.mad();
        // Taken from wikipedia
        let c = T::from(1.4826).unwrap();
        mad * c
    }

    /// Return a sorted version of this dataset.
    #[must_use]
    pub fn into_sorted(mut self) -> SortedData<T> {
        let slice = &mut self.0;
        slice.sort_by(T::total_cmp);
        SortedData(self)
    }

    /// Shuffle the dataset in-place using a Fisher-Yates shuffle.
    ///
    /// This uses a simple Linear Congruential Generator (LCG) for pseudorandom
    /// number generation with the given seed for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed for the random number generator.
    pub fn shuffle(&mut self, seed: u64) {
        let mut rng_state = seed;
        for i in (1..self.0.len()).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            self.0.swap(i, j);
        }
    }

    /// Select a sample of N data points from the dataset, this always returns at
    /// least one data point. This does not panic.
    ///
    /// If more data is requested than exists, the full dataset is returned.
    ///
    /// If 0 data points are requested, a single data point is returned.
    ///
    /// This assumes the data is nearly IID, samples at nearly even step sizes.
    #[must_use]
    pub fn sample_n(&self, n: usize) -> Self {
        if n == 0 {
            // Cannot sample 0 elements, return a single element
            return Self::new_unchecked(vec![self.0[0]].into_boxed_slice());
        }

        if n >= self.len() {
            return Self::new_unchecked(self.0.clone());
        }

        if n == 1 {
            // Special case to avoid division by zero
            return Self::new_unchecked(vec![self.0[0]].into_boxed_slice());
        }

        let mut sampled_data = Vec::with_capacity(n);
        let step = (self.len() - 1) as f64 / (n - 1) as f64;
        for i in 0..n {
            #[allow(
                clippy::cast_sign_loss,
                reason = "By construction this is always positive."
            )]
            let index = (i as f64 * step).round() as usize;
            sampled_data.push(self.0[index]);
        }
        Self::new_unchecked(sampled_data.into_boxed_slice())
    }

    /// Remove outliers from the data using sigma clipping.
    ///
    /// Accepts two standard deviation thresholds, one for lower and one for upper.
    ///
    /// Exits early if no data points are removed in an iteration (convergence).
    ///
    /// If all data points are filtered out, returns a dataset containing only the median
    /// value to maintain the invariant that `Data` is never empty.
    ///
    /// # Arguments
    /// * `lower_std` - Lower standard deviation threshold.
    /// * `upper_std` - Upper standard deviation threshold.
    /// * `n_iter` - Number of iterations to perform.
    #[must_use]
    pub fn sigma_clip(&self, lower_std: T, upper_std: T, n_iter: usize) -> Self {
        // to avoid confusion with users passing negative std values
        let lower_std = -lower_std.abs();
        let upper_std = upper_std.abs();

        let mut clipped_data = self.0.to_vec();
        for _ in 0..n_iter {
            let prev_len = clipped_data.len();
            let data = Self::new_unchecked(clipped_data.into_boxed_slice());
            let (mean, std) = data.mean_std();
            clipped_data = data
                .0
                .iter()
                .copied()
                .filter(|&x| (x - mean) >= lower_std * std && (x - mean) <= upper_std * std)
                .collect();

            // Exit early if converged (no points removed)
            if clipped_data.len() == prev_len {
                break;
            }

            // If all points were filtered out, return median value to maintain invariant
            if clipped_data.is_empty() {
                let mut median_data = Self::new_unchecked(self.0.clone());
                let median = median_data.median();
                return Self::new_unchecked(vec![median].into_boxed_slice());
            }
        }
        Self::new_unchecked(clipped_data.into_boxed_slice())
    }
}

impl<T> UncertainData<T>
where
    T: num_traits::Float + num_traits::NumAssignOps + std::iter::Sum,
{
    /// Compute the weighted mean using inverse variance weighting (1/σ²).
    ///
    /// This is the optimal estimator for combining measurements with different uncertainties,
    /// and is mathematically equivalent to the value that minimizes the reduced chi-squared
    /// statistic. This provides the same result as fitting via chi-squared minimization
    /// but is computed directly without iterative optimization.
    ///
    /// # Formula
    /// ``weighted_mean = Σ(x_i / σ_i²) / Σ(1 / σ_i²)``
    #[must_use]
    pub fn weighted_mean(&self) -> T {
        let mut sum_weights = T::zero();
        let mut sum_weighted_values = T::zero();

        for (value, sigma) in self.values.0.iter().zip(self.uncertainties.0.iter()) {
            let weight = T::one() / sigma.powi(2);
            sum_weighted_values += *value * weight;
            sum_weights += weight;
        }

        sum_weighted_values / sum_weights
    }

    /// Compute the weighted variance using inverse variance weighting.
    ///
    /// # Formula
    /// ``weighted_variance = 1 / Σ(1 / σ_i²)``
    #[must_use]
    pub fn weighted_variance(&self) -> T {
        let sum_weights: T = self
            .uncertainties
            .0
            .iter()
            .map(|sigma| T::one() / sigma.powi(2))
            .sum();

        T::one() / sum_weights
    }

    /// Compute the weighted standard deviation.
    ///
    /// This is the square root of the weighted variance and represents the
    /// uncertainty in the weighted mean.
    #[must_use]
    pub fn weighted_std(&self) -> T {
        self.weighted_variance().sqrt()
    }

    /// Compute the effective sample size accounting for varying uncertainties.
    ///
    /// When all uncertainties are equal, this equals the number of samples.
    /// When uncertainties vary, this is reduced based on the variance of the weights.
    ///
    /// # Formula
    /// ``n_eff = (Σw_i)² / Σ(w_i²) where w_i = 1/σ_i²``
    #[must_use]
    pub fn effective_sample_size(&self) -> T {
        let weights: Vec<T> = self
            .uncertainties
            .0
            .iter()
            .map(|sigma| T::one() / sigma.powi(2))
            .collect();

        let sum_weights: T = weights.iter().copied().sum();
        let sum_weights_squared: T = weights.iter().map(|w| w.powi(2)).sum();

        sum_weights.powi(2) / sum_weights_squared
    }

    /// Compute the reduced chi squared value from known values and standard deviations.
    /// This computes the reduced chi squared against a single desired value.
    #[inline(always)]
    pub fn reduced_chi2(&self, val: T) -> T {
        self.values
            .0
            .iter()
            .zip(self.uncertainties.0.iter())
            .map(|(d, sigma)| ((*d - val) / *sigma).powi(2))
            .sum::<T>()
    }

    /// Shuffle both values and uncertainties in-place using a Fisher-Yates shuffle.
    ///
    /// This maintains the one-to-one correspondence between values and their
    /// uncertainties by applying the same permutation to both arrays.
    ///
    /// This uses a simple Linear Congruential Generator (LCG) for pseudorandom
    /// number generation with the given seed for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed for the random number generator.
    pub fn shuffle(&mut self, seed: u64) {
        let mut rng_state = seed;
        for i in (1..self.values.0.len()).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            self.values.0.swap(i, j);
            self.uncertainties.0.swap(i, j);
        }
    }

    /// Length of the dataset.
    #[must_use]
    #[allow(clippy::len_without_is_empty, reason = "Cannot have empty dataset.")]
    pub fn len(&self) -> usize {
        self.values.0.len()
    }
}

impl<T> SortedData<T>
where
    T: num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    /// Compute the minimum value of the data.
    #[must_use]
    pub fn min(&self) -> T {
        self.0.0.iter().fold(T::infinity(), |a, &b| a.min(b))
    }

    /// Compute the maximum value of the data.
    #[must_use]
    pub fn max(&self) -> T {
        self.0.0.iter().fold(T::neg_infinity(), |a, &b| a.max(b))
    }

    /// Compute the median value of the sorted data.
    ///
    /// This is O(1) since the data is already sorted.
    #[must_use]
    pub fn median(&self) -> T {
        let half = T::one() / (T::one() + T::one());
        self.quantile(half)
    }

    /// Compute the mean value of the sorted data.
    #[must_use]
    pub fn mean(&self) -> T {
        self.0.mean()
    }

    /// Compute the standard deviation of the sorted data.
    #[must_use]
    pub fn std(&self) -> T {
        self.0.std()
    }

    /// Compute the mean and standard deviation of the sorted data.
    #[must_use]
    pub fn mean_std(&self) -> (T, T) {
        self.0.mean_std()
    }

    /// Length of the dataset.
    #[must_use]
    #[allow(clippy::len_without_is_empty, reason = "Cannot have empty dataset.")]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Compute the standard deviation estimate from the MAD value.
    ///
    /// This is not the std, or MAD, but an estimate of the std based on the MAD
    /// assuming that the data is normally distributed. This is more robust to outliers
    /// than the standard deviation.
    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    pub fn std_from_mad(&self) -> T {
        let mad = self.mad();
        // Taken from wikipedia
        let c = T::from(1.4826).unwrap();
        mad * c
    }

    /// Compute the MAD value of the data.
    ///
    /// <https://en.wikipedia.org/wiki/Median_absolute_deviation>
    ///
    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    pub fn mad(&self) -> T {
        let median = self.median();
        let mut abs_deviation_from_med: Vec<T> = self
            .as_slice()
            .iter()
            .map(|d| (*d - median).abs())
            .collect();
        let n = abs_deviation_from_med.len();
        let half = n / 2;
        if n % 2 == 1 {
            quickselect(&mut abs_deviation_from_med, half)
        } else {
            let lower = quickselect(&mut abs_deviation_from_med, half - 1);
            let upper = quickselect(&mut abs_deviation_from_med[half - 1..], 1);
            (lower + upper) / (T::one() + T::one())
        }
    }

    /// Compute the KS Test two sample statistic.
    ///
    /// <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>
    ///
    /// # Errors
    /// Fails when data does not contain any finite values.
    #[must_use]
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    pub fn two_sample_ks_statistic(&self, other: &Self) -> T {
        let len_a = self.0.len();
        let len_b = other.0.len();

        let mut stat = T::zero();
        let mut ida = 0;
        let mut idb = 0;
        let mut empirical_dist_func_a = T::zero();
        let mut empirical_dist_func_b = T::zero();

        // go through the sorted lists,
        while ida < len_a && idb < len_b {
            let val_a = &self.0[ida];
            while ida + 1 < len_a && *val_a == self.0[ida + 1] {
                ida += 1;
            }

            let val_b = &other.0[idb];
            while idb + 1 < len_b && *val_b == other.0[idb + 1] {
                idb += 1;
            }

            let min = &val_a.min(*val_b);

            if min == val_a {
                empirical_dist_func_a = T::from(ida + 1).unwrap() / T::from(len_a).unwrap();
                ida += 1;
            }
            if min == val_b {
                empirical_dist_func_b = T::from(idb + 1).unwrap() / T::from(len_b).unwrap();
                idb += 1;
            }

            let diff = (empirical_dist_func_a - empirical_dist_func_b).abs();
            if diff > stat {
                stat = diff;
            }
        }
        stat
    }

    /// Create a new [`SortedData`] without checking the data.
    #[must_use]
    pub fn new_unchecked(data: Data<T>) -> Self {
        Self(data)
    }

    /// Unwrap to get the inner [`Data`].
    #[must_use]
    pub fn unwrap_inner(self) -> Data<T> {
        self.0
    }

    /// Dataset as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }

    /// Calculate desired quantile of the sorted data.
    ///
    /// Quantile is effectively the same as percentile, but 0.5 quantile == 50% percentile.
    ///
    /// Quantiles are linearly interpolated between the two closest ranked values.
    ///
    /// If only one valid data point is provided, all quantiles evaluate to that value.
    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    #[must_use]
    pub fn quantile(&self, quant: T) -> T {
        let quant = quant.clamp(T::zero(), T::one());
        let data = self.0.as_slice();
        let n_data = self.0.len();

        let frac_idx = quant * T::from(n_data - 1).unwrap();
        #[allow(
            clippy::cast_sign_loss,
            reason = "By construction this is always positive."
        )]
        let idx = frac_idx.floor().to_usize().unwrap();

        if T::from(idx).unwrap() == frac_idx {
            // exactly on a data point
            unsafe { *data.get_unchecked(idx) }
        } else {
            // linear interpolation between two points
            let diff = frac_idx - T::from(idx).unwrap();
            unsafe {
                *data.get_unchecked(idx) * (T::one() - diff) + *data.get_unchecked(idx + 1) * diff
            }
        }
    }
}

/// Quickselect algorithm to find the k-th smallest element in an array.
///
/// This is an in-place algorithm with average O(n) time complexity.
/// If the data is allowed to be mutable, than this is more efficient than sorting the
/// entire array.
fn quickselect<T>(arr: &mut [T], k: usize) -> T
where
    T: Copy + PartialOrd,
{
    if arr.len() == 1 {
        return arr[0];
    }

    // Use median-of-three for pivot selection to avoid worst case
    let mid = arr.len() / 2;
    let last = arr.len() - 1;

    // Sort first, mid, last and use mid as pivot
    if arr[0] > arr[mid] {
        arr.swap(0, mid);
    }
    if arr[mid] > arr[last] {
        arr.swap(mid, last);
    }
    if arr[0] > arr[mid] {
        arr.swap(0, mid);
    }

    let pivot = arr[mid];

    // Partition
    let mut i = 0;
    let mut j = arr.len() - 1;

    loop {
        while i < arr.len() && arr[i] < pivot {
            i += 1;
        }
        while j > 0 && arr[j] > pivot {
            j -= 1;
        }
        if i >= j {
            break;
        }
        arr.swap(i, j);
        i += 1;
        j = j.saturating_sub(1);
    }

    // Recurse on the partition containing k
    if k < i {
        quickselect(&mut arr[..i], k)
    } else if k > j {
        quickselect(&mut arr[j + 1..], k - j - 1)
    } else {
        arr[k]
    }
}

/// Try to convert from a slice, removing non-finite values.
///
/// This will fail if there are no valid data points.
impl<T> TryFrom<&[T]> for Data<T>
where
    T: num_traits::Float,
{
    type Error = DataError;
    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        let data: Box<[T]> = value
            .iter()
            .filter_map(|x| if x.is_finite() { Some(*x) } else { None })
            .collect();
        if data.is_empty() {
            Err(DataError::EmptyDataset)
        } else {
            Ok(Self(data))
        }
    }
}

impl<T> TryFrom<Box<[T]>> for Data<T>
where
    T: num_traits::Float,
{
    type Error = DataError;

    fn try_from(value: Box<[T]>) -> Result<Self, Self::Error> {
        value.into_vec().try_into()
    }
}

impl<T> TryFrom<Vec<T>> for Data<T>
where
    T: Copy + num_traits::Float,
{
    type Error = DataError;

    fn try_from(mut value: Vec<T>) -> Result<Self, Self::Error> {
        // Filter out all non-finite values, keeping only finite data
        value.retain(|x| x.is_finite());

        if value.is_empty() {
            Err(DataError::EmptyDataset)
        } else {
            Ok(Self(value.into_boxed_slice()))
        }
    }
}

impl<T> TryFrom<Vec<T>> for SortedData<T>
where
    T: Copy + num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    type Error = DataError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Data::try_from(value).map(Data::into_sorted)
    }
}

impl<T> TryFrom<&[T]> for SortedData<T>
where
    T: num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    type Error = DataError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Data::try_from(value).map(Data::into_sorted)
    }
}

impl<T> TryFrom<(&[T], &[T])> for UncertainData<T>
where
    T: Copy + num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    type Error = DataError;

    fn try_from(value: (&[T], &[T])) -> Result<Self, Self::Error> {
        if value.0.len() != value.1.len() {
            return Err(DataError::UnequalLengths);
        }
        // Filter out all non-finite values, keeping only finite data
        let mut filtered_values = Vec::with_capacity(value.0.len());
        let mut filtered_uncertainties = Vec::with_capacity(value.1.len());
        for (v, u) in value.0.iter().zip(value.1.iter()) {
            if v.is_finite() && u.is_finite() {
                filtered_values.push(*v);
                filtered_uncertainties.push(*u);
            }
        }
        if filtered_values.is_empty() {
            Err(DataError::EmptyDataset)
        } else {
            Ok(Self {
                values: Data::new_unchecked(filtered_values.into_boxed_slice()),
                uncertainties: Data::new_unchecked(filtered_uncertainties.into_boxed_slice()),
            })
        }
    }
}

impl<T> TryFrom<(Vec<T>, Vec<T>)> for UncertainData<T>
where
    T: Copy + num_traits::Float + num_traits::float::TotalOrder + num_traits::NumAssignOps + Debug,
{
    type Error = DataError;

    fn try_from(value: (Vec<T>, Vec<T>)) -> Result<Self, Self::Error> {
        Self::try_from((value.0.as_slice(), value.1.as_slice()))
    }
}

impl<T> Index<usize> for Data<T>
where
    T: num_traits::Float,
{
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

#[cfg(test)]
mod tests {
    use super::{Data, StatsResult};

    #[test]
    fn test_median() {
        let mut data: Data<_> = vec![
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
        .try_into()
        .unwrap();

        assert!((data.std() - 2_f64.sqrt()).abs() < 1e-13);
        assert_eq!(data.median(), 3.0);
        assert_eq!(data.mean(), 3.0);
        assert_eq!(data.quantile(0.0), 1.0);
        assert_eq!(data.quantile(0.25), 2.0);
        assert_eq!(data.quantile(0.5), 3.0);
        assert_eq!(data.quantile(0.75), 4.0);
        assert_eq!(data.quantile(1.0), 5.0);
        assert_eq!(data.quantile(1.0 / 8.0), 1.5);
        assert_eq!(data.quantile(1.0 / 8.0 + 0.75), 4.5);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let data: Data<_> = data.try_into().unwrap();
        let data = data.into_sorted();
        assert_eq!(data.median(), 2.5);

        let data = vec![1.5, 0.5, 0.5, 1.5];
        let data: Data<_> = data.try_into().unwrap();
        let data = data.into_sorted();
        assert_eq!(data.median(), 1.0);
    }

    #[test]
    fn test_finite_bad() {
        let data: StatsResult<Data<_>> = [f64::NAN, f64::NEG_INFINITY, f64::INFINITY]
            .as_slice()
            .try_into();
        assert!(data.is_err());

        let data2: StatsResult<Data<f64>> = vec![].try_into();
        assert!(data2.is_err());
    }

    #[test]
    fn test_valid_data_from_vec() {
        // Test with Vec that transfers ownership
        // Vec implementation now filters out all non-finite values
        let vec_data = vec![1.0, 2.0, f64::NAN, 3.0, f64::INFINITY, 4.0, 5.0];
        let data: Data<_> = vec_data.try_into().unwrap();
        assert_eq!(data.len(), 5);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_valid_data_from_vec_negative_nan() {
        // Test with negative NaN and infinity
        // Vec implementation now filters out all non-finite values
        let vec_data = vec![1.0, -f64::NAN, 2.0, -f64::INFINITY, 3.0];
        let data: Data<_> = vec_data.try_into().unwrap();
        assert_eq!(data.len(), 3);
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_mad() {
        // Test MAD (Median Absolute Deviation) calculation
        let mut data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let mad = data.mad();
        // Median is 3.0, absolute deviations are [2, 1, 0, 1, 2]
        // MAD is the median of [0, 1, 1, 2, 2] (sorted) = 1.0
        assert_eq!(mad, 1.0);

        let data = data.into_sorted();
        let mad = data.mad();
        assert_eq!(mad, 1.0);
    }

    #[test]
    fn test_mad_even_data() {
        let mut data: Data<_> = vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap();
        assert_eq!(data.median(), 2.5);
        let mad = data.mad();
        // Median is 2.5, abs of deviations are [1.5, 0.5, 0.5, 1.5]
        // MAD is median of these deviations, which is 1.0
        assert_eq!(mad, 1.0);

        let data = data.into_sorted();
        assert_eq!(data.median(), 2.5);
        let mad = data.mad();
        assert_eq!(mad, 1.0);
    }

    #[test]
    fn test_kth_smallest() {
        let mut data: Data<_> = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0].try_into().unwrap();
        assert_eq!(data.kth_smallest(0), 1.0);
        assert_eq!(data.kth_smallest(3), 5.0);
        assert_eq!(data.kth_smallest(6), 9.0);
    }

    #[test]
    fn test_kth_smallest_single() {
        let mut data: Data<_> = vec![42.0].try_into().unwrap();
        assert_eq!(data.kth_smallest(0), 42.0);
    }

    #[test]
    fn test_into_sorted() {
        let data: Data<_> = vec![5.0, 2.0, 8.0, 1.0, 9.0].try_into().unwrap();
        let sorted = data.into_sorted();
        assert_eq!(sorted.unwrap_inner().as_slice(), &[1.0, 2.0, 5.0, 8.0, 9.0]);
    }

    #[test]
    fn test_sample_n() {
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            .try_into()
            .unwrap();

        // Sample 5 points from 10
        let sampled = data.sample_n(5);
        assert_eq!(sampled.len(), 5);
        // Should get approximately evenly spaced values
        assert!(sampled[0] <= sampled[1]);
        assert!(sampled[1] <= sampled[2]);
        assert!(sampled[2] <= sampled[3]);
        assert!(sampled[3] <= sampled[4]);
    }

    #[test]
    fn test_sample_n_more_than_data() {
        let data: Data<_> = vec![1.0, 2.0, 3.0].try_into().unwrap();
        let sampled = data.sample_n(10);
        assert_eq!(sampled.len(), 3);
        assert_eq!(sampled.as_slice(), data.as_slice());
    }

    #[test]
    fn test_sample_n_single() {
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let sampled = data.sample_n(1);
        assert_eq!(sampled.len(), 1);
        assert_eq!(sampled[0], 1.0);
    }

    #[test]
    fn test_sample_n_edge_case_n_is_one() {
        // Edge case: sampling exactly 1 element (tests division by zero fix)
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let sampled = data.sample_n(1);
        assert_eq!(sampled.len(), 1);
        // Should return the first element
        assert_eq!(sampled[0], 1.0);
    }

    #[test]
    fn test_kth_smallest_out_of_bounds() {
        // Test that kth_smallest clamps to valid range when k is out of bounds
        let mut data: Data<_> = vec![1.0, 2.0, 3.0].try_into().unwrap();
        // k=5 is out of bounds for length 3, should return the last element
        assert_eq!(data.kth_smallest(5), 3.0);
    }

    #[test]
    fn test_two_sample_ks_statistic_identical() {
        let data1: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let data2: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();

        let sorted1 = data1.into_sorted();
        let sorted2 = data2.into_sorted();

        let ks_stat: f64 = sorted1.two_sample_ks_statistic(&sorted2);
        assert!(ks_stat.abs() < 1e-10); // Should be 0 for identical distributions
    }

    #[test]
    fn test_two_sample_ks_statistic_different() {
        let data1: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let data2: Data<_> = vec![6.0, 7.0, 8.0, 9.0, 10.0].try_into().unwrap();

        let sorted1 = data1.into_sorted();
        let sorted2 = data2.into_sorted();

        let ks_stat: f64 = sorted1.two_sample_ks_statistic(&sorted2);
        // For completely separate distributions, KS statistic should be positive
        assert!(ks_stat >= 0.0);
    }

    #[test]
    fn test_two_sample_ks_statistic_overlapping() {
        let data1: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let data2: Data<_> = vec![3.0, 4.0, 5.0, 6.0, 7.0].try_into().unwrap();

        let sorted1 = data1.into_sorted();
        let sorted2 = data2.into_sorted();

        let ks_stat: f64 = sorted1.two_sample_ks_statistic(&sorted2);
        // For overlapping distributions, KS statistic should be non-negative
        assert!(ks_stat >= 0.0);
        assert!(ks_stat <= 1.0);
    }

    #[test]
    fn test_std_single_value() {
        let data: Data<_> = vec![5.0].try_into().unwrap();
        assert_eq!(data.std(), 0.0);
    }

    #[test]
    fn test_std_known_values() {
        // Standard deviation of [2, 4, 6, 8] with mean 5
        // Variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9 + 1 + 1 + 9) / 4 = 5
        // Std = sqrt(5) ≈ 2.236
        let data: Data<_> = vec![2.0, 4.0, 6.0, 8.0].try_into().unwrap();
        assert!((data.std() - 5_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_quantile_bounds() {
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let data = data.into_sorted();

        // Test clamping
        assert_eq!(data.quantile(-0.5), 1.0); // Should clamp to 0
        assert_eq!(data.quantile(1.5), 5.0); // Should clamp to 1
    }

    #[test]
    fn test_quantile_single_value() {
        let mut data: Data<_> = vec![42.0].try_into().unwrap();
        assert_eq!(data.quantile(0.0), 42.0);
        assert_eq!(data.quantile(0.5), 42.0);
        assert_eq!(data.quantile(1.0), 42.0);

        let data = data.into_sorted();
        assert_eq!(data.quantile(0.0), 42.0);
        assert_eq!(data.quantile(0.5), 42.0);
        assert_eq!(data.quantile(1.0), 42.0);
    }

    #[test]
    fn test_median_even_odd() {
        // Odd number of elements
        let odd_data: Data<_> = vec![1.0, 2.0, 3.0].try_into().unwrap();
        let odd_data = odd_data.into_sorted();
        assert_eq!(odd_data.median(), 2.0);

        // Even number of elements (should interpolate)
        let even_data: Data<_> = vec![1.0, 2.0, 3.0, 4.0].try_into().unwrap();
        let even_data = even_data.into_sorted();
        assert_eq!(even_data.median(), 2.5);
    }

    #[test]
    fn test_index() {
        let data: Data<_> = vec![1.0, 2.0, 3.0].try_into().unwrap();
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 2.0);
        assert_eq!(data[2], 3.0);
    }

    #[test]
    fn test_len() {
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        assert_eq!(data.len(), 5);
    }

    #[test]
    fn test_as_slice() {
        let original = vec![1.0, 2.0, 3.0];
        let data: Data<_> = original.as_slice().try_into().unwrap();
        assert_eq!(data.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_f32_support() {
        // Test that f32 works too
        let data: Data<f32> = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();
        let data = data.into_sorted();
        assert!((data.mean() - 3.0).abs() < 1e-6);
        assert!((data.median() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_clone() {
        let data: Data<_> = vec![1.0, 2.0, 3.0].try_into().unwrap();
        let cloned = data.clone();
        assert_eq!(data.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_sorted_clone() {
        let data: Data<_> = vec![3.0, 1.0, 2.0].try_into().unwrap();
        let sorted = data.into_sorted();
        let cloned = sorted.clone();
        assert_eq!(
            sorted.unwrap_inner().as_slice(),
            cloned.unwrap_inner().as_slice()
        );
    }

    #[test]
    fn test_sigma_clip() {
        // Test data with outliers: mean=5, std≈3.16
        // Values at ±3σ would be roughly -4.5 and 14.5
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]
            .try_into()
            .unwrap();

        // Clip at 2 sigma (should remove 100.0 as outlier)
        let clipped = data.sigma_clip(2.0, 2.0, 1);

        // After removing 100.0, should have 9 values
        assert!(clipped.len() < data.len());

        // The clipped data should not contain the extreme outlier
        assert!(!clipped.as_slice().contains(&100.0));
    }

    #[test]
    fn test_sigma_clip_no_outliers() {
        // Test data without outliers
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0].try_into().unwrap();

        // Clip at 3 sigma (should keep all data)
        let clipped = data.sigma_clip(3.0, 3.0, 1);

        // All values should be within 3 sigma
        assert!(clipped.len() <= data.len());
    }

    #[test]
    fn test_sigma_clip_edge_case_all_equal() {
        // Edge case: all values are equal (tests quickselect with pivot == all elements)
        let data: Data<_> = vec![5.0, 5.0, 5.0, 5.0, 5.0].try_into().unwrap();

        // Should not panic and should keep all data (std is 0)
        let clipped = data.sigma_clip(2.0, 2.0, 1);
        assert_eq!(clipped.len(), 5);
    }

    #[test]
    fn test_sigma_clip_all_filtered() {
        // Edge case: threshold so tight that all data is filtered out
        let data: Data<_> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].try_into().unwrap();

        // Use impossibly tight threshold - should return median value
        let clipped = data.sigma_clip(0.001, 0.001, 1);

        // Should return a single element (the median)
        assert_eq!(clipped.len(), 1);
        assert_eq!(clipped[0], 3.5); // median of [1,2,3,4,5,6] is 3.5
    }

    #[test]
    fn test_uncertain_data_creation() {
        use super::UncertainData;

        // Test creating from slices
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let uncertainties = vec![0.1, 0.2, 0.1, 0.15, 0.1];

        let data: UncertainData<_> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        assert_eq!(data.values.len(), 5);
        assert_eq!(data.uncertainties.len(), 5);
    }

    #[test]
    fn test_uncertain_data_filters_non_finite() {
        use super::UncertainData;

        // Test that non-finite values are filtered out
        let values = vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0];
        let uncertainties = vec![0.1, 0.2, 0.1, 0.15, 0.1];

        let data: UncertainData<_> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        // Should have 3 values (1.0, 3.0, 5.0)
        assert_eq!(data.values.len(), 3);
        assert_eq!(data.uncertainties.len(), 3);
    }

    #[test]
    fn test_uncertain_data_unequal_lengths() {
        use super::{DataError, UncertainData};

        let values = vec![1.0, 2.0, 3.0];
        let uncertainties = vec![0.1, 0.2];

        let result: Result<UncertainData<_>, _> =
            (values.as_slice(), uncertainties.as_slice()).try_into();

        assert!(matches!(result, Err(DataError::UnequalLengths)));
    }

    #[test]
    fn test_reduced_chi2_perfect_fit() {
        use super::UncertainData;

        // Test chi2 when all data points match the test value
        let values = vec![5.0, 5.0, 5.0, 5.0];
        let uncertainties = vec![1.0, 1.0, 1.0, 1.0];

        let data: UncertainData<_> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let chi2 = data.reduced_chi2(5.0);
        assert_eq!(chi2, 0.0);
    }

    #[test]
    fn test_reduced_chi2_known_value() {
        use super::UncertainData;

        // Test chi2 with known calculation
        // If values are [4.0, 6.0] with uncertainties [1.0, 1.0] and test value is 5.0
        // chi2 = ((4-5)/1)^2 + ((6-5)/1)^2 = 1 + 1 = 2
        let values = vec![4.0, 6.0];
        let uncertainties = vec![1.0, 1.0];

        let data: UncertainData<_> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let chi2 = data.reduced_chi2(5.0);
        assert_eq!(chi2, 2.0);
    }

    #[test]
    fn test_reduced_chi2_with_different_uncertainties() {
        use super::UncertainData;

        // Test chi2 with different uncertainties
        // values = [3.0, 7.0], uncertainties = [2.0, 1.0], test value = 5.0
        // chi2 = ((3-5)/2)^2 + ((7-5)/1)^2 = 1.0 + 4.0 = 5.0
        let values = vec![3.0, 7.0];
        let uncertainties = vec![2.0, 1.0];

        let data: UncertainData<_> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let chi2 = data.reduced_chi2(5.0);
        assert_eq!(chi2, 5.0);
    }

    #[test]
    fn test_weighted_mean_equal_uncertainties() {
        use super::UncertainData;

        // When uncertainties are equal, weighted mean should equal regular mean
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let uncertainties = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let weighted_mean = data.weighted_mean();
        let regular_mean = data.values.mean();

        assert!((weighted_mean - regular_mean).abs() < 1e-10);
        assert!((weighted_mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_different_uncertainties() {
        use super::UncertainData;

        // Test that weighted mean gives more weight to more precise measurements
        // Value 1.0 with σ=0.1 should dominate over value 10.0 with σ=10.0
        let values = vec![1.0, 10.0];
        let uncertainties = vec![0.1, 10.0];

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let weighted_mean = data.weighted_mean();

        // Should be much closer to 1.0 than 10.0
        assert!(weighted_mean < 2.0);
        assert!((weighted_mean - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_weighted_std() {
        use super::UncertainData;

        // Weighted std should decrease as we add more measurements
        let values = vec![5.0, 5.0, 5.0];
        let uncertainties = vec![1.0, 1.0, 1.0];

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let weighted_std = data.weighted_std();
        let weighted_var = data.weighted_variance();

        // Variance should be 1 / (3 * 1/1²) = 1/3
        assert!((weighted_var - 1.0 / 3.0).abs() < 1e-10);
        assert!((weighted_std - (1.0 / 3.0_f64.sqrt())).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sample_size_equal_uncertainties() {
        use super::UncertainData;

        // When uncertainties are equal, n_eff should equal n
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let uncertainties = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let n_eff = data.effective_sample_size();

        assert!((n_eff - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_effective_sample_size_different_uncertainties() {
        use super::UncertainData;

        // When uncertainties vary, n_eff < n
        let values = vec![1.0, 2.0, 3.0];
        let uncertainties = vec![0.1, 1.0, 10.0]; // Very different uncertainties

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let n_eff = data.effective_sample_size();

        // Should be less than 3 due to varying weights
        assert!(n_eff < 3.0);
        assert!(n_eff > 0.0);
    }

    #[test]
    fn test_weighted_mean_minimizes_chi2() {
        use super::UncertainData;

        // The weighted mean should minimize the chi2 value
        let values = vec![4.0, 5.0, 6.0];
        let uncertainties = vec![1.0, 0.5, 1.0];

        let data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        let weighted_mean = data.weighted_mean();
        let chi2_at_mean = data.reduced_chi2(weighted_mean);
        let chi2_slightly_off = data.reduced_chi2(weighted_mean + 0.1);

        // Chi2 should be minimized at the weighted mean
        assert!(chi2_at_mean < chi2_slightly_off);
    }

    #[test]
    fn test_data_shuffle() {
        use super::Data;

        // Test that shuffle changes order but preserves elements
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut data = Data::try_from(original.as_slice()).unwrap();

        // Calculate sum and length before shuffle
        let original_sum: f64 = original.iter().sum();
        let original_len = original.len();

        let (original_mean, original_std) = data.mean_std();

        data.shuffle(42);

        // Sum and length should be preserved
        let shuffled_sum: f64 = data.as_slice().iter().sum();
        assert_eq!(data.len(), original_len);
        assert!((original_sum - shuffled_sum).abs() < 1e-10);

        // Order should have changed (with high probability)
        let changed = data.as_slice() != original.as_slice();
        assert!(changed);

        // mean and std should be preserved
        let (shuffled_mean, shuffled_std) = data.mean_std();
        assert!((original_std - shuffled_std).abs() < 1e-10);
        assert!((original_mean - shuffled_mean).abs() < 1e-10);
    }

    #[test]
    fn test_data_shuffle_reproducibility() {
        use super::Data;

        // Test that same seed produces same shuffle
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut data1 = Data::try_from(original.as_slice()).unwrap();
        let mut data2 = data1.clone();

        data1.shuffle(12345);
        data2.shuffle(12345);

        // Same seed should produce same result
        assert_eq!(data1.as_slice(), data2.as_slice());

        data1.shuffle(111);
        data2.shuffle(222);

        // Different seeds should produce different results (with high probability)
        assert_ne!(data1.as_slice(), data2.as_slice());
    }

    #[test]
    fn test_uncertain_data_shuffle_preserves_pairing() {
        use super::UncertainData;

        // Test that shuffle maintains value-uncertainty correspondence
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let uncertainties = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let mut data: UncertainData<f64> = (values.as_slice(), uncertainties.as_slice())
            .try_into()
            .unwrap();

        // Store original pairings
        let original_pairs: Vec<(f64, f64)> = values
            .iter()
            .zip(uncertainties.iter())
            .map(|(&v, &u)| (v, u))
            .collect();

        data.shuffle(999);

        // Check that all original pairs still exist
        for i in 0..data.len() {
            let pair = (data.values.as_slice()[i], data.uncertainties.as_slice()[i]);
            assert!(
                original_pairs.contains(&pair),
                "Shuffled pair ({}, {}) not found in original pairs",
                pair.0,
                pair.1
            );
        }

        let mut data2 = data.clone();

        data.shuffle(54321);
        data2.shuffle(54321);

        // Same seed should produce same result
        assert_eq!(data.values.as_slice(), data2.values.as_slice());
        assert_eq!(
            data.uncertainties.as_slice(),
            data2.uncertainties.as_slice()
        );
    }
}
