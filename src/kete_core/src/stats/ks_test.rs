/// Implementation of the two sample KS test statistic.
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
use itertools::Itertools;

use crate::errors::{Error, KeteResult};

/// Compute the KS Test two sample statistic.
///
/// <https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test>
///
/// This ignores NAN or INF values in the samples.
///
pub fn two_sample_ks_statistic(sample_a: &[f64], sample_b: &[f64]) -> KeteResult<f64> {
    // Sort the two inputs and drop nan/inf
    let mut sample_a = sample_a
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect_vec();
    sample_a.sort_by(f64::total_cmp);

    let mut sample_b = sample_b
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect_vec();
    sample_b.sort_by(f64::total_cmp);

    let len_a = sample_a.len();
    let len_b = sample_b.len();

    if len_a == 0 || len_b == 0 {
        return Err(Error::ValueError(
            "Both samples must contain at least one finite value.".into(),
        ));
    }

    let mut stat = 0.0;
    let mut ida = 0;
    let mut idb = 0;
    let mut empirical_dist_func_a = 0.0;
    let mut empirical_dist_func_b = 0.0;

    // go through the sorted lists,
    while ida < len_a && idb < len_b {
        let val_a = &sample_a[ida];
        while ida + 1 < len_a && *val_a == sample_a[ida + 1] {
            ida += 1;
        }

        let val_b = &sample_b[idb];
        while idb + 1 < len_b && *val_b == sample_b[idb + 1] {
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
