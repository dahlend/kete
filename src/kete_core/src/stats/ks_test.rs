/// Implementation of the two sample KS test statistic.
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
    sample_a.sort_by(|a, b| a.total_cmp(b));

    let mut sample_b = sample_b
        .iter()
        .filter(|x| x.is_finite())
        .copied()
        .collect_vec();
    sample_b.sort_by(|a, b| a.total_cmp(b));

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
