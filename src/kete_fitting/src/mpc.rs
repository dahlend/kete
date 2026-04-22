const PRELOAD_OBS_ERRORS: &[u8] = include_bytes!("../../kete_core/data/obs_errs.csv");

/// MPC observation residual statistics, used to estimate expected residuals for a given
/// observatory code.
#[derive(Debug, Clone)]
pub struct ObsResiduals {
    /// MPC observatory code.
    pub code: String,

    /// Number of observations used to compute these statistics.
    pub n_obs: u32,

    /// Overall standard deviation of residuals in arcseconds, norm of RA and Dec.
    pub std: f32,

    /// The 2.5th percentile of RA residuals in arcseconds.
    pub ra_low: f32,

    /// The 50th percentile (median) of RA residuals in arcseconds.
    pub ra_med: f32,

    /// The 97.5th percentile of RA residuals in arcseconds.
    pub ra_high: f32,

    /// Standard deviation of RA residuals in arcseconds.
    pub ra_std: f32,

    /// The 2.5th percentile of Dec residuals in arcseconds.
    pub dec_low: f32,

    /// The 50th percentile (median) of Dec residuals in arcseconds.
    pub dec_med: f32,

    /// The 97.5th percentile of Dec residuals in arcseconds.
    pub dec_high: f32,

    /// Standard deviation of Dec residuals in arcseconds.
    pub dec_std: f32,
}

/// Load MPC Residual errors against all observations of the first 10k numbered
/// asteroids, as of 2025-04-20. This is used to estimate the expected residuals for a
/// given observatory code. These residuals are used to correct all other observations
/// from these observatories.
static OBS_ERRORS: std::sync::LazyLock<Box<[ObsResiduals]>> = std::sync::LazyLock::new(|| {
    let mut errors = Vec::new();
    let text = str::from_utf8(PRELOAD_OBS_ERRORS).unwrap().split('\n');
    for row in text.skip(1) {
        let cols: Vec<&str> = row.split(',').collect();
        if cols.len() == 11 {
            errors.push(ObsResiduals {
                code: cols[0].to_string(),
                n_obs: cols[1].parse().unwrap_or(0),
                std: cols[2].parse().unwrap_or(f32::NAN),
                ra_low: cols[3].parse().unwrap_or(f32::NAN),
                ra_med: cols[4].parse().unwrap_or(f32::NAN),
                ra_high: cols[5].parse().unwrap_or(f32::NAN),
                ra_std: cols[6].parse().unwrap_or(f32::NAN),
                dec_low: cols[7].parse().unwrap_or(f32::NAN),
                dec_med: cols[8].parse().unwrap_or(f32::NAN),
                dec_high: cols[9].parse().unwrap_or(f32::NAN),
                dec_std: cols[10].parse().unwrap_or(f32::NAN),
            });
        }
    }
    errors.into()
});

/// Get the residual statistics for a given observatory code, if available.
pub fn get_obs_residuals(code: &str) -> Option<&ObsResiduals> {
    OBS_ERRORS.iter().find(|err| err.code == code)
}
