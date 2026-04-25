const PRELOAD_OBS_ERRORS: &[u8] = include_bytes!("../../kete_core/data/obs_errs.csv");

/// MPC observation residual statistics, used to estimate expected residuals for a given
/// observatory code.
#[derive(Debug, Clone)]
pub struct ObservatoryStats {
    /// MPC observatory code.
    pub code: String,

    /// Number of observations used to compute these statistics.
    pub n_obs: u32,

    /// Standard deviation of RA residuals in arcseconds.
    pub ra_std: f32,

    /// Standard deviation of Dec residuals in arcseconds.
    pub dec_std: f32,
}

/// Load MPC Residual errors against all observations of the first 10k numbered
/// asteroids, as of 2025-04-20. This is used to estimate the expected residuals for a
/// given observatory code. These residuals are used to correct all other observations
/// from these observatories.
static OBS_ERRORS: std::sync::LazyLock<Box<[ObservatoryStats]>> = std::sync::LazyLock::new(|| {
    let mut errors = Vec::new();
    let text = str::from_utf8(PRELOAD_OBS_ERRORS).unwrap().split('\n');
    for row in text.skip(1) {
        let cols: Vec<&str> = row.split(',').collect();
        if cols.len() == 4 {
            errors.push(ObservatoryStats {
                code: cols[0].to_string(),
                n_obs: cols[1].parse().unwrap_or(0),
                ra_std: cols[2].parse().unwrap_or(f32::NAN),
                dec_std: cols[3].parse().unwrap_or(f32::NAN),
            });
        }
    }
    errors.into()
});

/// If available, return the RA and Dec standard deviations for a given MPC observatory.
pub fn get_observatory_stats(code: &str) -> Option<&ObservatoryStats> {
    OBS_ERRORS.iter().find(|err| err.code == code)
}
