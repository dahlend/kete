//! Spice replacement methods
//! Primarily includes the ability to read the contents of SPK files.
mod daf;
mod interpolation;
mod naif_ids;
mod obs_codes;
mod pck;
mod pck_segments;
#[allow(unused)]
mod sclk;
mod spk;
mod spk_segments;

pub use daf::*;
pub use naif_ids::try_name_from_id;
pub use obs_codes::OBS_CODES;
pub use pck::*;
pub use spk::*;

/// Convert seconds from J2000 into JD.
///
/// # Arguments
/// * `jds_sec` - The number of seconds from J2000.
///
/// # Returns
/// The Julian Date (TDB).
fn spice_jd_to_jd(jds_sec: f64) -> f64 {
    // 86400.0 = 60 * 60 * 24
    jds_sec / 86400.0 + 2451545.0
}

/// Convert JD to seconds from J2000.
fn jd_to_spice_jd(jd: f64) -> f64 {
    // 86400.0 = 60 * 60 * 24
    (jd - 2451545.0) * 86400.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spice_jd_to_jd() {
        let jd_sec = 0.0;
        let jd = spice_jd_to_jd(jd_sec);
        assert_eq!(jd, 2451545.0);

        let jd_sec = 86400.0; // 1 day in seconds
        let jd = spice_jd_to_jd(jd_sec);
        assert_eq!(jd, 2451546.0);
    }

    #[test]
    fn test_jd_to_spice_jd() {
        let jd = 2451545.0;
        let jd_sec = jd_to_spice_jd(jd);
        assert_eq!(jd_sec, 0.0);

        let jd = 2451546.0; // 1 day after J2000
        let jd_sec = jd_to_spice_jd(jd);
        assert_eq!(jd_sec, 86400.0);
    }

    #[test]
    fn test_spice_jd_to_jd_and_back() {
        let jd_sec = 1.0;
        let jd = spice_jd_to_jd(jd_sec);
        let jd_sec_back = jd_to_spice_jd(jd);
        assert!((jd_sec - jd_sec_back).abs() < 1e-5);
    }
}
