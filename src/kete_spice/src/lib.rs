//! # `kete_spice`
//!
//! SPICE kernel I/O, SPK-dependent propagation, and SPICE-related extensions for kete.
//!
//! This crate provides:
//! - SPICE kernel reading (SPK, PCK, CK, SCLK)
//! - SPK-dependent N-body propagation via the [`propagation`] module
//! - FOV SPICE-dependent visibility checks via the [`fov_checks`] module
//! - CK-dependent frame rotation via the [`frame_ext`] module
//!
//! Dependency direction: `kete_spice -> kete_core` (one-way, no cycles).

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod ck;
pub mod daf;
pub mod fov_checks;
pub mod frame_ext;
pub mod pck;
pub mod propagation;
pub mod sclk;
pub mod spk;
pub mod state_transition;

mod interpolation;
mod jacobian;

/// Common useful imports.
pub mod prelude {

    pub use crate::ck::LOADED_CK;
    pub use crate::daf::DafFile;
    pub use crate::pck::LOADED_PCK;
    pub use crate::sclk::LOADED_SCLK;
    pub use crate::spk::LOADED_SPK;

    pub use crate::fov_checks::{check_n_body, check_spks, check_visible};
    pub use crate::frame_ext::rotations_to_equatorial_full;
    pub use crate::propagation::propagate_n_body_spk;

    pub use crate::state_transition::compute_state_transition;
}

use kete_core::time::{TDB, Time};

/// Convert seconds from J2000 into JD.
///
/// # Arguments
/// * `jds_sec` - The number of TDB seconds from J2000.
///
/// # Returns
/// The Julian Date (TDB).
#[inline(always)]
fn spice_jd_to_jd(jds_sec: f64) -> Time<TDB> {
    // 86400.0 = 60 * 60 * 24
    (jds_sec / 86400.0 + 2451545.0).into()
}

/// Convert TDB JD to seconds from J2000.
#[inline(always)]
fn jd_to_spice_jd(epoch: Time<TDB>) -> f64 {
    // 86400.0 = 60 * 60 * 24
    (epoch.jd - 2451545.0) * 86400.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spice_jd_to_jd() {
        {
            let jd_sec = 0.0;
            let jd = spice_jd_to_jd(jd_sec);
            assert_eq!(jd, 2451545.0.into());
        }
        {
            // 1 day in seconds
            let jd_sec = 86400.0;
            let jd = spice_jd_to_jd(jd_sec);
            assert_eq!(jd, 2451546.0.into());
        }
    }

    #[test]
    fn test_jd_to_spice_jd() {
        {
            let jd = 2451545.0.into();
            let jd_sec = jd_to_spice_jd(jd);
            assert_eq!(jd_sec, 0.0);
        }
        {
            // 1 day after J2000
            let jd = 2451546.0.into();
            let jd_sec = jd_to_spice_jd(jd);
            assert_eq!(jd_sec, 86400.0);
        }
    }

    #[test]
    fn test_spice_jd_to_jd_and_back() {
        let jd_sec = 1.0;
        let jd = spice_jd_to_jd(jd_sec);
        let jd_sec_back = jd_to_spice_jd(jd);
        assert!((jd_sec - jd_sec_back).abs() < 1e-5);
    }
}

/// Test-only helpers for loading SPK data from `docs/data/`.
#[cfg(any(test, feature = "test"))]
pub mod test_data {
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Ensure the test planetary kernel and a sample asteroid SPK are loaded
    /// into [`LOADED_SPK`](crate::spk::LOADED_SPK).
    ///
    /// On CI or fresh checkouts the user cache is empty, so `load_core()`
    /// yields an empty collection. This helper loads committed test files
    /// (`de440s_1990_2050.bsp` and `20000042.bsp`) so that all
    /// SPICE-dependent tests can run without an external cache.
    ///
    /// Guarded by `Once` so the files are loaded at most once per binary.
    ///
    /// # Panics
    /// Panics if the test SPK files cannot be loaded.
    pub fn ensure_test_spk() {
        INIT.call_once(|| {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .parent()
                .unwrap();
            let mut spk = crate::spk::LOADED_SPK.write().unwrap();
            for name in ["de440s_1990_2050.bsp", "20000042.bsp"] {
                let path = root.join("docs/data").join(name);
                if let Err(e) = spk.load_file(path.to_str().unwrap()) {
                    panic!("Failed to load test SPK {name}: {e}");
                }
            }
        });
    }
}
