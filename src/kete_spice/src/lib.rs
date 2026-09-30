//! # `kete_spice`
//!
//! SPICE kernel I/O, SPK-dependent propagation, and SPICE-related extensions
//! for kete.
//!
//! This crate provides:
//! - SPICE kernel reading (SPK, PCK, CK, SCLK)
//! - SPICE kernel writing (SPK, PCK, CK)
//! - SPK-dependent N-body propagation in the [`propagation`] module
//! - SPICE-dependent FOV visibility checks in the [`fov_checks`] module
//! - CK-dependent frame rotation in the [`frame_ext`] module
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

mod interpolation;

/// Common useful imports.
pub mod prelude {

    pub use crate::ck::LOADED_CK;
    pub use crate::daf::DafFile;
    pub use crate::pck::LOADED_PCK;
    pub use crate::sclk::LOADED_SCLK;
    pub use crate::spk::LOADED_SPK;

    pub use crate::fov_checks::{check_n_body, check_spks, check_visible};
    pub use crate::frame_ext::rotations_to_equatorial_full;
    pub use crate::propagation::{SpkNBody, compute_state_transition};
}

use kete_core::time::{TDB, Time};

/// Add the segments of a newly loaded file to the front of a segment list.
///
/// `existing` is the list of loaded segments. `new` holds the segments of the
/// new file, in the order the file stores them. `key` gives the ID that
/// precedence is resolved by, such as the NAIF ID of the object.
///
/// SPICE resolves overlapping segments to the file loaded last. Within a file,
/// SPICE resolves them to the segment stored last. The readers scan forward and
/// return the first match. Those scans are on the hot path of propagation, so
/// this function sets precedence at load time instead of at query time.
///
/// The file goes in as one block ahead of the segments already loaded. Inside
/// the block, keys keep the order of their first appearance in the file. Only
/// the repeats of one key are reversed. Thus a file with one segment per key,
/// such as a planetary ephemeris, keeps its stored order.
pub(crate) fn prepend_by_precedence<S>(
    existing: &mut Vec<S>,
    new: Vec<S>,
    key: impl Fn(&S) -> i32,
) {
    if new.is_empty() {
        return;
    }
    let mut first_seen: Vec<i32> = Vec::new();
    let mut groups: std::collections::HashMap<i32, Vec<S>> = std::collections::HashMap::new();
    for seg in new.into_iter().rev() {
        let k = key(&seg);
        groups.entry(k).or_default().push(seg);
        first_seen.retain(|&x| x != k);
        first_seen.insert(0, k);
    }
    let block: Vec<S> = first_seen
        .into_iter()
        .flat_map(|k| groups.remove(&k).unwrap_or_default())
        .collect();
    let _ = existing.splice(0..0, block);
}

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

    /// Return the labels of (key, label) pairs. A label records the file and
    /// the position that a segment comes from.
    fn keys(v: &[(i32, &'static str)]) -> Vec<&'static str> {
        v.iter().map(|x| x.1).collect()
    }

    #[test]
    fn a_later_file_is_scanned_first() {
        let mut v = vec![(10, "a0"), (3, "a1")];
        prepend_by_precedence(&mut v, vec![(3, "b0"), (10, "b1")], |s| s.0);
        assert_eq!(keys(&v), ["b0", "b1", "a0", "a1"]);
    }

    /// Within one file, only the repeats of a key are reversed. Thus the scan
    /// finds the segment stored last for a key first.
    #[test]
    fn within_a_file_the_last_segment_of_a_key_comes_first() {
        let mut v = Vec::new();
        prepend_by_precedence(&mut v, vec![(7, "b0"), (5, "b1"), (7, "b2")], |s| s.0);
        assert_eq!(keys(&v), ["b2", "b0", "b1"]);
    }

    /// A file with one segment per key, such as de440s, keeps its stored
    /// order. The planet scan depends on this layout.
    #[test]
    fn one_segment_per_key_keeps_file_order() {
        let mut v: Vec<(i32, &'static str)> = Vec::new();
        let file = vec![(1, "1"), (2, "2"), (10, "10"), (399, "399"), (199, "199")];
        prepend_by_precedence(&mut v, file, |s| s.0);
        assert_eq!(keys(&v), ["1", "2", "10", "399", "199"]);
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
