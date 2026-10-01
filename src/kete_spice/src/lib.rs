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

use kete_core::errors::{Error, KeteResult};

use std::io::Read;

/// Load kernel files of any supported type into their singletons.
///
/// The type of each file comes from its first 8 bytes: binary SPK, PCK, and CK
/// files, and text SCLK files. All headers are read before any file loads, so a
/// file of an unsupported type loads nothing. A file of a supported type that
/// fails to load prints a message to stderr and is skipped, as the directory
/// loaders do.
///
/// # Errors
/// - [`Error::IOError`] if a file cannot be opened or read.
/// - [`Error::ValueError`] if a file does not have the header of a supported
///   kernel type.
/// - [`Error::LockFailed`] if a singleton lock cannot be acquired.
pub fn load_kernels(filenames: &[String]) -> KeteResult<()> {
    let mut kinds = Vec::with_capacity(filenames.len());
    for filename in filenames {
        // A file shorter than 8 bytes has a shorter header, which matches no type.
        let mut header = Vec::with_capacity(8);
        let _ = std::fs::File::open(filename)
            .and_then(|f| f.take(8).read_to_end(&mut header))
            .map_err(|e| Error::IOError(format!("{filename}: {e}")))?;
        match header.as_slice() {
            b"DAF/SPK " | b"DAF/PCK " | b"DAF/CK  " | b"KPL/SCLK" => kinds.push(header),
            _ => {
                return Err(Error::ValueError(format!(
                    "{filename} has header {:?}, which is not a supported kernel type. \
                     Supported types are binary SPK, PCK, and CK, and text SCLK.",
                    String::from_utf8_lossy(&header)
                )));
            }
        }
    }
    for (filename, kind) in filenames.iter().zip(kinds) {
        let load = match kind.as_slice() {
            b"DAF/SPK " => spk::LOADED_SPK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            b"DAF/PCK " => pck::LOADED_PCK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            b"DAF/CK  " => ck::LOADED_CK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            _ => sclk::LOADED_SCLK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
        };
        if let Err(err) = load {
            eprintln!("{filename} failed to load. {err}");
        }
    }
    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Return the labels of (key, label) pairs. A label records the file and
    /// the position that a segment comes from.
    fn keys(v: &[(i32, &'static str)]) -> Vec<&'static str> {
        v.iter().map(|x| x.1).collect()
    }

    /// An unsupported or short header is an error, and a missing file names itself.
    #[test]
    fn load_kernels_rejects_unsupported_files() {
        let path = std::env::temp_dir().join("kete_load_kernels_test.tf");
        std::fs::write(&path, "KPL/FK\n").unwrap();
        let path = path.to_str().unwrap().to_string();
        assert!(matches!(
            load_kernels(std::slice::from_ref(&path)),
            Err(Error::ValueError(_))
        ));
        let missing = format!("{path}.missing");
        assert!(matches!(
            load_kernels(std::slice::from_ref(&missing)),
            Err(Error::IOError(msg)) if msg.contains(&missing)
        ));
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
