//! # `kete_spice`
//!
//! SPICE kernel I/O and SPICE-related extensions for kete.
//!
//! The dependency is one way: `kete_spice` uses `kete_core`, and `kete_core`
//! does not use `kete_spice`.
//!
//! # Layout
//!
//! - [`daf`], [`spk`], [`pck`] and [`ck`]: the binary kernels. The SPK, PCK and
//!   CK modules read their kernels and write SPK, PCK and CK arrays.
//! - [`text`]: the text kernels (SCLK, frames, text PCK, instrument and
//!   meta-kernels), and what kete builds from them.
//! - [`frames`]: the frame model and the resolution of a frame to an inertial
//!   frame.
//! - [`instruments`]: the fields of view of instruments at a time.
//! - [`ephemeris`]: the loaded kernels as a
//!   [`kete_core::ephemeris::Ephemeris`], which the propagation and visibility
//!   code in `kete_core` takes.
//! - [`load_kernels`]: load files of any supported kernel type.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod ck;
pub mod daf;
pub mod ephemeris;
pub mod frames;
pub mod instruments;
pub mod pck;
pub mod spk;
pub mod text;

mod interpolation;

/// Common useful imports.
pub mod prelude {

    pub use crate::ck::LOADED_CK;
    pub use crate::daf::DafFile;
    pub use crate::pck::LOADED_PCK;
    pub use crate::spk::LOADED_SPK;
    pub use crate::text::LOADED_TEXT_KERNELS;

    pub use crate::ephemeris::SpiceEphemeris;
}

use kete_core::errors::{Error, KeteResult};

use std::io::Read;

/// Load kernel files of any supported type into their singletons.
///
/// The type of each file comes from its first bytes: binary SPK, PCK, and CK
/// files, and text SCLK, frames (FK), PCK, instrument (IK) and meta-kernels.
/// A text kernel that sets `KERNELS_TO_LOAD` is a meta-kernel (see
/// [`text::mk`]). The files it lists load right after it, in the order listed.
/// A listed file of an unsupported type, such as a leap seconds kernel or a
/// DSK, prints a message to stderr and is skipped.
///
/// All headers are read before any file loads, so a file in `filenames` of an
/// unsupported type loads nothing. A file of a supported type that fails to
/// load prints a message to stderr and is skipped, as the directory loaders
/// do.
///
/// All text kernels share one set of variables. Thus a frames kernel can define
/// a clock, and an SCLK kernel can hold frame variables.
///
/// # Errors
/// - [`Error::IOError`] if a file, or a file that a meta-kernel lists, cannot
///   be opened or read.
/// - [`Error::ValueError`] if a file in `filenames` does not have the header
///   of a supported kernel type, if a meta-kernel lists another meta-kernel,
///   or if its path symbols are malformed (see [`text::mk::kernels_to_load`]).
/// - [`Error::LockFailed`] if a singleton lock cannot be acquired.
pub fn load_kernels(filenames: &[String]) -> KeteResult<()> {
    let mut files = Vec::with_capacity(filenames.len());
    for filename in filenames {
        let kind = kernel_kind(filename)?.ok_or_else(|| {
            Error::ValueError(format!(
                "{filename} is not a supported kernel type. Supported types are binary \
                 SPK, PCK, and CK, and text SCLK, FK, PCK, IK, and meta-kernels."
            ))
        })?;
        files.push((filename.clone(), kind));
        let Some(listed) = meta_kernel_files(filename, kind)? else {
            continue;
        };
        for file in listed {
            let kind = kernel_kind(&file).map_err(|err| {
                let note = if std::path::Path::new(&file).is_relative() {
                    " (relative paths are from the working directory)"
                } else {
                    ""
                };
                add_context(err, &format!("Meta-kernel {filename}{note}"))
            })?;
            let Some(kind) = kind else {
                eprintln!(
                    "Meta-kernel {filename} lists {file}, which is not a supported kernel \
                     type. It is skipped."
                );
                continue;
            };
            if meta_kernel_files(&file, kind)?.is_some() {
                return Err(Error::ValueError(format!(
                    "Meta-kernel {filename} lists {file}, which is also a meta-kernel."
                )));
            }
            files.push((file, kind));
        }
    }
    for (filename, kind) in &files {
        let load = match kind {
            KernelKind::Spk => spk::LOADED_SPK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            KernelKind::Pck => pck::LOADED_PCK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            KernelKind::Ck => ck::LOADED_CK
                .write()
                .map_err(|_| Error::LockFailed)?
                .load_file(filename),
            KernelKind::Text => text::LOADED_TEXT_KERNELS
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

/// The kinds of kernel file [`load_kernels`] loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelKind {
    /// Binary SPK.
    Spk,
    /// Binary PCK.
    Pck,
    /// Binary CK.
    Ck,
    /// Text SCLK, FK, PCK, IK or meta-kernel.
    Text,
}

/// The kind of the kernel file `filename` from its header, or `None` if it is
/// not a supported kind.
///
/// # Errors
/// [`Error::IOError`] if the file cannot be opened or read.
fn kernel_kind(filename: &str) -> KeteResult<Option<KernelKind>> {
    // A file shorter than 8 bytes has a shorter header, which matches no kind.
    let mut header = Vec::with_capacity(8);
    let _ = std::fs::File::open(filename)
        .and_then(|f| f.take(8).read_to_end(&mut header))
        .map_err(|e| Error::IOError(format!("{filename}: {e}")))?;
    // Text kernel IDs other than "KPL/SCLK" are shorter than 8 bytes, and end
    // the line.
    let is_text = |id: &[u8]| {
        header.starts_with(id) && header.get(id.len()).is_none_or(u8::is_ascii_whitespace)
    };
    Ok(match header.as_slice() {
        b"DAF/SPK " => Some(KernelKind::Spk),
        b"DAF/PCK " => Some(KernelKind::Pck),
        b"DAF/CK  " => Some(KernelKind::Ck),
        _ if ["KPL/SCLK", "KPL/FK", "KPL/PCK", "KPL/IK", "KPL/MK"]
            .iter()
            .any(|id| is_text(id.as_bytes())) =>
        {
            Some(KernelKind::Text)
        }
        _ => None,
    })
}

/// The files the meta-kernel `filename` lists, or `None` if it is not a
/// meta-kernel.
///
/// A text kernel that cannot be read or does not parse is not a meta-kernel
/// here; it fails when it loads.
///
/// # Errors
/// [`Error::ValueError`] if its path symbols are malformed; see
/// [`text::mk::kernels_to_load`].
fn meta_kernel_files(filename: &str, kind: KernelKind) -> KeteResult<Option<Vec<String>>> {
    if kind != KernelKind::Text {
        return Ok(None);
    }
    let mut vars = text::TextKernelVars::default();
    if vars.load_file(filename).is_err() {
        return Ok(None);
    }
    text::mk::kernels_to_load(&vars)
        .map_err(|err| add_context(err, &format!("Meta-kernel {filename}")))
}

/// `err` with `prefix` before its message.
///
/// An error kind without a message is returned unchanged.
pub(crate) fn add_context(err: Error, prefix: &str) -> Error {
    // `Error` is non-exhaustive, so a match needs a wildcard arm. The wildcard
    // returns the kinds without a message unchanged.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "Error is non_exhaustive; kinds without a message pass through unchanged"
    )]
    match err {
        Error::Bounds(msg) => Error::Bounds(format!("{prefix}: {msg}")),
        Error::ValueError(msg) => Error::ValueError(format!("{prefix}: {msg}")),
        Error::IOError(msg) => Error::IOError(format!("{prefix}: {msg}")),
        other => other,
    }
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

    /// A frames kernel loads its frames, and the spacecraft clock it defines.
    #[test]
    fn load_kernels_loads_frames_kernels() {
        let path = std::env::temp_dir().join("kete_load_kernels_fk_test.tf");
        std::fs::write(
            &path,
            "KPL/FK\n\\begindata\nFRAME_LOAD_KERNELS_TEST = 1400701\n\
             SCLK_DATA_TYPE_1400702 = ( 1 )\nSCLK01_N_FIELDS_1400702 = ( 2 )\n\
             SCLK01_MODULI_1400702 = ( 4294967296 65536 )\n\
             SCLK01_OFFSETS_1400702 = ( 0 0 )\nSCLK01_OUTPUT_DELIM_1400702 = ( 1 )\n\
             SCLK_PARTITION_START_1400702 = ( 0.0 )\n\
             SCLK_PARTITION_END_1400702 = ( 2.8147497671065E+14 )\n\
             SCLK01_COEFFICIENTS_1400702 = ( 0.0 0.0 1.0 )\n",
        )
        .unwrap();
        load_kernels(&[path.to_str().unwrap().to_string()]).unwrap();
        let text = text::LOADED_TEXT_KERNELS.read().unwrap();
        assert_eq!(text.frame_id("load_kernels_test").unwrap().0, 1_400_701);
        let tick = text
            .clock(text::sclk::ClockId(-1_400_702))
            .unwrap()
            .time_to_tick(kete_core::time::Time::new(2_451_545.0))
            .unwrap();
        assert_eq!(tick, 0.0);
    }

    /// An unsupported or short header is an error, and a missing file names
    /// itself.
    #[test]
    fn load_kernels_rejects_unsupported_files() {
        let path = std::env::temp_dir().join("kete_load_kernels_test.tf");
        std::fs::write(&path, "KPL/LSK\n").unwrap();
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

    /// A meta-kernel loads its own variables except the meta-kernel ones, then
    /// the files it lists. A listed leap seconds kernel is skipped. A listed
    /// meta-kernel and a missing listed file are errors.
    #[test]
    fn load_kernels_meta_kernel() {
        let dir = std::env::temp_dir().join("kete_load_kernels_mk");
        std::fs::create_dir_all(&dir).unwrap();
        let path = |name: &str| dir.join(name).to_str().unwrap().to_string();
        std::fs::write(path("a.tf"), "KPL/FK\n\\begindata\nKETE_MK_TEST_A = 1\n").unwrap();
        std::fs::write(path("b.tls"), "KPL/LSK\n").unwrap();
        let meta = |name: &str, files: &str| {
            std::fs::write(
                path(name),
                format!(
                    "KPL/MK\n\\begindata\nPATH_VALUES = ( '{}' )\n\
                     PATH_SYMBOLS = ( 'D' )\nKERNELS_TO_LOAD = ( {files} )\n\
                     KETE_MK_TEST_{} = 2\n",
                    dir.to_str().unwrap(),
                    name.trim_end_matches(".tm").to_uppercase()
                ),
            )
            .unwrap();
            path(name)
        };
        let good = meta("good.tm", "'$D/a+' '.tf' '$D/b.tls'");
        load_kernels(std::slice::from_ref(&good)).unwrap();
        let text = text::LOADED_TEXT_KERNELS.read().unwrap();
        assert_eq!(text.vars().integer("KETE_MK_TEST_A").unwrap(), Some(1));
        assert_eq!(text.vars().integer("KETE_MK_TEST_GOOD").unwrap(), Some(2));
        assert!(text.vars().get("PATH_SYMBOLS").is_none());
        drop(text);

        let nested = meta("nested.tm", "'$D/good.tm'");
        assert!(matches!(
            load_kernels(std::slice::from_ref(&nested)),
            Err(Error::ValueError(msg)) if msg.contains("also a meta-kernel")
        ));
        let missing = meta("missing.tm", "'$D/a.tf' '$D/none.bsp'");
        assert!(matches!(
            load_kernels(std::slice::from_ref(&missing)),
            Err(Error::IOError(msg)) if msg.contains("missing.tm") && msg.contains("none.bsp")
        ));
        let text = text::LOADED_TEXT_KERNELS.read().unwrap();
        assert!(text.vars().get("KETE_MK_TEST_NESTED").is_none());
        assert!(text.vars().get("KETE_MK_TEST_MISSING").is_none());
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

    /// A file with one segment per key, such as de440s, keeps its stored order.
    /// The planet scan depends on this layout.
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
    /// (`de440s_1990_2050.bsp` and `20000042.bsp`) so that all SPICE-dependent
    /// tests can run without an external cache.
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
