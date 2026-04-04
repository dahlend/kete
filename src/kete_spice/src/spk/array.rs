use crate::daf::{DAFType, DafArray};
use crate::jd_to_spice_jd;
use kete_core::errors::Error;
use kete_core::time::{TDB, Time};

/// DAF Array of SPK data.
/// This is a wrapper around the [`DafArray`] which is specific to SPK data.
///
#[derive(Debug)]
pub struct SpkArray {
    /// The internal representation of the DAF array.
    pub daf: DafArray,

    /// JD Time in spice units of seconds from J2000.
    pub jds_start: f64,

    /// JD Time in spice units of seconds from J2000.
    pub jds_end: f64,

    /// The reference NAIF ID for the object in this Array.
    pub object_id: i32,

    /// The reference center NAIF ID for the central body in this Array.
    pub center_id: i32,

    /// The spice frame ID of the array.
    pub frame_id: i32,

    /// The spice segment type.
    pub segment_type: i32,
}

impl SpkArray {
    /// Is the specified JD within the range of this array.
    #[must_use]
    pub fn contains(&self, jd: Time<TDB>) -> bool {
        let jds = jd_to_spice_jd(jd);
        (jds >= self.jds_start) && (jds <= self.jds_end)
    }

    /// Construct a new SPK array from high-level parameters and data.
    ///
    /// `data` is the raw f64 array contents as defined by the SPK segment type
    /// specification (caller is responsible for formatting data correctly for the
    /// chosen segment type). The array start/end addresses are set to placeholder
    /// values (0) and filled in during file writing.
    #[must_use]
    pub fn new(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        segment_type: i32,
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
        data: Vec<f64>,
        name: String,
    ) -> Self {
        let jds_start = jd_to_spice_jd(jd_start);
        let jds_end = jd_to_spice_jd(jd_end);
        let summary_floats: Box<[f64]> = vec![jds_start, jds_end].into();
        let summary_ints: Box<[i32]> =
            vec![object_id, center_id, frame_id, segment_type, 0, 0].into();
        let daf = DafArray::new(
            summary_floats,
            summary_ints,
            data.into(),
            DAFType::Spk,
            name,
        );
        Self {
            daf,
            jds_start,
            jds_end,
            object_id,
            center_id,
            frame_id,
            segment_type,
        }
    }
}

impl TryFrom<DafArray> for SpkArray {
    type Error = Error;

    fn try_from(array: DafArray) -> Result<Self, Self::Error> {
        if array.daf_type != DAFType::Spk {
            Err(Error::IOError("DAF Array is not a SPK array.".into()))?;
        }

        if array.summary_floats.len() != 2 {
            Err(Error::IOError(
                "DAF Array is not a SPK array. Summary of array is incorrectly formatted, incorrect number of floats.".into(),
            ))?;
        }

        if array.summary_ints.len() != 6 {
            Err(Error::IOError("DAF Array is not a SPK array. Summary of array is incorrectly formatted, incorrect number of ints.".into()))?;
        }

        let jds_start = array.summary_floats[0];
        let jds_end = array.summary_floats[1];

        // The last two integers in the summary are the start and end of the array.
        let object_id = array.summary_ints[0];
        let center_id = array.summary_ints[1];
        let frame_id = array.summary_ints[2];
        let segment_type = array.summary_ints[3];

        Ok(Self {
            daf: array,
            jds_start,
            jds_end,
            object_id,
            center_id,
            frame_id,
            segment_type,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daf::{DafArray, DafFile};
    use crate::spk::segments::SpkSegment;
    use crate::spk::type2::SpkSegmentType2;
    use crate::spk::type9::SpkSegmentType9;
    use crate::spk::type13::SpkSegmentType13;

    /// Convert each
    /// [`DafArray`] through [`SpkArray`] -> [`SpkSegment`] -> [`DafArray`]
    /// exercising the full segment-type parsing and reconstruction.
    fn round_trip_through_segments(mut daf: DafFile) -> DafFile {
        let arrays: Vec<DafArray> = daf
            .arrays
            .drain(..)
            .map(|a| {
                let spk: SpkArray = a.try_into().unwrap();
                let seg: SpkSegment = spk.try_into().unwrap();
                seg.into()
            })
            .collect();
        daf.arrays = arrays;
        daf
    }

    #[test]
    fn spk_type9_round_trip() {
        use std::io::Cursor;

        let jd_start: Time<TDB> = 2451545.0.into(); // J2000
        let states: Vec<(Time<TDB>, [f64; 3], [f64; 3])> = (0..10)
            .map(|i| {
                let jd: Time<TDB> = (jd_start.jd + f64::from(i)).into();
                (jd, [1.0 + f64::from(i), 2.0, 3.0], [0.01, 0.02, 0.03])
            })
            .collect();

        let mut daf = DafFile::new_spk("test type 9", "round trip test");
        let spk_arr =
            SpkSegmentType9::new_array(-99999, 10, 1, &states, 3, "Test Type9 Segment").unwrap();
        daf.arrays.push(spk_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.arrays.len(), 1);

        let spk: SpkArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(spk.object_id, -99999);
        assert_eq!(spk.center_id, 10);
        assert_eq!(spk.frame_id, 1);
        assert_eq!(spk.segment_type, 9);
        assert_eq!(spk.jds_start, jd_to_spice_jd(jd_start));
        let jd_end: Time<TDB> = (jd_start.jd + 9.0).into();
        assert_eq!(spk.jds_end, jd_to_spice_jd(jd_end));
    }

    #[test]
    fn spk_type13_round_trip() {
        use std::io::Cursor;

        let jd_start: Time<TDB> = 2460000.0.into();
        let states: Vec<(Time<TDB>, [f64; 3], [f64; 3])> = (0..20)
            .map(|i| {
                let jd: Time<TDB> = (jd_start.jd + f64::from(i) * 0.5).into();
                (jd, [100.0 * f64::from(i), 200.0, 300.0], [1.0, 2.0, 3.0])
            })
            .collect();

        let mut daf = DafFile::new_spk("test type 13", "hermite test");
        let spk_arr =
            SpkSegmentType13::new_array(-88888, 0, 1, &states, 7, "Hermite Segment").unwrap();
        daf.arrays.push(spk_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.arrays.len(), 1);

        let spk: SpkArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(spk.object_id, -88888);
        assert_eq!(spk.segment_type, 13);
    }

    #[test]
    fn spk_type2_round_trip() {
        use std::io::Cursor;

        let polydg = 2;
        let ninrec = (polydg + 1) * 3; // 9
        let n = 3;
        let btime = 0.0;
        let intlen = 86400.0;
        let cdata: Vec<f64> = (0..ninrec * n).map(|i| i as f64 * 0.1).collect();
        let jd_start: Time<TDB> = 2451545.0.into();
        let jd_end: Time<TDB> = (2451545.0 + 3.0).into();

        let mut daf = DafFile::new_spk("test type 2", "chebyshev test");
        let spk_arr = SpkSegmentType2::new_array(
            -77777,
            0,
            1,
            &cdata,
            n,
            btime,
            intlen,
            polydg,
            jd_start,
            jd_end,
            "Cheby Segment",
        )
        .unwrap();
        daf.arrays.push(spk_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.arrays.len(), 1);

        let spk: SpkArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(spk.object_id, -77777);
        assert_eq!(spk.segment_type, 2);
    }

    #[test]
    fn spk_multiple_segments() {
        use std::io::Cursor;

        let jd_start: Time<TDB> = 2451545.0.into();
        let states9: Vec<(Time<TDB>, [f64; 3], [f64; 3])> = (0..5)
            .map(|i| {
                let jd: Time<TDB> = (jd_start.jd + f64::from(i)).into();
                (jd, [1.0; 3], [0.01; 3])
            })
            .collect();
        let states13: Vec<(Time<TDB>, [f64; 3], [f64; 3])> = (0..8)
            .map(|i| {
                let jd: Time<TDB> = (jd_start.jd + f64::from(i) * 2.0).into();
                (jd, [2.0; 3], [0.02; 3])
            })
            .collect();

        let mut daf = DafFile::new_spk("multi-seg", "two segments");
        daf.arrays.push(
            SpkSegmentType9::new_array(-11111, 10, 1, &states9, 3, "Seg A")
                .unwrap()
                .daf,
        );
        daf.arrays.push(
            SpkSegmentType13::new_array(-22222, 0, 1, &states13, 3, "Seg B")
                .unwrap()
                .daf,
        );

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.arrays.len(), 2);

        let ids: Vec<i32> = daf.arrays.iter().map(|a| a.summary_ints[0]).collect();
        assert_eq!(ids, vec![-11111, -22222]);
    }

    /// Read spherex.bsp (created by kete, not CSPICE), write it back, and
    /// verify the data round-trips correctly. Not byte-for-byte since the
    /// original was created by kete's old writer with different padding.
    #[test]
    fn round_trip_spherex() {
        use std::io::Cursor;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let original = std::fs::read(root.join("docs/data/spherex.bsp")).unwrap();
        let daf = DafFile::from_buffer(Cursor::new(&original)).unwrap();
        let daf = round_trip_through_segments(daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        // Re-read written output and compare structural data.
        let daf2 = DafFile::from_buffer(Cursor::new(&written)).unwrap();
        assert_eq!(daf.arrays.len(), daf2.arrays.len());
        assert_eq!(daf.comments, daf2.comments);
        assert_eq!(daf.internal_desc, daf2.internal_desc);
        for (a, b) in daf.arrays.iter().zip(daf2.arrays.iter()) {
            assert_eq!(a.summary_floats, b.summary_floats);
            assert_eq!(a.summary_ints, b.summary_ints);
            assert_eq!(a.data, b.data);
            assert_eq!(a.name, b.name);
        }
    }

    /// Same test for 20000042.bsp (7 Type-21 segments, 60 comment records).
    #[test]
    fn byte_for_byte_match_20000042() {
        use std::io::Cursor;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let original = std::fs::read(root.join("docs/data/20000042.bsp")).unwrap();
        let daf = DafFile::from_buffer(Cursor::new(&original)).unwrap();
        let daf = round_trip_through_segments(daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        assert_eq!(
            original.len(),
            written.len(),
            "File sizes differ: original={}, written={}",
            original.len(),
            written.len()
        );

        for (i, (a, b)) in original.iter().zip(written.iter()).enumerate() {
            assert_eq!(
                a,
                b,
                "Byte mismatch at offset {i} (0x{i:x}): original=0x{a:02x}, written=0x{b:02x} (record {}, offset-in-record {})",
                i / 1024 + 1,
                i % 1024,
            );
        }
    }

    /// Byte-for-byte test for wise.bsp (1296 Type-13 segments, 38 MB).
    #[test]
    fn byte_for_byte_match_wise() {
        use std::io::Cursor;

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let original = std::fs::read(root.join("docs/data/wise.bsp")).unwrap();
        let daf = DafFile::from_buffer(Cursor::new(&original)).unwrap();
        let daf = round_trip_through_segments(daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        assert_eq!(
            original.len(),
            written.len(),
            "File sizes differ: original={}, written={}",
            original.len(),
            written.len()
        );

        for (i, (a, b)) in original.iter().zip(written.iter()).enumerate() {
            assert_eq!(
                a,
                b,
                "Byte mismatch at offset {i} (0x{i:x}): original=0x{a:02x}, written=0x{b:02x} (record {}, offset-in-record {})",
                i / 1024 + 1,
                i % 1024,
            );
        }
    }

    /// Byte-for-byte test for a CSPICE-created binary PCK file.
    /// This test requires a PCK file to be present in the kete cache directory.
    #[test]
    fn byte_for_byte_match_pck() {
        use std::io::Cursor;

        let cache_dir = std::path::PathBuf::from(std::env::var("HOME").expect("HOME not set"))
            .join(".kete/kernels/core");
        // Find any .bpc file in the cache
        let pck_path = std::fs::read_dir(&cache_dir).ok().and_then(|entries| {
            entries.filter_map(Result::ok).map(|e| e.path()).find(|p| {
                p.extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("bpc"))
            })
        });
        let Some(pck_path) = pck_path else {
            eprintln!("Skipping PCK byte-for-byte test: no .bpc file in {cache_dir:?}");
            return;
        };

        let original = std::fs::read(&pck_path).unwrap();
        let daf = DafFile::from_buffer(Cursor::new(&original)).unwrap();
        assert_eq!(daf.daf_type, DAFType::Pck);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();
        let written = buf.into_inner();

        assert_eq!(
            original.len(),
            written.len(),
            "PCK file sizes differ: original={}, written={} ({:?})",
            original.len(),
            written.len(),
            pck_path,
        );

        for (i, (a, b)) in original.iter().zip(written.iter()).enumerate() {
            assert_eq!(
                a,
                b,
                "PCK byte mismatch at offset {i} (0x{i:x}): original=0x{a:02x}, written=0x{b:02x} (record {}, offset-in-record {}) in {:?}",
                i / 1024 + 1,
                i % 1024,
                pck_path,
            );
        }
    }
}
