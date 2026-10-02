//! Loading and reading of pointing from CK kernel files.
//!
//! CK files load into the singleton [`LOADED_CK`]. The singleton is wrapped in
//! a [`crossbeam::sync::ShardedLock`]. Acquire the lock before use. Most uses
//! only need the read lock.
//!
// BSD 3-Clause License
//
// Copyright (c) 2026, Dar Dahlen
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

mod array;
mod segments;
/// CK Type 2: Discrete pointing, constant rotation rate.
pub mod type2;
/// CK Type 3: Discrete pointing, linear interpolation.
pub mod type3;
pub mod type5;
pub mod type6;

pub use array::CkArray;
pub use type2::CkSegmentType2;
pub use type3::CkSegmentType3;
pub use type5::CkSegmentType5;
pub use type6::CkSegmentType6;

use kete_core::{
    errors::{Error, KeteResult},
    frames::{FrameId, NonInertialFrame},
    time::{TDB, Time},
};

use crate::daf::{DAFType, DafFile};
use crate::prepend_by_precedence;
use crate::text::sclk::Sclk;
use crossbeam::sync::ShardedLock;
use nalgebra::{Rotation3, Vector3};
use segments::CkSegment;
use std::collections::HashMap;

/// Largest difference, in days, between a requested time and the time of the CK
/// pointing used for it: about 1 ms.
pub(crate) const POINTING_TOLERANCE_DAYS: f64 = 1e-8;

/// A collection of segments.
#[derive(Debug, Default)]
pub struct CkCollection {
    /// The loaded segments, in precedence order: a lower index takes
    /// precedence.
    segments: Vec<CkSegment>,

    /// For each CK ID, the spans of its segments (see [`Span`]).
    index: HashMap<i32, Vec<Span>>,
}

impl CkCollection {
    /// Load all the segments of a CK file into this collection, ahead of those
    /// already loaded.
    ///
    /// # Errors
    /// [`Error::IOError`] if the file is not a CK formatted file.
    ///
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let file = DafFile::from_file(filename)?;
        if !matches!(file.daf_type, DAFType::Ck) {
            Err(Error::IOError(format!(
                "File {filename:?} is not a CK formatted file."
            )))?;
        }

        let mut segments = Vec::with_capacity(file.arrays.len());
        for array in file.arrays {
            let ck_array: CkArray = array.try_into()?;
            segments.push(CkSegment::try_from(ck_array)?);
        }
        // SPICE gives precedence to the file loaded last, and to the segment
        // stored last within a file.
        prepend_by_precedence(&mut self.segments, segments, |seg| {
            let arr: &CkArray = seg.into();
            arr.instrument_id
        });
        self.index = span_index(&self.segments);
        Ok(())
    }

    /// Clear all loaded CK kernels.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The pointing of the CK frame `ck_id` at `time`, on the spacecraft clock
    /// `sclk`.
    ///
    /// The function uses the first segment in precedence order that holds
    /// pointing at `time`. Thus inside a gap of a later segment, an earlier
    /// segment can supply the pointing. No segment extrapolates across a gap.
    ///
    /// The function returns the time of the pointing, and the frame relative
    /// to the reference frame that the segment stores.
    /// [`try_frame_at`](crate::frames::try_frame_at) resolves that reference.
    ///
    /// # Errors
    /// - [`Error::Bounds`] if no loaded segment holds pointing for `ck_id` at
    ///   `time`.
    /// - [`Error::IOError`] or [`Error::ValueError`] if the selected segment is
    ///   a type 6 segment with a malformed mini-segment.
    pub fn try_get_pointing(
        &self,
        time: Time<TDB>,
        ck_id: i32,
        sclk: &Sclk,
    ) -> KeteResult<(Time<TDB>, NonInertialFrame)> {
        let tick = sclk.time_to_tick(time)?;
        // Of the segments that span the tick, the one first in precedence order
        // that holds pointing there.
        let mut best: Option<usize> = None;
        if let Some(spans) = self.index.get(&ck_id) {
            for_each_spanning(spans, tick, &mut |idx| {
                if best.is_none_or(|b| idx < b) && self.segments[idx].has_data_at(tick) {
                    best = Some(idx);
                }
            });
        }
        let segment = best.map(|idx| &self.segments[idx]).ok_or_else(|| {
            Error::Bounds(format!(
                "CK frame {ck_id} has no pointing at JD {}.",
                time.jd()
            ))
        })?;
        segment.try_get_orientation(ck_id, time, tick, sclk)
    }

    /// Whether any loaded segment, at any time, holds pointing for `ck_id`.
    #[must_use]
    pub fn has_instrument(&self, ck_id: i32) -> bool {
        self.index.contains_key(&ck_id)
    }

    /// Return a list of all loaded instrument ids.
    #[must_use]
    pub fn loaded_instruments(&self) -> Vec<i32> {
        self.segments
            .iter()
            .map(|s| {
                let array: &CkArray = s.into();
                array.instrument_id
            })
            .collect::<Vec<i32>>()
            .into_iter()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect()
    }

    /// The loaded segments of an instrument, each as `(instrument id, reference
    /// frame id, segment type, start tick, end tick)`, with the ticks on the
    /// spacecraft clock.
    #[must_use]
    pub fn available_info(&self, instrument_id: i32) -> Vec<(i32, i32, i32, f64, f64)> {
        self.segments
            .iter()
            .filter_map(|s| {
                let array: &CkArray = s.into();
                if array.instrument_id == instrument_id {
                    Some((
                        array.instrument_id,
                        array.reference_frame_id,
                        array.segment_type,
                        array.tick_start,
                        array.tick_end,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

/// The time span of one segment, as a node of an implicit interval tree.
///
/// The spans of one CK ID are sorted by start tick. The node of a slice of them
/// is its middle span. Its children are the nodes of the slices below and above
/// it. `max_end` is the largest end tick in the slice of the node. Thus a search
/// skips every slice that ends before a tick.
#[derive(Debug)]
struct Span {
    start: f64,
    end: f64,
    max_end: f64,

    /// Index of the segment in [`CkCollection::segments`].
    segment: usize,
}

/// The spans of each CK ID in `segments`, sorted and with `max_end` set.
fn span_index(segments: &[CkSegment]) -> HashMap<i32, Vec<Span>> {
    let mut index: HashMap<i32, Vec<Span>> = HashMap::new();
    for (segment, seg) in segments.iter().enumerate() {
        let array: &CkArray = seg.into();
        index.entry(array.instrument_id).or_default().push(Span {
            start: array.tick_start,
            end: array.tick_end,
            max_end: array.tick_end,
            segment,
        });
    }
    for spans in index.values_mut() {
        spans.sort_by(|a, b| a.start.total_cmp(&b.start));
        let _ = set_max_end(spans);
    }
    index
}

/// Set `max_end` of the node of `spans` and its descendants.
///
/// The function returns the `max_end` of the node.
fn set_max_end(spans: &mut [Span]) -> f64 {
    let mid = spans.len() / 2;
    let (below, rest) = spans.split_at_mut(mid);
    let Some((node, above)) = rest.split_first_mut() else {
        return f64::NEG_INFINITY;
    };
    node.max_end = node.end.max(set_max_end(below)).max(set_max_end(above));
    node.max_end
}

/// Call `f` with the segment of each span of `spans` that holds `tick`.
///
/// Both ends of a span are inclusive.
fn for_each_spanning(spans: &[Span], tick: f64, f: &mut impl FnMut(usize)) {
    let mid = spans.len() / 2;
    let Some(node) = spans.get(mid) else {
        return;
    };
    if node.max_end < tick {
        return;
    }
    for_each_spanning(&spans[..mid], tick, f);
    // The spans above start at or after this one.
    if node.start <= tick {
        if tick <= node.end {
            f(node.segment);
        }
        for_each_spanning(&spans[mid + 1..], tick, f);
    }
}

/// CK singleton.
///
/// This is a lock protected [`CkCollection`]. Use `.try_read()` for read-only
/// access.
pub static LOADED_CK: std::sync::LazyLock<ShardedLock<CkCollection>> =
    std::sync::LazyLock::new(|| {
        let singleton = CkCollection::default();
        ShardedLock::new(singleton)
    });

/// Build an instrument frame from a CK C-matrix and angular velocity.
///
/// `time` is the time of the pointing. `c_matrix` rotates vectors from the
/// reference frame into the instrument frame, as CK stores it.
/// `angular_velocity` is the angular velocity of the instrument relative to the
/// reference frame, in radians per second. CK expresses it in the reference
/// frame. `reference_frame_id` is the ID of the reference frame.
///
/// The frame rotation is the inverse of the C-matrix. The rotation rate is the
/// time derivative of that rotation, `[w]x R`, in units of 1/day. The rotation
/// rate is `None` when `angular_velocity` is `None`.
fn instrument_frame(
    time: Time<TDB>,
    c_matrix: Rotation3<f64>,
    angular_velocity: Option<[f64; 3]>,
    reference_frame_id: i32,
) -> NonInertialFrame {
    let reference_frame_id = FrameId(reference_frame_id);
    let rotation = c_matrix.inverse();
    let rotation_rate =
        angular_velocity.map(|av| (Vector3::from(av) * 86400.0).cross_matrix() * rotation.matrix());
    NonInertialFrame::from_rotations(time, rotation, rotation_rate, reference_frame_id)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ck::type2::CkSegmentType2;
    use crate::daf::DafFile;
    use nalgebra::Matrix3;

    /// A constant-rate test clock with ID -999 on TDB, counting 65536 ticks per
    /// second from J2000.
    pub(crate) const TEST_CLOCK: &str = "KPL/SCLK\n\\begindata\n\
        SCLK_DATA_TYPE_999 = ( 1 )\n\
        SCLK01_TIME_SYSTEM_999 = ( 1 )\n\
        SCLK01_N_FIELDS_999 = ( 2 )\n\
        SCLK01_MODULI_999 = ( 4294967296 65536 )\n\
        SCLK01_OFFSETS_999 = ( 0 0 )\n\
        SCLK01_OUTPUT_DELIM_999 = ( 1 )\n\
        SCLK_PARTITION_START_999 = ( 0.0 )\n\
        SCLK_PARTITION_END_999 = ( 2.8147497671065E+14 )\n\
        SCLK01_COEFFICIENTS_999 = ( 0.0 0.0 1.0 )\n\
        \\begintext\n";

    /// The clock of [`TEST_CLOCK`].
    pub(crate) fn test_clock() -> &'static Sclk {
        static CLOCK: std::sync::OnceLock<crate::text::TextKernels> = std::sync::OnceLock::new();
        CLOCK
            .get_or_init(|| {
                let mut text = crate::text::TextKernels::default();
                text.load_text(TEST_CLOCK).unwrap();
                text
            })
            .clock(crate::text::sclk::ClockId(-999))
            .unwrap()
    }

    fn tick(jd: f64) -> f64 {
        test_clock().time_to_tick(Time::new(jd)).unwrap()
    }

    /// The index finds exactly the spans that hold a tick, as a scan of all of
    /// them does, for nested, overlapping, touching and disjoint spans.
    #[test]
    fn span_index_matches_a_scan() {
        // A fixed pseudo-random sequence keeps the test repeatable.
        let mut state: u64 = 3;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut spans: Vec<(f64, f64)> = (0..500)
            .map(|idx| {
                let start = (next() * 1000.0).floor();
                let length = if idx % 50 == 0 {
                    400.0
                } else {
                    (next() * 5.0).floor()
                };
                (start, start + length)
            })
            .collect();
        spans.push((10.0, 10.0));
        let mut index: Vec<Span> = spans
            .iter()
            .enumerate()
            .map(|(segment, &(start, end))| Span {
                start,
                end,
                max_end: end,
                segment,
            })
            .collect();
        index.sort_by(|a, b| a.start.total_cmp(&b.start));
        let _ = set_max_end(&mut index);
        for k in 0..2400 {
            let tick = f64::from(k) * 0.5 - 100.0;
            let mut found = Vec::new();
            for_each_spanning(&index, tick, &mut |idx| found.push(idx));
            found.sort_unstable();
            let expected: Vec<usize> = (0..spans.len())
                .filter(|&i| spans[i].0 <= tick && tick <= spans[i].1)
                .collect();
            assert_eq!(found, expected, "tick {tick}");
        }
        let mut found = Vec::new();
        for_each_spanning(&[], 1.0, &mut |idx| found.push(idx));
        assert!(found.is_empty());
    }

    /// Write a type 2 CK file for instrument -999000 and return its path.
    ///
    /// The file holds one fixed attitude over each of the given intervals.
    fn ck_file(name: &str, angle: f64, intervals: &[(f64, f64)]) -> String {
        let q = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle);
        let mut records = Vec::new();
        for _ in intervals {
            records.extend_from_slice(&[q.w, q.i, q.j, q.k, 0.0, 0.0, 0.0, 1.0]);
        }
        let starts: Vec<f64> = intervals.iter().map(|x| tick(x.0)).collect();
        let stops: Vec<f64> = intervals.iter().map(|x| tick(x.1)).collect();
        let array =
            CkSegmentType2::new_array(-999_000, 1, &records, &starts, &stops, name).unwrap();
        let mut daf = DafFile::new_ck("precedence test", "");
        daf.arrays.push(array.daf);
        let path = std::env::temp_dir().join(format!("kete_ck_precedence_{name}.bc"));
        let mut file = std::fs::File::create(&path).unwrap();
        daf.write_to(&mut file).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn angle_at(cks: &CkCollection, jd: f64) -> f64 {
        let (_, frame) = cks
            .try_get_pointing(Time::new(jd), -999_000, test_clock())
            .unwrap();
        let rot = Rotation3::from_matrix(frame.rotation.matrix());
        rot.angle()
    }

    /// Type 3 interpolates only between records of one interval.
    ///
    /// The interpolation has a constant rate for any signs of the stored
    /// quaternions. The segment holds no pointing in the gaps.
    #[test]
    fn type3_interpolates_within_each_interval() {
        use crate::ck::type3::CkSegmentType3;
        let jd0 = 2_457_100.0;
        let points = [(0.0, 0.0), (1.0, 0.2), (2.0, 0.4), (5.0, 1.0), (6.0, 2.2)];
        let mut records = Vec::new();
        for (idx, (_, angle)) in points.iter().enumerate() {
            let q = nalgebra::UnitQuaternion::from_axis_angle(&Vector3::z_axis(), *angle);
            // Flip the sign of alternate records. Both signs give the same
            // rotation.
            let sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
            records.extend_from_slice(&[sign * q.w, sign * q.i, sign * q.j, sign * q.k]);
        }
        let times: Vec<f64> = points.iter().map(|(dt, _)| tick(jd0 + dt)).collect();
        let starts = [times[0], times[3]];
        let array =
            CkSegmentType3::new_array(-999_000, 1, &records, &times, &starts, false, "type3")
                .unwrap();
        let mut daf = DafFile::new_ck("type 3 test", "");
        daf.arrays.push(array.daf);
        let path = std::env::temp_dir().join("kete_ck_type3_intervals.bc");
        daf.write_to(&mut std::fs::File::create(&path).unwrap())
            .unwrap();

        let mut cks = CkCollection::default();
        cks.load_file(path.to_str().unwrap()).unwrap();
        // Points away from the midpoints test the constant rate. A normalized
        // linear blend of the quaternions agrees with it only at the midpoint.
        for (dt, expected) in [
            (0.5, 0.1),
            (0.25, 0.05),
            (1.5, 0.3),
            (1.75, 0.35),
            (2.0, 0.4),
            (5.25, 1.3),
            (6.0, 2.2),
        ] {
            let angle = angle_at(&cks, jd0 + dt);
            assert!((angle - expected).abs() < 1e-9, "dt={dt}: {angle}");
        }
        assert!(
            cks.try_get_pointing(Time::new(jd0 + 3.0), -999_000, test_clock())
                .is_err()
        );
        assert!(
            cks.try_get_pointing(Time::new(jd0 + 6.5), -999_000, test_clock())
                .is_err()
        );
    }

    /// A later segment with gaps takes precedence inside its intervals.
    ///
    /// Inside its gaps, the earlier segment supplies the pointing, as in SPICE.
    #[test]
    fn a_gap_in_the_later_segment_falls_back_to_the_earlier() {
        let jd0 = 2_457_000.0;
        let older = ck_file("older", 0.1, &[(jd0, jd0 + 10.0)]);
        let newer = ck_file(
            "newer",
            0.7,
            &[(jd0 + 1.0, jd0 + 2.0), (jd0 + 6.0, jd0 + 7.0)],
        );

        let mut cks = CkCollection::default();
        cks.load_file(&older).unwrap();
        cks.load_file(&newer).unwrap();

        assert!((angle_at(&cks, jd0 + 1.5) - 0.7).abs() < 1e-12);
        assert!(
            (angle_at(&cks, jd0 + 4.0) - 0.1).abs() < 1e-12,
            "gap in newer"
        );
        assert!((angle_at(&cks, jd0 + 6.5) - 0.7).abs() < 1e-12);
        assert!(
            (angle_at(&cks, jd0 + 9.0) - 0.1).abs() < 1e-12,
            "after newer"
        );

        // With no earlier segment, a gap has no pointing. The code does not
        // extrapolate.
        let mut alone = CkCollection::default();
        alone.load_file(&newer).unwrap();
        assert!(
            alone
                .try_get_pointing(Time::new(jd0 + 4.0), -999_000, test_clock())
                .is_err()
        );
        assert!((angle_at(&alone, jd0 + 6.5) - 0.7).abs() < 1e-12);

        // With the load order reversed, the older file takes precedence at all
        // times.
        let mut cks = CkCollection::default();
        cks.load_file(&newer).unwrap();
        cks.load_file(&older).unwrap();
        assert!((angle_at(&cks, jd0 + 1.5) - 0.1).abs() < 1e-12);
    }

    pub(crate) const SPIN_JD0: f64 = 2_457_000.0;
    const SPIN_Q0: [f64; 4] = [0.980_066_577_841_241_6, 0.0, 0.198_669_330_795_061_2, 0.0];
    const SPIN_AV: [f64; 3] = [1e-5, -2e-5, 3e-5];

    /// Write a type 2 CK file with one day of constant spin.
    ///
    /// The function returns the path of the file.
    ///
    /// The spin rate is `SPIN_AV` and the start attitude is `SPIN_Q0`. The
    /// record uses the 65536 ticks per second of the test clock.
    pub(crate) fn spinning_ck_file(
        name: &str,
        instrument_id: i32,
        reference_frame_id: i32,
    ) -> String {
        let mut records = SPIN_Q0.to_vec();
        records.extend_from_slice(&SPIN_AV);
        records.push(1.0 / 65536.0);
        let starts = [tick(SPIN_JD0)];
        let stops = [tick(SPIN_JD0 + 1.0)];
        let array = CkSegmentType2::new_array(
            instrument_id,
            reference_frame_id,
            &records,
            &starts,
            &stops,
            name,
        )
        .unwrap();
        let mut daf = DafFile::new_ck("spin test", "");
        daf.arrays.push(array.daf);
        let path = std::env::temp_dir().join(format!("kete_ck_spin_{name}.bc"));
        let mut file = std::fs::File::create(&path).unwrap();
        daf.write_to(&mut file).unwrap();
        path.to_str().unwrap().to_string()
    }

    /// Return the central difference of a rotation over +/- `h` days.
    pub(crate) fn rate_by_difference(
        rotation_at: impl Fn(f64) -> Matrix3<f64>,
        jd: f64,
        h: f64,
    ) -> Matrix3<f64> {
        (rotation_at(jd + h) - rotation_at(jd - h)) / (2.0 * h)
    }

    /// Type 2 pointing matches the C-matrix from SPICE `ckgp`.
    ///
    /// The test uses the same segment in both readers. It also checks that the
    /// rotation rate is the derivative of the rotation.
    #[test]
    fn type2_spin_matches_spice() {
        let mut cks = CkCollection::default();
        cks.load_file(&spinning_ck_file("type2", -999_000, 1))
            .unwrap();

        // C-matrices from spiceypy ckgp for this segment and clock.
        let expected = [
            (
                600.0,
                Matrix3::new(
                    0.916_193_874_905_716,
                    0.014_166_179_273_209_28,
                    0.400_484_834_856_513,
                    -0.018_034_486_526_127_34,
                    0.999_820_007_559_872_9,
                    0.005_891_500_548_624_47,
                    -0.400_329_290_560_893_34,
                    -0.012_620_295_074_791_51,
                    0.916_284_444_703_771_6,
                ),
            ),
            (
                7200.0,
                Matrix3::new(
                    0.837_821_622_385_877_9,
                    0.158_085_536_416_831_18,
                    0.522_555_157_125_540_3,
                    -0.218_549_451_648_116_12,
                    0.974_236_385_407_248,
                    0.055_674_074_154_204_06,
                    -0.500_290_981_576_725_1,
                    -0.160_849_086_178_388_94,
                    0.850_785_816_306_650_7,
                ),
            ),
        ];
        for (seconds, c_matrix) in expected {
            let jd = SPIN_JD0 + seconds / 86400.0;
            let (_, frame) = cks
                .try_get_pointing(Time::new(jd), -999_000, test_clock())
                .unwrap();
            let err = (frame.rotation.inverse().matrix() - c_matrix).abs().max();
            assert!(err < 1e-8, "{seconds} s: C-matrix error {err:e}");

            let rotation_at = |jd| {
                *cks.try_get_pointing(Time::new(jd), -999_000, test_clock())
                    .unwrap()
                    .1
                    .rotation
                    .matrix()
            };
            let numeric = rate_by_difference(rotation_at, jd, 1e-3);
            let err = (frame.rotation_rate.unwrap() - numeric).abs().max();
            assert!(err < 1e-4, "{seconds} s: rotation rate error {err:e}");
        }
    }
}
