// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

use super::PckArray;
use crate::interpolation::{ChebyshevLayout, chebyshev_evaluate_both};
use crate::spk::type2::build_type2_data;
use kete_core::errors::Error;
use kete_core::frames::{FrameId, NonInertialFrame};
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Chebyshev polynomials (Euler angles only)
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html#Binary%20PCK%20Kernel>
///
#[derive(Debug)]
pub struct PckSegmentType2 {
    pub(in crate::pck) array: PckArray,
    layout: ChebyshevLayout,
}

impl PckSegmentType2 {
    fn get_record(&self, idx: usize) -> &[f64] {
        let record_len = self.layout.record_len;
        // SAFETY: `ChebyshevLayout::from_array` checked that the array holds
        // `n_records` records of `record_len` values. The only caller takes
        // `idx` from `record_index`, which returns at most `n_records - 1`.
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(idx * record_len..(idx + 1) * record_len)
        }
    }

    /// Return the stored orientation, along with the rate of change of the orientation.
    pub(in crate::pck) fn try_get_orientation(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<NonInertialFrame> {
        // Records in the segment contain information about the central position of the
        // north pole, as well as the position of the prime meridian. These values for
        // type 2 segments are stored as chebyshev polynomials of the first kind, in
        // essentially the exact same format as the Type 2 SPK segments.
        // Records for this type are structured as so:
        // - time at midpoint of record.
        // - (length of time record is valid for) / 2.0
        // - N Chebyshev polynomial coefficients for ra
        // - N Chebyshev polynomial coefficients for dec
        // - N Chebyshev polynomial coefficients for w
        //
        // Rate of change for each of these values can be calculated by using the
        // derivative of chebyshev of the first kind, which is done below.
        let record = self.get_record(self.layout.record_index(time.j2000_seconds()));
        let t_mid = record[0];
        let t_step = record[1];
        let t = time.j2000_seconds_minus(t_mid) / t_step;

        let n_coef = self.layout.n_coef;
        let ra_coef = &record[2..(n_coef + 2)];
        let dec_coef = &record[(n_coef + 2)..(2 * n_coef + 2)];
        let w_coef = &record[(2 * n_coef + 2)..(3 * n_coef + 2)];

        let ([ra, dec, w], [ra_der, dec_der, w_der]) =
            chebyshev_evaluate_both(t, ra_coef, dec_coef, w_coef)?;

        // rem_euclid is equivalent to the modulo operator, so this maps w to [0, 2pi]
        let w = w.rem_euclid(std::f64::consts::TAU);

        let frame = NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            time,
            [ra, dec, w],
            [
                // convert to radians per day
                ra_der / t_step * 86400.0,
                dec_der / t_step * 86400.0,
                w_der / t_step * 86400.0,
            ],
            FrameId(self.array.reference_frame_id),
        );

        Ok(frame)
    }

    /// Create a Type 2 (Chebyshev Euler angles, fixed intervals) PCK array.
    ///
    /// `frame_id` is the body-fixed frame ID, such as 3000 for Earth.
    /// `reference_frame_id` is the inertial reference frame ID, such as 17 for
    /// Ecliptic. `cdata` holds the flat Chebyshev coefficients,
    /// `3 * (polydg + 1)` values per record, in the order
    /// `[RA_0..RA_d, DEC_0..DEC_d, W_0..W_d]`. `n_records` is the number of
    /// records. `btime` is the start of the first interval, in TDB seconds from
    /// J2000. `intlen` is the length of each interval, in seconds. `polydg` is
    /// the polynomial degree. `jds_start` and `jds_end` are the segment start
    /// and end, in TDB seconds from J2000. `segment_name` is the name stored in
    /// the DAF name record.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `polydg` is greater than 27.
    /// - `intlen` is zero or negative.
    /// - The length of `cdata` is not `3 * (polydg + 1) * n_records`.
    pub fn new_array(
        frame_id: i32,
        reference_frame_id: i32,
        cdata: &[f64],
        n_records: usize,
        btime: f64,
        intlen: f64,
        polydg: usize,
        jds_start: f64,
        jds_end: f64,
        segment_name: &str,
    ) -> KeteResult<PckArray> {
        let data = build_type2_data(cdata, n_records, btime, intlen, polydg)?;
        Ok(PckArray::new(
            frame_id,
            reference_frame_id,
            2,
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }
}

impl TryFrom<PckArray> for PckSegmentType2 {
    type Error = Error;

    fn try_from(array: PckArray) -> Result<Self, Self::Error> {
        let layout = ChebyshevLayout::from_array(&array.daf, 3)?;
        Ok(Self { array, layout })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daf::DafFile;

    #[test]
    fn pck_type2_round_trip() {
        use std::io::Cursor;

        let polydg = 2;
        let ninrec = (polydg + 1) * 3; // 9
        let n = 3;
        let btime = 0.0;
        let intlen = 86400.0;
        let cdata: Vec<f64> = (0..ninrec * n).map(|i| i as f64 * 0.01).collect();
        let jds_start = 0.0;
        let jds_end = 3.0 * 86400.0;

        let mut daf = DafFile::new_pck("test pck", "pck round trip test");
        let pck_arr = PckSegmentType2::new_array(
            3000,
            17,
            &cdata,
            n,
            btime,
            intlen,
            polydg,
            jds_start,
            jds_end,
            "Earth orientation",
        )
        .unwrap();
        daf.arrays.push(pck_arr.daf);

        let mut buf = Cursor::new(Vec::new());
        daf.write_to(&mut buf).unwrap();

        let bytes = buf.into_inner();
        let daf = DafFile::from_buffer(Cursor::new(&bytes)).unwrap();
        assert_eq!(daf.daf_type, crate::daf::DAFType::Pck);
        assert_eq!(daf.arrays.len(), 1);
        assert_eq!(daf.n_doubles, 2);
        assert_eq!(daf.n_ints, 5);

        let pck: PckArray = daf.arrays.into_iter().next().unwrap().try_into().unwrap();
        assert_eq!(pck.frame_id, 3000);

        let seg = PckSegmentType2::try_from(pck).unwrap();
        assert!(
            seg.try_get_orientation(Time::from_j2000_seconds(1.5 * intlen))
                .is_ok()
        );
    }

    /// Check that the final instant of the segment uses the last record.
    ///
    /// Each record holds constant angles. The angles of the third record are
    /// 0.2, 0.25, and 0.3 rad.
    #[test]
    fn pck_type2_final_instant_uses_the_last_record() {
        let polydg = 1;
        let n = 3;
        let intlen = 86400.0;
        let mut cdata = Vec::new();
        for k in 0..n {
            let k = f64::from(k);
            for angle in [0.1 * k, 0.05 + 0.1 * k, 0.1 + 0.1 * k] {
                cdata.extend_from_slice(&[angle, 0.0]);
            }
        }
        let array = PckSegmentType2::new_array(
            3000,
            17,
            &cdata,
            3,
            0.0,
            intlen,
            polydg,
            0.0,
            3.0 * intlen,
            "constant records",
        )
        .unwrap();
        let seg = PckSegmentType2::try_from(array).unwrap();
        let frame = seg
            .try_get_orientation(Time::from_j2000_seconds(3.0 * intlen))
            .unwrap();
        let expected = NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(
            0.0,
            [0.2, 0.25, 0.3],
            [0.0; 3],
            FrameId::ECLIPJ2000,
        );
        assert!((frame.rotation.matrix() - expected.rotation.matrix()).norm() < 1e-14);
    }

    /// A segment relative to J2000 (1) evaluates like one relative to ECLIPJ2000 (17);
    /// the stored reference frame decides the rotation to equatorial.
    #[test]
    fn pck_type2_reference_frame_is_kept() {
        use crate::pck::segments::PckSegment;
        use kete_core::frames::{Ecliptic, InertialFrame};

        let angles = [0.2, 0.25, 0.3];
        let cdata: Vec<f64> = angles.iter().flat_map(|&a| [a, 0.0]).collect();
        let time = Time::from_j2000_seconds(0.5 * 86400.0);
        let frame_with_reference = |reference_frame_id| {
            let array = PckSegmentType2::new_array(
                3000,
                reference_frame_id,
                &cdata,
                1,
                0.0,
                86400.0,
                1,
                0.0,
                86400.0,
                "constant record",
            )
            .unwrap();
            PckSegment::try_from(array)
                .unwrap()
                .try_get_orientation(3000, time)
                .unwrap()
        };
        let stored =
            NonInertialFrame::from_euler::<'Z', 'X', 'Z'>(0.0, angles, [0.0; 3], FrameId::J2000);

        let j2000 = frame_with_reference(1);
        assert_eq!(j2000.reference_frame_id, FrameId::J2000);
        let (rot, _) = j2000.rotations_to_equatorial().unwrap();
        assert!((rot.matrix() - stored.rotation.matrix()).norm() < 1e-14);

        let ecliptic = frame_with_reference(17);
        assert_eq!(ecliptic.reference_frame_id, FrameId::ECLIPJ2000);
        let (rot, _) = ecliptic.rotations_to_equatorial().unwrap();
        let expected = Ecliptic::rotation_to_equatorial() * stored.rotation;
        assert!((rot.matrix() - expected.matrix()).norm() < 1e-14);
    }
}
