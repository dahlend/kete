//! Most users should interface with `pck.rs`, not this module.
//!
//! PCK Files are collections of `Segments`, which are ranges of times where the state
//! of an object is recorded. These segments are typically made up of many individual
//! `Records`, with an associated maximum and minimum time where they are valid for.
//!
//! There are unique structs for each possible segment type, not all are currently
//! supported. Each segment type must implement the PCKSegment trait, which allows for
//! the loading and querying of states contained within.
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html>
//!
//! There is a lot of repetition in this file, as many of the segment types have very
//! similar internal structures.
//!
use super::daf::DafArray;
use super::interpolation::*;
use super::{jd_to_spice_jd, spice_jds_to_jd};
use crate::errors::Error;
use crate::frames::Frame;
use crate::prelude::KeteResult;
use std::fmt::Debug;

#[derive(Debug)]
pub enum PckSegmentType {
    Type2(PckSegmentType2),
}

impl PckSegmentType {
    /// Load PCK Segment data from an array.
    pub fn from_array(segment_type: i32, array: DafArray) -> KeteResult<Self> {
        match segment_type {
            2 => Ok(PckSegmentType::Type2(array.into())),
            v => Err(Error::IOError(format!(
                "SPK Segment type {:?} not supported.",
                v
            )))?,
        }
    }
}

impl From<PckSegmentType> for DafArray {
    fn from(value: PckSegmentType) -> Self {
        match value {
            PckSegmentType::Type2(seg) => seg.array,
        }
    }
}

#[derive(Debug)]
pub struct PckSegment {
    /// The reference center NAIF ID for the central body in this Segment.
    pub center_id: i32,

    /// [`Frame`] of reference for this Segment.
    pub ref_frame: Frame,

    /// Start time of the segment.
    pub jd_start: f64,

    /// End time of the segment.
    pub jd_end: f64,

    /// Internal data representation.
    segment: PckSegmentType,
}

impl TryFrom<DafArray> for PckSegment {
    type Error = Error;

    fn try_from(array: DafArray) -> KeteResult<PckSegment> {
        let summary_floats = &array.summary_floats;
        let summary_ints = &array.summary_ints;
        let jd_start = spice_jds_to_jd(summary_floats[0]);
        let jd_end = spice_jds_to_jd(summary_floats[1]);

        let center_id = summary_ints[0];
        let frame_id = summary_ints[1];
        let segment_type = summary_ints[2];

        let ref_frame = match frame_id {
            1 => Frame::Equatorial, // J2000
            17 => Frame::Ecliptic,  // ECLIPJ2000
            _ => Frame::Unknown(frame_id),
        };

        let segment = PckSegmentType::from_array(segment_type, array)?;

        Ok(PckSegment {
            jd_start,
            jd_end,
            center_id,
            ref_frame,
            segment,
        })
    }
}

impl PckSegment {
    pub fn contains(&self, jd: f64) -> bool {
        (jd >= self.jd_start) && (jd <= self.jd_end)
    }

    /// Return the [`Frame`] at the specified JD. If the requested time is not within
    /// the available range, this will fail.
    pub fn try_get_orientation(&self, jd: f64) -> KeteResult<Frame> {
        if jd < self.jd_start || jd > self.jd_end {
            Err(Error::DAFLimits(
                "JD is not present in this record.".to_string(),
            ))?;
        }
        if self.ref_frame != Frame::Ecliptic {
            Err(Error::ValueError(
                "Non ecltiptic frames are not supported for PCK queries.".into(),
            ))?;
        }

        match &self.segment {
            PckSegmentType::Type2(v) => v.try_get_orientation(self, jd),
        }
    }
}

impl From<PckSegment> for DafArray {
    fn from(value: PckSegment) -> Self {
        value.segment.into()
    }
}

/// Chebyshev polynomials (Euler angles only)
///
/// https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html#Type%201:%20Modified%20Difference%20Arrays
///
#[derive(Debug)]
pub struct PckSegmentType2 {
    array: DafArray,
    jd_step: f64,
    n_coef: usize,
    record_len: usize,
}

impl PckSegmentType2 {
    fn get_record(&self, idx: usize) -> &[f64] {
        unsafe {
            self.array
                .data
                .get_unchecked(idx * self.record_len..(idx + 1) * self.record_len)
        }
    }

    /// Return the stored orientation, along with the rate of change of the orientation.
    fn try_get_orientation(&self, segment: &PckSegment, jd: f64) -> KeteResult<Frame> {
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
        let jd = jd_to_spice_jd(jd);
        let jd_start = jd_to_spice_jd(segment.jd_start);
        let record_index = ((jd - jd_start) / self.jd_step).floor() as usize;
        let record = self.get_record(record_index);
        let t_mid = record[0];
        let t_step = record[1];
        let t = (jd - t_mid) / t_step;

        let ra_coef = &record[2..(self.n_coef + 2)];
        let dec_coef = &record[(self.n_coef + 2)..(2 * self.n_coef + 2)];
        let w_coef = &record[(2 * self.n_coef + 2)..(3 * self.n_coef + 2)];

        let ([ra, dec, w], [ra_der, dec_der, w_der]) =
            chebyshev_evaluate_both(t, ra_coef, dec_coef, w_coef)?;

        // rem_euclid is equivalent to the modulo operator, so this maps w to [0, 2pi]
        let w = w.rem_euclid(std::f64::consts::TAU);

        Ok(Frame::EclipticNonInertial(
            segment.center_id,
            [
                ra,
                dec,
                w,
                ra_der / t_step * 86400.0,
                dec_der / t_step * 86400.0,
                w_der / t_step * 86400.0,
            ],
        ))
    }
}

impl From<DafArray> for PckSegmentType2 {
    fn from(array: DafArray) -> Self {
        let n_records = array[array.len() - 1] as usize;
        let record_len = array[array.len() - 2] as usize;
        let jd_step = array[array.len() - 3];

        let n_coef = (record_len - 2) / 3;

        if n_records * record_len + 4 != array.len() {
            panic!("PCK File not formatted correctly.")
        }

        PckSegmentType2 {
            array,
            jd_step,
            n_coef,
            record_len,
        }
    }
}
