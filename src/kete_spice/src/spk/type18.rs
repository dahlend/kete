//! SPK Segment Type 18 - ESOC/DDID Hermite/Lagrange Interpolation.
//!
//! Subtype 0: Hermite interpolation with 12-value records (pos, dpos, vel, dvel).
//! Subtype 1: Lagrange interpolation with 6-value records (pos, vel).
//!
//! <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%2018:%20ESOC/DDID%20Hermite/Lagrange%20Interpolation>

use super::SpkArray;
use crate::interpolation::{hermite_interpolation, lagrange_interpolation};
use kete_core::constants::AU_KM;
use kete_core::errors::Error;
use kete_core::prelude::KeteResult;
use kete_core::time::{TDB, Time};

/// Type 18 Record
///
/// This is actually 2 types in 1, under the stated goal of reducing the number
/// of unique SPICE kernel types.
///
/// Subtype 0 is a Hermite Interpolation of both position and velocity, a record
/// contains 12 numbers, 3 position, 3 derivative of position, 3 velocity, and 3
/// derivative of velocity. Note that it explicitly allows that the 3 velocity
/// values do not have to match the derivative of the position vectors.
/// Subtype 1 is a Lagrange Interpolation, a record of which contains 6 values,
/// 3 position, and 3 velocity.
///
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/FORTRAN/req/spk.html#Type%2018:%20ESOC/DDID%20Hermite/Lagrange%20Interpolation>
#[derive(Debug)]
pub struct SpkSegmentType18 {
    pub(crate) array: SpkArray,
    subtype: usize,
    window_size: usize,
    n_records: usize,
    record_size: usize,
}

impl SpkSegmentType18 {
    /// Create a Type 18 (ESOC/DDID Hermite or Lagrange interpolation) SPK array.
    ///
    /// # Arguments
    /// * `object_id`     - NAIF ID of the body.
    /// * `center_id`     - NAIF ID of the center body.
    /// * `frame_id`      - NAIF frame ID.
    /// * `records`       - Flat state data: `n * record_size` values (12 for subtype 0, 6 for subtype 1).
    /// * `epochs`        - n epoch values (SPICE seconds from J2000), strictly increasing.
    /// * `subtype`       - 0 for Hermite, 1 for Lagrange.
    /// * `window_size`   - Interpolation window size, must be <= n.
    /// * `jd_start`      - Segment start epoch.
    /// * `jd_end`        - Segment end epoch.
    /// * `segment_name`  - Name stored in the DAF name record (max 40 chars).
    ///
    /// # Errors
    /// Returns an error if `subtype` is not 0 or 1, or the data/epoch lengths
    /// are inconsistent.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        subtype: u32,
        window_size: u32,
        jd_start: Time<TDB>,
        jd_end: Time<TDB>,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let record_size: usize = match subtype {
            0 => 12,
            1 => 6,
            _ => return Err(Error::ValueError("Type 18: subtype must be 0 or 1.".into())),
        };
        let n = epochs.len();
        if n == 0 {
            return Err(Error::ValueError(
                "Type 18: need at least one record.".into(),
            ));
        }
        if records.len() != n * record_size {
            return Err(Error::ValueError(format!(
                "Type 18: records length ({}) must be n ({}) * record_size ({})",
                records.len(),
                n,
                record_size
            )));
        }
        if (window_size as usize) > n {
            return Err(Error::ValueError(
                "Type 18: window_size must be <= n.".into(),
            ));
        }
        for w in epochs.windows(2) {
            if w[1] <= w[0] {
                return Err(Error::ValueError(
                    "Type 18: epochs must be strictly increasing.".into(),
                ));
            }
        }

        // Layout: [n*record_size data][n epochs][directory][subtype][window_size][n]
        let n_dir = if n > 100 { (n - 1) / 100 } else { 0 };
        let mut data = Vec::with_capacity(n * (record_size + 1) + n_dir + 3);
        data.extend_from_slice(records);
        data.extend_from_slice(epochs);
        for i in 1..=n_dir {
            data.push(epochs[i * 100 - 1]);
        }
        data.push(f64::from(subtype));
        data.push(f64::from(window_size));
        data.push(n as f64);

        Ok(SpkArray::new(
            object_id,
            center_id,
            frame_id,
            18,
            jd_start,
            jd_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    fn get_record(&self, idx: usize) -> Type18RecordView<'_> {
        unsafe {
            let rec = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.record_size..(idx + 1) * self.record_size);
            Type18RecordView {
                pos: &rec[0..self.record_size / 2],
                vel: &rec[self.record_size / 2..self.record_size],
            }
        }
    }

    #[inline(always)]
    fn get_times(&self) -> &[f64] {
        unsafe {
            self.array.daf.data.get_unchecked(
                self.n_records * self.record_size..self.n_records * (self.record_size + 1),
            )
        }
    }

    #[inline(always)]
    #[allow(
        clippy::cast_possible_wrap,
        reason = "This is correct as long as the file is correct."
    )]
    pub(crate) fn try_get_pos_vel(&self, jds: f64) -> ([f64; 3], [f64; 3]) {
        let times = self.get_times();
        let start_idx: isize = match times.binary_search_by(|probe| probe.total_cmp(&jds)) {
            Ok(c) => c as isize - (self.window_size as isize) / 2,
            Err(c) => {
                if (jds - times[c - 1]).abs() < (jds - times[c]).abs() {
                    c as isize - 1 - self.window_size as isize / 2
                } else {
                    c as isize - self.window_size as isize / 2
                }
            }
        };

        #[allow(
            clippy::cast_sign_loss,
            reason = "This is correct as long as the file is correct."
        )]
        let start_idx =
            start_idx.clamp(0, self.n_records as isize - self.window_size as isize) as usize;

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        match self.subtype {
            0 => {
                for idx in 0..3 {
                    {
                        let p: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).pos[idx])
                            .collect();
                        let dp: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).pos[idx + 3])
                            .collect();
                        let (p, _) = hermite_interpolation(
                            &times[start_idx..start_idx + self.window_size],
                            &p,
                            &dp,
                            jds,
                        );
                        pos[idx] = p / AU_KM;
                    }
                    {
                        let p: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).vel[idx])
                            .collect();
                        let dp: Box<[f64]> = (0..self.window_size)
                            .map(|i| self.get_record(i + start_idx).vel[idx + 3])
                            .collect();
                        let (v, _) = hermite_interpolation(
                            &times[start_idx..start_idx + self.window_size],
                            &p,
                            &dp,
                            jds,
                        );
                        vel[idx] = v / AU_KM * 86400.;
                    }
                }
            }
            1 => {
                for idx in 0..3 {
                    let mut p: Box<[f64]> = (0..self.window_size)
                        .map(|i| self.get_record(i + start_idx).pos[idx])
                        .collect();
                    let mut dp: Box<[f64]> = (0..self.window_size)
                        .map(|i| self.get_record(i + start_idx).vel[idx])
                        .collect();
                    let p = lagrange_interpolation(
                        &times[start_idx..start_idx + self.window_size],
                        &mut p,
                        jds,
                    );
                    let v = lagrange_interpolation(
                        &times[start_idx..start_idx + self.window_size],
                        &mut dp,
                        jds,
                    );
                    pos[idx] = p / AU_KM;
                    vel[idx] = v / AU_KM * 86400.;
                }
            }
            _ => {
                unreachable!()
            }
        }
        (pos, vel)
    }
}

/// Type 18 Record View
/// A view into a record of type 18, provided mainly for clarity to the underlying
/// data structure.
struct Type18RecordView<'a> {
    pos: &'a [f64],
    vel: &'a [f64],
}

impl TryFrom<SpkArray> for SpkSegmentType18 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        reason = "This is correct as long as the file is correct."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let mut window_size = array.daf[array.daf.len() - 2] as usize;
        let subtype = array.daf[array.daf.len() - 3] as usize;
        let record_size = {
            if subtype == 0 {
                12
            } else if subtype == 1 {
                6
            } else {
                return Err(Error::ValueError(
                    "SPK Segment Type 18 only supports subtype of 0 or 1".into(),
                ));
            }
        };
        if window_size > n_records {
            eprintln!(
                "Spk Segment Type 18 must have at least as many records as the window size, n_records={n_records}, window_size={window_size}",
            );
            window_size = n_records;
        }

        Ok(Self {
            array,
            subtype,
            window_size,
            n_records,
            record_size,
        })
    }
}
