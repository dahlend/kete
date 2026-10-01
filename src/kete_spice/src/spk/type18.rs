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
    /// `object_id` is the NAIF ID of the body, and `center_id` is the NAIF ID
    /// of the center body. `frame_id` is the NAIF frame ID. `records` is the
    /// flat packet data, with one packet per epoch. A packet holds 12 values
    /// for subtype 0 and 6 values for subtype 1. `epochs` holds the packet
    /// epochs in TDB seconds from J2000. `subtype` is 0 for Hermite and 1 for
    /// Lagrange. `window_size` is the interpolation window size. `jds_start`
    /// and `jds_end` are the segment start and end, in TDB seconds from J2000.
    /// `segment_name` is the name stored in the DAF name record, which holds at
    /// most 40 characters.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - `subtype` is not 0 or 1.
    /// - `jds_start` is after `jds_end`, or either one is outside the first to
    ///   the last epoch.
    /// - The length of `records` is not the packet size times `epochs.len()`.
    /// - `window_size` is greater than `epochs.len()`.
    /// - The epochs are not strictly increasing.
    ///
    /// Returns [`Error::IOError`] from `check_packet_layout` in these cases:
    /// - `epochs` holds fewer than 2 values.
    /// - `window_size` is odd, less than 2, or greater than 8 for subtype 0 or
    ///   16 for subtype 1.
    pub fn new_array(
        object_id: i32,
        center_id: i32,
        frame_id: i32,
        records: &[f64],
        epochs: &[f64],
        subtype: u32,
        window_size: u32,
        jds_start: f64,
        jds_end: f64,
        segment_name: &str,
    ) -> KeteResult<SpkArray> {
        let record_size: usize = match subtype {
            0 => 12,
            1 => 6,
            _ => return Err(Error::ValueError("Type 18: subtype must be 0 or 1.".into())),
        };
        let n = epochs.len();
        check_packet_layout(
            subtype as usize,
            n,
            window_size as usize,
            record_size,
            n * (record_size + 1),
            TYPE18_MAX_DEGREE,
        )?;
        if !(epochs[0] <= jds_start && jds_start <= jds_end && jds_end <= epochs[n - 1]) {
            return Err(Error::ValueError(
                "Type 18: the segment start and end must be ordered and within the epochs.".into(),
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
            jds_start,
            jds_end,
            data,
            segment_name.to_string(),
        ))
    }

    #[inline(always)]
    pub(crate) fn try_get_pos_vel(&self, time: Time<TDB>) -> ([f64; 3], [f64; 3]) {
        PacketSeries {
            packets: &self.array.daf.data[..self.n_records * self.record_size],
            epochs: &self.array.daf.data
                [self.n_records * self.record_size..self.n_records * (self.record_size + 1)],
            subtype: self.subtype,
            window_size: self.window_size,
            record_size: self.record_size,
        }
        .try_get_pos_vel(time)
    }
}

/// Return the number of values in a packet of an ESOC/DDID subtype.
///
/// A subtype 0 packet holds 12 values. A subtype 1 or subtype 2 packet holds 6
/// values. Type 18 supports subtypes 0 and 1. Subtype 2 occurs only in type 19
/// mini-segments. Any other subtype gives `None`.
pub(in crate::spk) const fn packet_size(subtype: usize) -> Option<usize> {
    match subtype {
        0 => Some(12),
        1 | 2 => Some(6),
        _ => None,
    }
}

/// A run of ESOC/DDID packets with their epochs.
///
/// A type 18 segment and each mini-segment of a type 19 segment share this
/// layout, so both use this evaluator. The packets hold positions in km and
/// velocities in km/s. The evaluator returns positions in AU and velocities in
/// AU/day.
pub(in crate::spk) struct PacketSeries<'a> {
    pub packets: &'a [f64],
    pub epochs: &'a [f64],
    pub subtype: usize,
    pub window_size: usize,
    pub record_size: usize,
}

impl PacketSeries<'_> {
    #[inline(always)]
    fn record(&self, idx: usize) -> &[f64] {
        // SAFETY: Each constructor sets `packets` to hold `record_size` values
        // for each epoch. The only caller takes `idx` from `window`, which
        // keeps it below `epochs.len()`.
        unsafe {
            self.packets
                .get_unchecked(idx * self.record_size..(idx + 1) * self.record_size)
        }
    }

    /// Return the first index and the size of the interpolation window.
    ///
    /// This follows the SPKR18 and SPKR19 window rule. The interval starts at
    /// the last epoch strictly before `jds`. If no epoch is before `jds`, the
    /// interval starts at the first epoch. The window takes up to half its
    /// nominal size on each side of this interval. Near either end of the
    /// series, the window shrinks and does not shift.
    #[inline(always)]
    fn window(&self, jds: f64) -> (usize, usize) {
        let n = self.epochs.len();
        let low = self.epochs.partition_point(|&t| t < jds).max(1) - 1;
        let half = self.window_size / 2;
        let left = half.min(low + 1);
        let right = half.min(n - low - 1);
        (low + 1 - left, left + right)
    }

    #[inline(always)]
    pub(in crate::spk) fn try_get_pos_vel(&self, time: Time<TDB>) -> ([f64; 3], [f64; 3]) {
        let jds = time.j2000_seconds();
        let (start, window) = self.window(jds);
        let times = &self.epochs[start..start + window];
        let offset = time.j2000_seconds_minus(times[0]);

        let mut pos = [0.0; 3];
        let mut vel = [0.0; 3];
        for idx in 0..3 {
            match self.subtype {
                // Position and velocity each use a Hermite fit with their own
                // stored derivative.
                0 => {
                    let p: Box<[f64]> = (0..window).map(|i| self.record(i + start)[idx]).collect();
                    let dp: Box<[f64]> = (0..window)
                        .map(|i| self.record(i + start)[idx + 3])
                        .collect();
                    let (p, _) = hermite_interpolation(times, &p, &dp, offset);
                    pos[idx] = p / AU_KM;

                    let v: Box<[f64]> = (0..window)
                        .map(|i| self.record(i + start)[idx + 6])
                        .collect();
                    let dv: Box<[f64]> = (0..window)
                        .map(|i| self.record(i + start)[idx + 9])
                        .collect();
                    let (v, _) = hermite_interpolation(times, &v, &dv, offset);
                    vel[idx] = v / AU_KM * 86400.;
                }
                // The packets hold no derivative data. Position and velocity
                // each use a Lagrange fit.
                1 => {
                    let mut p: Box<[f64]> =
                        (0..window).map(|i| self.record(i + start)[idx]).collect();
                    let mut v: Box<[f64]> = (0..window)
                        .map(|i| self.record(i + start)[idx + 3])
                        .collect();
                    pos[idx] = lagrange_interpolation(times, &mut p, offset) / AU_KM;
                    vel[idx] = lagrange_interpolation(times, &mut v, offset) / AU_KM * 86400.;
                }
                // The stored velocity is the derivative for the Hermite fit of
                // the position. One fit gives both position and velocity.
                2 => {
                    let p: Box<[f64]> = (0..window).map(|i| self.record(i + start)[idx]).collect();
                    let dp: Box<[f64]> = (0..window)
                        .map(|i| self.record(i + start)[idx + 3])
                        .collect();
                    let (p, v) = hermite_interpolation(times, &p, &dp, offset);
                    pos[idx] = p / AU_KM;
                    vel[idx] = v / AU_KM * 86400.;
                }
                _ => unreachable!(),
            }
        }
        (pos, vel)
    }
}

impl TryFrom<SpkArray> for SpkSegmentType18 {
    type Error = Error;

    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "Values are checked to be non-negative whole numbers first."
    )]
    fn try_from(array: SpkArray) -> KeteResult<Self> {
        let len = array.daf.len();
        if len < 3 {
            return Err(Error::IOError("SPK Type 18: segment is truncated.".into()));
        }
        let control = [array.daf[len - 3], array.daf[len - 2], array.daf[len - 1]];
        if control
            .iter()
            .any(|x| !(x.is_finite() && *x >= 0.0 && x.fract() == 0.0))
        {
            return Err(Error::IOError(format!(
                "SPK Type 18: invalid control words {control:?}."
            )));
        }
        let [subtype, window_size, n_records] = control.map(|x| x as usize);

        // Subtype 2 uses the ESOC/DDID packet layout, but it occurs only inside
        // type 19 mini-segments. This reader rejects it.
        let record_size = match subtype {
            0 | 1 => packet_size(subtype).unwrap_or(0),
            _ => {
                return Err(Error::ValueError(
                    "SPK Segment Type 18 only supports subtype of 0 or 1".into(),
                ));
            }
        };
        check_packet_layout(
            subtype,
            n_records,
            window_size,
            record_size,
            len - 3,
            TYPE18_MAX_DEGREE,
        )?;

        Ok(Self {
            array,
            subtype,
            window_size,
            n_records,
            record_size,
        })
    }
}

/// Largest interpolation degree SPICE supports in type 18 segments.
pub(in crate::spk) const TYPE18_MAX_DEGREE: usize = 15;

/// Largest interpolation degree SPICE supports in type 19 mini-segments.
pub(in crate::spk) const TYPE19_MAX_DEGREE: usize = 27;

/// Check the control words of a type 18 segment or a type 19 mini-segment.
///
/// `available` is the number of values before the control words. `max_degree`
/// is the largest interpolation degree for the segment type. The largest window
/// is `(max_degree + 1) / 2` for subtype 0 and `max_degree + 1` for the other
/// subtypes. This check accepts a window larger than the number of packets. The
/// evaluator shrinks such a window to the packets available.
///
/// # Errors
/// Returns [`Error::IOError`] in these cases:
/// - `n_records` is less than 2.
/// - `window_size` is odd, less than 2, or greater than the largest window.
/// - `available` is less than `n_records * (record_size + 1)`.
pub(in crate::spk) fn check_packet_layout(
    subtype: usize,
    n_records: usize,
    window_size: usize,
    record_size: usize,
    available: usize,
    max_degree: usize,
) -> KeteResult<()> {
    if n_records < 2 {
        return Err(Error::IOError(format!(
            "ESOC/DDID segment needs at least 2 packets, found {n_records}."
        )));
    }
    let max_window = if subtype == 0 {
        max_degree.div_ceil(2)
    } else {
        max_degree + 1
    };
    if window_size < 2 || !window_size.is_multiple_of(2) || window_size > max_window {
        return Err(Error::IOError(format!(
            "ESOC/DDID window size must be even and in [2, {max_window}], found \
             {window_size}."
        )));
    }
    if n_records
        .checked_mul(record_size + 1)
        .is_none_or(|needed| needed > available)
    {
        return Err(Error::IOError(format!(
            "ESOC/DDID segment holds {available} values, too few for {n_records} packets."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(epochs: &[f64], window_size: usize) -> PacketSeries<'_> {
        PacketSeries {
            packets: &[],
            epochs,
            subtype: 1,
            window_size,
            record_size: 6,
        }
    }

    /// Check window placement against the SPKR18 rule, with epochs every 60 s.
    #[test]
    fn window_matches_spkr18() {
        let epochs: Vec<f64> = (0..11).map(|i| f64::from(i) * 60.0).collect();
        let s = series(&epochs, 4);

        // Half the window on each side of the containing interval.
        assert_eq!(s.window(130.0), (1, 4));
        assert_eq!(s.window(170.0), (1, 4));
        // An exact epoch belongs to the interval ending on it.
        assert_eq!(s.window(180.0), (1, 4));
        // At either end, the window shrinks and does not shift.
        assert_eq!(s.window(0.0), (0, 3));
        assert_eq!(s.window(30.0), (0, 3));
        assert_eq!(s.window(600.0), (8, 3));

        let s = series(&epochs[..2], 8);
        assert_eq!(s.window(30.0), (0, 2));
    }

    #[test]
    fn rejects_windows_spice_rejects() {
        let records = [0.0; 12];
        let epochs = [0.0, 60.0];
        let make = |window_size| {
            SpkSegmentType18::new_array(
                1000,
                10,
                1,
                &records,
                &epochs,
                1,
                window_size,
                0.0,
                0.0,
                "test",
            )
        };
        assert!(make(2).is_ok());
        assert!(make(0).is_err());
        assert!(make(1).is_err());
        assert!(make(3).is_err());
    }
}
