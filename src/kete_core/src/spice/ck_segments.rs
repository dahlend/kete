use crate::errors::{Error, KeteResult};
use crate::frames::{quaternion_to_euler, NonInertialFrame};
use crate::time::scales::TDB;
use crate::time::Time;
use nalgebra::{Quaternion, Rotation3, Unit};

use super::CkArray;
use super::LOADED_SCLK;

#[derive(Debug)]
pub(in crate::spice) enum CkSegment {
    Type3(CkSegmentType3),
}

impl CkSegment {
    pub fn try_get_orientation<T: NonInertialFrame>(
        &self,
        naif_id: i32,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, T)> {
        let arr_ref: &CkArray = self.into();
        if arr_ref.naif_id != naif_id {
            return Err(Error::DAFLimits(format!(
                "NAIF ID is not present in this record. {}",
                arr_ref.naif_id
            )));
        }

        match self {
            CkSegment::Type3(seg) => seg.try_get_orientation(time),
        }
    }
}

impl<'a> From<&'a CkSegment> for &'a CkArray {
    fn from(value: &'a CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => &seg.array,
        }
    }
}

impl From<CkSegment> for CkArray {
    fn from(value: CkSegment) -> Self {
        match value {
            CkSegment::Type3(seg) => seg.array,
        }
    }
}

impl TryFrom<CkArray> for CkSegment {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        match array.segment_type {
            3 => Ok(CkSegment::Type3(array.try_into()?)),
            v => Err(Error::IOError(format!(
                "CK Segment type {:?} not supported.",
                v
            ))),
        }
    }
}

/// Discrete pointing data with linear interpolation between.
///
/// This segment type is broken up into intervals, each with a beginning and
/// end. One or more data points may be contained within each intervals. Linear
/// interpolation may be performed within a intervals.
///
/// Queries may include a user supplied tolerance on the requested time.
///
/// Interpolation does not extend past the bounds of an interval, the closest
/// point may be returned, provided it is within the specified tolerance.
///
/// Single points of data are allowed (no interpolation as long as it is within
/// the tolerance).
///
/// https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/ck.html#Data%20Type%203

#[derive(Debug)]
#[allow(unused)]
pub(in crate::spice) struct CkSegmentType3 {
    array: CkArray,
    n_intervals: usize,
    n_records: usize,
    rec_size: usize,

    interval_start_idx: usize,
    time_start_idx: usize,
}

#[allow(unused)]
impl CkSegmentType3 {
    fn get_record(&self, idx: usize) -> Type3RecordView {
        unsafe {
            let rec = self
                .array
                .daf
                .data
                .get_unchecked(idx * self.rec_size..(idx + 1) * self.rec_size);
            Type3RecordView {
                quaternion: &rec[..4],
                accel: &rec[4..],
            }
        }
    }

    fn interval_starts(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.interval_start_idx..self.interval_start_idx + self.n_intervals)
        }
    }

    fn record_times(&self) -> &[f64] {
        unsafe {
            self.array
                .daf
                .data
                .get_unchecked(self.time_start_idx..self.time_start_idx + self.n_records)
        }
    }

    /// Get the list of times inside of the interval.
    ///
    /// This queries the directory of interval start times, then uses the
    /// start and stop of the matching interval to find the list of times
    /// in the SCLK time directory.
    ///
    /// If the time requested is not within any interval, return the closest
    /// interval, and the start index of the associated clock times.
    fn get_times_in_interval(&self, time_sclk: f64) -> KeteResult<(&[f64], usize)> {
        // first, check if the time is inside a known interval
        let interval_starts = self.interval_starts();
        if self.n_intervals == 1 {
            // If there is only one interval, return its times
            return Ok((self.record_times(), 0));
        }
        let mut interval_idx = interval_starts.partition_point(|&x| x <= time_sclk);

        // if the interval_index is the last one, or the second to last one, return the last interval
        if interval_idx >= self.n_intervals - 1 {
            interval_idx = self.n_intervals - 1;
        }
        let interval_start_time = interval_starts[interval_idx];
        let interval_stop_time = interval_starts[interval_idx + 1];

        // find the start and stop index in the time directory uing the interval times
        let record_times = self.record_times();
        let start_idx = record_times.partition_point(|&x| x < interval_start_time);
        let stop_idx = record_times.partition_point(|&x| x <= interval_stop_time);
        Ok((&record_times[start_idx..stop_idx], start_idx))
    }

    pub fn try_get_orientation<T: NonInertialFrame>(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, T)> {
        if T::spice_frame_id() != self.array.frame_id {
            return Err(Error::ValueError(format!(
                "Frame ID {} does not match segment frame ID {}.",
                T::spice_frame_id(),
                self.array.frame_id
            )));
        }
        let (time, quaternion, accel) = self.get_quaternion_at_time(time)?;
        let rot: Rotation3<_> = quaternion.into();

        let accel: [f64; 3] = accel.unwrap_or_default();

        /// convert quaternion to euler angles in the order X, Z, X
        let angles = quaternion_to_euler::<'X', 'Z', 'X'>(quaternion.into_inner());
        let values = [
            angles[0], angles[1], angles[2], accel[0], accel[1], accel[2],
        ];

        Ok((time, T::new(values)))
    }

    /// Return the record at the given time, interpolating if necessary.
    ///
    /// This will return the best effort record, along with the time of
    /// the record. If the requested time is outside of any interval, this
    /// will return the closest record.
    pub fn get_quaternion_at_time(
        &self,
        time: Time<TDB>,
    ) -> KeteResult<(Time<TDB>, Unit<Quaternion<f64>>, Option<[f64; 3]>)> {
        let sclk = LOADED_SCLK
            .try_read()
            .map_err(|_| Error::IOError("Failed to read SCLK data.".into()))?;
        let tick = sclk.try_time_to_tick(self.array.naif_id, time)?;

        // If there is only one record, return it immediately.
        if self.n_records == 1 {
            let record = self.get_record(0);
            let t = sclk.try_tick_to_time(self.array.naif_id, self.record_times()[0])?;
            let (quat, accel) = record.into();
            return Ok((t, Unit::from_quaternion(quat), accel));
        }

        let (interval_times, start_idx) = self.get_times_in_interval(tick)?;

        // find the closest two times in the interval
        let mut idx = interval_times.partition_point(|&x| x <= tick);
        if interval_times.len() == idx {
            // if the index is after the end of the interval, return the last record
            let record = self.get_record(idx - 1);
            let (quat, accel) = record.into();
            let t = sclk.try_tick_to_time(self.array.naif_id, *(interval_times.last().unwrap()))?;
            Ok((t, Unit::from_quaternion(quat), accel))
        } else if idx == 0 {
            // if the index is before the beginning of the interval, return the first record
            let record = self.get_record(0);
            let (quat, accel) = record.into();
            let t =
                sclk.try_tick_to_time(self.array.naif_id, *(interval_times.first().unwrap()))?;
            Ok((t, Unit::from_quaternion(quat), accel))
        } else {
            // otherwise, we have two records to interpolate between
            idx -= 1;
            let t1 = interval_times[idx];
            let t2 = interval_times[idx + 1];
            let (q0, acc0) = self.get_record(start_idx + idx).into();
            let (q1, acc1) = self.get_record(start_idx + idx + 1).into();
            let dt = (tick - t1) / (t2 - t1);
            let quaternion = q0.lerp(&q1, dt);

            let accel: Option<[f64; 3]> = acc0.map(|acc0| {
                acc0.iter()
                    .zip(acc1.unwrap())
                    .map(|(a1, a2)| a1 * (1.0 - dt) + a2 * dt)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            });

            Ok((time, Unit::from_quaternion(quaternion), accel))
        }
    }
}

#[allow(unused)]
struct Type3RecordView<'a> {
    quaternion: &'a [f64],
    accel: &'a [f64],
}

impl From<Type3RecordView<'_>> for (Quaternion<f64>, Option<[f64; 3]>) {
    fn from(record: Type3RecordView) -> Self {
        let quaternion: [f64; 4] = record.quaternion.try_into().unwrap();
        let quaternion =
            Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        let accel = if record.accel.is_empty() {
            None
        } else {
            Some(record.accel.try_into().unwrap())
        };
        (quaternion, accel)
    }
}

impl TryFrom<CkArray> for CkSegmentType3 {
    type Error = Error;

    fn try_from(array: CkArray) -> Result<Self, Self::Error> {
        let n_records = array.daf[array.daf.len() - 1] as usize;
        let n_intervals = array.daf[array.daf.len() - 2] as usize;

        if n_records == 0 {
            return Err(Error::IOError(
                "CK File does not contain any records.".into(),
            ));
        }
        if n_intervals == 0 {
            return Err(Error::IOError(
                "CK File does not contain any intervals of records.".into(),
            ));
        }

        let rec_size = if array.produces_angular_rates { 7 } else { 4 };

        // Times are also broken up into a 'directory' of every 100th time.
        // This calculates the size of the directory.
        let time_dir_size = (n_records - 1) / 100;

        // interval times are also broken up into a 'directory' of every 100th
        // interval start time. This calculates the size of the directory.
        let interval_dir_size = (n_intervals - 1) / 100;

        // there are n_records
        let mut expected_size = n_records * rec_size;
        // 2 lists of times + 2 numbers at the end
        expected_size += n_intervals + n_records + 2;
        // 2 directories
        expected_size += time_dir_size + interval_dir_size;

        if expected_size != array.daf.len() {
            return Err(Error::IOError(
                "CK File not formatted correctly. Number of records found in file don't match expected."
                    .into(),
            ));
        }

        let time_start_idx = n_records * rec_size;
        let interval_start_idx = time_start_idx + n_records + time_dir_size;

        Ok(CkSegmentType3 {
            array,
            n_intervals,
            n_records,
            rec_size,
            interval_start_idx,
            time_start_idx,
        })
    }
}
