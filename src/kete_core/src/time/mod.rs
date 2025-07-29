//! Time representation and conversions
//!
//! See [`TimeScale`] for a list of supported Time Scales.
//! See [`Time`] for the representation of time itself.
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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

use std::{
    marker::PhantomData,
    ops::{Add, Sub},
};

mod leap_second;
mod scales;

use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};

use crate::prelude::{Error, KeteResult};

// pub use self::leap_second::{};
pub use self::scales::{JD_TO_MJD, TAI, TDB, TimeScale, UTC};

/// Representation of Time.
///
/// This supports different time scaling standards via the [`TimeScale`] trait.
///
/// Machine precision between float 64s with numbers near J2000 (IE: 2451545.0) is
/// around 23 microseconds (2.7e-10 days). So times near J2000 by necessity can only
/// be represented with about 23 microsecond accuracy. This may be dealt with by
/// splitting time into two components, integer days and a float to represent the
/// fraction of a day. However as long as accuracy below ~30us is not required, then
/// a single f64 is sufficient.
///
/// The intended accuracy of conversions is at minimum on the 1-3ms time scale.
/// Due to this not requiring more accuracy than this, TDB can be assumed to be TT.
///
/// Leap seconds are defined at precisely the MJD in TAI scaled time when the IERS
/// leap second file specifies.
///
/// Any conversions to a single float will by necessity result in some small accuracy
/// loss due to the nature of the representation of numbers on computers.
///
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct Time<T: TimeScale> {
    /// Julian Date
    pub jd: f64,

    /// [`PhantomData`] is used here as the scale is only a record keeping convenience.
    scale_type: PhantomData<T>,
}

/// Convert Hours/Minutes/Seconds/millisecond to fraction of a day
pub fn hour_min_sec_to_day(h: u32, m: u32, s: u32, ms: u32) -> f64 {
    h as f64 / 24.0
        + m as f64 / 60. / 24.
        + s as f64 / 60. / 60. / 24.
        + ms as f64 / 1000. / 60. / 60. / 24.
}

/// Convert fraction of a day to hours/minutes/seconds/microseconds
pub fn frac_day_to_hmsms(mut frac: f64) -> Option<(u32, u32, u32, u32)> {
    if frac.is_sign_negative() || frac.abs() >= 1.0 {
        return None;
    }
    frac *= 24.0;
    let hour = frac as u32;
    frac -= hour as f64;
    frac *= 60.0;
    let minute = frac as u32;
    frac -= minute as f64;
    frac *= 60.0;
    let second = frac as u32;
    frac -= second as f64;
    frac *= 1000.0;

    Some((hour, minute, second, frac as u32))
}

/// Days in the provided year.
///
/// Returns 366 for leap years, 365 otherwise.
///
/// This is a proleptic implementation, meaning it does not take into account
/// the Gregorian calendar reform, which is correct for most applications.
///
fn days_in_year(year: i64) -> u64 {
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    if is_leap { 366 } else { 365 }
}

impl Time<UTC> {
    /// Read time from the standard ISO format for time.
    pub fn from_iso(s: &str) -> KeteResult<Self> {
        let time = DateTime::parse_from_rfc3339(s)?.to_utc();
        Self::from_datetime(&time)
    }

    /// Construct time from the current time.
    pub fn now() -> KeteResult<Self> {
        Self::from_datetime(&Utc::now())
    }

    /// Construct a Time object from a UTC [`DateTime`].
    pub fn from_datetime(time: &DateTime<Utc>) -> KeteResult<Self> {
        let frac_day = hour_min_sec_to_day(
            time.hour(),
            time.minute(),
            time.second(),
            time.timestamp_subsec_millis(),
        );
        Ok(Self::from_year_month_day(
            time.year() as i64,
            time.month(),
            time.day(),
            frac_day,
        ))
    }

    /// Return the Gregorian year, month, day, and fraction of a day.
    ///
    /// Algorithm from:
    /// "A Machine Algorithm for Processing Calendar Dates"
    /// <https://doi.org/10.1145/364096.364097>
    ///
    pub fn year_month_day(&self) -> (i64, u32, u32, f64) {
        let offset = self.jd + 0.5;
        let frac_day = offset.rem_euclid(1.0);

        let mut l = offset.div_euclid(1.0) as i64 + 68569;

        let n = (4 * l).div_euclid(146097);
        l -= (146097 * n + 3).div_euclid(4);
        let i = (4000 * (l + 1)).div_euclid(1461001);
        l -= (1461 * i).div_euclid(4) - 31;
        let k = (80 * l).div_euclid(2447);
        let day = l - (2447 * k).div_euclid(80);
        l = k.div_euclid(11);

        let month = k + 2 - 12 * l;
        let year = 100 * (n - 49) + i + l;
        (year, month as u32, day as u32, frac_day)
    }

    /// Return the current time as a fraction of the year.
    ///
    /// ```
    ///    use kete_core::time::{Time, UTC};
    ///
    ///    let time = Time::from_year_month_day(2010, 1, 1, 0.0);
    ///    assert_eq!(time.year_as_float(), 2010.0);
    ///
    ///    let time = Time::<UTC>::new(2457754.5);
    ///    assert_eq!(time.year_as_float(), 2017.0);
    ///
    ///    let time = Time::<UTC>::new(2457754.5 + 364.9999);
    ///    assert_eq!(time.year_as_float(), 2017.999999726028);
    ///
    ///    let time = Time::<UTC>::new(2457754.5 + 365.0 / 2.0);
    ///    assert_eq!(time.year_as_float(), 2017.5);
    ///
    ///    // 2016 was a leap year, so 366 days instead of 365.
    ///    let time = Time::<UTC>::new(2457754.5 - 366.0);
    ///    assert_eq!(time.year_as_float(), 2016.0);
    ///
    ///    let time = Time::<UTC>::new(2457754.5 - 366.0 / 2.0);
    ///    assert_eq!(time.year_as_float(), 2016.5);
    ///
    /// ```
    ///
    pub fn year_as_float(&self) -> f64 {
        let datetime = self.to_datetime().unwrap();

        // ordinal is the integer day of the year, starting at 0.
        let mut ordinal = datetime.ordinal0() as f64;

        // we need to add the fractional day to the ordinal.
        let (_, _, _, frac_day) = self.year_month_day();
        ordinal += frac_day;

        let year = datetime.year() as f64;
        let days_in_year = days_in_year(datetime.year() as i64) as f64;
        year + ordinal / days_in_year
    }

    /// Create Time from the date in the Gregorian calendar.
    ///
    /// Algorithm from:
    /// "A Machine Algorithm for Processing Calendar Dates"
    /// <https://doi.org/10.1145/364096.364097>
    ///
    pub fn from_year_month_day(year: i64, month: u32, day: u32, frac_day: f64) -> Self {
        let frac_day = frac_day - 0.5;
        let day = day as i64 + frac_day.div_euclid(1.0) as i64;
        let frac_day = frac_day.rem_euclid(1.0);
        let month = month as i64;

        let tmp = (month - 14) / 12;
        let days = day - 32075 + 1461 * (year + 4800 + tmp) / 4 + 367 * (month - 2 - tmp * 12) / 12
            - 3 * ((year + 4900 + tmp) / 100) / 4;

        Self::new(days as f64 + frac_day)
    }

    /// Create a [`DateTime`] object
    pub fn to_datetime(&self) -> KeteResult<DateTime<Utc>> {
        let (y, month, d, f) = self.year_month_day();
        let (h, m, s, ms) = frac_day_to_hmsms(f).unwrap();
        Ok(NaiveDate::from_ymd_opt(y as i32, month, d)
            .ok_or(Error::ValueError("Failed to convert ymd".into()))?
            .and_hms_milli_opt(h, m, s, ms)
            .ok_or(Error::ValueError("Failed to convert hms".into()))?
            .and_utc())
    }

    /// Construct a ISO compliant UTC string.
    pub fn to_iso(&self) -> KeteResult<String> {
        let datetime = self.to_datetime()?;
        Ok(datetime.to_rfc3339())
    }

    /// J2000 reference time.
    /// 2451545.0
    pub fn j2000() -> Time<TDB> {
        Time::<TDB>::new(2451545.0)
    }
}

impl<T: TimeScale> Time<T> {
    /// Construct a new Time object.
    pub fn new(jd: f64) -> Self {
        Self {
            jd,
            scale_type: PhantomData,
        }
    }

    /// Create Time from an Modified Julian Date (MJD).
    pub fn from_mjd(mjd: f64) -> Self {
        Self {
            jd: mjd - JD_TO_MJD,
            scale_type: PhantomData,
        }
    }

    /// Cast to UTC scaled time.
    pub fn utc(&self) -> Time<UTC> {
        Time::<UTC>::new(UTC::from_tdb(T::to_tdb(self.jd)))
    }

    /// Cast to TAI scaled time.
    pub fn tai(&self) -> Time<TAI> {
        Time::<TAI>::new(TAI::from_tdb(T::to_tdb(self.jd)))
    }

    /// Cast to TDB scaled time.
    pub fn tdb(&self) -> Time<TDB> {
        Time::<TDB>::new(T::to_tdb(self.jd))
    }

    /// Convert to an MJD float.
    pub fn mjd(&self) -> f64 {
        self.jd + JD_TO_MJD
    }

    /// Convert Time from one scale to another.
    pub fn into<Target: TimeScale>(&self) -> Time<Target> {
        Time::<Target>::new(Target::from_tdb(self.tdb().jd))
    }
}

impl<T: TimeScale> From<f64> for Time<T> {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl<A: TimeScale, B: TimeScale> Sub<Time<B>> for Time<A> {
    type Output = f64;

    /// Subtract two times, returning the difference in days.
    fn sub(self, other: Time<B>) -> Self::Output {
        self.tdb().jd - other.tdb().jd
    }
}

impl<A: TimeScale, B: TimeScale> Add<Time<B>> for Time<A> {
    type Output = Self;

    /// Add two times together, returning a new time in the same scale as `A`.
    fn add(self, other: Time<B>) -> Self {
        Self::new(A::from_tdb(self.tdb().jd + other.tdb().jd))
    }
}

#[cfg(test)]
mod tests {

    use scales::TT_TO_TAI;

    use super::*;

    #[test]
    fn test_time() {
        let t = Time::<UTC>::new(2451545.);
        assert!(t.year_month_day() == (2000, 1, 1, 0.5));

        let t2 = Time::<UTC>::from_year_month_day(2000, 1, 1, 0.5);
        assert!(t2.jd == 2451545.);

        let t3 = Time::<UTC>::from_year_month_day(2000, 1, 2, -0.5);
        assert!(t3.jd == 2451545.);

        let t4 = Time::<UTC>::new(2000000.);
        assert!(t4.year_month_day() == (763, 9, 18, 0.5));

        let t5 = Time::<UTC>::from_year_month_day(763, 9, 18, 0.5);
        assert!(t5.jd == 2000000.);

        let ymd = Time::<UTC>::new(-68774.4991992591).year_month_day();
        assert!(ymd.0 == -4901);
        assert!(ymd.1 == 8);
        assert!(ymd.2 == 8);
    }

    #[test]
    fn test_time_near_leap_second() {
        for offset in -1000..1000 {
            let offset = offset as f64 / 10.0;
            let mjd = 41683.0 + offset / 86400.0; // TIME IN TAI
            let t = Time::<TAI>::from_mjd(mjd);
            let t = t.tdb();
            let t = t.tai();

            // Numerical precision of times near J2000 is only around 1e-10
            assert!((t.mjd() - mjd).abs() < 1e-9,);
        }

        // Perform round trip conversions in the seconds around a leap second.
        for offset in -1000..1000 {
            let offset = offset as f64 / 10.0;
            let mjd = 41683.0 + offset / 86400.0; // TIME IN TAI
            let t = Time::<UTC>::from_mjd(mjd);
            let t = t.tai();
            let t = t.utc();

            // Numerical precision of times near J2000 is only around 1e-10
            assert!(
                (t.mjd() - mjd).abs() < 1e-9,
                "time = {} mjd = {} diff = {} sec",
                t.mjd(),
                mjd,
                (t.mjd() - mjd).abs() * 86400.0
            );
        }

        for offset in -1000..1000 {
            let offset = offset as f64 / 10.0;

            let mjd = 41683.0 + offset / 86400.0 + TT_TO_TAI;
            let t = Time::<UTC>::from_mjd(mjd);
            let t = t.tai();
            let t = t.utc();

            // Numerical precision of times near J2000 is only around 1e-10
            assert!(
                (t.mjd() - mjd).abs() < 1e-9,
                "time = {} mjd = {} diff = {} sec",
                t.mjd(),
                mjd,
                (t.mjd() - mjd).abs() * 86400.0
            );
        }
    }

    #[test]
    fn test_iso() {
        let t = Time::<UTC>::from_iso("2000-01-01T06:00:00.000Z").unwrap();
        assert!(t.year_month_day() == (2000, 1, 1, 0.25));

        let t1 = Time::<UTC>::from_iso("1987-12-25T00:00:00.000Z").unwrap();
        assert!(t1.year_month_day() == (1987, 12, 25, 0.0));
        assert!(t1.to_iso().unwrap() == "1987-12-25T00:00:00+00:00");
    }
}
