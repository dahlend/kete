// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-FileCopyrightText: 2025 California Institute of Technology
// SPDX-License-Identifier: BSD-3-Clause

//! Time representation and conversions
//!
//! See [`TimeScale`] for a list of supported Time Scales.
//! See [`Time`] for the representation of time itself.

use std::{
    cmp::Ordering,
    marker::PhantomData,
    ops::{Add, AddAssign, Sub, SubAssign},
};

mod leap_second;
mod scales;

use crate::prelude::{Error, KeteResult};
use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};

pub use self::scales::{JD_TO_MJD, TAI, TCB, TDB, TT, TimeScale, UTC};

/// Whole Julian day of the J2000 epoch, 2000-01-01 12:00 TDB.
const J2000_DAY: i64 = 2_451_545;

/// Seconds in a day.
const SECONDS_PER_DAY: f64 = 86400.0;

/// Largest separation in days of two times taken as the same instant, see
/// [`Time::same_instant`]. About 86 microseconds, above the rounding of a single f64
/// Julian date for dates within several thousand years of the present.
const SAME_INSTANT_DAYS: f64 = 1e-9;

/// Representation of Time.
///
/// This supports different time scaling standards via the [`TimeScale`] trait.
///
/// The Julian date is held as whole days plus a fraction of a day in `[0, 1)`. The
/// fraction resolves about 1e-16 day, around 10 picoseconds, at any epoch, where a
/// single f64 Julian date resolves only about 40 microseconds near the present.
/// Arithmetic on [`Time`] keeps the split, so a time plus an offset, and the
/// difference of two times, are accurate to that resolution. The split is internal:
/// times are built from and read as Julian dates, MJDs, calendar dates, or seconds
/// past J2000, and reading one back as a single f64 rounds it to f64 precision.
///
/// Conversions between time scales add the offset between the scales to the split
/// time. The offsets themselves are evaluated with the accuracy stated on each
/// [`TimeScale`].
///
/// Leap seconds take effect at 00:00 UTC on the dates in the IERS leap second file.
/// UTC before 1972, when leap seconds began, is taken to equal TAI.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Time<T: TimeScale> {
    /// Whole days of the Julian date.
    day: i64,

    /// Fraction of a day past `day`, in `[0, 1)`. A non-finite time is held here
    /// with `day` set to zero.
    frac: f64,

    /// [`PhantomData`] is used here as the scale is only a record keeping convenience.
    scale_type: PhantomData<T>,
}

impl<T: TimeScale> Time<T> {
    /// Construct a new Time object from a Julian date.
    pub fn new(jd: f64) -> Self {
        Self::from_raw(0, 0.0).add_days(jd)
    }

    /// Construct a Time object from a Julian date given in two parts, `jd1 + jd2`.
    ///
    /// The sum is kept without rounding it to a single f64, so splitting a date as
    /// whole days and a fraction of a day passes the fraction through at its own
    /// precision.
    pub fn from_parts(jd1: f64, jd2: f64) -> Self {
        Self::new(jd1).add_days(jd2)
    }

    /// Create Time from an Modified Julian Date (MJD).
    pub fn from_mjd(mjd: f64) -> Self {
        Self::from_parts(mjd, -JD_TO_MJD)
    }

    /// Construct a Time object from seconds past J2000 (JD 2451545.0) on this scale.
    ///
    /// On TDB this is SPICE ephemeris time (ET). Every day counts 86400 seconds, so on
    /// UTC the count skips leap seconds and is not elapsed time.
    pub fn from_j2000_seconds(seconds: f64) -> Self {
        if !seconds.is_finite() {
            return Self::from_raw(0, seconds);
        }
        // Whole days are taken off in seconds first, so only the remainder is divided
        // and rounded.
        let days = (seconds / SECONDS_PER_DAY).floor();
        let rest = seconds - days * SECONDS_PER_DAY;
        Self::from_parts(J2000_DAY as f64 + days, rest / SECONDS_PER_DAY)
    }

    /// The Julian date as a single f64.
    ///
    /// This rounds the time to f64 precision, about 40 microseconds near the present.
    /// Differences between times are more accurate taken as `Time - Time`.
    #[must_use]
    pub fn jd(&self) -> f64 {
        self.day as f64 + self.frac
    }

    /// The Julian date as two f64 that sum to it, the whole days and the fraction of
    /// a day in `[0, 1)`, which together keep its full precision.
    ///
    /// This is the two-part Julian date of SOFA and ERFA. A non-finite time is
    /// returned as `(0.0, jd)`.
    #[must_use]
    pub fn jd_parts(&self) -> (f64, f64) {
        (self.day as f64, self.frac)
    }

    /// Convert to an MJD float.
    #[must_use]
    pub fn mjd(&self) -> f64 {
        (self.day - 2_400_000) as f64 + (self.frac - 0.5)
    }

    /// Seconds past J2000 on this scale, SPICE ephemeris time (ET) on TDB, as a
    /// single f64.
    ///
    /// This rounds the time to f64 precision, about 60 nanoseconds near the present.
    /// An offset from a known time is more accurate from [`Self::j2000_seconds_minus`].
    #[must_use]
    pub fn j2000_seconds(&self) -> f64 {
        self.j2000_seconds_minus(0.0)
    }

    /// Seconds from `seconds`, seconds past J2000 on this scale, to this time.
    ///
    /// The whole days are subtracted before the fraction of a day is added, so the
    /// result keeps the precision of this time when `seconds` is close to it.
    #[must_use]
    pub fn j2000_seconds_minus(&self, seconds: f64) -> f64 {
        ((self.day - J2000_DAY) as f64 * SECONDS_PER_DAY - seconds) + self.frac * SECONDS_PER_DAY
    }

    /// Whether `other` is the same instant as this time, to within the rounding of a
    /// Julian date held as a single f64.
    ///
    /// Times are compared exactly by `==`. The same epoch built by two routes from f64
    /// input, such as an MJD and the equivalent JD, can differ by that rounding, so
    /// checks that two states share an epoch use this instead.
    #[must_use]
    pub fn same_instant(&self, other: &Self) -> bool {
        (*self - *other).elapsed.abs() <= SAME_INSTANT_DAYS
    }

    /// Cast to TDB scaled time.
    pub fn tdb(&self) -> Time<TDB> {
        // The split time is carried over as is and the offset between the scales added.
        Time::<TDB>::from_raw(self.day, self.frac).add_days(T::to_tdb_offset(self.jd()))
    }

    /// Convert Time from one scale to another.
    pub fn to_scale<Target: TimeScale>(&self) -> Time<Target> {
        let tdb = self.tdb();
        Time::<Target>::from_raw(tdb.day, tdb.frac).add_days(Target::from_tdb_offset(tdb.jd()))
    }

    /// Cast to UTC scaled time.
    pub fn utc(&self) -> Time<UTC> {
        self.to_scale::<UTC>()
    }

    /// Cast to TAI scaled time.
    pub fn tai(&self) -> Time<TAI> {
        self.to_scale::<TAI>()
    }

    /// Cast to TT scaled time.
    pub fn tt(&self) -> Time<TT> {
        self.to_scale::<TT>()
    }

    /// Add days to the split time on its own scale, without passing through TDB.
    ///
    /// This is the one place the split is formed: the whole days of `days` go to `day`
    /// and its fraction to `frac`. A non-finite result is held in `frac` with `day` at
    /// zero.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Whole days, checked finite"
    )]
    #[inline]
    fn add_days(self, days: f64) -> Self {
        if days == 0.0 {
            return self;
        }
        if !(days.is_finite() && self.frac.is_finite()) {
            return Self::from_raw(0, self.jd() + days);
        }
        let whole = days.floor();
        let mut day = self.day + whole as i64;
        // Both fractions are in [0, 1), so their sum carries at most one day, and
        // taking that day back off is exact.
        let mut frac = self.frac + (days - whole);
        if frac >= 1.0 {
            day += 1;
            frac -= 1.0;
        }
        Self::from_raw(day, frac)
    }

    /// The time with these exact parts, without checking them.
    const fn from_raw(day: i64, frac: f64) -> Self {
        Self {
            day,
            frac,
            scale_type: PhantomData,
        }
    }
}

impl Time<UTC> {
    /// Read time from the standard ISO format for time.
    ///
    /// # Errors
    /// An error is returned if ISO string parsing fails.
    pub fn from_iso(s: &str) -> KeteResult<Self> {
        let time = DateTime::parse_from_rfc3339(s)?.to_utc();
        Ok(Self::from_datetime(&time))
    }

    /// Construct time from the current time.
    pub fn now() -> Self {
        Self::from_datetime(&Utc::now())
    }

    /// Construct a Time object from a UTC [`DateTime`], to the nanosecond.
    pub fn from_datetime(time: &DateTime<Utc>) -> Self {
        let seconds = f64::from(time.hour() * 3600 + time.minute() * 60 + time.second());
        let frac_day = seconds / SECONDS_PER_DAY
            + f64::from(time.timestamp_subsec_nanos()) / (SECONDS_PER_DAY * 1e9);
        Self::from_year_month_day(i64::from(time.year()), time.month(), time.day(), frac_day)
    }

    /// Return the Gregorian year, month, day, and fraction of a day.
    ///
    /// Algorithm from:
    /// "A Machine Algorithm for Processing Calendar Dates"
    /// <https://doi.org/10.1145/364096.364097>
    ///
    #[must_use]
    #[allow(clippy::cast_possible_truncation, reason = "Truncation is intentional")]
    #[allow(clippy::cast_sign_loss, reason = "Sign is manually validated")]
    pub fn year_month_day(&self) -> (i32, u32, u32, f64) {
        // Calendar days start at midnight, half a day before the Julian day.
        let midnight = self.add_days(0.5);
        let frac_day = midnight.frac;

        let mut l = midnight.day + 68569;

        let n = (4 * l).div_euclid(146097);
        l -= (146097 * n + 3).div_euclid(4);
        let i = (4000 * (l + 1)).div_euclid(1461001);
        l -= (1461 * i).div_euclid(4) - 31;
        let k = (80 * l).div_euclid(2447);
        let day = l - (2447 * k).div_euclid(80);
        l = k.div_euclid(11);

        let month = k + 2 - 12 * l;
        let year = 100 * (n - 49) + i + l;
        (year as i32, month as u32, day as u32, frac_day)
    }

    /// Return the current time as a fraction of the year.
    ///
    /// ```
    ///    use kete_core::time::{Time, UTC};
    ///
    ///    let time = Time::from_year_month_day(2010, 1, 1, 0.0);
    ///    assert_eq!(time.year_as_float(), Ok(2010.0));
    ///
    ///    let time = Time::<UTC>::new(2457754.5);
    ///    assert_eq!(time.year_as_float(), Ok(2017.0));
    ///
    ///    let time = Time::<UTC>::new(2457754.5 + 364.9999);
    ///    assert_eq!(time.year_as_float(), Ok(2017.999999726028));
    ///
    ///    let time = Time::<UTC>::new(2457754.5 + 365.0 / 2.0);
    ///    assert_eq!(time.year_as_float(), Ok(2017.5));
    ///
    ///    // 2016 was a leap year, so 366 days instead of 365.
    ///    let time = Time::<UTC>::new(2457754.5 - 366.0);
    ///    assert_eq!(time.year_as_float(), Ok(2016.0));
    ///
    ///    let time = Time::<UTC>::new(2457754.5 - 366.0 / 2.0);
    ///    assert_eq!(time.year_as_float(), Ok(2016.5));
    ///
    /// ```
    ///
    /// # Errors
    /// Failure may occur if the [`NaiveDate::from_ymd_opt`] conversion fails.
    pub fn year_as_float(&self) -> KeteResult<f64> {
        let (year, month, day, frac_day) = self.year_month_day();
        let date = NaiveDate::from_ymd_opt(year, month, day)
            .ok_or(Error::ValueError("Failed to convert ymd".into()))?;

        // ordinal is the integer day of the year, starting at 0, plus the fraction
        // of the day.
        let ordinal = f64::from(date.ordinal0()) + frac_day;
        let days_in_year = f64::from(days_in_year(i64::from(year)));
        Ok(f64::from(year) + ordinal / days_in_year)
    }

    /// Create Time from the date in the Gregorian calendar.
    ///
    /// Algorithm from:
    /// "A Machine Algorithm for Processing Calendar Dates"
    /// <https://doi.org/10.1145/364096.364097>
    ///
    #[allow(clippy::cast_possible_truncation, reason = "Truncation is expected")]
    pub fn from_year_month_day(year: i64, month: u32, day: u32, frac_day: f64) -> Self {
        let month = i64::from(month);
        let tmp = (month - 14) / 12;
        let days = i64::from(day) - 32075
            + 1461 * (year + 4800 + tmp) / 4
            + 367 * (month - 2 - tmp * 12) / 12
            - 3 * ((year + 4900 + tmp) / 100) / 4;

        // `days` is the Julian day starting at noon of the date; the calendar day
        // starts half a day earlier.
        Self::from_parts(days as f64, frac_day - 0.5)
    }

    /// Create a [`DateTime`] object.
    ///
    /// The time of day is rounded to the nearest millisecond, carrying into the next
    /// day when it rounds up to midnight.
    ///
    /// # Errors
    /// Conversion to datetime object may fail for various reasons, such as the JD is
    /// too large or too small.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "Rounded and checked finite"
    )]
    pub fn to_datetime(&self) -> KeteResult<DateTime<Utc>> {
        let (year, month, day, frac) = self.year_month_day();
        if !frac.is_finite() {
            return Err(Error::ValueError("Time is not finite.".into()));
        }
        let millis = (frac * 86_400_000.0).round() as i64;
        let midnight = NaiveDate::from_ymd_opt(year, month, day)
            .ok_or(Error::ValueError("Failed to convert ymd".into()))?
            .and_time(chrono::NaiveTime::MIN);
        Ok((midnight + chrono::TimeDelta::milliseconds(millis)).and_utc())
    }

    /// Construct a ISO compliant UTC string.
    ///
    /// # Errors
    /// Conversion to datetime object may fail for various reasons, such as the JD is
    /// too large or too small.
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

impl<T: TimeScale> PartialOrd for Time<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !(self.frac.is_finite() && other.frac.is_finite()) {
            return self.jd().partial_cmp(&other.jd());
        }
        match self.day.cmp(&other.day) {
            Ordering::Equal => self.frac.partial_cmp(&other.frac),
            ord @ (Ordering::Less | Ordering::Greater) => Some(ord),
        }
    }
}

impl<T: TimeScale> From<f64> for Time<T> {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

impl<A: TimeScale, B: TimeScale> Sub<Time<B>> for Time<A> {
    type Output = Duration;

    /// Subtract two times, returning the duration in days of TDB.
    fn sub(self, other: Time<B>) -> Self::Output {
        let a = self.tdb();
        let b = other.tdb();
        Duration::new((a.day - b.day) as f64 + (a.frac - b.frac))
    }
}

impl<A: TimeScale> Add<f64> for Time<A> {
    type Output = Self;

    /// Add days on this time's own scale. On UTC that is calendar days, so the result
    /// has the same clock time; add a [`Duration`] for elapsed time across leap
    /// seconds.
    fn add(self, days: f64) -> Self::Output {
        self.add_days(days)
    }
}

impl<A: TimeScale> Sub<f64> for Time<A> {
    type Output = Self;

    /// Subtract days on this time's own scale, see [`Add<f64>`].
    fn sub(self, days: f64) -> Self::Output {
        self.add_days(-days)
    }
}

impl<A: TimeScale> AddAssign<f64> for Time<A> {
    fn add_assign(&mut self, days: f64) {
        *self = self.add_days(days);
    }
}

impl<A: TimeScale> SubAssign<f64> for Time<A> {
    fn sub_assign(&mut self, days: f64) {
        *self = self.add_days(-days);
    }
}

impl<A: TimeScale> Add<Duration> for Time<A> {
    type Output = Self;

    /// Add elapsed TDB time, the inverse of `Time - Time`.
    fn add(self, other: Duration) -> Self::Output {
        self.tdb().add_days(other.elapsed).to_scale::<A>()
    }
}

impl<A: TimeScale> Sub<Duration> for Time<A> {
    type Output = Self;

    /// Subtract elapsed TDB time, see [`Add<Duration>`].
    fn sub(self, other: Duration) -> Self::Output {
        self.tdb().add_days(-other.elapsed).to_scale::<A>()
    }
}

/// Elapsed time in TDB days.
///
/// Durations are elapsed time, not differences of labels: `Time - Time` converts both
/// times to TDB first, so a span across a UTC leap second includes it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Duration {
    /// Elapsed time in days.
    pub elapsed: f64,
}

impl Duration {
    /// Construct a new [`Duration`] object.
    pub fn new(elapsed: f64) -> Self {
        Self { elapsed }
    }
}

impl From<f64> for Duration {
    fn from(value: f64) -> Self {
        Self::new(value)
    }
}

/// Days in the provided year.
///
/// Returns 366 for leap years, 365 otherwise.
///
/// This is a proleptic implementation, meaning it does not take into account
/// the Gregorian calendar reform, which is correct for most applications.
///
fn days_in_year(year: i64) -> u32 {
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    if is_leap { 366 } else { 365 }
}

#[cfg(test)]
mod tests {

    use scales::TT_TO_TAI;

    use super::*;

    #[test]
    fn test_time() {
        let t = Time::<UTC>::new(2451545.);
        assert_eq!(t.year_month_day(), (2000, 1, 1, 0.5));

        let t2 = Time::<UTC>::from_year_month_day(2000, 1, 1, 0.5);
        assert_eq!(t2.jd(), 2451545.);

        let t3 = Time::<UTC>::from_year_month_day(2000, 1, 2, -0.5);
        assert_eq!(t3.jd(), 2451545.);

        let t4 = Time::<UTC>::new(2000000.);
        assert_eq!(t4.year_month_day(), (763, 9, 18, 0.5));

        let t5 = Time::<UTC>::from_year_month_day(763, 9, 18, 0.5);
        assert_eq!(t5.jd(), 2000000.);

        let ymd = Time::<UTC>::new(-68774.4991992591).year_month_day();
        assert_eq!(ymd.0, -4901);
        assert_eq!(ymd.1, 8);
        assert_eq!(ymd.2, 8);
    }

    #[test]
    fn test_time_near_leap_second() {
        for offset in -1000..1000 {
            let offset = f64::from(offset) / 10.0;
            // TIME IN TAI
            let mjd = 41683.0 + offset / 86400.0;
            let t = Time::<TAI>::from_mjd(mjd);
            let t = t.tdb();
            let t = t.tai();

            // Numerical precision of times near J2000 is only around 1e-10
            assert!((t.mjd() - mjd).abs() < 1e-9,);
        }

        // Perform round trip conversions in the seconds around a leap second.
        for offset in -1000..1000 {
            let offset = f64::from(offset) / 10.0;
            // TIME IN TAI
            let mjd = 41683.0 + offset / 86400.0;
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
            let offset = f64::from(offset) / 10.0;

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
    fn test_tcb_roundtrip() {
        // TCB/TDB round-trip: converting TDB -> TCB -> TDB should be the identity
        // to floating-point precision.
        for jd_offset in [-10000.0_f64, 0.0, 10000.0, 36525.0] {
            let tdb = Time::<TDB>::new(2_451_545.0 + jd_offset);
            let recovered = tdb.to_scale::<TCB>().tdb();
            let err = (recovered - tdb).elapsed.abs();
            assert!(err < 1e-15, "round-trip error at {tdb:?}: {err} days");
        }

        // At J2000.0 (jd = 2451545.0, ~23 years after T_0), TCB should be
        // ahead of TDB by roughly L_B * (J2000 - T_0) days.
        let jd_j2000 = 2_451_545.0_f64;
        let expected_drift_s = 1.550_519_768e-8_f64 * (jd_j2000 - 2_443_144.500_372_5) * 86400.0;
        let actual_drift_s = TCB::from_tdb_offset(jd_j2000) * 86400.0;
        assert!(
            (actual_drift_s - expected_drift_s).abs() < 1e-4,
            "drift mismatch: actual={actual_drift_s:.6} s expected={expected_drift_s:.6} s"
        );
    }

    /// A one nanosecond step survives addition and subtraction at a present-day
    /// epoch, where a single f64 Julian date resolves only about 40 microseconds.
    #[test]
    fn split_time_keeps_sub_microsecond_offsets() {
        let ns = 1e-9 / 86400.0;
        let t = Time::<TDB>::new(2_460_000.5);
        for k in 1..1000 {
            let offset = f64::from(k) * ns;
            let dt = ((t + offset) - t).elapsed;
            // The fraction of a day resolves about 1e-16 day, some 10 picoseconds.
            assert!((dt - offset).abs() < 1e-16, "{k} ns: {dt:e} vs {offset:e}");
        }
    }

    /// Whole days carry across the fraction, in both directions, and the parts
    /// recombine to the original Julian date.
    #[test]
    fn split_time_normalizes_across_days() {
        let t = Time::<TDB>::new(2_451_545.75);
        assert_eq!(t.jd_parts(), (2_451_545.0, 0.75));
        assert_eq!((t + 0.5).jd_parts(), (2_451_546.0, 0.25));
        assert_eq!((t - 1.0).jd_parts(), (2_451_544.0, 0.75));
        assert_eq!((t - 0.8).jd(), 2_451_544.95);
        let neg = Time::<TDB>::new(-3_800_000.25);
        assert_eq!(neg.jd_parts(), (-3_800_001.0, 0.75));
        assert_eq!(neg.jd(), -3_800_000.25);
        let parts = Time::<TDB>::from_parts(2_451_545.0, -1e-17);
        assert!(parts.jd_parts().1 < 1.0);
        assert!(Time::<TDB>::new(2_451_545.0) > parts || Time::<TDB>::new(2_451_545.0) == parts);
    }

    #[test]
    fn split_time_orders_and_handles_non_finite() {
        let a = Time::<TDB>::new(2_451_545.0);
        let b = a + 1e-12;
        assert!(a < b);
        assert!(b > a);
        let nan = Time::<TDB>::new(f64::NAN);
        assert!(nan.jd().is_nan());
        assert!((a + f64::NAN).jd().is_nan());
        assert!(nan.partial_cmp(&a).is_none());
        assert_ne!(nan, nan);
        assert_eq!((a + f64::INFINITY).jd(), f64::INFINITY);
    }

    /// Seconds past J2000 round trip, and the offset from a nearby epoch keeps the
    /// precision of the split time.
    #[test]
    fn j2000_seconds_offsets() {
        // A single f64 of seconds near 4.8e8 resolves only about 60 nanoseconds, so
        // the fine part is added to the split time rather than to the seconds.
        let t = Time::<TDB>::from_j2000_seconds(4.8e8) + 0.123_456_789 / 86400.0;
        assert!((t.j2000_seconds() - (4.8e8 + 0.123_456_789)).abs() < 1e-7);
        let offset = t.j2000_seconds_minus(4.8e8);
        assert!((offset - 0.123_456_789).abs() < 1e-10, "{offset}");
        assert_eq!(Time::<TDB>::from_j2000_seconds(0.0).jd(), 2_451_545.0);
        assert_eq!(Time::<TDB>::from_j2000_seconds(-86400.0).jd(), 2_451_544.0);
    }

    /// A number of days is added on the time's own scale, a `Duration` as elapsed TDB
    /// time, so across a leap second the two differ by that second.
    #[test]
    fn days_are_on_the_own_scale_and_durations_are_elapsed() {
        let before = Time::<UTC>::from_iso("2016-12-31T12:00:00+00:00").unwrap();
        let calendar = before + 1.0;
        assert_eq!(calendar.to_iso().unwrap(), "2017-01-01T12:00:00+00:00");
        // TDB and UTC seconds differ by the periodic TDB - TT term, tens of
        // microseconds over a day.
        assert!(((calendar - before).elapsed * 86400.0 - 86401.0).abs() < 1e-4);

        let elapsed = before + Duration::new(1.0);
        assert_eq!(elapsed.to_iso().unwrap(), "2017-01-01T11:59:59+00:00");
        assert!(((elapsed - before).elapsed - 1.0).abs() < 1e-15);

        // On TT a day is a TT day: the TT label moves by exactly one day.
        let tt = Time::<TT>::new(2_457_316.8);
        assert_eq!(((tt + 100.0).jd_parts().0 - tt.jd_parts().0), 100.0);
    }

    /// Scale conversions add their offsets to the split time, so converting there
    /// and back costs nothing measurable.
    #[test]
    fn scale_round_trips_keep_precision() {
        let t = Time::<TDB>::new(2_460_000.5) + 1.234e-9;
        for back in [
            t.utc().tdb(),
            t.tai().tdb(),
            t.tt().tdb(),
            t.to_scale::<TCB>().tdb(),
        ] {
            assert!(((back - t).elapsed * 86400.0).abs() < 1e-9);
        }
    }

    /// A UTC time read from ISO comes back as the same string after a round trip
    /// through TDB, including when it rounds up to the next day.
    #[test]
    fn iso_round_trips_through_tdb() {
        for s in [
            "2016-12-31T23:59:00+00:00",
            "2016-12-31T23:59:59.999+00:00",
            "2024-02-29T12:34:56.789+00:00",
        ] {
            let back = Time::<UTC>::from_iso(s)
                .unwrap()
                .tdb()
                .utc()
                .to_iso()
                .unwrap();
            assert_eq!(back, s);
        }
        let late = Time::<UTC>::from_iso("2016-12-31T23:59:59.9996+00:00").unwrap();
        assert_eq!(late.to_iso().unwrap(), "2017-01-01T00:00:00+00:00");
        assert!((late.year_as_float().unwrap() - 2017.0).abs() < 1e-10);
    }

    /// TAI - UTC steps at 00:00 UTC on a leap second date, not before.
    #[test]
    fn leap_second_takes_effect_at_utc_midnight() {
        for (iso, tai_m_utc) in [
            ("2016-12-31T23:59:00+00:00", 36.0),
            ("2016-12-31T23:59:59.5+00:00", 36.0),
            ("2017-01-01T00:00:00+00:00", 37.0),
            ("2017-01-01T00:00:30+00:00", 37.0),
        ] {
            let utc = Time::<UTC>::from_iso(iso).unwrap();
            let got = utc.tai().j2000_seconds_minus(utc.j2000_seconds());
            assert!((got - tai_m_utc).abs() < 1e-6, "{iso}: TAI - UTC = {got}");
        }
    }

    /// An MJD and the equivalent JD, each read from one f64, are different exact times
    /// but the same instant.
    #[test]
    fn same_instant_absorbs_f64_input_rounding() {
        let mjd = 60_000.123_456_789;
        let a = Time::<TDB>::from_mjd(mjd);
        let b = Time::<TDB>::new(mjd + 2_400_000.5);
        assert_ne!(a, b);
        assert!(a.same_instant(&b));
        assert!(!a.same_instant(&(a + 1e-6)));
    }

    #[test]
    fn test_iso() {
        let t = Time::<UTC>::from_iso("2000-01-01T06:00:00.000Z").unwrap();
        assert_eq!(t.year_month_day(), (2000, 1, 1, 0.25));

        let t1 = Time::<UTC>::from_iso("1987-12-25T00:00:00.000Z").unwrap();
        assert_eq!(t1.year_month_day(), (1987, 12, 25, 0.0));
        assert_eq!(t1.to_iso().unwrap(), "1987-12-25T00:00:00+00:00");
    }
}
