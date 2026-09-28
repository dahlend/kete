//! Utility functions which can't be easily classified into a specific module.
//
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

use nom::{
    Parser, bytes::complete::take_while1, character::complete::space0, multi::separated_list1,
    sequence::delimited,
};

use std::str::FromStr;

use crate::errors::{Error, KeteResult};

/// Degree angle representation.
///
/// Provides conversion between the angle representations used in astronomy
/// (degrees, hours, radians, and sexagesimal), along with text parsing and
/// formatting of the sexagesimal forms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Degrees {
    degrees: f64,
}

impl Degrees {
    /// Construct from radians.
    #[must_use]
    pub fn from_radians(radians: f64) -> Self {
        Self::from_degrees(radians.to_degrees())
    }

    /// Converts to radians.
    #[must_use]
    pub fn to_radians(&self) -> f64 {
        self.degrees.to_radians()
    }

    /// Construct from degrees.
    #[must_use]
    pub fn from_degrees(degrees: f64) -> Self {
        Self { degrees }
    }

    /// Converts to degrees.
    #[must_use]
    pub fn to_degrees(&self) -> f64 {
        self.degrees
    }

    /// Construct from hours.
    #[must_use]
    pub fn from_hours(hours: f64) -> Self {
        Self {
            degrees: hours * 15.0,
        }
    }

    /// Converts to hours.
    #[must_use]
    pub fn to_hours(&self) -> f64 {
        self.degrees / 15.0
    }

    /// Construct from degrees, arcminutes, and arcseconds.
    ///
    /// The sign of `degrees` applies to the whole angle, including a negative
    /// zero, so `(-0.0, 30.0, 0.0)` is -0.5 degrees.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `degrees` is not finite, or if `minutes` or
    /// `seconds` is outside [0, 60).
    pub fn try_from_degrees_minutes_seconds(
        degrees: f64,
        minutes: f64,
        seconds: f64,
    ) -> KeteResult<Self> {
        Ok(Self::from_degrees(combine_sexagesimal(
            degrees, minutes, seconds,
        )?))
    }

    /// Construct from hours, minutes, and seconds.
    ///
    /// The sign of `hours` applies to the whole angle, including a negative zero.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `hours` is not finite, or if `minutes` or
    /// `seconds` is outside [0, 60).
    pub fn try_from_hours_minutes_seconds(
        hours: f64,
        minutes: f64,
        seconds: f64,
    ) -> KeteResult<Self> {
        Ok(Self::from_hours(combine_sexagesimal(
            hours, minutes, seconds,
        )?))
    }

    /// Construct from a string containing the hours, minutes, and seconds
    /// representation.
    ///
    /// The string must contain one to three numbers; missing trailing values
    /// are zero. Separators may be any run of spaces, commas, colons, and
    /// semicolons. Numbers may be decimal or scientific notation. Only the
    /// first number may carry a sign, which applies to the whole angle.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if parsing fails, or for the conditions listed in
    /// [`Self::try_from_hours_minutes_seconds`].
    pub fn try_from_hms_str(text: &str) -> KeteResult<Self> {
        let (h, m, s) = parse_str_to_floats(text)?;
        Self::try_from_hours_minutes_seconds(h, m, s)
    }

    /// Construct from a string containing the degrees, arcminutes, and
    /// arcseconds representation.
    ///
    /// Accepts the same formats as [`Self::try_from_hms_str`].
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if parsing fails, or for the conditions listed in
    /// [`Self::try_from_degrees_minutes_seconds`].
    pub fn try_from_dms_str(text: &str) -> KeteResult<Self> {
        let (d, m, s) = parse_str_to_floats(text)?;
        Self::try_from_degrees_minutes_seconds(d, m, s)
    }

    /// Wrap the angle into the range [0, 360).
    #[must_use]
    pub fn bound_to_360(self) -> Self {
        let degrees = self.degrees.rem_euclid(360.0);
        // rem_euclid rounds to exactly 360 for tiny negative inputs.
        Self {
            degrees: if degrees >= 360.0 { 0.0 } else { degrees },
        }
    }

    /// Wrap the angle into the range [-180, 180).
    #[must_use]
    pub fn bound_to_pm_180(self) -> Self {
        let degrees = (self.degrees + 180.0).rem_euclid(360.0);
        Self {
            degrees: if degrees >= 360.0 {
                -180.0
            } else {
                degrees - 180.0
            },
        }
    }

    /// Converts to hours, minutes, and seconds, with the angle wrapped into
    /// [0, 24) hours and seconds rounded to `decimals` places.
    ///
    /// Rounding carries into minutes and hours, so the seconds are always less
    /// than 60 and the hours less than 24.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the angle is not finite.
    pub fn to_hours_minutes_seconds(&self, decimals: u32) -> KeteResult<(u32, u32, f64)> {
        if !self.degrees.is_finite() {
            return Err(Error::ValueError(format!(
                "Cannot convert a non-finite angle to hours: {}",
                self.degrees
            )));
        }
        let scale = 10_u64.pow(decimals);
        let day = 24 * 3600 * scale;
        let hours = self.degrees.rem_euclid(360.0) / 15.0;
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "finite, non-negative, and below one day of units"
        )]
        let units = (hours * 3600.0 * scale as f64).round() as u64 % day;
        let (h, m, s) = split_sexagesimal(units, scale);
        Ok((h, m, s))
    }

    /// Converts to degrees, arcminutes, and arcseconds, with arcseconds rounded
    /// to `decimals` places.
    ///
    /// Rounding carries into minutes and degrees, so the seconds are always
    /// less than 60. The returned degrees carry the sign of the angle, as a
    /// negative zero for angles between -1 and 0 degrees. An angle that rounds
    /// to zero is returned as positive.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the angle is not finite.
    pub fn to_degrees_minutes_seconds(&self, decimals: u32) -> KeteResult<(f64, u32, f64)> {
        if !self.degrees.is_finite() {
            return Err(Error::ValueError(format!(
                "Cannot convert a non-finite angle to degrees-minutes-seconds: {}",
                self.degrees
            )));
        }
        let scale = 10_u64.pow(decimals);
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "finite and non-negative"
        )]
        let units = (self.degrees.abs() * 3600.0 * scale as f64).round() as u64;
        let (d, m, s) = split_sexagesimal(units, scale);
        let d = f64::from(d);
        let d = if units > 0 && self.degrees < 0.0 {
            -d
        } else {
            d
        };
        Ok((d, m, s))
    }

    /// Format as "+dd mm ss.ss".
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the angle is not finite.
    pub fn to_dms_str(&self) -> KeteResult<String> {
        let (d, m, s) = self.to_degrees_minutes_seconds(2)?;
        Ok(format!("{d:+03} {m:02} {s:05.2}"))
    }

    /// Format as "hh mm ss.sss", wrapped into [0, 24) hours.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the angle is not finite.
    pub fn to_hms_str(&self) -> KeteResult<String> {
        let (h, m, s) = self.to_hours_minutes_seconds(3)?;
        Ok(format!("{h:02} {m:02} {s:06.3}"))
    }
}

/// Combine a signed leading term with minutes and seconds into one value in the
/// units of the leading term. The sign of the leading term applies to all three.
fn combine_sexagesimal(first: f64, minutes: f64, seconds: f64) -> KeteResult<f64> {
    if !first.is_finite() {
        return Err(Error::ValueError(format!(
            "Leading value must be finite: {first}"
        )));
    }
    if !(0.0..60.0).contains(&minutes) || !(0.0..60.0).contains(&seconds) {
        return Err(Error::ValueError(format!(
            "Minutes and seconds must be in [0, 60): {minutes}, {seconds}"
        )));
    }
    Ok(first + minutes.copysign(first) / 60.0 + seconds.copysign(first) / 3600.0)
}

/// Split a non-negative count of `1 / scale` seconds into whole leading units,
/// whole minutes, and seconds.
fn split_sexagesimal(units: u64, scale: u64) -> (u32, u32, f64) {
    let seconds = (units % (60 * scale)) as f64 / scale as f64;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "minutes are below 60; the leading term of a finite angle fits u32"
    )]
    let minutes = ((units / (60 * scale)) % 60) as u32;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "minutes are below 60; the leading term of a finite angle fits u32"
    )]
    let leading = (units / (3600 * scale)) as u32;
    (leading, minutes, seconds)
}

/// Parse a number from a string, consuming digits and decimal/exponent characters.
fn parse_num<T: FromStr>(input: &str) -> nom::IResult<&str, T> {
    nom::combinator::map_res(
        take_while1(|c: char| c.is_ascii_digit() || ".Ee+-".contains(c)),
        |s: &str| s.parse::<T>(),
    )
    .parse(input)
}

/// Parse a string of one to three numbers into a tuple.
///
/// The string may use any run of ` ,;:` as separators. Missing trailing values
/// are 0.0. Ranges are not checked here.
fn parse_str_to_floats(text: &str) -> KeteResult<(f64, f64, f64)> {
    let (rem, values): (_, Vec<f64>) = delimited(
        space0,
        separated_list1(take_while1(|c| " ,:;".contains(c)), parse_num),
        space0,
    )
    .parse(text)
    .map_err(|_| Error::ValueError(format!("Failed to parse string: {text}")))?;

    if !rem.trim().is_empty() {
        return Err(Error::ValueError(format!(
            "Failed to parse: {text} parsing failed at {rem}",
        )));
    }

    match *values.as_slice() {
        [x] => Ok((x, 0.0, 0.0)),
        [x, y] => Ok((x, y, 0.0)),
        [x, y, z] => Ok((x, y, z)),
        _ => Err(Error::ValueError(format!(
            "String has too many numbers: {text}",
        ))),
    }
}

/// Find the entire provided string in a collection of strings which may contain
/// partial matches.
///
/// Case insensitive search is performed.
///
/// Return all possible matches along with their indices.
#[must_use]
pub fn partial_str_match<'a>(needle: &str, haystack: &'a [&'a str]) -> Vec<(usize, &'a str)> {
    let needle = needle.trim().to_lowercase();
    haystack
        .iter()
        .enumerate()
        .filter(|&(_, &hay)| hay.to_lowercase().contains(&needle))
        .map(|(i, &hay)| (i, hay))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hms_to_deg() {
        let cases = [
            ((2.0, 0.0, 0.0), 30.0),
            ((2.0 - 24.0, 0.0, 0.0), -330.0),
            ((0.0, 0.0, 0.0), 0.0),
            ((24.0, 0.0, 0.0), 360.0),
            ((1.0, 2.0, 9.6), 15.54),
            ((-0.0, 30.0, 0.0), -7.5),
        ];
        for ((h, m, s), expected) in cases {
            let deg = Degrees::try_from_hours_minutes_seconds(h, m, s)
                .unwrap()
                .to_degrees();
            assert!((deg - expected).abs() < 1e-10, "{h} {m} {s}: {deg}");
        }
    }

    #[test]
    fn test_dms_to_deg() {
        let cases = [
            ((2.0, 0.0, 0.0), 2.0),
            ((2.0 - 360.0, 0.0, 0.0), -358.0),
            ((0.0, 0.0, 0.0), 0.0),
            ((24.0, 0.0, 0.0), 24.0),
            ((-0.0, 30.0, 0.0), -0.5),
            ((-10.0, 30.0, 36.0), -10.51),
        ];
        for ((d, m, s), expected) in cases {
            let deg = Degrees::try_from_degrees_minutes_seconds(d, m, s)
                .unwrap()
                .to_degrees();
            assert!((deg - expected).abs() < 1e-10, "{d} {m} {s}: {deg}");
        }
    }

    #[test]
    fn test_sexagesimal_ranges_rejected() {
        for (m, s) in [(60.0, 0.0), (0.0, 60.0), (-1.0, 0.0), (0.0, -1.0)] {
            assert!(Degrees::try_from_degrees_minutes_seconds(1.0, m, s).is_err());
            assert!(Degrees::try_from_hours_minutes_seconds(1.0, m, s).is_err());
        }
        assert!(Degrees::try_from_degrees_minutes_seconds(f64::NAN, 0.0, 0.0).is_err());
        assert!(Degrees::try_from_dms_str("10 75 00").is_err());
        assert!(Degrees::try_from_hms_str("10 -5 00").is_err());
        assert!(Degrees::try_from_hms_str("0 1 2 3").is_err());
        assert!(Degrees::try_from_hms_str("12h30m").is_err());
    }

    #[test]
    fn test_deg_to_hms() {
        let cases = [
            (30.0, (2, 0, 0.0)),
            (360.0, (0, 0, 0.0)),
            (15.54, (1, 2, 9.6)),
            (-15.0, (23, 0, 0.0)),
        ];
        for (deg, expected) in cases {
            let (h, m, s) = Degrees::from_degrees(deg)
                .to_hours_minutes_seconds(3)
                .unwrap();
            assert_eq!((h, m), (expected.0, expected.1), "{deg}");
            assert!((s - expected.2).abs() < 1e-10, "{deg}: {s}");
        }
    }

    #[test]
    fn test_strings_carry_rounded_seconds() {
        // Seconds that round up at the printed precision carry into the minutes.
        let hms = |h: f64, m: f64, s: f64| {
            Degrees::try_from_hours_minutes_seconds(h, m, s)
                .unwrap()
                .to_hms_str()
                .unwrap()
        };
        assert_eq!(hms(1.0, 2.0, 59.9996), "01 03 00.000");
        assert_eq!(hms(1.0, 59.0, 59.9996), "02 00 00.000");
        assert_eq!(hms(23.0, 59.0, 59.9996), "00 00 00.000");
        assert_eq!(hms(1.0, 2.0, 59.9994), "01 02 59.999");

        let dms = |d: f64, m: f64, s: f64| {
            Degrees::try_from_degrees_minutes_seconds(d, m, s)
                .unwrap()
                .to_dms_str()
                .unwrap()
        };
        assert_eq!(dms(10.0, 20.0, 59.996), "+10 21 00.00");
        assert_eq!(dms(-10.0, 59.0, 59.996), "-11 00 00.00");
        assert_eq!(dms(-0.0, 30.0, 0.0), "-00 30 00.00");
        assert_eq!(dms(-0.0, 0.0, 0.001), "+00 00 00.00");
        assert_eq!(dms(90.0, 0.0, 0.0), "+90 00 00.00");
    }

    #[test]
    fn test_non_finite_strings_rejected() {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(Degrees::from_degrees(x).to_hms_str().is_err());
            assert!(Degrees::from_degrees(x).to_dms_str().is_err());
        }
    }

    #[test]
    fn test_bounds() {
        assert_eq!(
            Degrees::from_degrees(-1e-20).bound_to_360().to_degrees(),
            0.0
        );
        assert_eq!(
            Degrees::from_degrees(-90.0).bound_to_360().to_degrees(),
            270.0
        );
        assert_eq!(
            Degrees::from_degrees(180.0).bound_to_pm_180().to_degrees(),
            -180.0
        );
        assert_eq!(
            Degrees::from_degrees(270.0).bound_to_pm_180().to_degrees(),
            -90.0
        );
        assert!(
            Degrees::from_degrees(-180.0 - 1e-14)
                .bound_to_pm_180()
                .to_degrees()
                < 180.0
        );
    }

    #[test]
    fn test_hms_str_roundtrip() {
        for hour in -24..24 {
            for minute in (0..60).step_by(10) {
                for second in 0..60 {
                    let minute = f64::from(minute);
                    let second = f64::from(second) + 0.123;
                    let hms =
                        Degrees::try_from_hours_minutes_seconds(f64::from(hour), minute, second)
                            .unwrap();
                    let hms_str = format!("{hour:02} {minute:02} {second:06.3}");
                    let parsed = Degrees::try_from_hms_str(&hms_str).unwrap();
                    assert!(
                        (hms.to_degrees() - parsed.to_degrees()).abs() < 1e-8,
                        "Failed for {hms_str}",
                    );
                    let formatted = Degrees::try_from_hms_str(&hms.to_hms_str().unwrap()).unwrap();
                    let diff = (formatted.to_degrees() - hms.bound_to_360().to_degrees()).abs();
                    assert!(diff < 1e-8, "Failed for {hms_str}");
                }
            }
        }
    }

    #[test]
    fn test_dms_str_roundtrip() {
        for degree in (-90..=90).step_by(5) {
            for minute in (0..60).step_by(10) {
                for second in 0..60 {
                    let minute = f64::from(minute);
                    let second = f64::from(second) + 0.12;
                    let dms = Degrees::try_from_degrees_minutes_seconds(
                        f64::from(degree),
                        minute,
                        second,
                    )
                    .unwrap();
                    let dms_str = format!("{degree:02} {minute:02} {second:05.2}");
                    let parsed = Degrees::try_from_dms_str(&dms_str).unwrap();
                    assert!(
                        (dms.to_degrees() - parsed.to_degrees()).abs() < 1e-8,
                        "Failed for {dms_str}",
                    );
                    let formatted = Degrees::try_from_dms_str(&dms.to_dms_str().unwrap()).unwrap();
                    assert!(
                        (formatted.to_degrees() - dms.to_degrees()).abs() < 1e-8,
                        "Failed for {dms_str}",
                    );
                }
            }
        }
    }
}
