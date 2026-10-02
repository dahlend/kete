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

/// Parse a string of one to three numbers into a tuple.
///
/// The numbers are separated by runs of ` ,;:`. Leading spaces and tabs, and
/// trailing whitespace, are allowed. A separator at the start or the end is not.
/// A number holds only digits, `.`, `E`, `e`, `+` and `-`. Missing trailing
/// values are 0.0. The ranges are not checked here.
///
/// # Errors
/// [`Error::ValueError`] if the string does not match this format, or holds
/// more than three numbers.
fn parse_str_to_floats(text: &str) -> KeteResult<(f64, f64, f64)> {
    let err = || Error::ValueError(format!("Failed to parse string: {text}"));
    let is_separator = |c: char| " ,:;".contains(c);
    let body = text.trim_start_matches([' ', '\t']).trim_end();
    if body.is_empty() || body.starts_with(is_separator) || body.ends_with(is_separator) {
        return Err(err());
    }
    let values = body
        .split(is_separator)
        .filter(|token| !token.is_empty())
        .map(|token| {
            if token
                .chars()
                .all(|c| c.is_ascii_digit() || ".Ee+-".contains(c))
            {
                token.parse::<f64>().map_err(|_| err())
            } else {
                Err(err())
            }
        })
        .collect::<KeteResult<Vec<f64>>>()?;

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

/// Sample points and weights for numerical integration over `[0, 1]`.
///
/// To estimate the integral of a function `f` from 0 to 1, evaluate `f` at
/// each returned point `x` and add up `w * f(x)`. The function returns `count`
/// pairs `(x, w)`. The weights sum to 1.
///
/// The points are not evenly spaced; they bunch toward the ends of the
/// interval. This placement makes the estimate exact when `f` is a polynomial
/// of degree up to `2 * count - 1`, and very accurate for smooth functions with
/// only a few dozen points. The rule is Gauss-Legendre quadrature: the points
/// are the roots of the Legendre polynomial of degree `count`, moved from
/// `[-1, 1]` to `[0, 1]`.
#[must_use]
pub fn gauss_legendre(count: usize) -> Vec<(f64, f64)> {
    // P_count and its derivative at x, by the three-term recurrence.
    let legendre = |x: f64| {
        let (mut prev, mut value) = (1.0, x);
        for k in 2..=count {
            let next = ((2 * k - 1) as f64 * x * value - (k - 1) as f64 * prev) / k as f64;
            prev = value;
            value = next;
        }
        (value, count as f64 * (x * value - prev) / (x * x - 1.0))
    };
    (0..count)
        .map(|i| {
            // Newton on P_count from the usual cosine guess
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (count as f64 + 0.5)).cos();
            for _ in 0..100 {
                let (value, slope) = legendre(x);
                let step = value / slope;
                x -= step;
                if step.abs() < 1e-16 {
                    break;
                }
            }
            let (_, slope) = legendre(x);
            ((1.0 - x) / 2.0, 1.0 / ((1.0 - x * x) * slope * slope))
        })
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

    /// Separators sit only between numbers; a tab is not a separator.
    #[test]
    fn test_string_separators() {
        assert_eq!(
            parse_str_to_floats(" 12,34;56 ").unwrap(),
            (12.0, 34.0, 56.0)
        );
        assert_eq!(parse_str_to_floats("12 ,  34").unwrap(), (12.0, 34.0, 0.0));
        assert_eq!(parse_str_to_floats("\t12 30\n").unwrap(), (12.0, 30.0, 0.0));
        for bad in [",12,34", "12,34,", "12\t30", "1 2 3 4", "nan", "- 12", ""] {
            assert!(parse_str_to_floats(bad).is_err(), "{bad:?}");
        }
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

    #[test]
    fn test_gauss_legendre_exact() {
        // Exact for x^k, k < 2 count, whose integral on [0, 1] is 1 / (k + 1).
        for count in 1..=64 {
            let points = gauss_legendre(count);
            assert_eq!(points.len(), count);
            assert!(points.iter().all(|&(x, w)| x > 0.0 && x < 1.0 && w > 0.0));
            for k in 0..(2 * count) {
                let power = i32::try_from(k).unwrap();
                let integral: f64 = points.iter().map(|&(x, w)| w * x.powi(power)).sum();
                assert!(
                    (integral * (k as f64 + 1.0) - 1.0).abs() < 1e-12,
                    "count {count}, k {k}: {integral}"
                );
            }
        }
    }
}
