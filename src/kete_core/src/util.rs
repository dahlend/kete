//! Utility functions which cant be easily classified into a specific module.
//
// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
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

use crate::{
    errors::{Error, KeteResult},
    spice::sclk::parse_num,
};

/// Degree angle representation.
///
/// This exists primarily to provide conversion methods between
/// the different representations of angles used in astronomy, such as
/// degrees, hours, and radians.
///
/// This also provide text parsing and generation for the standard
/// representations of these values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Degrees {
    degrees: f64,
}

impl Degrees {
    /// Construct from radians.
    pub fn from_radians(radians: f64) -> Self {
        Self::from_degrees(radians.to_degrees())
    }

    /// Converts to radians.
    pub fn to_radians(&self) -> f64 {
        self.degrees.to_radians()
    }

    /// Construct from degrees.
    pub fn from_degrees(degrees: f64) -> Self {
        Self { degrees }
    }

    /// Converts to degrees.
    pub fn to_degrees(&self) -> f64 {
        self.degrees
    }

    /// Construct from hours.
    pub fn from_hours(hours: f64) -> Self {
        Self {
            degrees: hours * 15.0,
        }
    }

    /// Converts to Hours as a float.
    pub fn to_hours(&self) -> f64 {
        self.degrees / 15.0
    }

    /// New Degrees from degrees and minutes.
    /// The degrees value can be negative, but the minutes must be non-negative.
    pub fn try_from_degrees_minutes(mut degrees: f64, minutes: f64) -> KeteResult<Self> {
        if minutes.is_sign_negative() {
            return Err(Error::ValueError(format!(
                "Minutes value must be non-negative: {minutes}"
            )));
        }
        degrees += minutes.copysign(degrees) / 60.0;
        Ok(Self { degrees })
    }

    /// New Degrees from degrees and minutes.
    /// The degrees value can be negative, but the minutes must be non-negative.
    pub fn try_from_degrees_minutes_seconds(
        mut degrees: f64,
        minutes: u32,
        seconds: f64,
    ) -> KeteResult<Self> {
        if seconds.is_sign_negative() {
            return Err(Error::ValueError(format!(
                "Seconds value must be non-negative: {seconds}"
            )));
        }
        degrees += (minutes as f64).copysign(degrees) / 60.0;
        degrees += seconds.copysign(degrees) / 3600.0;
        Ok(Self { degrees })
    }

    /// Construct from hours, minutes, and seconds.
    pub fn try_from_hours_minutes_seconds(
        hours: f64,
        minutes: u32,
        seconds: f64,
    ) -> KeteResult<Self> {
        if seconds.is_sign_negative() {
            return Err(Error::ValueError(format!(
                "Seconds values must be non-negative: {seconds}"
            )));
        }
        let minutes = (minutes as f64).copysign(hours);
        let seconds = seconds.copysign(hours);
        let seconds = hours * 3600.0 + minutes * 60.0 + seconds;
        Ok(Self {
            degrees: seconds / 3600.0 * 15.0,
        })
    }

    /// Construct from hours and minutes.
    pub fn try_from_hours_minutes(hours: f64, minutes: f64) -> KeteResult<Self> {
        if minutes.is_sign_negative() {
            return Err(Error::ValueError(format!(
                "Minutes values must be non-negative: {minutes}"
            )));
        }
        let minutes = minutes.copysign(hours);
        let minutes = hours * 60.0 + minutes;
        Ok(Self {
            degrees: minutes / (60.0 * 15.0),
        })
    }

    /// Converts to Hours Minutes Seconds.
    ///
    /// This will be returned in the range [0, 24).
    ///
    /// Parameters
    /// ----------
    /// tol: Tolerance for rounding seconds to zero.
    pub fn to_hours_minutes_seconds(&self, tol: f64) -> (f64, u32, f64) {
        let deg = self.degrees.rem_euclid(360.0);
        let hours = deg / 15.0;
        let mut h = hours.trunc();
        let mut m = (hours - h) * 60.0;
        let mut s = (m - m.trunc()) * 60.0;

        if (s - 60.0).abs() < tol {
            s = 0.0;
            m += 1.0;
        }
        if m.trunc() == 60.0 {
            m = 0.0;
            h += 1.0;
        }
        if h == 24.0 {
            h = 0.0;
        }
        (h, m.trunc() as u32, s)
    }

    /// Converts to Degrees Minutes Seconds.
    ///
    /// Parameters
    /// ----------
    /// tol: Tolerance for rounding seconds to zero.
    pub fn to_degrees_minutes_seconds(&self, tol: f64) -> (f64, u32, f64) {
        let degrees = self.degrees.abs();
        let mut deg = degrees.trunc();
        let mut m = ((degrees - deg) * 60.0).trunc();
        let mut s = (degrees - deg) * 3600.0 - m * 60.0;

        if (s - 60.0).abs() < tol {
            s = 0.0;
            m += 1.0;
        }
        if m.trunc() == 60.0 {
            m = 0.0;
            deg += 1.0;
        }
        (deg.copysign(self.degrees), m as u32, s)
    }

    /// Construct from a string containing the hours minutes and seconds
    /// representation.
    ///
    /// The strings must contain at least one number, and up to three.
    /// Only the first number may have a signed value.
    ///
    /// This is a generous implementation, it allows multiple separator
    /// characters such as spaces, commas, colons, and semicolons. It will
    /// also allow all numbers to be floating point or scientific notation.
    pub fn try_from_hms_str(text: &str) -> KeteResult<Self> {
        let (h, m, s) = parse_str_to_floats(text)?;
        let hours = h + m.copysign(h) / 60.0 + s.copysign(h) / 3600.0;
        Ok(Self::from_hours(hours))
    }

    /// Construct from a string containing the degree minutes and seconds
    /// representation.
    ///
    /// The strings must contain at least one number, and up to three.
    /// Only the first number may have a signed value.
    ///
    /// This is a generous implementation, it allows multiple separator
    /// characters such as spaces, commas, colons, and semicolons. It will
    /// also allow all numbers to be floating point or scientific notation.
    pub fn try_from_dms_str(text: &str) -> KeteResult<Self> {
        let (h, m, s) = parse_str_to_floats(text)?;
        let degrees = h + m.copysign(h) / 60.0 + s.copysign(h) / 3600.0;
        Ok(Self::from_degrees(degrees))
    }

    /// Wraps the degrees in the range [0, 360).
    pub fn bound_to_360(&mut self) -> f64 {
        self.degrees = self.degrees.rem_euclid(360.0);
        self.degrees
    }

    /// Wraps the degrees in the range [-180, 180).
    pub fn bound_to_pm_180(&mut self) -> f64 {
        self.degrees = (self.degrees + 180.0).rem_euclid(360.0) - 180.0;
        self.degrees
    }

    /// Convert to a string in the format "+ddd mm ss.ss".
    pub fn to_dms_str(&self) -> String {
        let (d, m, s) = self.to_degrees_minutes_seconds(1e-4);
        format!("{d:+03} {m:02} {s:05.2}")
    }

    /// Convert to a string in the format "hh mm ss.sss".
    pub fn to_hms_str(&self) -> String {
        let (h, m, s) = self.to_hours_minutes_seconds(1e-5);
        format!("{h:02} {m:02} {s:06.3}")
    }
}

/// Parse a string of up to 3 numbers into a tuple.
///
/// The second and third numbers are optional, but must not have a signed value.
/// The string may use any number of these ` ,;:` as separators.
/// If less than three numbers are provided, the missing values will default to 0.0.
///
fn parse_str_to_floats(text: &str) -> KeteResult<(f64, f64, f64)> {
    let (rem, hms): (_, Vec<f64>) = delimited(
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

    let (h, m, s) = match hms.len() {
        1 => (hms[0], 0.0, 0.0),
        2 => (hms[0], hms[1], 0.0),
        3 => (hms[0], hms[1], hms[2]),
        _ => {
            return Err(Error::ValueError(format!(
                "String has too many numbers: {text}",
            )));
        }
    };

    if m.is_sign_negative() || s.is_sign_negative() {
        return Err(Error::ValueError(format!(
            "String has negative values in the second or third terms: {text}"
        )));
    }
    Ok((h, m, s))
}

/// Find the entire provided string in a collection of strings which may contain
/// partial matches.
///
/// Case insensitive search is performed.
///
/// Return all possible matches along with their indices.
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
    fn test_hms_deg_roundtrip() {
        for deg in 0..36000 {
            let deg = deg as f64 / 100.0;
            let hms = Degrees::from_degrees(deg);
            let converted_deg = hms.to_degrees();
            assert!(
                (converted_deg - deg).abs() < 1e-10,
                "Failed for deg: {deg} != {converted_deg}  {hms:?}",
            );
        }
    }

    #[test]
    fn test_dms_deg_roundtrip() {
        for deg in 0..36000 {
            let deg = deg as f64 / 100.0;
            let hms = Degrees::from_degrees(deg);
            let converted_deg = hms.to_degrees();
            assert!(
                (converted_deg - deg).abs() < 1e-10,
                "Failed for deg: {deg} != {converted_deg}",
            );
        }
    }

    #[test]
    fn test_hms_to_deg() {
        assert!(
            (Degrees::try_from_hours_minutes_seconds(2.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 30.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_hours_minutes_seconds(2.0 - 24.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                + 330.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_hours_minutes_seconds(0.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 0.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_hours_minutes_seconds(24.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 360.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_hours_minutes_seconds(1.0, 2, 9.6)
                .unwrap()
                .to_degrees()
                - 15.54)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_dms_to_deg() {
        assert!(
            (Degrees::try_from_degrees_minutes_seconds(2.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 2.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_degrees_minutes_seconds(2.0 - 360.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                + 358.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_degrees_minutes_seconds(0.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 0.0)
                .abs()
                < 1e-10
        );
        assert!(
            (Degrees::try_from_degrees_minutes_seconds(24.0, 0, 0.0)
                .unwrap()
                .to_degrees()
                - 24.0)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_deg_to_hms() {
        {
            let (hours, minutes, seconds) =
                Degrees::from_degrees(30.0).to_hours_minutes_seconds(1e-8);
            assert_eq!(hours, 2.0);
            assert_eq!(minutes, 0);
            assert!((seconds - 0.0).abs() < 1e-10);
        }

        {
            let (hours, minutes, seconds) =
                Degrees::from_degrees(360.0).to_hours_minutes_seconds(1e-8);
            assert_eq!(hours, 0.0);
            assert_eq!(minutes, 0);
            assert!((seconds - 0.0).abs() < 1e-10);
        }

        {
            let (hours, minutes, seconds) =
                Degrees::from_degrees(15.54).to_hours_minutes_seconds(1e-8);
            assert_eq!(hours, 1.0);
            assert_eq!(minutes, 2);
            assert!((seconds - 9.6).abs() < 1e-10, "{}", seconds);
        }
    }

    #[test]
    fn test_hms_from_str() {
        for hour in -24..24 {
            for minute in 0..6 {
                let minute = minute * 10;
                for second in 0..60 {
                    let second = second as f64 + 0.123;
                    let hms = Degrees::try_from_hours_minutes_seconds(hour as f64, minute, second)
                        .unwrap();
                    let hms_str = format!("{hour:02} {minute:02} {second:06.3}");
                    assert!(
                        (hms.to_degrees()
                            - Degrees::try_from_hms_str(&hms_str).unwrap().to_degrees())
                        .abs()
                            < 1e-8,
                        "Failed for {hour}:{minute}:{second}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_dms_from_str() {
        for degree in (-360..360).step_by(10) {
            for minute in 0..6 {
                let minute = minute * 10;
                for second in 0..60 {
                    let second = second as f64 + 0.123;
                    let dms: Degrees =
                        Degrees::try_from_degrees_minutes_seconds(degree as f64, minute, second)
                            .unwrap();
                    let dms_str = format!("{degree:02} {minute:02} {second:06.3}");
                    assert!(
                        (dms.to_degrees()
                            - Degrees::try_from_dms_str(&dms_str).unwrap().to_degrees())
                        .abs()
                            < 1e-8,
                        "Failed for {degree}:{minute}:{second}",
                    );
                }
            }
        }
    }
}
