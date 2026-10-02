//! Spacecraft clocks (SCLK) from the variables of SPICE text kernels.
//!
//! The `SCLK_*` and `SCLK01_*` variables define a clock. An SCLK kernel or
//! another text kernel, such as a frames kernel, can hold them. Only type 1
//! clocks are supported.
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
use std::collections::HashMap;

use super::TextKernelVars;
use kete_core::{
    errors::{Error, KeteResult},
    time::{TDB, TT, Time},
};

/// NAIF ID of a spacecraft clock, such as -226.
///
/// The variables of a clock carry the absolute value of the ID in their names,
/// such as `SCLK01_COEFFICIENTS_226`. A CK frame names its clock by
/// `CK_<id>_SCLK` (see [`crate::text::fk::ck_clock_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockId(pub i32);

impl std::fmt::Display for ClockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A type 1 spacecraft clock, from the variables of the loaded text kernels.
///
/// [`crate::text::TextKernels::clock`] gives the clock of a spacecraft.
#[derive(Debug, Clone, PartialEq)]
pub struct Sclk {
    /// ID of the clock.
    id: ClockId,

    n_fields: u32,
    offsets: Vec<u64>,

    partition_start: Vec<f64>,
    partition_end: Vec<f64>,

    coefficients: Vec<[f64; 3]>,

    /// Rate at which each field of the clock ticks. The lowest value field
    /// ticks at 1 unit per tick.
    tick_rates: Vec<usize>,

    /// Whether the parallel time system of the clock is TDB, rather than TT.
    ///
    /// `SCLK01_TIME_SYSTEM_nn` is 1 for TDB and 2 for TT. A missing keyword
    /// means TDB. The coefficients map ticks to the parallel time system that
    /// the kernel declares. Thus a TT clock needs the conversion between TDB
    /// and TT.
    parallel_is_tdb: bool,
}

impl Sclk {
    /// ID of the clock.
    #[must_use]
    pub fn id(&self) -> ClockId {
        self.id
    }

    /// Convert a spacecraft clock string, such as `"1/0235850967.63768"`, into
    /// a time.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the string does not parse as fields of this
    /// clock, or its partition is not valid for its count.
    pub fn string_to_time(&self, time_str: &str) -> KeteResult<Time<TDB>> {
        let (_, tick) = self.string_to_tick(time_str)?;
        self.tick_to_time(tick)
    }

    /// Convert a spacecraft clock tick (SCLK time) into a [`Time<TDB>`].
    ///
    /// The parallel time is seconds past J2000 on the clock's time system, so
    /// it is read on that scale and converted to TDB afterward.
    ///
    /// # Errors
    /// [`Error::Bounds`] if `tick` is outside the partitions of the clock (see
    /// [`Self::tick_range`]).
    pub fn tick_to_time(&self, tick: f64) -> KeteResult<Time<TDB>> {
        self.check_tick(tick)?;
        let clock_rate = self.find_tick_rate(tick);

        let par_time =
            (tick - clock_rate[0]) * (clock_rate[2] / (self.tick_rates[0] as f64)) + clock_rate[1];
        Ok(if self.parallel_is_tdb {
            Time::<TDB>::from_j2000_seconds(par_time)
        } else {
            Time::<TT>::from_j2000_seconds(par_time).tdb()
        })
    }

    /// Convert time in TDB to a spacecraft clock tick count.
    ///
    /// The offset from the rate's reference time is taken from the split time,
    /// so the tick keeps the precision of `time` up to the resolution of an f64
    /// tick count.
    ///
    /// # Errors
    /// [`Error::Bounds`] if `time` is outside the partitions of the clock (see
    /// [`Self::tick_range`]).
    pub fn time_to_tick(&self, time: Time<TDB>) -> KeteResult<f64> {
        let (clock_rate, offset) = if self.parallel_is_tdb {
            let clock_rate = self.find_parallel_time_rate(time.j2000_seconds());
            (clock_rate, time.j2000_seconds_minus(clock_rate[1]))
        } else {
            let tt = time.tt();
            let clock_rate = self.find_parallel_time_rate(tt.j2000_seconds());
            (clock_rate, tt.j2000_seconds_minus(clock_rate[1]))
        };

        let tick = offset * ((self.tick_rates[0] as f64) / clock_rate[2]) + clock_rate[0];
        self.check_tick(tick)?;
        Ok(tick)
    }

    /// The range of tick counts the clock covers, from 0 to the total length
    /// of its partitions, both inclusive.
    #[must_use]
    pub fn tick_range(&self) -> (f64, f64) {
        let total = self
            .partition_start
            .iter()
            .zip(&self.partition_end)
            .map(|(start, end)| end - start)
            .sum();
        (0.0, total)
    }

    /// Check that `tick` is in [`Self::tick_range`].
    ///
    /// # Errors
    /// [`Error::Bounds`] if it is not.
    fn check_tick(&self, tick: f64) -> KeteResult<()> {
        let (first, last) = self.tick_range();
        if (first..=last).contains(&tick) {
            Ok(())
        } else {
            Err(Error::Bounds(format!(
                "Tick count {tick} is outside SCLK clock {}, which covers ticks {first} \
                 to {last}.",
                self.id
            )))
        }
    }

    /// Convert a spacecraft clock string into the partition and tick count.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] in these cases:
    /// - The string does not parse as clock fields.
    /// - The number of fields is zero or more than the clock has.
    /// - A field is below its offset.
    /// - The partition is not valid for the tick count.
    fn string_to_tick(&self, time_str: &str) -> KeteResult<(usize, f64)> {
        let (partition, mut fields) = parse_time_fields(time_str)?;

        if fields.len() > self.n_fields as usize || fields.is_empty() {
            return Err(Error::ValueError(format!(
                "Fields in time string must be between 1 and {}, found {}.",
                self.n_fields,
                fields.len()
            )));
        }

        // Each field counts from its offset. A field value is not required to
        // be below its modulus.
        for (field, &offset) in fields.iter_mut().zip(self.offsets.iter()) {
            *field = field.checked_sub(offset as usize).ok_or_else(|| {
                Error::ValueError(format!(
                    "Clock field value {field} is below its offset {offset}."
                ))
            })?;
        }

        // compute a floating point representation of the spacecraft clock time
        let mut tick: f64 = 0.0;
        fields
            .iter()
            .zip(self.tick_rates.iter())
            .for_each(|(field, rate)| {
                tick += (field * rate) as f64;
            });

        let (partition, partition_count) = self.partition_tick_count(tick, partition)?;
        tick += partition_count;
        Ok((partition, tick))
    }

    /// The coefficient row (tick, parallel time, rate) that holds `tick`: the
    /// last one starting at or before it.
    fn find_tick_rate(&self, tick: f64) -> [f64; 3] {
        let mut idx = self.coefficients.partition_point(|probe| probe[0] <= tick);
        idx = idx.saturating_sub(1);
        self.coefficients[idx]
    }

    /// The coefficient row (tick, parallel time, rate) that holds the parallel
    /// time `par_time`: the last one starting at or before it.
    fn find_parallel_time_rate(&self, par_time: f64) -> [f64; 3] {
        let mut idx = self
            .coefficients
            .partition_point(|probe| probe[1] <= par_time);
        idx = idx.saturating_sub(1);
        self.coefficients[idx]
    }

    /// Find the partition of a clock count, and the offset from count to ticks.
    ///
    /// `count` is the clock count within its partition. `partition` is the
    /// 1-based partition from the clock string, if the string gives one. The
    /// function returns the 1-based partition and an offset. The sum of `count`
    /// and the offset is the number of ticks from the start of the clock. The
    /// offset is the total length of the earlier partitions, minus the start of
    /// this partition.
    ///
    /// A given partition must contain the count. If no partition is given, the
    /// function uses the first partition that contains the count. Both
    /// partition bounds are inclusive.
    ///
    /// # Errors
    /// Returns [`Error::ValueError`] if `partition` is out of range or does not
    /// contain the count. Also returns [`Error::ValueError`] if `partition` is
    /// `None` and no partition contains the count.
    fn partition_tick_count(
        &self,
        count: f64,
        partition: Option<usize>,
    ) -> KeteResult<(usize, f64)> {
        let contains =
            |idx: usize| self.partition_start[idx] <= count && count <= self.partition_end[idx];
        let idx = match partition {
            Some(p) if (1..=self.partition_start.len()).contains(&p) && contains(p - 1) => p - 1,
            Some(p) => {
                return Err(Error::ValueError(format!(
                    "Clock count {count} is not in partition {p}."
                )));
            }
            None => (0..self.partition_start.len())
                .find(|&idx| contains(idx))
                .ok_or_else(|| {
                    Error::ValueError(format!("Clock count {count} is not in any partition."))
                })?,
        };
        let earlier: f64 = (0..idx)
            .map(|i| self.partition_end[i] - self.partition_start[i])
            .sum();
        Ok((idx + 1, earlier - self.partition_start[idx]))
    }
}

/// Parse a spacecraft clock string into a partition and field values.
///
/// The fields are unsigned integers. A delimiter is one dash, colon, comma or
/// period, with optional blanks around it, or a run of blanks alone. A blank is
/// a space or a tab. Two delimiters in a row, such as "::", mark an empty
/// field, which this parser rejects. An optional partition number and a slash
/// can come before the fields. The partition is `None` if the string does not
/// give one. Leading and trailing blanks are allowed.
///
/// # Errors
/// [`Error::ValueError`] if the string does not match this format, or if a
/// number does not fit in `usize`.
fn parse_time_fields(input: &str) -> KeteResult<(Option<usize>, Vec<usize>)> {
    let err = || Error::ValueError(format!("Failed to parse time fields of {input:?}."));
    let blank = [' ', '\t'];
    let integer = |s: &str| -> KeteResult<usize> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(err());
        }
        s.parse().map_err(|_| err())
    };
    let text = input.trim_matches(blank);
    let (partition, mut rest) = match text.split_once('/') {
        Some((partition, rest)) => (
            Some(integer(partition.trim_matches(blank))?),
            rest.trim_start_matches(blank),
        ),
        None => (None, text),
    };
    let mut fields = Vec::new();
    loop {
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        fields.push(integer(&rest[..end])?);
        rest = &rest[end..];
        if rest.is_empty() {
            return Ok((partition, fields));
        }
        let after_blanks = rest.trim_start_matches(blank);
        rest = match after_blanks.strip_prefix(['-', ':', ',', '.']) {
            Some(after) => after.trim_start_matches(blank),
            // A run of blanks alone is a delimiter.
            None if after_blanks.len() < rest.len() => after_blanks,
            None => return Err(err()),
        };
    }
}

/// The number of the clock `id` in its kernel variable names, such as 226 in
/// `SCLK01_COEFFICIENTS_226` for the clock -226.
fn variable_suffix(id: ClockId) -> i64 {
    -i64::from(id.0)
}

/// The non-negative integers of the variable `name`.
///
/// # Errors
/// [`Error::ValueError`] if the variable is missing, does not hold numbers, or
/// holds a number that is not a non-negative integer below 2^63.
fn unsigned(vars: &TextKernelVars, name: &str) -> KeteResult<Vec<u64>> {
    required(vars.numbers(name)?, name)?
        .iter()
        .map(|&x| {
            if x >= 0.0 && x.fract() == 0.0 && x < 2_f64.powi(63) {
                // The check above keeps the value integral, non-negative and
                // below 2^63, so the cast is exact.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "checked integral, non-negative and in range"
                )]
                Ok(x as u64)
            } else {
                Err(Error::ValueError(format!(
                    "{name} must hold non-negative integers, found {x}."
                )))
            }
        })
        .collect()
}

/// `value`, or an error that names the missing variable `name`.
///
/// # Errors
/// [`Error::ValueError`] if `value` is `None`.
fn required<T>(value: Option<T>, name: &str) -> KeteResult<T> {
    value.ok_or_else(|| Error::ValueError(format!("SCLK variable {name} is missing.")))
}

impl Sclk {
    /// The clock with NAIF ID `naif_id` from the variables of the loaded text
    /// kernels.
    ///
    /// # Errors
    /// [`Error::ValueError`] if a variable of the clock is missing or has the
    /// wrong type or size, if the clock type is not 1, if a modulus is 0, if a
    /// partition ends before it starts, or if the variables are inconsistent
    /// with each other.
    fn from_vars(vars: &TextKernelVars, id: ClockId) -> KeteResult<Self> {
        let n = variable_suffix(id);
        let data_type = format!("SCLK_DATA_TYPE_{n}");
        let dtype = required(vars.integer(&data_type)?, &data_type)?;
        if dtype != 1 {
            return Err(Error::ValueError(format!(
                "SCLK clock type must be 1, found {dtype}."
            )));
        }
        let name = format!("SCLK01_N_FIELDS_{n}");
        let n_fields = required(vars.integer(&name)?, &name)?;
        if n_fields < 1 {
            return Err(Error::ValueError(format!(
                "SCLK N_FIELDS must be at least 1, found {n_fields}."
            )));
        }
        // The check above keeps the count positive, so the cast is exact.
        #[allow(clippy::cast_sign_loss, reason = "checked positive")]
        let n_fields = n_fields as u32;
        let moduli = unsigned(vars, &format!("SCLK01_MODULI_{n}"))?;
        let offsets = unsigned(vars, &format!("SCLK01_OFFSETS_{n}"))?;
        let name = format!("SCLK01_OUTPUT_DELIM_{n}");
        let _ = required(vars.integer(&name)?, &name)?;
        let name = format!("SCLK_PARTITION_START_{n}");
        let partition_start = required(vars.numbers(&name)?, &name)?.to_vec();
        let name = format!("SCLK_PARTITION_END_{n}");
        let partition_end = required(vars.numbers(&name)?, &name)?.to_vec();
        let name = format!("SCLK01_COEFFICIENTS_{n}");
        let coefficients = required(vars.numbers(&name)?, &name)?;
        if coefficients.is_empty() || coefficients.len() % 3 != 0 {
            return Err(Error::ValueError(
                "SCLK Coefficients must be a non-empty list of triplets.".into(),
            ));
        }
        let coefficients = coefficients.as_chunks::<3>().0.to_vec();
        // The time system is optional. A missing value means TDB.
        let parallel_is_tdb = match vars.integer(&format!("SCLK01_TIME_SYSTEM_{n}"))? {
            None | Some(1) => true,
            Some(2) => false,
            Some(val) => {
                return Err(Error::ValueError(format!(
                    "SCLK Time System must be 1 (TDB) or 2 (TT), found {val}."
                )));
            }
        };

        if partition_start.is_empty() || partition_start.len() != partition_end.len() {
            return Err(Error::ValueError(format!(
                "SCLK PARTITION_START length ({}) does not match PARTITION_END ({})",
                partition_start.len(),
                partition_end.len()
            )));
        }
        if offsets.len() != n_fields as usize {
            return Err(Error::ValueError(format!(
                "SCLK OFFSETS length ({}) does not match N_FIELDS ({})",
                offsets.len(),
                n_fields
            )));
        }
        if moduli.len() != n_fields as usize {
            return Err(Error::ValueError(format!(
                "SCLK MODULI length ({:?}) does not match N_FIELDS ({})",
                moduli.len(),
                n_fields
            )));
        }

        if let Some(m) = moduli.iter().find(|&&m| m == 0) {
            return Err(Error::ValueError(format!(
                "SCLK MODULI must be at least 1, found {m}."
            )));
        }
        if let Some((start, end)) = partition_start
            .iter()
            .zip(&partition_end)
            .find(|(start, end)| !(start.is_finite() && end.is_finite() && start <= end))
        {
            return Err(Error::ValueError(format!(
                "SCLK partition ends must not be before their starts, found start {start} \
                 and end {end}."
            )));
        }

        // Each field has a tick rate: the number of ticks of the lowest field in
        // one count of the field.
        let mut tick_rates = vec![1_usize];
        for modulo in moduli.iter().skip(1).rev() {
            let last = tick_rates[tick_rates.len() - 1];
            let rate = usize::try_from(*modulo)
                .ok()
                .and_then(|m| m.checked_mul(last))
                .ok_or_else(|| {
                    Error::ValueError("SCLK MODULI give more ticks than fit in usize.".into())
                })?;
            tick_rates.push(rate);
        }
        tick_rates.reverse();

        Ok(Self {
            id,
            n_fields,
            offsets,
            partition_start,
            partition_end,
            coefficients,
            tick_rates,
            parallel_is_tdb,
        })
    }
}

/// Every clock the variables define, one for each `SCLK_DATA_TYPE_<n>`
/// variable whose `<n>` is an integer.
///
/// A clock that is missing a variable or is inconsistent keeps its error. The
/// error comes when the clock is used.
pub(crate) fn clocks_from(vars: &TextKernelVars) -> HashMap<ClockId, KeteResult<Sclk>> {
    vars.names()
        .filter_map(|name| {
            name.strip_prefix("SCLK_DATA_TYPE_")?
                .parse::<i32>()
                .ok()?
                .checked_neg()
        })
        .map(|id| {
            let id = ClockId(id);
            let clock = Sclk::from_vars(vars, id)
                .map_err(|e| Error::ValueError(format!("SCLK clock {id}: {e}")));
            (id, clock)
        })
        .collect()
}

#[cfg(test)]
mod tests {

    use super::*;

    /// The clock `id` of an SCLK kernel's text.
    fn clock_from(text: &str, id: i32) -> Sclk {
        let mut vars = TextKernelVars::default();
        vars.load_text(text).unwrap();
        clocks_from(&vars).remove(&ClockId(id)).unwrap().unwrap()
    }

    #[test]
    fn test_parse_time_field() {
        let input = "  1:2   3 - 5:8 . 9 9 ";
        let result = parse_time_fields(input).unwrap();
        assert_eq!(result, (None, vec![1, 2, 3, 5, 8, 9, 9]));

        let input = " 5 /  1:2   3 - 5:8 . 9 9 ";
        let result = parse_time_fields(input).unwrap();
        assert_eq!(result, (Some(5), vec![1, 2, 3, 5, 8, 9, 9]));

        // A Rosetta clock string, with a period between the fields.
        let result = parse_time_fields("1/0355251413.46270").unwrap();
        assert_eq!(result, (Some(1), vec![355_251_413, 46270]));
        let result = parse_time_fields("12-345").unwrap();
        assert_eq!(result, (None, vec![12, 345]));

        assert!(parse_time_fields("1/12:34x").is_err());

        // Two delimiters in a row mark an empty field, which is rejected.
        assert!(parse_time_fields("1/12::3").is_err());
        assert!(parse_time_fields("1/12:.3").is_err());
        assert!(parse_time_fields("1/12 : - 3").is_err());
    }

    #[test]
    fn test_data_block() {
        let input = r"
            KPL/SCLK
            Test Comments here.
            Text parsing is never a good time.
            This kernel is a copy of the Galileo time kernel.

            \begindata
            SCLK_KERNEL_ID            = ( @04-SEP-1990//4:23:00 )
            
            SCLK_DATA_TYPE_77         = ( 1                )
            SCLK01_N_FIELDS_77        = ( 4                )
            SCLK01_MODULI_77          = ( 16777215 91 10 8 )
            SCLK01_OFFSETS_77         = (        0  0  0 0 )
            SCLK01_OUTPUT_DELIM_77    = ( 2                )
            
            SCLK_PARTITION_START_77   = ( 0.0000000000000E+00
                                            2.5465440000000E+07
                                            7.2800001000000E+07
                                            1.3176800000000E+08 )
            
            SCLK_PARTITION_END_77      = ( 2.5465440000000E+07
                                            7.2800000000000E+07
                                            1.3176800000000E+08
                                            1.2213812519900E+11 )
            
            SCLK01_COEFFICIENTS_77    = (
            
            0.0000000000000E+00  -3.2287591517365E+08  6.0666283888000E+01
            7.2800000000000E+05  -3.2286984854565E+08  6.0666283888000E+01
            1.2365520000000E+06  -3.2286561063865E+08  6.0666283888000E+01
            1.2365600000000E+06  -3.2286558910065E+08  6.0697000438000E+01
            1.2368000000000E+06  -3.2286557090665E+08  6.0666283333000E+01
            1.2962400000000E+06  -3.2286507557565E+08  6.0666283333000E+01
            2.3296480000000E+07  -3.2286507491065E+08  6.0666300000000E+01
            2.3519280000000E+07  -3.2286321825465E+08  5.8238483608000E+02
            2.3519760000000E+07  -3.2286317985565E+08  6.0666272281000E+01
            2.4024000000000E+07  -3.2285897788265E+08  6.0666271175000E+01
            2.5378080000000E+07  -3.2284769395665E+08  6.0808150200000E+01
            2.5421760000000E+07  -3.2284732910765E+08  6.0666628073000E+01
            2.5465440000000E+07  -3.2284696510765E+08  6.0666628073000E+01
            3.6400000000000E+07  -3.2275584383265E+08  6.0666627957000E+01
            7.2800000000000E+07  -3.2245251069264E+08  6.0666628004000E+01
            1.0919999900000E+08  -3.2214917755262E+08  6.0666628004000E+01
            1.2769119900000E+08  -3.2199508431761E+08  6.0665620197000E+01
            1.3085799900000E+08  -3.2196869477261E+08  6.0666892494000E+01
            1.3176799900000E+08  -3.2196111141061E+08  6.0666722113000E+01
            1.3395199900000E+08  -3.2194291139361E+08  6.0666674091000E+01
            1.3613599900000E+08  -3.2192471139161E+08  6.0666590261000E+01
            1.4341599900000E+08  -3.2186404480160E+08  6.0666611658000E+01
            1.5069599900000E+08  -3.2180337818960E+08  6.0666611658000E+01
            1.7253599900000E+08  -3.2162137835458E+08  6.0666783566000E+01
            1.7515679900000E+08  -3.2159953831258E+08  6.0666629213000E+01
            1.7777759900000E+08  -3.2157769832557E+08  6.0666629213000E+01
            3.3451599900000E+08  -3.2027154579839E+08  6.0666505193000E+01
            3.3713679900000E+08  -3.2024970585638E+08  6.0666627480000E+01
            3.3975759900000E+08  -3.2022786587038E+08  6.0666627480000E+01
            5.6601999900000E+08  -3.1834234708794E+08  6.0666396876000E+01
            5.6733039900000E+08  -3.1833142713693E+08  6.0666626282000E+01
            5.6864079900000E+08  -3.1832050714393E+08  6.0666626282000E+01
            8.9797999900000E+08  -3.1557601563707E+08  5.9666626282000E+01
            8.9798727900000E+08  -3.1557595597007E+08  6.0666626282000E+01
            8.9799455900000E+08  -3.1557589430307E+08  6.0666626282000E+01 )
            
            \begintext";

        let clock = clock_from(input, -77);
        assert_eq!(clock.n_fields, 4);
        assert_eq!(clock.tick_rates, vec![7280, 80, 8, 1]);
        assert_eq!(
            clock.partition_start,
            vec![0.0, 2.546_544E+07, 7.280_000_1E+07, 1.31768E+08]
        );

        let t = clock.string_to_time("1/1000:00:00").unwrap();

        let ticks = clock.time_to_tick(t).unwrap();
        let t2 = clock.tick_to_time(ticks).unwrap();
        assert_eq!(t, t2);

        let (part, count) = clock.partition_tick_count(0.0, None).unwrap();
        assert_eq!(part, 1);
        assert_eq!(count, 0.0);

        let (part, count) = clock.partition_tick_count(7.290_000_3E+07, None).unwrap();
        assert_eq!(part, 3);
        assert_eq!(count, -1.0);

        // A count on a shared partition bound belongs to the first partition,
        // unless the other is given explicitly.
        let (part, _) = clock.partition_tick_count(2.546_544E+07, None).unwrap();
        assert_eq!(part, 1);
        let (part, _) = clock.partition_tick_count(2.546_544E+07, Some(2)).unwrap();
        assert_eq!(part, 2);
        assert!(clock.partition_tick_count(1.0E+08, Some(1)).is_err());
        assert!(clock.partition_tick_count(1.0E+12, None).is_err());
    }

    /// Field values count from their offsets.
    #[test]
    fn string_to_tick_subtracts_offsets() {
        let input = r"
            KPL/SCLK
            \begindata
            SCLK_KERNEL_ID            = ( @2000-JAN-01 )
            SCLK_DATA_TYPE_88         = ( 1 )
            SCLK01_N_FIELDS_88        = ( 2 )
            SCLK01_MODULI_88          = ( 100000 256 )
            SCLK01_OFFSETS_88         = ( 0 1 )
            SCLK01_OUTPUT_DELIM_88    = ( 1 )
            SCLK_PARTITION_START_88   = ( 0.0 )
            SCLK_PARTITION_END_88     = ( 2.56E+07 )
            SCLK01_COEFFICIENTS_88    = ( 0.0 0.0 1.0 )
            \begintext";
        let clock = clock_from(input, -88);

        assert_eq!(clock.string_to_tick("1/100:1").unwrap(), (1, 25600.0));
        assert_eq!(clock.string_to_tick("1/100:256").unwrap(), (1, 25855.0));
        assert!(clock.string_to_tick("1/100:0").is_err());
    }

    /// On a TT clock, a tick far from the rate's reference epoch converts to
    /// the time it came from. The TDB - TT term differs between the two epochs
    /// by milliseconds, so the parallel time must be read on TT before
    /// converting.
    #[test]
    fn tt_clock_round_trips_far_from_reference() {
        let input = r"
            KPL/SCLK
            \begindata
            SCLK_KERNEL_ID            = ( @2004-MAR-02 )
            SCLK_DATA_TYPE_226        = ( 1 )
            SCLK01_TIME_SYSTEM_226    = ( 2 )
            SCLK01_N_FIELDS_226       = ( 2 )
            SCLK01_MODULI_226         = ( 4294967296 65536 )
            SCLK01_OFFSETS_226        = ( 0 0 )
            SCLK01_OUTPUT_DELIM_226   = ( 1 )
            SCLK_PARTITION_START_226  = ( 0.0 )
            SCLK_PARTITION_END_226    = ( 2.8147497671065E+14 )
            SCLK01_COEFFICIENTS_226   = ( 0.0 1.3146108418400E+08 1.0 )
            \begintext";
        let clock = clock_from(input, -226);
        assert!(!clock.parallel_is_tdb);

        let time = Time::<TDB>::new(2_457_316.8);
        let back = clock
            .tick_to_time(clock.time_to_tick(time).unwrap())
            .unwrap();
        let err_s = (back - time).elapsed.abs() * 86400.0;
        assert!(err_s < 1e-6, "round trip error {err_s} s");
    }

    /// A modulus of 0 or a partition that ends before it starts is an error
    /// when the clock is used, and the other clocks still load.
    #[test]
    fn malformed_clocks_keep_their_errors() {
        let clock = |n: i32, moduli: &str, end: &str| {
            format!(
                "SCLK_DATA_TYPE_{n} = ( 1 )\nSCLK01_N_FIELDS_{n} = ( 2 )\n\
                 SCLK01_MODULI_{n} = ( {moduli} )\nSCLK01_OFFSETS_{n} = ( 0 0 )\n\
                 SCLK01_OUTPUT_DELIM_{n} = ( 1 )\nSCLK_PARTITION_START_{n} = ( 100 )\n\
                 SCLK_PARTITION_END_{n} = ( {end} )\n\
                 SCLK01_COEFFICIENTS_{n} = ( 0 0 1 )\n"
            )
        };
        let mut vars = TextKernelVars::default();
        vars.load_text(&format!(
            "\\begindata\n{}{}{}SCLK_DATA_TYPE_74 = ( 1 )\n",
            clock(71, "1000 0", "1E9"),
            clock(72, "1000 10", "50"),
            clock(73, "1000 10", "1E9"),
        ))
        .unwrap();
        let clocks = clocks_from(&vars);
        let err = |id| clocks[&ClockId(id)].as_ref().unwrap_err().to_string();
        assert!(err(-71).contains("MODULI must be at least 1"));
        assert!(err(-72).contains("partition ends"));
        assert!(err(-74).contains("missing"));
        assert!(clocks[&ClockId(-73)].is_ok());
    }

    /// Ticks and times outside the partitions of the clock are errors, as in
    /// CSPICE `sct2e` and `sce2c`. The ends of the range are inside.
    #[test]
    fn conversions_stay_in_the_clock_range() {
        let clock = clock_from(
            "\\begindata\nSCLK_DATA_TYPE_75 = ( 1 )\nSCLK01_N_FIELDS_75 = ( 2 )\n\
             SCLK01_MODULI_75 = ( 1000000 100 )\nSCLK01_OFFSETS_75 = ( 0 0 )\n\
             SCLK01_OUTPUT_DELIM_75 = ( 1 )\nSCLK_PARTITION_START_75 = ( 100 500 )\n\
             SCLK_PARTITION_END_75 = ( 300 1000 )\n\
             SCLK01_COEFFICIENTS_75 = ( 0 0 1 )\n",
            -75,
        );
        assert_eq!(clock.tick_range(), (0.0, 700.0));
        for tick in [0.0, 700.0] {
            let time = clock.tick_to_time(tick).unwrap();
            assert!((clock.time_to_tick(time).unwrap() - tick).abs() < 1e-6);
        }
        for tick in [-1.0, 701.0] {
            assert!(matches!(clock.tick_to_time(tick), Err(Error::Bounds(_))));
        }
        let before = Time::<TDB>::new(2_451_544.0);
        assert!(matches!(clock.time_to_tick(before), Err(Error::Bounds(_))));
    }
}
