use std::{marker::PhantomData, str::FromStr};

use crossbeam::channel::tick;
/// Parsing text kernels used by SPICE.
use nom::{
    branch::alt,
    bytes::{
        complete::{take, take_until, take_until1, take_while, take_while1},
        streaming::tag,
    },
    character::complete::{alpha1, char, digit1, space0, space1},
    combinator::{map_res, opt, Opt},
    error::{context, ParseError},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, pair, preceded, terminated},
    IResult, Input, Parser,
};

use crate::{
    errors::{Error, KeteResult},
    spice::spice_jd_to_jd,
    time::scales::TimeScale,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Sclk {
    naif_id: i32,
    kernel_id: String,

    // Required Fields
    n_fields: u32,
    moduli: Vec<u32>,
    offsets: Vec<u32>,

    partition_start: Vec<f64>,
    partition_end: Vec<f64>,

    coefficients: Vec<[f64; 3]>,

    // Delimiter here is only use for output formatting.
    output_delim: char,

    /// Timescale defaults to TDB if not specified.
    tdb: bool,
}

impl Sclk {
    /// Parse a spacecraft clock string into integers and a partition number.
    ///
    /// This handles all the rollover and partitioning logic.
    fn parse_time_string(&self, time_str: &str) -> KeteResult<(usize, f64)> {
        let (_, (partition, mut fields)) = parse_time_fields(time_str)
            .map_err(|_| Error::ValueError("Failed to parse time fields.".into()))?;

        if fields.len() > self.n_fields as usize || fields.is_empty() {
            return Err(Error::ValueError(format!(
                "Fields in time string must be between 1 and {}, found {}.",
                self.n_fields,
                fields.len()
            )));
        }

        // Add the offsets to all of the fields.
        fields
            .iter_mut()
            .zip(self.offsets.iter())
            .for_each(|(field, &offset)| {
                *field += offset as usize;
            });

        let mut rollover = false;
        fields
            .iter_mut()
            .zip(self.moduli.iter())
            .rev()
            .for_each(|(mut val, &modulus)| {
                if rollover {
                    *val += 1;
                    rollover = false;
                }
                if *val >= (modulus as usize) {
                    rollover = true;
                    *val -= modulus as usize;
                }
            });

        let mut tick_rates = vec![1];
        for modulo in self.moduli.iter().skip(1).rev() {
            let last = tick_rates.last().unwrap();
            tick_rates.push(*modulo as usize * last);
        }
        tick_rates.reverse();

        // compute a floating point representation of the spacecraft clock time
        let mut sc_ticks: usize = 0;
        fields
            .iter()
            .zip(tick_rates.iter())
            .for_each(|(field, rate)| {
                sc_ticks += field * rate;
            });

        let clock_rate = self.find_clock_rate(sc_ticks as f64)?;

        let par_time = clock_rate[2] * (sc_ticks as f64 - clock_rate[0])
            / (*tick_rates.first().unwrap() as f64)
            + clock_rate[1];

        // find partition from the par_time by looking at the bounds
        // self.partition_start and self.partition_end

        let mut idx = self
            .partition_start
            .partition_point(|&start| start <= sc_ticks as f64);

        if partition.is_some() && Some(idx) != partition {
            return Err(Error::ValueError(format!(
                "Partition mismatch: expected {}, found {}",
                partition.unwrap(),
                idx
            )));
        }
        if idx == 0 {
            return Err(Error::ValueError(format!(
                "Time {} is before the first partition start.",
                time_str
            )));
        }

        Ok((idx, par_time))
    }

    /// Go through the coefficients and find the clock rate for a given spacecraft clock.
    fn find_clock_rate(&self, sc_clock: f64) -> KeteResult<[f64; 3]> {
        let mut idx = self
            .coefficients
            .partition_point(|probe| probe[0] <= sc_clock);
        idx = idx.saturating_sub(1);

        let coef = self.coefficients[idx];
        Ok(coef)
    }
}

/// Parse a formatted time string into a vector of integers.
///
/// Time strings are integers separated by spaces, dashes, colons, commas, or periods.
/// Spaces between integers and separators are optional.
///
/// String formats include an optional partition number followed by a slash.
/// If this is not provided, it defaults to the first partition 1.
fn parse_time_fields(input: &str) -> IResult<&str, (Option<usize>, Vec<usize>)> {
    preceded(
        space0,
        pair(
            opt(terminated(
                delimited(space0, parse_num::<usize>, space0),
                terminated(char('/'), space0),
            )),
            separated_list1(take_while1(|c| " -:,.".contains(c)), parse_num),
        ),
    )
    .parse(input)
}

// Code below is used to parse SCLK files into the Sclk struct.

impl TryFrom<Vec<SclkToken>> for Sclk {
    type Error = Error;

    fn try_from(value: Vec<SclkToken>) -> Result<Self, Self::Error> {
        let mut naif_id: Option<i32> = None;
        let mut kernel_id: Option<String> = None;
        let mut n_fields: Option<u32> = None;
        let mut moduli: Option<Vec<u32>> = None;
        let mut offsets: Option<Vec<u32>> = None;
        let mut output_delim: Option<char> = None;
        let mut partition_start: Option<Vec<f64>> = None;
        let mut partition_end: Option<Vec<f64>> = None;
        let mut coefficients: Option<Vec<[f64; 3]>> = None;
        let mut tdb: Option<bool> = None;

        for token in value {
            match token {
                SclkToken::MagicNumber => continue,
                SclkToken::KernelID(id) => {
                    if kernel_id.is_some() {
                        return Err(Error::ValueError("Multiple SCLK_KERNEL_ID found.".into()));
                    }
                    kernel_id = Some(id);
                }
                SclkToken::Comments(_) => continue,
                SclkToken::DataType(id, dtype) => {
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    if dtype != 1 {
                        return Err(Error::ValueError(format!(
                            "SCLK clock type must be 1, found {}.",
                            dtype
                        )));
                    }
                } // Data type is always 1
                SclkToken::NFields01(id, n) => {
                    if n_fields.is_some() {
                        return Err(Error::ValueError("Multiple SCLK N_FIELDS found.".into()));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    if n < 1 {
                        return Err(Error::ValueError(format!(
                            "SCLK N_FIELDS must be at least 1, found {}.",
                            n
                        )));
                    }
                    naif_id = Some(id);
                    n_fields = Some(n);
                }
                SclkToken::Moduli01(id, val) => {
                    if moduli.is_some() {
                        return Err(Error::ValueError("Multiple SCLK MODULI found.".into()));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    moduli = Some(val);
                }
                SclkToken::Offsets01(id, val) => {
                    if offsets.is_some() {
                        return Err(Error::ValueError("Multiple SCLK OFFSETS found.".into()));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    offsets = Some(val);
                }
                SclkToken::OutputDelim01(id, val) => {
                    if output_delim.is_some() {
                        return Err(Error::ValueError(
                            "Multiple SCLK OUTPUT_DELIM found.".into(),
                        ));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    output_delim = Some(match val {
                        1 => '.',
                        2 => ':',
                        3 => '-',
                        4 => ',',
                        _ => ' ',
                    });
                }
                SclkToken::PartitionStart(id, val) => {
                    if partition_start.is_some() {
                        return Err(Error::ValueError(
                            "Multiple SCLK PARTITION_START found.".into(),
                        ));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    partition_start = Some(val);
                }
                SclkToken::PartitionEnd(id, val) => {
                    if partition_end.is_some() {
                        return Err(Error::ValueError(
                            "Multiple SCLK PARTITION_END found.".into(),
                        ));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    partition_end = Some(val);
                }
                SclkToken::Coefficients01(id, val) => {
                    if coefficients.is_some() {
                        return Err(Error::ValueError(
                            "Multiple SCLK Coefficients found.".into(),
                        ));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    coefficients = Some(
                        val.chunks(3)
                            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
                            .collect(),
                    );
                }
                SclkToken::TimeSystem01(id, val) => {
                    if tdb.is_some() {
                        return Err(Error::ValueError("Multiple SCLK Time System found.".into()));
                    }
                    let id = -(id as i32);
                    if naif_id.is_some() && Some(id) != naif_id {
                        return Err(Error::ValueError("Multiple SCLK NAIF ids found.".into()));
                    }
                    naif_id = Some(id);
                    // require that val be either 1 or 2
                    if val != 1 && val != 2 {
                        return Err(Error::ValueError(format!(
                            "SCLK Time System must be 1 (TDB) or 2 (TT), found {}.",
                            val
                        )));
                    }
                    tdb = Some(val == 1);
                }
                SclkToken::Unknown(s) => {
                    return Err(Error::ValueError(format!("Unknown SCLK line: {}", s)))
                }
            }
        }

        // validation
        let naif_id = naif_id.ok_or(Error::ValueError("SCLK NAIF ID is missing.".into()))?;
        let kernel_id = kernel_id.ok_or(Error::ValueError("SCLK Kernel ID is missing.".into()))?;
        let n_fields = n_fields.ok_or(Error::ValueError("SCLK N_FIELDS is missing.".into()))?;
        let moduli = moduli.ok_or(Error::ValueError("SCLK MODULI is missing.".into()))?;
        let offsets = offsets.ok_or(Error::ValueError("SCLK OFFSETS is missing.".into()))?;
        let output_delim =
            output_delim.ok_or(Error::ValueError("SCLK OUTPUT_DELIM is missing.".into()))?;
        let partition_start =
            partition_start.ok_or(Error::ValueError("SCLK PARTITION_START is missing.".into()))?;
        let partition_end =
            partition_end.ok_or(Error::ValueError("SCLK PARTITION_END is missing.".into()))?;
        let coefficients =
            coefficients.ok_or(Error::ValueError("SCLK Coefficients are missing.".into()))?;
        let tdb = tdb.unwrap_or(true); // Default to TDB if not specified
        if partition_start.len() != n_fields as usize {
            return Err(Error::ValueError(format!(
                "SCLK PARTITION_START length ({}) does not match N_FIELDS ({})",
                partition_start.len(),
                n_fields
            )));
        }
        if partition_end.len() != n_fields as usize {
            return Err(Error::ValueError(format!(
                "SCLK PARTITION_END length ({}) does not match N_FIELDS ({})",
                partition_end.len(),
                n_fields
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

        Ok(Sclk {
            naif_id,
            kernel_id,
            n_fields,
            moduli,
            offsets,
            output_delim,
            partition_start,
            partition_end,
            coefficients,
            tdb,
        })
    }
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
enum SclkToken {
    MagicNumber,
    KernelID(String),
    Comments(String),
    DataType(u32, i32),
    NFields01(u32, u32),
    Moduli01(u32, Vec<u32>),
    OutputDelim01(u32, u32),
    PartitionStart(u32, Vec<f64>),
    PartitionEnd(u32, Vec<f64>),
    Coefficients01(u32, Vec<f64>),
    TimeSystem01(u32, u32),
    Offsets01(u32, Vec<u32>),
    Unknown(String),
}

/// SCLK file data is stored as key value pairs.
///
/// Keys are made up of a text string with some of them ending in a numeric suffix.
/// This parses the optional numeric suffix from the key.
///    
/// parse_key_suffix("Thing_a_b_10") == Ok(("", ("Thing_a_b_", Some(10))))
/// parse_key_suffix("Thing_a_b") == Ok(("", ("Thing_a_b", None)))
///
fn parse_key_suffix(input: &str) -> IResult<&str, (&str, Option<u32>)> {
    let (rem, word) = take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_').parse(input)?;

    if word.len() < 3 {
        return Err(nom::Err::Error(ParseError::from_error_kind(
            input,
            nom::error::ErrorKind::TakeTill1,
        )));
    }
    let word_len = word.len();

    // without using nom for parsing, check if the last two characters are digits
    let is_digit = |c: char| c.is_ascii_digit();
    let (word, num) = if is_digit(word.chars().nth(word_len - 1).unwrap())
        && is_digit(word.chars().nth(word_len - 2).unwrap())
    {
        // take the last two characters as a number
        let num_str = &word[word_len - 2..];
        let num = num_str.parse::<u32>().ok();
        (&word[..word_len - 2], num)
    } else {
        // no number at the end
        (word, None)
    };

    Ok((rem, (word, num)))
}

/// SCLK file data is stored as key value pairs, this parses a specific key and its value.
///
///
/// Thing_a_b_10 = ( foo bar
///                   baz)
///
/// Parses into (Some(10), "foo bar \n baz")
///
fn parse_line<'a>(
    input: &'a str,
    has_id: bool,
    expected_str: &str,
) -> IResult<&'a str, (Option<u32>, &'a str)> {
    let (rem, (word, id)) = preceded(sp, parse_key_suffix).parse(input)?;

    if id.is_some() != has_id {
        return Err(nom::Err::Error(ParseError::from_error_kind(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }
    if word != expected_str {
        return Err(nom::Err::Error(ParseError::from_error_kind(
            input,
            nom::error::ErrorKind::Tag,
        )));
    }

    let (rem, _) = delimited(sp, char('='), sp).parse(rem)?;
    let (rem, contents) = delimited(char('('), take_while1(|c| c != ')'), char(')')).parse(rem)?;
    Ok((rem, (id, contents.trim())))
}

fn comments(input: &str) -> IResult<&str, SclkToken> {
    let (rem, comment) = take_until(r"\begindata")(input)?;
    Ok((rem, SclkToken::Comments(comment.to_string())))
}

/// Skip whitespace characters.
fn sp<'a, E: ParseError<&'a str>>(i: &'a str) -> IResult<&'a str, &'a str, E> {
    let chars = " \t\r\n";
    take_while(move |c| chars.contains(c))(i)
}

fn parse_num<T: FromStr>(input: &str) -> IResult<&str, T> {
    let chars = ".Ee+-";
    map_res(
        take_while1(|c: char| c.is_ascii_digit() || chars.contains(c)),
        |s: &str| s.parse::<T>(),
    )
    .parse(input)
}

fn parse_num_vec<T: FromStr>(input: &str) -> IResult<&str, Vec<T>> {
    context("vec", delimited(sp, separated_list1(sp, parse_num), sp)).parse(input)
}

fn kernel_id(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (id, contents)) = parse_line(input, false, "SCLK_KERNEL_ID")?;
    Ok((rem, SclkToken::KernelID(contents.to_string())))
}

fn n_fields(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK01_N_FIELDS_")?;
    let (_, val) = parse_num(contents)?;
    Ok((rem, SclkToken::NFields01(sc_id.unwrap(), val)))
}

/// Data type must be 1, as there is only one data type which has ever been defined
/// in the SPICE standard.
fn data_type(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK_DATA_TYPE_")?;
    let (_, val) = parse_num(contents)?;
    Ok((rem, SclkToken::DataType(sc_id.unwrap(), val)))
}

fn n_moduli(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK01_MODULI_")?;
    let (_, val) = parse_num_vec(contents)?;
    Ok((rem, SclkToken::Moduli01(sc_id.unwrap(), val)))
}

fn offsets(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK01_OFFSETS_")?;
    let (_, val) = parse_num_vec(contents)?;
    Ok((rem, SclkToken::Offsets01(sc_id.unwrap(), val)))
}

fn output_delim(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK01_OUTPUT_DELIM_")?;
    let (_, val) = parse_num(contents)?;
    Ok((rem, SclkToken::OutputDelim01(sc_id.unwrap(), val)))
}

fn partition_start(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK_PARTITION_START_")?;
    let (_, val) = parse_num_vec(contents)?;
    Ok((rem, SclkToken::PartitionStart(sc_id.unwrap(), val)))
}

fn partition_end(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK_PARTITION_END_")?;
    let (_, val) = parse_num_vec(contents)?;
    Ok((rem, SclkToken::PartitionEnd(sc_id.unwrap(), val)))
}

fn coefficients(input: &str) -> IResult<&str, SclkToken> {
    let (rem, (sc_id, contents)) = parse_line(input, true, "SCLK01_COEFFICIENTS_")?;
    let (_, val) = parse_num_vec(contents)?;
    Ok((rem, SclkToken::Coefficients01(sc_id.unwrap(), val)))
}

fn unknown(input: &str) -> IResult<&str, SclkToken> {
    let (rem, val) = context("unknown", preceded(sp, take_until1(" "))).parse(input)?;
    Ok((rem, SclkToken::Unknown(val.trim().to_string())))
}

fn magic_number(input: &str) -> IResult<&str, SclkToken> {
    preceded(sp, tag("KPL/SCLK"))
        .map(|_| SclkToken::MagicNumber)
        .parse(input)
}

fn parse_sclk_string(input: &str) -> IResult<&str, Vec<SclkToken>> {
    delimited(
        magic_number,
        preceded(
            preceded(comments, tag(r"\begindata")),
            separated_list0(
                sp,
                alt([
                    kernel_id,
                    n_fields,
                    data_type,
                    n_moduli,
                    offsets,
                    output_delim,
                    partition_start,
                    partition_end,
                    coefficients,
                    unknown,
                ]),
            ),
        ),
        preceded(sp, tag(r"\begintext")),
    )
    .parse(input)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_parse_time_field() {
        let input = "  1:2   3 - 5:8 . 9 9 ";
        let (_, result) = parse_time_fields(input).unwrap();
        assert_eq!(result, (None, vec![1, 2, 3, 5, 8, 9, 9]));

        let input = " 5 /  1:2   3 - 5:8 . 9 9 ";
        let (_, result) = parse_time_fields(input).unwrap();
        assert_eq!(result, (Some(5), vec![1, 2, 3, 5, 8, 9, 9]));
    }

    #[test]
    fn test_num_vec() {
        let input = "  1  2\n\t 3 3.14E-2 ";
        let result = parse_num_vec::<f64>(input);
        assert!(result.is_ok());
        let (_, vec) = result.unwrap();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 3.14e-2]);
    }

    #[test]
    fn test_num() {
        let input = "123 = ";
        let result = parse_num::<i32>(input);
        assert!(result.is_ok());
        let (res, vec) = result.unwrap();
        assert_eq!(vec, 123);
        assert_eq!(res, "= ");
    }

    #[test]
    fn test_offset() {
        let input = " SCLK01_OFFSETS_77         = (        0  0  0 0 )";
        let result = offsets(input);
        assert!(result.is_ok());
        let (_, vec) = result.unwrap();
        assert_eq!(vec, SclkToken::Offsets01(77, vec![0, 0, 0, 0]));
    }

    #[test]
    fn test_kernel_id() {
        let input = "SCLK_KERNEL_ID            = ( @04-SEP-1990//4:23:00 )";
        let (_, vec) = kernel_id(input).unwrap();
        assert_eq!(
            vec,
            SclkToken::KernelID("@04-SEP-1990//4:23:00".to_string())
        );
    }

    #[test]
    fn test_parse_key() {
        let a = parse_key_suffix("Thing_a_b_10");
        assert_eq!(a.unwrap(), ("", ("Thing_a_b_", Some(10))));

        let b = parse_key_suffix("SCLK01_OFFSETS_77 ");
        assert_eq!(b.unwrap(), (" ", ("SCLK01_OFFSETS_", Some(77))));

        let c = parse_key_suffix("SCLK01_OFFSETS_ ");
        assert_eq!(c.unwrap(), (" ", ("SCLK01_OFFSETS_", None)));
    }

    #[test]
    fn test_parse_line() {
        let a = "
               SCLK_PARTITION_START_77   = ( 0.0000000000000E+00
                                            2.5465440000000E+07
                                            7.2800001000000E+07
                                            1.3176800000000E+08 )
            ";
        let (res, sclk) = partition_start(a).unwrap();
        assert_eq!(
            sclk,
            SclkToken::PartitionStart(77, vec![0.0, 2.546544E+07, 7.2800001E+07, 1.31768E+08])
        );
    }

    #[test]
    fn test_partition_start() {
        let a = "Thing_a_b_10 = ( foo bar baz)";
        let (res, (id, rem)) = parse_line(a, true, "Thing_a_b_").unwrap();
        assert_eq!(id, Some(10));
        assert_eq!(rem, "foo bar baz");
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

        let (_, vec) = parse_sclk_string(input).unwrap();
        assert_eq!(vec.len(), 9);
        assert_eq!(
            vec[0],
            SclkToken::KernelID("@04-SEP-1990//4:23:00".to_string())
        );
        assert_eq!(vec[1], SclkToken::DataType(77, 1));
        assert_eq!(vec[2], SclkToken::NFields01(77, 4));
        assert_eq!(vec[3], SclkToken::Moduli01(77, vec![16777215, 91, 10, 8]));
        assert_eq!(vec[4], SclkToken::Offsets01(77, vec![0, 0, 0, 0]));
        assert_eq!(vec[5], SclkToken::OutputDelim01(77, 2));
        assert_eq!(
            vec[6],
            SclkToken::PartitionStart(77, vec![0.0, 2.546544E+07, 7.2800001E+07, 1.31768E+08])
        );

        let clock = Sclk::try_from(vec).unwrap();

        let _ = clock.parse_time_string("1/1000:00:00").unwrap();
    }
}
