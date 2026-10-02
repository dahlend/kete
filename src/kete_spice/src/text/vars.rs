//! Variables of SPICE text kernels.
//!
//! A text kernel holds `\begindata` blocks of assignments between `\begintext`
//! blocks of comments. The parser reads these forms:
//!
//! - An assignment is `NAME = value` or `NAME += value`. `=` replaces a
//!   variable and `+=` appends to it. Thus a file loaded later overrides an
//!   earlier one.
//! - A value is one item or a parenthesized list of items. The items of one
//!   value have one type.
//! - A number has an optional `E` or `D` exponent.
//! - A string is in single quotes. `''` inside a string is one quote. Trailing
//!   blanks are not part of the value.
//! - A date starts with `@`. The parser keeps a date as its text.
//!
//! A list that mixes dates and numbers is an error, because dates stay text.
//! The parser does not limit the line length.

use kete_core::errors::{Error, KeteResult};
use std::collections::HashMap;

/// The value of a text kernel variable.
#[derive(Debug, Clone, PartialEq)]
pub enum TextKernelValue {
    /// Numbers.
    Numbers(Vec<f64>),

    /// Strings, without their quotes.
    Strings(Vec<String>),

    /// `@` dates, as written after the `@`.
    Dates(Vec<String>),
}

/// The variables of the loaded text kernels.
#[derive(Debug, Clone, Default)]
pub struct TextKernelVars {
    vars: HashMap<String, TextKernelValue>,
}

impl TextKernelVars {
    /// Parse a text kernel file and add its variables.
    ///
    /// Bytes that are not UTF-8, such as Latin-1 letters in comments, are read
    /// as the replacement character (see [`read_kernel_text`]).
    ///
    /// # Errors
    /// [`Error::IOError`] if the file cannot be read or does not parse. A file
    /// that fails to parse adds none of its variables.
    pub fn load_file(&mut self, filename: &str) -> KeteResult<()> {
        let text = read_kernel_text(filename)?;
        self.load_text(&text)
            .map_err(|e| Error::IOError(format!("Text kernel {filename}: {e}")))
    }

    /// Parse the text of a text kernel and add its variables.
    ///
    /// # Errors
    /// [`Error::IOError`] if the text does not parse; nothing is added then.
    pub fn load_text(&mut self, text: &str) -> KeteResult<()> {
        let mut data = String::new();
        let mut in_data = false;
        for line in text.lines() {
            match line.trim() {
                r"\begindata" => in_data = true,
                r"\begintext" => in_data = false,
                _ if in_data => {
                    data.push_str(line);
                    data.push('\n');
                }
                _ => {}
            }
        }
        let assignments = parse_assignments(&data).map_err(Error::IOError)?;
        let mut vars = self.vars.clone();
        for (name, append, value) in assignments {
            match (append, vars.get_mut(&name)) {
                (true, Some(old)) => append_value(&name, old, value)?,
                _ => {
                    let _ = vars.insert(name, value);
                }
            }
        }
        self.vars = vars;
        Ok(())
    }

    /// The value of the variable `name`, if it is defined.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TextKernelValue> {
        self.vars.get(name)
    }

    /// The names of all defined variables.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.vars.keys().map(String::as_str)
    }

    /// The numbers of the variable `name`, if it is defined.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the variable does not hold numbers.
    pub fn numbers(&self, name: &str) -> KeteResult<Option<&[f64]>> {
        match self.vars.get(name) {
            None => Ok(None),
            Some(TextKernelValue::Numbers(v)) => Ok(Some(v)),
            Some(_) => Err(Error::ValueError(format!(
                "Kernel variable {name} does not hold numbers."
            ))),
        }
    }

    /// The integers of the variable `name`, if it is defined.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the variable does not hold numbers, or one of
    /// them is not an integer in the range of `i32`.
    pub fn integers(&self, name: &str) -> KeteResult<Option<Vec<i32>>> {
        let Some(v) = self.numbers(name)? else {
            return Ok(None);
        };
        v.iter()
            .map(|x| {
                if x.fract() == 0.0 && (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(x) {
                    // The check above keeps the value integral and in range, so
                    // the cast is exact.
                    #[allow(
                        clippy::cast_possible_truncation,
                        reason = "checked integral and in range"
                    )]
                    Ok(*x as i32)
                } else {
                    Err(Error::ValueError(format!(
                        "Kernel variable {name} must hold integers, found {x}."
                    )))
                }
            })
            .collect::<KeteResult<Vec<i32>>>()
            .map(Some)
    }

    /// The single integer of the variable `name`, if it is defined.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the variable does not hold exactly one number,
    /// or that number is not an integer in the range of `i32`.
    pub fn integer(&self, name: &str) -> KeteResult<Option<i32>> {
        match self.integers(name)?.as_deref() {
            None => Ok(None),
            Some(&[x]) => Ok(Some(x)),
            Some(v) => Err(Error::ValueError(format!(
                "Kernel variable {name} must hold one integer, found {v:?}."
            ))),
        }
    }

    /// The strings of the variable `name`, if it is defined.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the variable does not hold strings.
    pub fn strings(&self, name: &str) -> KeteResult<Option<&[String]>> {
        match self.vars.get(name) {
            None => Ok(None),
            Some(TextKernelValue::Strings(v)) => Ok(Some(v)),
            Some(_) => Err(Error::ValueError(format!(
                "Kernel variable {name} does not hold strings."
            ))),
        }
    }

    /// Remove the variable `name`, and return its value if it was defined.
    pub fn remove(&mut self, name: &str) -> Option<TextKernelValue> {
        self.vars.remove(name)
    }

    /// The single string of the variable `name`, if it is defined.
    ///
    /// # Errors
    /// [`Error::ValueError`] if the variable does not hold exactly one string.
    pub fn string(&self, name: &str) -> KeteResult<Option<&str>> {
        match self.vars.get(name) {
            None => Ok(None),
            Some(TextKernelValue::Strings(v)) if v.len() == 1 => Ok(Some(&v[0])),
            Some(_) => Err(Error::ValueError(format!(
                "Kernel variable {name} must hold one string."
            ))),
        }
    }
}

/// The text of the text kernel file `filename`.
///
/// SPICE requires the data of a text kernel to be ASCII, but comments can hold
/// other bytes, such as Latin-1 letters. Bytes that are not UTF-8 are read as
/// the replacement character, so they do not stop the file from loading.
///
/// # Errors
/// [`Error::IOError`] if the file cannot be read.
pub fn read_kernel_text(filename: &str) -> KeteResult<String> {
    let bytes = std::fs::read(filename).map_err(|e| Error::IOError(format!("{filename}: {e}")))?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Append the items of `new` to the variable `name`, which holds `old`.
///
/// # Errors
/// [`Error::IOError`] if `new` holds items of another type than `old`.
fn append_value(name: &str, old: &mut TextKernelValue, new: TextKernelValue) -> KeteResult<()> {
    match (old, new) {
        (TextKernelValue::Numbers(a), TextKernelValue::Numbers(b)) => a.extend(b),
        (TextKernelValue::Strings(a), TextKernelValue::Strings(b))
        | (TextKernelValue::Dates(a), TextKernelValue::Dates(b)) => a.extend(b),
        _ => {
            return Err(Error::IOError(format!(
                "Kernel variable {name}: += adds items of a different type."
            )));
        }
    }
    Ok(())
}

/// One item of a value.
enum Item {
    Number(f64),
    Str(String),
    Date(String),
}

/// Parse the assignments of the data blocks, joined with newlines.
///
/// Each assignment is `(name, is +=, value)`.
///
/// # Errors
/// A message that names the variable and the problem, if the text does not
/// parse.
fn parse_assignments(data: &str) -> Result<Vec<(String, bool, TextKernelValue)>, String> {
    let mut out = Vec::new();
    let mut rest = data.trim_start();
    while !rest.is_empty() {
        // The name runs to whitespace or '='. A name may hold '+', so "NAME+=" is the
        // name followed by "+=".
        let mut end = rest
            .find(|c: char| c.is_whitespace() || c == '=')
            .unwrap_or(rest.len());
        if rest[end..].starts_with('=') && rest[..end].ends_with('+') {
            end -= 1;
        }
        let name = &rest[..end];
        if name.is_empty() {
            return Err(format!("expected a variable name at {:?}", head(rest)));
        }
        rest = rest[end..].trim_start();
        let append = if let Some(r) = rest.strip_prefix("+=") {
            rest = r;
            true
        } else if let Some(r) = rest.strip_prefix('=') {
            rest = r;
            false
        } else {
            return Err(format!("expected = or += after {name}"));
        };
        rest = rest.trim_start();
        let mut items = Vec::new();
        if let Some(r) = rest.strip_prefix('(') {
            rest = r;
            loop {
                rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
                if let Some(r) = rest.strip_prefix(')') {
                    rest = r;
                    break;
                }
                if rest.is_empty() {
                    return Err(format!("{name}: the list is not closed with )"));
                }
                let (item, r) = parse_item(rest).map_err(|e| format!("{name}: {e}"))?;
                items.push(item);
                rest = r;
            }
        } else {
            let (item, r) = parse_item(rest).map_err(|e| format!("{name}: {e}"))?;
            items.push(item);
            rest = r;
        }
        let value = collect_items(items).map_err(|e| format!("{name}: {e}"))?;
        out.push((name.to_string(), append, value));
        rest = rest.trim_start();
    }
    Ok(out)
}

/// Parse one item at the start of `s`, and return it with the rest of `s`.
///
/// # Errors
/// A message, if `s` does not start with a number, a closed string or a
/// date.
fn parse_item(s: &str) -> Result<(Item, &str), String> {
    if let Some(body) = s.strip_prefix('\'') {
        let mut value = String::new();
        let mut chars = body.char_indices();
        while let Some((idx, c)) = chars.next() {
            match c {
                '\'' if body[idx + 1..].starts_with('\'') => {
                    value.push('\'');
                    let _ = chars.next();
                }
                // Trailing blanks are not part of a value.
                '\'' => return Ok((Item::Str(value.trim_end().to_string()), &body[idx + 1..])),
                '\n' => break,
                c => value.push(c),
            }
        }
        return Err(format!(
            "a string is not closed on its line at {:?}",
            head(s)
        ));
    }
    let end = s
        .find(|c: char| c.is_whitespace() || c == ',' || c == ')' || c == '(')
        .unwrap_or(s.len());
    let token = &s[..end];
    if token.is_empty() {
        return Err(format!("expected a value at {:?}", head(s)));
    }
    if let Some(date) = token.strip_prefix('@') {
        return Ok((Item::Date(date.to_string()), &s[end..]));
    }
    let number: f64 = token
        .replace(['D', 'd'], "E")
        .parse()
        .ok()
        .filter(|x: &f64| x.is_finite() && !token.starts_with(char::is_alphabetic))
        .ok_or_else(|| format!("{token:?} is not a number, a quoted string or an @ date"))?;
    Ok((Item::Number(number), &s[end..]))
}

/// The value of one assignment from its items.
///
/// # Errors
/// A message, if there are no items or the items have different types.
fn collect_items(items: Vec<Item>) -> Result<TextKernelValue, String> {
    let mixed = || "the items have different types".to_string();
    let mut value = match items.first() {
        None => return Err("the value is empty".into()),
        Some(Item::Number(_)) => TextKernelValue::Numbers(Vec::new()),
        Some(Item::Str(_)) => TextKernelValue::Strings(Vec::new()),
        Some(Item::Date(_)) => TextKernelValue::Dates(Vec::new()),
    };
    for item in items {
        match (&mut value, item) {
            (TextKernelValue::Numbers(v), Item::Number(x)) => v.push(x),
            (TextKernelValue::Strings(v), Item::Str(x))
            | (TextKernelValue::Dates(v), Item::Date(x)) => {
                v.push(x);
            }
            _ => return Err(mixed()),
        }
    }
    Ok(value)
}

/// The start of `s`, for error messages.
fn head(s: &str) -> &str {
    let end = s.char_indices().nth(30).map_or(s.len(), |(i, _)| i);
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(text: &str) -> TextKernelVars {
        let mut vars = TextKernelVars::default();
        vars.load_text(text).unwrap();
        vars
    }

    /// Each value form parses; comments outside data blocks are ignored.
    #[test]
    fn value_forms() {
        let p = vars(
            "KPL/FK\nA comment = ( not data )\n\\begindata\n\
             A = 1\n\
             B = ( 1, -2.5D3 +4.0e-1 )\n\
             FRAME_67P/C-G_CK = -1000012000\n\
             FRAME_ROS_SA+Y = -226015\n\
             G+= 3\n\
             C = 'it''s  '\n\
             D = ( 'x' 'y',\n  'z' )\n\
             E = @2011-01-05/12:01:58\n\
             \\begintext\nmore comment\n\\begindata\nF=2\n",
        );
        assert_eq!(p.integer("A").unwrap(), Some(1));
        assert_eq!(p.numbers("B").unwrap(), Some(&[1.0, -2500.0, 0.4][..]));
        assert_eq!(p.integer("FRAME_67P/C-G_CK").unwrap(), Some(-1_000_012_000));
        assert_eq!(p.integer("FRAME_ROS_SA+Y").unwrap(), Some(-226_015));
        assert_eq!(p.integer("G").unwrap(), Some(3));
        assert_eq!(p.string("C").unwrap(), Some("it's"));
        assert_eq!(
            p.get("D"),
            Some(&TextKernelValue::Strings(vec![
                "x".into(),
                "y".into(),
                "z".into()
            ]))
        );
        assert_eq!(
            p.get("E"),
            Some(&TextKernelValue::Dates(vec!["2011-01-05/12:01:58".into()]))
        );
        assert_eq!(p.integer("F").unwrap(), Some(2));
        assert!(p.get("A comment").is_none());
    }

    /// `=` replaces, `+=` appends, also across loads, and creates a missing
    /// variable.
    #[test]
    fn assignment_and_append() {
        let mut p = vars("\\begindata\nA = ( 1 2 )\nA += 3\nB += 'x'\n");
        assert_eq!(p.numbers("A").unwrap(), Some(&[1.0, 2.0, 3.0][..]));
        assert_eq!(p.string("B").unwrap(), Some("x"));
        p.load_text("\\begindata\nA += ( 4 )\n").unwrap();
        assert_eq!(p.numbers("A").unwrap(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
        p.load_text("\\begindata\nA = 5\n").unwrap();
        assert_eq!(p.integer("A").unwrap(), Some(5));
    }

    /// Malformed text is an error and adds nothing.
    #[test]
    fn malformed_text_adds_nothing() {
        let mut p = vars("\\begindata\nA = 1\n");
        for bad in [
            "\\begindata\nB = 2\nC = ( 1 2\n",
            "\\begindata\nB = 2\nC = 'open\n'\n",
            "\\begindata\nB = 2\nC = ( 1 'x' )\n",
            "\\begindata\nB = 2\nC 3\n",
            "\\begindata\nB = 2\nC = abc\n",
            "\\begindata\nB = 2\nC = inf\n",
            "\\begindata\nB = 2\nA += 'x'\n",
        ] {
            assert!(p.load_text(bad).is_err(), "{bad:?}");
            assert!(p.get("B").is_none(), "{bad:?}");
            assert_eq!(p.integer("A").unwrap(), Some(1));
        }
    }

    /// The typed getters reject values of the wrong type or count.
    #[test]
    fn typed_getters() {
        let p = vars("\\begindata\nA = ( 1 2 )\nB = 1.5\nC = 'x'\nD = ( 'x' 'y' )\n");
        assert!(p.integer("A").is_err());
        assert!(p.integer("B").is_err());
        assert!(p.integer("C").is_err());
        assert!(p.string("D").is_err());
        assert!(p.numbers("C").is_err());
        assert_eq!(p.integer("MISSING").unwrap(), None);
    }

    /// A byte that is not UTF-8 in a comment does not stop a file from
    /// loading.
    #[test]
    fn non_utf8_comments_load() {
        let path = std::env::temp_dir().join("kete_vars_latin1.tf");
        std::fs::write(&path, b"KPL/FK\nAuthor: Jos\xe9\n\\begindata\nX = 1\n").unwrap();
        let mut vars = TextKernelVars::default();
        vars.load_file(path.to_str().unwrap()).unwrap();
        assert_eq!(vars.integer("X").unwrap(), Some(1));
    }
}
