//! Meta-kernels: text kernels that list other kernels to load.
//!
//! A text kernel that sets `KERNELS_TO_LOAD` is a meta-kernel, whatever its
//! header, as in CSPICE `furnsh`. These rules also follow `furnsh`:
//!
//! - An element of `KERNELS_TO_LOAD` that ends in `+` continues into the next
//!   element, without the `+`.
//! - `PATH_SYMBOLS` and `PATH_VALUES` are lists of equal length. A `$` and a
//!   symbol in a file name is replaced by the value of the symbol. Symbols are
//!   case sensitive, the longest symbol that matches is used, and a value is
//!   not expanded again. A `$` that no symbol follows stays in the name.
//! - A relative path is relative to the working directory, not to the
//!   meta-kernel.
//! - A meta-kernel cannot list another meta-kernel.
//! - [`META_KERNEL_VARS`] are not kept with the variables of the loaded text
//!   kernels. The other variables of a meta-kernel are.

use super::TextKernelVars;
use kete_core::errors::{Error, KeteResult};

/// The variables that make a text kernel a meta-kernel.
pub const META_KERNEL_VARS: [&str; 3] = ["KERNELS_TO_LOAD", "PATH_SYMBOLS", "PATH_VALUES"];

/// The files a meta-kernel lists, in order, or `None` if `vars` does not set
/// `KERNELS_TO_LOAD`.
///
/// # Errors
/// [`Error::ValueError`] if one of [`META_KERNEL_VARS`] does not hold strings,
/// or `PATH_SYMBOLS` and `PATH_VALUES` differ in length.
pub fn kernels_to_load(vars: &TextKernelVars) -> KeteResult<Option<Vec<String>>> {
    let Some(elements) = vars.strings("KERNELS_TO_LOAD")? else {
        return Ok(None);
    };
    let symbols = vars.strings("PATH_SYMBOLS")?.unwrap_or_default();
    let values = vars.strings("PATH_VALUES")?.unwrap_or_default();
    if symbols.len() != values.len() {
        return Err(Error::ValueError(format!(
            "Meta-kernel has {} PATH_SYMBOLS and {} PATH_VALUES; the counts must match.",
            symbols.len(),
            values.len()
        )));
    }

    let mut files = Vec::new();
    let mut name = String::new();
    for element in elements {
        if let Some(start) = element.strip_suffix('+') {
            name.push_str(start);
        } else {
            name.push_str(element);
            files.push(substitute(&name, symbols, values));
            name.clear();
        }
    }
    if !name.is_empty() {
        files.push(substitute(&name, symbols, values));
    }
    Ok(Some(files))
}

/// `name` with each `$` and symbol replaced by the value of the symbol.
fn substitute(name: &str, symbols: &[String], values: &[String]) -> String {
    let mut out = String::with_capacity(name.len());
    let mut rest = name;
    while let Some(i) = rest.find('$') {
        out.push_str(&rest[..i]);
        let after = &rest[i + 1..];
        let longest = symbols
            .iter()
            .zip(values)
            .filter(|(symbol, _)| !symbol.is_empty() && after.starts_with(symbol.as_str()))
            .max_by_key(|(symbol, _)| symbol.len());
        if let Some((symbol, value)) = longest {
            out.push_str(value);
            rest = &after[symbol.len()..];
        } else {
            out.push('$');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(text: &str) -> KeteResult<Option<Vec<String>>> {
        let mut vars = TextKernelVars::default();
        vars.load_text(text).unwrap();
        kernels_to_load(&vars)
    }

    /// Symbols match case sensitively and by the longest symbol, a value is not
    /// expanded again, and `+` joins an element to the next. These cases were
    /// checked against CSPICE `furnsh`.
    #[test]
    fn expands_like_furnsh() {
        let listed = files(
            "\\begindata\n\
             PATH_VALUES = ( '/k/sub', '/k/deep', '$B', '/k' )\n\
             PATH_SYMBOLS = ( 'A', 'AB', 'C', 'B' )\n\
             KERNELS_TO_LOAD = ( '$A/a.tls' '$AB/b.tsc' '$ab/c' '$Ax' '$C/d'\n\
             '$Q/e' '/x$B$B/f' '$A/long+' '+' '.tf' '/last+' )\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            listed,
            [
                "/k/sub/a.tls",
                "/k/deep/b.tsc",
                "$ab/c",
                "/k/subx",
                "$B/d",
                "$Q/e",
                "/x/k/k/f",
                "/k/sub/long.tf",
                "/last",
            ]
        );
    }

    /// A text kernel without `KERNELS_TO_LOAD` is not a meta-kernel; unpaired
    /// symbols are an error.
    #[test]
    fn not_meta_and_malformed() {
        assert_eq!(files("\\begindata\nX = 1\n").unwrap(), None);
        assert!(files("\\begindata\nPATH_SYMBOLS = 'A'\nKERNELS_TO_LOAD = '$A/a'\n").is_err());
        assert!(files("\\begindata\nKERNELS_TO_LOAD = 1\n").is_err());
    }
}
