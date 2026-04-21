use crate::fitting::{ConvergenceError, FittingResult};

/// Find a root of `f` in the bracket `[a, b]` using the bisection method.
///
/// The function must have opposite signs at `a` and `b` (i.e. `f(a) * f(b) < 0`).
/// Returns the midpoint of the final bracket after at most `max_iter` halvings,
/// or when the bracket width drops below machine epsilon relative to the midpoint.
///
/// ```
///     use kete_stats::fitting::bisection;
///     let f = |x: f64| x * x - 2.0;
///     let root = bisection(f, 1.0, 2.0, 100).unwrap();
///     assert!((root - std::f64::consts::SQRT_2).abs() < 1e-15);
/// ```
///
/// # Errors
///
/// Returns [`ConvergenceError::NonFinite`] if any function evaluation produces a
/// non-finite value, or [`ConvergenceError::InvalidInput`] if `f(a)` and `f(b)` do
/// not have opposite signs.
pub fn bisection<T>(func: impl Fn(T) -> T, a: T, b: T, max_iter: usize) -> FittingResult<T>
where
    T: num_traits::Float,
{
    let mut lo = a;
    let mut hi = b;
    let f_lo = func(lo);
    let f_hi = func(hi);

    if !f_lo.is_finite() || !f_hi.is_finite() {
        return Err(ConvergenceError::NonFinite);
    }
    if f_lo * f_hi > T::zero() {
        return Err(ConvergenceError::InvalidInput(
            "bracket endpoints do not have opposite signs",
        ));
    }

    #[allow(
        clippy::missing_panics_doc,
        reason = "By construction this cannot panic."
    )]
    let epsilon = T::epsilon() * T::from(10.0).unwrap();

    for _ in 0..max_iter {
        let mid = (lo + hi) / (T::one() + T::one());
        if (hi - lo) < epsilon * mid.abs() {
            return Ok(mid);
        }
        let f_mid = func(mid);
        if !f_mid.is_finite() {
            return Err(ConvergenceError::NonFinite);
        }
        if func(lo) * f_mid <= T::zero() {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Ok((lo + hi) / (T::one() + T::one()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisection_sqrt2() {
        let f = |x: f64| x * x - 2.0;
        let root = bisection(f, 1.0, 2.0, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn test_bisection_negative_root() {
        let f = |x: f64| x + 3.0;
        let root = bisection(f, -5.0, 0.0, 100).unwrap();
        assert!((root - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_bisection_no_bracket() {
        let f = |x: f64| x * x + 1.0;
        let result = bisection(f, 0.0, 10.0, 100);
        assert!(result.is_err());
    }
}
