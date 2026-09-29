//! Minimum Orbital Intersection Distance (MOID) between two-body orbits.
//!
//! The MOID is the smallest distance between a point on one orbit and a point
//! on another orbit.
//!
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

use crate::constants::GMS;
use crate::errors::Error;
use crate::frames::{InertialFrame, SunCenter};
use crate::prelude::KeteResult;
use crate::state::State;
use nalgebra::{Complex, DMatrix, Matrix3x2, Matrix6, Schur, Vector3};
use std::f64::consts::TAU;

/// Compute the MOID between the two-body orbits of two states, in au.
///
/// Each Sun centered state defines a two-body orbit about the Sun. Bound and
/// unbound orbits are accepted.
///
/// # Errors
///
/// - [`Error::ValueError`] if the angular momentum of either state is zero or
///   not finite.
/// - [`Error::Convergence`] if an eigenvalue solve does not converge, or if no
///   critical point is found.
pub fn moid<T: InertialFrame>(
    state_a: &State<T, SunCenter>,
    state_b: &State<T, SunCenter>,
) -> KeteResult<f64> {
    let orbit_a = Conic::from_state(state_a)?;
    let orbit_b = Conic::from_state(state_b)?;
    // Method of Gronchi, Bau & Grassi (2023), "Revisiting the computation of the
    // critical points of the Keplerian distance". Every candidate is a distance
    // between two points on the orbits, so none is smaller than the MOID. A root
    // near aphelion of an orbit with eccentricity near 1 is poorly determined
    // when that orbit's anomaly is the one kept, so the elimination runs both
    // ways.
    let best = critical_min(&orbit_a, &orbit_b)?.min(critical_min(&orbit_b, &orbit_a)?);
    if best.is_finite() {
        Ok(best)
    } else {
        Err(Error::Convergence(
            "MOID: no critical point of the distance was found.".into(),
        ))
    }
}

/// A two-body orbit as a fixed conic about the Sun.
///
/// The point at true anomaly `f` is `r (cos f p_hat + sin f q_hat)`, with
/// `r = p / (1 + e cos f)`.
#[derive(Debug, Clone)]
struct Conic {
    /// Semi-latus rectum in au.
    p: f64,

    /// Eccentricity.
    e: f64,

    /// Unit vector toward perihelion.
    p_hat: Vector3<f64>,

    /// Unit vector in the orbit plane, 90 degrees ahead of `p_hat`.
    q_hat: Vector3<f64>,
}

impl Conic {
    /// Compute the conic of the two-body orbit through a Sun centered state.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the angular momentum is zero or not finite.
    fn from_state<T: InertialFrame>(state: &State<T, SunCenter>) -> KeteResult<Self> {
        let pos = Vector3::from(state.pos);
        let vel = Vector3::from(state.vel);
        let h = pos.cross(&vel);
        let h_norm = h.norm();
        if !h_norm.is_normal() {
            return Err(Error::ValueError(
                "MOID requires a state with nonzero angular momentum.".into(),
            ));
        }
        let h_hat = h / h_norm;
        // Remove the out-of-plane rounding error of the eccentricity vector.
        // Without this, the axes of a near circular orbit can leave the plane.
        let e_vec = vel.cross(&h) / GMS - pos.normalize();
        let e_vec = e_vec - e_vec.dot(&h_hat) * h_hat;
        let e = e_vec.norm();
        // A circular orbit has no perihelion. Any direction in the plane is
        // valid.
        let p_hat = if e > 0.0 { e_vec / e } else { pos.normalize() };
        Ok(Self {
            p: h_norm * h_norm / GMS,
            e,
            p_hat,
            q_hat: h_hat.cross(&p_hat),
        })
    }

    /// Return true if the true anomaly `f` is on the orbit.
    ///
    /// For a hyperbola, an anomaly past an asymptote is on the other branch.
    fn is_valid(&self, f: f64) -> bool {
        1.0 + self.e * f.cos() > 0.0
    }

    /// Return the position at true anomaly `f`, and its derivative with respect
    /// to `f`.
    fn position(&self, f: f64) -> (Vector3<f64>, Vector3<f64>) {
        let (sin_f, cos_f) = f.sin_cos();
        let den = 1.0 + self.e * cos_f;
        let pos = self.p / den * (cos_f * self.p_hat + sin_f * self.q_hat);
        let vel = self.p / (den * den) * (-sin_f * self.p_hat + (cos_f + self.e) * self.q_hat);
        (pos, vel)
    }
}

/// The critical point conditions of the squared distance at true anomaly `f2`
/// of orbit `b`, as polynomials in the true anomaly `f1` of orbit `a`.
///
/// Let `(c1, s1) = (cos f1, sin f1)`, with denominators cleared. The derivative
/// with respect to `f2` gives the line `alpha c1 + beta s1 + gamma = 0`. The
/// derivative with respect to `f1` gives the conic
/// `k_cc c1^2 + k_cs c1 s1 + k_c c1 + k_s s1 + k_0 = 0`. At a critical point,
/// `f1` satisfies both conditions.
struct Conditions {
    /// `1 + e2 cos f2`.
    den: f64,

    /// `[alpha, beta, gamma]`.
    line: [f64; 3],

    /// `[k_cc, k_cs, k_c, k_s, k_0]`.
    conic: [f64; 5],
}

impl Conditions {
    /// Compute the conditions at true anomaly `f2` of orbit `b`.
    fn new(a: &Conic, b: &Conic, f2: f64) -> Self {
        let (sin_f2, cos_f2) = f2.sin_cos();
        let den = 1.0 + b.e * cos_f2;

        // Derivative with respect to f2, times (1 + e1 c1) (1 + e2 c2).
        let t2 = -sin_f2 * b.p_hat + (cos_f2 + b.e) * b.q_hat;
        let line = [
            a.p * t2.dot(&a.p_hat) * den - a.e * b.p * b.e * sin_f2,
            a.p * t2.dot(&a.q_hat) * den,
            -b.p * b.e * sin_f2,
        ];

        // Derivative with respect to f1, times (1 + e1 c1) (1 + e2 c2).
        let x2 = b.p * (cos_f2 * b.p_hat + sin_f2 * b.q_hat);
        let (x2_p, x2_q) = (x2.dot(&a.p_hat), x2.dot(&a.q_hat));
        let conic = [
            -a.e * x2_q,
            a.e * x2_p,
            -x2_q * (1.0 + a.e * a.e),
            a.p * a.e * den + x2_p,
            -a.e * x2_q,
        ];
        Self { den, line, conic }
    }

    /// Return the eliminant of the two conditions.
    ///
    /// The eliminant is zero when the two conditions share a root `f1`. In
    /// `t = tan(f1 / 2)`, the line is a quadratic and the conic is a quartic.
    /// Their resultant is the determinant of their Sylvester matrix. The
    /// resultant has a factor `(1 + e2 cos f2)^2`. Division by this factor
    /// leaves a trigonometric polynomial in `f2` of degree 8.
    fn eliminant(&self) -> f64 {
        let [alpha, beta, gamma] = self.line;
        let [k_cc, k_cs, k_c, k_s, k_0] = self.conic;
        // Substitute c1 = (1 - t^2) / (1 + t^2) and s1 = 2t / (1 + t^2).
        // Multiply the line by (1 + t^2) and the conic by (1 + t^2)^2.
        // Coefficients start at the highest power of t.
        let line = [gamma - alpha, 2.0 * beta, gamma + alpha];
        let quartic = [
            k_cc - k_c + k_0,
            2.0 * (k_s - k_cs),
            2.0 * (k_0 - k_cc),
            2.0 * (k_s + k_cs),
            k_cc + k_c + k_0,
        ];
        let mut sylvester = Matrix6::<f64>::zeros();
        for row in 0..4 {
            for (col, &coef) in line.iter().enumerate() {
                sylvester[(row, row + col)] = coef;
            }
        }
        for row in 0..2 {
            for (col, &coef) in quartic.iter().enumerate() {
                sylvester[(4 + row, row + col)] = coef;
            }
        }
        sylvester.determinant() / (self.den * self.den)
    }

    /// Return the true anomalies `f1` where either condition is zero.
    ///
    /// A critical point satisfies both conditions. One condition can be zero
    /// for every `f1`, as the line is at the node of two orbits in
    /// perpendicular planes. This function therefore returns the roots of both
    /// conditions.
    ///
    /// # Errors
    ///
    /// [`Error::Convergence`] if an eigenvalue solve does not converge.
    fn anomalies(&self) -> KeteResult<Vec<f64>> {
        let [alpha, beta, gamma] = self.line;
        let [k_cc, k_cs, k_c, k_s, k_0] = self.conic;
        // Substitute c1 = (z + 1/z) / 2 and s1 = (z - 1/z) / 2i. Multiply the
        // line by 2z and the conic by 4z^2. Coefficients start at the lowest
        // power of z.
        let i = Complex::i();
        let line = [alpha + i * beta, (2.0 * gamma).into(), alpha - i * beta];
        let quartic = [
            k_cc + i * k_cs,
            2.0 * (k_c + i * k_s),
            (2.0 * k_cc + 4.0 * k_0).into(),
            2.0 * (k_c - i * k_s),
            k_cc - i * k_cs,
        ];
        let mut roots = unit_circle_roots(&line)?;
        roots.extend(unit_circle_roots(&quartic)?);
        Ok(roots)
    }
}

/// Degree of the eliminant as a trigonometric polynomial.
const DEGREE: i32 = 8;

/// Number of samples of the eliminant. It is more than `2 DEGREE + 1`, so the
/// Fourier coefficients do not alias.
const N_SAMPLES: u32 = 32;

/// Return the smallest distance over the critical points found by eliminating
/// `f1`, the true anomaly of orbit `a`.
///
/// The result is infinite if there is no candidate.
///
/// # Errors
///
/// [`Error::Convergence`] if an eigenvalue solve does not converge.
fn critical_min(a: &Conic, b: &Conic) -> KeteResult<f64> {
    // The eliminant is divided by (1 + e2 cos f2)^2. This factor is zero at the
    // asymptotes of a hyperbola and smallest at aphelion of an ellipse, at
    // +- edge. The samples are symmetric about perihelion. The offset keeps
    // every sample at least a quarter spacing from +- edge.
    let spacing = TAU / f64::from(N_SAMPLES);
    let edge = (-1.0 / b.e).max(-1.0).acos();
    let offset = if (0.25..0.75).contains(&(edge / spacing).fract()) {
        0.0
    } else {
        0.5
    };
    let samples: Vec<f64> = (0..N_SAMPLES)
        .map(|j| (f64::from(j) + offset) * spacing)
        .collect();
    let values: Vec<f64> = samples
        .iter()
        .map(|&f2| Conditions::new(a, b, f2).eliminant())
        .collect();

    // Fourier coefficients c_k, k = -DEGREE ..= DEGREE, of the eliminant. The
    // eliminant times z^DEGREE, with z = exp(i f2), is the polynomial with these
    // coefficients.
    let coef: Vec<Complex<f64>> = (-DEGREE..=DEGREE)
        .map(|k| {
            samples
                .iter()
                .zip(&values)
                .map(|(&f2, &value)| Complex::from_polar(value, -f64::from(k) * f2))
                .sum::<Complex<f64>>()
                / f64::from(N_SAMPLES)
        })
        .collect();

    let roots = if values.iter().any(|&x| x != 0.0) {
        unit_circle_roots(&coef)?
    } else {
        // For two identical orbits or two coplanar circles, the eliminant is
        // zero for every f2. Every sample is then a root.
        samples
    };

    let mut best = f64::INFINITY;
    for f2 in roots {
        if !b.is_valid(f2) {
            continue;
        }
        for f1 in Conditions::new(a, b, f2).anomalies()? {
            if a.is_valid(f1) {
                best = best.min(polish_critical_point(a, b, f1, f2));
            }
        }
    }
    Ok(best)
}

/// Return the arguments of the roots of `sum_j coef[j] z^j` near the unit
/// circle.
///
/// A root on the unit circle is a real zero of a trigonometric polynomial in
/// `z = exp(i f)`. This function keeps roots up to 10% off the unit circle. It
/// drops leading coefficients below `1e-13` times the largest coefficient.
/// These coefficients correspond to roots near infinity.
///
/// # Errors
///
/// [`Error::Convergence`] if the eigenvalue solve does not converge.
fn unit_circle_roots(coef: &[Complex<f64>]) -> KeteResult<Vec<f64>> {
    let max_coef = coef.iter().map(|x| x.norm()).fold(0.0, f64::max);
    let Some(n) = coef.iter().rposition(|x| x.norm() > 1e-13 * max_coef) else {
        return Ok(Vec::new());
    };
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut companion = DMatrix::<Complex<f64>>::zeros(n, n);
    for j in 0..n {
        companion[(0, j)] = -coef[n - 1 - j] / coef[n];
    }
    for j in 1..n {
        companion[(j, j - 1)] = Complex::new(1.0, 0.0);
    }
    let schur = Schur::try_new(companion, f64::EPSILON, 10_000).ok_or_else(|| {
        Error::Convergence("MOID: eigenvalues of the companion matrix did not converge.".into())
    })?;
    // The complex Schur form is upper triangular. Its diagonal holds the roots.
    Ok(schur
        .unpack()
        .1
        .diagonal()
        .iter()
        .filter(|z| (z.norm() - 1.0).abs() < 0.1)
        .map(|z| z.arg())
        .collect())
}

/// Refine a candidate critical point with Gauss-Newton steps, and return the
/// smallest distance seen.
///
/// Each step solves the linearized `x1 + dx1 d1 = x2 + dx2 d2` by least
/// squares. The fixed points of the step are the critical points of the
/// distance. The loop runs at most 10 steps. It stops early if the normal
/// equations are singular, or if a step leaves either orbit. Every distance
/// seen is between two points on the orbits, so the result is not smaller than
/// the MOID.
fn polish_critical_point(a: &Conic, b: &Conic, mut f1: f64, mut f2: f64) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..10 {
        let (x1, dx1) = a.position(f1);
        let (x2, dx2) = b.position(f2);
        let diff = x1 - x2;
        best = best.min(diff.norm());

        let jac = Matrix3x2::from_columns(&[dx1, -dx2]);
        let Some(step) = (jac.transpose() * jac)
            .lu()
            .solve(&(-jac.transpose() * diff))
        else {
            break;
        };
        f1 += step[0];
        f2 += step[1];
        // A step that is not finite also fails this check.
        if !a.is_valid(f1) || !b.is_valid(f2) {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::desigs::Desig;
    use crate::frames::Ecliptic;

    /// Return a Sun centered state at true anomaly `f` (degrees).
    ///
    /// The elements are the perihelion distance `q` (au), the eccentricity `e`,
    /// and the inclination, argument of perihelion and longitude of the
    /// ascending node (degrees).
    fn orbit_state(elements: [f64; 5], f: f64) -> State<Ecliptic, SunCenter> {
        let [q, e, inc, peri, node] = elements;
        let (inc, peri, node, f) = (
            inc.to_radians(),
            peri.to_radians(),
            node.to_radians(),
            f.to_radians(),
        );
        let (sw, cw) = peri.sin_cos();
        let (sn, cn) = node.sin_cos();
        let (si, ci) = inc.sin_cos();
        let p_hat = Vector3::new(cn * cw - sn * sw * ci, sn * cw + cn * sw * ci, sw * si);
        let q_hat = Vector3::new(-cn * sw - sn * cw * ci, -sn * sw + cn * cw * ci, cw * si);
        let p = q * (1.0 + e);
        let (sf, cf) = f.sin_cos();
        let pos = p / (1.0 + e * cf) * (cf * p_hat + sf * q_hat);
        let vel = (GMS / p).sqrt() * (-sf * p_hat + (e + cf) * q_hat);
        State::new(Desig::Empty, 2_460_000.5, pos, vel, SunCenter)
    }

    /// Compare against an independent reference for orbit pairs and a
    /// hyperbola. In these pairs the MOID is not at the closest pair of points
    /// on a coarse time grid. The reference is a dense grid in eccentric or
    /// true anomaly, with each grid local minimum refined, computed with scipy.
    #[test]
    fn moid_matches_reference() {
        let earth = [0.983, 0.0167, 0.0, 102.9, 0.0];
        for (a, b, expected) in [
            (
                [0.637, 0.454, 0.157, 247.068, 311.529],
                earth,
                0.000_566_564_171_126_785_8,
            ),
            (
                [0.629, 0.642, 15.521, 67.985, 97.504],
                earth,
                0.071_972_520_267_504_35,
            ),
            (
                [2.203, 0.951, 43.067, 76.282, 291.512],
                [1.159, 0.691, 22.587, 79.723, 343.526],
                0.458_758_972_904_498_13,
            ),
            // Hyperbola with an asymptote at a multiple of the sample spacing.
            // The sample offset keeps every sample away from it.
            (
                [0.5, std::f64::consts::SQRT_2, 20.0, 30.0, 40.0],
                [1.0, 0.1, 5.0, 10.0, 20.0],
                0.227_202_387_302_625_46,
            ),
        ] {
            let got = moid(&orbit_state(a, 10.0), &orbit_state(b, 250.0)).unwrap();
            assert!((got - expected).abs() < 1e-10, "{a:?}: {got} vs {expected}");
        }
    }

    /// Cases with known MOIDs: orbits through a shared point, the same orbit,
    /// and circles, where the MOID is the difference of the radii.
    #[test]
    fn moid_known_values() {
        // Two states at one position with different velocities, elliptic and
        // hyperbolic, have orbits that cross there.
        let pos = Vector3::new(0.8, -0.4, 0.1);
        for vel in [
            Vector3::new(0.004, 0.015, 0.002),
            Vector3::new(-0.01, 0.02, -0.005),
            Vector3::new(0.03, 0.01, 0.0),
        ] {
            let a = State::new(
                Desig::Empty,
                2_460_000.5,
                pos,
                Vector3::new(0.01, 0.012, -0.003),
                SunCenter,
            );
            let b: State<Ecliptic, SunCenter> =
                State::new(Desig::Empty, 2_460_000.5, pos, vel, SunCenter);
            let got = moid(&a, &b).unwrap();
            assert!(got < 1e-10, "crossing orbits: {got}");
        }

        // Crossing near aphelion of an e = 0.98 orbit. The eliminant is small
        // there, and its root is poorly determined.
        let pos = Vector3::new(
            6.195_375_517_760_655,
            11.673_989_847_903_838,
            12.795_580_873_996_078,
        );
        let a = State::new(
            Desig::Empty,
            2_460_000.5,
            pos,
            Vector3::new(
                -0.000_116_107_770_752_080_67,
                -0.000_961_148_197_817_903_2,
                -0.001_757_454_623_812_356_7,
            ),
            SunCenter,
        );
        let b: State<Ecliptic, SunCenter> = State::new(
            Desig::Empty,
            2_460_000.5,
            pos,
            Vector3::new(
                -0.001_015_368_901_420_766_8,
                -0.001_692_721_946_739_638_6,
                -0.002_500_610_442_175_163_6,
            ),
            SunCenter,
        );
        let got = moid(&a, &b).unwrap();
        assert!(got < 1e-10, "crossing near aphelion: {got}");

        let orbit = [0.983, 0.0167, 0.0, 102.9, 0.0];
        let got = moid(&orbit_state(orbit, 0.0), &orbit_state(orbit, 120.0)).unwrap();
        assert!(got < 1e-10, "same orbit: {got}");

        // At 90 degrees one condition is zero for every f1 at the node. The
        // anomalies then come from the other condition.
        for inc in [0.0, 20.0, 90.0, 150.0] {
            let got = moid(
                &orbit_state([1.0, 0.0, 0.0, 0.0, 0.0], 0.0),
                &orbit_state([1.5, 0.0, inc, 40.0, 10.0], 0.0),
            )
            .unwrap();
            assert!((got - 0.5).abs() < 1e-12, "circles at {inc} deg: {got}");
        }
    }

    #[test]
    fn moid_radial_state_is_error() {
        let radial = State::new(
            Desig::Empty,
            2_460_000.5,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.01, 0.0, 0.0),
            SunCenter,
        );
        let other = orbit_state([1.0, 0.1, 5.0, 0.0, 0.0], 0.0);
        assert!(moid(&radial, &other).is_err());
    }
}
