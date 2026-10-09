// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Gravity field as a spherical harmonic expansion.
//!
//! ```text
//! U(r) = GM / R  sum_{n=0}^{N} sum_{m=0}^{n}  ( C_nm V_nm + S_nm W_nm )
//!
//! V_nm = (R / r)^(n+1) P_nm(sin phi) cos(m lambda)
//! W_nm = (R / r)^(n+1) P_nm(sin phi) sin(m lambda)
//! ```
//!
//! in the body frame, with `phi` the latitude and `lambda` the longitude of the field
//! point, `R` the reference radius, and `P_nm` the associated Legendre functions
//! without the Condon-Shortley phase. `U` is positive and the acceleration is
//! `grad U`. The coefficients are fully normalized (the geodesy convention of
//! published fields): `C_nm = N_nm Cbar_nm` with
//!
//! ```text
//! N_nm = sqrt( (2 - delta_m0) (2n + 1) (n - m)! / (n + m)! )
//! ```
//!
//! so a J2 of the usual unnormalized sign convention is `Cbar_20 = -J2 / sqrt(5)`.
//! Degree 0 is the point mass (`Cbar_00 = 1` for the full mass) and degree 1 is an
//! offset of the center of mass from the origin.
//!
//! `V_nm` and `W_nm` come from the Cunningham recursions (Montenbruck and Gill,
//! "Satellite Orbits", 2000, section 3.2.4), rewritten for the normalized functions
//! `N_nm V_nm` and `N_nm W_nm`, which are bounded by `sqrt(2 (2n + 1)) (R / r)^(n+1)`
//! with no factorial growth at high degree. The partial derivatives of each `V_nm`,
//! `W_nm` are combinations of those of degree `n + 1` (Montenbruck and Gill
//! eq. 3.33), so the acceleration and its gradient are themselves series of degree
//! `N + 1` and `N + 2`; their coefficients are computed once, at construction, and
//! an evaluation is one pass of the recursion and dot products. There is no
//! singularity at the poles.
//!
//! The coefficients of a constant-density polyhedron are exact volume integrals of
//! the interior solid harmonics `N_nm (r / R)^n P_nm(sin phi) (cos, sin)(m lambda)`,
//! homogeneous polynomials of degree `n`:
//!
//! ```text
//! (Cbar_nm, Sbar_nm) = 1 / ((2n + 1) V)  integral_V  N_nm (r/R)^n P_nm (cos, sin)(m lambda) dV
//! ```
//!
//! By the divergence theorem, for `f` homogeneous of degree `n`,
//! `integral_V f dV = 1 / (n + 3) sum_faces h_f integral_face f dA`, with `h_f` the
//! signed distance of the face's plane from the origin; each face integral is exact
//! with a collapsed Gauss-Legendre rule of enough points.
//!
//! The series converges only outside the smallest sphere about the origin enclosing
//! all of the body's mass (the Brillouin sphere). Inside it the sum can diverge
//! without any sign of it in the result. A field built with a minimum radius (the
//! Brillouin radius, or larger) returns an error for any position inside it; one
//! built without cannot tell, and errors only at the origin.

use nalgebra::{Matrix3, Vector3};

use super::Polyhedron;
use crate::errors::{Error, KeteResult};
use crate::util::gauss_legendre;

/// A gravity field given by fully normalized spherical harmonic coefficients.
///
/// Positions are relative to the expansion origin, in the body frame and in the
/// units of `radius`; `gm` sets the units of the results. See the module
/// documentation for the conventions and the region of validity.
#[derive(Debug, Clone)]
pub struct SphericalHarmonics {
    /// Gravitational parameter.
    gm: f64,

    /// Reference radius.
    radius: f64,

    /// Evaluations closer to the origin than this are errors.
    min_radius: Option<f64>,

    /// The potential, `radius / gm U`.
    potential: Series,

    /// The acceleration along x, y, z, `radius^2 / gm accel_i`.
    accel: [Series; 3],

    /// The gradient entries of `GRAD_PAIRS`, `radius^3 / gm grad_ij`.
    grad: [Series; 6],

    /// Recursion factors of the basis, to the degree of `grad`.
    recursion: Recursion,
}

/// Number of parts the faces of a polyhedron are summed in, in order.
const FACE_PARTS: usize = 64;

/// The gradient entries held in `SphericalHarmonics::grad`, in order.
const GRAD_PAIRS: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

impl SphericalHarmonics {
    /// Build a field from normalized coefficients.
    ///
    /// `c[n]` and `s[n]` hold degree `n`, `n + 1` values each for orders `0..=n`; the
    /// largest degree is `c.len() - 1`. `min_radius`, in the units of `radius`, is
    /// the distance from the origin inside which an evaluation is an error: the
    /// Brillouin radius of the body, or larger.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `gm`, `radius` or a given `min_radius` is not positive
    /// and finite, if the
    /// coefficient tables are empty, do not match in shape, or are not triangular, if
    /// any coefficient is not finite, or if an `S_n0` is not zero (it multiplies
    /// `sin(0 lambda)` and has no meaning).
    pub fn new(
        gm: f64,
        radius: f64,
        c: &[Vec<f64>],
        s: &[Vec<f64>],
        min_radius: Option<f64>,
    ) -> KeteResult<Self> {
        if !gm.is_finite() || gm <= 0.0 {
            return Err(Error::ValueError(format!(
                "Spherical harmonic GM must be positive and finite, found {gm}."
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::ValueError(format!(
                "Spherical harmonic reference radius must be positive and finite, found {radius}."
            )));
        }
        if let Some(min) = min_radius
            && (!min.is_finite() || min <= 0.0)
        {
            return Err(Error::ValueError(format!(
                "Spherical harmonic minimum radius must be positive and finite, found {min}."
            )));
        }
        if c.is_empty() || c.len() != s.len() {
            return Err(Error::ValueError(format!(
                "Spherical harmonic C and S tables must be non-empty and of the same degree, \
                 found {} and {} rows.",
                c.len(),
                s.len()
            )));
        }
        for (n, (c_row, s_row)) in c.iter().zip(s).enumerate() {
            if c_row.len() != n + 1 || s_row.len() != n + 1 {
                return Err(Error::ValueError(format!(
                    "Spherical harmonic degree {n} must have {} coefficients in C and in S, \
                     found {} and {}.",
                    n + 1,
                    c_row.len(),
                    s_row.len()
                )));
            }
            if c_row.iter().chain(s_row).any(|v| !v.is_finite()) {
                return Err(Error::ValueError(format!(
                    "Spherical harmonic coefficients of degree {n} must be finite."
                )));
            }
            if s_row[0] != 0.0 {
                return Err(Error::ValueError(format!(
                    "Spherical harmonic S_{n}0 must be zero, found {}.",
                    s_row[0]
                )));
            }
        }
        let potential = Series {
            degree: c.len() - 1,
            c: c.concat(),
            s: s.concat(),
        };
        // The acceleration and its gradient are series too, built here once; an
        // evaluation is then one pass of the basis recursion and dot products.
        let accel = [0, 1, 2].map(|axis| potential.derivative(axis));
        let grad = GRAD_PAIRS.map(|(i, j)| accel[i].derivative(j));
        let recursion = Recursion::new(potential.degree + 2);
        Ok(Self {
            gm,
            radius,
            min_radius,
            potential,
            accel,
            grad,
            recursion,
        })
    }

    /// The exact field of a constant-density polyhedron to `degree`, expanded about
    /// the mesh origin. The reference radius and the minimum radius are both the
    /// mesh's bounding radius about the origin, its Brillouin radius. See the module
    /// documentation for the integrals.
    ///
    /// # Errors
    ///
    /// As [`Self::new`], which cannot fail for a valid polyhedron.
    pub fn from_polyhedron(poly: &Polyhedron, degree: usize) -> KeteResult<Self> {
        let mesh = poly.mesh();
        let radius = mesh.bounding_radius();
        let recursion = Recursion::new(degree);
        let nodes = gauss_legendre(usize::midpoint(degree, 3));
        let size = index(degree, degree) + 1;
        let add_faces = |faces: &[[u32; 3]], (c_sum, s_sum): &mut (Vec<f64>, Vec<f64>)| {
            for face in faces {
                let [a, b, c] = mesh.face_vertices(face);
                let (edge1, edge2) = (b - a, c - a);
                // h_f times twice the face area
                let h_area2 = a.dot(&edge1.cross(&edge2));
                // the face as a + u ((1 - v) edge1 + v edge2), dA = 2 A u du dv
                for &(u, wu) in &nodes {
                    for &(v, wv) in &nodes {
                        let point = a + ((1.0 - v) * edge1 + v * edge2) * u;
                        let weight = wu * wv * u * h_area2;
                        let basis = Basis::interior(&point, radius, degree, &recursion);
                        for (sum, value) in c_sum.iter_mut().zip(&basis.v) {
                            *sum += weight * value;
                        }
                        for (sum, value) in s_sum.iter_mut().zip(&basis.w) {
                            *sum += weight * value;
                        }
                    }
                }
            }
        };
        // The faces are summed in FACE_PARTS fixed parts, added in order, so the
        // result does not depend on the number of threads. Scoped threads rather than
        // rayon: this runs in the polyhedron's far-field initializer, and a rayon
        // worker waiting on its own tasks can run another that re-enters it.
        let faces = mesh.faces();
        let part_len = faces.len().div_ceil(FACE_PARTS).max(1);
        let mut parts = vec![(vec![0.0; size], vec![0.0; size]); faces.len().div_ceil(part_len)];
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        let parts_per_thread = parts.len().div_ceil(threads).max(1);
        std::thread::scope(|scope| {
            for (group, sums) in faces
                .chunks(part_len * parts_per_thread)
                .zip(parts.chunks_mut(parts_per_thread))
            {
                // joined, and any panic raised, at the end of the scope
                let _ = scope.spawn(|| {
                    for (part, sum) in group.chunks(part_len).zip(sums) {
                        add_faces(part, sum);
                    }
                });
            }
        });
        let (mut c_sum, mut s_sum) = (vec![0.0; size], vec![0.0; size]);
        for (c_part, s_part) in &parts {
            for (sum, value) in c_sum.iter_mut().zip(c_part) {
                *sum += value;
            }
            for (sum, value) in s_sum.iter_mut().zip(s_part) {
                *sum += value;
            }
        }
        let volume = mesh.volume();
        let row = |sums: &[f64], n: usize| -> Vec<f64> {
            let scale = 1.0 / ((2 * n + 1) as f64 * (n + 3) as f64 * volume);
            sums[index(n, 0)..=index(n, n)]
                .iter()
                .map(|x| x * scale)
                .collect()
        };
        let c: Vec<Vec<f64>> = (0..=degree).map(|n| row(&c_sum, n)).collect();
        let mut s: Vec<Vec<f64>> = (0..=degree).map(|n| row(&s_sum, n)).collect();
        // W_n0 is zero, so its integral is too; clear the rounding
        for row in &mut s {
            row[0] = 0.0;
        }
        Self::new(poly.gm(), radius, &c, &s, Some(radius))
    }

    /// The same field cut at `degree` (at most the current degree).
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if `degree` exceeds the current degree.
    pub fn truncated(&self, degree: usize) -> KeteResult<Self> {
        if degree > self.degree() {
            return Err(Error::ValueError(format!(
                "Cannot truncate a degree {} field to degree {degree}.",
                self.degree()
            )));
        }
        let rows = |flat: &[f64]| -> Vec<Vec<f64>> {
            (0..=degree)
                .map(|n| flat[index(n, 0)..=index(n, n)].to_vec())
                .collect()
        };
        Self::new(
            self.gm,
            self.radius,
            &rows(&self.potential.c),
            &rows(&self.potential.s),
            self.min_radius,
        )
    }

    /// Gravitational parameter.
    #[must_use]
    pub fn gm(&self) -> f64 {
        self.gm
    }

    /// Reference radius.
    #[must_use]
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Distance from the origin inside which an evaluation is an error, if set.
    #[must_use]
    pub fn min_radius(&self) -> Option<f64> {
        self.min_radius
    }

    /// Largest degree.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.potential.degree
    }

    /// Normalized `(C_nm, S_nm)`, or `None` beyond the degree or for `m > n`.
    #[must_use]
    pub fn coefficient(&self, n: usize, m: usize) -> Option<(f64, f64)> {
        let k = index(n, m);
        (n <= self.degree() && m <= n).then(|| (self.potential.c[k], self.potential.s[k]))
    }

    /// Gravitational potential at a position, positive, so that the acceleration is
    /// its gradient.
    ///
    /// # Errors
    ///
    /// [`Error::ValueError`] if the position is not finite or is the origin, and
    /// [`Error::Bounds`] if it is inside the minimum radius.
    pub fn potential(&self, pos: &Vector3<f64>) -> KeteResult<f64> {
        self.check(pos)?;
        let basis = Basis::new(pos, self.radius, self.degree(), &self.recursion);
        Ok(basis.dot(&self.potential) * self.gm / self.radius)
    }

    /// Acceleration at a position.
    ///
    /// # Errors
    ///
    /// As [`Self::potential`].
    pub fn field(&self, pos: &Vector3<f64>) -> KeteResult<Vector3<f64>> {
        self.check(pos)?;
        let basis = Basis::new(pos, self.radius, self.degree() + 1, &self.recursion);
        let scale = self.gm / (self.radius * self.radius);
        Ok(Vector3::from_fn(|i, _| basis.dot(&self.accel[i]) * scale))
    }

    /// Acceleration and its gradient with respect to position (`grad[i][j]` is
    /// `d accel_i / d pos_j`), at a position.
    ///
    /// # Errors
    ///
    /// As [`Self::potential`].
    pub fn field_and_gradient(
        &self,
        pos: &Vector3<f64>,
    ) -> KeteResult<(Vector3<f64>, Matrix3<f64>)> {
        self.check(pos)?;
        let basis = Basis::new(pos, self.radius, self.degree() + 2, &self.recursion);
        let scale = self.gm / (self.radius * self.radius);
        let accel = Vector3::from_fn(|i, _| basis.dot(&self.accel[i]) * scale);
        let mut grad = Matrix3::zeros();
        for (&(i, j), series) in GRAD_PAIRS.iter().zip(&self.grad) {
            grad[(i, j)] = basis.dot(series) * scale / self.radius;
            grad[(j, i)] = grad[(i, j)];
        }
        Ok((accel, grad))
    }

    /// An error for a position where the series cannot be evaluated.
    fn check(&self, pos: &Vector3<f64>) -> KeteResult<()> {
        let r = pos.norm();
        if !r.is_finite() || r == 0.0 {
            return Err(Error::ValueError(format!(
                "Spherical harmonic field evaluated at {pos:?}, which is not finite or is the origin."
            )));
        }
        // Bounds: the position is outside the region where the series is defined.
        if let Some(min) = self.min_radius
            && r < min
        {
            return Err(Error::Bounds(format!(
                "Spherical harmonic field evaluated at distance {r} from the origin, inside its \
                 minimum radius {min}, where the series is not valid."
            )));
        }
        Ok(())
    }
}

/// Position of degree `n`, order `m` in the triangular tables.
fn index(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// `a! / b!` for arguments that differ by a few.
fn factorial_ratio(a: usize, b: usize) -> f64 {
    if a >= b {
        ((b + 1)..=a).map(|k| k as f64).product()
    } else {
        1.0 / ((a + 1)..=b).map(|k| k as f64).product::<f64>()
    }
}

/// `N_nm / N_n2m2`, the ratio of the normalization factors.
fn norm_ratio(n: usize, m: usize, n2: usize, m2: usize) -> f64 {
    let e = |m: usize| if m == 0 { 1.0 } else { 2.0 };
    (e(m) / e(m2) * (2 * n + 1) as f64 / (2 * n2 + 1) as f64
        * factorial_ratio(n - m, n2 - m2)
        * factorial_ratio(n2 + m2, n + m))
    .sqrt()
}

/// A series `sum C_nm V_nm + S_nm W_nm` of the normalized basis functions, to a
/// degree, as triangular tables at `index(n, m)`.
#[derive(Debug, Clone)]
struct Series {
    degree: usize,
    c: Vec<f64>,
    s: Vec<f64>,
}

impl Series {
    fn zeros(degree: usize) -> Self {
        let size = index(degree, degree) + 1;
        Self {
            degree,
            c: vec![0.0; size],
            s: vec![0.0; size],
        }
    }

    /// Add to the `(n, m)` coefficients; the sine part is dropped at `m = 0`, where
    /// `W_n0` is zero.
    fn add(&mut self, n: usize, m: usize, c: f64, s: f64) {
        self.c[index(n, m)] += c;
        if m > 0 {
            self.s[index(n, m)] += s;
        }
    }

    /// `radius` times the derivative of the series along `axis` (0, 1, 2 for x, y,
    /// z), a series one degree higher: Montenbruck and Gill eq. 3.33 for the
    /// unnormalized functions, with the ratios of the normalizations.
    fn derivative(&self, axis: usize) -> Self {
        let mut out = Self::zeros(self.degree + 1);
        for n in 0..=self.degree {
            let n1 = n + 1;
            for m in 0..=n {
                let (c, s) = (self.c[index(n, m)], self.s[index(n, m)]);
                if axis == 2 {
                    // d X_nm / dz = -(n - m + 1) X_{n+1,m}, X = V, W
                    let t = -((n - m + 1) as f64) * norm_ratio(n, m, n1, m);
                    out.add(n1, m, t * c, t * s);
                } else if m == 0 {
                    // d V_n0 / dx = -V_{n+1,1}, d V_n0 / dy = -W_{n+1,1}
                    let t = -norm_ratio(n, 0, n1, 1);
                    if axis == 0 {
                        out.add(n1, 1, t * c, 0.0);
                    } else {
                        out.add(n1, 1, 0.0, t * c);
                    }
                } else {
                    let up = 0.5 * norm_ratio(n, m, n1, m + 1);
                    let down =
                        0.5 * ((n - m + 2) * (n - m + 1)) as f64 * norm_ratio(n, m, n1, m - 1);
                    if axis == 0 {
                        // d X_nm / dx = -up X_{n+1,m+1} + down X_{n+1,m-1}, X = V, W
                        out.add(n1, m + 1, -up * c, -up * s);
                        out.add(n1, m - 1, down * c, down * s);
                    } else {
                        // d V_nm / dy = -up W_{n+1,m+1} - down W_{n+1,m-1}
                        // d W_nm / dy =  up V_{n+1,m+1} + down V_{n+1,m-1}
                        out.add(n1, m + 1, up * s, -up * c);
                        out.add(n1, m - 1, down * s, -down * c);
                    }
                }
            }
        }
        out
    }
}

/// Factors of the normalized recursions, which depend only on degree and order.
#[derive(Debug, Clone)]
struct Recursion {
    /// Sectoral: `V_mm = sectoral[m] (x V_{m-1,m-1} - y W_{m-1,m-1})`, positions scaled
    /// by `radius / r^2`.
    sectoral: Vec<f64>,

    /// Vertical, at `index(n, m)`: `V_nm = first z V_{n-1,m} - second rho V_{n-2,m}`.
    first: Vec<f64>,

    /// See `first`.
    second: Vec<f64>,
}

impl Recursion {
    fn new(degree: usize) -> Self {
        let size = index(degree, degree) + 1;
        let mut first = vec![0.0; size];
        let mut second = vec![0.0; size];
        let sectoral = (0..=degree)
            .map(|m| {
                if m == 0 {
                    0.0
                } else {
                    (2 * m - 1) as f64 * norm_ratio(m, m, m - 1, m - 1)
                }
            })
            .collect();
        for m in 0..=degree {
            for n in (m + 1)..=degree {
                let k = index(n, m);
                first[k] = (2 * n - 1) as f64 / (n - m) as f64 * norm_ratio(n, m, n - 1, m);
                if n >= m + 2 {
                    second[k] = (n + m - 1) as f64 / (n - m) as f64 * norm_ratio(n, m, n - 2, m);
                }
            }
        }
        Self {
            sectoral,
            first,
            second,
        }
    }
}

/// The normalized `V_nm`, `W_nm` at one position, to some degree.
#[derive(Debug)]
struct Basis {
    /// Normalized `V_nm`, at `index(n, m)`.
    v: Vec<f64>,

    /// Normalized `W_nm`, at `index(n, m)`.
    w: Vec<f64>,
}

impl Basis {
    /// The exterior functions `N_nm (R / r)^(n+1) P_nm (cos, sin)(m lambda)`.
    fn new(pos: &Vector3<f64>, radius: f64, degree: usize, rec: &Recursion) -> Self {
        let r2 = pos.norm_squared();
        Self::recurse(
            pos * (radius / r2),
            radius * radius / r2,
            radius / r2.sqrt(),
            degree,
            rec,
        )
    }

    /// The interior functions `N_nm (r / R)^n P_nm (cos, sin)(m lambda)`, which follow
    /// the same recursions from the position scaled by `1 / R`.
    fn interior(pos: &Vector3<f64>, radius: f64, degree: usize, rec: &Recursion) -> Self {
        Self::recurse(
            pos / radius,
            pos.norm_squared() / (radius * radius),
            1.0,
            degree,
            rec,
        )
    }

    /// The recursions from `V_00 = start`, with the scaled position and `rho` the
    /// factor of the second vertical term.
    fn recurse(scaled: Vector3<f64>, rho: f64, start: f64, degree: usize, rec: &Recursion) -> Self {
        let size = index(degree, degree) + 1;
        let mut v = vec![0.0; size];
        let mut w = vec![0.0; size];
        v[0] = start;
        for m in 0..=degree {
            if m > 0 {
                let prev = index(m - 1, m - 1);
                let (v_prev, w_prev) = (v[prev], w[prev]);
                let f = rec.sectoral[m];
                v[index(m, m)] = f * (scaled.x * v_prev - scaled.y * w_prev);
                w[index(m, m)] = f * (scaled.x * w_prev + scaled.y * v_prev);
            }
            if m < degree {
                let k = index(m + 1, m);
                let first = rec.first[k] * scaled.z;
                v[k] = first * v[index(m, m)];
                w[k] = first * w[index(m, m)];
            }
            for n in (m + 2)..=degree {
                let k = index(n, m);
                let first = rec.first[k] * scaled.z;
                let second = rec.second[k] * rho;
                v[k] = first * v[index(n - 1, m)] - second * v[index(n - 2, m)];
                w[k] = first * w[index(n - 1, m)] - second * w[index(n - 2, m)];
            }
        }
        Self { v, w }
    }

    /// The value of a series of at most this degree.
    fn dot(&self, series: &Series) -> f64 {
        series
            .c
            .iter()
            .zip(&series.s)
            .zip(self.v.iter().zip(&self.w))
            .map(|((c, s), (v, w))| c * v + s * w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::Polyhedron;
    use crate::geometry::TriMesh;
    use nalgebra::Rotation3;
    use std::f64::consts::PI;

    /// Deterministic coefficients to `degree`, decaying like a real field.
    fn test_field(degree: usize, gm: f64, radius: f64) -> SphericalHarmonics {
        let mut c = Vec::new();
        let mut s = Vec::new();
        for n in 0..=degree {
            let scale = 0.3 / (1.0 + n as f64).powi(2);
            c.push(
                (0..=n)
                    .map(|m| {
                        if n == 0 {
                            1.0
                        } else {
                            scale * ((n * 7 + m * 3) as f64).sin()
                        }
                    })
                    .collect(),
            );
            s.push(
                (0..=n)
                    .map(|m| {
                        if m == 0 {
                            0.0
                        } else {
                            scale * ((n * 5 + m * 11) as f64).cos()
                        }
                    })
                    .collect(),
            );
        }
        SphericalHarmonics::new(gm, radius, &c, &s, None).unwrap()
    }

    fn points() -> Vec<Vector3<f64>> {
        vec![
            Vector3::new(2.1, -0.7, 1.3),
            Vector3::new(-0.4, 1.9, -2.2),
            Vector3::new(0.3, 0.2, 3.0),
            Vector3::new(-2.5, -1.0, 0.1),
            // on the pole, where latitude and longitude parameterizations are singular
            Vector3::new(0.0, 0.0, 2.4),
        ]
    }

    #[test]
    fn degree_zero_is_the_point_mass() {
        let gm = 3.7;
        let field = SphericalHarmonics::new(gm, 1.3, &[vec![1.0]], &[vec![0.0]], None).unwrap();
        for p in points() {
            let r = p.norm();
            assert!((field.potential(&p).unwrap() - gm / r).abs() < 1e-14 * gm / r);
            let (a, g) = field.field_and_gradient(&p).unwrap();
            let a_pm = -p * gm / r.powi(3);
            let g_pm = (p * p.transpose() * 3.0 - Matrix3::identity() * r * r) * (gm / r.powi(5));
            assert!((a - a_pm).norm() < 1e-14 * a_pm.norm());
            assert!((field.field(&p).unwrap() - a_pm).norm() < 1e-14 * a_pm.norm());
            assert!((g - g_pm).norm() < 1e-14 * g_pm.norm());
        }
    }

    /// The potential against the associated Legendre functions written out to degree 3.
    #[test]
    fn potential_matches_explicit_legendre_functions() {
        let (gm, radius) = (2.0, 1.5);
        let field = test_field(3, gm, radius);
        let legendre = |n: usize, m: usize, t: f64, c: f64| match (n, m) {
            (0, 0) => 1.0,
            (1, 0) => t,
            (1, 1) => c,
            (2, 0) => 1.5 * t * t - 0.5,
            (2, 1) => 3.0 * t * c,
            (2, 2) => 3.0 * c * c,
            (3, 0) => 2.5 * t.powi(3) - 1.5 * t,
            (3, 1) => 1.5 * c * (5.0 * t * t - 1.0),
            (3, 2) => 15.0 * t * c * c,
            (3, 3) => 15.0 * c.powi(3),
            _ => unreachable!(),
        };
        let fact = |k: usize| (1..=k).map(|i| i as f64).product::<f64>();
        for p in points() {
            let r = p.norm();
            let (t, c) = (p.z / r, p.x.hypot(p.y) / r);
            let lon = p.y.atan2(p.x);
            let mut total = 0.0;
            for n in 0..=3 {
                for m in 0..=n {
                    let norm =
                        ((if m == 0 { 1.0 } else { 2.0 }) * (2 * n + 1) as f64 * fact(n - m)
                            / fact(n + m))
                        .sqrt();
                    let (cnm, snm) = field.coefficient(n, m).unwrap();
                    total += (radius / r).powi(i32::try_from(n).unwrap() + 1)
                        * norm
                        * legendre(n, m, t, c)
                        * (cnm * (m as f64 * lon).cos() + snm * (m as f64 * lon).sin());
                }
            }
            let expected = total * gm / radius;
            assert!(
                (field.potential(&p).unwrap() - expected).abs() < 1e-13 * expected.abs(),
                "{p:?}: {} vs {expected}",
                field.potential(&p).unwrap()
            );
        }
    }

    /// J2 alone against the oblateness term the propagator already uses.
    #[test]
    fn j2_matches_the_oblate_term() {
        let (gm, radius, j2) = (4.0, 1.1, 1.2e-3);
        let c = [
            vec![1.0],
            vec![0.0, 0.0],
            vec![-j2 / 5_f64.sqrt(), 0.0, 0.0],
        ];
        let s = [vec![0.0], vec![0.0, 0.0], vec![0.0; 3]];
        let field = SphericalHarmonics::new(gm, radius, &c, &s, None).unwrap();
        let pole = Vector3::z();
        for p in points() {
            let expected = super::super::gravity::j2_correction(&p, &pole, radius, j2, gm);
            let got = field.field(&p).unwrap() + p * gm / p.norm().powi(3);
            assert!(
                (got - expected).norm() < 1e-12 * expected.norm().max(1e-30),
                "{p:?}: {got:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn field_is_the_gradient_of_the_potential() {
        let field = test_field(8, 1.0, 1.0);
        let h = 1e-6;
        for p in points() {
            let a = field.field(&p).unwrap();
            for axis in 0..3 {
                let mut dp = Vector3::zeros();
                dp[axis] = h;
                let fd = (field.potential(&(p + dp)).unwrap()
                    - field.potential(&(p - dp)).unwrap())
                    / (2.0 * h);
                assert!(
                    (a[axis] - fd).abs() < 1e-8 * a.norm(),
                    "{p:?} axis {axis}: {} vs {fd}",
                    a[axis]
                );
            }
        }
    }

    #[test]
    fn gradient_matches_finite_difference_and_laplace() {
        let field = test_field(8, 1.0, 1.0);
        let h = 1e-6;
        for p in points() {
            let (a, g) = field.field_and_gradient(&p).unwrap();
            assert!((a - field.field(&p).unwrap()).norm() < 1e-14 * a.norm());
            for axis in 0..3 {
                let mut dp = Vector3::zeros();
                dp[axis] = h;
                let fd =
                    (field.field(&(p + dp)).unwrap() - field.field(&(p - dp)).unwrap()) / (2.0 * h);
                assert!(
                    (g.column(axis) - fd).norm() < 1e-7 * g.norm(),
                    "{p:?} column {axis}: {:?} vs {fd:?}",
                    g.column(axis)
                );
            }
            // outside the mass the potential is harmonic
            assert!(g.trace().abs() < 1e-12 * g.norm(), "trace {}", g.trace());
        }
    }

    /// A sectoral term turns with the body: rotating the field point by 90 degrees about
    /// the pole flips the sign of a degree 2, order 2 potential.
    #[test]
    fn sectoral_term_turns_with_longitude() {
        let c = [vec![0.0], vec![0.0, 0.0], vec![0.0, 0.0, 0.4]];
        let s = [vec![0.0], vec![0.0, 0.0], vec![0.0, 0.0, 0.0]];
        let field = SphericalHarmonics::new(1.0, 1.0, &c, &s, None).unwrap();
        let turn = Rotation3::from_axis_angle(&Vector3::z_axis(), PI / 2.0);
        for p in points() {
            let u = field.potential(&p).unwrap();
            assert!(
                (field.potential(&(turn * p)).unwrap() + u).abs() < 1e-14 * u.abs().max(1e-300)
            );
        }
    }

    /// High degree stays finite and consistent: the normalized recursion does not
    /// overflow where the unnormalized functions would.
    #[test]
    fn high_degree_is_stable() {
        let field = test_field(200, 1.0, 1.0);
        let p = Vector3::new(0.6, -0.5, 0.75).normalize() * 1.02; // just outside r = 1
        let a = field.field(&p).unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        let h = 1e-7;
        for axis in 0..3 {
            let mut dp = Vector3::zeros();
            dp[axis] = h;
            let fd = (field.potential(&(p + dp)).unwrap() - field.potential(&(p - dp)).unwrap())
                / (2.0 * h);
            assert!(
                (a[axis] - fd).abs() < 1e-6 * a.norm(),
                "axis {axis}: {} vs {fd}",
                a[axis]
            );
        }
    }

    /// Box `[x0, x1] x [y0, y1] x [z0, z1]` as a constant-density polyhedron.
    fn box_polyhedron(lo: [f64; 3], hi: [f64; 3]) -> Polyhedron {
        let vertices = (0..8)
            .map(|i: u32| {
                Vector3::new(
                    if i & 1 == 0 { lo[0] } else { hi[0] },
                    if (i >> 1) & 1 == 0 { lo[1] } else { hi[1] },
                    if (i >> 2) & 1 == 0 { lo[2] } else { hi[2] },
                )
            })
            .collect();
        let faces = vec![
            [0, 2, 1],
            [1, 2, 3],
            [4, 5, 6],
            [5, 7, 6],
            [0, 1, 4],
            [1, 5, 4],
            [2, 6, 3],
            [3, 6, 7],
            [0, 4, 2],
            [2, 4, 6],
            [1, 3, 5],
            [3, 7, 5],
        ];
        Polyhedron::new(TriMesh::new(vertices, faces).unwrap(), 1.0)
            .unwrap()
            .without_far_field()
    }

    /// Degrees 0-2 of a box against closed forms: the mass, the centroid (degree 1),
    /// and the second moments `<x^2> = a^2 / 3` of a centered box of half sizes a, b, c.
    #[test]
    fn box_coefficients_match_closed_forms() {
        let (hx, hy, hz) = (0.9, 0.6, 0.4);
        let center = Vector3::new(0.3, -0.2, 0.15);
        let lo = [center.x - hx, center.y - hy, center.z - hz];
        let hi = [center.x + hx, center.y + hy, center.z + hz];
        let field = SphericalHarmonics::from_polyhedron(&box_polyhedron(lo, hi), 2).unwrap();
        let radius = field.radius();
        let (c00, _) = field.coefficient(0, 0).unwrap();
        let (c10, _) = field.coefficient(1, 0).unwrap();
        let (c11, s11) = field.coefficient(1, 1).unwrap();
        assert!((c00 - 1.0).abs() < 1e-14);
        let deg1 = 3_f64.sqrt() * radius;
        assert!((c10 - center.z / deg1).abs() < 1e-14);
        assert!((c11 - center.x / deg1).abs() < 1e-14);
        assert!((s11 - center.y / deg1).abs() < 1e-14);

        let centered =
            SphericalHarmonics::from_polyhedron(&box_polyhedron([-hx, -hy, -hz], [hx, hy, hz]), 2)
                .unwrap();
        let r2 = centered.radius().powi(2);
        let c20 = (hz * hz / 3.0 - (hx * hx + hy * hy) / 6.0) / r2 / 5_f64.sqrt();
        let c22 = (hx * hx - hy * hy) / (12.0 * r2) / (5.0 / 12.0_f64).sqrt();
        let (got20, _) = centered.coefficient(2, 0).unwrap();
        let (got22, s22) = centered.coefficient(2, 2).unwrap();
        assert!((got20 - c20).abs() < 1e-14, "{got20} vs {c20}");
        assert!((got22 - c22).abs() < 1e-14, "{got22} vs {c22}");
        assert!(s22.abs() < 1e-15);
    }

    /// The exact coefficients of an off-center box reproduce its field outside the
    /// Brillouin sphere, better with degree and with distance.
    #[test]
    fn polyhedron_coefficients_reproduce_its_field() {
        let poly = box_polyhedron([-0.8, -0.5, -0.3], [1.2, 0.9, 0.7]);
        let full = SphericalHarmonics::from_polyhedron(&poly, 24).unwrap();
        assert_eq!(full.min_radius(), Some(poly.mesh().bounding_radius()));
        let error = |field: &SphericalHarmonics, r: f64| {
            points()
                .iter()
                .map(|p| {
                    let q = p.normalize() * r;
                    let (expected, _) = poly.field(&q);
                    (field.field(&q).unwrap() - expected).norm() / expected.norm()
                })
                .fold(0.0, f64::max)
        };
        let rb = poly.mesh().bounding_radius();
        let (e12, e24) = (
            error(&full.truncated(12).unwrap(), 3.0 * rb),
            error(&full, 3.0 * rb),
        );
        println!("relative acceleration error at 3 R_B: degree 12 {e12:e}, degree 24 {e24:e}");
        assert!(e24 < 1e-11, "degree 24 at 3 R_B: {e24:e}");
        assert!(e24 < e12);
        assert!(error(&full.truncated(12).unwrap(), 6.0 * rb) < e12);
    }

    /// The addition theorem, exact at every degree: at `r = R`,
    /// `sum_m V_nm^2 + W_nm^2 = 2n + 1` for the fully normalized functions.
    #[test]
    fn normalization_holds_at_every_degree() {
        let degree = 400;
        let rec = Recursion::new(degree);
        for p in [
            Vector3::new(0.3, -0.8, 0.52),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.2, -0.01),
        ] {
            let p = p.normalize() * 1.7;
            let basis = Basis::new(&p, 1.7, degree, &rec);
            for n in 0..=degree {
                let sum: f64 = (0..=n)
                    .map(|m| basis.v[index(n, m)].powi(2) + basis.w[index(n, m)].powi(2))
                    .sum();
                let err = (sum / (2 * n + 1) as f64 - 1.0).abs();
                assert!(err < 1e-10, "{p:?} degree {n}: relative error {err:e}");
            }
        }
    }

    /// Inside the minimum radius, at the origin and at a non-finite position every
    /// evaluation is an error; just outside it is not.
    #[test]
    fn inside_the_minimum_radius_is_an_error() {
        let (c, s) = ([vec![1.0], vec![0.0, 0.0]], [vec![0.0], vec![0.0, 0.0]]);
        let field = SphericalHarmonics::new(1.0, 1.0, &c, &s, Some(1.5)).unwrap();
        assert_eq!(field.min_radius(), Some(1.5));
        let inside = Vector3::new(0.0, 1.4, 0.3);
        assert!(field.potential(&inside).is_err());
        assert!(field.field(&inside).is_err());
        assert!(field.field_and_gradient(&inside).is_err());
        // Bounds: outside the region where the series is defined.
        assert!(matches!(field.field(&inside), Err(Error::Bounds(_))));
        assert!(field.field(&Vector3::new(0.0, 1.5, 0.1)).is_ok());
        let unbounded = SphericalHarmonics::new(1.0, 1.0, &c, &s, None).unwrap();
        assert!(unbounded.field(&Vector3::zeros()).is_err());
        assert!(unbounded.field(&Vector3::new(f64::NAN, 0.0, 1.0)).is_err());
        assert!(unbounded.field(&inside).is_ok());
    }

    #[test]
    fn rejects_bad_input() {
        let one = [vec![1.0]];
        let zero = [vec![0.0]];
        assert!(SphericalHarmonics::new(0.0, 1.0, &one, &zero, None).is_err());
        assert!(SphericalHarmonics::new(1.0, f64::NAN, &one, &zero, None).is_err());
        assert!(SphericalHarmonics::new(1.0, 1.0, &[], &[], None).is_err());
        assert!(
            SphericalHarmonics::new(
                1.0,
                1.0,
                &[vec![1.0], vec![0.0]],
                &[vec![0.0], vec![0.0]],
                None
            )
            .is_err()
        );
        assert!(SphericalHarmonics::new(1.0, 1.0, &one, &[vec![0.5]], None).is_err());
        assert!(SphericalHarmonics::new(1.0, 1.0, &[vec![f64::INFINITY]], &zero, None).is_err());
        assert!(SphericalHarmonics::new(1.0, 1.0, &one, &zero, Some(0.0)).is_err());
        assert!(SphericalHarmonics::new(1.0, 1.0, &one, &zero, Some(f64::NAN)).is_err());
    }
}
