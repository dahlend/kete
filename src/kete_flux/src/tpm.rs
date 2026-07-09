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

//! Thermophysical model (TPM) with thermal inertia.
//!
//! Unlike NEATM and FRM, which assume instantaneous radiative equilibrium, the TPM
//! solves the 1D heat conduction equation into the subsurface, giving the surface a
//! thermal memory governed by the thermal inertia `Gamma`. The night side stays
//! warm and the diurnal temperature peak lags local noon.
//!
//! The default geometry is a smooth sphere. Non-dimensionalizing the heat equation
//! (depth in thermal skin depths, time in rotation phase, temperature in the
//! sub-solar equilibrium temperature `T_ss`) reduces the problem to
//!
//! ```text
//!   du/dtau = d2u/dx2
//!   surface (x = 0):   Theta * du/dx = u^4 - mu(tau)
//!   base   (x = x_max): du/dx = 0
//! ```
//!
//! which depends only on the thermal parameter `Theta = Gamma * sqrt(omega) /
//! (eps * sigma * T_ss^3)` and the insolation `mu(tau)`. The surface temperature
//! depends only on the facet-normal latitude (relative to the spin axis) and local
//! solar time, so the heat equation is solved once per latitude band -- independent
//! of body shape, since a convex body casts no self-shadows. The shape only enters
//! when rendering the disk-integrated flux ([`TpmShape`]): the default is a sphere,
//! but oblate spheroids and triaxial ellipsoids (or custom convex meshes) are
//! supported. A non-axisymmetric (triaxial) body produces a rotational thermal
//! lightcurve, so its flux depends on the rotation phase at the observation epoch.

use crate::crater::Crater;
use crate::{
    BandInfo, ColorCorrFn, ConvexShape, DEFAULT_SHAPE, Facet, ModelResults, TriangleShape,
    assemble_total, black_body_flux, bond_albedo, lambertian_vis_scale_factor,
    sub_solar_temperature,
};
use kete_core::constants::{AU_KM, STEFAN_BOLTZMANN};
use kete_core::errors::{Error, KeteResult};
use nalgebra::{DMatrix, DVector, UnitVector3, Vector3};
use rayon::prelude::*;
use std::f64::consts::{FRAC_PI_2, PI, TAU};

/// Spin state of the body.
///
/// The rotation phase at a time `epoch` (Julian date) is
/// `phase0 + 2*pi * (epoch - epoch0) * 86400 / period`. The phase only affects
/// non-axisymmetric shapes (a triaxial ellipsoid or a custom mesh); for a sphere or
/// an oblate spheroid the result is independent of phase.
#[derive(Debug, Clone)]
pub struct SpinState {
    /// Spin axis (ecliptic), rotation is right-handed about this vector. A reversed
    /// pole gives retrograde rotation, which sets the sign of the thermal lag.
    pub pole: UnitVector3<f64>,
    /// Rotation period in seconds.
    pub period: f64,
    /// Rotation phase (radians) at `epoch0`.
    pub phase0: f64,
    /// Reference epoch (Julian date) at which the phase is `phase0`.
    pub epoch0: f64,
}

impl SpinState {
    /// Rotation phase (radians) at the given epoch (Julian date).
    #[must_use]
    pub fn rotation_phase(&self, epoch: f64) -> f64 {
        if self.period == 0.0 {
            return self.phase0;
        }
        self.phase0 + TAU * (epoch - self.epoch0) * 86_400.0 / self.period
    }
}

/// Number of divisions (pole to equator) for the default ellipsoid mesh.
/// Facet count is `8 * n_div^2`.
const ELLIPSOID_DIV: u32 = 24;

/// A body-fixed convex shape used by the TPM, stored as facet normals and areas.
///
/// The body z-axis is the spin axis; `ellipsoid(a, b, c)` scales the x/y/z axes, so
/// an oblate spheroid is `ellipsoid(1, 1, c)` with `c < 1` and a triaxial ellipsoid
/// has `a != b`. Total facet area is normalized to 1.
#[derive(Debug, Clone)]
pub struct TpmShape {
    facets: Vec<Facet>,
    /// True when the shape is rotationally symmetric about the pole (sphere or oblate
    /// spheroid); the rotation phase then has no effect.
    axisymmetric: bool,
}

impl TpmShape {
    /// The default unit sphere (a well-distributed Fibonacci lattice).
    #[must_use]
    pub fn sphere() -> Self {
        Self {
            facets: DEFAULT_SHAPE.facets.to_vec(),
            axisymmetric: true,
        }
    }

    /// A unit sphere with an explicit facet count (a Fibonacci lattice). A coarse
    /// version is used as the cheap forward model for the fitting seed search.
    #[must_use]
    pub fn sphere_with_facets(n_facets: u32) -> Self {
        Self {
            facets: ConvexShape::new_fibonacci_lattice(n_facets).facets.to_vec(),
            axisymmetric: true,
        }
    }

    /// A triaxial ellipsoid with semi-axes `(a, b, c)` along body x/y/z (z = pole).
    #[must_use]
    pub fn ellipsoid(a: f64, b: f64, c: f64) -> Self {
        Self::ellipsoid_with_div(ELLIPSOID_DIV, a, b, c)
    }

    /// An ellipsoid with an explicit mesh resolution (facet count `8 * n_div^2`). A
    /// coarse version is used as the cheap forward model for the fitting seed search.
    #[must_use]
    pub fn ellipsoid_with_div(n_div: u32, a: f64, b: f64, c: f64) -> Self {
        let mesh = TriangleShape::new_ellipsoid(n_div, a, b, c);
        let facets = mesh
            .facets
            .iter()
            .map(|f| Facet {
                normal: f.normal,
                area: f.area,
            })
            .collect();
        Self {
            facets,
            axisymmetric: (a - b).abs() < 1e-9,
        }
    }

    /// Build a shape from explicit body-fixed facets (assumed convex). Not treated as
    /// axisymmetric.
    #[must_use]
    pub fn from_facets(facets: Vec<Facet>) -> Self {
        Self {
            facets,
            axisymmetric: false,
        }
    }
}

impl Default for TpmShape {
    fn default() -> Self {
        Self::sphere()
    }
}

/// A reusable ellipsoid tessellation: the fixed triangle topology of a unit sphere
/// stored as reference vertices, so a [`TpmShape`] for any axis ratios is produced by
/// scaling vertices and recomputing normals/areas -- skipping the `O(n_div^2)`
/// re-tessellation that [`TpmShape::ellipsoid`] pays on every call. Built once and
/// reused across fit steps where the axis ratios are a free parameter.
#[derive(Debug, Clone)]
pub struct EllipsoidTemplate {
    /// Per-facet unit-sphere vertices; the connectivity is baked in here.
    tris: Vec<[Vector3<f64>; 3]>,
}

impl EllipsoidTemplate {
    /// Build the template at the given mesh resolution (facet count `8 * n_div^2`).
    #[must_use]
    pub fn new(n_div: u32) -> Self {
        let mesh = TriangleShape::new_ellipsoid(n_div, 1.0, 1.0, 1.0);
        let tris = mesh.facets.iter().map(|f| f.vertices).collect();
        Self { tris }
    }

    /// Produce a [`TpmShape`] for semi-axes `(a, b, c)` along body x/y/z (z = pole).
    ///
    /// Each facet normal is recomputed from the cross product of the scaled edges -- an
    /// anisotropic scaling does *not* scale the normal, so the reference normal cannot
    /// just be rescaled -- and the total area is renormalized to 1. The result is
    /// facet-for-facet identical to [`TpmShape::ellipsoid_with_div`] at the same
    /// resolution; only cheaper. `axisymmetric` is set when `a == b`.
    #[must_use]
    pub fn shape(&self, a: f64, b: f64, c: f64) -> TpmShape {
        let scale = |v: &Vector3<f64>| Vector3::new(v.x * a, v.y * b, v.z * c);
        let mut facets: Vec<Facet> = Vec::with_capacity(self.tris.len());
        let mut total = 0.0;
        for tri in &self.tris {
            let v0 = scale(&tri[0]);
            let v1 = scale(&tri[1]);
            let v2 = scale(&tri[2]);
            let mut cross = (v1 - v0).cross(&(v2 - v0));
            // Orient outward (away from the origin), matching `TriangleFacet::new`,
            // which uses the first vertex to fix the sign.
            if v0.dot(&cross) < 0.0 {
                cross = -cross;
            }
            let area = cross.norm() / 2.0;
            total += area;
            facets.push(Facet {
                normal: UnitVector3::new_normalize(cross),
                area,
            });
        }
        let inv = total.recip();
        for f in &mut facets {
            f.area *= inv;
        }
        TpmShape {
            facets,
            axisymmetric: (a - b).abs() < 1e-9,
        }
    }
}

/// Thermal surface properties.
#[derive(Debug, Clone)]
pub struct ThermalParams {
    /// Thermal inertia `Gamma` in SI units (J m^-2 K^-1 s^-1/2).
    pub thermal_inertia: f64,
    /// Emissivity of the surface.
    pub emissivity: f64,
}

impl ThermalParams {
    /// Compute the thermal parameter `Theta = Gamma * sqrt(omega) / (eps * sigma *
    /// T_ss^3)` for a rotation `period` (seconds) and sub-solar temperature (Kelvin).
    #[must_use]
    pub fn thermal_parameter(&self, period: f64, sub_solar_temp: f64) -> f64 {
        let omega = TAU / period;
        self.thermal_inertia * omega.sqrt()
            / (self.emissivity * STEFAN_BOLTZMANN * sub_solar_temp.powi(3))
    }
}

/// Number of subsurface depth nodes. Fewer than a uniform grid needs because the
/// non-uniform grid clusters them near the surface (see [`DepthGrid`]).
const N_DEPTH: usize = 35;
/// Depth of the column in thermal skin depths. ~8 skin depths fully damps the
/// diurnal wave (exp(-8) ~ 3e-4).
const X_MAX: f64 = 8.0;
/// Exponential stretching factor for the non-uniform depth grid: nodes cluster near
/// the surface, where the diurnal gradient is steep.
const DEPTH_STRETCH: f64 = 4.0;
/// Number of time samples per rotation.
const N_TIME: usize = 360;
/// Number of latitude bands the heat equation is solved on.
const N_LAT: usize = 37;
/// Maximum rotations to integrate while seeking periodic steady state.
const MAX_ROTATIONS: usize = 500;
/// Convergence tolerance on the dimensionless surface temperature between rotations.
const CONV_TOL: f64 = 1e-5;
/// Maximum Newton iterations on the nonlinear radiative surface boundary per step
/// (stops early once the surface iterate converges; see `NEWTON_TOL`).
const N_SURFACE_ITERS: usize = 3;
/// Surface-Newton convergence tolerance (well below the steady-state `CONV_TOL`).
const NEWTON_TOL: f64 = 1e-6;
/// History window for Anderson acceleration of the periodic steady-state iteration.
const ANDERSON_M: usize = 5;
/// Default crater discretization for the roughness model (latitude rings).
const CRATER_RINGS: usize = 8;
/// Default azimuthal sectors per crater ring.
const CRATER_SECTORS: usize = 16;

/// Non-uniform subsurface depth grid and its Crank-Nicolson coefficients.
///
/// Nodes are exponentially clustered near the surface (where the diurnal temperature
/// gradient is steep), so the same accuracy needs fewer nodes than a uniform grid.
/// The grid and the interior/boundary coefficients are dimensionless and independent
/// of `Theta`, so they are precomputed once. Only the surface-flux factor
/// `k = 2 h0 / Theta` depends on `Theta` and is passed per solve.
struct DepthGrid {
    /// Number of time samples per rotation this grid's `dtau` was built for. The
    /// solver reads its sample count from here so a coarser grid (e.g. the offline
    /// roughness-table build) can be used without touching the runtime default.
    n_time: usize,
    /// Near-surface spacing `x[1] - x[0]` (used for `k = 2 h0 / Theta`).
    h0: f64,
    /// Surface-node diffusion number `dtau / (2 h0^2)`.
    r_surf: f64,
    /// Insulating-base diffusion number `dtau / h_last^2`.
    r_base: f64,
    /// Interior CN sub/diag/super coefficients `0.5*dtau*(a_i, b_i, c_i)` from the
    /// non-uniform 3-point Laplacian (indices `1..N_DEPTH-1`).
    alpha: Vec<f64>,
    beta: Vec<f64>,
    gamma: Vec<f64>,
}

impl DepthGrid {
    /// Build the depth grid for the default time resolution (`N_TIME`).
    fn new() -> Self {
        Self::with_n_time(N_TIME)
    }

    /// Build the depth grid whose Crank-Nicolson coefficients bake in `dtau =
    /// TAU / n_time`.
    fn with_n_time(n_time: usize) -> Self {
        let nz = N_DEPTH;
        let dtau = TAU / n_time as f64;
        let denom = DEPTH_STRETCH.exp() - 1.0;
        let x: Vec<f64> = (0..nz)
            .map(|i| {
                let t = i as f64 / (nz - 1) as f64;
                X_MAX * ((DEPTH_STRETCH * t).exp() - 1.0) / denom
            })
            .collect();

        let h0 = x[1] - x[0];
        let h_last = x[nz - 1] - x[nz - 2];
        let mut alpha = vec![0.0; nz];
        let mut beta = vec![0.0; nz];
        let mut gamma = vec![0.0; nz];
        for i in 1..nz - 1 {
            let hm = x[i] - x[i - 1];
            let hp = x[i + 1] - x[i];
            let a = 2.0 / (hm * (hm + hp));
            let c = 2.0 / (hp * (hm + hp));
            alpha[i] = 0.5 * dtau * a;
            gamma[i] = 0.5 * dtau * c;
            beta[i] = -0.5 * dtau * (a + c);
        }
        Self {
            n_time,
            h0,
            r_surf: dtau / (2.0 * h0 * h0),
            r_base: dtau / (h_last * h_last),
            alpha,
            beta,
            gamma,
        }
    }
}

/// The shared non-uniform depth grid (built once).
static DEPTH_GRID: std::sync::LazyLock<DepthGrid> = std::sync::LazyLock::new(DepthGrid::new);

/// Precomputed dimensionless surface temperature field over a sphere.
///
/// `surface[k][j]` is the dimensionless temperature `u = T / T_ss` at latitude band
/// `k` (evenly spaced from -pi/2 to pi/2) and local solar time
/// `tau_j = TAU * j / N_TIME`.
#[derive(Debug)]
struct DiurnalField {
    surface: Vec<Vec<f64>>,
}

impl DiurnalField {
    /// Solve the heat equation on each latitude band for a given thermal parameter
    /// and sub-solar latitude.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Convergence`] if any latitude band fails to converge.
    fn solve(theta: f64, sub_solar_lat: f64, grid: &DepthGrid) -> KeteResult<Self> {
        // Each latitude band is an independent heat solve. with_min_len keeps the
        // granularity coarse so batch (per-object) parallelism is not oversplit.
        let surface = (0..N_LAT)
            .into_par_iter()
            .with_min_len(4)
            .map(|k| {
                let lat = -FRAC_PI_2 + PI * k as f64 / (N_LAT - 1) as f64;
                solve_diurnal_surface(theta, lat, sub_solar_lat, grid)
            })
            .collect::<KeteResult<Vec<_>>>()?;
        Ok(Self { surface })
    }

    /// Bilinearly sample the dimensionless surface temperature at a latitude and
    /// local solar time, wrapping in time.
    fn sample(&self, lat: f64, tau: f64) -> f64 {
        self.sample_at(&self.sample_index(lat, tau))
    }

    /// Compute the bilinear (latitude, local-time) bracket and weights for a query
    /// point. The result depends only on the grid dimensions, so it can be reused
    /// across the four `(Theta, sslat)` corner fields (see [`sample_corners`]) instead
    /// of recomputing the index math four times.
    #[allow(clippy::cast_sign_loss, reason = "indices are clamped non-negative")]
    fn sample_index(&self, lat: f64, tau: f64) -> SampleIdx {
        // latitude index (latitudes are evenly spaced from -pi/2 to pi/2)
        let fk = ((lat + FRAC_PI_2) / PI * (N_LAT - 1) as f64).clamp(0.0, (N_LAT - 1) as f64);
        let k0 = fk.floor() as usize;
        let k1 = (k0 + 1).min(N_LAT - 1);
        let wk = fk - k0 as f64;

        // time index (wraps around the rotation); the row length is the field's
        // time-sample count, which may differ from the runtime default.
        let n_time = self.surface[0].len();
        let tau = tau.rem_euclid(TAU);
        let fj = tau / TAU * n_time as f64;
        let j0 = (fj.floor() as usize) % n_time;
        let j1 = (j0 + 1) % n_time;
        let wj = fj - fj.floor();

        SampleIdx {
            k0,
            k1,
            wk,
            j0,
            j1,
            wj,
        }
    }

    /// Bilinearly read this field at a precomputed [`SampleIdx`].
    fn sample_at(&self, idx: &SampleIdx) -> f64 {
        let row0 = &self.surface[idx.k0];
        let row1 = &self.surface[idx.k1];
        let a = row0[idx.j0] * (1.0 - idx.wj) + row0[idx.j1] * idx.wj;
        let b = row1[idx.j0] * (1.0 - idx.wj) + row1[idx.j1] * idx.wj;
        a * (1.0 - idx.wk) + b * idx.wk
    }
}

/// Precomputed bilinear sampling bracket (latitude/local-time indices and weights),
/// shared across the four corner fields, which have identical dimensions.
struct SampleIdx {
    k0: usize,
    k1: usize,
    wk: f64,
    j0: usize,
    j1: usize,
    wj: f64,
}

/// Default number of `Theta` grid nodes (log-spaced).
const GRID_N_THETA: usize = 24;
/// Default number of sub-solar-latitude grid nodes (linear over [-pi/2, pi/2]).
const GRID_N_SSLAT: usize = 19;

/// Precomputed grid of diurnal fields over (`Theta`, sub-solar latitude).
///
/// The diurnal field is expensive to solve but depends only on these two
/// quantities, so for repeated evaluation (e.g. fitting) it is solved once on a grid
/// and interpolated. `Theta` is gridded in log space; the sub-solar latitude in
/// linear space over `[-pi/2, pi/2]`. Build once with [`TpmFieldGrid::new`], then
/// pass to [`tpm_thermal_flux_cached`].
#[derive(Debug)]
pub struct TpmFieldGrid {
    log_theta_min: f64,
    log_theta_max: f64,
    n_theta: usize,
    n_sslat: usize,
    /// Row-major over (`theta`, sub-solar lat): index `i * n_sslat + j`.
    fields: Vec<DiurnalField>,
}

impl TpmFieldGrid {
    /// Build a grid spanning `[theta_min, theta_max]` at the default resolution.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Convergence`] if any grid node fails to converge.
    ///
    /// # Panics
    ///
    /// Panics if `theta_min` is not positive or `theta_max <= theta_min`.
    pub fn new(theta_min: f64, theta_max: f64) -> KeteResult<Self> {
        Self::with_resolution(theta_min, theta_max, GRID_N_THETA, GRID_N_SSLAT)
    }

    /// Build a grid with an explicit node count in each dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Convergence`] if any grid node fails to converge.
    ///
    /// # Panics
    ///
    /// Panics if `theta_min` is not positive, `theta_max <= theta_min`, or either
    /// node count is below 2.
    pub fn with_resolution(
        theta_min: f64,
        theta_max: f64,
        n_theta: usize,
        n_sslat: usize,
    ) -> KeteResult<Self> {
        assert!(
            theta_min > 0.0 && theta_max > theta_min,
            "require 0 < theta_min < theta_max"
        );
        assert!(
            n_theta >= 2 && n_sslat >= 2,
            "need at least 2 nodes per axis"
        );

        let log_theta_min = theta_min.ln();
        let log_theta_max = theta_max.ln();

        // Parallelize over grid nodes (the large dimension); each node solves its
        // latitudes serially so the two levels of parallelism do not oversubscribe.
        let fields = (0..n_theta * n_sslat)
            .into_par_iter()
            .with_min_len(1)
            .map(|idx| -> KeteResult<DiurnalField> {
                let i = idx / n_sslat;
                let j = idx % n_sslat;
                let theta = (log_theta_min
                    + (log_theta_max - log_theta_min) * i as f64 / (n_theta - 1) as f64)
                    .exp();
                let sslat = -FRAC_PI_2 + PI * j as f64 / (n_sslat - 1) as f64;
                let surface = (0..N_LAT)
                    .map(|k| {
                        let lat = -FRAC_PI_2 + PI * k as f64 / (N_LAT - 1) as f64;
                        solve_diurnal_surface(theta, lat, sslat, &DEPTH_GRID)
                    })
                    .collect::<KeteResult<Vec<_>>>()?;
                Ok(DiurnalField { surface })
            })
            .collect::<KeteResult<Vec<_>>>()?;

        Ok(Self {
            log_theta_min,
            log_theta_max,
            n_theta,
            n_sslat,
            fields,
        })
    }

    /// The four `(Theta, sub-solar-lat)` corner fields bracketing the query point,
    /// returned with the two interpolation weights `(w_theta, w_sslat)`. `Theta` is
    /// clamped to the grid range (linear in log-`Theta`); the sub-solar latitude is
    /// clamped to `[-pi/2, pi/2]` (linear).
    ///
    /// Sampling these four fields on demand (see `sample_corners`) gives the same
    /// multilinear interpolation as materializing a full interpolated field, but
    /// without building the entire `N_LAT x N_TIME` grid or allocating per evaluation
    /// -- which dominates the fitting cost where the renderer only needs a few thousand
    /// facet samples.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "indices are clamped non-negative and bounded by the grid size"
    )]
    fn corner_fields(&self, theta: f64, sub_solar_lat: f64) -> ([&DiurnalField; 4], f64, f64) {
        let lt = theta.ln().clamp(self.log_theta_min, self.log_theta_max);
        let fi = (lt - self.log_theta_min) / (self.log_theta_max - self.log_theta_min)
            * (self.n_theta - 1) as f64;
        let i0 = fi.floor() as usize;
        let i1 = (i0 + 1).min(self.n_theta - 1);
        let wi = fi - i0 as f64;

        let sslat = sub_solar_lat.clamp(-FRAC_PI_2, FRAC_PI_2);
        let fj = (sslat + FRAC_PI_2) / PI * (self.n_sslat - 1) as f64;
        let j0 = fj.floor() as usize;
        let j1 = (j0 + 1).min(self.n_sslat - 1);
        let wj = fj - j0 as f64;

        (
            [
                &self.fields[i0 * self.n_sslat + j0],
                &self.fields[i0 * self.n_sslat + j1],
                &self.fields[i1 * self.n_sslat + j0],
                &self.fields[i1 * self.n_sslat + j1],
            ],
            wi,
            wj,
        )
    }
}

/// Bilinearly interpolate the dimensionless surface temperature over four
/// `(Theta, sslat)` corner fields at a given latitude and local solar time.
fn sample_corners(corners: &[&DiurnalField; 4], wi: f64, wj: f64, lat: f64, tau: f64) -> f64 {
    // The (lat, tau) bracket is identical for all four corners, so compute it once.
    let idx = corners[0].sample_index(lat, tau);
    let v0 = corners[0].sample_at(&idx);
    let v1 = corners[1].sample_at(&idx);
    let v2 = corners[2].sample_at(&idx);
    let v3 = corners[3].sample_at(&idx);
    let a = v0 * (1.0 - wj) + v1 * wj;
    let b = v2 * (1.0 - wj) + v3 * wj;
    a * (1.0 - wi) + b * wi
}

/// Solve the dimensionless 1D heat equation for one latitude band to periodic
/// steady state, returning the surface temperature `u(tau)` over one rotation.
///
/// Crank-Nicolson in depth with an insulating base; the nonlinear radiative surface
/// boundary `Theta * du/dx = u^4 - mu` is linearized about the current iterate
/// (`u^4 ~= 4 a^3 u - 3 a^4`) and iterated a few times per step (Newton).
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the periodic steady state is not reached within
/// [`MAX_ROTATIONS`].
fn solve_diurnal_surface(
    theta: f64,
    lat: f64,
    sub_solar_lat: f64,
    grid: &DepthGrid,
) -> KeteResult<Vec<f64>> {
    let nz = N_DEPTH;
    let n_time = grid.n_time;
    let dtau = TAU / n_time as f64;
    let k_flux = 2.0 * grid.h0 / theta;

    // insolation mu(tau) = max(cos(solar zenith), 0) over one rotation
    let (s_lat, c_lat) = lat.sin_cos();
    let (s_sun, c_sun) = sub_solar_lat.sin_cos();
    let mu: Vec<f64> = (0..n_time)
        .map(|j| {
            let tau = dtau * j as f64;
            (s_lat * s_sun + c_lat * c_sun * tau.cos()).max(0.0)
        })
        .collect();

    let mean_mu = mu.iter().sum::<f64>() / n_time as f64;

    // Degenerate limits handled in closed form so they cannot produce NaN:
    // Theta -> inf (e.g. zero period, an infinitely fast rotator) is isothermal at
    // the mean-insolation temperature; Theta = 0 (zero inertia) is instantaneous
    // radiative equilibrium u^4 = mu.
    if !theta.is_finite() {
        return Ok(vec![mean_mu.powf(0.25); n_time]);
    }
    if theta <= 0.0 {
        return Ok(mu.iter().map(|m| m.powf(0.25)).collect());
    }

    // initialize the column at the mean-insolation equilibrium which makes
    // high-inertia cases converge immediately.
    let u_init = mean_mu.powf(0.25);
    let mut u = vec![u_init; nz];

    let mut surface = vec![0.0; n_time];
    let mut prev_surface = vec![f64::INFINITY; n_time];

    let mut scratch = ColumnScratch::new(nz);
    let mut anderson = Anderson::new();
    let mut f = vec![0.0; nz];

    for _ in 0..MAX_ROTATIONS {
        // f = F(u): one rotation from the current profile, recording the surface curve.
        f.copy_from_slice(&u);
        rotate_column(&mut f, &mut scratch, &mu, grid, k_flux, Some(&mut surface));

        let diff = surface
            .iter()
            .zip(&prev_surface)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if diff < CONV_TOL {
            return Ok(surface);
        }
        prev_surface.copy_from_slice(&surface);

        // accelerate the profile toward the fixed point F(u) = u.
        anderson.step(&mut u, &f);
    }
    Err(Error::Convergence(format!(
        "TPM diurnal solve failed to reach periodic steady state within {MAX_ROTATIONS} \
         rotations (Theta={theta}, latitude={lat} rad)"
    )))
}

/// Reusable scratch buffers for advancing one subsurface column.
struct ColumnScratch {
    u_old: Vec<f64>,
    sub: Vec<f64>,
    diag: Vec<f64>,
    sup: Vec<f64>,
    rhs: Vec<f64>,
}

impl ColumnScratch {
    fn new(nz: usize) -> Self {
        Self {
            u_old: vec![0.0; nz],
            sub: vec![0.0; nz],
            diag: vec![0.0; nz],
            sup: vec![0.0; nz],
            rhs: vec![0.0; nz],
        }
    }
}

/// Advance one subsurface column by one time step: Crank-Nicolson in depth with an
/// insulating base, and Newton iterations on the nonlinear radiative surface boundary
/// `Theta * du/dx = u^4 - mu`. `mu_n` / `mu_np1` are the absorbed insolation (direct
/// plus any crater self-heating) at the current and next time.
fn cn_column_step(
    u: &mut [f64],
    scratch: &mut ColumnScratch,
    mu_n: f64,
    mu_np1: f64,
    grid: &DepthGrid,
    k_flux: f64,
) {
    let nz = u.len();
    scratch.u_old.copy_from_slice(u);
    let ColumnScratch {
        u_old,
        sub,
        diag,
        sup,
        rhs,
    } = scratch;
    let rs = grid.r_surf;
    let rb = grid.r_base;

    for _ in 0..N_SURFACE_ITERS {
        // clamp the linearization point positive so the stiff low-inertia regime
        // cannot drive the cubic term negative.
        let prev = u[0];
        let a_lin = prev.max(1e-6);

        // surface node (linearized radiative flux boundary; near-surface spacing h0)
        diag[0] = 1.0 + 2.0 * rs + 4.0 * rs * k_flux * a_lin.powi(3);
        sup[0] = -2.0 * rs;
        rhs[0] = u_old[0] + 3.0 * rs * k_flux * a_lin.powi(4) - rs * k_flux * u_old[0].powi(4)
            + rs * k_flux * (mu_n + mu_np1)
            + rs * (2.0 * u_old[1] - 2.0 * u_old[0]);

        // interior nodes (non-uniform 3-point Crank-Nicolson)
        for i in 1..nz - 1 {
            sub[i] = -grid.alpha[i];
            diag[i] = 1.0 - grid.beta[i];
            sup[i] = -grid.gamma[i];
            rhs[i] = u_old[i]
                + grid.alpha[i] * u_old[i - 1]
                + grid.beta[i] * u_old[i]
                + grid.gamma[i] * u_old[i + 1];
        }

        // insulating base node
        sub[nz - 1] = -rb;
        diag[nz - 1] = 1.0 + rb;
        rhs[nz - 1] = rb * u_old[nz - 2] + (1.0 - rb) * u_old[nz - 1];

        thomas(sub, diag, sup, rhs, u);

        // adaptive: stop once the nonlinear surface iterate has converged
        if (u[0] - prev).abs() < NEWTON_TOL {
            break;
        }
    }

    // Temperature is physically non-negative. In the stiff low-inertia limit the
    // linearized night-side surface can briefly overshoot below zero; a negative value
    // would feed back through the u^4 term and diverge. Clamp to keep the solve well
    // posed (emitted flux there is ~0 regardless). Written so NaN is preserved rather
    // than masked to zero.
    for v in u.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

/// Advance a column through one full rotation (the period map `F`), optionally
/// recording the surface temperature curve.
fn rotate_column(
    u: &mut [f64],
    scratch: &mut ColumnScratch,
    mu: &[f64],
    grid: &DepthGrid,
    k_flux: f64,
    mut surface: Option<&mut [f64]>,
) {
    let n_time = grid.n_time;
    for j in 0..n_time {
        cn_column_step(u, scratch, mu[j], mu[(j + 1) % n_time], grid, k_flux);
        if let Some(s) = surface.as_deref_mut() {
            s[j] = u[0];
        }
    }
}

/// Anderson acceleration of a fixed-point iteration `u <- F(u)`.
///
/// The approach to periodic steady state is linearly convergent (a single dominant
/// transient mode), so extrapolating from a short history of residuals
/// `g = F(u) - u` collapses ~30 plain iterations to a handful. Walker-Ni (2011) form
/// with unit mixing: `u_{k+1} = F(u_k) - dF * gamma`, where `gamma` minimizes
/// `||g_k - dG * gamma||` over the windowed residual differences.
struct Anderson {
    f_hist: Vec<Vec<f64>>,
    g_hist: Vec<Vec<f64>>,
}

impl Anderson {
    fn new() -> Self {
        Self {
            f_hist: Vec::new(),
            g_hist: Vec::new(),
        }
    }

    /// Given the current iterate `u` and `f = F(u)`, overwrite `u` with the next
    /// accelerated iterate (clamped non-negative).
    fn step(&mut self, u: &mut [f64], f: &[f64]) {
        let resid: Vec<f64> = f.iter().zip(u.iter()).map(|(fi, ui)| fi - ui).collect();
        self.f_hist.push(f.to_vec());
        self.g_hist.push(resid.clone());
        if self.f_hist.len() > ANDERSON_M {
            let _ = self.f_hist.remove(0);
            let _ = self.g_hist.remove(0);
        }

        let n_diff = self.f_hist.len() - 1;
        if n_diff == 0 {
            // no history yet: a plain fixed-point step
            u.copy_from_slice(f);
            return;
        }

        let len = u.len();
        let mut dg = DMatrix::zeros(len, n_diff);
        let mut df = DMatrix::zeros(len, n_diff);
        for col in 0..n_diff {
            for row in 0..len {
                dg[(row, col)] = self.g_hist[col + 1][row] - self.g_hist[col][row];
                df[(row, col)] = self.f_hist[col + 1][row] - self.f_hist[col][row];
            }
        }
        // gamma = argmin ||resid - dG gamma||, via ridge-regularized normal equations.
        let resid_vec = DVector::from_column_slice(&resid);
        let mut ata = dg.transpose() * &dg;
        for d in 0..n_diff {
            ata[(d, d)] += 1e-10;
        }
        let atb = dg.transpose() * &resid_vec;
        let gamma = ata
            .lu()
            .solve(&atb)
            .unwrap_or_else(|| DVector::zeros(n_diff));

        let correction = &df * &gamma;
        for (ui, (fi, ci)) in u.iter_mut().zip(f.iter().zip(correction.iter())) {
            *ui = (fi - ci).max(0.0);
        }
    }
}

/// Advance all crater micro-facet columns through one rotation (the period map),
/// coupled by the lagged IR self-heating scalar. Optionally records the surface.
#[allow(
    clippy::too_many_arguments,
    reason = "crater rotation needs the full coupled state and geometry"
)]
fn rotate_crater(
    cols: &mut [Vec<f64>],
    scratch: &mut ColumnScratch,
    direct: &[Vec<f64>],
    grid: &DepthGrid,
    k_flux: f64,
    emissivity: f64,
    crater: &Crater,
    pow4: &mut [f64],
    mut surface: Option<&mut [Vec<f64>]>,
) {
    let n_micro = cols.len();
    let n_time = grid.n_time;
    for j in 0..n_time {
        for (p, col) in pow4.iter_mut().zip(cols.iter()) {
            *p = col[0].powi(4);
        }
        let h_self = emissivity * crater.recapture_irradiance(pow4);
        let jnp1 = (j + 1) % n_time;
        for i in 0..n_micro {
            let mu_n = direct[j][i] + h_self;
            let mu_np1 = direct[jnp1][i] + h_self;
            cn_column_step(&mut cols[i], scratch, mu_n, mu_np1, grid, k_flux);
            if let Some(s) = surface.as_deref_mut() {
                s[i][j] = cols[i][0];
            }
        }
    }
}

/// Sun direction in the crater-local frame (`z` = crater axis = host facet normal)
/// at hour angle `h`, for a host facet at latitude `lat` and sub-solar latitude
/// `sub_solar_lat`. The horizontal axes are local east/north; returns a unit vector.
fn sun_in_crater_frame(lat: f64, sub_solar_lat: f64, h: f64) -> Vector3<f64> {
    let (s_lat, c_lat) = lat.sin_cos();
    let (s_sun, c_sun) = sub_solar_lat.sin_cos();
    let (s_h, c_h) = h.sin_cos();
    let east = -c_sun * s_h;
    let north = s_sun * c_lat - c_sun * s_lat * c_h;
    let up = s_lat * s_sun + c_lat * c_sun * c_h; // = cos(solar zenith)
    Vector3::new(east, north, up)
}

/// Solve the coupled micro-facet temperatures of a roughness crater to periodic
/// steady state. Returns `surface[micro_facet][time]`, the dimensionless temperature.
///
/// Each micro-facet runs the same 1D column solver as the smooth surface
/// ([`cn_column_step`]), with the absorbed insolation augmented by two intra-crater
/// terms, both collapsed to a single scalar per time step by the spherical cap's
/// constant view factor: the (nonlinear) IR self-heating, and -- when `bond_albedo > 0`
/// -- the (linear) solar multiple scattering, i.e. reflected sunlight that bounces in
/// the cavity and is partly reabsorbed. `bond_albedo -> 0` (dark surface) drops the
/// scattering term; `gamma -> 0` recovers the smooth single-column result.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the coupled crater state does not reach periodic
/// steady state within [`MAX_ROTATIONS`].
fn solve_crater(
    theta: f64,
    lat: f64,
    sub_solar_lat: f64,
    crater: &Crater,
    emissivity: f64,
    bond_albedo: f64,
    grid: &DepthGrid,
) -> KeteResult<Vec<Vec<f64>>> {
    let n_micro = crater.len();
    let nz = N_DEPTH;
    let n_time = grid.n_time;
    let dtau = TAU / n_time as f64;
    let k_flux = 2.0 * grid.h0 / theta;

    // direct insolation factor per time step, per micro-facet (with rim shadowing)
    let mut direct: Vec<Vec<f64>> = (0..n_time)
        .map(|j| {
            let sun = sun_in_crater_frame(lat, sub_solar_lat, dtau * j as f64);
            crater.direct_factor(&sun)
        })
        .collect();

    // Solar multiple scattering: a fraction `A = bond_albedo` of the incident sunlight
    // is reflected, bounces via the constant crater view factor, and part is
    // reabsorbed as extra heating. With a constant view factor the recaptured
    // irradiance is uniform over the facets, so the absorbed scattered heating per time
    // step is a single scalar `q = A*recapture(direct)/(1 - A*R)` (the geometric series
    // of bounces), where `R = total_cap_area/(4 pi) = recapture(1)`. It is
    // temperature-independent, so fold it into `direct` before the thermal solve. The
    // `(1 - A)` absorbed fraction cancels because `direct` is already the absorbed
    // (not incident) sunlight in these units.
    if bond_albedo > 0.0 {
        let r_uniform = crater.recapture_irradiance(&vec![1.0; n_micro]);
        for dj in &mut direct {
            let recap = crater.recapture_irradiance(dj);
            let q = bond_albedo * recap / (1.0 - bond_albedo * r_uniform);
            for d in dj.iter_mut() {
                *d += q;
            }
        }
    }

    // Initialize each column near its mean-insolation equilibrium, including the
    // steady intra-crater self-heating. At high Theta the column is nearly isothermal
    // at u^4 = mean_direct + emissivity * H, where H is the (uniform, constant view
    // factor) recapture of the facets' own emission. H couples all facets, so solve
    // its scalar fixed point first; seeding with self-heating makes the high-Theta
    // crater converge as fast as the smooth column (whose mean-insolation seed is
    // already the answer) instead of dragging a slow transient past MAX_ROTATIONS.
    let mean_direct: Vec<f64> = (0..n_micro)
        .map(|i| (0..n_time).map(|j| direct[j][i]).sum::<f64>() / n_time as f64)
        .collect();
    let mut u4 = mean_direct.clone();
    for _ in 0..20 {
        let h_self = emissivity * crater.recapture_irradiance(&u4);
        for (q, m) in u4.iter_mut().zip(&mean_direct) {
            *q = m + h_self;
        }
    }
    let mut cols: Vec<Vec<f64>> = u4.iter().map(|q| vec![q.powf(0.25); nz]).collect();

    let mut surface = vec![vec![0.0; n_time]; n_micro];
    let mut prev_surface = vec![vec![f64::INFINITY; n_time]; n_micro];
    let mut scratch = ColumnScratch::new(nz);
    let mut pow4 = vec![0.0; n_micro];
    let mut anderson = Anderson::new();
    let mut f_cols = cols.clone();
    let mut stacked = vec![0.0; n_micro * nz];
    let mut f_stacked = vec![0.0; n_micro * nz];

    for _ in 0..MAX_ROTATIONS {
        // f = F(cols): one rotation from a copy of the state, recording the surface.
        for (fc, c) in f_cols.iter_mut().zip(&cols) {
            fc.copy_from_slice(c);
        }
        rotate_crater(
            &mut f_cols,
            &mut scratch,
            &direct,
            grid,
            k_flux,
            emissivity,
            crater,
            &mut pow4,
            Some(&mut surface),
        );

        let mut diff = 0.0_f64;
        for (s, p) in surface.iter().zip(&prev_surface) {
            for (a, b) in s.iter().zip(p) {
                diff = diff.max((a - b).abs());
            }
        }
        if diff < CONV_TOL {
            return Ok(surface);
        }
        for (p, s) in prev_surface.iter_mut().zip(&surface) {
            p.copy_from_slice(s);
        }

        // accelerate the stacked (n_micro x nz) crater state toward its fixed point.
        for (i, c) in cols.iter().enumerate() {
            stacked[i * nz..(i + 1) * nz].copy_from_slice(c);
        }
        for (i, c) in f_cols.iter().enumerate() {
            f_stacked[i * nz..(i + 1) * nz].copy_from_slice(c);
        }
        anderson.step(&mut stacked, &f_stacked);
        for (i, c) in cols.iter_mut().enumerate() {
            c.copy_from_slice(&stacked[i * nz..(i + 1) * nz]);
        }
    }
    Err(Error::Convergence(format!(
        "TPM crater solve failed to reach periodic steady state within {MAX_ROTATIONS} \
         rotations (Theta={theta}, latitude={lat} rad)"
    )))
}

/// Directional thermal emission of a crater toward the observer, per unit flat
/// footprint area. `vpa` is the per-micro-facet visible projected area toward the
/// observer (from [`Crater::visible_projected_area`], band-independent so callers
/// compute it once per geometry). `temps` are the micro-facet dimensionless
/// temperatures at the observation's local solar time.
///
/// This is the value that replaces the smooth Lambertian `bb(T) * (n . obs)`:
/// `sum_i visible_projected_area_i(obs) * bb(T_ss * u_i) / opening_area`. Only
/// observer-visible micro-facets contribute, so the hot sunlit walls dominate at low
/// phase (beaming) and the cold shadowed walls at high phase. As `gamma -> 0` it
/// reduces to `bb(T) * (n . obs)`.
fn crater_emission(
    crater: &Crater,
    vpa: &[f64],
    temps: &[f64],
    t_ss: f64,
    wavelength: f64,
    color_correction: Option<ColorCorrFn>,
) -> f64 {
    let mut sum = 0.0;
    for (a, &u) in vpa.iter().zip(temps) {
        if *a <= 0.0 {
            continue;
        }
        let temp = t_ss * u;
        let mut bb = black_body_flux(temp, wavelength);
        if let Some(func) = color_correction {
            bb *= func(temp);
        }
        sum += *a * bb;
    }
    sum / crater.opening_area()
}

/// Solve a tridiagonal system in place using the Thomas algorithm.
///
/// `diag` and `rhs` are used as scratch and overwritten; the solution is written to
/// `out`. `sub[0]` and `sup[n-1]` are unused.
fn thomas(sub: &[f64], diag: &mut [f64], sup: &[f64], rhs: &mut [f64], out: &mut [f64]) {
    let n = diag.len();
    for i in 1..n {
        let w = sub[i] / diag[i - 1];
        diag[i] -= w * sup[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    out[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        out[i] = (rhs[i] - sup[i] * out[i + 1]) / diag[i];
    }
}

/// Observation geometry reduced to the quantities the TPM solver and renderer need.
struct TpmGeom {
    /// Sub-solar equilibrium temperature in Kelvin.
    t_ss: f64,
    /// Thermal parameter.
    theta: f64,
    /// Sub-solar latitude relative to the spin equator (radians).
    sub_solar_lat: f64,
    /// Bond albedo of the surface (drives in-crater solar multiple scattering; 0 for the
    /// smooth path and the offline correction-table build).
    bond_albedo: f64,
    /// Unit vector from the object to the Sun.
    obj2sun_hat: Vector3<f64>,
    /// Spin axis.
    pole: Vector3<f64>,
    /// Vector from the observer to the object in AU.
    obs2obj: Vector3<f64>,
    /// Distance from the observer to the object in AU.
    obs2obj_r: f64,
}

/// Reduce the observation geometry to a [`TpmGeom`]. Returns `None` when the
/// sub-solar temperature is non-positive (object effectively unilluminated).
fn tpm_geom(
    spin: &SpinState,
    thermal: &ThermalParams,
    vis_albedo: f64,
    g_param: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
) -> Option<TpmGeom> {
    let obj2sun = -sun2obj;
    let obs2obj = sun2obj - sun2obs;
    let obs2obj_r = obs2obj.norm();

    // sub-solar equilibrium temperature (beaming = 1; the TPM resolves the
    // distribution explicitly rather than folding it into a beaming parameter).
    let t_ss = sub_solar_temperature(obj2sun.norm(), vis_albedo, g_param, 1.0, thermal.emissivity);
    if t_ss <= 0.0 {
        return None;
    }
    let theta = thermal.thermal_parameter(spin.period, t_ss);
    let pole = spin.pole.into_inner();
    let obj2sun_hat = obj2sun.normalize();
    let sub_solar_lat = obj2sun_hat.dot(&pole).clamp(-1.0, 1.0).asin();
    Some(TpmGeom {
        t_ss,
        theta,
        sub_solar_lat,
        bond_albedo: bond_albedo(vis_albedo, g_param),
        obj2sun_hat,
        pole,
        obs2obj,
        obs2obj_r,
    })
}

/// Body-fixed (x, y) axes expressed in inertial space, with body-z aligned to `pole`.
///
/// The x-axis is anchored to the ascending node of the body equator. A non-axisymmetric
/// shape is rotated by `phase` about the pole; an axisymmetric shape (sphere/oblate) is
/// phase-independent. Body normals map to inertial space as `b.x*x + b.y*y + b.z*pole`,
/// which keeps the symmetry axis on the spin pole. When `pole` is +z this returns the
/// identity frame `(x_hat, y_hat)`.
fn body_axes(pole: Vector3<f64>, phase: f64, axisymmetric: bool) -> (Vector3<f64>, Vector3<f64>) {
    let node = {
        let c = Vector3::z().cross(&pole);
        if c.norm() > 1e-8 {
            c.normalize()
        } else {
            Vector3::x()
        }
    };
    let e2_node = pole.cross(&node);
    if axisymmetric {
        (node, e2_node)
    } else {
        let (sin_p, cos_p) = phase.sin_cos();
        let x_body = cos_p * node + sin_p * e2_node;
        (x_body, pole.cross(&x_body))
    }
}

/// Render the disk-integrated thermal flux per band.
///
/// Sums Planck emission over all observer-visible facets (illuminated or not), each
/// at the temperature given by `temp_u(lat, tau)` -- the dimensionless surface
/// temperature `u = T / T_ss` at the facet's latitude and local solar time. The
/// sampler is a closure so the field can be either a single solved field or sampled
/// on demand from a [`TpmFieldGrid`] without materializing it.
fn render_thermal_flux<F: Fn(f64, f64) -> f64>(
    temp_u: F,
    geom: &TpmGeom,
    shape: &TpmShape,
    phase: f64,
    diameter: f64,
    emissivity: f64,
    obs_bands: &[BandInfo],
) -> Vec<f64> {
    let pole = geom.pole;

    // equatorial basis: e1 points to the sub-solar meridian, e2 = pole x e1 points
    // in the direction of rotation (increasing local solar time).
    let s_perp = geom.obj2sun_hat - geom.obj2sun_hat.dot(&pole) * pole;
    let e1 = if s_perp.norm() > 1e-8 {
        s_perp.normalize()
    } else {
        // sun is along the pole; the diurnal curve is flat so any basis works.
        let t = if pole.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        (t - t.dot(&pole) * pole).normalize()
    };
    let e2 = pole.cross(&e1);

    // Body-fixed frame expressed in inertial space: body-z is aligned with the pole and
    // the body x-axis is anchored to the ascending node of the body equator. A
    // non-axisymmetric shape is additionally rotated by `phase` about the pole; an
    // axisymmetric shape (sphere/oblate) is phase-independent, but body-z must STILL be
    // aligned with the pole so an oblate is flattened along the spin axis, not along
    // ecliptic z. (When the pole is +z this frame is the identity.)
    let (x_body, y_body) = body_axes(pole, phase, shape.axisymmetric);

    let obs2obj_hat = UnitVector3::new_normalize(geom.obs2obj);
    let bands: Vec<_> = obs_bands.iter().map(|x| x.wavelength).collect();
    let color_correction: Vec<_> = obs_bands.iter().map(|x| x.color_correction).collect();

    let mut fluxes = vec![0.0; obs_bands.len()];
    for facet in &shape.facets {
        // orient the body-fixed facet normal in inertial space (body-z -> pole)
        let b = facet.normal.into_inner();
        let normal = b.x * x_body + b.y * y_body + b.z * pole;
        let normal_unit = UnitVector3::new_normalize(normal);

        let obs_flux_scaling = lambertian_vis_scale_factor(
            &normal_unit,
            &obs2obj_hat,
            geom.obs2obj_r,
            diameter,
            emissivity,
        );
        if obs_flux_scaling == 0.0 {
            continue;
        }

        // map the facet to its latitude and local solar time, then look up the
        // temperature. Night-side facets are NOT skipped: they retain stored heat.
        let lat = normal.dot(&pole).clamp(-1.0, 1.0).asin();
        let tau = normal.dot(&e2).atan2(normal.dot(&e1));
        let temp = geom.t_ss * temp_u(lat, tau);

        for (idx, (wavelength, flux)) in bands.iter().zip(&mut fluxes).enumerate() {
            let mut facet_flux = black_body_flux(temp, *wavelength);
            if let Some(func) = color_correction[idx] {
                facet_flux *= func(temp);
            }
            facet_flux *= facet.area;
            *flux += obs_flux_scaling * facet_flux;
        }
    }
    fluxes
}

/// Solve the crater micro-facet temperature fields for every latitude band at a given
/// thermal parameter and sub-solar latitude (the dominant cost of the rough path).
///
/// Returns `fields[lat_band][micro_facet][time]`. Independent of rotation phase,
/// observation direction, and `T_ss` (the solution is dimensionless), so the renderer
/// can reuse one solve across many phases / temperatures (see [`build`]).
///
/// [`build`]: RoughnessCorrection::build
///
/// # Errors
///
/// Returns [`Error::Convergence`] if a crater solve does not reach periodic steady
/// state.
fn solve_crater_fields(
    theta: f64,
    sub_solar_lat: f64,
    crater: &Crater,
    emissivity: f64,
    bond_albedo: f64,
    grid: &DepthGrid,
) -> KeteResult<Vec<Vec<Vec<f64>>>> {
    (0..N_LAT)
        .into_par_iter()
        .with_min_len(1)
        .map(|k| {
            let lat = -FRAC_PI_2 + PI * k as f64 / (N_LAT - 1) as f64;
            solve_crater(
                theta,
                lat,
                sub_solar_lat,
                crater,
                emissivity,
                bond_albedo,
                grid,
            )
        })
        .collect()
}

/// Render the disk-integrated thermal flux with surface roughness (beaming).
///
/// Each host facet is replaced by a [`Crater`] whose micro-facet temperatures are
/// solved per latitude band (the expensive part -- ~`N_LAT` crater solves), then the
/// facet's beamed emission toward the observer is summed via [`crater_emission`].
/// This is the on-the-fly path (no cross-call cache); it is correct but costly.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if a crater solve does not reach periodic steady
/// state.
fn render_rough_thermal_flux(
    geom: &TpmGeom,
    shape: &TpmShape,
    phase: f64,
    diameter: f64,
    emissivity: f64,
    obs_bands: &[BandInfo],
    crater: &Crater,
    grid: &DepthGrid,
) -> KeteResult<Vec<f64>> {
    let crater_fields = solve_crater_fields(
        geom.theta,
        geom.sub_solar_lat,
        crater,
        emissivity,
        geom.bond_albedo,
        grid,
    )?;
    Ok(render_crater_disk(
        &crater_fields,
        geom,
        shape,
        phase,
        diameter,
        emissivity,
        obs_bands,
        crater,
        grid,
    ))
}

/// Render the disk-integrated rough thermal flux from precomputed crater temperature
/// fields (one per latitude band, from [`solve_crater_fields`]).
///
/// Split out from the solve so the offline table build can amortize the heat solve
/// over many rotation phases and sub-solar temperatures, which only affect this cheap
/// directional sum (`T_ss` enters through the Planck emission, phase through the
/// body-fixed orientation).
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "latitude/time indices are rounded from clamped non-negative values"
)]
fn render_crater_disk(
    crater_fields: &[Vec<Vec<f64>>],
    geom: &TpmGeom,
    shape: &TpmShape,
    phase: f64,
    diameter: f64,
    emissivity: f64,
    obs_bands: &[BandInfo],
    crater: &Crater,
    grid: &DepthGrid,
) -> Vec<f64> {
    let pole = geom.pole;

    // sun-meridian basis (e1 toward sub-solar meridian, e2 = pole x e1)
    let s_perp = geom.obj2sun_hat - geom.obj2sun_hat.dot(&pole) * pole;
    let e1 = if s_perp.norm() > 1e-8 {
        s_perp.normalize()
    } else {
        let t = if pole.x.abs() < 0.9 {
            Vector3::x()
        } else {
            Vector3::y()
        };
        (t - t.dot(&pole) * pole).normalize()
    };
    let e2 = pole.cross(&e1);

    // body-fixed frame: body-z aligned to the pole (identity when pole is +z), with the
    // phase rotation applied only for non-axisymmetric shapes.
    let (x_body, y_body) = body_axes(pole, phase, shape.axisymmetric);

    let obs2obj_hat = UnitVector3::new_normalize(geom.obs2obj);
    let obs_dir = -obs2obj_hat.into_inner(); // object -> observer
    let dist_scale = (geom.obs2obj_r * AU_KM / diameter).powi(-2);
    let n_time = grid.n_time;
    let dtau = TAU / n_time as f64;
    let n_micro = crater.len();

    let bands: Vec<_> = obs_bands.iter().map(|x| x.wavelength).collect();
    let color_correction: Vec<_> = obs_bands.iter().map(|x| x.color_correction).collect();

    let mut fluxes = vec![0.0; obs_bands.len()];
    let mut temps = vec![0.0; n_micro];
    for facet in &shape.facets {
        let b = facet.normal.into_inner();
        let normal = b.x * x_body + b.y * y_body + b.z * pole;

        // the observer must see the facet (the crater's directional projection is
        // handled inside crater_emission)
        let observed = normal.dot(&obs_dir);
        if observed <= 0.0 {
            continue;
        }

        // crater-local frame at this facet: up = normal, east = rotation direction,
        // north = toward the pole.
        let east = {
            let c = pole.cross(&normal);
            if c.norm() > 1e-8 { c.normalize() } else { e1 }
        };
        let north = {
            let nn = pole - pole.dot(&normal) * normal;
            if nn.norm() > 1e-8 { nn.normalize() } else { e2 }
        };
        let obs_local = Vector3::new(obs_dir.dot(&east), obs_dir.dot(&north), observed);

        // Bilinearly interpolate the crater field over latitude band and local solar
        // time (wrapping in time). This matches the smooth render's bilinear sampling,
        // so the rough and smooth disks are sampled consistently -- important because
        // the correction table stores their ratio.
        let lat = normal.dot(&pole).clamp(-1.0, 1.0).asin();
        let fk = ((lat + FRAC_PI_2) / PI * (N_LAT - 1) as f64).clamp(0.0, (N_LAT - 1) as f64);
        let k0 = fk.floor() as usize;
        let k1 = (k0 + 1).min(N_LAT - 1);
        let wk = fk - k0 as f64;

        let mut tau = normal.dot(&e2).atan2(normal.dot(&e1));
        if tau < 0.0 {
            tau += TAU;
        }
        let fj = tau / dtau;
        let j0 = (fj.floor() as usize) % n_time;
        let j1 = (j0 + 1) % n_time;
        let wj = fj - fj.floor();

        let field0 = &crater_fields[k0];
        let field1 = &crater_fields[k1];
        for (i, t) in temps.iter_mut().enumerate() {
            let a = field0[i][j0] * (1.0 - wj) + field0[i][j1] * wj;
            let b = field1[i][j0] * (1.0 - wj) + field1[i][j1] * wj;
            *t = a * (1.0 - wk) + b * wk;
        }

        let area_scale = emissivity * PI * dist_scale * facet.area;
        let vpa = crater.visible_projected_area(&obs_local);
        for ((wavelength, cc), flux) in bands.iter().zip(&color_correction).zip(&mut fluxes) {
            *flux +=
                area_scale * crater_emission(crater, &vpa, &temps, geom.t_ss, *wavelength, *cc);
        }
    }
    fluxes
}

/// Compute the TPM thermal flux for each band.
///
/// This solves the heat equation directly for the observation geometry. For
/// repeated evaluations at varying parameters (e.g. fitting), build a
/// [`TpmFieldGrid`] once and use [`tpm_thermal_flux_cached`].
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Effective diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the heat-equation solve does not reach periodic
/// steady state.
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_thermal_flux(
    obs_bands: &[BandInfo],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
) -> KeteResult<Vec<f64>> {
    let Some(geom) = tpm_geom(spin, thermal, vis_albedo, g_param, sun2obj, sun2obs) else {
        return Ok(vec![0.0; obs_bands.len()]);
    };
    let field = DiurnalField::solve(geom.theta, geom.sub_solar_lat, &DEPTH_GRID)?;
    let phase = spin.rotation_phase(epoch);
    Ok(render_thermal_flux(
        |lat, tau| field.sample(lat, tau),
        &geom,
        shape,
        phase,
        diameter,
        thermal.emissivity,
        obs_bands,
    ))
}

/// Compute the TPM thermal flux for each band using a precomputed field grid.
///
/// Identical to [`tpm_thermal_flux`] except the diurnal field is interpolated from
/// `grid` instead of solved, which is dramatically cheaper for repeated evaluation.
///
/// # Arguments
///
/// * `grid` - Precomputed field grid covering the relevant `Theta` range.
/// * `obs_bands` - Wavelength band information of the observer.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Effective diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_thermal_flux_cached(
    grid: &TpmFieldGrid,
    obs_bands: &[BandInfo],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
) -> Vec<f64> {
    let Some(geom) = tpm_geom(spin, thermal, vis_albedo, g_param, sun2obj, sun2obs) else {
        return vec![0.0; obs_bands.len()];
    };
    let (corners, wi, wj) = grid.corner_fields(geom.theta, geom.sub_solar_lat);
    let phase = spin.rotation_phase(epoch);
    render_thermal_flux(
        |lat, tau| sample_corners(&corners, wi, wj, lat, tau),
        &geom,
        shape,
        phase,
        diameter,
        thermal.emissivity,
        obs_bands,
    )
}

/// Compute TPM thermal + reflected flux and magnitudes for each band.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `band_albedos` - Albedo of the object for each band.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if the heat-equation solve does not reach periodic
/// steady state.
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_total_flux(
    obs_bands: &[BandInfo],
    band_albedos: &[f64],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    h_mag: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
) -> KeteResult<ModelResults> {
    let thermal_fluxes = tpm_thermal_flux(
        obs_bands, spin, shape, thermal, diameter, vis_albedo, g_param, sun2obj, sun2obs, epoch,
    )?;
    Ok(assemble_total(
        obs_bands,
        band_albedos,
        thermal_fluxes,
        diameter,
        g_param,
        h_mag,
        sun2obj,
        sun2obs,
    ))
}

/// Compute TPM thermal + reflected flux and magnitudes using a precomputed grid.
///
/// Identical to [`tpm_total_flux`] except the diurnal field is interpolated from
/// `grid`.
///
/// # Arguments
///
/// * `grid` - Precomputed field grid covering the relevant `Theta` range.
/// * `obs_bands` - Wavelength band information of the observer.
/// * `band_albedos` - Albedo of the object for each band.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
#[must_use]
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_total_flux_cached(
    grid: &TpmFieldGrid,
    obs_bands: &[BandInfo],
    band_albedos: &[f64],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    h_mag: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
) -> ModelResults {
    let thermal_fluxes = tpm_thermal_flux_cached(
        grid, obs_bands, spin, shape, thermal, diameter, vis_albedo, g_param, sun2obj, sun2obs,
        epoch,
    );
    assemble_total(
        obs_bands,
        band_albedos,
        thermal_fluxes,
        diameter,
        g_param,
        h_mag,
        sun2obj,
        sun2obs,
    )
}

/// Compute the TPM thermal flux for each band with surface roughness (beaming).
///
/// `roughness_angle` is the crater opening half-angle in radians (a proxy for RMS
/// slope), in `(0, pi/2]`. This solves a [`Crater`] per latitude band on the fly --
/// it is correct but far slower than the smooth path (no cache), and has no cached or
/// fitting variant yet.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Effective diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
/// * `roughness_angle` - Crater opening half-angle in radians, `(0, pi/2]`.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if a crater solve does not reach periodic steady
/// state.
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_thermal_flux_rough(
    obs_bands: &[BandInfo],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
    roughness_angle: f64,
) -> KeteResult<Vec<f64>> {
    let Some(geom) = tpm_geom(spin, thermal, vis_albedo, g_param, sun2obj, sun2obs) else {
        return Ok(vec![0.0; obs_bands.len()]);
    };
    let crater = Crater::new(roughness_angle, CRATER_RINGS, CRATER_SECTORS);
    let phase = spin.rotation_phase(epoch);
    render_rough_thermal_flux(
        &geom,
        shape,
        phase,
        diameter,
        thermal.emissivity,
        obs_bands,
        &crater,
        &DEPTH_GRID,
    )
}

/// Compute TPM thermal + reflected flux and magnitudes with surface roughness.
///
/// Like [`tpm_total_flux`] but uses the rough (beaming) thermal path; see
/// [`tpm_thermal_flux_rough`] for the cost caveat.
///
/// # Arguments
///
/// * `obs_bands` - Wavelength band information of the observer.
/// * `band_albedos` - Albedo of the object for each band.
/// * `spin` - Spin state (pole, rotation period, phase) of the object.
/// * `shape` - Body-fixed shape (sphere, ellipsoid, or custom mesh).
/// * `thermal` - Thermal inertia and emissivity of the surface.
/// * `diameter` - Effective diameter of the object in km.
/// * `vis_albedo` - Visible geometric albedo of the object.
/// * `g_param` - The G parameter in the HG system.
/// * `h_mag` - The H parameter of the object in the HG system.
/// * `sun2obj` - Position of the object with respect to the Sun in AU.
/// * `sun2obs` - Position of the observer with respect to the Sun in AU.
/// * `epoch` - Observation time (Julian date); sets the rotation phase.
/// * `roughness_angle` - Crater opening half-angle in radians, `(0, pi/2]`.
///
/// # Errors
///
/// Returns [`Error::Convergence`] if a crater solve does not reach periodic steady
/// state.
#[allow(
    clippy::too_many_arguments,
    reason = "flux models take the full observation geometry explicitly"
)]
pub fn tpm_total_flux_rough(
    obs_bands: &[BandInfo],
    band_albedos: &[f64],
    spin: &SpinState,
    shape: &TpmShape,
    thermal: &ThermalParams,
    diameter: f64,
    vis_albedo: f64,
    g_param: f64,
    h_mag: f64,
    sun2obj: &Vector3<f64>,
    sun2obs: &Vector3<f64>,
    epoch: f64,
    roughness_angle: f64,
) -> KeteResult<ModelResults> {
    let thermal_fluxes = tpm_thermal_flux_rough(
        obs_bands,
        spin,
        shape,
        thermal,
        diameter,
        vis_albedo,
        g_param,
        sun2obj,
        sun2obs,
        epoch,
        roughness_angle,
    )?;
    Ok(assemble_total(
        obs_bands,
        band_albedos,
        thermal_fluxes,
        diameter,
        g_param,
        h_mag,
        sun2obj,
        sun2obs,
    ))
}

/// Mean surface slope angle (radians) of the full-coverage spherical-cap roughness
/// surface with crater opening half-angle `gamma` (radians).
///
/// The model is parameterized internally by the crater opening half-angle, but the
/// literature usually quotes a *mean slope angle* `theta_bar` -- the convention-stable,
/// cross-model roughness currency (Hapke photometric roughness; what most TPM papers
/// report). For the spherical cap the local slope angle equals the micro-facet polar
/// angle, and true-surface-area weighting gives the slope PDF `sin(theta)` on
/// `[0, gamma]`, hence the closed form
///
/// ```text
///   theta_bar = (sin g - g cos g) / (1 - cos g).
/// ```
///
/// It is smooth and strictly increasing from 0 (smooth) to exactly 1 rad (~= 57.3 deg)
/// at `gamma = pi/2` (a hemisphere). Full-coverage caps therefore cannot represent a
/// mean slope above ~57 deg. Small-angle limit: `theta_bar -> (2/3) gamma`.
#[must_use]
pub fn mean_slope_angle(gamma: f64) -> f64 {
    // Below ~0.06 deg the closed form loses precision to cancellation; use the limit.
    if gamma <= 1e-3 {
        return 2.0 / 3.0 * gamma.max(0.0);
    }
    let (s, c) = gamma.sin_cos();
    (s - gamma * c) / (1.0 - c)
}

/// Inverse of [`mean_slope_angle`]: the crater opening half-angle `gamma` (radians)
/// whose mean slope angle is `theta_bar` (radians).
///
/// `theta_bar` is clamped to the representable range `(0, 1]` rad (the hemisphere
/// ceiling, ~= 57.3 deg). Computed by bisection -- `mean_slope_angle` is strictly
/// increasing, so the root is unique and the iteration is unconditionally robust (its
/// derivative stays ~0.57-0.66 over the whole range, so it is well conditioned even at
/// the ceiling).
#[must_use]
pub fn gamma_from_mean_slope(theta_bar: f64) -> f64 {
    if theta_bar <= 0.0 {
        return 0.0;
    }
    if theta_bar >= 1.0 {
        // mean_slope_angle(pi/2) == 1 exactly.
        return FRAC_PI_2;
    }
    let (mut lo, mut hi) = (0.0_f64, FRAC_PI_2);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mean_slope_angle(mid) < theta_bar {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Approximate RMS surface slope angle (radians) of the full-coverage spherical-cap
/// surface, for cross-comparison with Gaussian-random-surface roughness models that
/// quote an RMS slope (e.g. Rozitis & Green). True-area weighted:
///
/// ```text
///   <tan^2 theta> = (sec g + cos g - 2) / (1 - cos g),   returned as atan(sqrt(.)).
/// ```
///
/// NOTE: this is a *derived, approximate* cross-walk, not an internal model parameter.
/// Unlike the mean slope angle it is dominated by the near-vertical rim facets, so it
/// grows without bound as `gamma -> pi/2` and is sensitive to the crater
/// discretization. It tracks the mean slope angle to a few degrees in the realistic
/// regime (`gamma` up to ~50 deg) and should be treated as indicative beyond that.
#[must_use]
pub fn rms_slope(gamma: f64) -> f64 {
    if gamma <= 1e-3 {
        // small-angle limit: sqrt(<theta^2>) = gamma / sqrt(2).
        return gamma.max(0.0) / std::f64::consts::SQRT_2;
    }
    let c = gamma.cos();
    let mean_tan2 = (1.0 / c + c - 2.0) / (1.0 - c);
    mean_tan2.max(0.0).sqrt().atan()
}

/// Bracket `x` in an ascending grid, returning `(i0, i1, w)` such that the
/// interpolated value is `v[i0]*(1-w) + v[i1]*w`. Clamps outside the grid.
fn grid_bracket(grid: &[f64], x: f64) -> (usize, usize, f64) {
    let n = grid.len();
    if x <= grid[0] {
        return (0, 0, 0.0);
    }
    if x >= grid[n - 1] {
        return (n - 1, n - 1, 0.0);
    }
    let i1 = grid.partition_point(|&g| g < x).max(1);
    let i0 = i1 - 1;
    let w = (x - grid[i0]) / (grid[i1] - grid[i0]);
    (i0, i1, w)
}

/// A precomputed multiplicative roughness correction `R(Theta, gamma, phase, T_ss)`
/// per band, such that `rough_flux ~= smooth_flux * R`.
///
/// Building each grid node runs an on-the-fly rough disk render, so [`build`] is slow
/// and meant to be done once (offline) and reused; applying it via [`factor`] is then
/// cheap, which makes roughness usable inside a fit (smooth cache flux times `R`).
///
/// `R` carries an explicit sub-solar-temperature (`T_ss`) axis. The dimensionless
/// crater temperature field is fixed at a given `(Theta, gamma, phase)`, but the band
/// flux is a sum of Planck functions over the micro-facet temperatures, which is
/// nonlinear in `T_ss` (the Wien curvature): the same roughness beams more strongly in
/// a band that sits on the Wien side of the temperature distribution. A single-`T_ss`
/// table therefore mis-corrects objects far from that temperature, so `T_ss` is
/// tabulated and interpolated per observation rather than fixed.
///
/// Residual accuracy is a few percent: the disk-integrated `rough/smooth` ratio is
/// nearly independent of aspect (sub-solar latitude) for typical geometries (~1%,
/// rising to ~4% at pole-on), which is the one dimension still folded into a
/// representative value.
///
/// [`build`]: RoughnessCorrection::build
/// [`factor`]: RoughnessCorrection::factor
#[derive(Debug, Clone)]
pub struct RoughnessCorrection {
    log_thetas: Vec<f64>,
    gammas: Vec<f64>,
    phases: Vec<f64>,
    /// Sub-solar temperatures (K) the table was built at, ascending.
    t_sss: Vec<f64>,
    /// Band wavelengths (nm) the table was built for, one per band column.
    wavelengths: Vec<f64>,
    n_bands: usize,
    factors: Vec<f64>,
}

impl RoughnessCorrection {
    /// Construct directly from a tabulated grid (row-major over theta, gamma, phase,
    /// `t_ss`, band). `thetas` must be ascending and positive; `gammas`, `phases`, and
    /// `t_sss` ascending. Mainly for testing or loading a precomputed table.
    ///
    /// # Panics
    ///
    /// Panics if `factors.len()` does not equal
    /// `thetas * gammas * phases * t_sss * n_bands`.
    #[must_use]
    pub fn from_factors(
        thetas: &[f64],
        gammas: &[f64],
        phases: &[f64],
        t_sss: &[f64],
        wavelengths: &[f64],
        factors: Vec<f64>,
    ) -> Self {
        let n_bands = wavelengths.len();
        assert_eq!(
            factors.len(),
            thetas.len() * gammas.len() * phases.len() * t_sss.len() * n_bands,
            "factor table size mismatch"
        );
        Self {
            log_thetas: thetas.iter().map(|t| t.ln()).collect(),
            gammas: gammas.to_vec(),
            phases: phases.to_vec(),
            t_sss: t_sss.to_vec(),
            wavelengths: wavelengths.to_vec(),
            n_bands,
            factors,
        }
    }

    /// Build the correction table over the given grids by rendering the smooth and
    /// rough disk flux at each node (canonical sub-solar-latitude-0 geometry).
    ///
    /// `t_sss` are the sub-solar temperatures (K) the table is tabulated at;
    /// `obs_bands` fixes the band set; `rings`/`sectors` set the crater resolution.
    /// `n_time` is the time-sampling resolution of the offline solve: because the
    /// table stores the ratio `rough / smooth`, the systematic terminator-clipping
    /// bias largely cancels, so this build tolerates a coarser `n_time` than the
    /// runtime default for a near-linear speedup (~2x at `n_time = 180`).
    ///
    /// The phase and `T_ss` axes are cheap: the smooth field and the crater
    /// temperature field depend only on `(Theta, gamma)` (the dimensionless solution),
    /// so they are solved once per `(Theta, gamma)` and only re-rendered across phase
    /// and `T_ss`. The expensive heat solves therefore do not multiply with those two
    /// axes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Convergence`] if any smooth or rough solve at a grid node fails
    /// to reach periodic steady state.
    pub fn build(
        thetas: &[f64],
        gammas: &[f64],
        phases: &[f64],
        t_sss: &[f64],
        emissivity: f64,
        obs_bands: &[BandInfo],
        rings: usize,
        sectors: usize,
        n_time: usize,
    ) -> KeteResult<Self> {
        let shape = TpmShape::sphere();
        let grid = DepthGrid::with_n_time(n_time);
        let n_bands = obs_bands.len();
        let n_g = gammas.len();
        let n_p = phases.len();
        let n_s = t_sss.len();

        // One block per (Theta, gamma): solve the smooth and crater fields once, then
        // render the rough/smooth ratio at every (phase, T_ss). Each block is laid out
        // row-major over (phase, t_ss, band); concatenating the blocks in (theta,
        // gamma) order yields the global (theta, gamma, phase, t_ss, band) layout.
        let blocks: Vec<Vec<f64>> = (0..thetas.len() * n_g)
            .into_par_iter()
            .map(|node| -> KeteResult<Vec<f64>> {
                let it = node / n_g;
                let ig = node % n_g;
                let (theta, gamma) = (thetas[it], gammas[ig]);

                let field = DiurnalField::solve(theta, 0.0, &grid)?;
                let crater = Crater::new(gamma, rings, sectors);
                // Table build uses the dark-body limit (no solar scattering); see
                // `canonical_geom`.
                let crater_fields =
                    solve_crater_fields(theta, 0.0, &crater, emissivity, 0.0, &grid)?;

                let mut block = vec![0.0; n_p * n_s * n_bands];
                for (ip, &phase) in phases.iter().enumerate() {
                    for (is, &t_ss) in t_sss.iter().enumerate() {
                        let geom = canonical_geom(theta, phase, t_ss);
                        let smooth = render_thermal_flux(
                            |lat, tau| field.sample(lat, tau),
                            &geom,
                            &shape,
                            0.0,
                            1.0,
                            emissivity,
                            obs_bands,
                        );
                        let rough = render_crater_disk(
                            &crater_fields,
                            &geom,
                            &shape,
                            0.0,
                            1.0,
                            emissivity,
                            obs_bands,
                            &crater,
                            &grid,
                        );
                        let base = (ip * n_s + is) * n_bands;
                        for (b, (s, r)) in smooth.iter().zip(&rough).enumerate() {
                            block[base + b] = if *s > 0.0 { r / s } else { 1.0 };
                        }
                    }
                }
                Ok(block)
            })
            .collect::<KeteResult<Vec<_>>>()?;

        Ok(Self {
            log_thetas: thetas.iter().map(|t| t.ln()).collect(),
            gammas: gammas.to_vec(),
            phases: phases.to_vec(),
            t_sss: t_sss.to_vec(),
            wavelengths: obs_bands.iter().map(|b| b.wavelength).collect(),
            n_bands,
            factors: blocks.into_iter().flatten().collect(),
        })
    }

    fn idx(&self, it: usize, ig: usize, ip: usize, is: usize, band: usize) -> usize {
        (((it * self.gammas.len() + ig) * self.phases.len() + ip) * self.t_sss.len() + is)
            * self.n_bands
            + band
    }

    /// The correction factor for `band` at `(theta, gamma, phase, t_ss)`,
    /// multilinearly interpolated (linear in log-theta) and clamped to the grid.
    #[must_use]
    pub fn factor(&self, theta: f64, gamma: f64, phase: f64, t_ss: f64, band: usize) -> f64 {
        let (it0, it1, wt) = grid_bracket(&self.log_thetas, theta.max(1e-12).ln());
        let (ig0, ig1, wg) = grid_bracket(&self.gammas, gamma);
        let (ip0, ip1, wp) = grid_bracket(&self.phases, phase);
        let (is0, is1, ws) = grid_bracket(&self.t_sss, t_ss);
        let at = |it: usize, ig: usize, ip: usize, is: usize| {
            self.factors[self.idx(it, ig, ip, is, band)]
        };
        let lerp = |a: f64, b: f64, w: f64| a * (1.0 - w) + b * w;
        let on_s =
            |it: usize, ig: usize, ip: usize| lerp(at(it, ig, ip, is0), at(it, ig, ip, is1), ws);
        let on_p = |it: usize, ig: usize| lerp(on_s(it, ig, ip0), on_s(it, ig, ip1), wp);
        let on_g = |it: usize| lerp(on_p(it, ig0), on_p(it, ig1), wg);
        lerp(on_g(it0), on_g(it1), wt)
    }

    /// Apply the correction to a smooth flux vector in place: `flux[b] *= factor(...)`.
    pub fn apply(&self, smooth_flux: &mut [f64], theta: f64, gamma: f64, phase: f64, t_ss: f64) {
        for (band, flux) in smooth_flux.iter_mut().enumerate() {
            *flux *= self.factor(theta, gamma, phase, t_ss, band);
        }
    }

    /// The precomputed table shipped with the crate (embedded in the binary).
    ///
    /// Covers the four WISE bands over an `8 (Theta) x 6 (gamma) x 9 (phase) x 7 (T_ss)`
    /// grid. The `gamma` axis spans 10-90 deg opening half-angle, i.e. mean slope angle
    /// `theta_bar` from ~6.7 to 57.3 deg -- the full range the full-coverage cap can
    /// represent. Loading it avoids the slow per-fit table build; see the type docs for
    /// accuracy caveats (weak aspect dependence folded into a representative value).
    /// Regenerate at higher resolution if needed.
    ///
    /// # Panics
    ///
    /// Panics only if the embedded table is corrupt (a build-time invariant).
    #[must_use]
    pub fn shipped() -> Self {
        Self::from_bytes(include_bytes!("roughness_table.bin"))
            .expect("the embedded roughness table must be valid")
    }

    /// Serialize to a self-describing little-endian binary blob:
    /// `[n_theta, n_gamma, n_phase, n_t_ss, n_bands : u64]` then the `thetas`,
    /// `gammas`, `phases`, `t_sss`, `wavelengths`, and `factors` as `f64`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let thetas: Vec<f64> = self.log_thetas.iter().map(|l| l.exp()).collect();
        let counts = [
            self.log_thetas.len() as u64,
            self.gammas.len() as u64,
            self.phases.len() as u64,
            self.t_sss.len() as u64,
            self.n_bands as u64,
        ];
        let mut out = Vec::new();
        for c in counts {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for v in thetas
            .iter()
            .chain(&self.gammas)
            .chain(&self.phases)
            .chain(&self.t_sss)
            .chain(&self.wavelengths)
            .chain(&self.factors)
        {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Reconstruct from [`to_bytes`](Self::to_bytes). Returns `None` if the blob is
    /// truncated or inconsistent.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let read_f64 =
            |c: &[u8]| -> Option<f64> { Some(f64::from_le_bytes(<[u8; 8]>::try_from(c).ok()?)) };
        let read_u64 = |off: usize| -> Option<usize> {
            let c = bytes.get(off..off.checked_add(8)?)?;
            Some(u64::from_le_bytes(<[u8; 8]>::try_from(c).ok()?) as usize)
        };
        let (nt, ng, np, ns, nb) = (
            read_u64(0)?,
            read_u64(8)?,
            read_u64(16)?,
            read_u64(24)?,
            read_u64(32)?,
        );

        let mut off: usize = 40;
        let mut take = |n: usize| -> Option<Vec<f64>> {
            let end = off.checked_add(n.checked_mul(8_usize)?)?;
            let v: Option<Vec<f64>> = bytes.get(off..end)?.chunks_exact(8).map(read_f64).collect();
            off = end;
            v
        };
        let thetas = take(nt)?;
        let gammas = take(ng)?;
        let phases = take(np)?;
        let t_sss = take(ns)?;
        let wavelengths = take(nb)?;
        let factors = take(
            nt.checked_mul(ng)?
                .checked_mul(np)?
                .checked_mul(ns)?
                .checked_mul(nb)?,
        )?;
        Some(Self::from_factors(
            &thetas,
            &gammas,
            &phases,
            &t_sss,
            &wavelengths,
            factors,
        ))
    }

    /// The band wavelengths (nm) the table is tabulated at, one per band column.
    #[must_use]
    pub fn wavelengths(&self) -> &[f64] {
        &self.wavelengths
    }

    /// The tabulated thermal-parameter range `(min, max)`. Queries outside it are
    /// clamped by [`factor`](Self::factor).
    #[must_use]
    pub fn theta_range(&self) -> (f64, f64) {
        (
            self.log_thetas.first().map_or(0.0, |l| l.exp()),
            self.log_thetas.last().map_or(0.0, |l| l.exp()),
        )
    }

    /// The tabulated sub-solar-temperature range `(min, max)` in Kelvin. Queries outside
    /// it are clamped by [`factor`](Self::factor).
    #[must_use]
    pub fn t_ss_range(&self) -> (f64, f64) {
        (
            self.t_sss.first().copied().unwrap_or(0.0),
            self.t_sss.last().copied().unwrap_or(0.0),
        )
    }

    /// The correction factor for the band whose wavelength is nearest `wavelength`
    /// (nm), at sub-solar temperature `t_ss` (K). Used by the fitter, where each
    /// observation carries a single band and its own `T_ss`.
    #[must_use]
    pub fn factor_at_wavelength(
        &self,
        theta: f64,
        gamma: f64,
        phase: f64,
        t_ss: f64,
        wavelength: f64,
    ) -> f64 {
        let band = self
            .wavelengths
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - wavelength).abs().total_cmp(&(*b - wavelength).abs()))
            .map_or(0, |(i, _)| i);
        self.factor(theta, gamma, phase, t_ss, band)
    }
}

/// Canonical build geometry: object on +x from the Sun, observer at phase angle
/// `phase` in the x-y plane, pole on +z (so the sub-solar latitude is 0). Distance
/// and diameter cancel in the rough/smooth ratio.
fn canonical_geom(theta: f64, phase: f64, t_ss: f64) -> TpmGeom {
    let obj2sun_hat = Vector3::new(-1.0, 0.0, 0.0);
    let obj2obs_hat = Vector3::new(-phase.cos(), -phase.sin(), 0.0);
    TpmGeom {
        t_ss,
        theta,
        sub_solar_lat: 0.0,
        // The shipped correction table is built without solar multiple scattering (the
        // dark-body limit); the on-the-fly rough path applies it from the real albedo.
        bond_albedo: 0.0,
        obj2sun_hat,
        pole: Vector3::z(),
        obs2obj: -obj2obs_hat,
        obs2obj_r: 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frm_thermal_flux, neatm_thermal_flux};
    use std::f64::consts::FRAC_PI_4;

    /// Period-averaged insolation for a latitude band, the Theta -> inf limit.
    fn mean_insolation(lat: f64, sub_solar_lat: f64) -> f64 {
        let (s_lat, c_lat) = lat.sin_cos();
        let (s_sun, c_sun) = sub_solar_lat.sin_cos();
        let n = 100_000;
        let mut sum = 0.0;
        for j in 0..n {
            let tau = TAU * f64::from(j) / f64::from(n);
            sum += (s_lat * s_sun + c_lat * c_sun * tau.cos()).max(0.0);
        }
        sum / f64::from(n)
    }

    #[test]
    fn test_low_inertia_limit() {
        // Low inertia approaches instantaneous equilibrium u^4 = mu (NEATM-like).
        // The diurnal deficit/lag scales with Theta; check that it shrinks as Theta
        // decreases and is small at a low (but physical) inertia.
        let dtau = TAU / N_TIME as f64;
        let max_dev = |theta: f64| {
            let surface = solve_diurnal_surface(theta, 0.0, 0.0, &DEPTH_GRID).unwrap();
            surface
                .iter()
                .enumerate()
                .map(|(j, &u)| {
                    let mu = (dtau * j as f64).cos().max(0.0);
                    // only where there is appreciable illumination
                    if mu > 0.2 {
                        (u - mu.powf(0.25)).abs()
                    } else {
                        0.0
                    }
                })
                .fold(0.0, f64::max)
        };
        let dev_lo = max_dev(0.02);
        let dev_hi = max_dev(0.2);
        assert!(
            dev_lo < dev_hi,
            "deficit should shrink with Theta: {dev_lo} !< {dev_hi}"
        );
        assert!(dev_lo < 1.5e-2, "low-inertia deficit too large: {dev_lo}");
    }

    #[test]
    fn test_high_inertia_limit() {
        // Theta -> inf gives an isothermal column at the mean-insolation temperature.
        // The residual diurnal spread scales as ~1/Theta.
        let theta = 100.0;
        for &lat in &[0.0, 0.5, -0.7] {
            let surface = solve_diurnal_surface(theta, lat, 0.3, &DEPTH_GRID).unwrap();
            let max = surface.iter().copied().fold(f64::MIN, f64::max);
            let min = surface.iter().copied().fold(f64::MAX, f64::min);
            // nearly constant over the rotation
            assert!(max - min < 1.5e-2, "lat={lat}, spread={}", max - min);
            let mean = surface.iter().sum::<f64>() / surface.len() as f64;
            let expected = mean_insolation(lat, 0.3).powf(0.25);
            assert!(
                (mean - expected).abs() < 1e-2,
                "lat={lat}, mean={mean}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_energy_balance_closure() {
        // In periodic steady state the conducted flux integrates to zero over a
        // rotation, so emitted (u^4) must balance absorbed (mu): integral u^4 dtau ==
        // integral mu dtau.
        let dtau = TAU / N_TIME as f64;
        for &theta in &[0.1, 1.0, 5.0] {
            for &lat in &[0.0, 0.6, -0.4] {
                let sub_solar_lat = 0.2;
                let surface =
                    solve_diurnal_surface(theta, lat, sub_solar_lat, &DEPTH_GRID).unwrap();
                let (s_lat, c_lat) = lat.sin_cos();
                let (s_sun, c_sun) = sub_solar_lat.sin_cos();
                let mut emitted = 0.0;
                let mut absorbed = 0.0;
                for (j, &u) in surface.iter().enumerate() {
                    let tau = dtau * j as f64;
                    emitted += u.powi(4) * dtau;
                    absorbed += (s_lat * s_sun + c_lat * c_sun * tau.cos()).max(0.0) * dtau;
                }
                assert!(
                    (emitted - absorbed).abs() < 5e-3,
                    "theta={theta}, lat={lat}, emitted={emitted}, absorbed={absorbed}"
                );
            }
        }
    }

    #[test]
    fn test_zero_inertia_is_equilibrium() {
        // Theta = 0 is instantaneous radiative equilibrium u^4 = mu, returned in
        // closed form (no NaN from the infinitely stiff boundary).
        let dtau = TAU / N_TIME as f64;
        let surface = solve_diurnal_surface(0.0, 0.2, 0.3, &DEPTH_GRID).unwrap();
        for (j, &u) in surface.iter().enumerate() {
            let tau = dtau * j as f64;
            let mu = (0.2_f64.sin() * 0.3_f64.sin() + 0.2_f64.cos() * 0.3_f64.cos() * tau.cos())
                .max(0.0);
            assert!((u - mu.powf(0.25)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_infinite_theta_is_isothermal() {
        // Theta = inf (e.g. zero period) is isothermal at the mean-insolation
        // temperature, returned in closed form.
        let surface = solve_diurnal_surface(f64::INFINITY, 0.4, 0.1, &DEPTH_GRID).unwrap();
        // exactly constant over the rotation
        for &u in &surface {
            assert!((u - surface[0]).abs() < 1e-15);
        }
        // and equal to the mean-insolation temperature (quadrature-limited)
        let expected = mean_insolation(0.4, 0.1).powf(0.25);
        assert!((surface[0] - expected).abs() < 2e-3);
    }

    #[test]
    fn test_linear_thermal_wave() {
        // For a small sinusoidal insolation mu = m + a*cos(tau), linearized
        // thermal-wave theory gives the surface temperature oscillation in closed
        // form: amplitude |A| and phase lag arg(D), where the complex surface
        // response is A = a / D, D = 4*u0^3 + Theta*k, k = (1+i)/sqrt(2), and
        // u0 = m^(1/4). This validates the finite-Theta interior solver -- the
        // diurnal lag and amplitude that the Gamma->0 / Gamma->inf limits cannot
        // reach. A high latitude and sub-solar latitude keep the insolation unclipped
        // (polar day) and a << m (linear regime, single harmonic).
        let lat = 1.45_f64;
        let sslat = 1.45_f64;
        let (s_lat, c_lat) = lat.sin_cos();
        let (s_sun, c_sun) = sslat.sin_cos();
        let mean = s_lat * s_sun; // m
        let amp = c_lat * c_sun; // a
        assert!(
            mean - amp > 0.0,
            "insolation must stay positive (unclipped)"
        );
        let dtau = TAU / N_TIME as f64;
        let sqrt2 = 2.0_f64.sqrt();

        for &theta in &[0.5, 1.0, 3.0] {
            let surface = solve_diurnal_surface(theta, lat, sslat, &DEPTH_GRID).unwrap();

            // analytic surface response
            let u0 = mean.powf(0.25);
            let re_d = 4.0 * u0.powi(3) + theta / sqrt2;
            let im_d = theta / sqrt2;
            let analytic_amp = amp / re_d.hypot(im_d);
            let analytic_lag = im_d.atan2(re_d);

            // fundamental Fourier coefficient of the numerical surface curve.
            // surface[j] is the temperature at time (j + 1) * dtau.
            let mut proj_cos = 0.0;
            let mut proj_sin = 0.0;
            for (j, &u) in surface.iter().enumerate() {
                let t = (j as f64 + 1.0) * dtau;
                proj_cos += u * t.cos();
                proj_sin += u * t.sin();
            }
            let scale = 2.0 / N_TIME as f64;
            proj_cos *= scale;
            proj_sin *= scale;
            let measured_amp = proj_cos.hypot(proj_sin);
            let measured_lag = proj_sin.atan2(proj_cos);

            // The production solver matches the analytic thermal wave to well under
            // 0.1% in amplitude and ~0.001 rad in lag; assert with comfortable margin.
            assert!(
                (measured_amp - analytic_amp).abs() / analytic_amp < 5e-3,
                "theta={theta}: amplitude {measured_amp} vs analytic {analytic_amp}"
            );
            assert!(
                (measured_lag - analytic_lag).abs() < 5e-3,
                "theta={theta}: lag {measured_lag} vs analytic {analytic_lag}"
            );
        }
    }

    #[test]
    fn test_interior_scheme_second_order_mms() {
        // Method of Manufactured Solutions verifying the interior Crank-Nicolson scheme
        // is 2nd-order in time. The manufactured field is quadratic in depth -- for
        // which the non-uniform 3-point Laplacian is *exact* (it reproduces {1, x, x^2}
        // exactly) -- so the spatial discretization contributes zero error and the only
        // error is temporal. Adding the required source term makes the field the exact
        // solution of the semi-discrete system, so Crank-Nicolson must converge to it at
        // 2nd order as the time step halves (error ratio ~4). Reuses the production
        // Thomas solver and the same interior-coefficient formulas as `DepthGrid`.
        //
        //   u*(x,t) = phi(x) psi(t),  phi = 1 + x + 0.3 x^2,  psi = 1 + 0.5 sin(w t)
        //   u_t = phi psi',  u_xx = 0.6 psi  =>  source S = phi psi' - 0.6 psi
        let nz = N_DEPTH;
        // reconstruct the production non-uniform depth nodes
        let denom = DEPTH_STRETCH.exp() - 1.0;
        let x: Vec<f64> = (0..nz)
            .map(|i| {
                let t = i as f64 / (nz - 1) as f64;
                X_MAX * ((DEPTH_STRETCH * t).exp() - 1.0) / denom
            })
            .collect();

        let phi = |xi: f64| 1.0 + xi + 0.3 * xi * xi;
        let w = 1.5;
        let psi = |t: f64| 1.0 + 0.5 * (w * t).sin();
        let psi_dt = |t: f64| 0.5 * w * (w * t).cos();
        let u_exact = |xi: f64, t: f64| phi(xi) * psi(t);
        let source = |xi: f64, t: f64| phi(xi) * psi_dt(t) - 0.6 * psi(t);

        let t_end = 1.0;
        let run = |n_steps: usize| -> f64 {
            let dtau = t_end / n_steps as f64;
            // interior CN coefficients from the non-uniform Laplacian (same formulas as
            // DepthGrid), for this dtau.
            let mut alpha = vec![0.0; nz];
            let mut beta = vec![0.0; nz];
            let mut gamma = vec![0.0; nz];
            for i in 1..nz - 1 {
                let hm = x[i] - x[i - 1];
                let hp = x[i + 1] - x[i];
                let a = 2.0 / (hm * (hm + hp));
                let c = 2.0 / (hp * (hm + hp));
                alpha[i] = 0.5 * dtau * a;
                gamma[i] = 0.5 * dtau * c;
                beta[i] = -0.5 * dtau * (a + c);
            }
            let mut u: Vec<f64> = x.iter().map(|&xi| u_exact(xi, 0.0)).collect();
            let mut sub = vec![0.0; nz];
            let mut diag = vec![0.0; nz];
            let mut sup = vec![0.0; nz];
            let mut rhs = vec![0.0; nz];
            let mut out = vec![0.0; nz];
            for step in 0..n_steps {
                let tn = step as f64 * dtau;
                let tn1 = (step as f64 + 1.0) * dtau;
                // Dirichlet ends set to the exact manufactured values.
                diag[0] = 1.0;
                sup[0] = 0.0;
                rhs[0] = u_exact(x[0], tn1);
                sub[nz - 1] = 0.0;
                diag[nz - 1] = 1.0;
                rhs[nz - 1] = u_exact(x[nz - 1], tn1);
                for i in 1..nz - 1 {
                    sub[i] = -alpha[i];
                    diag[i] = 1.0 - beta[i];
                    sup[i] = -gamma[i];
                    let s_mid = 0.5 * (source(x[i], tn) + source(x[i], tn1));
                    rhs[i] = u[i]
                        + alpha[i] * u[i - 1]
                        + beta[i] * u[i]
                        + gamma[i] * u[i + 1]
                        + dtau * s_mid;
                }
                thomas(&sub, &mut diag, &sup, &mut rhs, &mut out);
                u.copy_from_slice(&out);
            }
            u.iter()
                .zip(&x)
                .map(|(&ui, &xi)| (ui - u_exact(xi, t_end)).abs())
                .fold(0.0, f64::max)
        };

        let e1 = run(40);
        let e2 = run(80);
        let e3 = run(160);
        let r1 = e1 / e2;
        let r2 = e2 / e3;
        assert!(
            r1 > 3.5 && r2 > 3.5,
            "Crank-Nicolson time convergence ratios {r1}, {r2} (expect ~4 for 2nd order); \
             errors {e1:e}, {e2:e}, {e3:e}"
        );
    }

    #[test]
    fn test_crater_smooth_limit() {
        // gamma -> 0: the crater collapses to a flat facet, so the area-weighted mean
        // micro-facet temperature matches the smooth single-column solution and
        // self-heating vanishes.
        let theta = 1.0;
        let lat = 0.3;
        let sslat = 0.2;
        let smooth = solve_diurnal_surface(theta, lat, sslat, &DEPTH_GRID).unwrap();
        let crater = Crater::new(0.08, 8, 16);
        let rough = solve_crater(theta, lat, sslat, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();
        let areas = crater.areas();
        let total: f64 = areas.iter().sum();
        for j in 0..N_TIME {
            let mean: f64 = rough.iter().zip(areas).map(|(s, a)| s[j] * a).sum::<f64>() / total;
            assert!(
                (mean - smooth[j]).abs() < 2e-2,
                "j={j}: crater mean {mean} vs smooth {}",
                smooth[j]
            );
        }
    }

    #[test]
    fn test_crater_self_heating_warms() {
        // Intra-crater self-heating only adds energy, so the coldest (shadowed /
        // night-side) micro-facets are warmer with self-heating on than off.
        let theta = 1.0;
        let crater = Crater::new(FRAC_PI_2, 8, 16); // hemisphere: strong cavity effect
        let with = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();
        let without = solve_crater(theta, 0.0, 0.0, &crater, 0.0, 0.0, &DEPTH_GRID).unwrap();
        let min_of = |s: &[Vec<f64>]| s.iter().flatten().copied().fold(f64::MAX, f64::min);
        let min_with = min_of(&with);
        let min_without = min_of(&without);
        assert!(
            min_with > min_without + 1e-3,
            "self-heating should warm the coldest facets: {min_with} vs {min_without}"
        );
    }

    #[test]
    fn test_crater_solar_scattering_warms() {
        // In-crater solar multiple scattering only adds absorbed energy, so a nonzero
        // bond albedo (some sunlight reflected and partly recaptured) warms the crater
        // everywhere versus no scattering. An exaggerated albedo makes the otherwise
        // small effect unambiguous.
        let theta = 1.0;
        let crater = Crater::new(FRAC_PI_2, 8, 16); // hemisphere: strong cavity effect
        let with = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.3, &DEPTH_GRID).unwrap();
        let without = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();
        let mut warmer_somewhere = false;
        for (sw, so) in with.iter().flatten().zip(without.iter().flatten()) {
            assert!(*sw >= *so - 1e-9, "scattering must not cool: {sw} vs {so}");
            if *sw > *so + 1e-6 {
                warmer_somewhere = true;
            }
        }
        assert!(warmer_somewhere, "solar scattering should warm the crater");
    }

    #[test]
    fn test_crater_scattering_vanishes_as_gamma_shrinks() {
        // As the cavity closes (gamma -> 0) the recaptured scattered fraction -> 0, so
        // the solar-scattering warming shrinks: a shallow crater is barely affected by
        // albedo while a hemisphere is warmed measurably.
        let theta = 1.0;
        let warming = |gamma: f64| {
            let crater = Crater::new(gamma, 8, 16);
            let hot = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.3, &DEPTH_GRID).unwrap();
            let cold = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();
            // peak noon micro-facet warming
            hot.iter()
                .flatten()
                .zip(cold.iter().flatten())
                .map(|(a, b)| a - b)
                .fold(0.0, f64::max)
        };
        assert!(
            warming(0.1) < warming(FRAC_PI_2),
            "scattering warming should grow with cavity depth"
        );
    }

    #[test]
    fn test_solar_scattering_matches_picard() {
        // Verification: the closed-form scattered-heating scalar folded into `direct`,
        //   q = A * recapture(direct) / (1 - A * R),
        // sums an infinite series of cavity bounces. Check it against an independent
        // bounce-by-bounce Picard iteration of the same recapture operator (which is
        // itself validated separately by test_recapture_uniform_radiosity), so this
        // isolates the geometric-series algebra (the (1-A) cancellation and the
        // 1/(1-A*R) denominator).
        let crater = Crater::new(1.2, 10, 20);
        let a = 0.2_f64; // bond albedo
        let sun = Vector3::new(0.3, 0.1, 0.95).normalize();
        let direct = crater.direct_factor(&sun);
        let n = direct.len();

        // closed form, exactly as solve_crater applies it
        let r_uniform = crater.recapture_irradiance(&vec![1.0; n]);
        let q_closed = a * crater.recapture_irradiance(&direct) / (1.0 - a * r_uniform);

        // independent Picard: incident I = direct/(1-A); iterate the uniform scattered
        // irradiance H = recapture(A*I + A*H) to a fixed point; extra absorbed = (1-A)*H.
        let incident: Vec<f64> = direct.iter().map(|d| d / (1.0 - a)).collect();
        let mut h = 0.0_f64;
        for _ in 0..10_000 {
            let radiosity: Vec<f64> = incident.iter().map(|i| a * i + a * h).collect();
            let h_new = crater.recapture_irradiance(&radiosity);
            if (h_new - h).abs() < 1e-15 {
                h = h_new;
                break;
            }
            h = h_new;
        }
        let q_picard = (1.0 - a) * h;

        assert!(
            (q_closed - q_picard).abs() < 1e-10,
            "closed form {q_closed} vs Picard bounces {q_picard}"
        );
        assert!(q_closed > 0.0, "scattered heating must be positive");
    }

    #[test]
    fn test_crater_render_smooth_limit() {
        // gamma -> 0: the crater's directional emission per flat area reduces to the
        // smooth Lambertian bb(T) * (n . obs).
        let theta = 1.0;
        let lat = 0.0;
        let sslat = 0.0;
        let crater = Crater::new(0.08, 8, 16);
        let field = solve_crater(theta, lat, sslat, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();
        let smooth = solve_diurnal_surface(theta, lat, sslat, &DEPTH_GRID).unwrap();
        let t_ss = 300.0;
        let wl = 11000.0;

        let jt = 0; // noon
        let temps: Vec<f64> = field.iter().map(|s| s[jt]).collect();
        let obs = Vector3::new(0.2_f64, 0.0, 0.98).normalize();
        let vpa = crater.visible_projected_area(&obs);
        let rough = crater_emission(&crater, &vpa, &temps, t_ss, wl, None);
        let smooth_em = black_body_flux(t_ss * smooth[jt], wl) * obs.z;
        assert!(
            (rough - smooth_em).abs() / smooth_em < 3e-2,
            "rough {rough} vs smooth {smooth_em}"
        );
    }

    #[test]
    fn test_crater_beams_toward_sun() {
        // At a tilted sun, the crater is brighter viewed from the sunward azimuth
        // (sees the hot sunlit far wall) than the anti-sunward azimuth (cold shadowed
        // wall) at the same emission angle -- the thermal beaming signature.
        let theta = 1.0;
        let crater = Crater::new(FRAC_PI_2, 12, 24); // hemisphere: strong beaming
        let field = solve_crater(theta, 0.0, 0.0, &crater, 0.9, 0.0, &DEPTH_GRID).unwrap();

        // tau = 60 deg: sun_in_crater_frame(0,0,tau) = (-sin tau, 0, cos tau), tilted.
        let jt = 60;
        let temps: Vec<f64> = field.iter().map(|s| s[jt]).collect();
        let e = FRAC_PI_4;
        let obs_sunward = Vector3::new(-e.sin(), 0.0, e.cos()); // sun's azimuth
        let obs_antisun = Vector3::new(e.sin(), 0.0, e.cos()); // opposite
        let vpa_sun = crater.visible_projected_area(&obs_sunward);
        let vpa_anti = crater.visible_projected_area(&obs_antisun);
        let em_sun = crater_emission(&crater, &vpa_sun, &temps, 300.0, 11000.0, None);
        let em_anti = crater_emission(&crater, &vpa_anti, &temps, 300.0, 11000.0, None);
        assert!(
            em_sun > em_anti * 1.05,
            "crater should beam toward the sun: sunward {em_sun} vs anti {em_anti}"
        );
    }

    #[test]
    fn test_rough_render_smooth_limit() {
        // gamma -> 0: the full rough disk render reduces to the smooth disk render.
        // (The directional beaming itself is validated per-crater in
        // test_crater_beams_toward_sun; the disk-level rough/smooth ratio mixes
        // beaming with cavity self-heating and is not monotonic in phase.)
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            period: 6.0 * 3600.0,
            phase0: 0.0,
            epoch0: 0.0,
        };
        let shape = TpmShape::sphere();
        let thermal = ThermalParams {
            thermal_inertia: 150.0,
            emissivity: 0.9,
        };
        let crater = Crater::new(0.08, 6, 12); // nearly flat
        let bands = [BandInfo::WISE[2], BandInfo::WISE[3]];
        let sun2obj = Vector3::new(1.5, 0.0, 0.0);
        let sun2obs = Vector3::new(0.5, 0.3, 0.0);

        let geom = tpm_geom(&spin, &thermal, 0.05, 0.15, &sun2obj, &sun2obs).unwrap();
        let field = DiurnalField::solve(geom.theta, geom.sub_solar_lat, &DEPTH_GRID).unwrap();
        let smooth = render_thermal_flux(
            |lat, tau| field.sample(lat, tau),
            &geom,
            &shape,
            0.0,
            1.0,
            0.9,
            &bands,
        );
        let rough =
            render_rough_thermal_flux(&geom, &shape, 0.0, 1.0, 0.9, &bands, &crater, &DEPTH_GRID)
                .unwrap();
        for (s, r) in smooth.iter().zip(&rough) {
            assert!(
                (s - r).abs() / s < 0.05,
                "rough {r} vs smooth {s} should agree as gamma -> 0"
            );
        }
    }

    #[test]
    fn test_roughness_correction_interpolation() {
        // mechanics only (no renders): a 2x2x2x2 single-band table over
        // (theta, gamma, phase, t_ss); check that grid nodes return the stored value
        // and the cube centre multilinearly interpolates to the mean of the corners.
        let thetas = [1.0, 10.0];
        let gammas = [0.2, 0.8];
        let phases = [0.0, FRAC_PI_2];
        let t_sss = [200.0, 400.0];
        // factors row-major over (theta, gamma, phase, t_ss, band=1)
        let factors: Vec<f64> = (0..16).map(|i| 1.0 + 0.02 * f64::from(i)).collect();
        let rc = RoughnessCorrection::from_factors(
            &thetas,
            &gammas,
            &phases,
            &t_sss,
            &[11000.0],
            factors.clone(),
        );

        // grid nodes reproduce stored values (first and last corner)
        assert!((rc.factor(1.0, 0.2, 0.0, 200.0, 0) - factors[0]).abs() < 1e-12);
        assert!((rc.factor(10.0, 0.8, FRAC_PI_2, 400.0, 0) - factors[15]).abs() < 1e-12);
        // clamping below/above every axis
        assert!((rc.factor(0.01, 0.2, 0.0, 50.0, 0) - factors[0]).abs() < 1e-12);
        assert!((rc.factor(1e6, 0.8, FRAC_PI_2, 1e4, 0) - factors[15]).abs() < 1e-12);
        // centre of the 4-cube = mean of the 16 corners
        let mid = rc.factor((1.0_f64 * 10.0).sqrt(), 0.5, FRAC_PI_4, 300.0, 0);
        let mean = factors.iter().sum::<f64>() / 16.0;
        assert!((mid - mean).abs() < 1e-9, "mid {mid} vs mean {mean}");
    }

    #[test]
    fn test_mean_slope_angle_matches_quadrature() {
        // Area-weighted mean slope of the spherical cap: slope PDF sin(theta) on
        // [0, gamma]. Compare the closed form to direct numerical quadrature.
        let quad = |gamma: f64| {
            let n = 200_000;
            let (mut num, mut den) = (0.0, 0.0);
            for j in 0..n {
                let th = gamma * (f64::from(j) + 0.5) / f64::from(n);
                let w = th.sin();
                num += th * w;
                den += w;
            }
            num / den
        };
        for &g in &[0.2_f64, 0.5, 0.8, 1.0, 1.3, FRAC_PI_2] {
            let cf = mean_slope_angle(g);
            let q = quad(g);
            assert!((cf - q).abs() < 1e-4, "gamma={g}: closed {cf} vs quad {q}");
        }
        // small-angle limit theta_bar -> (2/3) gamma, and exact 1 rad ceiling.
        let g = 1e-4;
        assert!((mean_slope_angle(g) - 2.0 / 3.0 * g).abs() < 1e-12);
        assert!((mean_slope_angle(FRAC_PI_2) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gamma_from_mean_slope_round_trip() {
        // mean_slope_angle is strictly increasing and the inverse recovers gamma.
        let mut prev = -1.0;
        for k in 1..=20 {
            let g = FRAC_PI_2 * f64::from(k) / 20.0;
            let tb = mean_slope_angle(g);
            assert!(tb > prev, "mean slope must increase with gamma");
            prev = tb;
            let g_back = gamma_from_mean_slope(tb);
            assert!(
                (g_back - g).abs() < 1e-6,
                "round trip gamma {g} -> {g_back}"
            );
        }
        // clamps outside the representable (0, 1] rad range.
        assert_eq!(gamma_from_mean_slope(-0.1), 0.0);
        assert!((gamma_from_mean_slope(1.5) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_rms_slope_exceeds_mean_and_is_finite() {
        // RMS slope >= mean slope, finite for gamma < pi/2, and matches the closed form
        // at gamma = 45 deg (~0.572 rad).
        for &g in &[0.2_f64, 0.5, FRAC_PI_4, 1.0, 1.4] {
            let rms = rms_slope(g);
            assert!(
                rms.is_finite() && rms >= mean_slope_angle(g) - 1e-9,
                "gamma={g}: rms {rms}"
            );
        }
        assert!((rms_slope(FRAC_PI_4) - 0.5724).abs() < 2e-3);
    }

    #[test]
    fn test_shipped_roughness_table_loads() {
        // the embedded table parses and gives a plausible thermal enhancement
        let rc = RoughnessCorrection::shipped();
        let f = rc.factor_at_wavelength(5.0, 0.7, FRAC_PI_2, 300.0, 11000.0);
        assert!(f.is_finite() && f > 0.9 && f < 2.0, "shipped factor {f}");
    }

    #[test]
    fn test_roughness_correction_temperature_dependence() {
        // The shipped table must actually vary with T_ss in a band that sits on the
        // Wien side of the temperature distribution (W3): a hotter object beams less
        // than a cold one at the same (Theta, gamma, phase). A flat (single-T_ss)
        // table would make these identical.
        let rc = RoughnessCorrection::shipped();
        let cold = rc.factor_at_wavelength(2.0, 0.9, 0.0, 220.0, 11000.0);
        let hot = rc.factor_at_wavelength(2.0, 0.9, 0.0, 400.0, 11000.0);
        assert!(
            cold > hot,
            "W3 beaming should weaken with temperature: {cold} vs {hot}"
        );
        assert!(
            (cold - hot).abs() > 0.01,
            "temperature effect too small: {cold} vs {hot}"
        );
    }

    #[test]
    fn test_roughness_correction_bytes_roundtrip() {
        let thetas = [1.0, 10.0];
        let gammas = [0.2, 0.8];
        let phases = [0.0, FRAC_PI_2];
        let t_sss = [220.0, 380.0];
        let wavelengths = [11000.0, 22000.0];
        let factors: Vec<f64> = (0..32).map(|i| 1.0 + 0.01 * f64::from(i)).collect();
        let rc = RoughnessCorrection::from_factors(
            &thetas,
            &gammas,
            &phases,
            &t_sss,
            &wavelengths,
            factors,
        );
        let back = RoughnessCorrection::from_bytes(&rc.to_bytes()).unwrap();
        // a few sample points must match exactly
        for &(t, g, p, s, w) in &[
            (1.0, 0.2, 0.0, 220.0, 11000.0),
            (10.0, 0.8, FRAC_PI_2, 380.0, 22000.0),
        ] {
            let a = rc.factor_at_wavelength(t, g, p, s, w);
            let b = back.factor_at_wavelength(t, g, p, s, w);
            assert!((a - b).abs() < 1e-15, "{a} vs {b}");
        }
        assert!(RoughnessCorrection::from_bytes(&[0_u8; 8]).is_none());
    }

    #[test]
    #[ignore = "generates the shipped roughness table (slow); writes src/roughness_table.bin"]
    fn generate_roughness_table() {
        // Real table over (Theta, gamma, phase, T_ss) for the WISE bands. R is convex
        // (it curves through the beaming peak), so Theta and phase are sampled finely
        // enough to keep linear-interpolation error ~1%; gamma is smoother but still
        // refined. The T_ss axis captures the Wien dependence of the per-band beaming
        // (the crater emits a sum of Planck functions, nonlinear in absolute
        // temperature) and spans the NEA/MBA range. The phase and T_ss axes are cheap
        // -- the heat solve is shared across them (see build()). Built at n_time=180:
        // the stored rough/smooth ratio cancels the terminator-sampling bias.
        let lt0 = 0.05_f64.ln();
        let lt1 = 50.0_f64.ln();
        let thetas: Vec<f64> = (0..8)
            .map(|i| (lt0 + (lt1 - lt0) * f64::from(i) / 7.0).exp())
            .collect();
        let gammas: Vec<f64> = (0..6)
            .map(|i| (10.0 + 80.0 * f64::from(i) / 5.0).to_radians())
            .collect();
        let phases: Vec<f64> = (0..9).map(|i| PI * f64::from(i) / 8.0).collect();
        // 150-450 K in 50 K steps spans cool main-belt to hot near-Earth sub-solar
        // temperatures; linear interpolation over this smooth axis is ~1%.
        let t_sss: Vec<f64> = (0..7).map(|i| 150.0 + 50.0 * f64::from(i)).collect();
        let bands = BandInfo::WISE;
        let rc =
            RoughnessCorrection::build(&thetas, &gammas, &phases, &t_sss, 0.9, &bands, 6, 12, 180)
                .unwrap();
        let bytes = rc.to_bytes();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/roughness_table.bin");
        std::fs::write(path, &bytes).unwrap();
        eprintln!(
            "wrote {} bytes ({} nodes) to {path}",
            bytes.len(),
            thetas.len() * gammas.len() * phases.len() * t_sss.len()
        );
    }

    #[test]
    #[ignore = "slow: builds a roughness table via on-the-fly rough renders"]
    fn test_roughness_correction_build_roundtrip() {
        // Build a tiny table, then confirm that at a grid node the smooth flux times
        // the correction reproduces the directly-rendered rough flux.
        let bands = [BandInfo::WISE[2], BandInfo::WISE[3]];
        let thetas = [1.0, 6.0];
        let gammas = [0.7];
        let phases = [0.3, 1.2];
        let t_ss = 260.0;
        let t_sss = [t_ss];
        let rc = RoughnessCorrection::build(
            &thetas, &gammas, &phases, &t_sss, 0.9, &bands, 8, 16, N_TIME,
        )
        .unwrap();

        // at the (theta=6, gamma=0.7, phase=1.2, t_ss) node, smooth*R == rough
        let geom = canonical_geom(6.0, 1.2, t_ss);
        let field = DiurnalField::solve(6.0, 0.0, &DEPTH_GRID).unwrap();
        let shape = TpmShape::sphere();
        let mut smooth = render_thermal_flux(
            |lat, tau| field.sample(lat, tau),
            &geom,
            &shape,
            0.0,
            1.0,
            0.9,
            &bands,
        );
        let crater = Crater::new(0.7, 8, 16);
        let rough =
            render_rough_thermal_flux(&geom, &shape, 0.0, 1.0, 0.9, &bands, &crater, &DEPTH_GRID)
                .unwrap();
        rc.apply(&mut smooth, 6.0, 0.7, 1.2, t_ss);
        for (a, b) in smooth.iter().zip(&rough) {
            assert!((a - b).abs() / b < 1e-6, "corrected {a} vs rough {b}");
        }
    }

    #[test]
    fn test_zero_inertia_matches_neatm_disk_flux() {
        // Absolute-calibration anchor: at Gamma = 0 the TPM is instantaneous radiative
        // equilibrium (u^4 = mu), which is exactly NEATM with beaming eta = 1. The
        // disk-integrated thermal flux must therefore match `neatm_thermal_flux` at
        // eta = 1 -- this pins the whole rendering chain (projected areas, the
        // emissivity*pi factor, the distance scaling, the Planck assembly) against the
        // independently implemented, analytically-quadratured NEATM path. The residual
        // is the TPM's facet sum + (lat, local-time) field resolution vs NEATM's exact
        // azimuthal quadrature. The Gamma = 0 field is pole-independent, so any pole
        // works; use a fine sphere to keep the facet-sum error small.
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.2, 0.1, 1.0)),
            period: 6.0 * 3600.0,
            phase0: 0.0,
            epoch0: 0.0,
        };
        let shape = TpmShape::sphere_with_facets(8192);
        let thermal = ThermalParams {
            thermal_inertia: 0.0, // Gamma = 0 -> Theta = 0 -> u^4 = mu (NEATM, eta = 1)
            emissivity: 0.9,
        };
        let bands = BandInfo::WISE;
        let (vis_albedo, g_param, diameter) = (0.05, 0.15, 5.0);

        // a spread of phase angles (observer off the sun line)
        for &(s2o, s2b) in &[
            (Vector3::new(1.4, 0.0, 0.0), Vector3::new(0.4, 0.0, 0.0)),
            (Vector3::new(1.8, 0.3, 0.1), Vector3::new(0.9, -0.2, 0.0)),
            (Vector3::new(2.2, -0.5, 0.2), Vector3::new(1.1, 0.4, -0.1)),
        ] {
            let tpm = tpm_thermal_flux(
                &bands, &spin, &shape, &thermal, diameter, vis_albedo, g_param, &s2o, &s2b, 0.0,
            )
            .unwrap();
            let neatm =
                neatm_thermal_flux(&bands, diameter, vis_albedo, g_param, 1.0, 0.9, &s2o, &s2b);
            for (t, n) in tpm.iter().zip(&neatm) {
                let rel = (t - n).abs() / n;
                assert!(
                    rel < 0.01,
                    "TPM(Gamma=0) {t} vs NEATM(eta=1) {n}, rel={rel:.4}"
                );
            }
        }
    }

    #[test]
    fn test_infinite_inertia_matches_frm_disk_flux() {
        // Absolute-calibration anchor at the opposite limit: Theta -> inf is isothermal
        // at the mean-insolation temperature per latitude band, which is the FRM. kete's
        // `frm_thermal_flux` hard-codes the pole to ecliptic +z and uses
        // cos(lat - sub_solar_lat)^(1/4); that equals the mean-insolation limit only
        // when the sub-solar latitude is 0, so the Sun must lie in the equatorial
        // (x-y) plane. Using the same default sphere facets as FRM makes the comparison
        // facet-for-facet up to the (lat, local-time) field interpolation. period = 0
        // drives Theta -> inf via the closed-form isothermal branch.
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            period: 0.0, // Theta -> inf (isothermal closed form)
            phase0: 0.0,
            epoch0: 0.0,
        };
        let shape = TpmShape::sphere();
        let thermal = ThermalParams {
            thermal_inertia: 200.0,
            emissivity: 0.9,
        };
        let bands = BandInfo::WISE;
        let (vis_albedo, g_param, diameter) = (0.05, 0.15, 5.0);

        // sub-solar latitude 0: Sun in the x-y plane; observer also in-plane.
        for &(s2o, s2b) in &[
            (Vector3::new(1.5, 0.0, 0.0), Vector3::new(0.5, 0.2, 0.0)),
            (Vector3::new(1.9, 0.4, 0.0), Vector3::new(1.0, -0.3, 0.0)),
        ] {
            let tpm = tpm_thermal_flux(
                &bands, &spin, &shape, &thermal, diameter, vis_albedo, g_param, &s2o, &s2b, 0.0,
            )
            .unwrap();
            let frm = frm_thermal_flux(&bands, diameter, vis_albedo, g_param, 0.9, &s2o, &s2b);
            for (t, f) in tpm.iter().zip(&frm) {
                let rel = (t - f).abs() / f;
                assert!(rel < 0.01, "TPM(Theta=inf) {t} vs FRM {f}, rel={rel:.4}");
            }
        }
    }

    #[test]
    fn test_stiff_night_side_finite() {
        // Very low inertia (stiff Theta) viewed on the night side must stay finite:
        // the surface plunges toward zero at sunset without diverging.
        for &theta in &[1e-3, 1e-2, 0.1] {
            let surface = solve_diurnal_surface(theta, 0.3, 0.0, &DEPTH_GRID).unwrap();
            assert!(
                surface.iter().all(|u| u.is_finite() && *u >= 0.0),
                "theta={theta} produced a non-finite or negative temperature"
            );
        }
    }

    #[test]
    fn test_cached_field_matches_direct() {
        // An interpolated field from the grid should closely match a direct solve.
        let grid = TpmFieldGrid::new(0.1, 50.0).unwrap();
        for &(theta, sslat) in &[(0.3, 0.0), (1.5, 0.4), (12.0, -0.6), (30.0, 0.2)] {
            let (corners, wi, wj) = grid.corner_fields(theta, sslat);
            let direct = DiurnalField::solve(theta, sslat, &DEPTH_GRID).unwrap();
            for &lat in &[0.25, -0.5, 0.8] {
                for &tau in &[0.0, 1.5, 3.0, 4.5] {
                    let a = sample_corners(&corners, wi, wj, lat, tau);
                    let b = direct.sample(lat, tau);
                    // pointwise field interpolation error; largest in cold, highly
                    // structured high-latitude winter regions (small contribution to
                    // integrated flux -- see test_cached_flux_matches_direct).
                    assert!(
                        (a - b).abs() < 4e-2,
                        "theta={theta}, sslat={sslat}, lat={lat}, tau={tau}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "facet count accuracy/speed study"]
    fn facet_count_study() {
        // Disk-integrated flux accuracy and cached-render cost vs sphere facet count,
        // relative to a high-resolution reference. Drives the sampling facet choice.
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.2, 0.1, 1.0)),
            period: 6.0 * 3600.0,
            phase0: 0.0,
            epoch0: 0.0,
        };
        let thermal = ThermalParams {
            thermal_inertia: 180.0,
            emissivity: 0.9,
        };
        let bands = BandInfo::WISE;
        let grid = TpmFieldGrid::new(0.1, 50.0).unwrap();
        let geoms = [
            (Vector3::new(2.4, 0.3, 0.1), Vector3::new(1.2, 0.1, 0.0)),
            (Vector3::new(1.8, -0.5, 0.2), Vector3::new(0.9, -0.2, 0.1)),
        ];
        let flux_at = |n: u32, s2o: &Vector3<f64>, s2b: &Vector3<f64>| {
            let shape = TpmShape::sphere_with_facets(n);
            tpm_thermal_flux_cached(
                &grid, &bands, &spin, &shape, &thermal, 5.0, 0.1, 0.15, s2o, s2b, 0.0,
            )
        };
        let reference: Vec<Vec<f64>> = geoms.iter().map(|(a, b)| flux_at(8192, a, b)).collect();
        for &n in &[128_u32, 256, 512, 1024, 2048] {
            let shape = TpmShape::sphere_with_facets(n);
            let (s2o, s2b) = &geoms[0];
            let t0 = std::time::Instant::now();
            for _ in 0..2000 {
                let _ = tpm_thermal_flux_cached(
                    &grid, &bands, &spin, &shape, &thermal, 5.0, 0.1, 0.15, s2o, s2b, 0.0,
                );
            }
            let us = t0.elapsed().as_secs_f64() / 2000.0 * 1e6;
            let mut worst = 0.0_f64;
            for ((s2o, s2b), refl) in geoms.iter().zip(&reference) {
                let f = flux_at(n, s2o, s2b);
                for (a, r) in f.iter().zip(refl) {
                    worst = worst.max(100.0 * (a - r).abs() / r);
                }
            }
            eprintln!("FACETS n={n:5} render={us:6.1}us worst_relerr={worst:.2}%");
        }
    }

    #[test]
    fn test_cached_flux_matches_direct() {
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            period: 6.0 * 3600.0,
            phase0: 0.0,
            epoch0: 0.0,
        };
        let shape = TpmShape::sphere();
        let thermal = ThermalParams {
            thermal_inertia: 150.0,
            emissivity: 0.9,
        };
        let bands = [BandInfo::WISE[2], BandInfo::WISE[3]];
        let sun2obj = Vector3::new(1.5, 0.0, 0.0);
        let sun2obs = Vector3::new(0.5, 0.4, 0.0);

        // grid must span the Theta this geometry produces
        let grid = TpmFieldGrid::new(0.05, 100.0).unwrap();
        let direct = tpm_thermal_flux(
            &bands, &spin, &shape, &thermal, 1.0, 0.05, 0.15, &sun2obj, &sun2obs, 0.0,
        )
        .unwrap();
        let cached = tpm_thermal_flux_cached(
            &grid, &bands, &spin, &shape, &thermal, 1.0, 0.05, 0.15, &sun2obj, &sun2obs, 0.0,
        );
        for (d, c) in direct.iter().zip(&cached) {
            let rel = (d - c).abs() / d;
            assert!(rel < 0.02, "direct={d}, cached={c}, rel={rel}");
        }
    }

    #[test]
    fn test_total_flux_basic() {
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            period: 6.0 * 3600.0,
            phase0: 0.0,
            epoch0: 0.0,
        };
        let shape = TpmShape::sphere();
        let thermal = ThermalParams {
            thermal_inertia: 200.0,
            emissivity: 0.9,
        };
        let bands = [BandInfo::WISE[2], BandInfo::WISE[3]];
        let albedos = [0.05, 0.05];
        let sun2obj = Vector3::new(1.5, 0.0, 0.0);
        let sun2obs = Vector3::new(0.5, 0.1, 0.0);

        let res = tpm_total_flux(
            &bands, &albedos, &spin, &shape, &thermal, 1.0, 0.05, 0.15, 18.0, &sun2obj, &sun2obs,
            0.0,
        )
        .unwrap();
        for f in &res.fluxes {
            assert!(f.is_finite() && *f > 0.0, "flux={f}");
        }
        // thermal should dominate in the WISE W3/W4 bands for a dark 1 km object.
        for (t, total) in res.thermal_fluxes.iter().zip(&res.fluxes) {
            assert!(t / total > 0.5);
        }
    }

    fn flux_at_phase(shape: &TpmShape, phase0: f64) -> f64 {
        let spin = SpinState {
            pole: UnitVector3::new_normalize(Vector3::new(0.0, 0.0, 1.0)),
            period: 6.0 * 3600.0,
            phase0,
            epoch0: 0.0,
        };
        let thermal = ThermalParams {
            thermal_inertia: 150.0,
            emissivity: 0.9,
        };
        let bands = [BandInfo::WISE[3]];
        // observer off the equator so the long axis sweeps through the line of sight
        let sun2obj = Vector3::new(1.5, 0.0, 0.0);
        let sun2obs = Vector3::new(0.5, 0.4, 0.0);
        tpm_thermal_flux(
            &bands, &spin, shape, &thermal, 1.0, 0.05, 0.15, &sun2obj, &sun2obs, 0.0,
        )
        .unwrap()[0]
    }

    #[test]
    fn test_oblate_is_phase_independent() {
        // An oblate spheroid (a = b) is rotationally symmetric: rotation phase has no
        // effect on the disk-integrated flux.
        let shape = TpmShape::ellipsoid(1.0, 1.0, 0.6);
        let f0 = flux_at_phase(&shape, 0.0);
        let f1 = flux_at_phase(&shape, 1.0);
        let f2 = flux_at_phase(&shape, 2.5);
        assert!((f0 - f1).abs() / f0 < 1e-6, "{f0} vs {f1}");
        assert!((f0 - f2).abs() / f0 < 1e-6, "{f0} vs {f2}");
    }

    #[test]
    fn test_oblate_flux_rotation_invariant() {
        // A rigid rotation of the whole system (pole + sun + observer) cannot change the
        // disk-integrated flux. This fails if the oblate is flattened along a fixed axis
        // (e.g. ecliptic z) instead of the spin pole -- the bug fixed by `body_axes`.
        let shape = TpmShape::ellipsoid(1.0, 1.0, 0.55); // oblate, pole = body z
        let thermal = ThermalParams {
            thermal_inertia: 150.0,
            emissivity: 0.9,
        };
        let bands = [BandInfo::WISE[2], BandInfo::WISE[3]];
        let sun2obj = Vector3::new(1.8, 0.3, 0.1);
        let sun2obs = Vector3::new(0.9, -0.2, 0.4);
        let pole = UnitVector3::new_normalize(Vector3::new(0.2, 0.1, 1.0));

        let flux = |pole: UnitVector3<f64>, s2o: Vector3<f64>, s2b: Vector3<f64>| {
            let spin = SpinState {
                pole,
                period: 6.0 * 3600.0,
                phase0: 0.0,
                epoch0: 0.0,
            };
            tpm_thermal_flux(
                &bands, &spin, &shape, &thermal, 5.0, 0.05, 0.15, &s2o, &s2b, 0.0,
            )
            .unwrap()
        };

        let base = flux(pole, sun2obj, sun2obs);

        // rotate the entire configuration rigidly about an arbitrary axis
        let rot = nalgebra::Rotation3::from_axis_angle(
            &UnitVector3::new_normalize(Vector3::new(0.4, -0.7, 0.5)),
            1.1,
        );
        let rotated = flux(
            UnitVector3::new_normalize(rot * pole.into_inner()),
            rot * sun2obj,
            rot * sun2obs,
        );

        // Exact in the continuum; the residual here is only azimuthal facet sampling.
        for (a, b) in base.iter().zip(&rotated) {
            let rel = (a - b).abs() / a;
            assert!(
                rel < 5e-3,
                "rotation changed oblate flux by {rel:.1e}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_ellipsoid_template_matches_direct() {
        // The cheap template remap must reproduce the direct ellipsoid builder
        // facet-for-facet (same topology, normals recomputed from scaled vertices).
        let n_div = 12;
        let template = EllipsoidTemplate::new(n_div);
        for &(a, b, c) in &[(1.0, 1.0, 0.6), (2.0, 1.0, 0.8), (1.0, 0.7, 0.5)] {
            let direct = TpmShape::ellipsoid_with_div(n_div, a, b, c);
            let remapped = template.shape(a, b, c);
            assert_eq!(remapped.facets.len(), direct.facets.len());
            assert_eq!(remapped.axisymmetric, direct.axisymmetric);
            for (r, d) in remapped.facets.iter().zip(&direct.facets) {
                assert!(
                    (r.area - d.area).abs() < 1e-12,
                    "area {} vs {}",
                    r.area,
                    d.area
                );
                let dot = r.normal.dot(&d.normal);
                assert!(dot > 1.0 - 1e-12, "normal mismatch, dot={dot}");
            }
        }
        // total area is normalized to 1
        let total: f64 = template
            .shape(2.0, 1.3, 0.7)
            .facets
            .iter()
            .map(|f| f.area)
            .sum();
        assert!((total - 1.0).abs() < 1e-12, "total area {total}");
    }

    #[test]
    fn test_triaxial_has_rotational_lightcurve() {
        // A triaxial ellipsoid (a != b) is not symmetric, so the disk-integrated
        // thermal flux varies with rotation phase.
        let shape = TpmShape::ellipsoid(2.0, 1.0, 0.8);
        let fluxes: Vec<f64> = (0..8)
            .map(|i| flux_at_phase(&shape, TAU * f64::from(i) / 8.0))
            .collect();
        let max = fluxes.iter().copied().fold(f64::MIN, f64::max);
        let min = fluxes.iter().copied().fold(f64::MAX, f64::min);
        assert!(
            (max - min) / max > 0.02,
            "expected a rotational lightcurve, got spread {}",
            (max - min) / max
        );
    }
}
