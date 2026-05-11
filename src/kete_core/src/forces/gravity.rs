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

use std::str::FromStr;

use crossbeam::sync::ShardedLock;
use nalgebra::{Matrix3, Vector3};

use crate::{
    constants::{C_AU_PER_DAY_INV_SQUARED, EARTH_J2, GMS, JUPITER_J2, SUN_J2},
    desigs::Desig,
    errors::{Error, KeteResult},
    frames::{Ecliptic, InertialFrame},
};

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

/// Gravitational parameters for an object which follows a SPICE kernel.
/// Radius is in AU, mass is in AU^3 / (Day^2 * Solar Mass)
/// Typically mass should be defined by (GMS * size compared to the Sun).
#[derive(Debug, Clone, Copy)]
pub struct GravParams {
    /// Associated NAIF id
    pub naif_id: i32,

    /// Mass of the object in GMS
    pub mass: f64,

    /// Radius of the object in AU.
    pub radius: f32,
}

impl FromStr for GravParams {
    type Err = Error;

    /// Load a [`GravParams`] from a single string.
    fn from_str(row: &str) -> KeteResult<Self> {
        let mut iter = row.split_whitespace();
        let err = || Error::IOError(format!("GravParams row incorrectly formatted. {row}"));
        let naif_id: i32 = iter.next().ok_or_else(err)?.parse()?;
        let mass: f64 = iter.next().ok_or_else(err)?.parse()?;
        // default radius: 100 m expressed in AU.
        let radius: f32 = iter.next().unwrap_or("6.684587122268446e-10").parse()?;
        Ok(Self {
            naif_id,
            mass: mass * GMS,
            radius,
        })
    }
}

/// Gravity parameter Singleton
pub static MASSES_KNOWN: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = Vec::new();
        let text = std::str::from_utf8(include_bytes!("../../data/masses.tsv"))
            .expect("masses.tsv is not valid UTF-8")
            .split('\n');
        for row in text.filter(|x| !x.starts_with('#') & (!x.trim().is_empty())) {
            let code = GravParams::from_str(row)
                .unwrap_or_else(|e| panic!("failed to parse masses.tsv row {row:?}: {e}"));
            singleton.push(code);
        }
        singleton.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        ShardedLock::new(singleton)
    });

/// Gravity parameter Singleton
pub static MASSES_SELECTED: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        // pre-add the planets and the 5 most massive asteroids from the masses_known list
        // 20000001, 20000002, 20000004, 20000010, 20000704
        // Ceres, Vesta, Pallas, Hygiea, and Interamnia
        let known = MASSES_KNOWN.read().unwrap();
        let mut singleton = select_by_naif_id(
            &known,
            &[
                10, 1, 2, 399, 301, 4, 5, 6, 7, 8, 20000001, 20000002, 20000004, 20000010, 20000704,
            ],
        );
        singleton.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        ShardedLock::new(singleton)
    });

/// Planets and Moon, in the order `[Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter,
/// Saturn, Uranus, Neptune]`. Initialized once from [`MASSES_KNOWN`].
///
/// Wrapped in [`ShardedLock`] purely so it returns the same guard type as
/// [`MASSES_SELECTED`] -- the data is read-only after init and the lock is
/// uncontended in practice.
static PLANETS: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        ShardedLock::new(select_by_naif_id(
            &MASSES_KNOWN.read().unwrap(),
            &[10, 1, 2, 399, 301, 4, 5, 6, 7, 8],
        ))
    });

/// Planets only (Earth and Moon merged into Earth-Moon barycenter id 3).
///
/// Wrapped in [`ShardedLock`] for the same reason as [`PLANETS`].
static SIMPLIFIED_PLANETS: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        ShardedLock::new(select_by_naif_id(
            &MASSES_KNOWN.read().unwrap(),
            &[10, 1, 2, 3, 4, 5, 6, 7, 8],
        ))
    });

/// Register a new massive object to be used in the extended list of objects.
///
/// Masses must be provided as a fraction of the Sun's mass, and radius in AU.
///
/// If an object is already registered with the same NAIF ID, it will not be added
/// again.
#[cfg_attr(feature = "pyo3", pyfunction, pyo3(signature=(naif_id, mass, radius=0.0)))]
pub fn register_custom_mass(naif_id: i32, mass: f64, radius: f32) {
    GravParams {
        naif_id,
        mass: mass * GMS,
        radius,
    }
    .register();
}

/// Register a massive object from the known-masses table by its NAIF ID.
///
/// Use [`register_custom_mass`] to add a body that is not in the table.
///
/// # Errors
///
/// Returns an error if the NAIF ID is not present in the built-in mass table.
#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn register_mass(naif_id: i32) -> KeteResult<()> {
    let known_masses = GravParams::known_masses();
    if let Some(params) = known_masses.iter().find(|p| p.naif_id == naif_id) {
        params.register();
        return Ok(());
    }
    Err(Error::ValueError(format!(
        "NAIF ID {naif_id} is not in the built-in mass table; \
         use register_custom_mass to add it manually"
    )))
}

/// List the massive objects in the extended list of objects to be used during orbit
/// propagation.
///
/// This is meant to be human readable, and will return:
/// (the name of the object,
///  the NAIF ID,
///  the mass,
///  the radius)
#[cfg_attr(feature = "pyo3", pyfunction)]
#[must_use]
pub fn registered_masses() -> Vec<(String, i32, f64, f32)> {
    GravParams::selected_masses()
        .iter()
        .map(|p| {
            (
                Desig::Naif(p.naif_id).try_naif_id_to_name().to_string(),
                p.naif_id,
                p.mass / GMS,
                p.radius,
            )
        })
        .collect()
}

/// List the preloaded massive objects known to kete.
///
/// This is meant to be human readable, and will return:
/// (the name of the object,
///  the NAIF ID,
///  the mass,
///  the radius)
#[cfg_attr(feature = "pyo3", pyfunction)]
#[must_use]
pub fn known_masses() -> Vec<(String, i32, f64, f32)> {
    GravParams::known_masses()
        .iter()
        .map(|p| {
            (
                Desig::Naif(p.naif_id).try_naif_id_to_name().to_string(),
                p.naif_id,
                p.mass / GMS,
                p.radius,
            )
        })
        .collect()
}

/// Pick out the [`GravParams`] entries with NAIF ids matching `ids`,
/// preserving the order of `ids` and skipping any not present in `known`.
fn select_by_naif_id(known: &[GravParams], ids: &[i32]) -> Vec<GravParams> {
    ids.iter()
        .filter_map(|id| known.iter().find(|p| p.naif_id == *id).copied())
        .collect()
}

impl GravParams {
    /// Add acceleration to the provided accel vector.
    #[inline(always)]
    pub fn add_acceleration(
        &self,
        accel: &mut Vector3<f64>,
        rel_pos: &Vector3<f64>,
        rel_vel: &Vector3<f64>,
    ) {
        let mass = self.mass;

        // Special cases for different objects
        match self.naif_id {
            // Sun and Jupiter share an identical correction shape (GR plus
            // ecliptic-frame J2); only the J2 coefficient differs.
            5 | 10 => {
                let j2 = if self.naif_id == 10 {
                    SUN_J2
                } else {
                    JUPITER_J2
                };
                apply_gr_correction(accel, rel_pos, rel_vel, mass);
                let rel_pos_eclip = Ecliptic::from_equatorial(*rel_pos);
                *accel +=
                    Ecliptic::to_equatorial(j2_correction(&rel_pos_eclip, self.radius, j2, mass));
            }
            399 => *accel += j2_correction(rel_pos, self.radius, EARTH_J2, mass),
            _ => (),
        }

        // Basic newtonian gravity
        *accel -= &(rel_pos * (mass * rel_pos.norm().powi(-3)));
    }

    /// Add this [`GravParams`] to the singleton.
    ///
    /// # Panics
    /// Panic if a write lock cannot be put on [`MASSES_SELECTED`].
    pub fn register(self) {
        let mut params = MASSES_SELECTED.write().unwrap();
        // Check if the GravParams already exists
        if !params.iter().any(|p| p.naif_id == self.naif_id) {
            params.push(self);
            params.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        }
    }

    /// Get a read-only reference to the singleton.
    ///
    /// # Panics
    /// Panic if a read lock cannot be put on [`MASSES_KNOWN`].
    pub fn known_masses() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        MASSES_KNOWN.read().unwrap()
    }

    /// Currently selected masses for use in orbit propagation.
    ///
    /// # Panics
    /// Panic if a read lock cannot be put on [`MASSES_SELECTED`].
    pub fn selected_masses() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        MASSES_SELECTED.read().unwrap()
    }

    /// List of all known massive planets and the Moon.
    ///
    /// # Panics
    /// Panic if a read lock cannot be put on [`PLANETS`].
    pub fn planets() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        PLANETS.read().unwrap()
    }

    /// List of Massive planets, but merge the moon and earth together.
    ///
    /// # Panics
    /// Panic if a read lock cannot be put on [`SIMPLIFIED_PLANETS`].
    pub fn simplified_planets() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        SIMPLIFIED_PLANETS.read().unwrap()
    }
}

/// Calculate the effects of the J2 term
///
/// Z is the z component of the unit vector.
#[inline(always)]
fn j2_correction(rel_pos: &Vector3<f64>, radius: f32, j2: f64, mass: f64) -> Vector3<f64> {
    let r = rel_pos.norm();
    let z_squared = 5.0 * (rel_pos.z / r).powi(2);

    // this is formatted a little funny in an attempt to reduce numerical noise
    // 3/2 * j2 * mass * radius^2 / distance^5
    let coef = 1.5 * j2 * mass * (f64::from(radius) / r).powi(2) * r.powi(-3);
    Vector3::<f64>::new(
        rel_pos.x * coef * (z_squared - 1.0),
        rel_pos.y * coef * (z_squared - 1.0),
        rel_pos.z * coef * (z_squared - 3.0),
    )
}

/// Analytical Jacobian of the J2 oblateness acceleration `da_J2/dd` in
/// the body's pole-aligned frame.
///
/// - `d`: relative position in the pole-aligned frame (AU)
/// - `radius`: equatorial radius of the body (AU)
/// - `j2`: J2 coefficient
/// - `mass`: GM of the body (AU^3/day^2)
fn j2_jacobian(d: &Vector3<f64>, radius: f64, j2: f64, mass: f64) -> Matrix3<f64> {
    let d = *d;
    let r = d.norm();
    let r2 = r * r;
    let z = d.z;

    let lambda = 1.5 * j2 * mass * (radius / r).powi(2) / (r2 * r);
    let big_z = 5.0 * z * z / r2;

    let a_norm = Vector3::new(
        d.x * (big_z - 1.0),
        d.y * (big_z - 1.0),
        d.z * (big_z - 3.0),
    );
    let f_diag = Matrix3::from_diagonal(&Vector3::new(big_z - 1.0, big_z - 1.0, big_z - 3.0));
    let dz_dd = (10.0 * z / r2) * (Vector3::new(0.0, 0.0, 1.0) - (z / r2) * d);

    lambda * (-5.0 / r2 * a_norm * d.transpose() + f_diag + d * dz_dd.transpose())
}

/// Analytical `da/dr` and `da/dv` for the N-body gravity force model.
///
/// Includes contributions from:
/// - Newtonian point-mass gravity (all bodies)
/// - General relativity correction (Sun, Jupiter: NAIF IDs 10 and 5)
/// - J2 oblateness (Sun and Jupiter in ecliptic frame, Earth in equatorial)
///
/// `cached_states` must be the SSB-relative `(pos, vel)` of each body in
/// `massive_obj`, in the same order. Non-gravitational contributions are not
/// included; they are handled by the relevant [`Force`] implementation and
/// summed at the [`ForceSet`] composition layer.
#[must_use]
pub fn analytical_jacobians(
    pos: &Vector3<f64>,
    vel: &Vector3<f64>,
    cached_states: &[(Vector3<f64>, Vector3<f64>)],
    massive_obj: &[GravParams],
) -> (Matrix3<f64>, Matrix3<f64>) {
    let pos = *pos;
    let vel = *vel;
    let mut da_dr = Matrix3::<f64>::zeros();
    let mut da_dv = Matrix3::<f64>::zeros();
    let ident = Matrix3::<f64>::identity();

    for (grav_params, (body_pos, body_vel)) in massive_obj.iter().zip(cached_states) {
        let d = pos - body_pos;
        let v = vel - body_vel;
        let r = d.norm();
        let r2 = r * r;
        let r3 = r2 * r;
        let r5 = r2 * r3;
        let mass = grav_params.mass;

        da_dr -= (mass / r5) * (r2 * ident - 3.0 * d * d.transpose());

        match grav_params.naif_id {
            5 | 10 => {
                let cinv2 = C_AU_PER_DAY_INV_SQUARED;
                let kappa = mass * cinv2 / r3;
                let v2 = v.norm_squared();
                let big_c = 4.0 * mass / r - v2;
                let big_r = 4.0 * d.dot(&v);
                let a_gr = big_c * d + big_r * v;

                da_dr += (-3.0 * kappa / r2) * a_gr * d.transpose()
                    + kappa
                        * ((-4.0 * mass / r3) * d * d.transpose()
                            + big_c * ident
                            + 4.0 * v * v.transpose());
                da_dv +=
                    kappa * (-2.0 * d * v.transpose() + 4.0 * v * d.transpose() + big_r * ident);

                let j2_val = if grav_params.naif_id == 10 {
                    SUN_J2
                } else {
                    JUPITER_J2
                };
                let d_ec = Ecliptic::from_equatorial(d);
                let j2_jac = j2_jacobian(&d_ec, f64::from(grav_params.radius), j2_val, mass);
                let rot = *Ecliptic::rotation_to_equatorial().matrix();
                da_dr += rot * j2_jac * rot.transpose();
            }
            399 => {
                da_dr += j2_jacobian(&d, f64::from(grav_params.radius), EARTH_J2, mass);
            }
            _ => {}
        }
    }

    (da_dr, da_dv)
}

/// Add the effects of general relativistic motion to an acceleration vector
#[inline(always)]
fn apply_gr_correction(
    accel: &mut Vector3<f64>,
    rel_pos: &Vector3<f64>,
    rel_vel: &Vector3<f64>,
    mass: f64,
) {
    let r_v = 4.0 * rel_pos.dot(rel_vel);

    let rel_v2: f64 = rel_vel.norm_squared();
    let r = rel_pos.norm();

    let gr_const: f64 = mass * C_AU_PER_DAY_INV_SQUARED * r.powi(-3);
    let c: f64 = 4. * mass / r - rel_v2;
    *accel += gr_const * (c * rel_pos + r_v * rel_vel);
}
