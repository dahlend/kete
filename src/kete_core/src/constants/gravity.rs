// BSD 3-Clause License
//
// Copyright (c) 2025, Dar Dahlen
// Copyright (c) 2025, California Institute of Technology
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
use nalgebra::Vector3;

use crate::{
    desigs::Desig,
    errors::{Error, KeteResult},
    frames::{Ecliptic, InertialFrame},
};

use super::C_AU_PER_DAY_INV_SQUARED;

/// Standard Gravitational Constants of the Sun
/// AU^3 / (Day^2 * Solar Mass)
pub const GMS: f64 = 0.00029591220828411956;

/// Gaussian gravitational constant, equivalent to sqrt of GMS.
/// AU^(3/2) per (Day sqrt(Solar Mass))
pub const GMS_SQRT: f64 = 0.01720209894996;

/// Sun J2 Parameter
///
/// This paper below a source, however there are several papers which all put
/// the Sun's J2 at 2.2e-7.
///
/// "Prospects of Dynamical Determination of General Relativity Parameter β and Solar
/// Quadrupole Moment J2 with Asteroid Radar Astronomy"
/// The Astrophysical Journal, 845:166 (5pp), 2017 August 20
pub const SUN_J2: f64 = 2.2e-7;

/// Earth J2 Parameter
/// See "Revisiting Spacetrack Report #3" - Final page of appendix.
pub const EARTH_J2: f64 = 0.00108262998905;

/// Earth J3 Parameter
pub const EARTH_J3: f64 = -0.00000253215306;

/// Earth J4 Parameter
pub const EARTH_J4: f64 = -0.00000161098761;

/// Jupiter J2 Parameter
///
/// "Measurement of Jupiter’s asymmetric gravity field"
/// <https://www.nature.com/articles/nature25776>
/// Nature 555, 220-220, 2018 March 8
pub const JUPITER_J2: f64 = 0.014696572;

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
        let naif_id = iter.next();
        let mass = iter.next();
        let radius = iter.next().unwrap_or("0.0");
        if naif_id.is_none() || mass.is_none() {
            return Err(Error::IOError(format!(
                "GravParams row incorrectly formatted. {row}",
            )));
        }
        let mass: f64 = mass.unwrap().parse()?;

        Ok(Self {
            naif_id: naif_id.unwrap().parse()?,
            mass: mass * GMS,
            radius: radius.parse()?,
        })
    }
}

/// Gravity parameter Singleton
static MASSES_KNOWN: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = Vec::new();
        let text = std::str::from_utf8(include_bytes!("../../data/masses.tsv"))
            .unwrap()
            .split('\n');
        for row in text.filter(|x| !x.starts_with('#') & (!x.trim().is_empty())) {
            let code = GravParams::from_str(row).unwrap();
            singleton.push(code);
        }
        singleton.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        ShardedLock::new(singleton)
    });

/// Gravity parameter Singleton
static MASSES_SELECTED: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        let mut singleton = Vec::new();
        // pre-add the planets and the 5 most massive asteroids from the masses_known list
        // 20000001, 20000002, 20000004, 20000010, 20000704
        // Ceres, Vesta, Pallas, Hygiea, and Interamnia
        let known_masses = MASSES_KNOWN.read().unwrap();
        for id in [
            10, 1, 2, 399, 301, 4, 5, 6, 7, 8, 20000001, 20000002, 20000004, 20000010, 20000704,
        ] {
            if let Some(param) = known_masses.iter().find(|p| p.naif_id == id) {
                singleton.push(*param);
            }
        }
        singleton.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        ShardedLock::new(singleton)
    });

/// Register a new massive object to be used in the extended list of objects.
///
/// Masses must be provided as a fraction of the Sun's mass, and radius in AU.
///
/// If an object is already registered with the same NAIF ID, it will not be added again.
#[cfg_attr(feature = "pyo3", pyfunction, pyo3(signature=(naif_id, mass, radius=0.0)))]
pub fn register_custom_mass(naif_id: i32, mass: f64, radius: f32) {
    let params = GravParams::new(naif_id, mass * GMS, radius);
    params.register();
}

/// Register a new massive object to be used in the extended list of objects.
///
/// This looks up the mass and radius of the object by its NAIF ID of the known
/// masses, panics if the object is not found.
#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn register_mass(naif_id: i32) {
    let known_masses = GravParams::known_masses();
    if let Some(params) = known_masses.iter().find(|p| p.naif_id == naif_id) {
        params.register();
        return;
    }
    panic!("Failed to find mass for NAIF ID {naif_id}");
}

/// List the massive objects in the extended list of objects to be used during orbit propagation.
///
/// This is meant to be human readable, and will return:
/// (the name of the object,
///  the NAIF ID,
///  the mass,
///  the radius)
#[cfg_attr(feature = "pyo3", pyfunction)]
pub fn registered_masses() -> Vec<(String, i32, f64, f32)> {
    let params = GravParams::selected_masses();
    params
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
pub fn known_masses() -> Vec<(String, i32, f64, f32)> {
    let params = GravParams::known_masses();
    params
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

impl GravParams {
    /// Create a new [`GravParams`] object.
    pub fn new(naif_id: i32, mass: f64, radius: f32) -> Self {
        Self {
            naif_id,
            mass,
            radius,
        }
    }

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
            5 => {
                let radius = self.radius;

                // GR correction
                apply_gr_correction(accel, rel_pos, rel_vel, &mass);

                // J2 correction
                let rel_pos_eclip = Ecliptic::from_equatorial(*rel_pos);
                *accel += Ecliptic::to_equatorial(j2_correction(
                    &rel_pos_eclip,
                    radius,
                    &JUPITER_J2,
                    &mass,
                ));
            }
            10 => {
                let radius = self.radius;

                // GR correction
                apply_gr_correction(accel, rel_pos, rel_vel, &mass);

                // J2 correction
                let rel_pos_eclip = Ecliptic::from_equatorial(*rel_pos);
                *accel +=
                    Ecliptic::to_equatorial(j2_correction(&rel_pos_eclip, radius, &SUN_J2, &mass));
            }
            399 => *accel += j2_correction(rel_pos, self.radius, &EARTH_J2, &mass),
            _ => (),
        }

        // Basic newtonian gravity
        *accel -= &(rel_pos * (mass * rel_pos.norm().powi(-3)));
    }

    /// Add this [`GravParams`] to the singleton.
    pub fn register(self) {
        let mut params = MASSES_SELECTED.write().unwrap();
        // Check if the GravParams already exists
        if !params.iter().any(|p| p.naif_id == self.naif_id) {
            params.push(self);
            params.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        }
    }

    /// Get a read-only reference to the singleton.
    pub fn known_masses() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        MASSES_KNOWN.read().unwrap()
    }

    /// Currently selected masses for use in orbit propagation.
    pub fn selected_masses() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        MASSES_SELECTED.read().unwrap()
    }

    /// List of all known massive planets and the Moon.
    pub fn planets() -> Vec<Self> {
        let known = Self::known_masses();
        let mut planets = Vec::new();
        for id in [10, 1, 2, 399, 301, 4, 5, 6, 7, 8] {
            if let Some(param) = known.iter().find(|p| p.naif_id == id) {
                planets.push(*param);
            }
        }
        planets
    }

    /// List of Massive planets, but merge the moon and earth together.
    pub fn simplified_planets() -> Vec<Self> {
        let known = Self::known_masses();
        let mut planets = Vec::new();
        for id in [10, 1, 2, 3, 4, 5, 6, 7, 8] {
            if let Some(param) = known.iter().find(|p| p.naif_id == id) {
                planets.push(*param);
            }
        }
        planets
    }
}

/// Calculate the effects of the J2 term
///
/// Z is the z component of the unit vector.
#[inline(always)]
fn j2_correction(rel_pos: &Vector3<f64>, radius: f32, j2: &f64, mass: &f64) -> Vector3<f64> {
    let r = rel_pos.norm();
    let z_squared = 5.0 * (rel_pos.z / r).powi(2);

    // this is formatted a little funny in an attempt to reduce numerical noise
    // 3/2 * j2 * mass * earth_r^2 / distance^5
    let coef = 1.5 * j2 * mass * (radius as f64 / r).powi(2) * r.powi(-3);
    Vector3::<f64>::new(
        rel_pos.x * coef * (z_squared - 1.0),
        rel_pos.y * coef * (z_squared - 1.0),
        rel_pos.z * coef * (z_squared - 3.0),
    )
}

/// Add the effects of general relativistic motion to an acceleration vector
#[inline(always)]
fn apply_gr_correction(
    accel: &mut Vector3<f64>,
    rel_pos: &Vector3<f64>,
    rel_vel: &Vector3<f64>,
    mass: &f64,
) {
    let r_v = 4.0 * rel_pos.dot(rel_vel);

    let rel_v2: f64 = rel_vel.norm_squared();
    let r = rel_pos.norm();

    let gr_const: f64 = mass * C_AU_PER_DAY_INV_SQUARED * r.powi(-3);
    let c: f64 = 4. * mass / r - rel_v2;
    *accel += gr_const * (c * rel_pos + r_v * rel_vel);
}
