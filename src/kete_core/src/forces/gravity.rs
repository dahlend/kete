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
use std::sync::Arc;

use crossbeam::sync::ShardedLock;
use nalgebra::{Matrix3, Vector3};

use super::{Polyhedron, SphericalHarmonics};
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
#[derive(Debug, Clone)]
pub struct GravParams {
    /// Associated NAIF id
    pub naif_id: i32,

    /// Gravitational parameter `GM` of the object, in AU^3 / Day^2.
    ///
    /// Parsed from the mass table as a fraction of the Sun's mass and scaled by
    /// [`GMS`](crate::constants::GMS), so this is an absolute `GM` rather than a ratio.
    pub mass: f64,

    /// Radius of the object in AU.
    pub radius: f32,

    /// Whether the first-order relativistic correction is applied for this body.
    pub relativistic: bool,

    /// Gravitational field of the body beyond the point-mass term.
    pub shape: Shape,
}

/// Gravitational field of a massive body beyond the point-mass term.
///
/// [`Shape::Point`] and [`Shape::Oblate`] add to the point-mass term.
/// [`Shape::Polyhedron`] and [`Shape::SphericalHarmonics`] replace it inside their
/// switch radius, since those fields contain the point mass, and fall back to it
/// outside.
#[derive(Debug, Clone)]
pub enum Shape {
    /// No additional term.
    Point,

    /// The J2 zonal term of an oblate body.
    Oblate {
        /// J2 coefficient, normalized by the body's `radius`.
        j2: f64,
        /// Unit spin axis of the body on equatorial axes.
        pole: Vector3<f64>,
    },

    /// A constant-density polyhedron.
    ///
    /// Within `switch_radius` of the body the acceleration is the polyhedron field;
    /// beyond it, the point mass. The polyhedron is in the body frame, in AU, with
    /// its origin at the body's position and its `GM` in AU^3 / Day^2 (the
    /// `GravParams` mass is used for the point-mass term, and the two are expected
    /// to agree). The point mass is at the body's position, so a mesh whose volume
    /// centroid is off its origin also changes the field across the switch by that
    /// offset.
    Polyhedron {
        /// The field model, in the body frame.
        model: Arc<Polyhedron>,
        /// Orientation of the body frame.
        orientation: Orientation,
        /// Distance from the body, in AU, inside which the polyhedron field is used.
        switch_radius: f64,
    },

    /// A spherical harmonic field.
    ///
    /// Within `switch_radius` of the body the acceleration is the series; beyond it,
    /// the point mass. The series is in the body frame, in AU, expanded about the
    /// body's position, with its `GM` in AU^3 / Day^2 equal to the `GravParams`
    /// mass. Inside the model's minimum radius, where the series is not valid, the
    /// acceleration is the point mass, and a position there counts as inside the body;
    /// see [`GravParams::is_inside`].
    SphericalHarmonics {
        /// The field model, in the body frame.
        model: Arc<SphericalHarmonics>,
        /// Orientation of the body frame.
        orientation: Orientation,
        /// Distance from the body, in AU, inside which the series is used.
        switch_radius: f64,
    },
}

/// The field model of a shaped body selected for one evaluation.
#[derive(Debug, Clone, Copy)]
enum ShapeField<'a> {
    Polyhedron(&'a Polyhedron),
    Harmonics(&'a SphericalHarmonics),
}

impl ShapeField<'_> {
    /// Acceleration at a body-frame position.
    fn accel(self, pos: &Vector3<f64>) -> KeteResult<Vector3<f64>> {
        match self {
            Self::Polyhedron(model) => Ok(model.field(pos).0),
            Self::Harmonics(model) => model.field(pos),
        }
    }

    /// Acceleration and its gradient at a body-frame position.
    fn field_and_gradient(self, pos: &Vector3<f64>) -> KeteResult<(Vector3<f64>, Matrix3<f64>)> {
        match self {
            Self::Polyhedron(model) => {
                let (accel, grad, _) = model.field_and_gradient(pos);
                Ok((accel, grad))
            }
            Self::Harmonics(model) => model.field_and_gradient(pos),
        }
    }
}

/// Orientation of a body frame relative to equatorial axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    /// A fixed rotation from the body frame to equatorial axes.
    Fixed(Matrix3<f64>),

    /// The rotation from the body frame to equatorial axes is the frame `frame_id` of
    /// an [`Ephemeris`](crate::ephemeris::Ephemeris) at each evaluation; the caller
    /// resolves it (see [`GravParams::orientation_needed`]).
    Frame {
        /// Id of the body frame.
        frame_id: i32,
    },
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
        Ok(Self::new(naif_id, mass * GMS, radius))
    }
}

/// Every body in the built-in mass table, sorted by mass.
static MASSES_KNOWN: std::sync::LazyLock<Vec<GravParams>> = std::sync::LazyLock::new(|| {
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
    singleton
});

/// Bodies selected for the extended N-body model; changed by registration.
static MASSES_SELECTED: std::sync::LazyLock<ShardedLock<Vec<GravParams>>> =
    std::sync::LazyLock::new(|| {
        // pre-add the planets and the 5 most massive asteroids from the masses_known list
        // 20000001, 20000002, 20000004, 20000010, 20000704
        // Ceres, Vesta, Pallas, Hygiea, and Interamnia
        let mut singleton = select_by_naif_id(
            &MASSES_KNOWN,
            &[
                10, 1, 2, 399, 301, 4, 5, 6, 7, 8, 20000001, 20000002, 20000004, 20000010, 20000704,
            ],
        );
        singleton.sort_by(|a, b| a.mass.total_cmp(&b.mass));
        ShardedLock::new(singleton)
    });

/// Planets and Moon, in the order `[Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter,
/// Saturn, Uranus, Neptune]`. Initialized once from [`MASSES_KNOWN`].
static PLANETS: std::sync::LazyLock<Vec<GravParams>> = std::sync::LazyLock::new(|| {
    select_by_naif_id(&MASSES_KNOWN, &[10, 1, 2, 399, 301, 4, 5, 6, 7, 8])
});

/// Planets only (Earth and Moon merged into Earth-Moon barycenter id 3).
static SIMPLIFIED_PLANETS: std::sync::LazyLock<Vec<GravParams>> =
    std::sync::LazyLock::new(|| select_by_naif_id(&MASSES_KNOWN, &[10, 1, 2, 3, 4, 5, 6, 7, 8]));

/// Register a new massive object to be used in the extended list of objects.
///
/// Masses must be provided as a fraction of the Sun's mass, and radius in AU.
///
/// An object already registered with the same NAIF ID is replaced.
#[cfg_attr(feature = "pyo3", pyfunction, pyo3(signature=(naif_id, mass, radius=0.0)))]
pub fn register_custom_mass(naif_id: i32, mass: f64, radius: f32) {
    GravParams::new(naif_id, mass * GMS, radius).register();
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
        params.clone().register();
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
    describe_masses(&GravParams::selected_masses())
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
    describe_masses(GravParams::known_masses())
}

/// `(name, NAIF id, mass as a fraction of the Sun's, radius)` for each body.
fn describe_masses(params: &[GravParams]) -> Vec<(String, i32, f64, f32)> {
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

/// Pick out the [`GravParams`] entries with NAIF ids matching `ids`,
/// preserving the order of `ids` and skipping any not present in `known`.
fn select_by_naif_id(known: &[GravParams], ids: &[i32]) -> Vec<GravParams> {
    ids.iter()
        .filter_map(|id| known.iter().find(|p| p.naif_id == *id).cloned())
        .collect()
}

impl GravParams {
    /// Build the parameters of a body, with `mass` as `GM` in AU^3 / Day^2 and `radius`
    /// in AU.
    ///
    /// The relativistic correction and the [`Shape`] are assigned from the NAIF id. The
    /// Sun and Jupiter are relativistic and oblate about the ecliptic pole (both poles
    /// are approximated by it, a fraction-of-a-degree-scale approximation on an already
    /// small term), the Earth is oblate about the equatorial pole, and every other body
    /// is a point mass.
    #[must_use]
    pub fn new(naif_id: i32, mass: f64, radius: f32) -> Self {
        let (relativistic, shape) = match naif_id {
            10 => (
                true,
                Shape::Oblate {
                    j2: SUN_J2,
                    pole: *ECLIPTIC_POLE_EQUATORIAL,
                },
            ),
            5 => (
                true,
                Shape::Oblate {
                    j2: JUPITER_J2,
                    pole: *ECLIPTIC_POLE_EQUATORIAL,
                },
            ),
            399 => (
                false,
                Shape::Oblate {
                    j2: EARTH_J2,
                    pole: Vector3::z(),
                },
            ),
            _ => (false, Shape::Point),
        };
        Self {
            naif_id,
            mass,
            radius,
            relativistic,
            shape,
        }
    }

    /// The frame id whose rotation [`Self::add_acceleration`] and
    /// [`Self::add_acceleration_and_jacobians`] need at relative position `rel_pos`, if any: a
    /// shaped body with a frame orientation, evaluated inside its switch radius.
    #[inline(always)]
    #[must_use]
    pub fn orientation_needed(&self, rel_pos: &Vector3<f64>) -> Option<i32> {
        match &self.shape {
            Shape::Polyhedron {
                orientation: Orientation::Frame { frame_id },
                switch_radius,
                ..
            }
            | Shape::SphericalHarmonics {
                orientation: Orientation::Frame { frame_id },
                switch_radius,
                ..
            } if rel_pos.norm() < *switch_radius => Some(*frame_id),
            Shape::Point
            | Shape::Oblate { .. }
            | Shape::Polyhedron { .. }
            | Shape::SphericalHarmonics { .. } => None,
        }
    }

    /// The shape field to use at `rel_pos`, with the body-to-equatorial rotation, or
    /// `None` when the point-mass term applies.
    #[inline(always)]
    fn shape_field_at(
        &self,
        rel_pos: &Vector3<f64>,
        body_to_equatorial: Option<&Matrix3<f64>>,
    ) -> KeteResult<Option<(ShapeField<'_>, Matrix3<f64>)>> {
        let (model, orientation, switch_radius) = match &self.shape {
            Shape::Polyhedron {
                model,
                orientation,
                switch_radius,
            } => (ShapeField::Polyhedron(model), orientation, switch_radius),
            Shape::SphericalHarmonics {
                model,
                orientation,
                switch_radius,
            } => (ShapeField::Harmonics(model), orientation, switch_radius),
            Shape::Point | Shape::Oblate { .. } => return Ok(None),
        };
        if rel_pos.norm() >= *switch_radius {
            return Ok(None);
        }
        // Inside the minimum radius the series is not valid. The integrators evaluate
        // trial points there during a step, so this is the point mass rather than an
        // error, and an accepted state there is an impact.
        if let ShapeField::Harmonics(harmonics) = model
            && harmonics
                .min_radius()
                .is_some_and(|min| rel_pos.norm() < min)
        {
            return Ok(None);
        }
        let rot = match orientation {
            Orientation::Fixed(rot) => *rot,
            Orientation::Frame { frame_id } => *body_to_equatorial.ok_or_else(|| {
                Error::ValueError(format!(
                    "Body {} has a gravity field model oriented by frame {frame_id}, \
                     but no orientation was provided; this propagator cannot evaluate it.",
                    self.naif_id
                ))
            })?,
        };
        Ok(Some((model, rot)))
    }

    /// Whether `rel_pos`, relative to the body on equatorial axes, is inside the body.
    ///
    /// For [`Shape::Polyhedron`] this is the polyhedron's solid-angle inside test on
    /// its surface. For [`Shape::SphericalHarmonics`] it is `|rel_pos|` at most the
    /// larger of `radius` and the model's minimum radius, since the series is not
    /// valid inside the minimum radius. For every other shape it is
    /// `|rel_pos| <= radius`.
    /// `body_to_equatorial` is as for [`Self::add_acceleration`].
    ///
    /// # Errors
    /// Fails if a required orientation is not provided.
    pub fn is_inside(
        &self,
        rel_pos: &Vector3<f64>,
        body_to_equatorial: Option<&Matrix3<f64>>,
    ) -> KeteResult<bool> {
        if let Shape::Polyhedron { model, .. } = &self.shape {
            // The mesh lies within its bounding radius, which is inside the switch
            // radius, so the polyhedron field applies wherever the point can be inside.
            if rel_pos.norm() > model.mesh().bounding_radius() {
                return Ok(false);
            }
            if let Some((ShapeField::Polyhedron(model), rot)) =
                self.shape_field_at(rel_pos, body_to_equatorial)?
            {
                // The solid angle is 4 pi inside the surface and 0 outside it.
                let (_, solid_angle) = model.field(&(rot.transpose() * rel_pos));
                return Ok(solid_angle > 2.0 * std::f64::consts::PI);
            }
        }
        let radius = match &self.shape {
            Shape::SphericalHarmonics { model, .. } => {
                f64::from(self.radius).max(model.min_radius().unwrap_or(0.0))
            }
            Shape::Point | Shape::Oblate { .. } | Shape::Polyhedron { .. } => {
                f64::from(self.radius)
            }
        };
        Ok(rel_pos.norm() <= radius)
    }

    /// Add acceleration to the provided accel vector.
    ///
    /// `rel_pos` and `rel_vel` are relative to the body, on equatorial axes.
    /// `body_to_equatorial` is the body-frame rotation for a shaped body with a
    /// frame orientation, required when [`Self::orientation_needed`] returns a frame,
    /// ignored otherwise.
    ///
    /// # Errors
    /// Fails if a required orientation is not provided.
    #[inline(always)]
    pub fn add_acceleration(
        &self,
        accel: &mut Vector3<f64>,
        rel_pos: &Vector3<f64>,
        rel_vel: &Vector3<f64>,
        body_to_equatorial: Option<&Matrix3<f64>>,
    ) -> KeteResult<()> {
        let mass = self.mass;

        if self.relativistic {
            apply_gr_correction(accel, rel_pos, rel_vel, mass);
        }
        match &self.shape {
            Shape::Point => (),
            Shape::Oblate { j2, pole } => {
                *accel += j2_correction(rel_pos, pole, f64::from(self.radius), *j2, mass);
            }
            Shape::Polyhedron { .. } | Shape::SphericalHarmonics { .. } => {
                if let Some((model, rot)) = self.shape_field_at(rel_pos, body_to_equatorial)? {
                    // the field contains the point mass
                    *accel += rot * model.accel(&(rot.transpose() * rel_pos))?;
                    return Ok(());
                }
            }
        }

        // Basic newtonian gravity
        *accel -= &(rel_pos * (mass * rel_pos.norm().powi(-3)));
        Ok(())
    }

    /// Add this body's acceleration to `accel`, and its derivatives, `da/dr` and
    /// `da/dv`, to the provided matrices, evaluating a shape field once.
    ///
    /// The acceleration is that of [`Self::add_acceleration`], and the derivatives
    /// its analytical derivative, term for term:
    /// - Newtonian point-mass gravity
    /// - General relativity correction, for bodies flagged `relativistic`
    /// - J2 oblateness, for bodies with [`Shape::Oblate`]
    /// - the polyhedron or spherical harmonic field in place of the point mass, for
    ///   [`Shape::Polyhedron`] and [`Shape::SphericalHarmonics`] inside their switch
    ///   radius
    ///
    /// `body_to_equatorial` is as for [`Self::add_acceleration`].
    ///
    /// # Errors
    /// As [`Self::add_acceleration`].
    #[inline(always)]
    pub fn add_acceleration_and_jacobians(
        &self,
        accel: &mut Vector3<f64>,
        da_dr: &mut Matrix3<f64>,
        da_dv: &mut Matrix3<f64>,
        rel_pos: &Vector3<f64>,
        rel_vel: &Vector3<f64>,
        body_to_equatorial: Option<&Matrix3<f64>>,
    ) -> KeteResult<()> {
        let d = *rel_pos;
        let v = *rel_vel;
        let ident = Matrix3::<f64>::identity();
        let r = d.norm();
        let r2 = r * r;
        let r3 = r2 * r;
        let r5 = r2 * r3;
        let mass = self.mass;

        // the acceleration in the order of `add_acceleration`, so it matches exactly
        if self.relativistic {
            apply_gr_correction(accel, rel_pos, rel_vel, mass);
        }
        if let Shape::Oblate { j2, pole } = &self.shape {
            *accel += j2_correction(rel_pos, pole, f64::from(self.radius), *j2, mass);
        }
        if let Some((model, rot)) = self.shape_field_at(&d, body_to_equatorial)? {
            // the field contains the point mass
            let (field, grad) = model.field_and_gradient(&(rot.transpose() * d))?;
            *accel += rot * field;
            *da_dr += rot * grad * rot.transpose();
        } else {
            *accel -= &(rel_pos * (mass * rel_pos.norm().powi(-3)));
            *da_dr -= (mass / r5) * (r2 * ident - 3.0 * d * d.transpose());
        }

        if self.relativistic {
            let cinv2 = C_AU_PER_DAY_INV_SQUARED;
            let kappa = mass * cinv2 / r3;
            let v2 = v.norm_squared();
            let big_c = 4.0 * mass / r - v2;
            let big_r = 4.0 * d.dot(&v);
            let a_gr = big_c * d + big_r * v;

            *da_dr += (-3.0 * kappa / r2) * a_gr * d.transpose()
                + kappa
                    * ((-4.0 * mass / r3) * d * d.transpose()
                        + big_c * ident
                        + 4.0 * v * v.transpose());
            *da_dv += kappa * (-2.0 * d * v.transpose() + 4.0 * v * d.transpose() + big_r * ident);
        }

        match &self.shape {
            Shape::Point | Shape::Polyhedron { .. } | Shape::SphericalHarmonics { .. } => (),
            Shape::Oblate { j2, pole } => {
                *da_dr += j2_jacobian(&d, pole, f64::from(self.radius), *j2, mass);
            }
        }
        Ok(())
    }

    /// Add this [`GravParams`] to the singleton, replacing any entry with the same
    /// NAIF id.
    ///
    /// # Panics
    /// Panic if a write lock cannot be put on [`MASSES_SELECTED`].
    pub fn register(self) {
        let mut params = MASSES_SELECTED.write().unwrap();
        params.retain(|p| p.naif_id != self.naif_id);
        params.push(self);
        params.sort_by(|a, b| a.mass.total_cmp(&b.mass));
    }

    /// Every body in the built-in mass table, sorted by mass.
    #[must_use]
    pub fn known_masses() -> &'static [Self] {
        &MASSES_KNOWN
    }

    /// Currently selected masses for use in orbit propagation.
    ///
    /// # Panics
    /// Panic if a read lock cannot be put on [`MASSES_SELECTED`].
    pub fn selected_masses() -> crossbeam::sync::ShardedLockReadGuard<'static, Vec<Self>> {
        MASSES_SELECTED.read().unwrap()
    }

    /// The planets and the Moon.
    #[must_use]
    pub fn planets() -> &'static [Self] {
        &PLANETS
    }

    /// The planets, with the Earth and Moon merged into the Earth-Moon barycenter.
    #[must_use]
    pub fn simplified_planets() -> &'static [Self] {
        &SIMPLIFIED_PLANETS
    }

    /// Gravitational parameter of the body with this NAIF id, in AU^3 / Day^2.
    ///
    /// An unknown id is an error rather than a fallback to the Sun. The two places this
    /// matters are both silent when guessed: a two-body reference is only meaningful
    /// about a gravitating body, and the solar system barycenter, NAIF 0, is not in the
    /// mass table at all, so barycentric quantities would have been built with the Sun's
    /// `mu` about a focus with no body at it.
    ///
    /// # Errors
    /// Fails if `naif_id` has no entry in [`Self::known_masses`].
    pub fn try_mass_from_naif_id(naif_id: i32) -> KeteResult<f64> {
        Self::known_masses()
            .iter()
            .find(|p| p.naif_id == naif_id)
            .map(|p| p.mass)
            .ok_or_else(|| {
                Error::ValueError(format!(
                    "NAIF id {naif_id} has no known mass, so it has no gravitational \
                     parameter. Note that the solar system barycenter has no body at it."
                ))
            })
    }
}

/// The ecliptic pole expressed on equatorial axes.
static ECLIPTIC_POLE_EQUATORIAL: std::sync::LazyLock<Vector3<f64>> =
    std::sync::LazyLock::new(|| Ecliptic::to_equatorial(Vector3::z()));

/// Acceleration from the J2 oblateness term of a body.
///
/// `rel_pos` is the position relative to the body and `pole` the body's unit
/// spin axis, both expressed on the same axes; the result is returned on
/// those axes. `radius` is the body's equatorial radius in AU and `mass` its
/// GM in AU^3/Day^2.
#[inline(always)]
pub(crate) fn j2_correction(
    rel_pos: &Vector3<f64>,
    pole: &Vector3<f64>,
    radius: f64,
    j2: f64,
    mass: f64,
) -> Vector3<f64> {
    let r = rel_pos.norm();
    let z = rel_pos.dot(pole);
    let z_squared = 5.0 * (z / r).powi(2);

    // this is formatted a little funny in an attempt to reduce numerical noise
    // 3/2 * j2 * mass * radius^2 / distance^5
    let coef = 1.5 * j2 * mass * (radius / r).powi(2) * r.powi(-3);
    coef * ((z_squared - 1.0) * rel_pos - (2.0 * z) * pole)
}

/// Analytical Jacobian of the J2 oblateness acceleration `da_J2/dd`.
///
/// The derivative of [`j2_correction`], with the same arguments: `d` is the position
/// relative to the body and `pole` its unit spin axis, on the same axes.
///
/// - `radius`: equatorial radius of the body (AU)
/// - `j2`: J2 coefficient
/// - `mass`: GM of the body (AU^3/day^2)
fn j2_jacobian(
    d: &Vector3<f64>,
    pole: &Vector3<f64>,
    radius: f64,
    j2: f64,
    mass: f64,
) -> Matrix3<f64> {
    let d = *d;
    let r = d.norm();
    let r2 = r * r;
    let z = d.dot(pole);

    let lambda = 1.5 * j2 * mass * (radius / r).powi(2) / (r2 * r);
    let big_z = 5.0 * z * z / r2;

    // accel = lambda * a_norm, with lambda scaling as r^-5.
    let a_norm = (big_z - 1.0) * d - (2.0 * z) * pole;
    let da_norm = (big_z - 1.0) * Matrix3::identity() - 2.0 * pole * pole.transpose();
    let dz_dd = (10.0 * z / r2) * (pole - (z / r2) * d);

    lambda * (-5.0 / r2 * a_norm * d.transpose() + da_norm + d * dz_dd.transpose())
}

/// Add the effects of general relativistic motion to an acceleration vector.
///
/// This is the first-order Schwarzschild (single-body 1PN) acceleration
/// `(GM/c^2 r^3) [(4GM/r - v^2) r + 4 (r.v) v]`, with `rel_pos`/`rel_vel`
/// relative to the central body. It reproduces both the secular apsidal
/// precession and the relativistic mean motion.
#[inline(always)]
pub(crate) fn apply_gr_correction(
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The ecliptic-pole rotation form of the Sun and Jupiter J2 jacobian: rotate into the
    /// frame whose z axis is the pole, differentiate there, rotate back.
    fn j2_jacobian_by_rotation(d: &Vector3<f64>, radius: f64, j2: f64, mass: f64) -> Matrix3<f64> {
        let d_ec = Ecliptic::from_equatorial(*d);
        let rot = *Ecliptic::rotation_to_equatorial().matrix();
        rot * j2_jacobian(&d_ec, &Vector3::z(), radius, j2, mass) * rot.transpose()
    }

    /// `j2_jacobian` about an arbitrary pole agrees with the same derivative taken in the
    /// pole-aligned frame and rotated back.
    #[test]
    fn j2_jacobian_general_pole_matches_rotated_frame() {
        let d = Vector3::new(0.31, -0.84, 0.12);
        let radius = 4.65e-3;
        let direct = j2_jacobian(&d, &ECLIPTIC_POLE_EQUATORIAL, radius, SUN_J2, GMS);
        let rotated = j2_jacobian_by_rotation(&d, radius, SUN_J2, GMS);
        let rel = (direct - rotated).abs().max() / rotated.abs().max();
        println!("j2_jacobian general pole vs rotated frame: relative difference {rel:e}");
        assert!(rel < 1e-12, "relative difference {rel:e}");
    }

    /// `add_acceleration_and_jacobians` gives the acceleration of `add_acceleration` and
    /// its derivative for every kind of body: relativistic and oblate (Sun, Jupiter),
    /// oblate only (Earth), and point mass (Mars).
    #[test]
    fn add_acceleration_and_jacobians_matches_finite_difference() {
        let rel_pos = Vector3::new(0.31, -0.84, 0.12);
        let rel_vel = Vector3::new(0.011, 0.004, -0.002);
        let accel_at = |body: &GravParams, p: &Vector3<f64>, v: &Vector3<f64>| {
            let mut accel = Vector3::zeros();
            body.add_acceleration(&mut accel, p, v, None).unwrap();
            accel
        };
        for naif_id in [10, 5, 399, 4] {
            let body = GravParams::known_masses()
                .iter()
                .find(|p| p.naif_id == naif_id)
                .cloned()
                .unwrap();
            let mut accel = Vector3::zeros();
            let mut da_dr = Matrix3::zeros();
            let mut da_dv = Matrix3::zeros();
            body.add_acceleration_and_jacobians(
                &mut accel, &mut da_dr, &mut da_dv, &rel_pos, &rel_vel, None,
            )
            .unwrap();
            assert_eq!(accel, accel_at(&body, &rel_pos, &rel_vel));

            let mut fd_dr = Matrix3::zeros();
            let mut fd_dv = Matrix3::zeros();
            for axis in 0..3 {
                let mut step = Vector3::zeros();
                step[axis] = 1e-6;
                let col = (accel_at(&body, &(rel_pos + step), &rel_vel)
                    - accel_at(&body, &(rel_pos - step), &rel_vel))
                    / 2e-6;
                fd_dr.set_column(axis, &col);
                step[axis] = 1e-8;
                let col = (accel_at(&body, &rel_pos, &(rel_vel + step))
                    - accel_at(&body, &rel_pos, &(rel_vel - step)))
                    / 2e-8;
                fd_dv.set_column(axis, &col);
            }
            let err_dr = (da_dr - fd_dr).abs().max() / da_dr.abs().max();
            let err_dv = (da_dv - fd_dv).abs().max();
            println!("body {naif_id}: da/dr relative error {err_dr:e}, da/dv error {err_dv:e}");
            assert!(
                err_dr < 1e-7,
                "body {naif_id}: da/dr relative error {err_dr:e}"
            );
            assert!(
                err_dv < 1e-7 * da_dr.abs().max(),
                "body {naif_id}: da/dv error {err_dv:e}"
            );
        }
    }
    /// A polyhedron body: the test prism of `forces::polyhedron`, scaled to `size`,
    /// with a fixed rotation or a frame orientation.
    fn polyhedron_body(orientation: Orientation, switch_radius: f64) -> GravParams {
        let size = 1e-3;
        let v = (0..8_u32)
            .map(|i| {
                Vector3::new(
                    if i & 1 == 0 { -1.3 } else { 1.1 },
                    if (i >> 1) & 1 == 0 { -0.7 } else { 0.9 },
                    if (i >> 2) & 1 == 0 { -0.45 } else { 0.35 },
                ) * size
            })
            .collect();
        let f = vec![
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
        let gm = 3e-12;
        // centered, so the point mass beyond the switch is at the center of mass
        let (mesh, _) = crate::geometry::TriMesh::new(v, f).unwrap().centered();
        let model = Polyhedron::new(mesh, gm).unwrap();
        let mut body = GravParams::new(-99, gm, 2e-3);
        body.shape = Shape::Polyhedron {
            model: Arc::new(model),
            orientation,
            switch_radius,
        };
        body
    }

    fn rotation() -> Matrix3<f64> {
        *nalgebra::Rotation3::from_euler_angles(0.4, -0.9, 1.7).matrix()
    }

    /// A polyhedron body is tested against its surface, not its bounding sphere. The
    /// test prism reaches at most 0.45e-3 AU along -z of the body frame, while its
    /// bounding radius is about 1.8e-3 AU.
    #[test]
    fn polyhedron_body_inside_is_the_surface() {
        let rot = rotation();
        let body = polyhedron_body(Orientation::Fixed(rot), 0.05);
        let Shape::Polyhedron { model, .. } = &body.shape else {
            unreachable!("built as a polyhedron")
        };
        let bounding = model.mesh().bounding_radius();
        // Between the surface and the bounding sphere, along the body z axis.
        let outside = rot * Vector3::new(0.0, 0.0, 0.9e-3);
        assert!(outside.norm() < bounding);
        assert!(!body.is_inside(&outside, None).unwrap());
        let inside = rot * Vector3::new(0.0, 0.0, 0.1e-3);
        assert!(body.is_inside(&inside, None).unwrap());
        // A point body keeps the radius test.
        let point = GravParams::new(-98, 3e-12, 2e-3);
        assert!(point.is_inside(&outside, None).unwrap());
    }

    /// Inside the switch radius the acceleration is the rotated polyhedron field;
    /// outside it is the point mass; and the two agree across the switch.
    #[test]
    fn polyhedron_body_field_and_switch() {
        let rot = rotation();
        let switch = 0.05;
        let body = polyhedron_body(Orientation::Fixed(rot), switch);
        let Shape::Polyhedron { model, .. } = &body.shape else {
            unreachable!("built as a polyhedron")
        };
        let accel_at = |p: &Vector3<f64>| {
            let mut a = Vector3::zeros();
            body.add_acceleration(&mut a, p, &Vector3::zeros(), None)
                .unwrap();
            a
        };
        let near = Vector3::new(2.1e-3, -0.8e-3, 1.4e-3);
        let expect = rot * model.field(&(rot.transpose() * near)).0;
        assert!(
            (accel_at(&near) - expect).norm() <= 1e-14 * expect.norm(),
            "inside: rotated polyhedron field"
        );

        let dir = Vector3::new(0.3, 0.5, -0.8).normalize();
        let (a_in, a_out) = (
            accel_at(&(dir * switch * 0.999_999)),
            accel_at(&(dir * switch)),
        );
        let point = -body.mass * dir / switch.powi(2);
        assert!(
            (a_out - point).norm() <= 1e-14 * point.norm(),
            "outside: point mass"
        );
        // at 50 prism sizes the quadrupole is ~1e-4 of the monopole
        assert!(
            (a_in - a_out).norm() < 1e-3 * point.norm(),
            "continuous across the switch, same sign"
        );
    }

    #[test]
    fn polyhedron_body_jacobian_matches_finite_difference() {
        let body = polyhedron_body(Orientation::Fixed(rotation()), 0.05);
        let p = Vector3::new(2.1e-3, -0.8e-3, 1.4e-3);
        let v = Vector3::new(1e-4, 2e-4, -1e-4);
        let mut accel = Vector3::zeros();
        let mut da_dr = Matrix3::zeros();
        let mut da_dv = Matrix3::zeros();
        body.add_acceleration_and_jacobians(&mut accel, &mut da_dr, &mut da_dv, &p, &v, None)
            .unwrap();
        let accel_at = |q: &Vector3<f64>| {
            let mut a = Vector3::zeros();
            body.add_acceleration(&mut a, q, &v, None).unwrap();
            a
        };
        assert_eq!(accel, accel_at(&p));
        let h = 1e-9;
        for axis in 0..3 {
            let step = Vector3::ith(axis, h);
            let col = (accel_at(&(p + step)) - accel_at(&(p - step))) / (2.0 * h);
            assert!(
                (da_dr.column(axis) - col).norm() < 1e-6 * da_dr.norm(),
                "da/dr column {axis}"
            );
        }
        assert_eq!(da_dv, Matrix3::zeros(), "no velocity dependence");
    }

    #[test]
    fn polyhedron_body_ck_orientation_required_inside_only() {
        let body = polyhedron_body(Orientation::Frame { frame_id: -42 }, 0.05);
        let near = Vector3::new(2.1e-3, -0.8e-3, 1.4e-3);
        let far = Vector3::new(0.3, 0.0, 0.0);
        assert_eq!(body.orientation_needed(&near), Some(-42), "needed inside");
        assert_eq!(body.orientation_needed(&far), None, "not needed outside");
        let mut a = Vector3::zeros();
        assert!(
            body.add_acceleration(&mut a, &near, &Vector3::zeros(), None)
                .is_err(),
            "missing orientation is an error, not a point mass"
        );
        assert!(
            body.add_acceleration(&mut a, &far, &Vector3::zeros(), None)
                .is_ok(),
            "outside, the point mass needs no orientation"
        );
        let rot = rotation();
        let mut with_ck = Vector3::zeros();
        body.add_acceleration(&mut with_ck, &near, &Vector3::zeros(), Some(&rot))
            .unwrap();
        let fixed = polyhedron_body(Orientation::Fixed(rot), 0.05);
        let mut with_fixed = Vector3::zeros();
        fixed
            .add_acceleration(&mut with_fixed, &near, &Vector3::zeros(), None)
            .unwrap();
        assert_eq!(with_ck, with_fixed, "a provided rotation is used as given");
    }

    /// A spherical harmonic body with its model in the body frame, in AU.
    fn harmonics_body(
        model: SphericalHarmonics,
        orientation: Orientation,
        switch_radius: f64,
    ) -> GravParams {
        let mut body = GravParams::new(-98, model.gm(), 2e-3);
        body.shape = Shape::SphericalHarmonics {
            model: Arc::new(model),
            orientation,
            switch_radius,
        };
        body
    }

    /// Degree 4 coefficients, the minimum radius the reference radius.
    fn test_harmonics(gm: f64, radius: f64) -> SphericalHarmonics {
        let c: Vec<Vec<f64>> = (0..=4_usize)
            .map(|n| {
                (0..=n)
                    .map(|m| {
                        if n == 0 {
                            1.0
                        } else {
                            0.05 * ((n * 3 + m) as f64).sin()
                        }
                    })
                    .collect()
            })
            .collect();
        let s: Vec<Vec<f64>> = (0..=4_usize)
            .map(|n| {
                (0..=n)
                    .map(|m| {
                        if m == 0 {
                            0.0
                        } else {
                            0.05 * ((n + m * 5) as f64).cos()
                        }
                    })
                    .collect()
            })
            .collect();
        SphericalHarmonics::new(gm, radius, &c, &s, Some(radius)).unwrap()
    }

    fn accel_of(body: &GravParams, p: &Vector3<f64>) -> KeteResult<Vector3<f64>> {
        let mut a = Vector3::zeros();
        body.add_acceleration(&mut a, p, &Vector3::zeros(), None)?;
        Ok(a)
    }

    /// J2 as a spherical harmonic body equals the oblate body, acceleration and
    /// Jacobian, with the pole carried by the orientation.
    #[test]
    fn harmonics_body_j2_matches_oblate() {
        // a radius exact in f32, the type of GravParams::radius
        let (gm, radius, j2) = (3e-12, 0.001_953_125, 2e-2);
        let rot = rotation();
        let c = [
            vec![1.0],
            vec![0.0, 0.0],
            vec![-j2 / 5_f64.sqrt(), 0.0, 0.0],
        ];
        let s = [vec![0.0], vec![0.0; 2], vec![0.0; 3]];
        let model = SphericalHarmonics::new(gm, radius, &c, &s, Some(radius)).unwrap();
        let body = harmonics_body(model, Orientation::Fixed(rot), 0.05);
        let mut oblate = GravParams::new(-97, gm, radius as f32);
        oblate.shape = Shape::Oblate {
            j2,
            pole: rot * Vector3::z(),
        };
        for p in [
            Vector3::new(3.1e-3, -0.8e-3, 1.4e-3),
            Vector3::new(-2.0e-3, 2.5e-3, -4.0e-3),
        ] {
            let (a, a0) = (accel_of(&body, &p).unwrap(), accel_of(&oblate, &p).unwrap());
            assert!(
                (a - a0).norm() <= 1e-13 * a0.norm(),
                "{p:?}: {a:?} vs {a0:?}"
            );
            let (mut dr, mut dv, mut dr0, mut dv0) = (
                Matrix3::zeros(),
                Matrix3::zeros(),
                Matrix3::zeros(),
                Matrix3::zeros(),
            );
            body.add_acceleration_and_jacobians(
                &mut Vector3::zeros(),
                &mut dr,
                &mut dv,
                &p,
                &Vector3::zeros(),
                None,
            )
            .unwrap();
            oblate
                .add_acceleration_and_jacobians(
                    &mut Vector3::zeros(),
                    &mut dr0,
                    &mut dv0,
                    &p,
                    &Vector3::zeros(),
                    None,
                )
                .unwrap();
            assert!((dr - dr0).norm() <= 1e-12 * dr0.norm(), "{p:?}: da/dr");
        }
    }

    #[test]
    fn harmonics_body_jacobian_matches_finite_difference() {
        let body = harmonics_body(
            test_harmonics(3e-12, 1.5e-3),
            Orientation::Fixed(rotation()),
            0.05,
        );
        let p = Vector3::new(2.6e-3, -1.8e-3, 1.4e-3);
        let mut accel = Vector3::zeros();
        let mut da_dr = Matrix3::zeros();
        let mut da_dv = Matrix3::zeros();
        body.add_acceleration_and_jacobians(
            &mut accel,
            &mut da_dr,
            &mut da_dv,
            &p,
            &Vector3::zeros(),
            None,
        )
        .unwrap();
        assert_eq!(accel, accel_of(&body, &p).unwrap());
        let h = 1e-9;
        for axis in 0..3 {
            let step = Vector3::ith(axis, h);
            let col = (accel_of(&body, &(p + step)).unwrap()
                - accel_of(&body, &(p - step)).unwrap())
                / (2.0 * h);
            assert!(
                (da_dr.column(axis) - col).norm() < 1e-6 * da_dr.norm(),
                "da/dr column {axis}"
            );
        }
        assert_eq!(da_dv, Matrix3::zeros(), "no velocity dependence");
    }

    /// Inside the model's minimum radius the body is the point mass, not a diverging
    /// series, and the position counts as inside the body; beyond the switch radius
    /// it is also the point mass.
    #[test]
    fn harmonics_body_inside_minimum_radius_is_the_point_mass() {
        let body = harmonics_body(
            test_harmonics(3e-12, 1.5e-3),
            Orientation::Fixed(rotation()),
            0.05,
        );
        let inside = Vector3::new(1.0e-3, 0.5e-3, -0.2e-3);
        let point = -inside * body.mass / inside.norm().powi(3);
        assert!((accel_of(&body, &inside).unwrap() - point).norm() <= 1e-15 * point.norm());
        let (mut dr, mut dv) = (Matrix3::zeros(), Matrix3::zeros());
        body.add_acceleration_and_jacobians(
            &mut Vector3::zeros(),
            &mut dr,
            &mut dv,
            &inside,
            &Vector3::zeros(),
            None,
        )
        .unwrap();
        assert!(body.is_inside(&inside, None).unwrap());
        assert!(
            !body
                .is_inside(&Vector3::new(2.5e-3, 0.0, 0.0), None)
                .unwrap()
        );
        let far = Vector3::new(0.06, 0.0, 0.0);
        let point = -far * body.mass / far.norm().powi(3);
        assert!((accel_of(&body, &far).unwrap() - point).norm() <= 1e-15 * point.norm());
    }

    #[test]
    fn harmonics_body_ck_orientation_required_inside_only() {
        let rot = rotation();
        let body = harmonics_body(
            test_harmonics(3e-12, 1.5e-3),
            Orientation::Frame { frame_id: -42 },
            0.05,
        );
        let near = Vector3::new(2.6e-3, -1.8e-3, 1.4e-3);
        let far = Vector3::new(0.3, 0.0, 0.0);
        assert_eq!(body.orientation_needed(&near), Some(-42));
        assert_eq!(body.orientation_needed(&far), None);
        assert!(
            accel_of(&body, &near).is_err(),
            "missing orientation is an error"
        );
        assert!(accel_of(&body, &far).is_ok());
        let mut with_ck = Vector3::zeros();
        body.add_acceleration(&mut with_ck, &near, &Vector3::zeros(), Some(&rot))
            .unwrap();
        let fixed = harmonics_body(test_harmonics(3e-12, 1.5e-3), Orientation::Fixed(rot), 0.05);
        assert_eq!(with_ck, accel_of(&fixed, &near).unwrap());
    }

    /// A harmonic body built from a polyhedron's exact coefficients matches the
    /// polyhedron body's closed-form field outside the Brillouin sphere.
    #[test]
    fn harmonics_body_matches_polyhedron_body() {
        let rot = rotation();
        let poly_body = polyhedron_body(Orientation::Fixed(rot), 0.05);
        let Shape::Polyhedron { model, .. } = &poly_body.shape else {
            unreachable!("built as a polyhedron")
        };
        let exact = (**model).clone().without_far_field();
        let harmonics = SphericalHarmonics::from_polyhedron(&exact, 20).unwrap();
        let rb = harmonics.radius();
        let body = harmonics_body(harmonics, Orientation::Fixed(rot), 0.05);
        for (dir, scale) in [
            (Vector3::new(0.3, -0.8, 0.5), 2.5),
            (Vector3::new(-1.0, 0.2, 0.1), 4.0),
            (Vector3::new(0.1, 0.1, -1.0), 8.0),
        ] {
            let p = dir.normalize() * rb * scale;
            let expect = rot * exact.field(&(rot.transpose() * p)).0;
            let got = accel_of(&body, &p).unwrap();
            assert!(
                (got - expect).norm() <= 1e-9 * expect.norm(),
                "{scale} R_B: {:e}",
                (got - expect).norm() / expect.norm()
            );
        }
    }

    #[test]
    fn register_replaces() {
        let id = -777_001;
        GravParams::new(id, 1e-20, 1e-9).register();
        GravParams::new(id, 2e-20, 3e-9).register();
        let selected = GravParams::selected_masses();
        let entries: Vec<_> = selected.iter().filter(|p| p.naif_id == id).collect();
        assert_eq!(entries.len(), 1, "one entry per id");
        assert_eq!(entries[0].mass, 2e-20, "the later registration wins");
        drop(selected);
        MASSES_SELECTED.write().unwrap().retain(|p| p.naif_id != id);
    }
}
