// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Coordinate frames and related conversions.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.

mod body_rotation;
mod center;
mod definitions;
mod earth;
mod rotation;
mod source;
mod table;
mod vector;

pub use body_rotation::BodyRotation;
pub use center::{CenterBody, DynCenter, EarthCenter, SSB, SunCenter};
pub use definitions::{
    B1950, DynFrameId, Ecliptic, EclipticB1950, Equatorial, FK4, FrameId, Galactic, InstantFrame,
    MarsIAU, OffsetFrame, StaticFrame,
};
pub use earth::{
    EARTH_A, approx_delta_t, approx_solar_noon, approx_sun_dec, earth_frame, earth_nutation,
    earth_obliquity, earth_pos_to_ecliptic, earth_precession_rotation, ecef_to_geodetic_lat_lon,
    equation_of_time, geocentric_radius, geodetic_lat_lon_to_ecef, geodetic_lat_to_geocentric,
    geodetic_to_parallax, greenwich_mean_sidereal_time, next_sunset_sunrise, prime_vert_radius,
    teme_frame,
};
pub use rotation::{euler_rotation, quaternion_to_euler};
pub use source::{FrameSource, MAX_FRAME_CHAIN};
pub use table::{FrameDefinition, FrameTable};
pub use vector::Vector;
