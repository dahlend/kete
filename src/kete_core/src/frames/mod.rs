// SPDX-FileCopyrightText: 2026 Dar Dahlen
// SPDX-License-Identifier: BSD-3-Clause

//! Coordinate frames and related conversions.
//!
//! Distances measured in AU, time is in units of days with TDB scaling.

mod center;
mod definitions;
mod earth;
mod rotation;
mod vector;

pub use center::{CenterBody, DynCenter, EarthCenter, SSB, SunCenter};
pub use definitions::{
    Ecliptic, Equatorial, FK4, FrameId, Galactic, InertialFrame, NonInertialFrame,
};
pub use earth::{
    EARTH_A, approx_delta_t, approx_earth_frame, approx_earth_pos_to_ecliptic, approx_solar_noon,
    approx_sun_dec, approx_ut1, earth_nutation, earth_obliquity, earth_precession_rotation,
    ecef_to_geodetic_lat_lon, equation_of_time, geocentric_radius, geodetic_lat_lon_to_ecef,
    geodetic_lat_to_geocentric, geodetic_to_parallax, greenwich_mean_sidereal_time,
    next_sunset_sunrise, prime_vert_radius, teme_frame,
};
pub use rotation::{euler_rotation, quaternion_to_euler};
pub use vector::Vector;
